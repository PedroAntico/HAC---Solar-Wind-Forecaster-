#!/usr/bin/env python3
"""
HAC - Heliospheric Accumulated Coupling (Versão Científica Final v5)
- Limpeza MÍNIMA (preserva > 95% dos dados)
- Filtro de speed quase removido
- Calibração linear automática
- gammas ajustados para melhor escala
- Proteções contra overflow
"""

import pandas as pd
import numpy as np
import warnings
import argparse
import os
import json
from scipy.stats import pearsonr
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================
# CONSTANTES FÍSICAS
# ============================
M_P = 1.6726e-27
CM3_TO_M3 = 1e6
KMS_TO_MS = 1e3

def compute_pdyn_nPa(N_cm3, V_kms):
    N_m3 = N_cm3 * CM3_TO_M3
    V_ms = V_kms * KMS_TO_MS
    return M_P * N_m3 * V_ms**2 * 1e9

# ============================
# PREPARE OMNI - LIMPEZA MÍNIMA
# ============================
def prepare_omni(df):
    df = df.copy().sort_values('time_tag').reset_index(drop=True)
    if 'dst' in df.columns:
        df = df.drop(columns=['dst'])

    # Conversão automática m/s → km/s
    if 'speed' in df.columns and len(df) > 100:
        med = df['speed'].median()
        if med > 10000:
            print(f"⚠️ Speed detectado em m/s (mediana={med:.0f}) → convertendo para km/s")
            df['speed'] /= 1000.0

    # LIMPEZA MÍNIMA - apenas valores impossíveis fisicamente
    bad = (
        (df['bz_gsm'].abs() > 100) |
        (df['speed'] < 100) | (df['speed'] > 3000) |
        (df['density'] < 0.001) | (df['density'] > 5000) |
        (df['bx_gsm'].abs() > 100) |
        (df['by_gsm'].abs() > 100)
    )
    before = len(df)
    df.loc[bad, ['bz_gsm', 'bx_gsm', 'by_gsm', 'density', 'speed']] = np.nan

    # Preenchimento generoso
    df['bz_gsm'] = df['bz_gsm'].ffill(limit=12)
    for col in ['bx_gsm', 'by_gsm', 'density', 'speed']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=24)

    df = df.dropna(subset=['bz_gsm', 'speed', 'density']).reset_index(drop=True)
    after = len(df)

    print(f"   prepare_omni: {before:,} → {after:,} pontos ({(before - after) / before * 100:.1f}% removidos)")
    if after > 0:
        print(f"   Speed final: min={df['speed'].min():.0f}, max={df['speed'].max():.0f}, mediana={df['speed'].median():.0f} km/s")
        print(f"   Bz: min={df['bz_gsm'].min():.1f}, max={df['bz_gsm'].max():.1f} nT")
    return df

def split_data(df, split_date='2010-01-01'):
    train = df[df['time_tag'] < split_date].copy()
    test = df[df['time_tag'] >= split_date].copy()
    return train, test

def smooth_dst(merged, hours):
    if hours <= 0:
        merged = merged.copy()
        merged['dst_smooth'] = merged['dst']
        return merged
    merged = merged.set_index('time_tag')
    if len(merged) > 1:
        med_dt = merged.index.to_series().diff().median().total_seconds()
        win = max(3, int(hours * 3600 / med_dt))
        win = win + 1 if win % 2 == 0 else win
        merged['dst_smooth'] = merged['dst'].rolling(win, center=True, min_periods=1).mean()
    else:
        merged['dst_smooth'] = merged['dst']
    merged = merged.dropna(subset=['dst_smooth'])
    return merged.reset_index()

# ============================
# HAC VETORIAL
# ============================
def integrate_hac_vectorial(df, driver_type='akasofu'):
    if len(df) == 0:
        return df.copy()
    times = df['time_tag'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    N = df['density'].values
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values

    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    theta = np.arctan2(By, Bz)
    sin4 = np.sin(theta / 2)**4

    if driver_type == 'akasofu':
        phi_raw = V * (Btot**2) * sin4
        pos = phi_raw[phi_raw > 0]
        scale = np.percentile(pos, 90) if len(pos) > 0 else 1.0
        phi = phi_raw / (scale + 1e-6)
    else:
        Pdyn = compute_pdyn_nPa(N, V)
        phi = 0.7 * (Pdyn * Btot) + 0.3 * (V * Btot**2 * sin4)
        phi = phi / 1000.0

    dt_sec = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec[1:] = diffs
    dt_sec[0] = np.median(diffs) if len(diffs) > 0 else 3600.0
    dt_h = dt_sec / 3600.0

    Hload = np.zeros(len(times))
    Hrel = np.zeros(len(times))
    Hsat = np.zeros(len(times))

    alpha_L = 1.2e-5
    beta_L = 0.15
    alpha_R = 0.08
    beta_R = 0.25
    eta = 0.3
    theta_sat = 2.5
    sigma_sat = 0.5

    for i in range(1, len(times)):
        dt = dt_h[i]
        Hsat[i-1] = 1 / (1 + np.exp(-(Hload[i-1] - theta_sat) / sigma_sat))
        dHload = alpha_L * phi[i] * (1 - Hsat[i-1]) - beta_L * Hload[i-1]
        dHrel = alpha_R * Hload[i-1] - beta_R * Hrel[i-1] + eta * Hsat[i-1] * Hload[i-1]
        Hload[i] = Hload[i-1] + dHload * dt
        Hrel[i] = Hrel[i-1] + dHrel * dt
        Hload[i] = max(0, Hload[i])
        Hrel[i] = max(0, Hrel[i])

    Hsat[-1] = 1 / (1 + np.exp(-(Hload[-1] - theta_sat) / sigma_sat))
    C = np.maximum(-Bz, 0) * (V / 400.0)
    HCI = C + Hload

    out = df.copy()
    out['Hload'] = Hload
    out['Hrel'] = Hrel
    out['Hsat'] = Hsat
    out['HCI'] = HCI
    out['phi'] = phi
    out['C'] = C
    out['Bz'] = Bz
    return out

# ============================
# MODELO DST (COM ESCALA AJUSTADA)
# ============================
def compute_dst_direct(df, tau_R=6.0, gamma1=0.6, gamma2=0.05, gamma3=0.05,
                       gamma4=0.0, gamma5=0.0, gamma6=1.2, threshold=1.5,
                       delay_internal=2.0, tau_driver=0.0, b_pressure=1.5,
                       vbs_boost=2.0, debug=False):
    if len(df) == 0:
        return np.array([])

    times = df['time_tag'].values
    Hload = df['Hload'].values
    Hrel = df['Hrel'].values
    C = df['C'].values
    Bz = df['Bz'].values
    V = df['speed'].values
    Pdyn = compute_pdyn_nPa(df['density'].values, V)

    Bs = np.maximum(-Bz, 0)
    VBs = (V * Bs) / 1000.0

    interaction = (Hload * C) / (1 + Hload)
    extra = np.zeros_like(Hload)
    mask = Hload > threshold
    if np.any(mask):
        delta = Hload[mask] - threshold
        extra[mask] = (delta**2) / (1 + delta)

    driver_raw = (gamma1 * Hload + gamma2 * Hrel + gamma3 * C +
                  gamma4 * interaction + gamma5 * extra +
                  gamma6 * VBs * vbs_boost)

    # Clipping suave (apenas extremos)
    if np.any(driver_raw > 100):
        cap = np.percentile(driver_raw[driver_raw > 0], 99.9)
        driver_raw = np.clip(driver_raw, 0, cap)

    dt_sec_arr = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec_arr[1:] = diffs
    dt_sec_arr[0] = np.median(diffs) if len(diffs) > 0 else 3600.0
    dt_h = dt_sec_arr / 3600.0

    if delay_internal > 0:
        median_dt = np.median(dt_sec_arr[dt_sec_arr > 0])
        delay_steps = int(round(delay_internal * 3600 / median_dt))
        if delay_steps > 0:
            driver_delayed = np.zeros_like(driver_raw)
            driver_delayed[delay_steps:] = driver_raw[:-delay_steps]
        else:
            driver_delayed = driver_raw
    else:
        driver_delayed = driver_raw

    if tau_driver > 0:
        driver_eff = np.zeros_like(driver_delayed)
        for i in range(1, len(driver_eff)):
            decay = np.exp(-dt_h[i] / tau_driver)
            driver_eff[i] = driver_eff[i-1] * decay + driver_delayed[i] * (1 - decay)
        driver = driver_eff
    else:
        driver = driver_delayed

    tau_eff = tau_R / (1 + 0.5 * C)

    # Interpoladores para solve_ivp
    driver_interp = interp1d(np.arange(len(driver)), driver, bounds_error=False, fill_value=(driver[0], driver[-1]))
    tau_interp = interp1d(np.arange(len(tau_eff)), tau_eff, bounds_error=False, fill_value=(tau_eff[0], tau_eff[-1]))
    pdyn_interp = interp1d(np.arange(len(Pdyn)), Pdyn, bounds_error=False, fill_value=(Pdyn[0], Pdyn[-1]))

    def dst_ode(t, Dst):
        t_idx = np.clip(int(round(t)), 0, len(driver)-1)
        return -driver_interp(t_idx) - Dst / tau_interp(t_idx) + b_pressure * np.sqrt(pdyn_interp(t_idx))

    sol = solve_ivp(dst_ode, [0, len(times)-1], [0.0], method='LSODA', t_eval=np.arange(len(times)), rtol=1e-6, atol=1e-8)
    Dst = sol.y[0]

    if debug:
        print(f"\n🔍 DEBUG compute_dst_direct:")
        print(f"   driver: min={np.min(driver):.3f}, max={np.max(driver):.3f}, mean={np.mean(driver):.3f}")
        print(f"   Dst_raw: min={np.min(Dst):.1f}, max={np.max(Dst):.1f}, mean={np.mean(Dst):.1f}")

    return Dst

# ============================
# BASELINE BURTON (COM PROTEÇÃO)
# ============================
def compute_dst_burton(df, tau=6.0, alpha=4.5, delay_internal=2.0):
    if len(df) == 0:
        return np.array([])
    times = df['time_tag'].values
    Bz = df['Bz'].values
    V = df['speed'].values
    Bs = np.maximum(-Bz, 0)
    VBs = (V * Bs) / 1000.0

    dt_sec = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec[1:] = diffs
    dt_sec[0] = np.median(diffs) if len(diffs) > 0 else 3600.0
    dt_h = dt_sec / 3600.0

    if delay_internal > 0:
        median_dt = np.median(dt_sec[dt_sec > 0])
        delay_steps = int(round(delay_internal * 3600 / median_dt))
        if delay_steps > 0:
            VBs_delayed = np.zeros_like(VBs)
            VBs_delayed[delay_steps:] = VBs[:-delay_steps]
        else:
            VBs_delayed = VBs
    else:
        VBs_delayed = VBs

    Dst = np.zeros(len(times))
    for i in range(1, len(times)):
        dt = dt_h[i]
        vbs_val = np.clip(VBs_delayed[i], 0, 100)
        dDst = -alpha * vbs_val - Dst[i-1] / tau
        dDst = np.clip(dDst, -500, 500)
        Dst[i] = Dst[i-1] + dDst * dt
        Dst[i] = np.clip(Dst[i], -500, 50)
    return Dst

# ============================
# MÉTRICAS
# ============================
def storm_metrics(y_true, y_pred, threshold=-50):
    mask = y_true < threshold
    if mask.sum() == 0:
        return {'mae_storm': np.nan, 'n_storm': 0}
    mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
    return {'mae_storm': mae, 'n_storm': mask.sum()}

def compute_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return None
    corr = pearsonr(y_true, y_pred)[0] if np.std(y_true) * np.std(y_pred) > 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    storm = storm_metrics(y_true, y_pred)
    return {'correlation': corr, 'r2': r2, 'rmse': rmse, 'mae': mae,
            'mae_storm': storm['mae_storm'], 'n_storm': storm['n_storm']}

def validation_metrics_direct(df, pred_col='Dst_model', smooth_hours=0):
    df_smooth = smooth_dst(df[['time_tag', 'dst']], smooth_hours)
    df_aligned = pd.merge_asof(
        df_smooth.sort_values('time_tag'),
        df[['time_tag', pred_col]].sort_values('time_tag'),
        on='time_tag', direction='nearest'
    ).dropna()
    y_true = df_aligned['dst_smooth'].values
    y_pred = df_aligned[pred_col].values
    return compute_metrics(y_true, y_pred)

# ============================
# OTIMIZAÇÃO (OPCIONAL)
# ============================
def objective(params, train_df):
    gamma1, gamma2, gamma3, gamma6, delay_internal, b_pressure = params
    train_copy = train_df.copy()
    train_copy['Dst_raw'] = compute_dst_direct(train_copy,
                                               gamma1=gamma1, gamma2=gamma2, gamma3=gamma3,
                                               gamma6=gamma6, delay_internal=delay_internal,
                                               b_pressure=b_pressure, vbs_boost=2.0)
    metrics = validation_metrics_direct(train_copy, pred_col='Dst_raw', smooth_hours=0)
    return metrics['mae_storm'] if np.isfinite(metrics['mae_storm']) else 100.0

def optimize_parameters(train_df):
    bounds = [(0.2, 1.5), (0.01, 0.5), (0.01, 0.5), (0.2, 1.5), (0, 6), (0.0, 2.0)]
    from scipy.optimize import differential_evolution
    result = differential_evolution(objective, bounds, args=(train_df,), workers=1, tol=1e-4, popsize=8, maxiter=20)
    return result.x

# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--omni', default='data/omni_2000_2020.csv')
    parser.add_argument('--dst', default='data/dst_kyoto_2000_2020.csv')
    parser.add_argument('--output', default='hac_results.csv')
    parser.add_argument('--driver_type', default='akasofu', choices=['akasofu', 'hybrid'])
    parser.add_argument('--tau_R', type=float, default=6.0)
    parser.add_argument('--gamma1', type=float, default=0.6)
    parser.add_argument('--gamma2', type=float, default=0.05)
    parser.add_argument('--gamma3', type=float, default=0.05)
    parser.add_argument('--gamma4', type=float, default=0.0)
    parser.add_argument('--gamma5', type=float, default=0.0)
    parser.add_argument('--gamma6', type=float, default=1.2)
    parser.add_argument('--threshold', type=float, default=1.5)
    parser.add_argument('--delay_internal', type=float, default=2.0)
    parser.add_argument('--tau_driver', type=float, default=0.0)
    parser.add_argument('--b_pressure', type=float, default=1.5)
    parser.add_argument('--vbs_boost', type=float, default=2.0)
    parser.add_argument('--smooth_hours', type=float, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train_test', action='store_true')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--save_params', type=str, default=None)
    parser.add_argument('--load_params', type=str, default=None)
    args = parser.parse_args()

    if args.load_params and os.path.exists(args.load_params):
        with open(args.load_params, 'r') as f:
            params = json.load(f)
        for key, val in params.items():
            setattr(args, key, val)
        print(f"📂 Parâmetros carregados de {args.load_params}")

    print("\n" + "="*80)
    print("🛰️  HAC - Modelo Científico Final v5 (limpeza mínima)")
    print("="*80)

    # Carregar OMNI
    if not os.path.exists(args.omni):
        print(f"❌ OMNI não encontrado: {args.omni}")
        return
    omni = pd.read_csv(args.omni, parse_dates=['time_tag'])
    omni = prepare_omni(omni)
    if len(omni) == 0:
        return
    print(f"✅ OMNI carregado: {len(omni)} pontos ({omni['time_tag'].min()} a {omni['time_tag'].max()})")

    # Carregar Dst
    if not os.path.exists(args.dst):
        print(f"❌ Dst não encontrado: {args.dst}")
        return
    dst = pd.read_csv(args.dst, parse_dates=['time_tag']).sort_values('time_tag')
    dst = dst[dst['time_tag'] <= omni['time_tag'].max()]

    # Merge
    time_grid = omni[['time_tag']].copy()
    dst_interp = pd.merge_asof(
        time_grid.sort_values('time_tag'),
        dst[['time_tag', 'dst']].sort_values('time_tag'),
        on='time_tag', direction='nearest', tolerance=pd.Timedelta('2h')
    ).set_index('time_tag').interpolate(method='time', limit=3).reset_index()

    df = pd.merge(omni, dst_interp, on='time_tag', how='left').dropna(subset=['dst'])
    print(f"   Merge final: {len(df)} pontos, Dst min = {df['dst'].min():.1f} nT")

    df = integrate_hac_vectorial(df, driver_type=args.driver_type)

    if not args.train_test:
        df['Dst_raw'] = compute_dst_direct(df, debug=args.debug,
                                           tau_R=args.tau_R,
                                           gamma1=args.gamma1, gamma2=args.gamma2, gamma3=args.gamma3,
                                           gamma4=args.gamma4, gamma5=args.gamma5, gamma6=args.gamma6,
                                           threshold=args.threshold,
                                           delay_internal=args.delay_internal,
                                           tau_driver=args.tau_driver,
                                           b_pressure=args.b_pressure,
                                           vbs_boost=args.vbs_boost)
        reg = LinearRegression().fit(df['Dst_raw'].values.reshape(-1,1), df['dst'].values)
        df['Dst_model'] = reg.predict(df['Dst_raw'].values.reshape(-1,1))
        print(f"📐 Calibração linear: Dst = {reg.coef_[0]:.4f} * Dst_raw + {reg.intercept_:.2f}")
        df.to_csv(args.output, index=False)
        print(f"💾 Resultados salvos em {args.output}")
        return

    # ==================== MODO TREINO/TESTE ====================
    print("\n📊 VALIDAÇÃO TREINO/TESTE (2000-2010 / 2010-2020)")
    omni_raw = pd.read_csv(args.omni, parse_dates=['time_tag'])
    omni_raw = prepare_omni(omni_raw)
    dst_raw = pd.read_csv(args.dst, parse_dates=['time_tag']).sort_values('time_tag')
    dst_raw = dst_raw[dst_raw['time_tag'] <= omni_raw['time_tag'].max()]
    dst_interp_raw = pd.merge_asof(
        omni_raw[['time_tag']].sort_values('time_tag'),
        dst_raw[['time_tag', 'dst']].sort_values('time_tag'),
        on='time_tag', direction='nearest', tolerance=pd.Timedelta('2h')
    ).set_index('time_tag').interpolate(method='time', limit=3).reset_index()
    df_raw = pd.merge(omni_raw, dst_interp_raw, on='time_tag', how='left').dropna(subset=['dst'])
    train_df, test_df = split_data(df_raw)

    print(f"Treino: {len(train_df)} pontos ({train_df['time_tag'].min()} a {train_df['time_tag'].max()})")
    print(f"Teste:  {len(test_df)} pontos ({test_df['time_tag'].min()} a {test_df['time_tag'].max()})")

    train_df = integrate_hac_vectorial(train_df, driver_type=args.driver_type)
    test_df = integrate_hac_vectorial(test_df, driver_type=args.driver_type)

    if args.optimize:
        print("\n🔧 Otimizando parâmetros...")
        opt_params = optimize_parameters(train_df)
        print(f"✅ Parâmetros otimizados: {opt_params}")
        args.gamma1, args.gamma2, args.gamma3, args.gamma6, args.delay_internal, args.b_pressure = opt_params
        if args.save_params:
            params_to_save = {k: getattr(args, k) for k in ['gamma1','gamma2','gamma3','gamma6','delay_internal','b_pressure','tau_R','vbs_boost']}
            with open(args.save_params, 'w') as f:
                json.dump(params_to_save, f, indent=2)
            print(f"💾 Parâmetros salvos em {args.save_params}")

    # Calcular Dst_raw
    train_df['Dst_raw'] = compute_dst_direct(train_df, debug=args.debug,
                                             tau_R=args.tau_R,
                                             gamma1=args.gamma1, gamma2=args.gamma2, gamma3=args.gamma3,
                                             gamma4=args.gamma4, gamma5=args.gamma5, gamma6=args.gamma6,
                                             threshold=args.threshold,
                                             delay_internal=args.delay_internal,
                                             tau_driver=args.tau_driver,
                                             b_pressure=args.b_pressure,
                                             vbs_boost=args.vbs_boost)
    test_df['Dst_raw'] = compute_dst_direct(test_df, debug=args.debug,
                                            tau_R=args.tau_R,
                                            gamma1=args.gamma1, gamma2=args.gamma2, gamma3=args.gamma3,
                                            gamma4=args.gamma4, gamma5=args.gamma5, gamma6=args.gamma6,
                                            threshold=args.threshold,
                                            delay_internal=args.delay_internal,
                                            tau_driver=args.tau_driver,
                                            b_pressure=args.b_pressure,
                                            vbs_boost=args.vbs_boost)

    # Calibração linear usando treino
    reg = LinearRegression().fit(train_df['Dst_raw'].values.reshape(-1,1), train_df['dst'].values)
    print(f"\n📐 Calibração linear (treino): Dst = {reg.coef_[0]:.4f} * Dst_raw + {reg.intercept_:.2f}")
    train_df['Dst_hac'] = reg.predict(train_df['Dst_raw'].values.reshape(-1,1))
    test_df['Dst_hac'] = reg.predict(test_df['Dst_raw'].values.reshape(-1,1))

    # Baseline Burton
    train_df['Dst_burton'] = compute_dst_burton(train_df, tau=args.tau_R, alpha=4.5, delay_internal=args.delay_internal)
    test_df['Dst_burton'] = compute_dst_burton(test_df, tau=args.tau_R, alpha=4.5, delay_internal=args.delay_internal)

    # Métricas
    hac_train = validation_metrics_direct(train_df, 'Dst_hac', smooth_hours=0)
    hac_test = validation_metrics_direct(test_df, 'Dst_hac', smooth_hours=args.smooth_hours)
    burton_train = validation_metrics_direct(train_df, 'Dst_burton', smooth_hours=0)
    burton_test = validation_metrics_direct(test_df, 'Dst_burton', smooth_hours=args.smooth_hours)

    print("\n📊 TREINO (2000-2010) - SEM SUAVIZAÇÃO:")
    print(f"   HAC    : r={hac_train['correlation']:.3f}  R²={hac_train['r2']:.3f}  RMSE={hac_train['rmse']:.1f}  MAE_storm={hac_train.get('mae_storm',0):.1f}")
    print(f"   Burton : r={burton_train['correlation']:.3f}  R²={burton_train['r2']:.3f}  RMSE={burton_train['rmse']:.1f}  MAE_storm={burton_train.get('mae_storm',0):.1f}")

    print("\n📊 TESTE (2010-2020) - SUAVIZAÇÃO {}h:".format(args.smooth_hours))
    print(f"   HAC    : r={hac_test['correlation']:.3f}  R²={hac_test['r2']:.3f}  RMSE={hac_test['rmse']:.1f}  MAE_storm={hac_test.get('mae_storm',0):.1f}")
    print(f"   Burton : r={burton_test['correlation']:.3f}  R²={burton_test['r2']:.3f}  RMSE={burton_test['rmse']:.1f}  MAE_storm={burton_test.get('mae_storm',0):.1f}")

    test_df.to_csv(args.output.replace('.csv', '_test.csv'), index=False)
    print(f"\n💾 Resultados do teste salvos em {args.output.replace('.csv', '_test.csv')}")

if __name__ == "__main__":
    main()
