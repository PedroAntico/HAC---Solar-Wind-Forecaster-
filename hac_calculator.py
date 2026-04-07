#!/usr/bin/env python3
"""
HAC - Heliospheric Accumulated Coupling (Versão Científica Final)

Modelo Dst baseado em Burton com:
- Limpeza física dos dados OMNI (outliers removidos)
- Escala fixa (sem normalização perigosa)
- Integração numérica estável (solve_ivp)
- Otimização automática de parâmetros (differential_evolution)
- Baseline Burton para comparação
- Validação treino/teste com métricas focadas em tempestades

Uso:
    python hac_calculator.py --train_test
    python hac_calculator.py --train_test --optimize
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import os
import json
from scipy.stats import pearsonr
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================
# CONSTANTES FÍSICAS
# ============================
M_P = 1.6726e-27      # kg
CM3_TO_M3 = 1e6
KMS_TO_MS = 1e3

def compute_pdyn_nPa(N_cm3, V_kms):
    N_m3 = N_cm3 * CM3_TO_M3
    V_ms = V_kms * KMS_TO_MS
    return M_P * N_m3 * V_ms**2 * 1e9

# ============================
# LIMPEZA ROBUSTA DOS DADOS
# ============================
def prepare_omni(df):
    df = df.copy().sort_values('time_tag').reset_index(drop=True)
    if 'dst' in df.columns:
        df = df.drop(columns=['dst'])
    
    # Limpeza física (remove valores espúrios)
    bad = (
        (df['bz_gsm'].abs() > 100) |          # |Bz| > 100 nT é fill value ou erro
        (df['speed'] < 100) | (df['speed'] > 2000) |
        (df['density'] < 0.1) | (df['density'] > 1000) |
        (df['bx_gsm'].abs() > 100) |
        (df['by_gsm'].abs() > 100)
    )
    df.loc[bad, ['bz_gsm', 'bx_gsm', 'by_gsm', 'density', 'speed']] = np.nan
    
    # Preenchimento conservador
    df['bz_gsm'] = df['bz_gsm'].ffill(limit=3)
    for col in ['bx_gsm', 'by_gsm', 'density', 'speed']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=6)
    
    df = df.dropna(subset=['bz_gsm', 'speed', 'density']).reset_index(drop=True)
    return df

def apply_time_shift(df, hours):
    df = df.copy()
    df['time_tag'] = df['time_tag'] - pd.Timedelta(hours=hours)
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
# HAC VETORIAL (DRIVER BASE)
# ============================
def integrate_hac_vectorial(df, driver_type='akasofu'):
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
    dt_sec[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    dt_h = dt_sec / 3600.0

    Hload = np.zeros(len(times))
    Hrel  = np.zeros(len(times))
    Hsat  = np.zeros(len(times))

    alpha_L = 1.2e-5
    beta_L  = 0.15
    alpha_R = 0.08
    beta_R  = 0.25
    eta     = 0.3
    theta_sat = 2.5
    sigma_sat = 0.5

    for i in range(1, len(times)):
        dt = dt_h[i]
        Hsat[i-1] = 1 / (1 + np.exp(-(Hload[i-1] - theta_sat) / sigma_sat))
        dHload = alpha_L * phi[i] * (1 - Hsat[i-1]) - beta_L * Hload[i-1]
        dHrel  = alpha_R * Hload[i-1] - beta_R * Hrel[i-1] + eta * Hsat[i-1] * Hload[i-1]
        Hload[i] = Hload[i-1] + dHload * dt
        Hrel[i]  = Hrel[i-1]  + dHrel  * dt
        Hload[i] = max(0, Hload[i])
        Hrel[i]  = max(0, Hrel[i])

    Hsat[-1] = 1 / (1 + np.exp(-(Hload[-1] - theta_sat) / sigma_sat))
    C = np.maximum(-Bz, 0) * (V / 400.0)
    HCI = C + Hload

    out = df.copy()
    out['Hload'] = Hload
    out['Hrel']  = Hrel
    out['Hsat']  = Hsat
    out['HCI']   = HCI
    out['phi']   = phi
    out['C']     = C
    out['Bz']    = Bz
    return out

# ============================
# MODELO DST DIRETO (COM INTEGRAÇÃO ESTÁVEL)
# ============================
def compute_dst_direct(df, tau_R=6.0, gamma1=0.2, gamma2=0.05, gamma3=0.05,
                       gamma4=0.0, gamma5=0.0, gamma6=0.05, threshold=1.5,
                       delay_internal=2.0, tau_driver=0.0, b_pressure=2.0,
                       debug=False):
    """
    Modelo Dst com equação física:
        dDst/dt = -driver - Dst/τ + b_pressure * sqrt(Pdyn)
    driver = gamma1*Hload + gamma2*Hrel + gamma3*C + gamma4*interaction + gamma5*extra + gamma6*VBs
    VBs = V * max(-Bz,0) / 1000 (escala física)
    delay_internal: atraso aplicado ao driver (horas)
    """
    times = df['time_tag'].values
    Hload = df['Hload'].values
    Hrel  = df['Hrel'].values
    C     = df['C'].values
    Bz    = df['Bz'].values
    V     = df['speed'].values
    Pdyn  = compute_pdyn_nPa(df['density'].values, df['speed'].values)

    # Driver físico: VBs normalizado (escala física)
    Bs = np.maximum(-Bz, 0)
    VBs = (V * Bs) / 1000.0

    # Termos de memória (opcionais)
    interaction = (Hload * C) / (1 + Hload)
    extra = np.zeros_like(Hload)
    mask = Hload > threshold
    if np.any(mask):
        delta = Hload[mask] - threshold
        extra[mask] = (delta**2) / (1 + delta)

    driver_raw = (gamma1 * Hload + gamma2 * Hrel + gamma3 * C +
                  gamma4 * interaction + gamma5 * extra +
                  gamma6 * VBs)

    # Proteção contra outliers (apenas 0.1% mais extremo)
    if np.any(driver_raw > 100):
        cap = np.percentile(driver_raw[driver_raw > 0], 99.9)
        driver_raw = np.clip(driver_raw, 0, cap)

    # Atraso interno (shift puro)
    dt_sec_arr = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec_arr[1:] = diffs
    dt_sec_arr[0] = np.median(diffs) if len(diffs) > 0 else 60.0
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

    # Memória exponencial do driver (opcional)
    if tau_driver > 0:
        driver_eff = np.zeros_like(driver_delayed)
        for i in range(1, len(driver_eff)):
            decay = np.exp(-dt_h[i] / tau_driver)
            driver_eff[i] = driver_eff[i-1] * decay + driver_delayed[i] * (1 - decay)
        driver = driver_eff
    else:
        driver = driver_delayed

    # Constante de tempo efetiva (depende de C)
    tau_eff = tau_R / (1 + 0.5 * C)

    # Integração numérica estável com solve_ivp
    def dst_ode(t, Dst):
        # Interpolação linear para obter valores no tempo t
        idx = int(round(t))
        if idx >= len(times):
            idx = len(times) - 1
        driver_t = driver[idx]
        tau_t = tau_eff[idx]
        pdyn_t = Pdyn[idx]
        return -driver_t - Dst / tau_t + b_pressure * np.sqrt(pdyn_t)

    t_span = (0, len(times) - 1)
    t_eval = np.arange(len(times))
    sol = solve_ivp(dst_ode, t_span, [0.0], method='LSODA', t_eval=t_eval)
    Dst = sol.y[0]

    if debug:
        print(f"\n🔍 DEBUG:")
        print(f"   driver: min={np.min(driver):.3f}, max={np.max(driver):.3f}, mean={np.mean(driver):.3f}")
        print(f"   tau_eff: min={np.min(tau_eff):.3f}, max={np.max(tau_eff):.3f}, mean={np.mean(tau_eff):.3f}")
        print(f"   Dst_raw: min={np.min(Dst):.3f}, max={np.max(Dst):.3f}, mean={np.mean(Dst):.3f}, std={np.std(Dst):.3f}")

    return Dst

# ============================
# BASELINE BURTON CLÁSSICO
# ============================
def compute_dst_burton(df, tau=6.0, alpha=4.5, delay_internal=2.0):
    """
    Modelo Burton simples: dDst/dt = -α * VBs - Dst/τ
    """
    times = df['time_tag'].values
    Bz = df['Bz'].values
    V = df['speed'].values
    Bs = np.maximum(-Bz, 0)
    VBs = (V * Bs) / 1000.0

    dt_sec = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec[1:] = diffs
    dt_sec[0] = np.median(diffs) if len(diffs) > 0 else 60.0
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
        dDst = -alpha * VBs_delayed[i] - Dst[i-1] / tau
        Dst[i] = Dst[i-1] + dDst * dt
    return Dst

# ============================
# MÉTRICAS (COM FOCO EM TEMPESTADES)
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
# OTIMIZAÇÃO DE PARÂMETROS
# ============================
def objective(params, train_df):
    gamma1, gamma2, gamma3, gamma6, delay_internal, b_pressure = params
    train_df['Dst_raw'] = compute_dst_direct(train_df,
                                             gamma1=gamma1, gamma2=gamma2, gamma3=gamma3,
                                             gamma6=gamma6, delay_internal=delay_internal,
                                             b_pressure=b_pressure)
    metrics = validation_metrics_direct(train_df, pred_col='Dst_raw', smooth_hours=0)
    # Minimizar MAE em tempestades ou -correlação
    if np.isfinite(metrics['mae_storm']):
        return metrics['mae_storm']
    else:
        return 100.0

def optimize_parameters(train_df):
    bounds = [(0.01, 1.0), (0.001, 0.5), (0.001, 0.5), (0.001, 0.5), (0, 6), (0.5, 5)]
    result = differential_evolution(objective, bounds, args=(train_df,), workers=1, tol=1e-4)
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
    parser.add_argument('--gamma1', type=float, default=0.2)
    parser.add_argument('--gamma2', type=float, default=0.05)
    parser.add_argument('--gamma3', type=float, default=0.05)
    parser.add_argument('--gamma4', type=float, default=0.0)
    parser.add_argument('--gamma5', type=float, default=0.0)
    parser.add_argument('--gamma6', type=float, default=0.05)
    parser.add_argument('--threshold', type=float, default=1.5)
    parser.add_argument('--delay_internal', type=float, default=2.0)
    parser.add_argument('--tau_driver', type=float, default=0.0)
    parser.add_argument('--b_pressure', type=float, default=2.0)
    parser.add_argument('--smooth_hours', type=float, default=0, help='Suavização apenas na validação final')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train_test', action='store_true')
    parser.add_argument('--optimize', action='store_true', help='Otimiza parâmetros no treino')
    parser.add_argument('--save_params', type=str, default=None)
    parser.add_argument('--load_params', type=str, default=None)
    args = parser.parse_args()

    if args.load_params:
        with open(args.load_params, 'r') as f:
            params = json.load(f)
        for key, val in params.items():
            setattr(args, key, val)
        print(f"📂 Parâmetros carregados de {args.load_params}")

    print("\n" + "="*70)
    print("🛰️  HAC - Modelo Científico Final")
    print(f"   tau_R = {args.tau_R}, gamma1={args.gamma1}, gamma2={args.gamma2}, gamma3={args.gamma3}, gamma6={args.gamma6}")
    print(f"   b_pressure = {args.b_pressure}, delay_internal = {args.delay_internal}")
    print("="*70)

    # Carregar OMNI
    if not os.path.exists(args.omni):
        print(f"❌ OMNI não encontrado: {args.omni}")
        return
    omni = pd.read_csv(args.omni, parse_dates=['time_tag'])
    omni = prepare_omni(omni)
    print(f"✅ OMNI: {len(omni)} pontos")

    # Carregar Dst Kyoto
    if not os.path.exists(args.dst):
        print(f"❌ Dst não encontrado: {args.dst}")
        return
    dst = pd.read_csv(args.dst, parse_dates=['time_tag']).sort_values('time_tag')
    print(f"✅ Dst Kyoto: {len(dst)} pontos")
    dst = dst[dst['time_tag'] <= omni['time_tag'].max()]
    print(f"   Dst cortado: {len(dst)} pontos")

    # Interpolação do Dst
    time_grid = omni[['time_tag']].copy()
    dst_interp = pd.merge_asof(
        time_grid.sort_values('time_tag'),
        dst[['time_tag', 'dst']].sort_values('time_tag'),
        on='time_tag', direction='nearest', tolerance=pd.Timedelta('2h')
    ).set_index('time_tag').interpolate(method='time', limit=3).reset_index()
    assert 'dst' in dst_interp.columns, "ERRO: coluna dst sumiu"

    if 'dst' in omni.columns:
        omni = omni.drop(columns=['dst'])
    df = pd.merge(omni, dst_interp, on='time_tag', how='left').dropna(subset=['dst'])
    print(f"   Merge final: {len(df)} pontos, Dst min = {df['dst'].min():.1f} nT")

    # Integração HAC (driver)
    df = integrate_hac_vectorial(df, driver_type=args.driver_type)

    if not args.train_test:
        df['Dst_model'] = compute_dst_direct(df, tau_R=args.tau_R,
                                             gamma1=args.gamma1, gamma2=args.gamma2,
                                             gamma3=args.gamma3, gamma4=args.gamma4,
                                             gamma5=args.gamma5, gamma6=args.gamma6,
                                             threshold=args.threshold,
                                             delay_internal=args.delay_internal,
                                             tau_driver=args.tau_driver,
                                             b_pressure=args.b_pressure,
                                             debug=args.debug)
        df.to_csv(args.output, index=False)
        print(f"💾 Resultados salvos em {args.output}")
        return

    # ========== MODO TREINO/TESTE ==========
    print("\n📊 VALIDAÇÃO TREINO/TESTE (2000-2010 / 2010-2020)")
    # Recarregar dados crus para divisão antes da integração HAC
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
    train_df, test_df = split_data(df_raw, split_date='2010-01-01')
    print(f"Treino: {len(train_df)} pontos")
    print(f"Teste: {len(test_df)} pontos")

    # Integrar HAC separadamente
    train_df = integrate_hac_vectorial(train_df, driver_type=args.driver_type)
    test_df = integrate_hac_vectorial(test_df, driver_type=args.driver_type)

    # Otimização de parâmetros (se solicitada)
    if args.optimize:
        print("\n🔧 Otimizando parâmetros no treino (pode levar alguns minutos)...")
        opt_params = optimize_parameters(train_df)
        print(f"Parâmetros otimizados: gamma1={opt_params[0]:.3f}, gamma2={opt_params[1]:.3f}, gamma3={opt_params[2]:.3f}, gamma6={opt_params[3]:.3f}, delay={opt_params[4]:.1f}, b={opt_params[5]:.1f}")
        gamma1, gamma2, gamma3, gamma6, delay_internal, b_pressure = opt_params
        args.gamma1 = gamma1
        args.gamma2 = gamma2
        args.gamma3 = gamma3
        args.gamma6 = gamma6
        args.delay_internal = delay_internal
        args.b_pressure = b_pressure
        if args.save_params:
            params_to_save = {k: getattr(args, k) for k in ['gamma1','gamma2','gamma3','gamma6','delay_internal','b_pressure','tau_R']}
            with open(args.save_params, 'w') as f:
                json.dump(params_to_save, f, indent=2)
            print(f"💾 Parâmetros salvos em {args.save_params}")

    # Modelo HAC
    train_df['Dst_hac'] = compute_dst_direct(train_df, tau_R=args.tau_R,
                                             gamma1=args.gamma1, gamma2=args.gamma2,
                                             gamma3=args.gamma3, gamma4=args.gamma4,
                                             gamma5=args.gamma5, gamma6=args.gamma6,
                                             threshold=args.threshold,
                                             delay_internal=args.delay_internal,
                                             tau_driver=args.tau_driver,
                                             b_pressure=args.b_pressure,
                                             debug=args.debug)
    test_df['Dst_hac'] = compute_dst_direct(test_df, tau_R=args.tau_R,
                                            gamma1=args.gamma1, gamma2=args.gamma2,
                                            gamma3=args.gamma3, gamma4=args.gamma4,
                                            gamma5=args.gamma5, gamma6=args.gamma6,
                                            threshold=args.threshold,
                                            delay_internal=args.delay_internal,
                                            tau_driver=args.tau_driver,
                                            b_pressure=args.b_pressure,
                                            debug=args.debug)

    # Baseline Burton
    train_df['Dst_burton'] = compute_dst_burton(train_df, tau=args.tau_R, alpha=4.5, delay_internal=args.delay_internal)
    test_df['Dst_burton'] = compute_dst_burton(test_df, tau=args.tau_R, alpha=4.5, delay_internal=args.delay_internal)

    # Métricas (sem suavização para treino, com suavização opcional para validação)
    hac_train = validation_metrics_direct(train_df, pred_col='Dst_hac', smooth_hours=0)
    hac_test = validation_metrics_direct(test_df, pred_col='Dst_hac', smooth_hours=args.smooth_hours)
    burton_train = validation_metrics_direct(train_df, pred_col='Dst_burton', smooth_hours=0)
    burton_test = validation_metrics_direct(test_df, pred_col='Dst_burton', smooth_hours=args.smooth_hours)

    print("\n📊 TREINO (2000-2010) - SEM SUAVIZAÇÃO:")
    print(f"   HAC:     r={hac_train['correlation']:.3f}, R²={hac_train['r2']:.3f}, RMSE={hac_train['rmse']:.2f}, MAE_storm={hac_train['mae_storm']:.2f}")
    print(f"   Burton:  r={burton_train['correlation']:.3f}, R²={burton_train['r2']:.3f}, RMSE={burton_train['rmse']:.2f}, MAE_storm={burton_train['mae_storm']:.2f}")

    print("\n📊 TESTE (2010-2020) - COM SUAVIZAÇÃO ({}h):".format(args.smooth_hours))
    print(f"   HAC:     r={hac_test['correlation']:.3f}, R²={hac_test['r2']:.3f}, RMSE={hac_test['rmse']:.2f}, MAE_storm={hac_test['mae_storm']:.2f}")
    print(f"   Burton:  r={burton_test['correlation']:.3f}, R²={burton_test['r2']:.3f}, RMSE={burton_test['rmse']:.2f}, MAE_storm={burton_test['mae_storm']:.2f}")

    # Salvar resultados do teste
    test_df.to_csv(args.output.replace('.csv', '_test.csv'), index=False)
    print(f"\n💾 Resultados salvos em {args.output.replace('.csv', '_test.csv')}")

if __name__ == "__main__":
    main()
