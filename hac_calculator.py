#!/usr/bin/env python3
"""
HAC - Heliospheric Accumulated Coupling (Versão Final Corrigida)

Modelo Dst baseado em Burton com:
- Driver físico: VBs = V * max(-Bz,0) / 1000
- Termos de memória: Hload, Hrel, C
- Equação: dDst/dt = -driver - Dst/τ + b_pressure * sqrt(Pdyn)
- Atraso interno no driver (delay_internal)
- Normalização opcional do driver
- Validação treino/teste com calibração linear (opcional)

Uso recomendado:
    python hac_calculator.py --train_test --delay_internal 2 --gamma6 0.02 --b_pressure 7 --normalize_driver
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import os
import json
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

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
# FUNÇÕES AUXILIARES
# ============================
def prepare_omni(df):
    df = df.copy().sort_values('time_tag').reset_index(drop=True)
    if 'dst' in df.columns:
        df = df.drop(columns=['dst'])
    df['bz_gsm'] = df['bz_gsm'].ffill(limit=2)
    for col in ['bx_gsm', 'by_gsm', 'density', 'speed']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=3)
    return df.dropna(subset=['bz_gsm', 'speed', 'density'])

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
# MODELO DST DIRETO (COM SINAL CORRETO E PRESSÃO)
# ============================
def compute_dst_direct(df, tau_R=6.0, gamma1=0.5, gamma2=0.1, gamma3=0.1,
                       gamma4=0.0, gamma5=0.0, gamma6=0.02, threshold=1.5,
                       delay_internal=2.0, tau_driver=0.0, b_pressure=7.0,
                       normalize_driver=True, debug=False):
    """
    Modelo Dst com equação física:
        dDst/dt = -driver - Dst/τ + b_pressure * sqrt(Pdyn)
    driver = gamma1*Hload + gamma2*Hrel + gamma3*C + gamma4*interaction + gamma5*extra + gamma6*VBs_normalizado
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

    # Driver físico: VBs normalizado (escala ~ centenas)
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

    if debug:
        print(f"\n🔍 DEBUG ANTES DA NORMALIZAÇÃO:")
        print(f"   driver_raw: min={np.min(driver_raw):.3f}, max={np.max(driver_raw):.3f}, mean={np.mean(driver_raw):.3f}")
        print(f"   VBs: min={np.min(VBs):.3f}, max={np.max(VBs):.3f}, mean={np.mean(VBs):.3f}")

    # Normalização do driver (recomendada)
    if normalize_driver:
        pos = driver_raw[driver_raw > 0]
        if len(pos) > 0:
            scale = np.percentile(pos, 95)
            print(f"   scale (p95): {scale:.3f}")
        else:
            scale = 1.0
        driver_raw = driver_raw / (scale + 1e-6)
        if debug:
            print(f"   driver_raw após normalização: min={np.min(driver_raw):.3f}, max={np.max(driver_raw):.3f}")

    # Clipping para estabilidade
    driver_raw = np.clip(driver_raw, 0, 50)

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
    tau_eff = tau_R / (1 + 0.5 * C)   # k_tau fixo = 0.5

    # Equação diferencial correta: dDst/dt = -driver - Dst/τ + b_pressure * sqrt(Pdyn)
    Dst = np.zeros(len(times))
    for i in range(1, len(times)):
        dt = dt_h[i]
        dDst = -driver[i] - Dst[i-1] / tau_eff[i] + b_pressure * np.sqrt(Pdyn[i])
        Dst[i] = Dst[i-1] + dDst * dt
        # Limite físico
        Dst[i] = max(-500, min(50, Dst[i]))

    if debug:
        print(f"\n🔍 DEBUG APÓS INTEGRAÇÃO:")
        print(f"   driver: min={np.min(driver):.3f}, max={np.max(driver):.3f}, mean={np.mean(driver):.3f}")
        print(f"   tau_eff: min={np.min(tau_eff):.3f}, max={np.max(tau_eff):.3f}, mean={np.mean(tau_eff):.3f}")
        print(f"   Dst_raw: min={np.min(Dst):.3f}, max={np.max(Dst):.3f}, mean={np.mean(Dst):.3f}, std={np.std(Dst):.3f}")

    return Dst

# ============================
# VALIDAÇÃO COM CALIBRAÇÃO LINEAR (OPCIONAL)
# ============================
def validation_metrics_calibrated(train_df, test_df, pred_col='Dst_raw', smooth_hours=1):
    train_smooth = smooth_dst(train_df[['time_tag', 'dst']], smooth_hours)
    train_aligned = pd.merge_asof(
        train_smooth.sort_values('time_tag'),
        train_df[['time_tag', pred_col]].sort_values('time_tag'),
        on='time_tag', direction='nearest'
    ).dropna(subset=['dst_smooth', pred_col])
    X_train = train_aligned[pred_col].values.reshape(-1, 1)
    y_train = train_aligned['dst_smooth'].values

    corr_raw = pearsonr(X_train.ravel(), y_train)[0] if len(X_train) > 1 else np.nan
    print(f"\n🔍 CORRELAÇÃO RAW (treino, Dst_raw vs Dst_real): {corr_raw:.3f}")
    print(f"   X_train (Dst_raw): min={np.min(X_train):.3f}, max={np.max(X_train):.3f}, std={np.std(X_train):.3f}")
    print(f"   y_train (Dst_real): min={np.min(y_train):.3f}, max={np.max(y_train):.3f}, std={np.std(y_train):.3f}")

    reg = LinearRegression().fit(X_train, y_train)
    coef = reg.coef_[0]
    interc = reg.intercept_
    print(f"\n📐 Calibração linear (treino): Dst_real = {coef:.6f} * Dst_raw + {interc:.2f}")

    train_df['Dst_calib'] = coef * train_df[pred_col] + interc
    test_df['Dst_calib']  = coef * test_df[pred_col]  + interc

    train_metrics = validation_metrics_direct(train_df, pred_col='Dst_calib', smooth_hours=smooth_hours)
    test_metrics = validation_metrics_direct(test_df, pred_col='Dst_calib', smooth_hours=smooth_hours)
    return train_metrics, test_metrics, coef, interc

def validation_metrics_direct(df, dst_col='dst', pred_col='Dst_calib', smooth_hours=1):
    df_smooth = smooth_dst(df[['time_tag', dst_col]], smooth_hours)
    df_aligned = pd.merge_asof(
        df_smooth.sort_values('time_tag'),
        df[['time_tag', pred_col]].sort_values('time_tag'),
        on='time_tag', direction='nearest'
    ).dropna()
    y_true = df_aligned['dst_smooth'].values
    y_pred = df_aligned[pred_col].values
    corr = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'correlation': corr, 'r2': r2, 'rmse': rmse, 'mae': mae}

# ============================
# DIAGNÓSTICOS
# ============================
def plot_scatter(train_aligned, test_aligned, train_metrics, test_metrics, fname='scatter_train_test.png'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(train_aligned['dst_smooth'], train_aligned['Dst_calib'], alpha=0.1, s=1)
    plt.plot([-500, 50], [-500, 50], 'r--', label='y=x')
    plt.xlabel('Dst Real (nT)')
    plt.ylabel('Dst Previsto (nT)')
    plt.title(f'Treino (r={train_metrics["correlation"]:.3f}, R²={train_metrics["r2"]:.3f})')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(test_aligned['dst_smooth'], test_aligned['Dst_calib'], alpha=0.1, s=1)
    plt.plot([-500, 50], [-500, 50], 'r--', label='y=x')
    plt.xlabel('Dst Real (nT)')
    plt.ylabel('Dst Previsto (nT)')
    plt.title(f'Teste (r={test_metrics["correlation"]:.3f}, R²={test_metrics["r2"]:.3f})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✅ Scatter plot salvo: {fname}")

def plot_residuals_by_magnitude(test_aligned, fname='residuals_magnitude.png'):
    test_aligned = test_aligned.copy()
    bins = [-np.inf, -200, -100, -50, 0]
    labels = ['Extremo (<-200)', 'G3 (-200 a -100)', 'G1/G2 (-100 a -50)', 'Quiet (-50 a 0)']
    test_aligned['bin'] = pd.cut(test_aligned['dst_smooth'], bins=bins, labels=labels, right=False)
    stats = []
    for label in labels:
        group = test_aligned[test_aligned['bin'] == label]
        if len(group) > 0:
            mae = np.mean(np.abs(group['dst_smooth'] - group['Dst_calib']))
            stats.append((label, mae, len(group)))
    plt.figure(figsize=(10, 6))
    categories = [s[0] for s in stats]
    mae_vals = [s[1] for s in stats]
    counts = [s[2] for s in stats]
    bars = plt.bar(categories, mae_vals, color='skyblue')
    for bar, cnt in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'N={cnt}', ha='center', fontsize=9)
    plt.ylabel('MAE (nT)')
    plt.title('Erro Absoluto Médio por Intensidade da Tempestade')
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✅ Gráfico de resíduos salvo: {fname}")

def plot_event(df, event_name, start, end, pred_col='Dst_calib', fname=None):
    mask = (df['time_tag'] >= start) & (df['time_tag'] <= end)
    event = df[mask].copy()
    if len(event) == 0:
        print(f"⚠️ Evento {event_name} não encontrado.")
        return
    plt.figure(figsize=(12, 5))
    plt.plot(event['time_tag'], event['dst'], 'k-', label='Dst Real', lw=2)
    plt.plot(event['time_tag'], event[pred_col], 'b-', label='Dst Previsto (calibrado)', lw=1.5)
    plt.axhline(-50, color='orange', ls='--', label='G1 (-50 nT)')
    plt.axhline(-100, color='red', ls='--', label='G3 (-100 nT)')
    plt.ylabel('Dst [nT]')
    plt.title(f'{event_name} (Dst min = {event["dst"].min():.1f} nT)')
    plt.legend()
    plt.grid(True)
    if fname is None:
        fname = f"{event_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✅ Evento plotado: {fname}")

# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser(description='HAC - Modelo Dst Corrigido (Sinal + Pressão)')
    parser.add_argument('--omni', default='data/omni_2000_2020.csv')
    parser.add_argument('--dst', default='data/dst_kyoto_2000_2020.csv')
    parser.add_argument('--output', default='hac_results.csv')
    parser.add_argument('--driver_type', default='akasofu', choices=['akasofu', 'hybrid'])
    # Parâmetros do modelo
    parser.add_argument('--tau_R', type=float, default=6.0, help='Constante de tempo base (horas)')
    parser.add_argument('--gamma1', type=float, default=0.5, help='Peso de Hload')
    parser.add_argument('--gamma2', type=float, default=0.1, help='Peso de Hrel')
    parser.add_argument('--gamma3', type=float, default=0.1, help='Peso de C')
    parser.add_argument('--gamma4', type=float, default=0.0, help='Peso da interação Hload*C')
    parser.add_argument('--gamma5', type=float, default=0.0, help='Peso do termo extra (limiar)')
    parser.add_argument('--gamma6', type=float, default=0.02, help='Peso do driver físico VBs')
    parser.add_argument('--threshold', type=float, default=1.5)
    parser.add_argument('--delay_internal', type=float, default=2.0, help='Atraso interno (horas)')
    parser.add_argument('--tau_driver', type=float, default=0.0, help='Memória exponencial do driver')
    parser.add_argument('--b_pressure', type=float, default=7.0, help='Coeficiente da pressão dinâmica')
    parser.add_argument('--normalize_driver', action='store_true', help='Normalizar o driver (recomendado)')
    parser.add_argument('--smooth_hours', type=float, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train_test', action='store_true')
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
    print("🛰️  HAC - Modelo Dst Corrigido (Sinal + Pressão Dinâmica)")
    print(f"   tau_R = {args.tau_R} h, delay_internal = {args.delay_internal} h")
    print(f"   gamma1={args.gamma1}, gamma2={args.gamma2}, gamma3={args.gamma3}, gamma6={args.gamma6}")
    print(f"   b_pressure = {args.b_pressure}, normalize_driver = {args.normalize_driver}")
    print("="*70)

    # Carregar OMNI
    if not os.path.exists(args.omni):
        print(f"❌ OMNI não encontrado: {args.omni}")
        return
    omni = pd.read_csv(args.omni, parse_dates=['time_tag'])
    omni = prepare_omni(omni)
    print(f"✅ OMNI: {len(omni)} pontos, {omni['time_tag'].min()} a {omni['time_tag'].max()}")

    # Carregar Dst Kyoto
    if not os.path.exists(args.dst):
        print(f"❌ Dst não encontrado: {args.dst}")
        return
    dst = pd.read_csv(args.dst, parse_dates=['time_tag']).sort_values('time_tag')
    print(f"✅ Dst Kyoto: {len(dst)} pontos, {dst['time_tag'].min()} a {dst['time_tag'].max()}")
    dst = dst[dst['time_tag'] <= omni['time_tag'].max()]
    print(f"   Dst cortado: {len(dst)} pontos")

    # Interpolação do Dst na grade do OMNI
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
    if df['dst'].min() > -50:
        print("⚠️ ALERTA: Período sem tempestades fortes (Dst > -50 nT).")

    # Integração HAC (driver)
    df = integrate_hac_vectorial(df, driver_type=args.driver_type)

    if not args.train_test:
        df['Dst_raw'] = compute_dst_direct(df, tau_R=args.tau_R,
                                           gamma1=args.gamma1, gamma2=args.gamma2,
                                           gamma3=args.gamma3, gamma4=args.gamma4,
                                           gamma5=args.gamma5, gamma6=args.gamma6,
                                           threshold=args.threshold,
                                           delay_internal=args.delay_internal,
                                           tau_driver=args.tau_driver,
                                           b_pressure=args.b_pressure,
                                           normalize_driver=args.normalize_driver,
                                           debug=args.debug)
        df.to_csv(args.output, index=False)
        print(f"💾 Resultados salvos em {args.output}")
        return

    # ========== MODO TREINO/TESTE ==========
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
    train_df, test_df = split_data(df_raw, split_date='2010-01-01')
    print(f"Treino: {len(train_df)} pontos ({train_df['time_tag'].min()} a {train_df['time_tag'].max()})")
    print(f"Teste: {len(test_df)} pontos ({test_df['time_tag'].min()} a {test_df['time_tag'].max()})")

    # Integrar HAC separadamente
    train_df = integrate_hac_vectorial(train_df, driver_type=args.driver_type)
    test_df = integrate_hac_vectorial(test_df, driver_type=args.driver_type)

    train_df['Dst_raw'] = compute_dst_direct(train_df, tau_R=args.tau_R,
                                             gamma1=args.gamma1, gamma2=args.gamma2,
                                             gamma3=args.gamma3, gamma4=args.gamma4,
                                             gamma5=args.gamma5, gamma6=args.gamma6,
                                             threshold=args.threshold,
                                             delay_internal=args.delay_internal,
                                             tau_driver=args.tau_driver,
                                             b_pressure=args.b_pressure,
                                             normalize_driver=args.normalize_driver,
                                             debug=args.debug)
    test_df['Dst_raw'] = compute_dst_direct(test_df, tau_R=args.tau_R,
                                            gamma1=args.gamma1, gamma2=args.gamma2,
                                            gamma3=args.gamma3, gamma4=args.gamma4,
                                            gamma5=args.gamma5, gamma6=args.gamma6,
                                            threshold=args.threshold,
                                            delay_internal=args.delay_internal,
                                            tau_driver=args.tau_driver,
                                            b_pressure=args.b_pressure,
                                            normalize_driver=args.normalize_driver,
                                            debug=args.debug)

    # Calibração linear (opcional)
    train_metrics, test_metrics, coef, interc = validation_metrics_calibrated(
        train_df, test_df, pred_col='Dst_raw', smooth_hours=args.smooth_hours
    )

    # Para plots, aplicar calibração também no dataframe completo
    df['Dst_raw'] = compute_dst_direct(df, tau_R=args.tau_R,
                                       gamma1=args.gamma1, gamma2=args.gamma2,
                                       gamma3=args.gamma3, gamma4=args.gamma4,
                                       gamma5=args.gamma5, gamma6=args.gamma6,
                                       threshold=args.threshold,
                                       delay_internal=args.delay_internal,
                                       tau_driver=args.tau_driver,
                                       b_pressure=args.b_pressure,
                                       normalize_driver=args.normalize_driver,
                                       debug=args.debug)
    df['Dst_calib'] = coef * df['Dst_raw'] + interc

    print("\n📊 TREINO (2000-2010) - CALIBRADO:")
    if train_metrics:
        print(f"   Correlação: {train_metrics['correlation']:.3f}")
        print(f"   R²: {train_metrics['r2']:.3f}")
        print(f"   RMSE: {train_metrics['rmse']:.2f}")
        print(f"   MAE: {train_metrics['mae']:.2f}")

    print("\n📊 TESTE (2010-2020) - CALIBRADO:")
    if test_metrics:
        print(f"   Correlação: {test_metrics['correlation']:.3f}")
        print(f"   R²: {test_metrics['r2']:.3f}")
        print(f"   RMSE: {test_metrics['rmse']:.2f}")
        print(f"   MAE: {test_metrics['mae']:.2f}")

    # Alinhar dados para gráficos
    train_smooth = smooth_dst(train_df[['time_tag', 'dst']], args.smooth_hours)
    train_aligned = pd.merge_asof(
        train_smooth.sort_values('time_tag'),
        train_df[['time_tag', 'Dst_calib']].sort_values('time_tag'),
        on='time_tag', direction='nearest'
    ).dropna(subset=['dst_smooth', 'Dst_calib'])

    test_smooth = smooth_dst(test_df[['time_tag', 'dst']], args.smooth_hours)
    test_aligned = pd.merge_asof(
        test_smooth.sort_values('time_tag'),
        test_df[['time_tag', 'Dst_calib']].sort_values('time_tag'),
        on='time_tag', direction='nearest'
    ).dropna(subset=['dst_smooth', 'Dst_calib'])

    plot_scatter(train_aligned, test_aligned, train_metrics, test_metrics)
    plot_residuals_by_magnitude(test_aligned)
    plot_event(df, 'Halloween Storm 2003', '2003-10-28', '2003-11-02', pred_col='Dst_calib')
    plot_event(df, 'St. Patrick Storm 2015', '2015-03-17', '2015-03-19', pred_col='Dst_calib')
    plot_event(df, 'Quase Carrington 2012', '2012-07-22', '2012-07-24', pred_col='Dst_calib')

    test_df.to_csv(args.output.replace('.csv', '_test_calib.csv'), index=False)
    print(f"\n💾 Resultados do teste salvos em {args.output.replace('.csv', '_test_calib.csv')}")

if __name__ == "__main__":
    main()
