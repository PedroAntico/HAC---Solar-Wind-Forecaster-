#!/usr/bin/env python3
"""
HAC - Heliospheric Accumulated Coupling (Versão Final com Melhorias)

Modelo vetorial de acumulação de energia (Hload, Hrel, Hsat) com:
- Driver puro de Akasofu: ε = V * Btot² * sin⁴(θ/2)  (normalizado)
- Memória exponencial no driver (τ_driver)
- Correção de pressão dinâmica no decaimento (Dst* = Dst - b * sqrt(Pdyn))
- Peso ajustável para Hload, Hrel e C (acoplamento complementar)
- Busca automática de delay

Uso:
    python hac_calculator.py --find_delay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import os
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, linregress
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

# ============================
# HAC VETORIAL (Hload, Hrel, Hsat)
# ============================
def integrate_hac_vectorial(df, driver_type='akasofu'):
    """
    Calcula os estados Hload, Hrel, Hsat e HCI a partir dos dados OMNI.
    driver_type: 'akasofu' (padrão) ou 'hybrid' (anterior).
    """
    times = df['time_tag'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    N = df['density'].values
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values

    # ----- Driver φ(t) -----
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    theta = np.arctan2(By, Bz)
    sin4 = np.sin(theta / 2)**4

    if driver_type == 'akasofu':
        # Akasofu puro: ε = V * Btot² * sin⁴(θ/2)
        phi_raw = V * (Btot**2) * sin4
        phi = phi_raw / 1000.0   # normalização empírica (ε típico ~1000)
    else:
        # Híbrido antigo (preservado para comparação)
        Pdyn = compute_pdyn_nPa(N, V)
        phi = 0.7 * (Pdyn * Btot) + 0.3 * (V * Btot**2 * sin4)
        phi = phi / 1000.0

    # ----- Tempo -----
    dt_sec = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec[1:] = diffs
    dt_sec[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    dt_h = dt_sec / 3600.0

    # ----- Estados -----
    Hload = np.zeros(len(times))
    Hrel  = np.zeros(len(times))
    Hsat  = np.zeros(len(times))

    # ----- Parâmetros (ajustáveis) -----
    alpha_L = 1.2e-5
    beta_L  = 0.15
    alpha_R = 0.08
    beta_R  = 0.25
    eta     = 0.3
    theta_sat = 2.5
    sigma_sat = 0.5

    for i in range(1, len(times)):
        dt = dt_h[i]

        # saturação sigmoidal (baseada em Hload anterior)
        Hsat[i-1] = 1 / (1 + np.exp(-(Hload[i-1] - theta_sat) / sigma_sat))

        # LOADING
        dHload = alpha_L * phi[i] * (1 - Hsat[i-1]) - beta_L * Hload[i-1]

        # RELEASE
        dHrel = alpha_R * Hload[i-1] - beta_R * Hrel[i-1] + eta * Hsat[i-1] * Hload[i-1]

        # integração Euler
        Hload[i] = Hload[i-1] + dHload * dt
        Hrel[i]  = Hrel[i-1]  + dHrel  * dt

        # limites físicos
        Hload[i] = max(0, Hload[i])
        Hrel[i]  = max(0, Hrel[i])

    # último Hsat
    Hsat[-1] = 1 / (1 + np.exp(-(Hload[-1] - theta_sat) / sigma_sat))

    # ----- HCI (acoplamento complementar) -----
    C = np.maximum(-Bz, 0) * (V / 400.0)
    HCI = C + Hload

    out = df.copy()
    out['Hload'] = Hload
    out['Hrel']  = Hrel
    out['Hsat']  = Hsat
    out['HCI']   = HCI
    out['phi']   = phi
    out['C']     = C
    return out

# ============================
# MODELO DST BASEADO NO HAC VETORIAL (COM MELHORIAS)
# ============================
def compute_dst_hac(df, alpha=2.0, tau=10.0, gamma1=1.0, gamma2=0.0,
                    delay_internal=2.0, tau_driver=2.0, b_pressure=6.0):
    """
    Dst a partir dos estados HAC.
    - Driver combinado: γ1*Hload + γ2*Hrel + (1-γ1-γ2)*C
    - Memória exponencial no driver (τ_driver)
    - Correção de pressão dinâmica: Dst* = Dst - b_pressure * sqrt(Pdyn)
    - Equação: dDst/dt = -α * driver_eff - Dst* / τ
    """
    times = df['time_tag'].values
    Hload = df['Hload'].values
    Hrel  = df['Hrel'].values
    C     = df['C'].values
    Pdyn  = compute_pdyn_nPa(df['density'].values, df['speed'].values)

    # Combinação linear dos estados
    gamma3 = 1.0 - gamma1 - gamma2
    driver_raw = gamma1 * Hload + gamma2 * Hrel + gamma3 * C

    # Atraso interno (shift)
    dt_sec_arr = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec_arr[1:] = diffs
    dt_sec_arr[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    dt_h = dt_sec_arr / 3600.0

    if delay_internal > 0:
        delay_steps = int(round(delay_internal * 3600 / np.median(dt_sec_arr[dt_sec_arr>0])))
        if delay_steps > 0:
            driver_delayed = np.zeros_like(driver_raw)
            if delay_steps < len(driver_raw):
                driver_delayed[delay_steps:] = driver_raw[:-delay_steps]
        else:
            driver_delayed = driver_raw
    else:
        driver_delayed = driver_raw

    # Memória exponencial no driver (τ_driver)
    if tau_driver > 0:
        driver_eff = np.zeros_like(driver_delayed)
        for i in range(1, len(driver_eff)):
            decay = np.exp(-dt_h[i] / tau_driver)
            driver_eff[i] = driver_eff[i-1] * decay + driver_delayed[i] * (1 - decay)
        driver = driver_eff
    else:
        driver = driver_delayed

    # Integração do Dst com correção de pressão
    Dst = np.zeros(len(times))
    for i in range(1, len(times)):
        dt = dt_h[i]
        # Dst corrigido pela pressão dinâmica
        Dst_star = Dst[i-1] - b_pressure * np.sqrt(Pdyn[i])
        injection = -alpha * driver[i]
        decay = Dst_star / tau
        dDst = injection - decay
        Dst[i] = Dst[i-1] + dDst * dt
        Dst[i] = max(-500, min(50, Dst[i]))
    return Dst

# ============================
# VALIDAÇÃO
# ============================
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

def compute_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return None
    corr = pearsonr(y_true, y_pred)[0] if np.std(y_true) * np.std(y_pred) > 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'correlation': corr, 'r2': r2, 'rmse': rmse, 'mae': mae}

def validation_metrics(df_model, dst_series, smooth_hours):
    if dst_series is None or len(dst_series) == 0:
        return None
    df_model = df_model.copy()
    dst = dst_series.copy()
    df_model['time_tag'] = pd.to_datetime(df_model['time_tag'])
    dst['time_tag'] = pd.to_datetime(dst['time_tag'])
    df_model = df_model.dropna(subset=['time_tag', 'Dst_model'])
    dst = dst.dropna(subset=['time_tag', 'dst'])
    if len(df_model) < 10 or len(dst) < 10:
        return None
    merged = pd.merge_asof(df_model.sort_values('time_tag'),
                           dst.sort_values('time_tag'),
                           on='time_tag', direction='backward')
    merged = merged.dropna(subset=['dst', 'Dst_model'])
    merged = smooth_dst(merged, smooth_hours)
    y_true = -merged['dst_smooth'].values
    y_pred = -merged['Dst_model'].values
    return compute_metrics(y_true, y_pred)

def find_optimal_delay(df_model, dst_series, max_delay, step, smooth_hours):
    best_corr = -1
    best_delay = 0
    print(f"\n🔍 Buscando atraso ótimo (Dst_model) até {max_delay}h...")
    delays = np.arange(0, max_delay + step, step)
    for delay in delays:
        shifted = apply_time_shift(dst_series, delay)
        m = validation_metrics(df_model, shifted, smooth_hours)
        if m and not np.isnan(m['correlation']):
            corr = m['correlation']
            print(f"   Delay {delay:3.0f}h → r = {corr:.3f}")
            if corr > best_corr:
                best_corr = corr
                best_delay = delay
    if best_corr < 0.1:
        print("⚠️ Correlação máxima muito baixa. Usando delay=0.")
        best_delay = 0
        best_corr = 0.0
    print(f"🔥 Melhor delay: {best_delay:.0f}h (r = {best_corr:.3f})")
    return best_delay, best_corr

# ============================
# GRÁFICO
# ============================
def plot_comparison(df, df_model, dst_df, delay, smooth_hours, fname):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    if 'Hload' in df.columns:
        axes[0].plot(df['time_tag'], df['Hload'], label='Hload', color='#d62728', lw=2)
        axes[0].plot(df['time_tag'], df['Hrel'], label='Hrel', color='#1f77b4', lw=1.5, alpha=0.7)
        axes[0].plot(df['time_tag'], df['Hsat'], label='Hsat', color='#2ca02c', lw=1.5, alpha=0.7)
        axes[0].set_ylabel('HAC')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Estados HAC (Hload, Hrel, Hsat)')
    else:
        axes[0].set_visible(False)
    if dst_df is not None:
        if delay > 0:
            dst_shifted = apply_time_shift(dst_df, delay)
        else:
            dst_shifted = dst_df.copy()
        dst_shifted = smooth_dst(dst_shifted, smooth_hours)
        axes[1].plot(df['time_tag'], df_model['Dst_model'], label='Dst_model', color='blue', lw=1.5)
        axes[1].plot(dst_shifted['time_tag'], dst_shifted['dst_smooth'], 'k', lw=1.5, label='Dst real suavizado')
        axes[1].axhline(-50, color='red', ls='--', label='Limiar tempestade')
        axes[1].set_ylabel('Dst [nT]')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title(f'Comparação Dst_model vs Dst real (delay {delay}h)')
    else:
        axes[1].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figura salva: {fname}")

# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser(description='HAC Calculator - Modelo Vetorial Melhorado')
    parser.add_argument('--omni', default='data/omni_2000_2020.csv')
    parser.add_argument('--dst', default='data/dst_kyoto_2000_2020.csv')
    parser.add_argument('--output', default='hac_results.csv')
    parser.add_argument('--driver_type', default='akasofu', choices=['akasofu', 'hybrid'],
                        help='Tipo de driver: akasofu (puro) ou hybrid')
    parser.add_argument('--alpha', type=float, default=2.0, help='Coeficiente de injeção')
    parser.add_argument('--tau', type=float, default=10.0, help='Constante de decaimento (horas)')
    parser.add_argument('--gamma1', type=float, default=1.0, help='Peso de Hload')
    parser.add_argument('--gamma2', type=float, default=0.0, help='Peso de Hrel')
    parser.add_argument('--delay_internal', type=float, default=2.0, help='Atraso interno (horas)')
    parser.add_argument('--tau_driver', type=float, default=2.0, help='Memória exponencial do driver (horas)')
    parser.add_argument('--b_pressure', type=float, default=6.0, help='Coeficiente de correção de pressão (Dst*)')
    parser.add_argument('--smooth_hours', type=float, default=1, help='Suavização do Dst (horas)')
    parser.add_argument('--find_delay', action='store_true', help='Busca atraso ótimo')
    parser.add_argument('--max_delay', type=int, default=24, help='Máximo delay para busca')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🛰️  HAC - Modelo Vetorial Melhorado")
    print(f"   Driver: {args.driver_type.upper()}")
    print(f"   Parâmetros: alpha={args.alpha}, tau={args.tau}, gamma1={args.gamma1}, gamma2={args.gamma2}")
    print(f"   delay_internal={args.delay_internal}h, tau_driver={args.tau_driver}h, b_pressure={args.b_pressure}")
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

    # Cortar Dst para período do OMNI
    dst = dst[dst['time_tag'] <= omni['time_tag'].max()]
    print(f"   Dst cortado: {len(dst)} pontos")

    # Interpolação robusta
    time_grid = omni[['time_tag']].copy()
    dst_interp = pd.merge_asof(
        time_grid.sort_values('time_tag'),
        dst[['time_tag', 'dst']].sort_values('time_tag'),
        on='time_tag',
        direction='nearest',
        tolerance=pd.Timedelta('2h')
    )
    dst_interp = dst_interp.set_index('time_tag').interpolate(method='time', limit=3).reset_index()
    assert 'dst' in dst_interp.columns, "ERRO: coluna dst sumiu"

    # Merge final
    if 'dst' in omni.columns:
        omni = omni.drop(columns=['dst'])
    df = pd.merge(omni, dst_interp, on='time_tag', how='left', suffixes=('', '_dst'))
    if 'dst_dst' in df.columns:
        df = df.rename(columns={'dst_dst': 'dst'})
    df = df.dropna(subset=['dst'])
    print(f"   Merge final: {len(df)} pontos, Dst min = {df['dst'].min():.1f} nT")
    if df['dst'].min() > -50:
        print("⚠️ ALERTA: Período sem tempestades fortes (Dst > -50 nT).")

    # HAC vetorial
    df = integrate_hac_vectorial(df, driver_type=args.driver_type)
    df['Dst_model'] = compute_dst_hac(df,
                                      alpha=args.alpha,
                                      tau=args.tau,
                                      gamma1=args.gamma1,
                                      gamma2=args.gamma2,
                                      delay_internal=args.delay_internal,
                                      tau_driver=args.tau_driver,
                                      b_pressure=args.b_pressure)

    # Validação
    df_model = df[['time_tag', 'Dst_model']].copy()
    dst_df = df[['time_tag', 'dst']].copy()
    smooth_hours = args.smooth_hours

    best_delay = 0
    if args.find_delay:
        best_delay, best_corr = find_optimal_delay(df_model, dst_df, args.max_delay, 1, smooth_hours)
    else:
        best_delay = 0
        shifted = apply_time_shift(dst_df, best_delay)
        metrics = validation_metrics(df_model, shifted, smooth_hours)
        if metrics:
            best_corr = metrics['correlation']
            print(f"\n📊 Correlação sem delay externo: r = {best_corr:.3f}")

    print("\n📊 VALIDAÇÃO DO MODELO Dst")
    print("="*50)
    shifted = apply_time_shift(dst_df, best_delay)
    metrics = validation_metrics(df_model, shifted, smooth_hours)
    if metrics:
        print(f"✅ Dst_model vs Dst real (delay={best_delay}h):")
        print(f"   Correlação: {metrics['correlation']:.3f}")
        print(f"   R²: {metrics['r2']:.3f}")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   MAE: {metrics['mae']:.2f}")

    # Salvar resultados
    df.to_csv(args.output, index=False)
    print(f"\n💾 Resultados salvos em {args.output}")

    # Gráfico
    plot_comparison(df, df_model, dst_df, best_delay, smooth_hours, "hac_comparison.png")

if __name__ == "__main__":
    main()
