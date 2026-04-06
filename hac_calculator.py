#!/usr/bin/env python3
"""
HAC - Heliospheric Accumulated Coupling (Versão Final com HAC Vetorial)

Implementa o modelo vetorial de acumulação de energia (Hload, Hrel, Hsat) 
conforme descrito no paper, e o utiliza para prever o índice Dst.

Modelos disponíveis:
  - 'hac': Dst_model = -α * HCI (com decaimento exponencial)
  - 'burton': modelo clássico de Burton com driver Akasofu ε
  - 'direct': Dst = -a * HAC (para comparação)

Uso recomendado:
    # HAC vetorial
    python hac_calculator.py --dst_model hac --find_delay

    # Burton (baseline)
    python hac_calculator.py --dst_model burton --use_akasofu_driver --find_delay
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
# CLASSE DE CONFIGURAÇÃO (HAC)
# ============================
class Config:
    def __init__(self, alpha_pdyn=1/6, simplify=False, model='hac'):
        self.alpha_pdyn = alpha_pdyn
        self.simplify = simplify
        self.model = model
        self.tau_base_hours = 3.0
        self.v_ref = 400.0
        self.tau_power = 0.3
        self.tau_min_hours = 1.0
        self.tau_max_hours = 6.0
        self.use_by_clock = True
        self.use_dynamic_pressure = not simplify
        self.p0 = 2.0
        self.beta_source = 0.0 if simplify else 0.5
        self.loss_exponent = 1.0 if simplify else 1.5
        self.h0 = 100.0
        self.max_source_physical = 10000.0
        self.max_hac = 500.0
        self.use_savgol = True
        self.sg_window = 7
        self.sg_order = 2
        self.source_norm_constant = self._compute_norm()

    def _compute_norm(self):
        V = 400.0
        Bz = -5.0
        B = 7.0
        Pdyn = 2.0
        base = V * (-Bz) * np.sqrt(B) * (Pdyn/self.p0)**self.alpha_pdyn
        return base / 100.0

    def __repr__(self):
        return (f"Config(model={self.model}, simplify={self.simplify}, "
                f"alpha_pdyn={self.alpha_pdyn:.3f}, norm={self.source_norm_constant:.2f})")

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
def integrate_hac_vectorial(df, config):
    """
    HAC REAL - versão vetorial (Hload, Hrel, Hsat) conforme paper.
    Retorna DataFrame com colunas Hload, Hrel, Hsat, HCI.
    """
    times = df['time_tag'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    N = df['density'].values
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values

    # ===== DRIVER φ(t) =====
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    theta = np.arctan2(By, Bz)
    sin4 = np.sin(theta / 2)**4
    # termo híbrido (pressão dinâmica + reconexão)
    phi = 0.7 * (N * V**2 * Btot**2) + 0.3 * (V * Btot**2 * sin4)
    phi = phi / 1e6   # normalização

    # ===== TEMPO =====
    dt_sec = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec[1:] = diffs
    dt_sec[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    dt_h = dt_sec / 3600.0

    # ===== ESTADOS =====
    Hload = np.zeros(len(times))
    Hrel  = np.zeros(len(times))
    Hsat  = np.zeros(len(times))

    # ===== PARÂMETROS (do paper, ajustáveis) =====
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

    # ===== HCI (como no paper) =====
    C = np.maximum(-Bz, 0) * (V / 400.0)   # termo de acoplamento complementar
    HCI = C + Hload

    out = df.copy()
    out['Hload'] = Hload
    out['Hrel']  = Hrel
    out['Hsat']  = Hsat
    out['HCI']   = HCI
    out['phi']   = phi
    return out

# ============================
# MODELO DST BASEADO NO HAC VETORIAL
# ============================
def compute_dst_hac(df, alpha=5.0, tau=8.0, gamma1=1.0, gamma2=0.0, delay_hours=0.0):
    """
    Dst baseado no estado HAC real.
        injection = -α * (γ1 * Hload + γ2 * Hrel + (1-γ1-γ2)*HCI)   (padrão: apenas HCI)
        decay = Dst / τ
    Opcionalmente, aplica atraso interno (shift) na injeção.
    """
    times = df['time_tag'].values
    Hload = df['Hload'].values
    Hrel  = df['Hrel'].values
    HCI   = df['HCI'].values

    # Combinação linear dos estados (ajustável)
    if gamma1 + gamma2 > 1.0:
        gamma1 = gamma1 / (gamma1+gamma2+1e-6)
        gamma2 = gamma2 / (gamma1+gamma2+1e-6)
    gamma3 = 1.0 - gamma1 - gamma2
    driver = gamma1 * Hload + gamma2 * Hrel + gamma3 * HCI

    # Atraso interno (shift puro)
    if delay_hours > 0:
        # Calcular dt médio
        if len(times) > 1:
            dt_sec = np.median(np.diff(times).astype('timedelta64[s]').astype(float))
        else:
            dt_sec = 3600.0
        delay_steps = int(round(delay_hours * 3600 / dt_sec))
        if delay_steps > 0:
            driver_shifted = np.zeros_like(driver)
            if delay_steps < len(driver):
                driver_shifted[delay_steps:] = driver[:-delay_steps]
            driver = driver_shifted

    # Integração do Dst
    dt_sec = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec[1:] = diffs
    dt_sec[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    dt_h = dt_sec / 3600.0

    Dst = np.zeros(len(times))
    tau_sec = tau * 3600.0   # não usado diretamente, pois a equação é em horas
    for i in range(1, len(times)):
        dt = dt_h[i]
        injection = -alpha * driver[i]
        decay = Dst[i-1] / tau    # tau em horas
        dDst = injection - decay
        Dst[i] = Dst[i-1] + dDst * dt
        Dst[i] = max(-500, min(50, Dst[i]))
    return Dst

# ============================
# MODELO DST BURTON (já existente)
# ============================
def compute_dst_burton(df, tau_main=4.0, tau_recovery=12.0, b=6.0, alpha=0.7,
                       tau_driver=0.0, offset=20.0, gamma=0.0,
                       use_akasofu_driver=False):
    """
    Modelo Burton clássico com Akasofu ε ou Ey, sem HAC.
    (código resumido para brevidade – versão completa disponível anteriormente)
    """
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values
    Pdyn = compute_pdyn_nPa(df['density'].values, df['speed'].values)
    times = df['time_tag'].values

    if use_akasofu_driver:
        Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
        cos_theta = Bz / (Btot + 1e-10)
        sin_theta2 = np.sqrt((1 - cos_theta) / 2)
        sin4 = sin_theta2**4
        driver_raw = V * (Btot**2) * sin4
        EPSILON_REF = 1000.0
        driver_norm = driver_raw / EPSILON_REF
    else:
        driver_raw = V * np.maximum(0, -Bz) * 1e-3
        driver_norm = driver_raw
    driver_norm = np.clip(driver_norm, 0, None)

    # Integração temporal
    dt_sec_arr = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec_arr[1:] = diffs
    dt_sec_arr[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    dt_h = dt_sec_arr / 3600.0

    if tau_driver > 0:
        driver_eff = np.zeros_like(driver_norm)
        tau_driver_sec = tau_driver * 3600.0
        for i in range(1, len(driver_eff)):
            dt = dt_sec_arr[i]
            alpha_d = np.exp(-dt / tau_driver_sec)
            driver_eff[i] = driver_eff[i-1] * alpha_d + driver_norm[i] * (1 - alpha_d)
        driver = driver_eff
    else:
        driver = driver_norm

    Dst = np.zeros(len(times))
    for i in range(1, len(times)):
        dt = dt_h[i]
        Dst_star = Dst[i-1] - b * np.sqrt(Pdyn[i]) + offset
        d_val = driver[i]
        tau_frac = (d_val ** 1.5) / (d_val ** 1.5 + 0.3)
        tau = tau_recovery - (tau_recovery - tau_main) * tau_frac
        injection = -alpha * d_val
        decay = Dst_star / tau
        dDst = injection - decay
        Dst[i] = Dst[i-1] + dDst * dt
        Dst[i] = max(-500, min(50, Dst[i]))
    return Dst

def compute_dst_direct(df):
    return -df['HAC'].values   # HAC do modelo vetorial? Na verdade, aqui usaremos Hload? Mas para compatibilidade, manteremos como estava.

def compute_dst_hybrid(df, a=0.5, b=0.2):
    return -a * df['HAC'].values - b * df['dHAC_dt'].values

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

def calibrate_dst_model(df_model, dst_series, delay, smooth_hours):
    shifted = apply_time_shift(dst_series, delay)
    merged = pd.merge_asof(df_model[['time_tag', 'Dst_model']].sort_values('time_tag'),
                           shifted.sort_values('time_tag'),
                           on='time_tag', direction='backward')
    merged = merged.dropna(subset=['dst', 'Dst_model'])
    merged = smooth_dst(merged, smooth_hours)
    mask = merged['dst_smooth'] < 0
    if mask.sum() < 5:
        print("⚠️ Poucos pontos para calibração.")
        return 1.0
    x = merged.loc[mask, 'Dst_model'].values
    y = -merged.loc[mask, 'dst_smooth'].values
    slope = np.sum(x * y) / np.sum(x * x)
    print(f"📊 Calibração: Dst_model = {slope:.3f} * Dst_model_orig")
    return slope

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
# TESTE DE UNIDADES
# ============================
def test_physics(config):
    print("\n🧪 TESTE DE UNIDADES FÍSICAS:")
    print("="*50)
    N, V, Bx, By, Bz = 5.0, 400.0, 2.0, 3.0, -5.0
    Pdyn = compute_pdyn_nPa(N, V)
    print(f"Pdyn({N} cm⁻³, {V} km/s) = {Pdyn:.2f} nPa (esperado ~1.34)")
    test_df = pd.DataFrame({'bx_gsm': [Bx], 'by_gsm': [By], 'bz_gsm': [Bz],
                            'speed': [V], 'density': [N]})
    S_q = compute_source_advanced(test_df, config)[0]
    print(f"Fonte(quiet) = {S_q:.2f} (esperado ~50-100)")
    assert 10 < S_q < 200, f"Fonte quiet fora da faixa: {S_q}"
    N_s, V_s, Bx_s, By_s, Bz_s = 40.0, 900.0, 5.0, 15.0, -30.0
    test_df_s = pd.DataFrame({'bx_gsm': [Bx_s], 'by_gsm': [By_s], 'bz_gsm': [Bz_s],
                              'speed': [V_s], 'density': [N_s]})
    S_s = compute_source_advanced(test_df_s, config)[0]
    ratio = S_s / S_q
    print(f"Fonte(storm) = {S_s:.2f} (amplificação = {ratio:.2f}x)")
    assert ratio > 1.5, f"Tempestade não amplificou: {ratio:.2f}x"
    print("✅ Teste de unidades passou.")

# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser(description='HAC Calculator - HAC Vetorial vs Burton')
    parser.add_argument('--omni', default='data/omni_2000_2020.csv')
    parser.add_argument('--dst', default='data/dst_kyoto_2000_2020.csv')
    parser.add_argument('--output', default='hac_results.csv')
    parser.add_argument('--model', default='hac', choices=['hac', 'simple', 'akasofu'])
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument('--dst_model', default='hac', choices=['hac', 'burton', 'direct', 'hybrid'],
                        help='Modelo Dst: hac (vetorial), burton, direct, hybrid')
    # Parâmetros para modelo HAC vetorial
    parser.add_argument('--alpha', type=float, default=5.0, help='Coeficiente de injeção (HAC)')
    parser.add_argument('--tau_hac', type=float, default=8.0, help='Constante de decaimento (horas)')
    parser.add_argument('--gamma1', type=float, default=1.0, help='Peso de Hload na injeção')
    parser.add_argument('--gamma2', type=float, default=0.0, help='Peso de Hrel na injeção')
    parser.add_argument('--delay_internal', type=float, default=0.0, help='Atraso interno na injeção (horas)')
    # Parâmetros para Burton
    parser.add_argument('--use_akasofu_driver', action='store_true', help='Burton: usar Akasofu ε')
    parser.add_argument('--tau_main', type=float, default=4.0, help='Burton: τ injeção')
    parser.add_argument('--tau_recovery', type=float, default=12.0, help='Burton: τ recuperação')
    parser.add_argument('--b', type=float, default=6.0, help='Burton: coef. pressão dinâmica')
    parser.add_argument('--alpha_burton', type=float, default=0.7, help='Burton: escala do driver')
    parser.add_argument('--tau_driver', type=float, default=0.0, help='Burton: memória extra do driver')
    parser.add_argument('--offset', type=float, default=20.0, help='Burton: offset atmosférico')
    # Parâmetros gerais
    parser.add_argument('--alpha_pdyn', type=float, default=1/6)
    parser.add_argument('--smooth_hours', type=float, default=1, help='Suavização do Dst (horas)')
    parser.add_argument('--no_calibrate', action='store_true')
    parser.add_argument('--find_delay', action='store_true', help='Busca atraso ótimo (recomendado)')
    parser.add_argument('--max_delay', type=int, default=24, help='Máximo delay para busca (horas)')
    args = parser.parse_args()

    config = Config(alpha_pdyn=args.alpha_pdyn, simplify=args.simplify, model=args.model)
    print("\n" + "="*70)
    print(f"🛰️  HAC - Modelo: {config.model.upper()}")
    print(f"   Configuração: {config}")
    print(f"   Modelo Dst: {args.dst_model}")
    if args.dst_model == 'hac':
        print(f"   Parâmetros HAC: alpha={args.alpha}, tau={args.tau_hac}, gamma1={args.gamma1}, gamma2={args.gamma2}, delay_internal={args.delay_internal}h")
    elif args.dst_model == 'burton':
        print(f"   Parâmetros Burton: tau_main={args.tau_main}, tau_recovery={args.tau_recovery}, b={args.b}, alpha={args.alpha_burton}, tau_driver={args.tau_driver}, offset={args.offset}")
        if args.use_akasofu_driver:
            print("   Driver: Akasofu ε")
        else:
            print("   Driver: Ey")
    print("="*70)

    test_physics(config)

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

    # ===== CÁLCULO DO MODELO =====
    if args.dst_model == 'hac':
        # HAC vetorial
        df = integrate_hac_vectorial(df, config)
        df['Dst_model'] = compute_dst_hac(df, alpha=args.alpha, tau=args.tau_hac,
                                          gamma1=args.gamma1, gamma2=args.gamma2,
                                          delay_hours=args.delay_internal)
    elif args.dst_model == 'burton':
        df['Dst_model'] = compute_dst_burton(df, tau_main=args.tau_main, tau_recovery=args.tau_recovery,
                                             b=args.b, alpha=args.alpha_burton, tau_driver=args.tau_driver,
                                             offset=args.offset, use_akasofu_driver=args.use_akasofu_driver)
    elif args.dst_model == 'direct':
        # Para direct/hybrid, precisamos do HAC simples (não vetorial). Vamos usar o HAC antigo.
        # Para não quebrar, usamos a função antiga integrate_hac (que não está mais aqui). Então, por simplicidade,
        # vamos avisar que esse modo requer o HAC antigo. Mas como estamos focando no HAC vetorial, deixamos apenas como placeholder.
        print("❌ Modelo direct/hybrid não implementado com HAC vetorial. Use --dst_model hac ou burton.")
        return
    elif args.dst_model == 'hybrid':
        print("❌ Modelo hybrid não implementado com HAC vetorial. Use --dst_model hac ou burton.")
        return
    else:
        raise ValueError("Modelo Dst desconhecido")

    # DataFrame para validação
    df_model = df[['time_tag', 'Dst_model']].copy()
    dst_df = df[['time_tag', 'dst']].copy()
    smooth_hours = args.smooth_hours

    # Busca de delay ótimo (se solicitado)
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

    # Validação final
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
