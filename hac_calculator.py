#!/usr/bin/env python3
"""
HAC - Heliospheric Accumulated Coupling (Versão Final com Correção Conceitual)

Modelo Dst baseado em Burton com:
- Campo elétrico de reconexão: Ey = V * max(0, -Bz) * 1e-3  (mV/m)
- Limiar (threshold) e **sem saturação** (Ey_eff = max(0, Ey - threshold))
- Atraso físico de 2h na injeção (shift causal)
- Constante de tempo **contínua**: τ = τ_recovery - (τ_recovery - τ_main) * (Ey_eff/(Ey_eff+1))
- Termo de pressão dinâmica: Dst_star = Dst - b * sqrt(Pdyn)
- **Termo de energia acumulada (HAC) no decaimento**: (Dst_star + γ * HAC) / τ

Uso:
    python hac_calculator.py --dst_model burton --alpha 1.0 --b 6 \
        --tau_main 4 --tau_recovery 12 --delay_hours 2 --gamma 0.05
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
# TERMOS FONTE PARA HAC
# ============================
def compute_source_simple(df, _config=None):
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    source = np.zeros_like(Bz)
    mask = Bz < 0
    source[mask] = -Bz[mask] * V[mask] / 1000.0
    return source

def compute_source_akasofu(df, _config=None):
    Bx, By, Bz = df['bx_gsm'].values, df['by_gsm'].values, df['bz_gsm'].values
    V = df['speed'].values
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_theta = Bz / (Btot + 1e-10)
    sin_theta2 = np.sqrt((1 - cos_theta) / 2)
    sin4 = sin_theta2**4
    return V * (Btot**2) * sin4 / 1e6

def compute_source_advanced(df, config):
    Bx, By, Bz = df['bx_gsm'].values, df['by_gsm'].values, df['bz_gsm'].values
    V, N = df['speed'].values, df['density'].values
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    theta = np.arctan2(By, Bz)
    recon = np.sin(theta / 2)**2 if config.use_by_clock else 1.0
    bz_sul = -np.minimum(Bz, 0)
    base = V * bz_sul * recon * np.sqrt(Btot)
    if config.use_dynamic_pressure:
        Pdyn = compute_pdyn_nPa(N, V)
        dyn_factor = (Pdyn / config.p0) ** config.alpha_pdyn
        dyn_factor = np.where(Pdyn > 0, dyn_factor, 1.0)
        base *= dyn_factor
    base = np.clip(base, 0, config.max_source_physical)
    return base / config.source_norm_constant

# ============================
# INTEGRAÇÃO DO HAC
# ============================
def integrate_hac(df, source_func, config, calib_factor=1.0):
    times = df['time_tag'].values
    Vsw = df['speed'].values
    source = source_func(df, config)

    dt_sec = np.zeros(len(times))
    if len(times) > 1:
        diffs = np.diff(times).astype('timedelta64[s]').astype(float)
        dt_sec[1:] = diffs
        dt_sec[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    else:
        dt_sec[:] = 60.0

    tau_hours = config.tau_base_hours * (config.v_ref / np.maximum(Vsw, 1.0)) ** config.tau_power
    tau_hours = np.clip(tau_hours, config.tau_min_hours, config.tau_max_hours)
    tau_sec = tau_hours * 3600.0

    H = np.zeros(len(times))
    for i in range(1, len(H)):
        dt = dt_sec[i]
        tau = tau_sec[i]

        if config.loss_exponent != 1.0:
            tau_loss = tau * (1 + (H[i-1] / config.h0) ** (config.loss_exponent - 1))
        else:
            tau_loss = tau

        tau_eff = tau_loss / (1 + config.beta_source * source[i])
        alpha = np.exp(-dt / tau_eff)
        H[i] = H[i-1] * alpha + source[i] * (1 - alpha)
        H[i] = max(0, min(H[i], config.max_hac))

    H_scaled = H * calib_factor

    if config.use_savgol and len(H_scaled) >= config.sg_window:
        H_smooth = savgol_filter(H_scaled, config.sg_window, config.sg_order)
    else:
        H_smooth = H_scaled
    dH_dt = np.gradient(H_smooth, dt_sec) * 3600.0

    out = df.copy()
    out['HAC'] = H_scaled
    out['HAC_raw'] = H
    out['HAC_smooth'] = H_smooth
    out['dHAC_dt'] = dH_dt
    return out

# ============================
# MODELO DST BURTON (COM HAC NO DECAIMENTO)
# ============================
def compute_dst_burton(df, tau_main=4.0, tau_recovery=12.0, b=6.0, alpha=1.0,
                       threshold=0.5, delay_hours=2.0, gamma=0.05):
    """
    Modelo Burton com:
        - Ey = V * max(0, -Bz) * 1e-3 (mV/m)
        - Ey_eff = max(0, Ey - threshold)  (sem saturação)
        - Atraso causal na injeção
        - τ contínuo: τ = τ_recovery - (τ_recovery - τ_main) * (Ey_eff/(Ey_eff+1))
        - Pressão dinâmica: Dst_star = Dst - b * sqrt(Pdyn)
        - Decaimento modificado: (Dst_star + γ * HAC) / τ
    """
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    Pdyn = compute_pdyn_nPa(df['density'].values, df['speed'].values)

    # Ey bruto (mV/m)
    Ey_raw = V * np.maximum(0, -Bz) * 1e-3
    Ey_eff = np.maximum(0, Ey_raw - threshold)          # sem saturação extra

    # Atraso causal (shift puro)
    times = df['time_tag'].values
    if len(times) > 1:
        dt_sec = np.median(np.diff(times).astype('timedelta64[s]').astype(float))
    else:
        dt_sec = 3600.0
    delay_steps = int(round(delay_hours * 3600 / dt_sec))
    if delay_steps > 0:
        Ey_eff_shifted = np.zeros_like(Ey_eff)
        if delay_steps < len(Ey_eff):
            Ey_eff_shifted[delay_steps:] = Ey_eff[:-delay_steps]
        Ey_eff = Ey_eff_shifted

    # Integração
    dt_sec_arr = np.zeros(len(times))
    if len(times) > 1:
        diffs = np.diff(times).astype('timedelta64[s]').astype(float)
        dt_sec_arr[1:] = diffs
        dt_sec_arr[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    else:
        dt_sec_arr[:] = 60.0

    Dst = np.zeros(len(times))
    # Se gamma > 0, precisamos do HAC (já calculado)
    if gamma > 0 and 'HAC' not in df.columns:
        raise ValueError("gamma > 0 requer HAC. Execute primeiro integrate_hac.")
    HAC = df['HAC'].values if gamma > 0 else np.zeros(len(times))

    for i in range(1, len(times)):
        dt_hours = dt_sec_arr[i] / 3600.0
        # Dst corrigido pela pressão dinâmica
        Dst_star = Dst[i-1] - b * np.sqrt(Pdyn[i])
        # τ contínuo (transição suave)
        tau = tau_recovery - (tau_recovery - tau_main) * (Ey_eff[i] / (Ey_eff[i] + 1.0))
        # Termo de injeção (Ey) e decaimento com HAC
        injection = -alpha * Ey_eff[i]
        decay = (Dst_star + gamma * HAC[i]) / tau
        dDst = injection - decay
        Dst[i] = Dst[i-1] + dDst * dt_hours
        Dst[i] = max(-500, min(50, Dst[i]))
    return Dst

def compute_dst_hybrid(df, a=0.5, b=0.2):
    return -a * df['HAC'].values - b * df['dHAC_dt'].values

def compute_dst_direct(df):
    return -df['HAC'].values

def compute_dst_deriv(df, k=0.5):
    return -df['HAC'].values + k * df['dHAC_dt'].values

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
    if 'HAC' in df.columns:
        axes[0].plot(df['time_tag'], df['HAC'], label='HAC', color='#d62728', lw=2)
        axes[0].set_ylabel('HAC')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Acumulação de Energia (HAC)')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--omni', default='data/omni_2000_2020.csv')
    parser.add_argument('--dst', default='data/dst_kyoto_2000_2020.csv')
    parser.add_argument('--output', default='hac_results.csv')
    parser.add_argument('--model', default='hac', choices=['hac', 'simple', 'akasofu'])
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument('--dst_model', default='burton', choices=['direct', 'deriv', 'hybrid', 'burton'],
                        help='Modelo Dst: direct, deriv, hybrid, burton')
    # Parâmetros para hybrid
    parser.add_argument('--a', type=float, default=0.5, help='Coeficiente HAC (hybrid)')
    parser.add_argument('--b_hybrid', type=float, default=0.2, help='Coeficiente dH/dt (hybrid)')
    # Parâmetros para deriv
    parser.add_argument('--k', type=float, default=0.5, help='Coeficiente para modelo deriv')
    # Parâmetros para burton (corrigidos)
    parser.add_argument('--tau_main', type=float, default=4.0, help='τ durante injeção (horas)')
    parser.add_argument('--tau_recovery', type=float, default=12.0, help='τ durante recuperação (horas)')
    parser.add_argument('--b', type=float, default=6.0, help='Coeficiente da pressão dinâmica')
    parser.add_argument('--alpha', type=float, default=1.0, help='Fator de escala para Ey')
    parser.add_argument('--threshold', type=float, default=0.5, help='Limiar para Ey (mV/m)')
    parser.add_argument('--delay_hours', type=float, default=2.0, help='Atraso físico da injeção (horas)')
    parser.add_argument('--gamma', type=float, default=0.05,
                        help='Coeficiente de energia acumulada no decaimento (HAC/τ)')
    # Parâmetros gerais
    parser.add_argument('--alpha_pdyn', type=float, default=1/6)
    parser.add_argument('--smooth_hours', type=float, default=1, help='Suavização do Dst (horas)')
    parser.add_argument('--no_calibrate', action='store_true', help='Não aplicar calibração linear')
    parser.add_argument('--find_delay', action='store_true', help='Busca delay externo (não recomendado para Burton)')
    args = parser.parse_args()

    config = Config(alpha_pdyn=args.alpha_pdyn, simplify=args.simplify, model=args.model)
    print("\n" + "="*70)
    print(f"🛰️  HAC - Modelo: {config.model.upper()}")
    print(f"   Configuração: {config}")
    print(f"   Modelo Dst: {args.dst_model}")
    if args.dst_model == 'hybrid':
        print(f"   a = {args.a}, b_hybrid = {args.b_hybrid}")
    elif args.dst_model == 'deriv':
        print(f"   k = {args.k}")
    elif args.dst_model == 'burton':
        print(f"   tau_main = {args.tau_main} h, tau_recovery = {args.tau_recovery} h")
        print(f"   b = {args.b}, alpha = {args.alpha}, threshold = {args.threshold} mV/m")
        print(f"   delay = {args.delay_hours} h, gamma = {args.gamma}")
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

    # Selecionar fonte HAC (necessário se dst_model usa HAC ou se gamma>0)
    need_hac = (args.dst_model in ['direct', 'deriv', 'hybrid']) or (args.dst_model == 'burton' and args.gamma > 0)
    if need_hac:
        if config.model == 'simple':
            source_func = lambda d, c: compute_source_simple(d)
        elif config.model == 'akasofu':
            source_func = lambda d, c: compute_source_akasofu(d)
        else:
            source_func = compute_source_advanced
        df = integrate_hac(df, source_func, config, calib_factor=1.0)

    # Gerar Dst_model conforme escolha
    if args.dst_model == 'direct':
        df['Dst_model'] = compute_dst_direct(df)
    elif args.dst_model == 'deriv':
        df['Dst_model'] = compute_dst_deriv(df, k=args.k)
    elif args.dst_model == 'hybrid':
        df['Dst_model'] = compute_dst_hybrid(df, a=args.a, b=args.b_hybrid)
    else:  # burton
        df['Dst_model'] = compute_dst_burton(df, tau_main=args.tau_main, tau_recovery=args.tau_recovery,
                                             b=args.b, alpha=args.alpha, threshold=args.threshold,
                                             delay_hours=args.delay_hours, gamma=args.gamma)

    # DataFrame para validação
    df_model = df[['time_tag', 'Dst_model']].copy()
    dst_df = df[['time_tag', 'dst']].copy()
    max_delay = 24
    step = 1
    smooth_hours = args.smooth_hours

    # Busca de delay externo (apenas se solicitado)
    best_delay = 0
    if args.find_delay:
        best_delay, best_corr = find_optimal_delay(df_model, dst_df, max_delay, step, smooth_hours)
    else:
        best_delay = 0
        shifted = apply_time_shift(dst_df, best_delay)
        metrics = validation_metrics(df_model, shifted, smooth_hours)
        if metrics:
            best_corr = metrics['correlation']
            print(f"\n📊 Correlação sem delay externo: r = {best_corr:.3f}")

    # Calibração linear (apenas para modelos não físicos)
    if not args.no_calibrate and args.dst_model in ['direct', 'deriv', 'hybrid']:
        factor = calibrate_dst_model(df_model, dst_df, best_delay, smooth_hours)
        if factor != 1.0:
            df['Dst_model'] = df['Dst_model'] * factor
            df_model['Dst_model'] = df_model['Dst_model'] * factor
            print(f"✅ Dst_model calibrado com fator {factor:.3f}")

    # Validação final
    print("\n📊 VALIDAÇÃO DO MODELO Dst")
    print("="*50)
    shifted = apply_time_shift(dst_df, best_delay)
    metrics = validation_metrics(df_model, shifted, smooth_hours)
    if metrics:
        print(f"✅ Dst_model vs Dst real (delay externo={best_delay}h):")
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
