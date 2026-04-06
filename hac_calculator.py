#!/usr/bin/env python3
"""
HAC - Heliospheric Accumulated Coupling (Versão Final Ajustada)
Modelo com escala física real: S_quiet ≈ 50-100, HAC limitado, busca de delay até 24h,
preditor derivativo (opcional), suavização ajustável do Dst.
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
# CLASSE DE CONFIGURAÇÃO
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
        # Normalização: S_quiet ≈ 50-100 (física)
        self.source_norm_constant = self._compute_norm()

    def _compute_norm(self):
        V = 400.0
        Bz = -5.0
        B = 7.0
        Pdyn = 2.0
        base = V * (-Bz) * np.sqrt(B) * (Pdyn/self.p0)**self.alpha_pdyn
        # Desejamos que S_quiet seja ~100, então divisor = base/100
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
# TERMOS FONTE
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
# INTEGRAÇÃO (sem multiplicação por tau_eff)
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
        H[i] = H[i-1] * alpha + source[i] * (1 - alpha)   # correção crucial
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
# VALIDAÇÃO COM OPÇÕES DE PREDITOR E SMOOTHING
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

def compute_metrics(y_true, y_pred, th_dst=None, th_hac=None, hac_vals=None):
    if len(y_true) < 2:
        return None
    corr = pearsonr(y_true, y_pred)[0] if np.std(y_true) * np.std(y_pred) > 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pod = far = csi = None
    if th_dst is not None and th_hac is not None and hac_vals is not None:
        true_storm = y_true > th_dst
        pred_storm = hac_vals > th_hac
        tp = np.sum(true_storm & pred_storm)
        fp = np.sum(~true_storm & pred_storm)
        fn = np.sum(true_storm & ~pred_storm)
        pod = tp / (tp+fn) if (tp+fn) > 0 else 0.0
        far = fp / (tp+fp) if (tp+fp) > 0 else 0.0
        csi = tp / (tp+fp+fn) if (tp+fp+fn) > 0 else 0.0
    return {'correlation': corr, 'r2': r2, 'rmse': rmse, 'mae': mae,
            'pod': pod, 'far': far, 'csi': csi}

def validation_metrics(df_hac, dst_series, predictor, smooth_hours, th_dst=None, th_hac=None):
    if dst_series is None or len(dst_series) == 0:
        return None
    df_hac = df_hac.copy()
    dst = dst_series.copy()
    df_hac['time_tag'] = pd.to_datetime(df_hac['time_tag'])
    dst['time_tag'] = pd.to_datetime(dst['time_tag'])
    needed = ['time_tag', 'HAC', 'dHAC_dt']
    if predictor == 'hac_ma':
        needed.append('HAC_ma')
    df_hac = df_hac.dropna(subset=needed)
    dst = dst.dropna(subset=['time_tag', 'dst'])
    if len(df_hac) < 10 or len(dst) < 10:
        return None
    merged = pd.merge_asof(df_hac.sort_values('time_tag'),
                           dst.sort_values('time_tag'),
                           on='time_tag', direction='backward')
    merged = merged.dropna(subset=['dst', 'HAC', 'dHAC_dt'])
    merged = smooth_dst(merged, smooth_hours)
    merged = merged[merged['dst_smooth'] < 0]
    if len(merged) < 10:
        return None
    if predictor == 'hac':
        y_pred = merged['HAC'].values
        hac_vals = merged['HAC'].values
    elif predictor == 'deriv':
        y_pred = np.maximum(0, merged['dHAC_dt'].values)
        hac_vals = y_pred
    elif predictor == 'hac_ma':
        y_pred = merged['HAC_ma'].values
        hac_vals = y_pred
    else:
        return None
    y_true = -merged['dst_smooth'].values
    return compute_metrics(y_true, y_pred, th_dst, th_hac, hac_vals)

def find_optimal_delay(df_hac, dst_series, max_delay, step, smooth_hours, predictor, test_invert=False):
    best_corr = -1
    best_delay = 0
    print(f"\n🔍 Buscando atraso ótimo (predictor={predictor}, smooth={smooth_hours}h) até {max_delay}h...")
    delays = np.arange(0, max_delay + step, step)
    for delay in delays:
        shifted = apply_time_shift(dst_series, delay)
        m = validation_metrics(df_hac, shifted, predictor, smooth_hours)
        if m and not np.isnan(m['correlation']):
            corr = m['correlation']
            print(f"   Delay {delay:3.0f}h → r = {corr:.3f}")
            if test_invert:
                # testar correlação com sinal invertido
                df_inv = df_hac.copy()
                df_inv['HAC'] = -df_hac['HAC']
                m_inv = validation_metrics(df_inv, shifted, predictor, smooth_hours)
                if m_inv and m_inv['correlation'] > corr:
                    print(f"       → sinal invertido daria r = {m_inv['correlation']:.3f} (melhor)")
            if corr > best_corr:
                best_corr = corr
                best_delay = delay
    if best_corr < 0.1:
        print("⚠️ Correlação máxima muito baixa. Usando delay=0.")
        best_delay = 0
        best_corr = 0.0
    print(f"🔥 Melhor delay: {best_delay:.0f}h (r = {best_corr:.3f})")
    return best_delay, best_corr

def calibrate_nonlinear(df_raw, dst_series, delay, min_hac, min_dst,
                        smooth_hours, predictor, transform='log1p'):
    shifted = apply_time_shift(dst_series, delay)
    df_raw = df_raw.copy()
    shifted = shifted.copy()
    df_raw['time_tag'] = pd.to_datetime(df_raw['time_tag'])
    shifted['time_tag'] = pd.to_datetime(shifted['time_tag'])
    merged = pd.merge_asof(df_raw.sort_values('time_tag'),
                           shifted.sort_values('time_tag'),
                           on='time_tag', direction='backward')
    merged = merged.dropna(subset=['HAC_raw', 'dHAC_dt', 'dst'])
    merged = smooth_dst(merged, smooth_hours)
    if predictor == 'hac':
        col = 'HAC_raw'
    elif predictor == 'deriv':
        col = 'dHAC_dt'
    else:
        col = 'HAC_raw'
    mask = (merged['dst_smooth'] < 0) & (merged['dst_smooth'].abs() > min_dst) & (merged[col] > min_hac)
    if mask.sum() < 5:
        print("⚠️ Pontos insuficientes para calibração.")
        return None, None
    x = merged.loc[mask, col].values
    y = -merged.loc[mask, 'dst_smooth'].values
    if transform == 'log1p':
        xt = np.log1p(x)
        yt = np.log1p(y)
    else:
        xt, yt = x, y
    slope, intercept, r_value, _, _ = linregress(xt, yt)
    print(f"📊 Calibração (delay={delay}h, transform={transform}):")
    print(f"   f(y) = {slope:.3f} * f(HAC) + {intercept:.3f} (R²={r_value**2:.3f})")
    return slope, intercept

def predict_from_calibration(hac, slope, intercept, transform='log1p'):
    if transform == 'log1p':
        return np.expm1(slope * np.log1p(hac) + intercept)
    else:
        return slope * hac + intercept

def hac_threshold_from_calib(slope, intercept, target_dst, transform='log1p'):
    if slope is None or intercept is None:
        return None
    if transform == 'log1p':
        return np.expm1((np.log1p(target_dst) - intercept) / slope)
    else:
        return (target_dst - intercept) / slope

# ============================
# GRÁFICO
# ============================
def plot_comparison(df, simple_vals, akasofu_vals, dst_df, delay, smooth_hours, predictor, th_dst_neg, fname):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(df['time_tag'], df['HAC'], label='HAC (principal)', color='#d62728', lw=2)
    axes[0].plot(df['time_tag'], simple_vals, label='Baseline (-Bz*V)', color='#1f77b4', lw=1.5, alpha=0.7)
    axes[0].plot(df['time_tag'], akasofu_vals, label='Baseline Akasofu ε', color='#ff7f0e', lw=1.5, alpha=0.7)
    axes[0].set_ylabel('Índice')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Comparação de Modelos (predictor={predictor})')
    if dst_df is not None:
        if delay > 0:
            dst_shifted = apply_time_shift(dst_df, delay)
        else:
            dst_shifted = dst_df.copy()
        dst_shifted = smooth_dst(dst_shifted, smooth_hours)
        axes[1].plot(dst_shifted['time_tag'], dst_shifted['dst_smooth'], 'k', lw=1.5, label='Dst suavizado')
        axes[1].axhline(th_dst_neg, color='red', ls='--', label=f'Limiar tempestade ({th_dst_neg} nT)')
        axes[1].set_ylabel('Dst [nT]')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title(f'Dst (atrasado {delay}h)')
    else:
        axes[1].set_visible(False)
    axes[2].plot(df['time_tag'], df['dHAC_dt'], color='#9b59b6', lw=1.5, label='dHAC/dt')
    axes[2].axhline(0, color='black', ls='--', alpha=0.5)
    axes[2].set_ylabel('dHAC/dt [1/h]')
    axes[2].set_xlabel('Tempo (UTC)')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figura salva: {fname}")

# ============================
# TESTE DE UNIDADES (CORRIGIDO)
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
    print(f"Fonte(quiet) = {S_q:.2f} (esperado ~50-100, compatível com normalização /100)")
    # Agora aceita valores entre 10 e 200 (escala física real)
    assert 10 < S_q < 200, f"Fonte quiet fora da faixa esperada (10-200): {S_q}"
    # Teste de amplificação com tempestade realística
    N_s, V_s, Bx_s, By_s, Bz_s = 40.0, 900.0, 5.0, 15.0, -30.0
    test_df_s = pd.DataFrame({'bx_gsm': [Bx_s], 'by_gsm': [By_s], 'bz_gsm': [Bz_s],
                              'speed': [V_s], 'density': [N_s]})
    S_s = compute_source_advanced(test_df_s, config)[0]
    ratio = S_s / S_q
    print(f"Fonte(storm) = {S_s:.2f} (amplificação = {ratio:.2f}x)")
    assert ratio > 1.5, f"Tempestade não amplificou o suficiente: {ratio:.2f}x"
    print("✅ Teste de unidades passou (escala física consistente).")

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
    parser.add_argument('--no_calibrate', action='store_true')
    parser.add_argument('--alpha_pdyn', type=float, default=1/6)
    parser.add_argument('--predictor', default='hac', choices=['hac', 'deriv'],
                        help='Preditor para validação: hac (acumulado) ou deriv (taxa)')
    parser.add_argument('--smooth_hours', type=float, default=1,
                        help='Suavização do Dst em horas (0 = sem suavização)')
    parser.add_argument('--test_invert', action='store_true',
                        help='Testar correlação com HAC invertido')
    args = parser.parse_args()

    config = Config(alpha_pdyn=args.alpha_pdyn, simplify=args.simplify, model=args.model)
    print("\n" + "="*70)
    print(f"🛰️  HAC - Modelo: {config.model.upper()}")
    print(f"   Configuração: {config}")
    print(f"   Preditor: {args.predictor}, Suavização Dst: {args.smooth_hours}h")
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

    # Selecionar fonte
    if config.model == 'simple':
        source_func = lambda d, c: compute_source_simple(d)
    elif config.model == 'akasofu':
        source_func = lambda d, c: compute_source_akasofu(d)
    else:
        source_func = compute_source_advanced

    df = integrate_hac(df, source_func, config, calib_factor=1.0)

    # Baselines (precisam da coluna dHAC_dt)
    if config.model == 'hac':
        df_simple = integrate_hac(df, lambda d, c: compute_source_simple(d), config, 1.0)
        df_akasofu = integrate_hac(df, lambda d, c: compute_source_akasofu(d), config, 1.0)
        simple_vals = df_simple['HAC'].values
        akasofu_vals = df_akasofu['HAC'].values
        # Derivadas para baselines
        dt_sec = np.diff(df['time_tag'].values).astype('timedelta64[s]').astype(float)
        dt_sec = np.insert(dt_sec, 0, dt_sec[0] if len(dt_sec) > 0 else 60.0)
        dH_simple = np.gradient(simple_vals, dt_sec) * 3600.0
        dH_akasofu = np.gradient(akasofu_vals, dt_sec) * 3600.0
    else:
        simple_vals = akasofu_vals = np.zeros(len(df))
        dH_simple = dH_akasofu = np.zeros(len(df))

    # Preparar DataFrames para validação
    df_hac = df[['time_tag', 'HAC', 'dHAC_dt']].copy()
    dst_df = df[['time_tag', 'dst']].copy()
    max_delay = 24
    step = 1
    smooth_hours = args.smooth_hours
    predictor = args.predictor
    test_invert = args.test_invert

    best_delay, best_corr = find_optimal_delay(df_hac, dst_df, max_delay, step, smooth_hours, predictor, test_invert)

    # Calibração (apenas se delay > 0)
    calib_slope, calib_intercept = 1.0, 0.0
    if not args.no_calibrate and best_delay > 0:
        df_raw = df[['time_tag', 'HAC_raw', 'dHAC_dt']].copy()
        sl, ic = calibrate_nonlinear(df_raw, dst_df, best_delay, 0.01, 5,
                                     smooth_hours, predictor, 'log1p')
        if sl is not None:
            calib_slope, calib_intercept = sl, ic
            # recalcular HAC com fator linear aproximado
            if predictor == 'hac':
                pred_raw = df['HAC_raw'].values
            else:
                pred_raw = df['dHAC_dt'].values
            hac_mean = pred_raw.mean()
            dst_pred = predict_from_calibration(hac_mean, calib_slope, calib_intercept, 'log1p')
            linear_factor = dst_pred / hac_mean if hac_mean > 0 else 1.0
            if not (0.01 < linear_factor < 100):
                linear_factor = 1.0
            df = integrate_hac(df, source_func, config, linear_factor)
            if config.model == 'hac':
                df_simple = integrate_hac(df, lambda d, c: compute_source_simple(d), config, linear_factor)
                df_akasofu = integrate_hac(df, lambda d, c: compute_source_akasofu(d), config, linear_factor)
                simple_vals = df_simple['HAC'].values
                akasofu_vals = df_akasofu['HAC'].values
                dt_sec = np.diff(df['time_tag'].values).astype('timedelta64[s]').astype(float)
                dt_sec = np.insert(dt_sec, 0, dt_sec[0] if len(dt_sec) > 0 else 60.0)
                dH_simple = np.gradient(simple_vals, dt_sec) * 3600.0
                dH_akasofu = np.gradient(akasofu_vals, dt_sec) * 3600.0
            df_hac = df[['time_tag', 'HAC', 'dHAC_dt']].copy()
            print(f"✅ HAC recalibrado com fator linear: {linear_factor:.3f}")

    # Limiar HAC para Dst = -50
    th_dst = 50
    if not args.no_calibrate and calib_slope != 1.0 and best_delay > 0:
        if predictor == 'hac':
            hac_th = hac_threshold_from_calib(calib_slope, calib_intercept, th_dst, 'log1p')
        else:
            # Para derivada, usamos percentil 90
            hac_th = np.percentile(df['dHAC_dt'].dropna(), 90)
        hac_th = hac_th if hac_th is not None else np.percentile(df['HAC'].dropna(), 75)
    else:
        if predictor == 'hac':
            hac_th = np.percentile(df['HAC'].dropna(), 75)
        else:
            hac_th = np.percentile(df['dHAC_dt'].dropna(), 75)
    print(f"\n🎯 THRESHOLD (|Dst| = {th_dst} nT, predictor={predictor}): {hac_th:.2f}")

    # Validação direta
    print(f"\n📊 VALIDAÇÃO DIRETA (predictor={predictor}, delay={best_delay}h, smooth={smooth_hours}h)")
    print("="*50)
    shifted = apply_time_shift(dst_df, best_delay)
    if predictor == 'hac':
        pred_col = 'HAC_raw'
    else:
        pred_col = 'dHAC_dt'
    merged = pd.merge_asof(df[['time_tag', pred_col]].sort_values('time_tag'),
                           shifted.sort_values('time_tag'),
                           on='time_tag', direction='backward').dropna(subset=['dst', pred_col])
    merged = smooth_dst(merged, smooth_hours)
    merged = merged[merged['dst_smooth'] < 0]
    metrics_direct = None
    if len(merged) > 10:
        y_true = -merged['dst_smooth'].values
        if predictor == 'hac':
            y_pred = predict_from_calibration(merged[pred_col].values, calib_slope, calib_intercept, 'log1p')
        else:
            # Para derivada, não aplicamos calibração (já está em escala)
            y_pred = merged[pred_col].values
        hac_vals = merged[pred_col].values
        metrics_direct = compute_metrics(y_true, y_pred, th_dst, hac_th, hac_vals)
        if metrics_direct:
            print(f"✅ Modelo principal vs Dst suavizado (delay={best_delay}h):")
            print(f"   Correlação: {metrics_direct['correlation']:.3f}")
            print(f"   R²: {metrics_direct['r2']:.3f}")
            print(f"   RMSE: {metrics_direct['rmse']:.2f}")
            print(f"   MAE: {metrics_direct['mae']:.2f}")
            if metrics_direct['pod'] is not None:
                print(f"   POD: {metrics_direct['pod']:.3f} | FAR: {metrics_direct['far']:.3f} | CSI: {metrics_direct['csi']:.3f}")

    # Comparação com baselines
    if config.model == 'hac' and dst_df is not None:
        print("\n📊 COMPARAÇÃO COM BASELINES")
        print("="*50)
        # Baseline simples (usando HAC ou derivada conforme predictor)
        if predictor == 'hac':
            simple_pred = simple_vals
            akasofu_pred = akasofu_vals
        else:
            simple_pred = dH_simple
            akasofu_pred = dH_akasofu
        df_simple_eval = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': simple_pred, 'dHAC_dt': dH_simple})
        metrics_simple = validation_metrics(df_simple_eval, shifted, predictor, smooth_hours, th_dst, hac_th)
        if metrics_simple:
            print(f"Baseline (-Bz*V): Correlação={metrics_simple['correlation']:.3f}, R²={metrics_simple['r2']:.3f}")
        df_akasofu_eval = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': akasofu_pred, 'dHAC_dt': dH_akasofu})
        metrics_akasofu = validation_metrics(df_akasofu_eval, shifted, predictor, smooth_hours, th_dst, hac_th)
        if metrics_akasofu:
            print(f"Baseline Akasofu ε: Correlação={metrics_akasofu['correlation']:.3f}, R²={metrics_akasofu['r2']:.3f}")
        if metrics_direct:
            print(f"Seu modelo HAC: Correlação={metrics_direct['correlation']:.3f}, R²={metrics_direct['r2']:.3f}")

    # Salvar resultados
    df.to_csv(args.output, index=False)
    print(f"\n💾 Resultados salvos em {args.output}")

    # Gráfico
    plot_comparison(df, simple_vals, akasofu_vals, dst_df, best_delay,
                    smooth_hours, predictor, -th_dst, "hac_comparison.png")

if __name__ == "__main__":
    main()
