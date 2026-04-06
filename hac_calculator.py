"""
HAC - Heliospheric Accumulated Coupling (Versão Final Refatorada)
Modelo principal: V * (-Bz) * sin²(θ/2) * sqrt(B_total) * (Pdyn/P0)^{α}
Baselines: (-Bz*V) e Akasofu ε (mesma dinâmica de integração)

Uso:
    python hac_calculator.py --omni data/omni_2000_2020.csv --dst data/dst_kyoto_2000_2020.csv
    python hac_calculator.py --model simple   # baseline (-Bz*V)
    python hac_calculator.py --model akasofu  # baseline Akasofu ε
    python hac_calculator.py --simplify       # remove perda não linear e Pdyn
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
# CONSTANTES FÍSICAS (invariantes)
# ============================
M_P = 1.6726e-27      # kg (massa do próton)
CM3_TO_M3 = 1e6       # conversão cm⁻³ → m⁻³
KMS_TO_MS = 1e3       # conversão km/s → m/s

# ============================
# CLASSE DE CONFIGURAÇÃO
# ============================
class Config:
    """Encapsula todos os parâmetros ajustáveis do modelo."""
    def __init__(self, alpha_pdyn=1/6, simplify=False, model='hac'):
        self.alpha_pdyn = alpha_pdyn
        self.simplify = simplify
        self.model = model

        # Parâmetros de memória
        self.tau_base_hours = 3.0
        self.v_ref = 400.0
        self.tau_power = 0.3
        self.tau_min_hours = 1.0
        self.tau_max_hours = 6.0

        # Termo fonte
        self.use_by_clock = True
        self.use_dynamic_pressure = not simplify   # desliga se simplify=True
        self.p0 = 2.0                              # nPa

        # Adaptação e perda
        self.beta_source = 0.0 if simplify else 0.5
        self.loss_exponent = 1.0 if simplify else 1.5
        self.h0 = 100.0

        # Clipping e suavização
        self.max_source_physical = 10000.0
        self.use_savgol = True
        self.sg_window = 7
        self.sg_order = 2

        # Normalização da fonte (calculada dinamicamente)
        self.source_norm_constant = self._compute_normalization()

    def _compute_normalization(self):
        """Deriva SOURCE_NORM de valores típicos do vento solar."""
        v_typical = 400.0
        bz_typical = 5.0
        b_typical = 7.0
        pdyn_typical = 2.0
        s_typical = (v_typical * bz_typical * np.sqrt(b_typical) *
                     (pdyn_typical / self.p0) ** self.alpha_pdyn)
        hac_quiet_target = 100.0
        return s_typical / hac_quiet_target

    def __repr__(self):
        return (f"Config(model={self.model}, simplify={self.simplify}, "
                f"alpha_pdyn={self.alpha_pdyn:.3f}, norm={self.source_norm_constant:.2f})")

# ============================
# FUNÇÕES AUXILIARES (não dependem de config)
# ============================
def prepare_omni(df):
    df = df.copy()
    df = df.sort_values('time_tag').reset_index(drop=True)
    df['bz_gsm'] = df['bz_gsm'].ffill(limit=2)
    cols = ['bx_gsm', 'by_gsm', 'density', 'speed']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=3)
    df = df.dropna(subset=['bz_gsm', 'speed', 'density'])
    return df

def apply_time_shift(df, hours):
    shifted = df.copy()
    shifted['time_tag'] = shifted['time_tag'] - pd.Timedelta(hours=hours)
    return shifted

def compute_pdyn_nPa(N_cm3, V_kms):
    N_m3 = N_cm3 * CM3_TO_M3
    V_ms = V_kms * KMS_TO_MS
    Pdyn_Pa = M_P * N_m3 * V_ms**2
    return Pdyn_Pa * 1e9

def compute_source_simple(df):
    """Fonte do baseline: -Bz * V (normalizado para escala similar)"""
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    source = np.zeros_like(Bz)
    south = Bz < 0
    source[south] = -Bz[south] * V[south] / 1000.0
    return source

def compute_source_akasofu(df):
    """Fonte de Akasofu: V * B^2 * sin^4(theta/2) (normalizado)"""
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_theta = Bz / (Btot + 1e-10)
    sin_theta2 = np.sqrt((1 - cos_theta) / 2)
    sin4 = sin_theta2**4
    epsilon = V * (Btot**2) * sin4
    return epsilon / 1e6   # normalização empírica

def compute_source_advanced(df, config):
    """Fonte do modelo principal: V * (-Bz) * sin²(θ/2) * sqrt(B_total) * (Pdyn/P0)^{α}"""
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    N = df['density'].values

    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    theta = np.arctan2(By, Bz)
    recon = np.sin(theta / 2)**2 if config.use_by_clock else 1.0
    bz_sul = -np.minimum(Bz, 0)   # positivo quando Bz < 0

    base = V * bz_sul * recon * np.sqrt(Btot)

    if config.use_dynamic_pressure:
        Pdyn = compute_pdyn_nPa(N, V)
        Pdyn_norm = Pdyn / config.p0
        dyn_factor = np.where(Pdyn > 0, Pdyn_norm ** config.alpha_pdyn, 1.0)
        base *= dyn_factor

    base = np.clip(base, 0, config.max_source_physical)
    source = base / config.source_norm_constant
    return source

# ============================
# INTEGRAÇÃO GENÉRICA
# ============================
def integrate_hac(df, source_func, config, calib_factor=1.0):
    """
    Integra qualquer termo fonte usando o mesmo esquema de memória.
    source_func deve aceitar (df, config) e retornar array source.
    """
    times = df['time_tag'].values
    Vsw = df['speed'].values
    source = source_func(df, config)

    dt_seconds = np.zeros(len(times))
    if len(times) > 1:
        time_diffs = np.diff(times)
        dt_seconds[1:] = time_diffs.astype('timedelta64[s]').astype(np.float64)
        median_dt = np.median(dt_seconds[1:]) if len(dt_seconds) > 2 else 60.0
        dt_seconds[0] = median_dt
    else:
        dt_seconds[:] = 60.0

    tau_hours = config.tau_base_hours * (config.v_ref / np.maximum(Vsw, 1.0)) ** config.tau_power
    tau_hours = np.clip(tau_hours, config.tau_min_hours, config.tau_max_hours)
    tau_seconds = tau_hours * 3600.0

    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        dt = dt_seconds[i]
        tau = tau_seconds[i]

        source_eff = source[i]   # sem saturação adicional

        # Perda não linear (τ_eff base)
        if config.loss_exponent != 1.0:
            tau_loss_eff = tau * (1 + (hac[i-1] / config.h0) ** (config.loss_exponent - 1))
        else:
            tau_loss_eff = tau

        # Adaptação à fonte (acelera resposta quando S é alto)
        tau_eff = tau_loss_eff / (1 + config.beta_source * source_eff)

        alpha = np.exp(-dt / tau_eff)
        hac[i] = hac[i-1] * alpha + source_eff * tau_eff * (1 - alpha)
        hac[i] = max(0, hac[i])

    hac_scaled = hac * calib_factor

    # Derivada suavizada
    if config.use_savgol and len(hac_scaled) >= config.sg_window:
        hac_smooth = savgol_filter(hac_scaled, window_length=config.sg_window,
                                   polyorder=config.sg_order)
    else:
        hac_smooth = hac_scaled
    dHAC_dt = np.gradient(hac_smooth, dt_seconds) * 3600.0

    df_out = df.copy()
    df_out['HAC'] = hac_scaled
    df_out['HAC_raw'] = hac
    df_out['HAC_smooth'] = hac_smooth
    df_out['dHAC_dt'] = dHAC_dt
    return df_out

# ============================
# FUNÇÕES DE VALIDAÇÃO
# ============================
def smooth_dst(merged, smooth_hours):
    merged = merged.set_index('time_tag')
    if len(merged) > 1:
        median_dt = merged.index.to_series().diff().median().total_seconds()
        window_size = max(3, int(smooth_hours * 3600 / median_dt))
        if window_size % 2 == 0:
            window_size += 1
        merged['dst_smooth'] = merged['dst'].rolling(window=window_size, center=True, min_periods=1).mean()
    else:
        merged['dst_smooth'] = merged['dst']
    merged = merged.dropna(subset=['dst_smooth'])
    return merged.reset_index()

def transform_xy(x, y, transform_type):
    if transform_type == 'log':
        x_t = np.log1p(x)
        y_t = np.log1p(y)
    elif transform_type == 'sqrt':
        x_t = np.sqrt(x)
        y_t = np.sqrt(y)
    else:
        x_t, y_t = x, y
    return x_t, y_t

def inverse_transform(y_pred_t, transform_type):
    if transform_type == 'log':
        return np.expm1(y_pred_t)
    elif transform_type == 'sqrt':
        return y_pred_t**2
    else:
        return y_pred_t

def compute_metrics(y_true, y_pred, threshold_dst=None, threshold_hac=None, hac_vals=None):
    if len(y_true) < 2:
        return None
    corr = pearsonr(y_true, y_pred)[0] if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    pod = far = csi = None
    if threshold_dst is not None and threshold_hac is not None and hac_vals is not None:
        true_storm = y_true > threshold_dst   # y_true é -dst_smooth
        pred_storm = hac_vals > threshold_hac
        tp = np.sum(true_storm & pred_storm)
        fp = np.sum(~true_storm & pred_storm)
        fn = np.sum(true_storm & ~pred_storm)
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {'correlation': corr, 'r2': r2, 'rmse': rmse, 'mae': mae,
            'pod': pod, 'far': far, 'csi': csi}

def compute_validation_metrics(df_hac, dst_series, predictor, smooth_dst_hours,
                               threshold_dst=None, threshold_hac=None):
    if dst_series is None or len(dst_series) == 0:
        return None
    df_hac = df_hac.copy()
    dst_series = dst_series.copy()
    df_hac['time_tag'] = pd.to_datetime(df_hac['time_tag'], errors='coerce')
    dst_series['time_tag'] = pd.to_datetime(dst_series['time_tag'], errors='coerce')
    needed = ['time_tag', 'HAC', 'dHAC_dt']
    if predictor == 'hac_ma':
        needed.append('HAC_ma')
    for col in needed:
        if col not in df_hac.columns:
            return None
    df_hac = df_hac.dropna(subset=needed)
    dst_series = dst_series.dropna(subset=['time_tag', 'dst'])
    if len(df_hac) < 10 or len(dst_series) < 10:
        return None
    merged = pd.merge_asof(
        df_hac.sort_values('time_tag'),
        dst_series.sort_values('time_tag'),
        on='time_tag',
        direction='backward'
    )
    merged = merged.dropna(subset=['dst', 'HAC', 'dHAC_dt'])
    merged = smooth_dst(merged, smooth_dst_hours)
    merged = merged[merged['dst_smooth'] < 0]
    if len(merged) < 10:
        return None
    if predictor == 'hac':
        y_pred = merged['HAC'].values
        hac_vals = merged['HAC'].values
    elif predictor == 'deriv':
        y_pred = merged['dHAC_dt'].values
        y_pred = np.maximum(0, y_pred)
        hac_vals = y_pred
    elif predictor == 'hac_ma':
        y_pred = merged['HAC_ma'].values
        hac_vals = y_pred
    else:
        return None
    y_true = -merged['dst_smooth'].values
    return compute_metrics(y_true, y_pred,
                           threshold_dst=threshold_dst,
                           threshold_hac=threshold_hac,
                           hac_vals=hac_vals)

def find_optimal_delay(df_hac, dst_series, max_delay, step, smooth_dst_hours, predictor):
    best_corr = -1
    best_delay = 0
    print(f"\n🔍 Buscando atraso ótimo (predictor={predictor})...")
    for delay in np.arange(0, max_delay + step, step):
        shifted_dst = apply_time_shift(dst_series, delay)
        metrics = compute_validation_metrics(df_hac, shifted_dst, predictor,
                                             smooth_dst_hours)
        if metrics and not np.isnan(metrics['correlation']):
            corr = metrics['correlation']
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

def calibrate_hac_nonlinear(df_hac_raw, dst_series, delay, min_hac, min_dst,
                            smooth_dst_hours, predictor, transform):
    shifted_dst = apply_time_shift(dst_series, delay)
    df_hac_raw = df_hac_raw.copy()
    shifted_dst = shifted_dst.copy()
    df_hac_raw['time_tag'] = pd.to_datetime(df_hac_raw['time_tag'])
    shifted_dst['time_tag'] = pd.to_datetime(shifted_dst['time_tag'])
    merged = pd.merge_asof(df_hac_raw.sort_values('time_tag'),
                           shifted_dst.sort_values('time_tag'),
                           on='time_tag', direction='backward')
    merged = merged.dropna(subset=['HAC_raw', 'dHAC_dt', 'dst'])
    merged = smooth_dst(merged, smooth_dst_hours)
    if predictor == 'hac':
        X_col = 'HAC_raw'
    elif predictor == 'deriv':
        X_col = 'dHAC_dt'
    else:
        X_col = 'HAC_raw'
    mask = (merged['dst_smooth'] < 0) & (merged['dst_smooth'].abs() > min_dst) & (merged[X_col] > min_hac)
    if mask.sum() < 5:
        print("⚠️ Pontos insuficientes para calibração.")
        return None, None
    x = merged.loc[mask, X_col].values
    y = -merged.loc[mask, 'dst_smooth'].values
    x_t, y_t = transform_xy(x, y, transform)
    slope, intercept, r_value, p_value, stderr = linregress(x_t, y_t)
    print(f"📊 Calibração não linear (delay={delay}h, transform={transform}):")
    print(f"   f(y) = {slope:.3f} * f(HAC) + {intercept:.3f} (R²={r_value**2:.3f})")
    return slope, intercept

def predict_from_calibration(hac, slope, intercept, transform):
    if transform == 'log':
        return np.expm1(slope * np.log1p(hac) + intercept)
    elif transform == 'sqrt':
        return (slope * np.sqrt(hac) + intercept)**2
    else:
        return slope * hac + intercept

def get_hac_threshold_from_calibration(calib_slope, calib_intercept, target_dst, transform):
    if calib_slope is None or calib_intercept is None:
        return None
    if transform == 'log':
        log1p_hac = (np.log1p(target_dst) - calib_intercept) / calib_slope
        hac_thresh = np.expm1(log1p_hac)
    elif transform == 'sqrt':
        sqrt_hac = (np.sqrt(target_dst) - calib_intercept) / calib_slope
        hac_thresh = sqrt_hac**2
    else:
        hac_thresh = (target_dst - calib_intercept) / calib_slope
    return max(0, hac_thresh)

# ============================
# FIGURA COMPARATIVA
# ============================
def create_figure(df, baseline_simple_vals, baseline_akasofu_vals, dst_df, delay,
                  smooth_dst_hours, predictor, dst_threshold, filename="hac_comparison.png"):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(df['time_tag'], df['HAC'], label='HAC (modelo principal)', color='#d62728', linewidth=2)
    axes[0].plot(df['time_tag'], baseline_simple_vals, label='Baseline (-Bz*V)', color='#1f77b4', linewidth=1.5, alpha=0.7)
    axes[0].plot(df['time_tag'], baseline_akasofu_vals, label='Baseline Akasofu ε', color='#ff7f0e', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Índice')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Comparação dos Modelos (preditor = {predictor})')
    if dst_df is not None:
        if delay > 0:
            dst_shifted = apply_time_shift(dst_df, delay)
        else:
            dst_shifted = dst_df.copy()
        dst_shifted = smooth_dst(dst_shifted, smooth_dst_hours)
        axes[1].plot(dst_shifted['time_tag'], dst_shifted['dst_smooth'], color='black', linewidth=1.5, label='Dst suavizado')
        axes[1].axhline(y=dst_threshold, color='red', linestyle='--', label=f'Limiar tempestade ({dst_threshold} nT)')
        axes[1].set_ylabel('Dst [nT]')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title(f'Dst (atrasado {delay}h)')
    else:
        axes[1].set_visible(False)
    axes[2].plot(df['time_tag'], df['dHAC_dt'], color='#9b59b6', linewidth=1.5, label='dHAC/dt')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('dHAC/dt [1/h]')
    axes[2].set_xlabel('Tempo (UTC)')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figura salva: {filename}")

# ============================
# TESTES DE UNIDADES (CORRIGIDOS)
# ============================
def test_physics(config):
    print("\n🧪 TESTE DE UNIDADES FÍSICAS:")
    print("="*50)
    # Condições típicas (quiet)
    N = 5.0
    V = 400.0
    Bx = 2.0
    By = 3.0
    Bz = -5.0

    # 1. Pdyn
    Pdyn = compute_pdyn_nPa(N, V)
    print(f"1. Pdyn({N} cm⁻³, {V} km/s) = {Pdyn:.2f} nPa")
    assert 1.0 < Pdyn < 2.0, f"Pdyn fora do esperado: {Pdyn}"

    # 2. Fonte do modelo principal em quiet
    test_df = pd.DataFrame({'bx_gsm': [Bx], 'by_gsm': [By], 'bz_gsm': [Bz],
                            'speed': [V], 'density': [N]})
    S = compute_source_advanced(test_df, config)[0]
    print(f"2. Fonte(quiet) = {S:.2f}")
    # Agora a escala física é ~100 (pois HAC_quiet_target=100)
    assert 50 < S < 150, f"Fonte quiet fora do esperado (deveria ser ~100): {S}"

    # 3. Evento extremo (storm)
    N_storm = 30.0
    V_storm = 800.0
    Bz_storm = -20.0
    test_df_storm = pd.DataFrame({'bx_gsm': [Bx], 'by_gsm': [By], 'bz_gsm': [Bz_storm],
                                  'speed': [V_storm], 'density': [N_storm]})
    S_storm = compute_source_advanced(test_df_storm, config)[0]
    print(f"3. Fonte(storm) = {S_storm:.2f}")
    # Verifica amplificação relativa (fator > 5)
    assert S_storm > 5 * S, f"Fonte de tempestade não amplificou: {S_storm} vs {S}"

    # 4. Normalização
    print(f"4. SOURCE_NORM = {config.source_norm_constant:.2f}")
    assert 30 < config.source_norm_constant < 100, f"Normalização fora da faixa: {config.source_norm_constant}"

    print("\n✅ Todos os testes passaram!")

# ============================
# FUNÇÃO PRINCIPAL
# ============================
def main():
    parser = argparse.ArgumentParser(description='HAC Calculator - Modelo Físico com Baselines')
    parser.add_argument('--omni', default='data/omni_2000_2020.csv', help='Arquivo OMNI (sem Dst)')
    parser.add_argument('--dst', default='data/dst_kyoto_2000_2020.csv', help='Arquivo Dst Kyoto')
    parser.add_argument('--output', default='hac_results.csv', help='Arquivo de saída')
    parser.add_argument('--model', default='hac', choices=['hac', 'simple', 'akasofu'],
                        help='Modelo a ser executado: hac (principal), simple (-Bz*V), akasofu (ε)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplifica o modelo: desativa perda não linear, Pdyn e adaptação')
    parser.add_argument('--no_calibrate', action='store_true', help='Pula a calibração (usa fator 1)')
    parser.add_argument('--alpha_pdyn', type=float, default=1/6,
                        help='Expoente da pressão dinâmica (teste: 0.0, 0.5, 1.0)')
    args = parser.parse_args()

    # Criar objeto de configuração
    config = Config(alpha_pdyn=args.alpha_pdyn, simplify=args.simplify, model=args.model)
    print("\n" + "="*70)
    print(f"🛰️  HAC - Modelo: {config.model.upper()}")
    print(f"   Configuração: {config}")
    print("="*70)

    # Teste de unidades (opcional, pode ser desligado com flag)
    test_physics(config)

    # Carregar OMNI
    if not os.path.exists(args.omni):
        print(f"❌ Arquivo OMNI não encontrado: {args.omni}")
        return
    omni = pd.read_csv(args.omni, parse_dates=['time_tag'])
    omni = prepare_omni(omni)
    print(f"✅ OMNI: {len(omni)} pontos, {omni['time_tag'].min()} a {omni['time_tag'].max()}")

    # Carregar Dst Kyoto
    if not os.path.exists(args.dst):
        print(f"❌ Arquivo Dst não encontrado: {args.dst}")
        return
    dst = pd.read_csv(args.dst, parse_dates=['time_tag'])
    dst = dst.sort_values('time_tag')
    print(f"✅ Dst Kyoto: {len(dst)} pontos, {dst['time_tag'].min()} a {dst['time_tag'].max()}")

    # Merge robusto: interpolar Dst na grade temporal do OMNI
    omni = omni.sort_values('time_tag')
    dst = dst.sort_values('time_tag')
    time_grid = omni['time_tag'].copy()
    dst_interp = dst.set_index('time_tag')
    dst_interp = dst_interp.reindex(time_grid, method='nearest', limit=2)
    dst_interp = dst_interp.interpolate(method='time', limit=3)
    dst_interp = dst_interp.reset_index().rename(columns={'index': 'time_tag'})
    df = pd.merge(omni, dst_interp, on='time_tag', how='left')
    df = df.dropna(subset=['dst'])
    print(f"   Dados após merge: {len(df)} pontos, Dst mínimo = {df['dst'].min():.1f} nT")
    if df['dst'].min() > -50:
        print("⚠️  ALERTA: O Dst Kyoto ainda não mostra tempestades fortes!")
        print("   Verifique se o download cobriu anos com eventos (ex.: 2003, 2015).")

    # Selecionar fonte conforme modelo
    if config.model == 'simple':
        source_func = lambda df_, cfg: compute_source_simple(df_)
    elif config.model == 'akasofu':
        source_func = lambda df_, cfg: compute_source_akasofu(df_)
    else:  # 'hac'
        source_func = compute_source_advanced

    # Calcular HAC (primeira passagem sem calibração)
    df = integrate_hac(df, source_func, config, calib_factor=1.0)

    # Calcular baselines (apenas para o modelo principal, para comparação)
    if config.model == 'hac':
        # Baselines usam a mesma dinâmica de integração, mas com fontes diferentes
        df_simple = integrate_hac(df, lambda d, c: compute_source_simple(d), config, calib_factor=1.0)
        df_akasofu = integrate_hac(df, lambda d, c: compute_source_akasofu(d), config, calib_factor=1.0)
        baseline_simple_vals = df_simple['HAC'].values
        baseline_akasofu_vals = df_akasofu['HAC'].values
    else:
        baseline_simple_vals = np.zeros(len(df))
        baseline_akasofu_vals = np.zeros(len(df))

    # Preparar dados para validação
    df_hac = df[['time_tag', 'HAC', 'dHAC_dt']].copy()
    dst_df = df[['time_tag', 'dst']].copy()

    # Parâmetros de validação (fixos, poderiam vir do config)
    max_delay_hours = 12
    delay_step = 1
    smooth_dst_hours = 3
    predictor = 'hac'
    transform = 'log'
    min_dst_calib = 5

    # Encontrar delay ótimo
    best_delay, best_corr = find_optimal_delay(df_hac, dst_df, max_delay_hours, delay_step,
                                               smooth_dst_hours, predictor)

    # Calibração (opcional)
    calib_slope, calib_intercept = 1.0, 0.0
    if not args.no_calibrate:
        df_hac_raw = df[['time_tag', 'HAC_raw', 'dHAC_dt']].copy()
        calib_slope, calib_intercept = calibrate_hac_nonlinear(
            df_hac_raw, dst_df, delay=best_delay, min_hac=0.01, min_dst=min_dst_calib,
            smooth_dst_hours=smooth_dst_hours, predictor=predictor, transform=transform)
        if calib_slope is None:
            calib_slope, calib_intercept = 1.0, 0.0
        else:
            # Ajustar fator linear para visualização (apenas para o gráfico)
            hac_mean = df['HAC_raw'].mean()
            dst_pred_mean = predict_from_calibration(hac_mean, calib_slope, calib_intercept, transform)
            linear_factor = dst_pred_mean / hac_mean if hac_mean > 0 else 1.0
            if not (0.01 < linear_factor < 100):
                linear_factor = 1.0
            df = integrate_hac(df, source_func, config, calib_factor=linear_factor)
            if config.model == 'hac':
                df_simple = integrate_hac(df, lambda d, c: compute_source_simple(d), config, calib_factor=linear_factor)
                df_akasofu = integrate_hac(df, lambda d, c: compute_source_akasofu(d), config, calib_factor=linear_factor)
                baseline_simple_vals = df_simple['HAC'].values
                baseline_akasofu_vals = df_akasofu['HAC'].values
            df_hac = df[['time_tag', 'HAC', 'dHAC_dt']].copy()
            print(f"✅ HAC recalibrado com fator linear aproximado: {linear_factor:.3f}")

    # Calcular limiar HAC para detecção (Dst = -50 nT)
    dst_threshold = -50
    if calib_slope is not None and calib_intercept is not None and not args.no_calibrate:
        hac_threshold = get_hac_threshold_from_calibration(calib_slope, calib_intercept,
                                                           target_dst=abs(dst_threshold),
                                                           transform=transform)
        if hac_threshold is None:
            hac_threshold = np.percentile(df['HAC'].dropna(), 75)
    else:
        hac_threshold = np.percentile(df['HAC'].dropna(), 75)
    print(f"\n🎯 HAC_THRESHOLD (Dst={dst_threshold} nT): {hac_threshold:.2f}")

    # Validação direta com calibração não linear
    print("\n📊 VALIDAÇÃO DIRETA COM CALIBRAÇÃO NÃO LINEAR")
    print("="*50)
    shifted_dst = apply_time_shift(dst_df, best_delay)
    merged = pd.merge_asof(df[['time_tag', 'HAC_raw']].sort_values('time_tag'),
                           shifted_dst.sort_values('time_tag'),
                           on='time_tag', direction='backward')
    merged = merged.dropna(subset=['dst', 'HAC_raw'])
    merged = smooth_dst(merged, smooth_dst_hours)
    merged = merged[merged['dst_smooth'] < 0]
    if len(merged) > 10:
        y_true = -merged['dst_smooth'].values
        y_pred = predict_from_calibration(merged['HAC_raw'].values, calib_slope, calib_intercept, transform)
        metrics = compute_metrics(y_true, y_pred,
                                  threshold_dst=abs(dst_threshold),
                                  threshold_hac=hac_threshold,
                                  hac_vals=merged['HAC_raw'].values)
        if metrics:
            print(f"✅ Modelo principal (predição direta) vs Dst suavizado (delay={best_delay}h):")
            print(f"   Correlação: {metrics['correlation']:.3f}")
            print(f"   R²: {metrics['r2']:.3f}")
            print(f"   RMSE: {metrics['rmse']:.2f}")
            print(f"   MAE: {metrics['mae']:.2f}")
            if metrics['pod'] is not None:
                print(f"   POD: {metrics['pod']:.3f} | FAR: {metrics['far']:.3f} | CSI: {metrics['csi']:.3f}")

    # Validação com HAC calibrado (apenas para referência)
    print("\n📊 VALIDAÇÃO COM HAC CALIBRADO")
    print("="*50)
    shifted_dst = apply_time_shift(dst_df, best_delay)
    metrics_hac = compute_validation_metrics(df_hac, shifted_dst, predictor, smooth_dst_hours,
                                             threshold_dst=abs(dst_threshold),
                                             threshold_hac=hac_threshold)
    if metrics_hac:
        print(f"✅ HAC (calibrado linear) vs Dst suavizado (delay={best_delay}h):")
        print(f"   Correlação: {metrics_hac['correlation']:.3f}")
        print(f"   R²: {metrics_hac['r2']:.3f}")
        print(f"   RMSE: {metrics_hac['rmse']:.2f}")
        print(f"   MAE: {metrics_hac['mae']:.2f}")
        if metrics_hac['pod'] is not None:
            print(f"   POD: {metrics_hac['pod']:.3f} | FAR: {metrics_hac['far']:.3f} | CSI: {metrics_hac['csi']:.3f}")

    # Salvar resultados
    df.to_csv(args.output, index=False)
    print(f"\n💾 Resultados salvos em {args.output}")

    # Gráfico
    create_figure(df, baseline_simple_vals, baseline_akasofu_vals, dst_df, best_delay,
                  smooth_dst_hours, predictor, dst_threshold, "hac_comparison.png")

if __name__ == "__main__":
    main()
