import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

# ============================
# CONFIGURAÇÃO PRINCIPAL
# ============================
MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"
DST_FILE = "data/dst_data.csv"          # Opcional: colunas 'time_tag', 'dst'

# ============================
# PARÂMETROS DO MODELO FÍSICO (AJUSTÁVEIS)
# ============================
# Memória (τ) e variação com velocidade
TAU_BASE_HOURS = 2.0
V_REF = 400.0
TAU_POWER = 0.3
TAU_MIN_HOURS = 1.0
TAU_MAX_HOURS = 5.0

# Termo fonte – densidade e ângulo de clock (acoplamento real)
USE_AKASOFU = False
USE_DENSITY = True
USE_BY_CLOCK = True

# Escala do termo fonte (agora neutra; calibração fará o ajuste)
SOURCE_SCALE_FACTOR = 1.0          # Deixe 1.0 para deixar a calibração fazer o ajuste
MAX_SOURCE = 5000.0                # Clipping de segurança (ainda pode ser útil)

# Expoente de perda (1.0 = linear, >1.0 = supra-linear)
LOSS_EXPONENT = 1.2                # Teste: 1.0, 1.2, 1.5

# Saturação física: S_eff = S / (1 + H / H_sat)
# Se H_SATURATION <= 0, a saturação é desativada.
H_SATURATION = 1000.0              # Ajuste conforme necessário

# Suavização da derivada (para análise, não usado no núcleo)
USE_SAVITZKY_GOLAY = True
SG_WINDOW = 7
SG_ORDER = 2

# Limiares para validação de tempestade
DST_THRESHOLD = -50                # nT – tempestade se Dst < -50
# O limiar HAC será calculado dinamicamente como mediana dos HAC nos momentos de Dst < -50
USE_DYNAMIC_HAC_THRESHOLD = True

# Parâmetros de validação
MAX_DELAY_HOURS = 12
DELAY_STEP_HOURS = 1
CALIBRATE_WITH_DELAY = True

# Suavização do Dst na validação (rolling window em horas)
DST_SMOOTH_HOURS = 3

# Min |Dst| para pontos de calibração (reduzido para capturar mais eventos)
MIN_DST_CALIB = 5

# ============================
# FUNÇÕES AUXILIARES
# ============================
def load_omni_data(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
        for col in headers:
            if col != 'time_tag':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"✅ {filepath}: {len(df)} registros")
        return df
    except Exception as e:
        print(f"❌ Erro em {filepath}: {e}")
        return None

def load_dst_data(filepath):
    try:
        df = pd.read_csv(filepath, parse_dates=['time_tag'])
        df = df.sort_values('time_tag')
        df['dst'] = pd.to_numeric(df['dst'], errors='coerce')
        df = df.dropna(subset=['dst'])
        print(f"✅ DST: {len(df)} registros")
        return df
    except Exception as e:
        print(f"⚠️ DST não carregado: {e}")
        return None

def prepare_merged_data(mag_df, plasma_df):
    df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
    df = df.sort_values('time_tag').reset_index(drop=True)
    if 'bz_gsm' in df.columns:
        df['bz_gsm'] = df['bz_gsm'].ffill(limit=2)
    cols_to_interpolate = ['bx_gsm', 'by_gsm', 'density', 'speed']
    for col in cols_to_interpolate:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=3)
    df = df.dropna(subset=['bz_gsm', 'speed', 'density'])
    return df

def apply_time_shift(df, hours):
    shifted = df.copy()
    shifted['time_tag'] = shifted['time_tag'] - pd.Timedelta(hours=hours)
    return shifted

# ============================
# TERMOS FONTE (COM DENSIDADE E ÂNGULO DE CLOCK)
# ============================
def compute_source_simplified(Bx, By, Bz, V, N, use_density, use_by_clock):
    source = np.zeros_like(Bz)
    south_mask = Bz < 0
    if not np.any(south_mask):
        return source

    base = -Bz[south_mask] * V[south_mask]

    if use_density:
        base *= np.sqrt(N[south_mask])

    if use_by_clock:
        theta = np.arctan2(By[south_mask], Bz[south_mask])
        recon_factor = np.sin(theta / 2)**2
        source[south_mask] = base * recon_factor
    else:
        source[south_mask] = base

    source = np.clip(source, 0, MAX_SOURCE)
    source = source / SOURCE_SCALE_FACTOR
    return source

def compute_source(df, use_akasofu=USE_AKASOFU, use_density=USE_DENSITY, use_by_clock=USE_BY_CLOCK):
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    N = df['density'].values
    if use_akasofu:
        Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
        cos_theta = Bz / (Btot + 1e-10)
        sin_theta2 = np.sqrt((1 - cos_theta) / 2)
        sin4 = sin_theta2**4
        epsilon = V * (Btot**2) * sin4
        epsilon = np.clip(epsilon, 0, MAX_SOURCE * 1e6)
        return epsilon / 1e6
    else:
        return compute_source_simplified(Bx, By, Bz, V, N, use_density, use_by_clock)

# ============================
# MODELOS BASELINE (COM PERDA NÃO LINEAR E SATURAÇÃO)
# ============================
def compute_baseline_simple(df, tau_base, v_ref, power, tau_min, tau_max,
                            loss_exp=LOSS_EXPONENT, h_sat=H_SATURATION):
    source = np.zeros_like(df['bz_gsm'].values)
    south = df['bz_gsm'].values < 0
    source[south] = -df['bz_gsm'].values[south] * df['speed'].values[south] / 1000.0
    source = np.clip(source, 0, MAX_SOURCE / 1000.0)
    times = df['time_tag'].values
    Vsw = df['speed'].values
    dt_seconds = np.zeros(len(times))
    if len(times) > 1:
        time_diffs = np.diff(times)
        dt_seconds[1:] = time_diffs.astype('timedelta64[s]').astype(np.float64)
        median_dt = np.median(dt_seconds[1:]) if len(dt_seconds) > 2 else 60.0
        dt_seconds[0] = median_dt
    else:
        dt_seconds[:] = 60.0

    tau_hours = tau_base * (v_ref / np.maximum(Vsw, 1.0)) ** power
    tau_hours = np.clip(tau_hours, tau_min, tau_max)
    tau_seconds = tau_hours * 3600.0

    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        # Aplica saturação na fonte (se h_sat > 0)
        if h_sat > 0:
            source_eff = source[i] / (1 + hac[i-1] / h_sat)
        else:
            source_eff = source[i]
        loss = (hac[i-1]**loss_exp) / (tau_seconds[i] + 1e-6)
        hac[i] = hac[i-1] + (source_eff - loss) * dt_seconds[i]
        hac[i] = max(0, hac[i])
    return hac

def compute_baseline_akasofu(df, tau_base, v_ref, power, tau_min, tau_max,
                             loss_exp=LOSS_EXPONENT, h_sat=H_SATURATION):
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    N = df['density'].values
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_theta = Bz / (Btot + 1e-10)
    sin_theta2 = np.sqrt((1 - cos_theta) / 2)
    sin4 = sin_theta2**4
    source = V * (Btot**2) * sin4 / 1e6
    source = np.clip(source, 0, MAX_SOURCE / 1e6)

    times = df['time_tag'].values
    Vsw = df['speed'].values
    dt_seconds = np.zeros(len(times))
    if len(times) > 1:
        time_diffs = np.diff(times)
        dt_seconds[1:] = time_diffs.astype('timedelta64[s]').astype(np.float64)
        median_dt = np.median(dt_seconds[1:]) if len(dt_seconds) > 2 else 60.0
        dt_seconds[0] = median_dt
    else:
        dt_seconds[:] = 60.0

    tau_hours = tau_base * (v_ref / np.maximum(Vsw, 1.0)) ** power
    tau_hours = np.clip(tau_hours, tau_min, tau_max)
    tau_seconds = tau_hours * 3600.0

    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        if h_sat > 0:
            source_eff = source[i] / (1 + hac[i-1] / h_sat)
        else:
            source_eff = source[i]
        loss = (hac[i-1]**loss_exp) / (tau_seconds[i] + 1e-6)
        hac[i] = hac[i-1] + (source_eff - loss) * dt_seconds[i]
        hac[i] = max(0, hac[i])
    return hac

# ============================
# CÁLCULO DO HAC PRINCIPAL (COM PERDA NÃO LINEAR E SATURAÇÃO)
# ============================
def calculate_hac_physical(df, calib_factor=1.0, tau_base=TAU_BASE_HOURS, v_ref=V_REF,
                           power=TAU_POWER, tau_min=TAU_MIN_HOURS, tau_max=TAU_MAX_HOURS,
                           loss_exp=LOSS_EXPONENT, h_sat=H_SATURATION):
    times = df['time_tag'].values
    Vsw = df['speed'].values
    source = compute_source(df)

    dt_seconds = np.zeros(len(times))
    if len(times) > 1:
        time_diffs = np.diff(times)
        dt_seconds[1:] = time_diffs.astype('timedelta64[s]').astype(np.float64)
        median_dt = np.median(dt_seconds[1:]) if len(dt_seconds) > 2 else 60.0
        dt_seconds[0] = median_dt
    else:
        dt_seconds[:] = 60.0

    tau_hours = tau_base * (v_ref / np.maximum(Vsw, 1.0)) ** power
    tau_hours = np.clip(tau_hours, tau_min, tau_max)
    tau_seconds = tau_hours * 3600.0

    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        # Saturação física
        if h_sat > 0:
            source_eff = source[i] / (1 + hac[i-1] / h_sat)
        else:
            source_eff = source[i]
        # Perda não linear
        loss = (hac[i-1]**loss_exp) / (tau_seconds[i] + 1e-6)
        hac[i] = hac[i-1] + (source_eff - loss) * dt_seconds[i]
        hac[i] = max(0, hac[i])

    hac_scaled = hac * calib_factor

    # Derivada suavizada (para análise)
    if USE_SAVITZKY_GOLAY and len(hac_scaled) >= SG_WINDOW:
        hac_smooth = savgol_filter(hac_scaled, window_length=SG_WINDOW, polyorder=SG_ORDER)
    else:
        hac_smooth = hac_scaled
    dHAC_dt = np.gradient(hac_smooth, dt_seconds) * 3600.0

    df['HAC'] = hac_scaled
    df['HAC_raw'] = hac
    df['HAC_smooth'] = hac_smooth
    df['tau_hours'] = tau_hours
    df['dHAC_dt'] = dHAC_dt
    return df

# ============================
# MÉTRICAS DE VALIDAÇÃO (PREDITOR = HAC, ALVO = -Dst suavizado)
# ============================
def compute_metrics(df_hac, dst_series, threshold_dst=-50, threshold_hac=None,
                    smooth_dst_hours=DST_SMOOTH_HOURS):
    if dst_series is None or len(dst_series) == 0:
        return None

    df_hac = df_hac.copy()
    dst_series = dst_series.copy()

    df_hac['time_tag'] = pd.to_datetime(df_hac['time_tag'], errors='coerce')
    dst_series['time_tag'] = pd.to_datetime(dst_series['time_tag'], errors='coerce')

    df_hac = df_hac.dropna(subset=['time_tag', 'HAC'])
    dst_series = dst_series.dropna(subset=['time_tag', 'dst'])

    if len(df_hac) < 10 or len(dst_series) < 10:
        return None

    merged = pd.merge_asof(
        df_hac.sort_values('time_tag'),
        dst_series.sort_values('time_tag'),
        on='time_tag',
        direction='nearest'
    )
    merged = merged.dropna(subset=['dst', 'HAC'])

    merged = merged.set_index('time_tag')
    if len(merged) > 1:
        median_dt = merged.index.to_series().diff().median().total_seconds()
        window_size = max(3, int(smooth_dst_hours * 3600 / median_dt))
        if window_size % 2 == 0:
            window_size += 1
        merged['dst_smooth'] = merged['dst'].rolling(window=window_size, center=True, min_periods=1).mean()
    else:
        merged['dst_smooth'] = merged['dst']
    merged = merged.dropna(subset=['dst_smooth'])
    merged = merged.reset_index()

    merged = merged[merged['dst_smooth'] < 0]
    if len(merged) < 10:
        return None

    y_pred = merged['HAC'].values
    y_true = -merged['dst_smooth'].values

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        corr = float('nan')
    else:
        corr = pearsonr(y_true, y_pred)[0]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    pod = far = csi = None
    if threshold_hac is not None:
        true_storm = merged['dst_smooth'] < threshold_dst
        pred_storm = merged['HAC'] > threshold_hac
        tp = np.sum(true_storm & pred_storm)
        fp = np.sum(~true_storm & pred_storm)
        fn = np.sum(true_storm & ~pred_storm)
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        'correlation': corr,
        'rmse': rmse,
        'mae': mae,
        'pod': pod,
        'far': far,
        'csi': csi,
        'n_points': len(merged)
    }

def find_optimal_delay(df_hac, dst_series, max_delay=MAX_DELAY_HOURS, step=DELAY_STEP_HOURS,
                       smooth_dst_hours=DST_SMOOTH_HOURS):
    best_corr = -1
    best_delay = 0
    print(f"\n🔍 Buscando atraso ótimo (HAC → -Dst suavizado):")
    for delay in np.arange(0, max_delay + step, step):
        shifted_dst = apply_time_shift(dst_series, delay)
        metrics = compute_metrics(df_hac, shifted_dst, DST_THRESHOLD, threshold_hac=None,
                                  smooth_dst_hours=smooth_dst_hours)
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

def calibrate_hac_via_regression(df_hac_raw, dst_series, delay=0, min_hac=0.01, min_dst=MIN_DST_CALIB,
                                 smooth_dst_hours=DST_SMOOTH_HOURS):
    shifted_dst = apply_time_shift(dst_series, delay)
    df_hac_raw = df_hac_raw.copy()
    shifted_dst = shifted_dst.copy()
    df_hac_raw['time_tag'] = pd.to_datetime(df_hac_raw['time_tag'])
    shifted_dst['time_tag'] = pd.to_datetime(shifted_dst['time_tag'])

    merged = pd.merge_asof(df_hac_raw.sort_values('time_tag'),
                           shifted_dst.sort_values('time_tag'),
                           on='time_tag', direction='nearest')
    merged = merged.dropna(subset=['HAC_raw', 'dst'])

    merged = merged.set_index('time_tag')
    if len(merged) > 1:
        median_dt = merged.index.to_series().diff().median().total_seconds()
        window_size = max(3, int(smooth_dst_hours * 3600 / median_dt))
        if window_size % 2 == 0:
            window_size += 1
        merged['dst_smooth'] = merged['dst'].rolling(window=window_size, center=True, min_periods=1).mean()
    else:
        merged['dst_smooth'] = merged['dst']
    merged = merged.dropna(subset=['dst_smooth'])
    merged = merged.reset_index()

    mask = (merged['dst_smooth'] < 0) & (merged['dst_smooth'].abs() > min_dst) & (merged['HAC_raw'] > min_hac)
    if mask.sum() < 5:
        print("⚠️ Pontos insuficientes para calibração.")
        return None

    x = merged.loc[mask, 'HAC_raw'].values
    y = -merged.loc[mask, 'dst_smooth'].values

    slope, intercept, r_value, p_value, stderr = linregress(x, y)
    print(f"📊 Calibração via regressão (delay={delay}h, min|Dst|={min_dst} nT):")
    print(f"   |Dst_smooth| = {slope:.3f} * HAC_raw + {intercept:.3f} (R²={r_value**2:.3f})")
    factor = slope
    print(f"   Fator calibração: {factor:.3f}")
    return factor

def predictive_analysis(df, dH_threshold=5.0, lookahead_hours=3, dst_series=None, threshold_dst=-50):
    if dst_series is None:
        print("   Sem dados de Dst, não é possível avaliar predição.")
        return
    df['alert'] = df['dHAC_dt'] > dH_threshold
    dst_shifted = apply_time_shift(dst_series, -lookahead_hours)
    merged = pd.merge_asof(df[['time_tag', 'alert']].sort_values('time_tag'),
                           dst_shifted[['time_tag', 'dst']].sort_values('time_tag'),
                           on='time_tag', direction='nearest')
    merged = merged.dropna()
    true_storm = merged['dst'] < threshold_dst
    pred_alert = merged['alert']
    tp = np.sum(true_storm & pred_alert)
    fp = np.sum(~true_storm & pred_alert)
    fn = np.sum(true_storm & ~pred_alert)
    pod = tp / (tp + fn) if (tp+fn) > 0 else 0
    far = fp / (tp + fp) if (tp+fp) > 0 else 0
    print(f"\n📈 Análise preditiva (dHAC/dt > {dH_threshold} antecede {lookahead_hours}h):")
    print(f"   POD = {pod:.3f}, FAR = {far:.3f}")

def create_comparison_figure(df, baseline_simple, baseline_akasofu, dst_df=None, filename="hac_comparison.png",
                             delay=0, smooth_dst_hours=DST_SMOOTH_HOURS):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(df['time_tag'], df['HAC'], label='HAC (modelo com perda)', color='#d62728', linewidth=2)
    axes[0].plot(df['time_tag'], baseline_simple, label='Baseline: (-Bz*V) (com perda)', color='#1f77b4', linewidth=1.5, alpha=0.7)
    axes[0].plot(df['time_tag'], baseline_akasofu, label='Baseline: Akasofu ε (com perda)', color='#ff7f0e', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Índice')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Comparação dos Modelos de Acoplamento (perda não linear, saturação)')

    if dst_df is not None:
        if delay > 0:
            dst_shifted = apply_time_shift(dst_df, delay)
        else:
            dst_shifted = dst_df.copy()
        dst_shifted = dst_shifted.set_index('time_tag')
        if len(dst_shifted) > 1:
            median_dt = dst_shifted.index.to_series().diff().median().total_seconds()
            window_size = max(3, int(smooth_dst_hours * 3600 / median_dt))
            if window_size % 2 == 0:
                window_size += 1
            dst_shifted['dst_smooth'] = dst_shifted['dst'].rolling(window=window_size, center=True, min_periods=1).mean()
        else:
            dst_shifted['dst_smooth'] = dst_shifted['dst']
        dst_shifted = dst_shifted.reset_index()
        axes[1].plot(dst_shifted['time_tag'], dst_shifted['dst_smooth'], color='black', linewidth=1.5, label='Dst suavizado')
        axes[1].axhline(y=DST_THRESHOLD, color='red', linestyle='--', label=f'Limiar tempestade ({DST_THRESHOLD} nT)')
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
    print(f"✅ Figura comparativa salva: {filename}")

def main():
    print("\n" + "="*70)
    print("🛰️  HAC - MODELO AVANÇADO: perda não linear + saturação + Dst baseado")
    print("="*70)

    # Carregar dados OMNI
    print("\n📥 Carregando dados OMNI...")
    mag_df = load_omni_data(MAG_FILE)
    plasma_df = load_omni_data(PLASMA_FILE)
    if mag_df is None or plasma_df is None:
        print("❌ Falha crítica no carregamento de dados.")
        return

    # Preparar dados
    print("\n🔧 Preparando e fundindo dados...")
    df = prepare_merged_data(mag_df, plasma_df)
    if len(df) < 10:
        print("❌ Dados insuficientes após preparação.")
        return
    print(f"   Dados válidos: {len(df)} pontos")
    print(f"   Período: {df['time_tag'].min()} a {df['time_tag'].max()}")

    bz_neg_count = (df['bz_gsm'] < 0).sum()
    print(f"\n🔎 Bz sul: {bz_neg_count} pontos ({bz_neg_count/len(df)*100:.1f}%)")
    if bz_neg_count == 0:
        print("⚠️  ALERTA: Nenhum ponto com Bz negativo detectado! O acoplamento será zero.")

    # Calcular HAC_raw (sem calibração)
    df = calculate_hac_physical(df, calib_factor=1.0)

    # Calcular baselines
    baseline_simple = compute_baseline_simple(df, TAU_BASE_HOURS, V_REF, TAU_POWER, TAU_MIN_HOURS, TAU_MAX_HOURS)
    baseline_akasofu = compute_baseline_akasofu(df, TAU_BASE_HOURS, V_REF, TAU_POWER, TAU_MIN_HOURS, TAU_MAX_HOURS)

    # Carregar Dst
    dst_df = load_dst_data(DST_FILE) if DST_FILE else None

    # Filtrar Dst para o período dos dados OMNI
    if dst_df is not None:
        start = df['time_tag'].min()
        end = df['time_tag'].max()
        original_len = len(dst_df)
        dst_df = dst_df[(dst_df['time_tag'] >= start) & (dst_df['time_tag'] <= end)]
        print(f"\n📅 Filtragem Dst: {original_len} → {len(dst_df)} pontos no período comum.")
        if len(dst_df) == 0:
            print("⚠️  Nenhum dado Dst no período dos dados OMNI. Validação impossível.")
            dst_df = None

    # Inicializar variáveis
    best_delay = 0
    best_corr = float('nan')
    calib_factor = 1.0

    if dst_df is not None:
        df_hac = df[['time_tag', 'HAC']].copy()
        df_hac_raw = df[['time_tag', 'HAC_raw']].copy()

        print("\n📋 Amostra HAC (raw):")
        print(df_hac_raw.head())
        print("\n📋 Amostra Dst:")
        print(dst_df.head())
        print("\n📊 Estatísticas HAC_raw:")
        print(df['HAC_raw'].describe())
        print("\n📊 Estatísticas Dst (apenas negativos):")
        print(dst_df[dst_df['dst'] < 0]['dst'].describe())

        best_delay, best_corr = find_optimal_delay(df_hac, dst_df, max_delay=MAX_DELAY_HOURS,
                                                   smooth_dst_hours=DST_SMOOTH_HOURS)

        if CALIBRATE_WITH_DELAY:
            calib_factor = calibrate_hac_via_regression(df_hac_raw, dst_df, delay=best_delay,
                                                        min_dst=MIN_DST_CALIB,
                                                        smooth_dst_hours=DST_SMOOTH_HOURS)
            if calib_factor is None or calib_factor <= 0:
                print("⚠️ Calibração falhou ou fator não positivo. Usando fator=1.0.")
                calib_factor = 1.0
            else:
                df = calculate_hac_physical(df, calib_factor=calib_factor)
                baseline_simple = baseline_simple * calib_factor
                baseline_akasofu = baseline_akasofu * calib_factor
                df_hac = df[['time_tag', 'HAC']].copy()
        else:
            df = calculate_hac_physical(df, calib_factor=calib_factor)
            baseline_simple = baseline_simple * calib_factor
            baseline_akasofu = baseline_akasofu * calib_factor
            df_hac = df[['time_tag', 'HAC']].copy()

        # Calcular limiar HAC baseado em Dst real (somente nos momentos de tempestade)
        # Primeiro, alinhar HAC com Dst (com delay ótimo) para extrair os valores nos tempos de tempestade
        shifted_dst = apply_time_shift(dst_df, best_delay)
        # Suavizar Dst (como nas métricas) para obter os índices de tempestade
        tmp = pd.merge_asof(df_hac.sort_values('time_tag'),
                            shifted_dst.sort_values('time_tag'),
                            on='time_tag', direction='nearest')
        tmp = tmp.dropna(subset=['dst'])
        # Suavização rápida para definir Dst suavizado
        tmp = tmp.set_index('time_tag')
        if len(tmp) > 1:
            median_dt = tmp.index.to_series().diff().median().total_seconds()
            window_size = max(3, int(DST_SMOOTH_HOURS * 3600 / median_dt))
            if window_size % 2 == 0:
                window_size += 1
            tmp['dst_smooth'] = tmp['dst'].rolling(window=window_size, center=True, min_periods=1).mean()
        else:
            tmp['dst_smooth'] = tmp['dst']
        tmp = tmp.reset_index()
        storm_hac_vals = tmp[tmp['dst_smooth'] < DST_THRESHOLD]['HAC']
        if len(storm_hac_vals) > 0:
            hac_threshold = np.median(storm_hac_vals)
            print(f"\n📊 Limiar HAC baseado em tempestade real (Dst < {DST_THRESHOLD} nT): {hac_threshold:.2f}")
        else:
            # fallback: percentil 90
            hac_threshold = np.percentile(df['HAC'], 90)
            print(f"\n⚠️ Nenhum ponto com Dst < {DST_THRESHOLD} nT. Usando percentil 90: {hac_threshold:.2f}")

        # Validação final
        print("\n📊 VALIDAÇÃO DO MODELO (HAC → -Dst suavizado)")
        print("="*50)
        shifted_dst = apply_time_shift(dst_df, best_delay)
        metrics_hac = compute_metrics(df_hac, shifted_dst, DST_THRESHOLD, threshold_hac=hac_threshold,
                                      smooth_dst_hours=DST_SMOOTH_HOURS)
        if metrics_hac:
            print(f"✅ Modelo principal (calibrado) vs Dst suavizado (delay={best_delay}h):")
            print(f"   Correlação: {metrics_hac['correlation']:.3f}")
            print(f"   RMSE: {metrics_hac['rmse']:.2f}")
            print(f"   MAE: {metrics_hac['mae']:.2f}")
            if metrics_hac['pod'] is not None:
                print(f"   POD: {metrics_hac['pod']:.3f} | FAR: {metrics_hac['far']:.3f} | CSI: {metrics_hac['csi']:.3f}")

        df_simple = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': baseline_simple})
        metrics_simple = compute_metrics(df_simple, shifted_dst, DST_THRESHOLD, threshold_hac=hac_threshold,
                                         smooth_dst_hours=DST_SMOOTH_HOURS)
        if metrics_simple:
            print(f"\n📉 Baseline (-Bz*V) (calibrado) vs Dst suavizado (delay={best_delay}h):")
            print(f"   Correlação: {metrics_simple['correlation']:.3f}")
            print(f"   RMSE: {metrics_simple['rmse']:.2f}")
            if metrics_simple['pod'] is not None:
                print(f"   POD: {metrics_simple['pod']:.3f} | FAR: {metrics_simple['far']:.3f} | CSI: {metrics_simple['csi']:.3f}")

        df_akasofu = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': baseline_akasofu})
        metrics_akasofu = compute_metrics(df_akasofu, shifted_dst, DST_THRESHOLD, threshold_hac=hac_threshold,
                                          smooth_dst_hours=DST_SMOOTH_HOURS)
        if metrics_akasofu:
            print(f"\n📉 Baseline Akasofu ε (calibrado) vs Dst suavizado (delay={best_delay}h):")
            print(f"   Correlação: {metrics_akasofu['correlation']:.3f}")
            print(f"   RMSE: {metrics_akasofu['rmse']:.2f}")
            if metrics_akasofu['pod'] is not None:
                print(f"   POD: {metrics_akasofu['pod']:.3f} | FAR: {metrics_akasofu['far']:.3f} | CSI: {metrics_akasofu['csi']:.3f}")

        predictive_analysis(df, dH_threshold=5.0, lookahead_hours=3, dst_series=dst_df, threshold_dst=DST_THRESHOLD)
    else:
        print("⚠️ Dados de Dst não disponíveis no período comum. Validação não realizada.")

    # Gráfico
    create_comparison_figure(df, baseline_simple, baseline_akasofu, dst_df, "hac_comparison.png",
                             delay=best_delay if dst_df is not None else 0,
                             smooth_dst_hours=DST_SMOOTH_HOURS)

    csv_file = "hac_resultados_finais.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n💾 Dados processados salvos: {csv_file}")

    # Relatório final
    print("\n" + "="*70)
    print("🎯 RELATÓRIO FINAL")
    print("="*70)
    print(f"⚙️  Parâmetros do modelo (configuração final):")
    print(f"   • Perda: dH/dt = S_eff - H^{LOSS_EXPONENT}/τ")
    print(f"   • Saturação: S_eff = S / (1 + H/{H_SATURATION})" if H_SATURATION > 0 else "   • Saturação: desativada")
    print(f"   • τ = {TAU_BASE_HOURS} * ({V_REF}/V)^{TAU_POWER:.1f} h, limitado entre {TAU_MIN_HOURS} e {TAU_MAX_HOURS} h")
    print(f"   • Termo fonte: (-Bz*V) * sqrt(N) * sin²(θ/2) (densidade e ângulo de clock ATIVOS)")
    print(f"   • Escala do termo fonte: /{SOURCE_SCALE_FACTOR:.1f} (neutra, calibração ajusta)")
    if dst_df is not None:
        print(f"   • Atraso ótimo: {best_delay}h (correlação máxima: {best_corr:.3f})")
        print(f"   • Calibração: fator multiplicativo {calib_factor:.3f}")
        print(f"   • Suavização do Dst: janela de {DST_SMOOTH_HOURS} horas")
        print(f"   • Alvo de validação: -Dst suavizado (intensidade da tempestade)")
        print(f"   • Limiar HAC: mediana nos momentos de Dst < {DST_THRESHOLD} nT = {hac_threshold:.2f}")
    print(f"   • Derivada: {'suavizada (SG)' if USE_SAVITZKY_GOLAY else 'bruta'}")
    print("\n✅ Modelo com todas as melhorias: perda não linear, saturação física e limiar baseado em Dst.")
    print("="*70)

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    main()
