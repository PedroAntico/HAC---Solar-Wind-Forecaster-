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
CALIBRATION_FILE = "data/eventos_referencia.csv"  # Opcional: colunas 'data', 'hac_target'

# ============================
# PARÂMETROS DO MODELO FÍSICO (COM PERDA)
# ============================
# Constante de tempo (memória) – agora usada como τ no termo de perda
TAU_BASE_HOURS = 2.0        # Constante de tempo base (horas)
V_REF = 400.0               # Velocidade de referência (km/s)
TAU_POWER = 0.3             # Expoente para variação suave
TAU_MIN_HOURS = 1.0
TAU_MAX_HOURS = 5.0

# Termo fonte – simplificado (sem ângulo de clock, sem densidade)
USE_AKASOFU = False
USE_DENSITY = False
USE_BY_CLOCK = False

# Fator de escala do termo fonte (ajuste fino da intensidade)
SOURCE_SCALE_FACTOR = 100.0   # Antes era /1000; agora /100 (maior amplitude)

# Limite para evitar explosões numéricas no termo fonte
MAX_SOURCE = 5000.0

# Suavização da derivada
USE_SAVITZKY_GOLAY = True
SG_WINDOW = 7
SG_ORDER = 2

# Limiares para validação de tempestade (Dst e HAC)
DST_THRESHOLD = -50          # nT – tempestade se Dst < -50
HAC_THRESHOLD = 50           # Valor correspondente de HAC (será ajustado dinamicamente)

# Parâmetros de validação
MAX_DELAY_HOURS = 6          # Atraso máximo a testar (horas)
DELAY_STEP_HOURS = 1         # Passo de atraso
CALIBRATE_WITH_DELAY = True  # Se True, usa o delay ótimo para calibração

# Tipo de preditor usado na validação ('hac', 'deriv', 'product')
PREDICTOR_TYPE = 'product'   # 'product' = HAC * dHAC/dt; 'deriv' = dHAC/dt; 'hac' = HAC

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
# TERMOS FONTE (SIMPLIFICADO)
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

    # Limite para evitar explosões numéricas
    source = np.clip(source, 0, MAX_SOURCE)

    # Aplica fator de escala
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
        epsilon = np.clip(epsilon, 0, MAX_SOURCE * 1e6)  # Ajuste de escala
        return epsilon / 1e6
    else:
        return compute_source_simplified(Bx, By, Bz, V, N, use_density, use_by_clock)

# ============================
# MODELOS BASELINE (com perda)
# ============================
def compute_baseline_simple(df, tau_base, v_ref, power, tau_min, tau_max):
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

    # Modelo com perda: dH/dt = source - H/τ
    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        dt_hours = dt_seconds[i] / 3600.0
        loss = hac[i-1] / (tau_seconds[i] + 1e-6)  # perda por segundo
        hac[i] = hac[i-1] + (source[i] - loss) * dt_seconds[i]  # incremento em segundos
        hac[i] = max(0, hac[i])  # evitar negativo
    return hac

def compute_baseline_akasofu(df, tau_base, v_ref, power, tau_min, tau_max):
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
        dt_hours = dt_seconds[i] / 3600.0
        loss = hac[i-1] / (tau_seconds[i] + 1e-6)
        hac[i] = hac[i-1] + (source[i] - loss) * dt_seconds[i]
        hac[i] = max(0, hac[i])
    return hac

# ============================
# CÁLCULO DO HAC PRINCIPAL (COM PERDA)
# ============================
def calculate_hac_physical(df, calib_factor=1.0, tau_base=TAU_BASE_HOURS, v_ref=V_REF,
                           power=TAU_POWER, tau_min=TAU_MIN_HOURS, tau_max=TAU_MAX_HOURS):
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

    # Modelo com perda: dH/dt = source - H/τ
    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        # Perda por segundo: H / τ
        loss = hac[i-1] / (tau_seconds[i] + 1e-6)
        # Integração explícita (Euler)
        hac[i] = hac[i-1] + (source[i] - loss) * dt_seconds[i]
        # Impedir valores negativos
        hac[i] = max(0, hac[i])

    hac_scaled = hac * calib_factor

    # Derivada suavizada (necessária para o preditor composto)
    if USE_SAVITZKY_GOLAY and len(hac_scaled) >= SG_WINDOW:
        hac_smooth = savgol_filter(hac_scaled, window_length=SG_WINDOW, polyorder=SG_ORDER)
    else:
        hac_smooth = hac_scaled
    dHAC_dt = np.gradient(hac_smooth, dt_seconds) * 3600.0  # por hora

    df['HAC'] = hac_scaled
    df['HAC_raw'] = hac
    df['HAC_smooth'] = hac_smooth
    df['tau_hours'] = tau_hours
    df['dHAC_dt'] = dHAC_dt
    return df

# ============================
# MÉTRICAS DE VALIDAÇÃO (COM PREDITOR FLEXÍVEL)
# ============================
def compute_metrics(df_hac, dst_series, threshold_dst=-50, threshold_hac=50, predictor='product'):
    """
    df_hac: DataFrame com colunas 'time_tag', 'HAC', 'dHAC_dt'
    dst_series: DataFrame com 'time_tag', 'dst'
    predictor: 'hac', 'deriv', 'product' (HAC * dHAC/dt)
    """
    if dst_series is None or len(dst_series) == 0:
        return None

    df_hac = df_hac.copy()
    dst_series = dst_series.copy()

    df_hac['time_tag'] = pd.to_datetime(df_hac['time_tag'], errors='coerce')
    dst_series['time_tag'] = pd.to_datetime(dst_series['time_tag'], errors='coerce')

    df_hac = df_hac.dropna(subset=['time_tag', 'HAC', 'dHAC_dt'])
    dst_series = dst_series.dropna(subset=['time_tag', 'dst'])

    if len(df_hac) < 10 or len(dst_series) < 10:
        return None

    merged = pd.merge_asof(
        df_hac.sort_values('time_tag'),
        dst_series.sort_values('time_tag'),
        on='time_tag',
        direction='nearest'
    )
    merged = merged.dropna(subset=['dst', 'HAC', 'dHAC_dt'])

    # Apenas pontos com Dst < 0 (tempestade ou perturbação)
    merged = merged[merged['dst'] < 0]

    if len(merged) < 10:
        return None

    # Construir preditor
    if predictor == 'hac':
        y_pred = merged['HAC'].values
    elif predictor == 'deriv':
        y_pred = merged['dHAC_dt'].values
        y_pred = np.maximum(0, y_pred)  # apenas taxa positiva
    elif predictor == 'product':
        y_pred = merged['HAC'].values * merged['dHAC_dt'].values
        y_pred = np.maximum(0, y_pred)  # não negativo
    else:
        raise ValueError(f"Predictor desconhecido: {predictor}")

    # Alvo: magnitude positiva do Dst (para correlação positiva)
    y_true = -merged['dst'].values  # positivo

    # Correlação
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        corr = float('nan')
    else:
        corr = pearsonr(y_true, y_pred)[0]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Métricas de detecção (ainda baseadas em HAC puro para classificação)
    true_storm = merged['dst'] < threshold_dst
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

def find_optimal_delay(df_hac, dst_series, max_delay=MAX_DELAY_HOURS, step=DELAY_STEP_HOURS, predictor='product'):
    best_corr = -1
    best_delay = 0
    print(f"\n🔍 Buscando atraso ótimo (preditor={predictor} → Dst):")
    for delay in np.arange(0, max_delay + step, step):
        shifted_dst = apply_time_shift(dst_series, delay)
        metrics = compute_metrics(df_hac, shifted_dst, DST_THRESHOLD, HAC_THRESHOLD, predictor=predictor)
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

def calibrate_hac_via_regression(df_hac, dst_series, delay=0, min_hac=0.01, min_dst=20, predictor='hac'):
    """
    Calibra HAC (ou derivada) para alinhar com -Dst (positivo) nos pontos onde Dst < 0 e |Dst| > min_dst.
    """
    shifted_dst = apply_time_shift(dst_series, delay)
    df_hac = df_hac.copy()
    shifted_dst = shifted_dst.copy()
    df_hac['time_tag'] = pd.to_datetime(df_hac['time_tag'])
    shifted_dst['time_tag'] = pd.to_datetime(shifted_dst['time_tag'])

    merged = pd.merge_asof(df_hac.sort_values('time_tag'),
                           shifted_dst.sort_values('time_tag'),
                           on='time_tag', direction='nearest')
    merged = merged.dropna(subset=['HAC_raw', 'dst', 'dHAC_dt'])

    # Filtra apenas Dst < 0 e magnitude > min_dst
    mask = (merged['dst'] < 0) & (merged['dst'].abs() > min_dst)

    if predictor == 'hac':
        mask &= (merged['HAC_raw'] > min_hac)
        if mask.sum() < 5:
            print("⚠️ Pontos insuficientes para calibração do HAC.")
            return None
        x = merged.loc[mask, 'HAC_raw'].values
        y = -merged.loc[mask, 'dst'].values
    elif predictor == 'deriv':
        mask &= (merged['dHAC_dt'] > min_hac)
        if mask.sum() < 5:
            print("⚠️ Pontos insuficientes para calibração da derivada.")
            return None
        x = merged.loc[mask, 'dHAC_dt'].values
        y = -merged.loc[mask, 'dst'].values
    elif predictor == 'product':
        # Calibrar usando o produto HAC * dHAC/dt
        prod = merged['HAC_raw'] * merged['dHAC_dt']
        mask &= (prod > min_hac)
        if mask.sum() < 5:
            print("⚠️ Pontos insuficientes para calibração do produto.")
            return None
        x = prod.loc[mask].values
        y = -merged.loc[mask, 'dst'].values
    else:
        return None

    slope, intercept, r_value, p_value, stderr = linregress(x, y)
    print(f"📊 Calibração via regressão (delay={delay}h, predictor={predictor}):")
    print(f"   |Dst| = {slope:.3f} * X + {intercept:.3f} (R²={r_value**2:.3f})")
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

def create_comparison_figure(df, baseline_simple, baseline_akasofu, dst_df=None, filename="hac_comparison.png", delay=0, predictor='product'):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Painel 1: HAC e baselines
    axes[0].plot(df['time_tag'], df['HAC'], label='HAC (modelo com perda)', color='#d62728', linewidth=2)
    axes[0].plot(df['time_tag'], baseline_simple, label='Baseline: (-Bz*V) (com perda)', color='#1f77b4', linewidth=1.5, alpha=0.7)
    axes[0].plot(df['time_tag'], baseline_akasofu, label='Baseline: Akasofu ε (com perda)', color='#ff7f0e', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Índice')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Comparação dos Modelos de Acoplamento (com perda)')

    # Painel 2: Dst (negativo) com delay
    if dst_df is not None:
        if delay > 0:
            dst_shifted = apply_time_shift(dst_df, delay)
            axes[1].plot(dst_shifted['time_tag'], dst_shifted['dst'], label=f'Dst (atrasado {delay}h)', color='black', linewidth=1.2)
        else:
            axes[1].plot(dst_df['time_tag'], dst_df['dst'], label='Dst (nT)', color='black', linewidth=1.2)
        axes[1].axhline(y=DST_THRESHOLD, color='red', linestyle='--', label=f'Limiar tempestade ({DST_THRESHOLD} nT)')
        axes[1].set_ylabel('Dst [nT]')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Dst (Real)')
    else:
        axes[1].set_visible(False)

    # Painel 3: dHAC/dt e produto (opcional)
    axes[2].plot(df['time_tag'], df['dHAC_dt'], color='#9b59b6', linewidth=1.5, label='dHAC/dt')
    if predictor == 'product':
        product = df['HAC'] * df['dHAC_dt']
        axes[2].plot(df['time_tag'], product, color='#2ecc71', linewidth=1.0, linestyle='--', label='HAC × dHAC/dt')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Taxa / Produto')
    axes[2].set_xlabel('Tempo (UTC)')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figura comparativa salva: {filename}")

def main():
    print("\n" + "="*70)
    print("🛰️  HAC - MODELO FÍSICO COM PERDA E VALIDAÇÃO CONSISTENTE")
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

    # Verificação de Bz sul
    bz_neg_count = (df['bz_gsm'] < 0).sum()
    print(f"\n🔎 Bz sul: {bz_neg_count} pontos ({bz_neg_count/len(df)*100:.1f}%)")
    if bz_neg_count == 0:
        print("⚠️  ALERTA: Nenhum ponto com Bz negativo detectado! O acoplamento será zero.")
        print("   Verifique os dados de IMF ou considere usar By/Bz combinados.")

    # Calcular HAC_raw (sem calibração)
    df = calculate_hac_physical(df, calib_factor=1.0)

    # Calcular baselines (com perda)
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

    # Calibração e validação
    if dst_df is not None:
        # DataFrame para validação (contém HAC e dHAC/dt)
        df_hac = df[['time_tag', 'HAC', 'dHAC_dt']].copy()
        # Também precisamos de HAC_raw para calibração (se não usar derivada)
        df_hac_raw = df[['time_tag', 'HAC_raw', 'dHAC_dt']].copy()

        print("\n📋 Amostra HAC (raw) e derivada:")
        print(df_hac_raw.head())
        print("\n📋 Amostra Dst:")
        print(dst_df.head())
        print("\n📊 Estatísticas HAC_raw:")
        print(df['HAC_raw'].describe())
        print("\n📊 Estatísticas Dst (apenas negativos):")
        print(dst_df[dst_df['dst'] < 0]['dst'].describe())

        # Encontrar delay ótimo usando o preditor selecionado
        best_delay, best_corr = find_optimal_delay(df_hac, dst_df, predictor=PREDICTOR_TYPE)

        # Calibrar usando o mesmo preditor (opcional)
        if CALIBRATE_WITH_DELAY:
            # Para calibração, usamos o mesmo tipo de preditor, mas com a versão raw (não calibrada)
            # Nota: a calibração do produto requer HAC_raw e dHAC_dt juntos
            calib_factor = calibrate_hac_via_regression(df_hac_raw, dst_df, delay=best_delay, predictor=PREDICTOR_TYPE)
            if calib_factor is None:
                calib_factor = 1.0
            else:
                # Recalcular HAC com o novo fator
                df = calculate_hac_physical(df, calib_factor=calib_factor)
                # Recalcular baselines com mesmo fator (para comparação justa)
                baseline_simple = baseline_simple * calib_factor
                baseline_akasofu = baseline_akasofu * calib_factor
                # Atualizar df_hac
                df_hac = df[['time_tag', 'HAC', 'dHAC_dt']].copy()
        else:
            df = calculate_hac_physical(df, calib_factor=calib_factor)
            baseline_simple = baseline_simple * calib_factor
            baseline_akasofu = baseline_akasofu * calib_factor
            df_hac = df[['time_tag', 'HAC', 'dHAC_dt']].copy()

    # Validação final
    print("\n📊 VALIDAÇÃO DO MODELO (apenas Dst < 0, preditor = {})".format(PREDICTOR_TYPE))
    print("="*50)
    if dst_df is not None:
        shifted_dst = apply_time_shift(dst_df, best_delay)
        metrics_hac = compute_metrics(df_hac, shifted_dst, DST_THRESHOLD, HAC_THRESHOLD, predictor=PREDICTOR_TYPE)
        if metrics_hac:
            print(f"✅ Modelo principal (calibrado) vs Dst (delay={best_delay}h):")
            print(f"   Correlação: {metrics_hac['correlation']:.3f}")
            print(f"   RMSE: {metrics_hac['rmse']:.2f}")
            print(f"   MAE: {metrics_hac['mae']:.2f}")
            print(f"   POD: {metrics_hac['pod']:.3f} | FAR: {metrics_hac['far']:.3f} | CSI: {metrics_hac['csi']:.3f}")

        # Baselines: precisamos criar df_hac para eles (com derivada)
        # Para baseline simples
        df_simple = pd.DataFrame({
            'time_tag': df['time_tag'],
            'HAC': baseline_simple,
            'dHAC_dt': np.gradient(baseline_simple, df['time_tag'].diff().dt.total_seconds().values) * 3600.0
        })
        metrics_simple = compute_metrics(df_simple, shifted_dst, DST_THRESHOLD, HAC_THRESHOLD, predictor=PREDICTOR_TYPE)
        if metrics_simple:
            print(f"\n📉 Baseline (-Bz*V) (calibrado) vs Dst (delay={best_delay}h):")
            print(f"   Correlação: {metrics_simple['correlation']:.3f}")
            print(f"   RMSE: {metrics_simple['rmse']:.2f}")
            print(f"   POD: {metrics_simple['pod']:.3f} | FAR: {metrics_simple['far']:.3f} | CSI: {metrics_simple['csi']:.3f}")

        df_akasofu = pd.DataFrame({
            'time_tag': df['time_tag'],
            'HAC': baseline_akasofu,
            'dHAC_dt': np.gradient(baseline_akasofu, df['time_tag'].diff().dt.total_seconds().values) * 3600.0
        })
        metrics_akasofu = compute_metrics(df_akasofu, shifted_dst, DST_THRESHOLD, HAC_THRESHOLD, predictor=PREDICTOR_TYPE)
        if metrics_akasofu:
            print(f"\n📉 Baseline Akasofu ε (calibrado) vs Dst (delay={best_delay}h):")
            print(f"   Correlação: {metrics_akasofu['correlation']:.3f}")
            print(f"   RMSE: {metrics_akasofu['rmse']:.2f}")
            print(f"   POD: {metrics_akasofu['pod']:.3f} | FAR: {metrics_akasofu['far']:.3f} | CSI: {metrics_akasofu['csi']:.3f}")

        # Análise preditiva com dHAC/dt
        predictive_analysis(df, dH_threshold=5.0, lookahead_hours=3, dst_series=dst_df, threshold_dst=DST_THRESHOLD)
    else:
        print("⚠️ Dados de Dst não disponíveis no período comum. Validação não realizada.")

    # Gráfico
    create_comparison_figure(df, baseline_simple, baseline_akasofu, dst_df, "hac_comparison.png",
                             delay=best_delay if dst_df is not None else 0, predictor=PREDICTOR_TYPE)

    # Salvar dados
    csv_file = "hac_resultados_finais.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n💾 Dados processados salvos: {csv_file}")

    # Relatório final
    print("\n" + "="*70)
    print("🎯 RELATÓRIO FINAL")
    print("="*70)
    print(f"⚙️  Parâmetros do modelo:")
    print(f"   • Modelo de acúmulo: dH/dt = S - H/τ (perda linear)")
    print(f"   • τ = {TAU_BASE_HOURS} * ({V_REF}/V)^{TAU_POWER:.1f} h, limitado entre {TAU_MIN_HOURS} e {TAU_MAX_HOURS} h")
    print(f"   • Termo fonte: {'Akasofu ε' if USE_AKASOFU else '(-Bz*V) (simplificado)'}")
    print(f"   • Escala do termo fonte: /{SOURCE_SCALE_FACTOR:.1f}, com clipping em {MAX_SOURCE}")
    if dst_df is not None:
        print(f"   • Atraso ótimo: {best_delay}h (correlação máxima: {best_corr:.3f})")
        print(f"   • Calibração: fator multiplicativo {calib_factor:.3f} (aplicado a todos os modelos)")
    print(f"   • Preditor usado na validação: {PREDICTOR_TYPE}")
    print(f"   • Derivada: {'suavizada (SG)' if USE_SAVITZKY_GOLAY else 'bruta'}")
    print("\n✅ Modelo agora fisicamente consistente (perda, preditor composto, alvo positivo).")
    print("="*70)

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    main()
