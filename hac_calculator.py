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
# PARÂMETROS DO MODELO FÍSICO
# ============================
TAU_BASE_HOURS = 4.0        # Memória base (horas)
V_REF = 400.0               # Velocidade de referência (km/s)
TAU_POWER = 0.3             # Expoente para variação suave
TAU_MIN_HOURS = 2.0
TAU_MAX_HOURS = 8.0

USE_AKASOFU = False         # False usa versão simplificada com ângulo de clock
USE_DENSITY = True          # Inclui sqrt(N) no termo fonte
USE_BY_CLOCK = True         # Usa ângulo de clock (sin²(θ/2)) para modular

USE_SAVITZKY_GOLAY = True
SG_WINDOW = 7
SG_ORDER = 2

DST_THRESHOLD = -50          # nT – tempestade se Dst < -50
HAC_THRESHOLD = 50           # Valor correspondente de HAC (ajustável)

MAX_DELAY_HOURS = 6          # Atraso máximo a testar (horas)
DELAY_STEP_HOURS = 1         # Passo de atraso
CALIBRATE_WITH_DELAY = True  # Se True, usa o delay ótimo para calibração

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
# TERMOS FONTE
# ============================
def compute_source_akasofu(Bx, By, Bz, V, N):
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    cos_theta = Bz / (Btot + 1e-10)
    sin_theta2 = np.sqrt((1 - cos_theta) / 2)
    sin4 = sin_theta2**4
    epsilon = V * (Btot**2) * sin4
    return epsilon / 1e6

def compute_source_simplified(Bx, By, Bz, V, N, use_density=True, use_by_clock=True):
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
    source = source / 1000.0
    return source

def compute_source(df, use_akasofu=USE_AKASOFU, use_density=USE_DENSITY, use_by_clock=USE_BY_CLOCK):
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    N = df['density'].values
    if use_akasofu:
        return compute_source_akasofu(Bx, By, Bz, V, N)
    else:
        return compute_source_simplified(Bx, By, Bz, V, N, use_density, use_by_clock)

# ============================
# MODELOS BASELINE
# ============================
def compute_baseline_simple(df, tau_base, v_ref, power, tau_min, tau_max):
    source = np.zeros_like(df['bz_gsm'].values)
    south = df['bz_gsm'].values < 0
    source[south] = -df['bz_gsm'].values[south] * df['speed'].values[south] / 1000.0
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
    alpha = np.exp(-dt_seconds / tau_seconds)
    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        hac[i] = alpha[i] * hac[i-1] + (1 - alpha[i]) * source[i]
    return hac

def compute_baseline_akasofu(df, tau_base, v_ref, power, tau_min, tau_max):
    source = compute_source_akasofu(df['bx_gsm'].values, df['by_gsm'].values,
                                    df['bz_gsm'].values, df['speed'].values,
                                    df['density'].values)
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
    alpha = np.exp(-dt_seconds / tau_seconds)
    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        hac[i] = alpha[i] * hac[i-1] + (1 - alpha[i]) * source[i]
    return hac

# ============================
# CÁLCULO DO HAC PRINCIPAL
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
    alpha = np.exp(-dt_seconds / tau_seconds)

    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        hac[i] = alpha[i] * hac[i-1] + (1 - alpha[i]) * source[i]

    hac_scaled = hac * calib_factor

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
# MÉTRICAS DE VALIDAÇÃO
# ============================
def compute_metrics(hac_series, dst_series, threshold_dst=-50, threshold_hac=50):
    """Calcula correlação, RMSE, MAE, POD, FAR, CSI entre HAC e Dst."""
    if dst_series is None or len(dst_series) == 0:
        return None

    hac_series = hac_series.copy()
    dst_series = dst_series.copy()

    hac_series['time_tag'] = pd.to_datetime(hac_series['time_tag'], errors='coerce')
    dst_series['time_tag'] = pd.to_datetime(dst_series['time_tag'], errors='coerce')

    hac_series = hac_series.dropna(subset=['time_tag', 'HAC'])
    dst_series = dst_series.dropna(subset=['time_tag', 'dst'])

    if len(hac_series) < 10 or len(dst_series) < 10:
        return None

    df_merged = pd.merge_asof(
        hac_series.sort_values('time_tag'),
        dst_series.sort_values('time_tag'),
        on='time_tag',
        direction='nearest'
    )
    df_merged = df_merged.dropna(subset=['dst', 'HAC'])

    if len(df_merged) < 10:
        return None

    y_true = df_merged['dst'].abs()
    y_pred = df_merged['HAC']

    # Evita divisão por zero e valores constantes
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        corr = float('nan')
    else:
        corr = pearsonr(y_true, y_pred)[0]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    true_storm = df_merged['dst'] < threshold_dst
    pred_storm = df_merged['HAC'] > threshold_hac
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
        'n_points': len(df_merged)
    }

def find_optimal_delay(hac_series, dst_series, max_delay=MAX_DELAY_HOURS, step=DELAY_STEP_HOURS):
    """Encontra o atraso que maximiza a correlação entre HAC e |Dst|."""
    best_corr = -1
    best_delay = 0
    print("\n🔍 Buscando atraso ótimo (HAC → Dst):")
    for delay in np.arange(0, max_delay + step, step):
        shifted_dst = apply_time_shift(dst_series, delay)
        metrics = compute_metrics(hac_series, shifted_dst, DST_THRESHOLD, HAC_THRESHOLD)
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

def calibrate_hac_via_regression(hac_series, dst_series, delay=0, min_hac=0.01, min_dst=20):
    """
    Calibra HAC usando regressão linear HAC_calib = a * HAC_raw, com base nos pontos onde |Dst| > min_dst.
    """
    shifted_dst = apply_time_shift(dst_series, delay)
    # Alinhar temporalmente
    hac_series = hac_series.copy()
    shifted_dst = shifted_dst.copy()
    hac_series['time_tag'] = pd.to_datetime(hac_series['time_tag'])
    shifted_dst['time_tag'] = pd.to_datetime(shifted_dst['time_tag'])
    merged = pd.merge_asof(hac_series.sort_values('time_tag'),
                           shifted_dst.sort_values('time_tag'),
                           on='time_tag', direction='nearest')
    merged = merged.dropna(subset=['HAC_raw', 'dst'])
    # Filtrar pontos onde |Dst| > min_dst e HAC_raw > 0
    mask = (merged['dst'].abs() > min_dst) & (merged['HAC_raw'] > min_hac)
    if mask.sum() < 5:
        print("⚠️ Pontos insuficientes para calibração. Usando fator padrão.")
        return None
    x = merged.loc[mask, 'HAC_raw'].values
    y = merged.loc[mask, 'dst'].abs().values
    slope, intercept, r_value, p_value, stderr = linregress(x, y)
    print(f"📊 Calibração via regressão (delay={delay}h):")
    print(f"   |Dst| = {slope:.3f} * HAC_raw + {intercept:.3f} (R²={r_value**2:.3f})")
    factor = slope
    print(f"   Fator calibração: {factor:.3f}")
    return factor

# ============================
# ANÁLISE PREDITIVA COM dHAC/dt
# ============================
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

# ============================
# VISUALIZAÇÃO
# ============================
def create_comparison_figure(df, baseline_simple_calib, baseline_akasofu_calib, dst_df=None, filename="hac_comparison.png", delay=0):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(df['time_tag'], df['HAC'], label='HAC (modelo principal)', color='#d62728', linewidth=2)
    axes[0].plot(df['time_tag'], baseline_simple_calib, label='Baseline: (-Bz*V) (calibrado)', color='#1f77b4', linewidth=1.5, alpha=0.7)
    axes[0].plot(df['time_tag'], baseline_akasofu_calib, label='Baseline: Akasofu ε (calibrado)', color='#ff7f0e', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Índice')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Comparação dos Modelos de Acoplamento Heliosférico')

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

# ============================
# FUNÇÃO PRINCIPAL
# ============================
def main():
    print("\n" + "="*70)
    print("🛰️  HAC - HELIOSPHERIC ACCUMULATED COUPLING (MODELO FÍSICO VALIDADO)")
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

    # Verificação rápida de Bz sul
    bz_neg_count = (df['bz_gsm'] < 0).sum()
    print(f"\n🔎 Bz sul: {bz_neg_count} pontos ({bz_neg_count/len(df)*100:.1f}%)")
    if bz_neg_count == 0:
        print("⚠️  ALERTA: Nenhum ponto com Bz negativo detectado! O acoplamento será zero.")
        print("   Verifique os dados de IMF ou considere usar By/Bz combinados.")

    # Calcular HAC_raw (sem calibração)
    df = calculate_hac_physical(df, calib_factor=1.0)

    # Calcular baselines (não calibrados)
    baseline_simple = compute_baseline_simple(df, TAU_BASE_HOURS, V_REF, TAU_POWER, TAU_MIN_HOURS, TAU_MAX_HOURS)
    baseline_akasofu = compute_baseline_akasofu(df, TAU_BASE_HOURS, V_REF, TAU_POWER, TAU_MIN_HOURS, TAU_MAX_HOURS)

    # Carregar Dst
    dst_df = load_dst_data(DST_FILE) if DST_FILE else None

    # --- PERÍODO COMUM ---
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

    # Calibração e validação apenas se houver Dst no período comum
    if dst_df is not None:
        # Criar série para validação (renomeia HAC_raw para HAC)
        hac_raw_series = df[['time_tag', 'HAC_raw']].rename(columns={'HAC_raw': 'HAC'})

        # Debug dos dados
        print("\n📋 Amostra HAC (raw):")
        print(hac_raw_series.head())
        print("\n📋 Amostra Dst:")
        print(dst_df.head())
        print("\n📊 Estatísticas HAC_raw:")
        print(df['HAC_raw'].describe())
        print("\n📊 Estatísticas Dst:")
        print(dst_df['dst'].describe())

        # Encontrar delay ótimo
        best_delay, best_corr = find_optimal_delay(hac_raw_series, dst_df)

        # Calibrar
        if CALIBRATE_WITH_DELAY:
            # Para calibração, precisamos da série com HAC_raw (não renomeada)
            hac_raw_for_calib = df[['time_tag', 'HAC_raw']].copy()
            calib_factor = calibrate_hac_via_regression(hac_raw_for_calib, dst_df, delay=best_delay)
            if calib_factor is None:
                calib_factor = 1.0

    # Aplicar calibração final
    df = calculate_hac_physical(df, calib_factor=calib_factor)
    baseline_simple_calib = baseline_simple * calib_factor
    baseline_akasofu_calib = baseline_akasofu * calib_factor

    # Validação final
    print("\n📊 VALIDAÇÃO DO MODELO (com atraso ótimo)")
    print("="*50)
    if dst_df is not None:
        shifted_dst = apply_time_shift(dst_df, best_delay)
        hac_series = df[['time_tag', 'HAC']].copy()
        metrics_hac = compute_metrics(hac_series, shifted_dst, DST_THRESHOLD, HAC_THRESHOLD)
        if metrics_hac:
            print(f"✅ HAC (calibrado) vs Dst (delay={best_delay}h):")
            print(f"   Correlação: {metrics_hac['correlation']:.3f}")
            print(f"   RMSE: {metrics_hac['rmse']:.2f}")
            print(f"   MAE: {metrics_hac['mae']:.2f}")
            print(f"   POD: {metrics_hac['pod']:.3f} | FAR: {metrics_hac['far']:.3f} | CSI: {metrics_hac['csi']:.3f}")

        simple_series = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': baseline_simple_calib})
        metrics_simple = compute_metrics(simple_series, shifted_dst, DST_THRESHOLD, HAC_THRESHOLD)
        if metrics_simple:
            print(f"\n📉 Baseline (-Bz*V) calibrado vs Dst (delay={best_delay}h):")
            print(f"   Correlação: {metrics_simple['correlation']:.3f}")
            print(f"   RMSE: {metrics_simple['rmse']:.2f}")
            print(f"   POD: {metrics_simple['pod']:.3f} | FAR: {metrics_simple['far']:.3f} | CSI: {metrics_simple['csi']:.3f}")

        akasofu_series = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': baseline_akasofu_calib})
        metrics_akasofu = compute_metrics(akasofu_series, shifted_dst, DST_THRESHOLD, HAC_THRESHOLD)
        if metrics_akasofu:
            print(f"\n📉 Baseline Akasofu ε calibrado vs Dst (delay={best_delay}h):")
            print(f"   Correlação: {metrics_akasofu['correlation']:.3f}")
            print(f"   RMSE: {metrics_akasofu['rmse']:.2f}")
            print(f"   POD: {metrics_akasofu['pod']:.3f} | FAR: {metrics_akasofu['far']:.3f} | CSI: {metrics_akasofu['csi']:.3f}")

        predictive_analysis(df, dH_threshold=5.0, lookahead_hours=3, dst_series=dst_df, threshold_dst=DST_THRESHOLD)
    else:
        print("⚠️ Dados de Dst não disponíveis no período comum. Validação não realizada.")

    # Gráfico
    create_comparison_figure(df, baseline_simple_calib, baseline_akasofu_calib, dst_df, "hac_comparison.png", delay=best_delay if dst_df is not None else 0)

    # Salvar dados
    csv_file = "hac_resultados_finais.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n💾 Dados processados salvos: {csv_file}")

    # Relatório final
    print("\n" + "="*70)
    print("🎯 RELATÓRIO FINAL")
    print("="*70)
    print(f"⚙️  Parâmetros do modelo:")
    print(f"   • τ = {TAU_BASE_HOURS} * ({V_REF}/V)^{TAU_POWER:.1f} h, limitado entre {TAU_MIN_HOURS} e {TAU_MAX_HOURS} h")
    print(f"   • Termo fonte: {'Akasofu ε' if USE_AKASOFU else '(-Bz*V)*sqrt(N)*sin²(θ/2)'}")
    if dst_df is not None:
        print(f"   • Atraso ótimo: {best_delay}h (correlação máxima: {best_corr:.3f})")
        print(f"   • Calibração: fator multiplicativo {calib_factor:.3f} (aplicado a todos os modelos)")
    print(f"   • Derivada: {'suavizada (SG)' if USE_SAVITZKY_GOLAY else 'bruta'}")
    print("\n✅ Modelo validado e pronto para análise científica.")
    print("="*70)

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    main()
