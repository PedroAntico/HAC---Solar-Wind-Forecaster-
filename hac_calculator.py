import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from scipy.signal import savgol_filter
from scipy.stats import linregress, pearsonr
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
TAU_BASE_HOURS = 3.0
V_REF = 400.0
TAU_POWER = 0.5
TAU_MIN_HOURS = 1.5
TAU_MAX_HOURS = 5.0

USE_AKASOFU = False
USE_DENSITY = True
USE_BY_CLOCK = True
INCLUDE_BY_MAGNITUDE = False

# Suavização da derivada
USE_SAVITZKY_GOLAY = True
SG_WINDOW = 7
SG_ORDER = 2

# Calibração (escala física independente)
CALIBRATION_FACTOR = 30.0               # Fator multiplicativo padrão
USE_CALIBRATION_FROM_EVENTS = True      # Se True e arquivo existir, ajusta fator via regressão

# Limiares para validação de tempestade (usando Dst)
DST_THRESHOLD = -50          # nT (tempestade se Dst < -50)
HAC_THRESHOLD = 50           # Valor de HAC correspondente (pode ser ajustado)

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
        if INCLUDE_BY_MAGNITUDE:
            source[south_mask] += 0.5 * np.abs(By[south_mask]) * V[south_mask] * np.sqrt(N[south_mask])
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

def compute_baseline_simple(df, tau_hours, v_ref, tau_power, tau_min, tau_max):
    """Baseline: acumulado de (-Bz*V) (ignora densidade e By)"""
    source = np.zeros_like(df['bz_gsm'].values)
    south = df['bz_gsm'].values < 0
    source[south] = -df['bz_gsm'].values[south] * df['speed'].values[south] / 1000.0
    # Aplicar mesmo filtro exponencial
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
    tau_hours_arr = tau_base_hours * (v_ref / np.maximum(Vsw, 1.0)) ** tau_power
    tau_hours_arr = np.clip(tau_hours_arr, tau_min, tau_max)
    tau_seconds = tau_hours_arr * 3600.0
    alpha = np.exp(-dt_seconds / tau_seconds)
    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        hac[i] = alpha[i] * hac[i-1] + (1 - alpha[i]) * source[i]
    return hac

def compute_baseline_akasofu(df, tau_base_hours, v_ref, tau_power, tau_min, tau_max):
    """Baseline: acumulado do parâmetro ε de Akasofu"""
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
    tau_hours_arr = tau_base_hours * (v_ref / np.maximum(Vsw, 1.0)) ** tau_power
    tau_hours_arr = np.clip(tau_hours_arr, tau_min, tau_max)
    tau_seconds = tau_hours_arr * 3600.0
    alpha = np.exp(-dt_seconds / tau_seconds)
    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        hac[i] = alpha[i] * hac[i-1] + (1 - alpha[i]) * source[i]
    return hac

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
    # Suavização para derivada
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

def calibrate_from_events(calib_file, df):
    """
    Ajusta fator de escala (multiplicativo) para que HAC_calibrado se aproxime dos valores alvo.
    Arquivo deve ter colunas 'data' e 'hac_target'.
    """
    try:
        calib = pd.read_csv(calib_file, parse_dates=['data'])
        if 'hac_target' not in calib.columns:
            print("⚠️ Arquivo de calibração não tem coluna 'hac_target'. Usando fator padrão.")
            return None
        raw_vals = []
        target_vals = []
        for _, row in calib.iterrows():
            idx = np.argmin(np.abs(df['time_tag'] - row['data']))
            if idx is not None:
                raw = df.iloc[idx]['HAC_raw']
                if raw > 0:
                    raw_vals.append(raw)
                    target_vals.append(row['hac_target'])
        if len(raw_vals) < 2:
            print("⚠️ Poucos pontos para calibração. Usando fator padrão.")
            return None
        # Regressão linear sem intercepto (HAC_target = fator * HAC_raw)
        # pois queremos um fator multiplicativo puro.
        # Para forçar intercepto zero, usamos soma dos produtos / soma dos quadrados.
        factor = np.sum(np.array(target_vals) * np.array(raw_vals)) / np.sum(np.array(raw_vals)**2)
        print(f"📊 Calibração: fator multiplicativo = {factor:.3f}")
        return factor
    except Exception as e:
        print(f"⚠️ Erro na calibração: {e}")
        return None

def compute_metrics(hac_series, dst_series, threshold_dst=-50, threshold_hac=50):
    """
    Calcula métricas de validação comparando HAC (ou modelo) com Dst real.
    """
    # Alinhar temporalmente (mais próximo)
    df_merged = pd.merge_asof(hac_series.reset_index(), dst_series.reset_index(),
                              left_on='time_tag', right_on='time_tag', direction='nearest')
    df_merged = df_merged.dropna(subset=['dst', 'HAC'])
    if len(df_merged) == 0:
        return None
    corr, p = pearsonr(df_merged['dst'].abs(), df_merged['HAC'])
    rmse = np.sqrt(mean_squared_error(df_merged['dst'].abs(), df_merged['HAC']))
    mae = mean_absolute_error(df_merged['dst'].abs(), df_merged['HAC'])
    # Detecção de tempestade
    true_storm = df_merged['dst'] < threshold_dst
    pred_storm = df_merged['HAC'] > threshold_hac
    tp = np.sum(true_storm & pred_storm)
    fp = np.sum(~true_storm & pred_storm)
    fn = np.sum(true_storm & ~pred_storm)
    tn = np.sum(~true_storm & ~pred_storm)
    pod = tp / (tp + fn) if (tp+fn) > 0 else 0
    far = fp / (tp + fp) if (tp+fp) > 0 else 0
    csi = tp / (tp + fp + fn) if (tp+fp+fn) > 0 else 0
    return {
        'correlation': corr,
        'rmse': rmse,
        'mae': mae,
        'pod': pod,
        'far': far,
        'csi': csi,
        'n_points': len(df_merged)
    }

def create_comparison_figure(df, baseline_simple, baseline_akasofu, dst_df=None, filename="hac_comparison.png"):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Painel 1: HAC vs baselines
    axes[0].plot(df['time_tag'], df['HAC'], label='HAC (modelo principal)', color='#d62728', linewidth=2)
    axes[0].plot(df['time_tag'], baseline_simple, label='Baseline: (-Bz*V)', color='#1f77b4', linewidth=1.5, alpha=0.7)
    axes[0].plot(df['time_tag'], baseline_akasofu, label='Baseline: Akasofu ε', color='#ff7f0e', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Índice')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Comparação dos Modelos de Acoplamento')

    # Painel 2: Dst (se disponível)
    if dst_df is not None:
        axes[1].plot(dst_df['time_tag'], dst_df['dst'], label='Dst (nT)', color='black', linewidth=1.2)
        axes[1].axhline(y=DST_THRESHOLD, color='red', linestyle='--', label=f'Limiar tempestade ({DST_THRESHOLD} nT)')
        axes[1].set_ylabel('Dst [nT]')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Dst (Real)')
    else:
        axes[1].set_visible(False)

    # Painel 3: dHAC/dt
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
    print("🛰️  HAC - HELIOSPHERIC ACCUMULATED COUPLING (MODELO FINAL COM VALIDAÇÃO)")
    print("="*70)

    # Carregar dados
    mag_df = load_omni_data(MAG_FILE)
    plasma_df = load_omni_data(PLASMA_FILE)
    if mag_df is None or plasma_df is None:
        print("❌ Falha crítica.")
        return
    df = prepare_merged_data(mag_df, plasma_df)
    if len(df) < 10:
        print("❌ Dados insuficientes.")
        return
    print(f"   Dados válidos: {len(df)} pontos | Período: {df['time_tag'].min()} a {df['time_tag'].max()}")

    # Calcular HAC_raw
    df = calculate_hac_physical(df, calib_factor=1.0)
    # Calcular baselines
    baseline_simple = compute_baseline_simple(df, TAU_BASE_HOURS, V_REF, TAU_POWER, TAU_MIN_HOURS, TAU_MAX_HOURS)
    baseline_akasofu = compute_baseline_akasofu(df, TAU_BASE_HOURS, V_REF, TAU_POWER, TAU_MIN_HOURS, TAU_MAX_HOURS)

    # Calibração
    calib_factor = CALIBRATION_FACTOR
    if USE_CALIBRATION_FROM_EVENTS and CALIBRATION_FILE:
        new_factor = calibrate_from_events(CALIBRATION_FILE, df)
        if new_factor is not None:
            calib_factor = new_factor
    # Recalcular com fator calibrado
    df = calculate_hac_physical(df, calib_factor=calib_factor)
    # Recalcular baselines? Não recalibramos baselines, eles são comparados em escala raw.
    # Mas para comparação justa, podemos normalizá-los para ter mesma faixa? Melhor comparar correlação com Dst.

    # Carregar Dst se disponível
    dst_df = load_dst_data(DST_FILE) if DST_FILE else None

    # Validação
    print("\n📊 VALIDAÇÃO DO MODELO")
    print("="*50)
    metrics = {}
    if dst_df is not None:
        # Alinhar HAC com Dst
        hac_series = df[['time_tag', 'HAC']].copy()
        metrics_hac = compute_metrics(hac_series, dst_df, DST_THRESHOLD, HAC_THRESHOLD)
        if metrics_hac:
            metrics['HAC'] = metrics_hac
            print(f"✅ HAC (calibrado) vs Dst:")
            print(f"   Correlação: {metrics_hac['correlation']:.3f}")
            print(f"   RMSE: {metrics_hac['rmse']:.2f}")
            print(f"   MAE: {metrics_hac['mae']:.2f}")
            print(f"   POD: {metrics_hac['pod']:.3f} | FAR: {metrics_hac['far']:.3f} | CSI: {metrics_hac['csi']:.3f}")

        # Avaliar baselines
        baseline_simple_series = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': baseline_simple})
        metrics_simple = compute_metrics(baseline_simple_series, dst_df, DST_THRESHOLD, HAC_THRESHOLD)
        if metrics_simple:
            metrics['Simple'] = metrics_simple
            print(f"\n📉 Baseline (-Bz*V) vs Dst:")
            print(f"   Correlação: {metrics_simple['correlation']:.3f}")
            print(f"   RMSE: {metrics_simple['rmse']:.2f}")
            print(f"   POD: {metrics_simple['pod']:.3f} | FAR: {metrics_simple['far']:.3f} | CSI: {metrics_simple['csi']:.3f}")

        baseline_akasofu_series = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': baseline_akasofu})
        metrics_akasofu = compute_metrics(baseline_akasofu_series, dst_df, DST_THRESHOLD, HAC_THRESHOLD)
        if metrics_akasofu:
            metrics['Akasofu'] = metrics_akasofu
            print(f"\n📉 Baseline Akasofu ε vs Dst:")
            print(f"   Correlação: {metrics_akasofu['correlation']:.3f}")
            print(f"   RMSE: {metrics_akasofu['rmse']:.2f}")
            print(f"   POD: {metrics_akasofu['pod']:.3f} | FAR: {metrics_akasofu['far']:.3f} | CSI: {metrics_akasofu['csi']:.3f}")

    else:
        print("⚠️ Dados de Dst não disponíveis. Validação não realizada.")

    # Gerar figura comparativa
    create_comparison_figure(df, baseline_simple, baseline_akasofu, dst_df, "hac_comparison.png")

    # Salvar dados
    df.to_csv("hac_resultados_finais.csv", index=False, encoding='utf-8')
    print("\n💾 Dados salvos em hac_resultados_finais.csv")

    # Relatório final
    print("\n" + "="*70)
    print("🎯 RELATÓRIO FINAL")
    print("="*70)
    print(f"⚙️  Parâmetros: τ = {TAU_BASE_HOURS} * ({V_REF}/V)^{TAU_POWER:.1f} | Fonte: {'Akasofu' if USE_AKASOFU else 'Simplificado com ângulo de clock'}")
    print(f"   Calibração: fator {calib_factor:.3f}")
    if dst_df is not None and metrics:
        print("\n📈 Desempenho comparativo (correlação com |Dst|):")
        for name, m in metrics.items():
            print(f"   {name}: r = {m['correlation']:.3f} (CSI={m['csi']:.3f})")
    print("\n✅ Modelo pronto para análise científica.")
    print("="*70)

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    main()
