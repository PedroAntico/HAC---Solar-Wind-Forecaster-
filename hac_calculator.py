"""
HAC - Heliospheric Accumulated Coupling (Versão Final com Física Aprimorada)
Modelo com:
- Termo fonte estabilizado: V * (-Bz) * sin²(θ/2) * sqrt(B_total) * (Pdyn/P0)^(1/6)
- Normalização por constante física (independente do período)
- Memória adaptativa: τ_eff = τ / (1 + β * S)
- Perda não linear: τ_eff_base = τ * (1 + (H/H₀)^{γ-1})
- Filtro exponencial correto: H[i] = H[i-1]*α + S_eff * τ_eff * (1-α)
- Calibração não linear (log) com predição direta
- Validação com correlação, R², RMSE, MAE, POD, FAR, CSI
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime, timedelta
import warnings
import argparse
import os
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================
# CONFIGURAÇÃO DO MODELO (FÍSICA)
# ============================
# Parâmetros de memória
TAU_BASE_HOURS = 3.0          # constante de tempo base (horas)
V_REF = 400.0                 # velocidade de referência (km/s)
TAU_POWER = 0.3               # expoente da variação com V
TAU_MIN_HOURS = 1.0
TAU_MAX_HOURS = 6.0

# Termo fonte
USE_DENSITY = True            # influência da densidade via pressão dinâmica
USE_BY_CLOCK = True           # usa ângulo de clock
USE_DYNAMIC_PRESSURE = True   # usa pressão dinâmica como fator
# Parâmetros físicos da pressão dinâmica
P0 = 2.0                      # nPa (pressão de referência)
ALPHA_PDYN = 1/6              # expoente da compressão magnetosférica

# Normalização da fonte: constante física (independente do dataset)
SOURCE_NORM_CONSTANT = 1000.0  # valor calibrado a partir de um período de referência (pode ser ajustado)

# Saturação/adaptação da memória
BETA_SOURCE = 0.5             # aceleração da resposta: τ_eff = τ / (1 + β * S)

# Perda não linear (γ)
LOSS_EXPONENT = 1.5           # γ > 1 acelera dissipação em altas energias
H0 = 100.0                    # energia de referência para perda não linear (escala)

# Saturação da fonte (desligada)
H_SATURATION = 0.0

# Suavização da derivada (para gráfico)
USE_SAVITZKY_GOLAY = True
SG_WINDOW = 7
SG_ORDER = 2

# Parâmetros de validação
DST_THRESHOLD = -50
MAX_DELAY_HOURS = 12
DELAY_STEP_HOURS = 1
CALIBRATE_WITH_DELAY = True
DST_SMOOTH_HOURS = 3
MIN_DST_CALIB = 5

# Preditor para validação (usaremos HAC puro)
PREDICTOR_TYPE = 'hac'

# Transformação na calibração: 'log' ou 'sqrt'
CALIBRATION_TRANSFORM = 'log'

# Limiar HAC dinâmico baseado em Dst real
USE_DYNAMIC_HAC_THRESHOLD = True
HAC_THRESHOLD_PERCENTILE = 90

# ============================
# DOWNLOAD DOS DADOS OMNI
# ============================
def download_omni(start_date, end_date, resolution="hour"):
    """Baixa dados OMNI diretamente do servidor NASA."""
    url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
    payload = {
        "activity": "retrieve",
        "res": resolution,
        "spacecraft": "omni2",
        "start_date": start_date,
        "end_date": end_date,
        "vars": ["12", "13", "14", "21", "22", "26"],
        "scale": "Linear",
        "view": "0",
        "table": "0"
    }
    print(f"📡 Baixando OMNI {start_date} → {end_date}...")
    try:
        response = requests.post(url, data=payload, timeout=60)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Erro no download: {e}")
        return None

    text = response.text
    lines = text.split("\n")
    data_lines = [l for l in lines if l.strip() and l[0].isdigit()]
    if not data_lines:
        print("⚠️ Nenhum dado encontrado para o período.")
        return None

    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        delim_whitespace=True,
        header=None,
        na_values=[9999, 9999.99, 99999.0, 99999.99]
    )
    df.columns = [
        "year", "doy", "hour",
        "bx_gsm", "by_gsm", "bz_gsm",
        "density", "speed", "dst"
    ]
    df["time_tag"] = pd.to_datetime(df["year"], format="%Y") \
        + pd.to_timedelta(df["doy"] - 1, unit="D") \
        + pd.to_timedelta(df["hour"], unit="h")
    df = df[["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "density", "speed", "dst"]]
    df = df.dropna()
    print(f"✅ {len(df)} pontos baixados")
    return df

def download_omni_range(start_str, end_str, chunk_years=True):
    """Baixa um intervalo de datas, dividindo por anos se necessário."""
    start = datetime.strptime(start_str, "%Y%m%d")
    end = datetime.strptime(end_str, "%Y%m%d")
    if not chunk_years:
        return download_omni(start_str, end_str)
    years = range(start.year, end.year + 1)
    dfs = []
    for year in years:
        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31)
        if year_start < start:
            year_start = start
        if year_end > end:
            year_end = end
        if year_start > year_end:
            continue
        s = year_start.strftime("%Y%m%d")
        e = year_end.strftime("%Y%m%d")
        df_year = download_omni(s, e)
        if df_year is not None:
            dfs.append(df_year)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

# ============================
# FUNÇÕES AUXILIARES (PREPARAÇÃO)
# ============================
def prepare_data(df):
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

# ============================
# TERMO FONTE (COM FÍSICA APERFEIÇOADA)
# ============================
def compute_source_advanced(Bx, By, Bz, V, N):
    """
    Calcula S(t) = V * (-Bz) * sin²(θ/2) * sqrt(B_total) * (Pdyn/P0)**(1/6)
    Normalizado por constante física SOURCE_NORM_CONSTANT.
    """
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    theta = np.arctan2(By, Bz)
    recon = np.sin(theta / 2)**2 if USE_BY_CLOCK else 1.0
    bz_sul = -np.minimum(Bz, 0)   # positivo quando Bz < 0

    # Termo base: V * bz_sul * recon * sqrt(Btot)
    base = V * bz_sul * recon * np.sqrt(Btot)

    if USE_DENSITY and USE_DYNAMIC_PRESSURE:
        Pdyn = N * V**2               # nPa (se N em cm⁻³, V em km/s → escala adequada)
        Pdyn_norm = Pdyn / P0
        # Fator de compressão: (Pdyn/P0)^(1/6)
        dyn_factor = np.where(Pdyn > 0, Pdyn_norm**ALPHA_PDYN, 1.0)
        base *= dyn_factor

    base = np.clip(base, 0, 1e9)
    source = base / SOURCE_NORM_CONSTANT
    return source

def compute_source(df, use_akasofu=False):
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
        return epsilon / 1e6
    else:
        return compute_source_advanced(Bx, By, Bz, V, N)

# ============================
# CÁLCULO DO HAC (COM MEMÓRIA ADAPTATIVA E PERDA NÃO LINEAR)
# ============================
def calculate_hac(df, calib_factor=1.0,
                  tau_base=TAU_BASE_HOURS, v_ref=V_REF, power=TAU_POWER,
                  tau_min=TAU_MIN_HOURS, tau_max=TAU_MAX_HOURS,
                  loss_exp=LOSS_EXPONENT, h0=H0, h_sat=H_SATURATION,
                  beta_source=BETA_SOURCE):
    times = df['time_tag'].values
    Vsw = df['speed'].values
    source = compute_source(df, use_akasofu=False)

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
        dt = dt_seconds[i]
        tau = tau_seconds[i]

        # Saturação da fonte (opcional)
        if h_sat > 0:
            source_eff = source[i] / (1 + hac[i-1] / h_sat)
        else:
            source_eff = source[i]

        # Perda não linear: τ_base_eff = τ * (1 + (H/h0)^{γ-1})
        if loss_exp != 1.0:
            tau_loss_eff = tau * (1 + (hac[i-1] / h0)**(loss_exp - 1))
        else:
            tau_loss_eff = tau

        # Adaptação à fonte: acelera resposta quando há forte injeção
        tau_eff = tau_loss_eff / (1 + beta_source * source_eff)

        # Filtro exponencial correto (solução analítica)
        alpha = np.exp(-dt / tau_eff)
        hac[i] = hac[i-1] * alpha + source_eff * tau_eff * (1 - alpha)
        hac[i] = max(0, hac[i])

    hac_scaled = hac * calib_factor

    # Derivada suavizada (para gráfico)
    if USE_SAVITZKY_GOLAY and len(hac_scaled) >= SG_WINDOW:
        hac_smooth = savgol_filter(hac_scaled, window_length=SG_WINDOW, polyorder=SG_ORDER)
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
# MODELOS BASELINE (mesma estrutura, fonte diferente)
# ============================
def baseline_simple(df, tau_base, v_ref, power, tau_min, tau_max,
                    loss_exp=LOSS_EXPONENT, h0=H0, h_sat=H_SATURATION,
                    beta_source=BETA_SOURCE):
    source = np.zeros_like(df['bz_gsm'].values)
    south = df['bz_gsm'].values < 0
    source[south] = -df['bz_gsm'].values[south] * df['speed'].values[south] / 1000.0
    source = np.clip(source, 0, 1e9)
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
        dt = dt_seconds[i]
        tau = tau_seconds[i]
        if h_sat > 0:
            source_eff = source[i] / (1 + hac[i-1] / h_sat)
        else:
            source_eff = source[i]
        if loss_exp != 1.0:
            tau_loss_eff = tau * (1 + (hac[i-1] / h0)**(loss_exp - 1))
        else:
            tau_loss_eff = tau
        tau_eff = tau_loss_eff / (1 + beta_source * source_eff)
        alpha = np.exp(-dt / tau_eff)
        hac[i] = hac[i-1] * alpha + source_eff * tau_eff * (1 - alpha)
        hac[i] = max(0, hac[i])
    return hac

def baseline_akasofu(df, tau_base, v_ref, power, tau_min, tau_max,
                     loss_exp=LOSS_EXPONENT, h0=H0, h_sat=H_SATURATION,
                     beta_source=BETA_SOURCE):
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
    source = np.clip(source, 0, 1e9)
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
        dt = dt_seconds[i]
        tau = tau_seconds[i]
        if h_sat > 0:
            source_eff = source[i] / (1 + hac[i-1] / h_sat)
        else:
            source_eff = source[i]
        if loss_exp != 1.0:
            tau_loss_eff = tau * (1 + (hac[i-1] / h0)**(loss_exp - 1))
        else:
            tau_loss_eff = tau
        tau_eff = tau_loss_eff / (1 + beta_source * source_eff)
        alpha = np.exp(-dt / tau_eff)
        hac[i] = hac[i-1] * alpha + source_eff * tau_eff * (1 - alpha)
        hac[i] = max(0, hac[i])
    return hac

# ============================
# FUNÇÕES DE VALIDAÇÃO (COM PREDIÇÃO DIRETA)
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

def compute_metrics(y_true, y_pred, threshold_dst=-50, threshold_hac=None, hac_vals=None):
    if len(y_true) < 2:
        return None
    corr = pearsonr(y_true, y_pred)[0] if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    pod = far = csi = None
    if threshold_hac is not None and hac_vals is not None:
        true_storm = y_true > threshold_dst  # y_true é -dst_smooth, então positivo
        pred_storm = hac_vals > threshold_hac
        tp = np.sum(true_storm & pred_storm)
        fp = np.sum(~true_storm & pred_storm)
        fn = np.sum(true_storm & ~pred_storm)
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {'correlation': corr, 'r2': r2, 'rmse': rmse, 'mae': mae,
            'pod': pod, 'far': far, 'csi': csi}

def compute_validation_metrics(df_hac, dst_series, predictor='hac', smooth_dst_hours=DST_SMOOTH_HOURS):
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
    elif predictor == 'deriv':
        y_pred = merged['dHAC_dt'].values
        y_pred = np.maximum(0, y_pred)
    elif predictor == 'hac_ma':
        y_pred = merged['HAC_ma'].values
    else:
        return None
    y_true = -merged['dst_smooth'].values
    return compute_metrics(y_true, y_pred)

def find_optimal_delay(df_hac, dst_series, max_delay=MAX_DELAY_HOURS, step=DELAY_STEP_HOURS,
                       smooth_dst_hours=DST_SMOOTH_HOURS, predictor='hac'):
    best_corr = -1
    best_delay = 0
    print(f"\n🔍 Buscando atraso ótimo (predictor={predictor})...")
    for delay in np.arange(0, max_delay + step, step):
        shifted_dst = apply_time_shift(dst_series, delay)
        metrics = compute_validation_metrics(df_hac, shifted_dst, predictor=predictor,
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

def calibrate_hac_nonlinear(df_hac_raw, dst_series, delay=0, min_hac=0.01, min_dst=MIN_DST_CALIB,
                            smooth_dst_hours=DST_SMOOTH_HOURS, predictor='hac',
                            transform=CALIBRATION_TRANSFORM):
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

# ============================
# FIGURA COMPARATIVA
# ============================
def create_figure(df, baseline_simple_vals, baseline_akasofu_vals, dst_df=None, filename="hac_comparison.png",
                  delay=0, smooth_dst_hours=DST_SMOOTH_HOURS, predictor='hac'):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(df['time_tag'], df['HAC'], label='HAC (modelo)', color='#d62728', linewidth=2)
    axes[0].plot(df['time_tag'], baseline_simple_vals, label='Baseline (-Bz*V)', color='#1f77b4', linewidth=1.5, alpha=0.7)
    axes[0].plot(df['time_tag'], baseline_akasofu_vals, label='Baseline Akasofu ε', color='#ff7f0e', linewidth=1.5, alpha=0.7)
    if predictor == 'hac_ma' and 'HAC_ma' in df.columns:
        axes[0].plot(df['time_tag'], df['HAC_ma'], label='HAC médio (preditor)', color='green', linewidth=1.2, linestyle='--')
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
    print(f"✅ Figura salva: {filename}")

# ============================
# FUNÇÃO PRINCIPAL
# ============================
def main():
    parser = argparse.ArgumentParser(description='HAC - Heliospheric Accumulated Coupling (versão final)')
    parser.add_argument('--start', default='20000101', help='Data inicial (YYYYMMDD)')
    parser.add_argument('--end', default='20200101', help='Data final (YYYYMMDD)')
    parser.add_argument('--download', action='store_true', help='Forçar download dos dados')
    parser.add_argument('--input', default='omni_data.csv', help='Arquivo CSV local')
    parser.add_argument('--output', default='hac_results.csv', help='Arquivo de saída')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🛰️  HAC - Heliospheric Accumulated Coupling (Versão Final)")
    print("="*70)

    # Carrega ou baixa dados
    if args.download or not os.path.exists(args.input):
        print(f"\n📡 Baixando dados OMNI de {args.start} a {args.end}...")
        df_raw = download_omni_range(args.start, args.end, chunk_years=True)
        if df_raw is None:
            print("❌ Falha no download. Abortando.")
            return
        df_raw.to_csv(args.input, index=False)
        print(f"💾 Dados salvos em {args.input}")
    else:
        print(f"\n📂 Lendo dados locais de {args.input}...")
        df_raw = pd.read_csv(args.input, parse_dates=['time_tag'])

    # Prepara dados
    df = prepare_data(df_raw)
    print(f"   Dados após limpeza: {len(df)} pontos")
    print(f"   Período: {df['time_tag'].min()} a {df['time_tag'].max()}")

    # Verifica se há tempestades no período
    dst_min = df['dst'].min()
    print(f"\n📊 Dst mínimo no período: {dst_min:.1f} nT")
    if dst_min > -30:
        print("⚠️  ALERTA: O período selecionado tem tempestades muito fracas (Dst > -30 nT).")
        print("   Recomendamos usar um período com tempestades reais, por exemplo:")
        print("   --start 20000101 --end 20200101  (inclui Halloween Storm, 2015 storm, etc.)")
        print("   A validação pode ser prejudicada.\n")

    bz_neg = (df['bz_gsm'] < 0).sum()
    print(f"🔎 Bz sul: {bz_neg} pontos ({bz_neg/len(df)*100:.1f}%)")
    if bz_neg == 0:
        print("⚠️  ALERTA: Nenhum ponto com Bz negativo! O acoplamento será zero.")

    # Calcula HAC (primeira passagem sem calibração)
    df = calculate_hac(df, calib_factor=1.0)

    # Calcula baselines (usando nomes diferentes das funções)
    baseline_simple_vals = baseline_simple(df, TAU_BASE_HOURS, V_REF, TAU_POWER, TAU_MIN_HOURS, TAU_MAX_HOURS)
    baseline_akasofu_vals = baseline_akasofu(df, TAU_BASE_HOURS, V_REF, TAU_POWER, TAU_MIN_HOURS, TAU_MAX_HOURS)

    # Prepara dados de Dst
    dst_df = df[['time_tag', 'dst']].copy()
    if len(dst_df) == 0:
        print("❌ Nenhum dado de Dst disponível. Abortando validação.")
        return

    # DataFrames para validação
    df_hac = df[['time_tag', 'HAC', 'dHAC_dt']].copy()
    df_hac_raw = df[['time_tag', 'HAC_raw', 'dHAC_dt']].copy()

    print("\n📊 Estatísticas HAC_raw:")
    print(df['HAC_raw'].describe())
    print("\n📊 Estatísticas Dst (apenas negativos):")
    print(dst_df[dst_df['dst'] < 0]['dst'].describe())

    # Encontra delay ótimo
    best_delay, best_corr = find_optimal_delay(df_hac, dst_df, max_delay=MAX_DELAY_HOURS,
                                               smooth_dst_hours=DST_SMOOTH_HOURS, predictor=PREDICTOR_TYPE)

    # Calibração não linear
    calib_slope = None
    calib_intercept = None
    if CALIBRATE_WITH_DELAY:
        calib_slope, calib_intercept = calibrate_hac_nonlinear(
            df_hac_raw, dst_df, delay=best_delay, min_dst=MIN_DST_CALIB,
            smooth_dst_hours=DST_SMOOTH_HOURS, predictor=PREDICTOR_TYPE,
            transform=CALIBRATION_TRANSFORM)
        if calib_slope is None:
            print("⚠️ Calibração falhou. Usando fator linear padrão.")
            # Fallback: regressão linear simples
            shifted_dst = apply_time_shift(dst_df, best_delay)
            merged = pd.merge_asof(df_hac_raw.sort_values('time_tag'),
                                   shifted_dst.sort_values('time_tag'),
                                   on='time_tag', direction='backward')
            merged = smooth_dst(merged, DST_SMOOTH_HOURS)
            mask = (merged['dst_smooth'] < 0) & (merged['dst_smooth'].abs() > MIN_DST_CALIB) & (merged['HAC_raw'] > 0.01)
            if mask.sum() > 5:
                x = merged.loc[mask, 'HAC_raw'].values
                y = -merged.loc[mask, 'dst_smooth'].values
                slope, intercept, r, p, se = linregress(x, y)
                calib_slope = slope
                calib_intercept = intercept
                print(f"   Fallback linear: |Dst| = {slope:.3f} * HAC_raw + {intercept:.3f}")
            else:
                calib_slope = 1.0
                calib_intercept = 0.0

    # Aplicar calibração: recalcular HAC com fator linear aproximado (apenas para visualização)
    if calib_slope is not None and calib_intercept is not None:
        hac_mean = df['HAC_raw'].mean()
        dst_pred_mean = predict_from_calibration(hac_mean, calib_slope, calib_intercept, CALIBRATION_TRANSFORM)
        linear_factor = dst_pred_mean / hac_mean if hac_mean > 0 else 1.0
        if not (0.01 < linear_factor < 100):
            print(f"⚠️ Fator linear {linear_factor:.3f} fora da faixa aceitável (0.01-100). Usando 1.0.")
            linear_factor = 1.0
        df = calculate_hac(df, calib_factor=linear_factor)
        baseline_simple_vals = baseline_simple_vals * linear_factor
        baseline_akasofu_vals = baseline_akasofu_vals * linear_factor
        df_hac = df[['time_tag', 'HAC', 'dHAC_dt']].copy()
        if PREDICTOR_TYPE == 'hac_ma':
            df['HAC_ma'] = df['HAC'].rolling(window=5, min_periods=1).mean()
            df_hac['HAC_ma'] = df['HAC_ma'].copy()
        print(f"✅ HAC recalibrado com fator linear aproximado: {linear_factor:.3f}")

    # Validação direta com calibração não linear
    print("\n📊 VALIDAÇÃO DIRETA COM CALIBRAÇÃO NÃO LINEAR")
    print("="*50)
    if calib_slope is not None and calib_intercept is not None:
        shifted_dst = apply_time_shift(dst_df, best_delay)
        merged = pd.merge_asof(df[['time_tag', 'HAC_raw']].sort_values('time_tag'),
                               shifted_dst.sort_values('time_tag'),
                               on='time_tag', direction='backward')
        merged = merged.dropna(subset=['dst', 'HAC_raw'])
        merged = smooth_dst(merged, DST_SMOOTH_HOURS)
        merged = merged[merged['dst_smooth'] < 0]
        if len(merged) > 10:
            y_true = -merged['dst_smooth'].values
            merged['dst_pred'] = predict_from_calibration(merged['HAC_raw'].values, calib_slope, calib_intercept, CALIBRATION_TRANSFORM)
            y_pred = merged['dst_pred'].values
            metrics_direct = compute_metrics(y_true, y_pred)
            if metrics_direct:
                print(f"✅ Modelo principal (predição direta) vs Dst suavizado (delay={best_delay}h):")
                print(f"   Correlação: {metrics_direct['correlation']:.3f}")
                print(f"   R²: {metrics_direct['r2']:.3f}")
                print(f"   RMSE: {metrics_direct['rmse']:.2f}")
                print(f"   MAE: {metrics_direct['mae']:.2f}")

    # Validação com HAC calibrado linear (para comparação com baselines)
    print("\n📊 VALIDAÇÃO COM HAC CALIBRADO (LINEAR)")
    print("="*50)
    shifted_dst = apply_time_shift(dst_df, best_delay)
    metrics_hac = compute_validation_metrics(df_hac, shifted_dst, predictor=PREDICTOR_TYPE,
                                             smooth_dst_hours=DST_SMOOTH_HOURS)
    if metrics_hac:
        print(f"✅ HAC (calibrado linear) vs Dst suavizado (delay={best_delay}h):")
        print(f"   Correlação: {metrics_hac['correlation']:.3f}")
        print(f"   R²: {metrics_hac['r2']:.3f}")
        print(f"   RMSE: {metrics_hac['rmse']:.2f}")
        print(f"   MAE: {metrics_hac['mae']:.2f}")

    # Baselines
    df_simple = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': baseline_simple_vals})
    metrics_simple = compute_validation_metrics(df_simple, shifted_dst, predictor='hac',
                                                smooth_dst_hours=DST_SMOOTH_HOURS)
    if metrics_simple:
        print(f"\n📉 Baseline (-Bz*V) (calibrado) vs Dst suavizado (delay={best_delay}h):")
        print(f"   Correlação: {metrics_simple['correlation']:.3f}")
        print(f"   R²: {metrics_simple['r2']:.3f}")
        print(f"   RMSE: {metrics_simple['rmse']:.2f}")

    df_akasofu = pd.DataFrame({'time_tag': df['time_tag'], 'HAC': baseline_akasofu_vals})
    metrics_akasofu = compute_validation_metrics(df_akasofu, shifted_dst, predictor='hac',
                                                 smooth_dst_hours=DST_SMOOTH_HOURS)
    if metrics_akasofu:
        print(f"\n📉 Baseline Akasofu ε (calibrado) vs Dst suavizado (delay={best_delay}h):")
        print(f"   Correlação: {metrics_akasofu['correlation']:.3f}")
        print(f"   R²: {metrics_akasofu['r2']:.3f}")
        print(f"   RMSE: {metrics_akasofu['rmse']:.2f}")

    # Limiar HAC dinâmico
    tmp = pd.merge_asof(df_hac[['time_tag', 'HAC']].sort_values('time_tag'),
                        shifted_dst.sort_values('time_tag'),
                        on='time_tag', direction='backward')
    tmp = tmp.dropna(subset=['dst'])
    tmp = smooth_dst(tmp, DST_SMOOTH_HOURS)
    storm_vals = tmp[tmp['dst_smooth'] < DST_THRESHOLD]['HAC']
    if len(storm_vals) > 0:
        hac_threshold = np.median(storm_vals)
        print(f"\n📊 Limiar HAC baseado em tempestade real (Dst < {DST_THRESHOLD} nT): {hac_threshold:.2f}")
    else:
        hac_threshold = np.percentile(df['HAC'], HAC_THRESHOLD_PERCENTILE)
        print(f"\n⚠️ Nenhum ponto com Dst < {DST_THRESHOLD} nT. Usando percentil {HAC_THRESHOLD_PERCENTILE}: {hac_threshold:.2f}")

    # Gráfico
    create_figure(df, baseline_simple_vals, baseline_akasofu_vals, dst_df, "hac_comparison.png",
                  delay=best_delay, smooth_dst_hours=DST_SMOOTH_HOURS, predictor=PREDICTOR_TYPE)

    # Salva resultados
    df.to_csv(args.output, index=False)
    print(f"\n💾 Resultados salvos em {args.output}")

    print("\n" + "="*70)
    print("🎯 RELATÓRIO FINAL")
    print("="*70)
    print(f"⚙️  Parâmetros do modelo (física final):")
    print(f"   • Termo fonte: V * (-Bz) * sin²(θ/2) * sqrt(B_total) * (Pdyn/{P0:.1f})^{ALPHA_PDYN:.2f}")
    print(f"   • Normalização: constante física SOURCE_NORM = {SOURCE_NORM_CONSTANT:.1f}")
    print(f"   • Memória adaptativa: τ_eff = τ / (1 + {BETA_SOURCE} * S)")
    print(f"   • Perda não linear: τ_base = τ * (1 + (H/{H0})^{LOSS_EXPONENT-1})")
    print(f"   • Filtro: H[i] = H[i-1]*α + S_eff * τ_eff * (1-α)")
    print(f"   • τ = {TAU_BASE_HOURS} * ({V_REF}/V)^{TAU_POWER:.1f} h, limitado entre {TAU_MIN_HOURS} e {TAU_MAX_HOURS} h")
    print(f"   • Atraso ótimo: {best_delay}h (correlação máxima: {best_corr:.3f})")
    print(f"   • Calibração: transformação {CALIBRATION_TRANSFORM}")
    if calib_slope is not None:
        print(f"   • Parâmetros calibração: slope={calib_slope:.3f}, intercept={calib_intercept:.3f}")
    print(f"   • Limiar HAC (referência): {hac_threshold:.2f}")
    print("\n✅ Modelo agora com normalização física, memória adaptativa e pressão dinâmica realista.")
    print("   Execute com período contendo tempestades (ex: 2000-2020) para validar.")

if __name__ == "__main__":
    main()
