"""
Módulo para cálculo de métricas de coerência do campo magnético.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from dataclasses_json import dataclass_json
import warnings
from scipy import signal, stats, fft

@dataclass_json
@dataclass
class CoherenceMetrics:
    """
    Métricas de coerência do campo magnético.
    
    Attributes
    ----------
    bt_std : float
        Desvio padrão de Bt.
    bt_rel_std : float
        Desvio padrão relativo de Bt (std/mean).
    bz_std : float
        Desvio padrão de Bz.
    bz_rel_std : float
        Desvio padrão relativo de Bz.
    bz_lag1_autocorr : float
        Autocorrelação lag-1 de Bz.
    bt_lag1_autocorr : float
        Autocorrelação lag-1 de Bt.
    spectral_coherence : float
        Coerência espectral entre Bz e Bt (0-1).
    trend_stability : float
        Medida de estabilidade da tendência (0-1).
    field_rotation_rate : float
        Taxa média de rotação do campo (graus/min).
    """
    bt_std: float
    bt_rel_std: float
    bz_std: float
    bz_rel_std: float
    bz_lag1_autocorr: float
    bt_lag1_autocorr: float
    spectral_coherence: float
    trend_stability: float
    field_rotation_rate: float


def calculate_autocorrelation(series: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """
    Calcula autocorrelação de uma série.
    
    Parameters
    ----------
    series : np.ndarray
        Série temporal.
    max_lag : int
        Lag máximo para cálculo.
    
    Returns
    -------
    np.ndarray
        Autocorrelação para lags 0-max_lag.
    """
    n = len(series)
    if n < max_lag * 2:
        max_lag = n // 2
    
    autocorr = np.zeros(max_lag + 1)
    series_mean = np.nanmean(series)
    series_var = np.nanvar(series)
    
    if series_var == 0:
        return autocorr
    
    for lag in range(max_lag + 1):
        if lag < n:
            cov = np.nanmean((series[lag:] - series_mean) * (series[:n-lag] - series_mean))
            autocorr[lag] = cov / series_var
    
    return autocorr


def calculate_spectral_coherence(bz: np.ndarray, bt: np.ndarray, 
                                sampling_rate: float = 1/60) -> float:
    """
    Calcula coerência espectral entre Bz e Bt.
    
    Parameters
    ----------
    bz, bt : np.ndarray
        Séries temporais.
    sampling_rate : float
        Taxa de amostragem em Hz.
    
    Returns
    -------
    float
        Coerência espectral média (0-1).
    """
    n = min(len(bz), len(bt))
    if n < 10:
        return 0.0
    
    # Remove tendência linear
    bz_detrend = signal.detrend(bz[:n])
    bt_detrend = signal.detrend(bt[:n])
    
    # Calcula coerência
    f, Cxy = signal.coherence(bz_detrend, bt_detrend, 
                              fs=1/sampling_rate, nperseg=min(256, n//4))
    
    # Retorna coerência média nas frequências baixas (< 1 mHz)
    low_freq_mask = f < 1e-3  # < 1 mHz
    if np.any(low_freq_mask):
        return float(np.mean(Cxy[low_freq_mask]))
    else:
        return float(np.mean(Cxy))


def calculate_field_rotation(bz: np.ndarray, by: Optional[np.ndarray] = None,
                            bx: Optional[np.ndarray] = None) -> float:
    """
    Calcula a taxa de rotação do campo magnético.
    
    Parameters
    ----------
    bz : np.ndarray
        Componente Bz.
    by, bx : np.ndarray, optional
        Componentes By e Bx para cálculo 3D.
    
    Returns
    -------
    float
        Taxa média de rotação (graus/min).
    """
    n = len(bz)
    if n < 2:
        return 0.0
    
    if by is not None and bx is not None and len(by) == n and len(bx) == n:
        # Cálculo 3D
        b_norm = np.sqrt(bx**2 + by**2 + bz**2)
        mask = b_norm > 0.1  # Evita divisão por zero
        
        if np.sum(mask) < 2:
            return 0.0
        
        bx_norm = bx[mask] / b_norm[mask]
        by_norm = by[mask] / b_norm[mask]
        bz_norm = bz[mask] / b_norm[mask]
        
        # Ângulos esféricos
        phi = np.arctan2(by_norm, bx_norm)  # longitude
        theta = np.arccos(bz_norm)  # colatitude
        
        # Variação angular
        d_phi = np.diff(phi)
        d_theta = np.diff(theta)
        
        # Ângulo de rotação total
        rotation_angles = np.arccos(
            np.cos(theta[:-1]) * np.cos(theta[1:]) +
            np.sin(theta[:-1]) * np.sin(theta[1:]) * np.cos(d_phi)
        )
        
        rotation_rate = np.nanmean(np.degrees(rotation_angles))
    else:
        # Cálculo simplificado apenas com Bz
        bz_normalized = bz / (np.abs(bz) + 1e-6)
        angles = np.arccos(np.clip(bz_normalized, -1, 1))
        rotation_angles = np.abs(np.diff(angles))
        rotation_rate = np.nanmean(np.degrees(rotation_angles)) if len(angles) > 1 else 0.0
    
    return float(rotation_rate)


def compute_coherence_metrics(
    bz_series: pd.Series,
    bt_series: pd.Series,
    by_series: Optional[pd.Series] = None,
    bx_series: Optional[pd.Series] = None
) -> CoherenceMetrics:
    """
    Calcula métricas de coerência do campo magnético.
    
    Parameters
    ----------
    bz_series, bt_series : pd.Series
        Séries temporais de Bz e Bt.
    by_series, bx_series : pd.Series, optional
        Componentes By e Bx para análise 3D.
    
    Returns
    -------
    CoherenceMetrics
        Métricas de coerência.
    """
    from .alignment import validate_time_series
    
    # Valida e alinha séries
    bz, bt = validate_time_series(bz_series, bt_series)
    n = len(bz)
    
    # Calcula estatísticas básicas
    bt_mean = float(np.nanmean(np.abs(bt.values)))
    bz_mean = float(np.nanmean(np.abs(bz.values)))
    
    bt_std = float(np.nanstd(bt.values))
    bz_std = float(np.nanstd(bz.values))
    
    bt_rel_std = bt_std / (bt_mean + 1e-6)
    bz_rel_std = bz_std / (bz_mean + 1e-6)
    
    # Autocorrelações
    bz_autocorr = calculate_autocorrelation(bz.values, max_lag=10)
    bt_autocorr = calculate_autocorrelation(bt.values, max_lag=10)
    
    bz_lag1 = bz_autocorr[1] if len(bz_autocorr) > 1 else 0.0
    bt_lag1 = bt_autocorr[1] if len(bt_autocorr) > 1 else 0.0
    
    # Coerência espectral
    spectral_coherence = calculate_spectral_coherence(bz.values, bt.values)
    
    # Estabilidade da tendência
    if n > 10:
        # Fit linear para detectar tendências
        x = np.arange(n)
        bz_slope, bz_intercept = np.polyfit(x, bz.values, 1)[:2]
        bt_slope, bt_intercept = np.polyfit(x, bt.values, 1)[:2]
        
        # R² das tendências
        bz_pred = bz_slope * x + bz_intercept
        bt_pred = bt_slope * x + bt_intercept
        
        bz_r2 = 1 - np.var(bz.values - bz_pred) / (np.var(bz.values) + 1e-6)
        bt_r2 = 1 - np.var(bt.values - bt_pred) / (np.var(bt.values) + 1e-6)
        
        trend_stability = float((bz_r2 + bt_r2) / 2)
    else:
        trend_stability = 0.0
    
    # Taxa de rotação do campo
    if by_series is not None and bx_series is not None:
        # Alinha todas as séries
        aligned = bz.align(bt, by_series, bx_series, join='inner')
        bz_aligned, bt_aligned, by_aligned, bx_aligned = aligned
        field_rotation_rate = calculate_field_rotation(
            bz_aligned.values, by_aligned.values, bx_aligned.values
        )
    else:
        field_rotation_rate = calculate_field_rotation(bz.values)
    
    return CoherenceMetrics(
        bt_std=bt_std,
        bt_rel_std=float(bt_rel_std),
        bz_std=bz_std,
        bz_rel_std=float(bz_rel_std),
        bz_lag1_autocorr=float(bz_lag1),
        bt_lag1_autocorr=float(bt_lag1),
        spectral_coherence=float(spectral_coherence),
        trend_stability=float(trend_stability),
        field_rotation_rate=float(field_rotation_rate)
    )


def calculate_angular_dispersion(bz: np.ndarray, bt: np.ndarray) -> float:
    """
    Calcula dispersão angular do fluxo.
    
    Parameters
    ----------
    bz, bt : np.ndarray
        Componentes do campo magnético.
    
    Returns
    -------
    float
        Dispersão angular em graus (0-90).
    """
    if len(bz) < 2:
        return 90.0  # Máxima dispersão
    
    # Ângulo do campo: θ = arccos(Bz/Bt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = np.degrees(np.arccos(np.clip(bz / (bt + 1e-6), -1, 1)))
    
    # Remove NaNs
    angles = angles[np.isfinite(angles)]
    
    if len(angles) < 2:
        return 90.0
    
    # Dispersão como desvio padrão circular
    angles_rad = np.radians(angles)
    mean_angle = np.arctan2(np.mean(np.sin(angles_rad)), 
                           np.mean(np.cos(angles_rad)))
    
    # Dispersão circular
    R = np.sqrt(np.mean(np.cos(angles_rad))**2 + np.mean(np.sin(angles_rad))**2)
    circular_std = np.sqrt(-2 * np.log(R))
    
    return float(np.degrees(circular_std))


# Testes
if __name__ == "__main__":
    # Dados de teste
    np.random.seed(42)
    n = 200
    t = np.linspace(0, 4*np.pi, n)
    
    # Bz com coerência
    bz = 10 * np.sin(t) + np.random.normal(0, 2, n)
    bt = 15 + 5 * np.sin(t + 0.1) + np.random.normal(0, 1, n)
    by = 5 * np.cos(t) + np.random.normal(0, 1, n)
    bx = 8 * np.sin(0.5*t) + np.random.normal(0, 1, n)
    
    dates = pd.date_range('2023-12-03', periods=n, freq='1min')
    bz_series = pd.Series(bz, index=dates)
    bt_series = pd.Series(bt, index=dates)
    by_series = pd.Series(by, index=dates)
    bx_series = pd.Series(bx, index=dates)
    
    # Calcula métricas
    metrics = compute_coherence_metrics(bz_series, bt_series, by_series, bx_series)
    
    print("Métricas de coerência:")
    print(f"  Std Bt: {metrics.bt_std:.2f} nT")
    print(f"  Std rel Bt: {metrics.bt_rel_std:.3f}")
    print(f"  Autocorr Bz (lag1): {metrics.bz_lag1_autocorr:.3f}")
    print(f"  Coerência espectral: {metrics.spectral_coherence:.3f}")
    print(f"  Estabilidade tendência: {metrics.trend_stability:.3f}")
    print(f"  Taxa rotação campo: {metrics.field_rotation_rate:.2f} °/min")
    
    # Dispersão angular
    dispersion = calculate_angular_dispersion(bz, bt)
    print(f"  Dispersão angular: {dispersion:.1f}°")
