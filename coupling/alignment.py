"""
Módulo para cálculo de métricas de alinhamento magnético.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from dataclasses_json import dataclass_json
import warnings
from scipy import stats

@dataclass_json
@dataclass
class AlignmentMetrics:
    """
    Métricas de alinhamento magnético.
    
    Attributes
    ----------
    south_fraction : float
        Fração do tempo com Bz < limiar sul.
    south_persistence_score : float
        Score de persistência (0-1) baseado em duração contínua.
    mean_Bt : float
        Magnitude média do campo magnético total.
    mean_Bz : float
        Componente Bz média.
    bz_variability : float
        Desvio padrão de Bz.
    max_south_duration : float
        Duração máxima contínua de Bz sul em minutos.
    n_points : int
        Número de pontos válidos.
    data_coverage : float
        Fração de dados válidos (0-1).
    """
    south_fraction: float
    south_persistence_score: float
    mean_Bt: float
    mean_Bz: float
    bz_variability: float
    max_south_duration: float
    n_points: int
    data_coverage: float


def validate_time_series(bz_series: pd.Series, bt_series: pd.Series, 
                        min_points: int = 10) -> Tuple[pd.Series, pd.Series]:
    """
    Valida e prepara séries temporais para análise.
    
    Parameters
    ----------
    bz_series, bt_series : pd.Series
        Séries temporais com índice datetime.
    min_points : int
        Número mínimo de pontos requeridos.
    
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Séries validadas e alinhadas.
    
    Raises
    ------
    ValueError
        Se os dados forem insuficientes ou inválidos.
    """
    # Verifica tipos
    if not isinstance(bz_series, pd.Series) or not isinstance(bt_series, pd.Series):
        raise TypeError("As entradas devem ser pandas Series")
    
    # Verifica se são séries temporais
    if not isinstance(bz_series.index, pd.DatetimeIndex):
        raise TypeError("bz_series deve ter índice DatetimeIndex")
    if not isinstance(bt_series.index, pd.DatetimeIndex):
        raise TypeError("bt_series deve ter índice DatetimeIndex")
    
    # Remove NaNs e infs
    bz_clean = bz_series.replace([np.inf, -np.inf], np.nan).dropna()
    bt_clean = bt_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Verifica dados suficientes
    if len(bz_clean) < min_points or len(bt_clean) < min_points:
        raise ValueError(f"Dados insuficientes: bz={len(bz_clean)}, bt={len(bt_clean)}")
    
    # Alinha temporalmente
    aligned = bz_clean.align(bt_clean, join='inner')
    bz_aligned, bt_aligned = aligned
    
    if len(bz_aligned) < min_points:
        raise ValueError(f"Poucos dados após alinhamento: {len(bz_aligned)} pontos")
    
    # Verifica consistência temporal
    time_diff = bz_aligned.index.to_series().diff().dt.total_seconds()
    if time_diff.std() > time_diff.mean() * 0.5:
        warnings.warn("Série temporal pode ter intervalos inconsistentes")
    
    return bz_aligned, bt_aligned


def detect_persistence_blocks(bz_values: np.ndarray, 
                             is_south: np.ndarray,
                             dt_minutes: float) -> Dict[str, Any]:
    """
    Detecta blocos de persistência em séries de Bz.
    
    Returns
    -------
    Dict com informações sobre persistência.
    """
    blocks = []
    current_block_start = None
    current_block_length = 0
    
    for i, (value, south) in enumerate(zip(bz_values, is_south)):
        if south:
            if current_block_start is None:
                current_block_start = i
            current_block_length += 1
        else:
            if current_block_start is not None:
                blocks.append({
                    'start_idx': current_block_start,
                    'length': current_block_length,
                    'duration_min': current_block_length * dt_minutes,
                    'mean_bz': np.mean(bz_values[current_block_start:current_block_start + current_block_length])
                })
                current_block_start = None
                current_block_length = 0
    
    # Último bloco se terminar em sul
    if current_block_start is not None:
        blocks.append({
            'start_idx': current_block_start,
            'length': current_block_length,
            'duration_min': current_block_length * dt_minutes,
            'mean_bz': np.mean(bz_values[current_block_start:current_block_start + current_block_length])
        })
    
    return {
        'blocks': blocks,
        'n_blocks': len(blocks),
        'max_duration': max([b['duration_min'] for b in blocks]) if blocks else 0,
        'total_south_time': sum([b['duration_min'] for b in blocks])
    }


def compute_alignment_metrics(
    bz_series: pd.Series,
    bt_series: pd.Series,
    south_threshold: float = -10.0,
    min_persistence_minutes: int = 30,
    require_continuity: bool = True
) -> AlignmentMetrics:
    """
    Calcula métricas de alinhamento magnético.
    
    Parameters
    ----------
    bz_series, bt_series : pd.Series
        Séries temporais de Bz e Bt.
    south_threshold : float
        Limiar para considerar Bz sul (nT).
    min_persistence_minutes : int
        Duração mínima para considerar persistência (minutos).
    require_continuity : bool
        Se True, verifica continuidade temporal.
    
    Returns
    -------
    AlignmentMetrics
        Métricas calculadas.
    """
    # Validação inicial
    bz, bt = validate_time_series(bz_series, bt_series)
    
    n_total = len(bz)
    n_expected = len(pd.date_range(start=bz.index[0], end=bz.index[-1], 
                                   freq='1min')) if require_continuity else n_total
    
    # Cálculo do intervalo médio
    if n_total > 1:
        dt_minutes = np.median(
            np.diff(bz.index.values).astype('timedelta64[ms]').astype(float) / (1000 * 60)
        )
    else:
        dt_minutes = 1.0
    
    # Máscara para Bz sul
    is_south = bz.values < south_threshold
    south_fraction = float(np.mean(is_south))
    
    # Análise de persistência
    persistence_info = detect_persistence_blocks(bz.values, is_south, dt_minutes)
    max_duration = persistence_info['max_duration']
    
    # Score de persistência (normalizado)
    if max_duration >= min_persistence_minutes:
        persistence_score = min(1.0, max_duration / (min_persistence_minutes * 3))
    else:
        persistence_score = max_duration / min_persistence_minutes
    
    # Cálculo de estatísticas
    mean_bt = float(np.nanmean(bt.values))
    mean_bz = float(np.nanmean(bz.values))
    bz_std = float(np.nanstd(bz.values))
    
    # Cobertura de dados
    data_coverage = n_total / n_expected if n_expected > 0 else 1.0
    
    return AlignmentMetrics(
        south_fraction=south_fraction,
        south_persistence_score=persistence_score,
        mean_Bt=mean_bt,
        mean_Bz=mean_bz,
        bz_variability=bz_std,
        max_south_duration=max_duration,
        n_points=n_total,
        data_coverage=data_coverage
    )


def calculate_integrated_magnetic_density(bt_series: pd.Series) -> float:
    """
    Calcula a densidade magnética integrada: ∫|Bt|dt.
    
    Parameters
    ----------
    bt_series : pd.Series
        Série temporal de Bt.
    
    Returns
    -------
    float
        Densidade magnética integrada (nT·min).
    """
    bt_clean = bt_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(bt_clean) < 2:
        return 0.0
    
    # Calcula intervalo de tempo em minutos
    time_minutes = (bt_clean.index - bt_clean.index[0]).total_seconds() / 60.0
    
    # Integração trapezoidal
    integrated_density = np.trapz(np.abs(bt_clean.values), time_minutes)
    
    return float(integrated_density)


# Testes da função
if __name__ == "__main__":
    # Cria dados de teste
    np.random.seed(42)
    n_points = 120
    dates = pd.date_range('2023-12-03', periods=n_points, freq='1min')
    
    # Simula Bz com períodos sul
    bz_values = np.random.normal(-5, 5, n_points)
    bz_values[30:60] = np.random.normal(-15, 3, 30)  # Período sul intenso
    bz = pd.Series(bz_values, index=dates)
    
    # Simula Bt correlacionado
    bt = pd.Series(np.random.normal(15, 3, n_points) + np.abs(bz_values)/3, index=dates)
    
    # Calcula métricas
    metrics = compute_alignment_metrics(bz, bt)
    print("Métricas de alinhamento:")
    print(f"  Fração sul: {metrics.south_fraction:.3f}")
    print(f"  Score persistência: {metrics.south_persistence_score:.3f}")
    print(f"  Duração máxima sul: {metrics.max_south_duration:.1f} min")
    print(f"  Bz médio: {metrics.mean_Bz:.1f} nT")
    print(f"  Bt médio: {metrics.mean_Bt:.1f} nT")
    
    # Densidade integrada
    integrated = calculate_integrated_magnetic_density(bt)
    print(f"  Densidade magnética integrada: {integrated:.0f} nT·min")
