"""
Módulo principal para cálculo do índice de acoplamento Sol-Terra.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses_json import dataclass_json

from .geometry import EventGeometry, impact_angle_score, calculate_flank_encounter_angle
from .alignment import compute_alignment_metrics, AlignmentMetrics, calculate_integrated_magnetic_density
from .coherence import compute_coherence_metrics, CoherenceMetrics, calculate_angular_dispersion


@dataclass_json
@dataclass
class CouplingIndexResult:
    """
    Resultado do cálculo do índice de acoplamento.
    
    Attributes
    ----------
    coupling_index : float
        Índice principal de acoplamento (0-1).
    geom_score : float
        Score geométrico (0-1).
    south_fraction : float
        Fração de tempo com Bz sul.
    south_persistence_score : float
        Score de persistência de Bz sul.
    coherence_score : float
        Score de coerência do campo.
    integrated_magnetic_density : float
        Densidade magnética integrada ∫|Bt|dt.
    angular_dispersion : float
        Dispersão angular do fluxo (graus).
    flank_angle : float
        Ângulo de encontro com flanco (graus).
    confidence : float
        Confiança no cálculo (0-1).
    details : dict
        Detalhes completos das métricas.
    """
    coupling_index: float
    geom_score: float
    south_fraction: float
    south_persistence_score: float
    coherence_score: float
    integrated_magnetic_density: float
    angular_dispersion: float
    flank_angle: float
    confidence: float
    details: Dict[str, Any]


class CouplingIndexCalculator:
    """
    Calculadora do índice de acoplamento com configuração flexível.
    """
    
    def __init__(self, 
                 weights: Optional[Dict[str, float]] = None,
                 south_threshold: float = -10.0,
                 min_persistence_minutes: int = 30):
        """
        Inicializa a calculadora.
        
        Parameters
        ----------
        weights : dict, optional
            Pesos para cada componente:
            - geometry: score geométrico
            - alignment: alinhamento magnético
            - coherence: coerência do campo
            - integrated_density: densidade magnética
            - angular_dispersion: dispersão angular
        south_threshold : float
            Limiar para Bz sul (nT).
        min_persistence_minutes : int
            Duração mínima para persistência.
        """
        # Pesos padrão
        default_weights = {
            'geometry': 0.25,
            'alignment': 0.30,
            'coherence': 0.20,
            'integrated_density': 0.15,
            'angular_dispersion': 0.10
        }
        
        self.weights = default_weights.copy()
        if weights:
            self.weights.update(weights)
            
        # Normaliza pesos para soma = 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        self.south_threshold = south_threshold
        self.min_persistence_minutes = min_persistence_minutes
        
    def calculate_confidence(self, 
                           alignment_metrics: AlignmentMetrics,
                           coherence_metrics: CoherenceMetrics,
                           data_points: int) -> float:
        """
        Calcula confiança no resultado baseado na qualidade dos dados.
        
        Returns
        -------
        float
            Score de confiança (0-1).
        """
        confidence_factors = []
        
        # Fator 1: Número de pontos
        if data_points >= 120:  # 2 horas a 1 min
            points_score = 1.0
        elif data_points >= 60:
            points_score = 0.8
        elif data_points >= 30:
            points_score = 0.6
        elif data_points >= 10:
            points_score = 0.4
        else:
            points_score = 0.1
        
        confidence_factors.append(points_score)
        
        # Fator 2: Cobertura de dados
        coverage_score = alignment_metrics.data_coverage
        confidence_factors.append(coverage_score)
        
        # Fator 3: Variabilidade confiável
        variability_score = min(1.0, 10.0 / (coherence_metrics.bz_std + 1e-6))
        confidence_factors.append(variability_score)
        
        # Média ponderada
        confidence = np.mean(confidence_factors)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def compute(self,
                geom: EventGeometry,
                bz_series: pd.Series,
                bt_series: pd.Series,
                by_series: Optional[pd.Series] = None,
                bx_series: Optional[pd.Series] = None) -> CouplingIndexResult:
        """
        Calcula o índice de acoplamento completo.
        
        Parameters
        ----------
        geom : EventGeometry
            Geometria do evento.
        bz_series, bt_series : pd.Series
            Séries temporais de Bz e Bt.
        by_series, bx_series : pd.Series, optional
            Componentes By e Bx para análise 3D.
        
        Returns
        -------
        CouplingIndexResult
            Resultado completo.
        """
        # 1. Score geométrico
        geom_score = impact_angle_score(geom)
        flank_angle = calculate_flank_encounter_angle(geom)
        
        # 2. Métricas de alinhamento
        alignment_metrics = compute_alignment_metrics(
            bz_series, bt_series,
            south_threshold=self.south_threshold,
            min_persistence_minutes=self.min_persistence_minutes
        )
        
        # 3. Densidade magnética integrada
        integrated_density = calculate_integrated_magnetic_density(bt_series)
        # Normaliza: valor típico bom = 1000 nT·min
        density_score = min(1.0, integrated_density / 1000.0)
        
        # 4. Métricas de coerência
        coherence_metrics = compute_coherence_metrics(
            bz_series, bt_series, by_series, bx_series
        )
        
        # 5. Dispersão angular
        angular_dispersion = calculate_angular_dispersion(
            bz_series.values, bt_series.values
        )
        # Quanto menor dispersão, melhor
        dispersion_score = max(0.0, 1.0 - angular_dispersion / 90.0)
        
        # 6. Calcula scores individuais
        alignment_score = 0.6 * alignment_metrics.south_fraction + \
                         0.4 * alignment_metrics.south_persistence_score
        
        # Score de coerência combina várias métricas
        coherence_score = (
            0.3 * (1.0 - coherence_metrics.bz_rel_std) +
            0.3 * max(0.0, coherence_metrics.bz_lag1_autocorr) +
            0.2 * coherence_metrics.spectral_coherence +
            0.2 * (1.0 - min(1.0, coherence_metrics.field_rotation_rate / 10.0))
        )
        
        # 7. Combinação ponderada
        weighted_sum = (
            self.weights['geometry'] * geom_score +
            self.weights['alignment'] * alignment_score +
            self.weights['coherence'] * coherence_score +
            self.weights['integrated_density'] * density_score +
            self.weights['angular_dispersion'] * dispersion_score
        )
        
        # 8. Aplica correção por ângulo de flanco
        flank_correction = max(0.1, 1.0 - flank_angle / 90.0)
        coupling_index = weighted_sum * flank_correction
        
        # 9. Calcula confiança
        confidence = self.calculate_confidence(
            alignment_metrics, coherence_metrics, len(bz_series)
        )
        
        # 10. Prepara detalhes
        details = {
            'weights': self.weights,
            'geometry': {
                'score': geom_score,
                'longitude': geom.longitude_deg,
                'latitude': geom.latitude_deg,
                'width': geom.angular_width_deg,
                'halo': geom.halo_flag,
                'flank_angle': flank_angle
            },
            'alignment': alignment_metrics.to_dict(),
            'coherence': coherence_metrics.to_dict(),
            'integrated_density': {
                'value': integrated_density,
                'score': density_score
            },
            'angular_dispersion': {
                'value': angular_dispersion,
                'score': dispersion_score
            },
            'component_scores': {
                'geometry': geom_score,
                'alignment': alignment_score,
                'coherence': coherence_score,
                'density': density_score,
                'dispersion': dispersion_score
            },
            'weighted_sum': weighted_sum,
            'flank_correction': flank_correction
        }
        
        return CouplingIndexResult(
            coupling_index=float(np.clip(coupling_index, 0.0, 1.0)),
            geom_score=geom_score,
            south_fraction=alignment_metrics.south_fraction,
            south_persistence_score=alignment_metrics.south_persistence_score,
            coherence_score=float(coherence_score),
            integrated_magnetic_density=integrated_density,
            angular_dispersion=angular_dispersion,
            flank_angle=flank_angle,
            confidence=confidence,
            details=details
        )


# Função de conveniência
def compute_coupling_index(
    geom: EventGeometry,
    bz_series: pd.Series,
    bt_series: pd.Series,
    by_series: Optional[pd.Series] = None,
    bx_series: Optional[pd.Series] = None,
    **kwargs
) -> CouplingIndexResult:
    """
    Função de conveniência para cálculo do índice de acoplamento.
    
    Parameters
    ----------
    geom : EventGeometry
        Geometria do evento.
    bz_series, bt_series : pd.Series
        Séries temporais de Bz e Bt.
    by_series, bx_series : pd.Series, optional
        Componentes By e Bx.
    **kwargs : dict
        Argumentos adicionais para CouplingIndexCalculator.
    
    Returns
    -------
    CouplingIndexResult
        Resultado do cálculo.
    """
    calculator = CouplingIndexCalculator(**kwargs)
    return calculator.compute(geom, bz_series, bt_series, by_series, bx_series)


# Testes
if __name__ == "__main__":
    # Dados de teste
    np.random.seed(42)
    n = 180  # 3 horas de dados
    
    # Cria séries temporais
    dates = pd.date_range('2023-12-03 00:00', periods=n, freq='1min')
    
    # Simula diferentes cenários
    t = np.linspace(0, 6*np.pi, n)
    
    # Cenário 1: Evento bem acoplado
    bz_good = 5 * np.sin(t) - 8  # Predominante sul
    bt_good = 20 + 3 * np.sin(t + 0.1)
    
    # Cenário 2: Evento mal acoplado
    bz_bad = np.random.normal(0, 5, n)  # Ruidoso
    bt_bad = 10 + np.random.normal(0, 5, n)
    
    # Geometrias
    geom_good = EventGeometry(
        longitude_deg=-10.0,  # Perto do centro
        latitude_deg=5.0,
        angular_width_deg=220.0,
        halo_flag=True
    )
    
    geom_bad = EventGeometry(
        longitude_deg=70.0,  # Flanco
        latitude_deg=30.0,
        angular_width_deg=60.0,
        halo_flag=False
    )
    
    # Testa cenário bom
    print("=== CENÁRIO BEM ACOPLADO ===")
    bz_series = pd.Series(bz_good, index=dates)
    bt_series = pd.Series(bt_good, index=dates)
    
    result = compute_coupling_index(geom_good, bz_series, bt_series)
    
    print(f"Índice de acoplamento: {result.coupling_index:.3f}")
    print(f"Confiança: {result.confidence:.3f}")
    print(f"Score geométrico: {result.geom_score:.3f}")
    print(f"Fração Bz sul: {result.south_fraction:.3f}")
    print(f"Dispersão angular: {result.angular_dispersion:.1f}°")
    print(f"Ângulo flanco: {result.flank_angle:.1f}°")
    
    print("\n=== CENÁRIO MAL ACOPLADO ===")
    bz_series = pd.Series(bz_bad, index=dates)
    bt_series = pd.Series(bt_bad, index=dates)
    
    result = compute_coupling_index(geom_bad, bz_series, bt_series)
    
    print(f"Índice de acoplamento: {result.coupling_index:.3f}")
    print(f"Confiança: {result.confidence:.3f}")
    print(f"Score geométrico: {result.geom_score:.3f}")
    print(f"Fração Bz sul: {result.south_fraction:.3f}")
    print(f"Dispersão angular: {result.angular_dispersion:.1f}°")
    print(f"Ângulo flanco: {result.flank_angle:.1f}°")
