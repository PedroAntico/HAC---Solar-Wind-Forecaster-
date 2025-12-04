"""
Módulo para previsão operacional baseada no índice de acoplamento.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from datetime import datetime, timedelta
import warnings

from .coupling_index import CouplingIndexResult, compute_coupling_index
from .geometry import EventGeometry


@dataclass_json
@dataclass
class ForecastResult:
    """
    Resultado de previsão operacional.
    
    Attributes
    ----------
    probability_bz_south : float
        Probabilidade de Bz sul sustentado (0-1).
    expected_min_bz : float
        Bz mínimo esperado (nT).
    expected_duration : float
        Duração esperada de Bz sul (minutos).
    severity_level : str
        Nível de severidade: 'low', 'moderate', 'high', 'extreme'.
    recommended_actions : List[str]
        Ações recomendadas.
    confidence : float
        Confiança na previsão (0-1).
    timestamp : datetime
        Timestamp da previsão.
    """
    probability_bz_south: float
    expected_min_bz: float
    expected_duration: float
    severity_level: str
    recommended_actions: List[str]
    confidence: float
    timestamp: datetime


class CouplingForecaster:
    """
    Sistema de previsão baseado no índice de acoplamento.
    """
    
    # Limiares calibrados empiricamente
    SEVERITY_THRESHOLDS = {
        'low': 0.3,
        'moderate': 0.5,
        'high': 0.7,
        'extreme': 0.85
    }
    
    # Mapeamento severidade → ações
    ACTION_MAPPING = {
        'low': [
            "Monitorar parâmetros",
            "Atualizar previsões a cada 6h"
        ],
        'moderate': [
            "Alerta para operadores de satélite",
            "Monitorar campo magnético em tempo real",
            "Preparar atualizações de modelos"
        ],
        'high': [
            "Alerta de tempestade geomagnética",
            "Ativar procedimentos de mitigação",
            "Monitorar contínuo (1h)",
            "Comunicar operadores críticos"
        ],
        'extreme': [
            "ALERTA VERMELHO: Tempestade geomagnética severa",
            "Ativar todos os procedimentos de emergência",
            "Monitorar em tempo real (15min)",
            "Comunicar todas as partes interessadas",
            "Preparar para possíveis blecautes"
        ]
    }
    
    def __init__(self, calibration_data: Optional[pd.DataFrame] = None):
        """
        Inicializa o sistema de previsão.
        
        Parameters
        ----------
        calibration_data : pd.DataFrame, optional
            Dados históricos para calibração.
        """
        self.calibration_data = calibration_data
        self.calibrated = calibration_data is not None
        
        if self.calibrated:
            self._calibrate_thresholds()
    
    def _calibrate_thresholds(self):
        """Calibra limiares com dados históricos."""
        # Implementação simplificada
        warnings.warn("Calibração completa requer mais dados históricos")
    
    def predict_from_coupling_index(self, 
                                   coupling_result: CouplingIndexResult,
                                   solar_wind_speed: Optional[float] = None,
                                   solar_wind_density: Optional[float] = None) -> ForecastResult:
        """
        Gera previsão a partir do índice de acoplamento.
        
        Parameters
        ----------
        coupling_result : CouplingIndexResult
            Resultado do cálculo do índice.
        solar_wind_speed : float, optional
            Velocidade do vento solar (km/s).
        solar_wind_density : float, optional
            Densidade do vento solar (partículas/cm³).
        
        Returns
        -------
        ForecastResult
            Previsão operacional.
        """
        ci = coupling_result.coupling_index
        
        # 1. Probabilidade de Bz sul sustentado
        # Baseado no índice e na persistência
        base_prob = ci
        persistence_factor = coupling_result.south_persistence_score
        
        # Correção por velocidade do vento solar
        if solar_wind_speed is not None:
            # Velocidades típicas ~400 km/s, alta >600 km/s
            speed_factor = min(1.0, solar_wind_speed / 800.0)
            base_prob *= speed_factor
        
        probability = min(0.95, base_prob * (0.7 + 0.3 * persistence_factor))
        
        # 2. Bz mínimo esperado
        # Relação empírica: índice mais alto → Bz mais negativo
        expected_min_bz = -5 - 25 * ci
        
        # 3. Duração esperada
        base_duration = 30 + 180 * ci  # 30 min a 3.5 horas
        expected_duration = base_duration * (0.5 + 0.5 * persistence_factor)
        
        # 4. Determina nível de severidade
        severity_level = 'low'
        for level, threshold in self.SEVERITY_THRESHOLDS.items():
            if ci >= threshold:
                severity_level = level
        
        # 5. Ações recomendadas
        recommended_actions = self.ACTION_MAPPING.get(severity_level, [])
        
        # Adiciona ações específicas baseadas em métricas
        if coupling_result.angular_dispersion < 30:
            recommended_actions.append("Campo bem organizado - risco concentrado")
        if coupling_result.integrated_magnetic_density > 1500:
            recommended_actions.append("Alta densidade magnética - energia total elevada")
        
        # 6. Confiança na previsão
        forecast_confidence = coupling_result.confidence * 0.8 + 0.2
        
        return ForecastResult(
            probability_bz_south=float(np.clip(probability, 0.0, 1.0)),
            expected_min_bz=float(expected_min_bz),
            expected_duration=float(expected_duration),
            severity_level=severity_level,
            recommended_actions=recommended_actions,
            confidence=float(np.clip(forecast_confidence, 0.0, 1.0)),
            timestamp=datetime.now()
        )
    
    def predict_from_raw_data(self,
                             geom: EventGeometry,
                             bz_series: pd.Series,
                             bt_series: pd.Series,
                             solar_wind_speed: Optional[float] = None,
                             **kwargs) -> Tuple[CouplingIndexResult, ForecastResult]:
        """
        Previsão completa a partir de dados brutos.
        
        Returns
        -------
        tuple
            (resultado do índice, previsão)
        """
        # Calcula índice de acoplamento
        coupling_result = compute_coupling_index(geom, bz_series, bt_series, **kwargs)
        
        # Gera previsão
        forecast = self.predict_from_coupling_index(
            coupling_result, solar_wind_speed
        )
        
        return coupling_result, forecast
    
    def generate_alert_message(self, forecast: ForecastResult) -> str:
        """
        Gera mensagem de alerta formatada.
        
        Returns
        -------
        str
            Mensagem de alerta.
        """
        timestamp = forecast.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        
        message = f"""
        ⚠️ ALERTA DE ACOPLAMENTO MAGNÉTICO ⚠️
        Hora: {timestamp}
        Severidade: {forecast.severity_level.upper()}
        
        PROBABILIDADE Bz SUL: {forecast.probability_bz_south:.0%}
        Bz MÍNIMO ESPERADO: {forecast.expected_min_bz:.0f} nT
        DURAÇÃO ESPERADA: {forecast.expected_duration:.0f} min
        
        AÇÕES RECOMENDADAS:
        {chr(10).join(f'  • {action}' for action in forecast.recommended_actions)}
        
        Confiança da previsão: {forecast.confidence:.0%}
        """
        
        return message.strip()


# Funções auxiliares
def calculate_optimal_forecast_window(bz_series: pd.Series,
                                     min_points: int = 30,
                                     max_hours: int = 6) -> pd.DatetimeIndex:
    """
    Determina janela ótima para previsão.
    
    Parameters
    ----------
    bz_series : pd.Series
        Série temporal de Bz.
    min_points : int
        Pontos mínimos necessários.
    max_hours : int
        Janela máxima em horas.
    
    Returns
    -------
    pd.DatetimeIndex
        Janela temporal recomendada.
    """
    if len(bz_series) < min_points:
        raise ValueError(f"Dados insuficientes: {len(bz_series)} < {min_points}")
    
    # Janela de análise: últimos dados disponíveis
    end_time = bz_series.index[-1]
    start_time = end_time - timedelta(hours=max_hours)
    
    # Filtra para garantir dados contínuos
    window = bz_series.loc[start_time:end_time]
    
    if len(window) < min_points:
        # Usa todos os dados disponíveis
        window = bz_series
    
    return window.index


# Testes
if __name__ == "__main__":
    # Dados de teste
    np.random.seed(42)
    n = 120  # 2 horas
    
    dates = pd.date_range('2023-12-03 00:00', periods=n, freq='1min')
    t = np.linspace(0, 4*np.pi, n)
    
    # Evento de teste
    bz = 8 * np.sin(t) - 12  # Forte componente sul
    bt = 20 + 5 * np.sin(t + 0.2)
    
    bz_series = pd.Series(bz, index=dates)
    bt_series = pd.Series(bt, index=dates)
    
    geom = EventGeometry(
        longitude_deg=-15.0,
        latitude_deg=8.0,
        angular_width_deg=180.0,
        halo_flag=True
    )
    
    # Cria previsor
    forecaster = CouplingForecaster()
    
    # Gera previsão
    coupling_result, forecast = forecaster.predict_from_raw_data(
        geom, bz_series, bt_series,
        solar_wind_speed=550.0
    )
    
    print("=== PREVISÃO DE ACOPLAMENTO ===")
    print(f"Índice de acoplamento: {coupling_result.coupling_index:.3f}")
    print(f"Probabilidade Bz sul: {forecast.probability_bz_south:.1%}")
    print(f"Bz mínimo esperado: {forecast.expected_min_bz:.0f} nT")
    print(f"Duração esperada: {forecast.expected_duration:.0f} min")
    print(f"Nível severidade: {forecast.severity_level}")
    print(f"Confiança: {forecast.confidence:.1%}")
    
    print("\n=== MENSAGEM DE ALERTA ===")
    print(forecaster.generate_alert_message(forecast))
