"""
Testes unitários para o módulo de acoplamento geomagnético.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Adiciona o diretório pai ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hac.coupling.geometry import (
    EventGeometry, impact_angle_score, 
    calculate_flank_encounter_angle
)
from hac.coupling.alignment import (
    compute_alignment_metrics, 
    calculate_integrated_magnetic_density
)
from hac.coupling.coherence import (
    compute_coherence_metrics,
    calculate_angular_dispersion
)
from hac.coupling.coupling_index import (
    compute_coupling_index, CouplingIndexCalculator
)
from hac.coupling.forecast import CouplingForecaster


class TestGeometry:
    """Testes para o módulo de geometria."""
    
    def test_event_geometry_validation(self):
        """Testa validação da geometria do evento."""
        # Geometria válida
        geom = EventGeometry(
            longitude_deg=0.0,
            latitude_deg=0.0,
            angular_width_deg=60.0
        )
        assert geom.longitude_deg == 0.0
        
        # Geometria inválida deve lançar erro
        with pytest.raises(ValueError):
            EventGeometry(
                longitude_deg=200.0,  # Inválido
                latitude_deg=0.0,
                angular_width_deg=60.0
            )
    
    def test_impact_angle_score(self):
        """Testa cálculo do score de impacto."""
        # Evento central
        geom_central = EventGeometry(
            longitude_deg=0.0,
            latitude_deg=0.0,
            angular_width_deg=180.0,
            halo_flag=True
        )
        score_central = impact_angle_score(geom_central)
        assert 0.8 <= score_central <= 1.0
        
        # Evento de flanco
        geom_flank = EventGeometry(
            longitude_deg=80.0,
            latitude_deg=30.0,
            angular_width_deg=30.0,
            halo_flag=False
        )
        score_flank = impact_angle_score(geom_flank)
        assert 0.0 <= score_flank <= 0.5
        
        # Verifica que score central > score flanco
        assert score_central > score_flank
    
    def test_flank_encounter_angle(self):
        """Testa cálculo do ângulo de encontro com flanco."""
        geom = EventGeometry(
            longitude_deg=45.0,
            latitude_deg=0.0,
            angular_width_deg=60.0
        )
        angle = calculate_flank_encounter_angle(geom, earth_position=0.0)
        assert 0.0 <= angle <= 90.0
        # Para longitude 45° e largura 60°, o ângulo efetivo é ~15°
        assert abs(angle - 15.0) < 5.0


class TestAlignment:
    """Testes para o módulo de alinhamento."""
    
    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo para testes."""
        np.random.seed(42)
        n = 120  # 2 horas
        
        dates = pd.date_range('2023-12-03', periods=n, freq='1min')
        
        # Bz com período sul intenso
        bz = np.random.normal(-5, 3, n)
        bz[30:90] = np.random.normal(-15, 2, 60)  # 1 hora de Bz sul
        
        # Bt correlacionado
        bt = np.random.normal(15, 2, n) + np.abs(bz)/2
        
        return pd.Series(bz, index=dates), pd.Series(bt, index=dates)
    
    def test_compute_alignment_metrics(self, sample_data):
        """Testa cálculo de métricas de alinhamento."""
        bz, bt = sample_data
        
        metrics = compute_alignment_metrics(bz, bt)
        
        # Verificações básicas
        assert 0.0 <= metrics.south_fraction <= 1.0
        assert 0.0 <= metrics.south_persistence_score <= 1.0
        assert metrics.n_points == len(bz)
        assert metrics.data_coverage > 0.5
        
        # Para nossos dados simulados, esperamos fração sul alta
        assert metrics.south_fraction > 0.4
    
    def test_integrated_magnetic_density(self, sample_data):
        """Testa cálculo da densidade magnética integrada."""
        _, bt = sample_data
        
        integrated = calculate_integrated_magnetic_density(bt)
        
        # Valor deve ser positivo
        assert integrated > 0.0
        
        # Para Bt ~15 nT por 120 min, valor esperado ~1800 nT·min
        assert 1000 < integrated < 3000
    
    def test_persistence_detection(self):
        """Testa detecção de persistência de Bz sul."""
        # Cria série com bloco sul de 45 minutos
        n = 180
        dates = pd.date_range('2023-12-03', periods=n, freq='1min')
        
        bz = np.ones(n) * 5.0  # Inicia positivo
        bz[60:105] = -15.0  # 45 minutos de Bz sul
        bt = np.ones(n) * 10.0
        
        bz_series = pd.Series(bz, index=dates)
        bt_series = pd.Series(bt, index=dates)
        
        metrics = compute_alignment_metrics(bz_series, bt_series)
        
        # Deve detectar persistência
        assert metrics.max_south_duration == 45.0
        assert metrics.south_persistence_score > 0.5


class TestCoherence:
    """Testes para o módulo de coerência."""
    
    @pytest.fixture
    def sample_coherent_data(self):
        """Cria dados coerentes para teste."""
        np.random.seed(42)
        n = 200
        
        t = np.linspace(0, 4*np.pi, n)
        
        # Dados coerentes (senoidal com pouco ruído)
        bz = 10 * np.sin(t) + np.random.normal(0, 1, n)
        bt = 15 + 3 * np.sin(t + 0.1) + np.random.normal(0, 0.5, n)
        by = 5 * np.cos(t) + np.random.normal(0, 1, n)
        bx = 8 * np.sin(0.5*t) + np.random.normal(0, 1, n)
        
        dates = pd.date_range('2023-12-03', periods=n, freq='1min')
        
        return (
            pd.Series(bz, index=dates),
            pd.Series(bt, index=dates),
            pd.Series(by, index=dates),
            pd.Series(bx, index=dates)
        )
    
    def test_coherence_metrics(self, sample_coherent_data):
        """Testa cálculo de métricas de coerência."""
        bz, bt, by, bx = sample_coherent_data
        
        metrics = compute_coherence_metrics(bz, bt, by, bx)
        
        # Verificações
        assert metrics.bt_std > 0
        assert -1 <= metrics.bz_lag1_autocorr <= 1
        assert 0 <= metrics.spectral_coherence <= 1
        assert 0 <= metrics.trend_stability <= 1
        
        # Para dados coerentes, autocorrelação deve ser alta
        assert metrics.bz_lag1_autocorr > 0.5
    
    def test_angular_dispersion(self):
        """Testa cálculo da dispersão angular."""
        # Campo bem organizado (Bz sempre negativo)
        n = 100
        bz_organized = np.full(n, -10.0)
        bt_organized = np.full(n, 15.0)
        
        disp_organized = calculate_angular_dispersion(bz_organized, bt_organized)
        assert disp_organized < 10.0  # Dispersão baixa
        
        # Campo desorganizado
        bz_random = np.random.uniform(-20, 20, n)
        bt_random = np.random.uniform(5, 25, n)
        
        disp_random = calculate_angular_dispersion(bz_random, bt_random)
        assert disp_random > 30.0  # Dispersão alta


class TestCouplingIndex:
    """Testes para o índice de acoplamento."""
    
    @pytest.fixture
    def test_scenario(self):
        """Cria cenário de teste completo."""
        np.random.seed(42)
        n = 180
        
        dates = pd.date_range('2023-12-03', periods=n, freq='1min')
        t = np.linspace(0, 6*np.pi, n)
        
        # Dados para evento bem acoplado
        bz_good = 8 * np.sin(t) - 12
        bt_good = 20 + 4 * np.sin(t + 0.1)
        
        geom_good = EventGeometry(
            longitude_deg=-10.0,
            latitude_deg=5.0,
            angular_width_deg=200.0,
            halo_flag=True
        )
        
        return (
            geom_good,
            pd.Series(bz_good, index=dates),
            pd.Series(bt_good, index=dates)
        )
    
    def test_coupling_index_calculation(self, test_scenario):
        """Testa cálculo do índice de acoplamento."""
        geom, bz, bt = test_scenario
        
        result = compute_coupling_index(geom, bz, bt)
        
        # Verificações básicas
        assert 0.0 <= result.coupling_index <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert 'geometry' in result.details
        assert 'alignment' in result.details
        
        # Para cenário bom, índice deve ser alto
        assert result.coupling_index > 0.5
    
    def test_coupling_index_consistency(self):
        """Testa consistência do índice."""
        # Cria calculadora customizada
        calculator = CouplingIndexCalculator(
            weights={
                'geometry': 0.5,
                'alignment': 0.5,
                'coherence': 0.0,
                'integrated_density': 0.0,
                'angular_dispersion': 0.0
            }
        )
        
        # Evento frontal com Bz sul
        geom = EventGeometry(0, 0, 180, True)
        n = 100
        dates = pd.date_range('2023-12-03', periods=n, freq='1min')
        bz = pd.Series(np.full(n, -15.0), index=dates)
        bt = pd.Series(np.full(n, 20.0), index=dates)
        
        result = calculator.compute(geom, bz, bt)
        
        # Deve ser muito alto
        assert result.coupling_index > 0.8
    
    def test_coupling_components(self, test_scenario):
        """Testa que todos os componentes afetam o índice."""
        geom, bz, bt = test_scenario
        
        # Calcula índice completo
        result_full = compute_coupling_index(geom, bz, bt)
        
        # Testa que índices individuais estão presentes
        assert 'geometry' in result_full.details['component_scores']
        assert 'alignment' in result_full.details['component_scores']
        assert 'coherence' in result_full.details['component_scores']


class TestForecast:
    """Testes para o módulo de previsão."""
    
    @pytest.fixture
    def sample_forecast_input(self):
        """Cria entrada de exemplo para previsão."""
        np.random.seed(42)
        n = 120
        
        dates = pd.date_range('2023-12-03', periods=n, freq='1min')
        t = np.linspace(0, 4*np.pi, n)
        
        bz = 10 * np.sin(t) - 8
        bt = 18 + 3 * np.sin(t + 0.1)
        
        geom = EventGeometry(
            longitude_deg=-15.0,
            latitude_deg=10.0,
            angular_width_deg=180.0,
            halo_flag=True
        )
        
        return geom, pd.Series(bz, index=dates), pd.Series(bt, index=dates)
    
    def test_forecast_generation(self, sample_forecast_input):
        """Testa geração de previsão."""
        geom, bz, bt = sample_forecast_input
        
        forecaster = CouplingForecaster()
        coupling_result, forecast = forecaster.predict_from_raw_data(
            geom, bz, bt, solar_wind_speed=500.0
        )
        
        # Verificações
        assert 0.0 <= forecast.probability_bz_south <= 1.0
        assert forecast.expected_min_bz < 0  # Esperado negativo
        assert forecast.expected_duration > 0
        assert forecast.severity_level in ['low', 'moderate', 'high', 'extreme']
        assert len(forecast.recommended_actions) > 0
        assert 0.0 <= forecast.confidence <= 1.0
    
    def test_severity_classification(self):
        """Testa classificação de severidade."""
        forecaster = CouplingForecaster()
        
        # Testa diferentes níveis
        test_cases = [
            (0.2, 'low'),
            (0.4, 'moderate'),
            (0.75, 'high'),
            (0.9, 'extreme')
        ]
        
        for coupling_value, expected_level in test_cases:
            # Cria resultado simulado
            class MockResult:
                coupling_index = coupling_value
                south_persistence_score = 0.5
                confidence = 0.8
                south_fraction = 0.5
                angular_dispersion = 30.0
                integrated_magnetic_density = 1000.0
                flank_angle = 20.0
                geom_score = 0.7
                coherence_score = 0.6
                details = {}
            
            forecast = forecaster.predict_from_coupling_index(MockResult())
            assert forecast.severity_level == expected_level
    
    def test_alert_message_generation(self, sample_forecast_input):
        """Testa geração de mensagem de alerta."""
        geom, bz, bt = sample_forecast_input
        
        forecaster = CouplingForecaster()
        _, forecast = forecaster.predict_from_raw_data(geom, bz, bt)
        
        message = forecaster.generate_alert_message(forecast)
        
        # Verifica elementos essenciais na mensagem
        assert 'ALERTA' in message
        assert forecast.severity_level.upper() in message
        assert str(int(forecast.probability_bz_south * 100)) in message
        assert 'nT' in message  # Unidade de Bz
        assert 'min' in message  # Unidade de duração


class TestIntegration:
    """Testes de integração do sistema completo."""
    
    def test_end_to_end_pipeline(self):
        """Testa pipeline completo de dados à previsão."""
        np.random.seed(42)
        
        # 1. Cria dados simulados
        n = 180
        dates = pd.date_range('2023-12-03 00:00', periods=n, freq='1min')
        t = np.linspace(0, 6*np.pi, n)
        
        bz = 12 * np.sin(t) - 10  # Forte componente sul
        bt = 22 + 5 * np.sin(t + 0.15)
        by = 8 * np.cos(t)
        bx = 10 * np.sin(0.3*t)
        
        bz_series = pd.Series(bz, index=dates)
        bt_series = pd.Series(bt, index=dates)
        by_series = pd.Series(by, index=dates)
        bx_series = pd.Series(bx, index=dates)
        
        # 2. Define geometria do evento
        geom = EventGeometry(
            longitude_deg=-5.0,
            latitude_deg=8.0,
            angular_width_deg=240.0,
            halo_flag=True,
            source_region="AR 13579"
        )
        
        # 3. Calcula índice de acoplamento
        calculator = CouplingIndexCalculator()
        coupling_result = calculator.compute(
            geom, bz_series, bt_series, by_series, bx_series
        )
        
        # 4. Gera previsão
        forecaster = CouplingForecaster()
        forecast = forecaster.predict_from_coupling_index(
            coupling_result, solar_wind_speed=600.0
        )
        
        # 5. Verificações finais
        assert coupling_result.coupling_index > 0.0
        assert forecast.probability_bz_south > 0.0
        assert len(forecast.recommended_actions) > 0
        
        print(f"\nPipeline completo executado com sucesso:")
        print(f"  Índice: {coupling_result.coupling_index:.3f}")
        print(f"  Probabilidade: {forecast.probability_bz_south:.1%}")
        print(f"  Severidade: {forecast.severity_level}")
        print(f"  Confiança: {forecast.confidence:.1%}")


if __name__ == "__main__":
    # Executa testes
    pytest.main([__file__, "-v", "--tb=short"])
