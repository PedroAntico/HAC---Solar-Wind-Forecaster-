#!/usr/bin/env python3
"""
HAC v6 PREDICTOR - VERSÃO DEFINITIVA PARA PRODUÇÃO
Predictor robusto com dados reais, sem fallbacks sintéticos.
"""

import os
import re
import json
import logging
import gc
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Configuração para produção
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError as e:
    logger.error(f"TensorFlow não disponível: {e}")
    TF_AVAILABLE = False

try:
    from hac_v6_config import HACConfig
except ImportError:
    import sys
    sys.path.append('.')
    from hac_v6_config import HACConfig


@dataclass
class ForecastResult:
    """Resultado de previsão com validação"""
    horizon: int
    values: Dict[str, float]
    valid: bool
    alerts: List[str]
    warnings: List[str]


class HACv6Predictor:
    """Predictor definitivo com dados reais"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = HACConfig(config_path)
        paths = self.config.get("paths")
        self.model_dir = paths["model_dir"]
        self.default_targets = self.config.get("targets")["primary"]
        
        # Estado
        self.models: Dict[int, Any] = {}
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.scalers_y: Dict[int, Any] = {}
        
        logger.info("🧠 Inicializando HACv6Predictor (versão definitiva)")
        
    def load_models(self) -> bool:
        """Carrega todos os modelos e scalers disponíveis"""
        if not os.path.isdir(self.model_dir):
            logger.error(f"Diretório de modelos não encontrado: {self.model_dir}")
            return False
        
        pattern = re.compile(r"^.+_h(\d+)_\d+")
        loaded = []
        
        for folder in sorted(os.listdir(self.model_dir)):
            folder_path = os.path.join(self.model_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Extrai horizonte
            match = pattern.match(folder)
            if not match:
                continue
            
            horizon = int(match.group(1))
            
            # Caminhos dos arquivos
            model_path = os.path.join(folder_path, "model.keras")
            meta_path = os.path.join(folder_path, "metadata.json")
            scaler_y_path = os.path.join(folder_path, "scaler_Y.pkl")
            
            # Verifica se arquivos existem
            if not os.path.exists(model_path):
                logger.warning(f"Modelo não encontrado: {model_path}")
                continue
            
            if not os.path.exists(scaler_y_path):
                logger.error(f"Scaler Y não encontrado para H{horizon}: {scaler_y_path}")
                continue
            
            try:
                # Carrega modelo
                gc.collect()
                model = load_model(model_path)
                
                # Carrega metadata
                metadata = {}
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                
                # Carrega scaler Y
                import joblib
                scaler_y = joblib.load(scaler_y_path)
                
                # Valida scaler Y
                if not hasattr(scaler_y, 'mean_') or not hasattr(scaler_y, 'scale_'):
                    logger.error(f"Scaler Y inválido para H{horizon}")
                    continue
                
                # Armazena
                self.models[horizon] = model
                self.metadata[horizon] = metadata
                self.scalers_y[horizon] = scaler_y
                loaded.append(horizon)
                
                logger.info(f"✅ H{horizon} carregado: {folder}")
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar H{horizon}: {e}")
                continue
        
        if loaded:
            logger.info(f"🎯 {len(loaded)} modelos carregados: {sorted(loaded)}")
            return True
        else:
            logger.error("⚠️ Nenhum modelo foi carregado")
            return False
    
    def get_available_horizons(self) -> List[int]:
        """Retorna horizontes disponíveis"""
        return sorted(self.models.keys())
    
    def _validate_physical_ranges(self, values: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Valida se os valores estão em ranges físicos realistas"""
        warnings = []
        valid = True
        
        # Ranges baseados em dados OMNI reais
        physical_ranges = {
            "speed": (200, 2000),      # km/s
            "density": (0.1, 100),     # cm⁻³
            "bz_gsm": (-50, 50),       # nT
            "bz": (-50, 50),           # nT (alias)
            "bt": (0, 100)             # nT
        }
        
        for param, value in values.items():
            if param in physical_ranges:
                min_val, max_val = physical_ranges[param]
                
                if value < min_val or value > max_val:
                    warnings.append(f"{param} fora do range: {value:.1f} (esperado: {min_val}-{max_val})")
                    valid = False
        
        return valid, warnings
    
    def _calculate_alerts(self, values: Dict[str, float]) -> List[str]:
        """Calcula alertas baseados nos thresholds do config"""
        alerts = []
        thresholds = self.config.get("alerts", {}).get("thresholds", {})
        
        # Valores padrão se não especificados
        speed_high = thresholds.get("speed_high", 600)
        speed_low = thresholds.get("speed_low", 300)
        density_high = thresholds.get("density_high", 20)
        bz_extreme = thresholds.get("bz_extreme", 10)
        
        speed = values.get("speed", 0)
        density = values.get("density", 0)
        bz = values.get("bz_gsm", values.get("bz", 0))
        
        if speed > speed_high:
            alerts.append("HIGH_SPEED")
        if speed < speed_low:
            alerts.append("LOW_SPEED")
        if density > density_high:
            alerts.append("HIGH_DENSITY")
        if abs(bz) > bz_extreme:
            alerts.append("EXTREME_BZ")
        
        return alerts
    
    def predict(self, X_window: np.ndarray, horizon: int) -> ForecastResult:
        """
        Faz previsão para um horizonte específico
        
        Args:
            X_window: Array (lookback, n_features) ou (1, lookback, n_features)
            horizon: Horizonte de previsão em horas
            
        Returns:
            ForecastResult: Resultado completo da previsão
        """
        if horizon not in self.models:
            available = self.get_available_horizons()
            raise ValueError(f"Modelo H{horizon} não disponível. Horizontes: {available}")
        
        if horizon not in self.scalers_y:
            raise ValueError(f"Scaler Y não disponível para H{horizon}")
        
        try:
            # Prepara entrada
            X = np.asarray(X_window, dtype=np.float32)
            if X.ndim == 2:
                X = np.expand_dims(X, axis=0)
            
            if X.ndim != 3:
                raise ValueError(f"Shape inválido: {X.shape}. Esperado (lookback, n_features) ou (1, lookback, n_features)")
            
            # Previsão
            model = self.models[horizon]
            y_scaled = model.predict(X, verbose=0)[0]
            
            # Aplica scaler Y
            scaler_y = self.scalers_y[horizon]
            y_real = scaler_y.inverse_transform(y_scaled.reshape(1, -1))[0]
            
            # Mapeia para targets
            targets = self.metadata.get(horizon, {}).get("targets", self.default_targets)
            values = {targets[i]: float(y_real[i]) for i in range(len(targets))}
            
            # Validação e alertas
            valid, warnings = self._validate_physical_ranges(values)
            alerts = self._calculate_alerts(values)
            
            return ForecastResult(
                horizon=horizon,
                values=values,
                valid=valid,
                alerts=alerts,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"❌ Erro na previsão H{horizon}: {e}")
            raise
    
    def predict_from_dataframe(self, df: pd.DataFrame, horizon: int, lookback: int = None) -> ForecastResult:
        """Previsão a partir de DataFrame"""
        if lookback is None:
            lookback = self.metadata.get(horizon, {}).get(
                "lookback",
                self.config.get("training")["main_lookback"]
            )
        
        if len(df) < lookback:
            raise ValueError(f"DataFrame muito curto: {len(df)} linhas, precisa {lookback}")
        
        # Usa as últimas 'lookback' linhas
        window = df.iloc[-lookback:].values
        return self.predict(window, horizon)
    
    def predict_all_horizons(self, X_window: np.ndarray) -> Dict[int, ForecastResult]:
        """Previsão para todos os horizontes disponíveis"""
        results = {}
        
        for horizon in self.get_available_horizons():
            try:
                result = self.predict(X_window, horizon)
                results[horizon] = result
            except Exception as e:
                logger.error(f"❌ Falha na previsão H{horizon}: {e}")
        
        return results
    
    def get_model_info(self, horizon: int) -> Dict[str, Any]:
        """Retorna informações do modelo"""
        if horizon not in self.models:
            return {}
        
        return {
            "horizon": horizon,
            "metadata": self.metadata.get(horizon, {}),
            "scaler_y_means": self.scalers_y[horizon].mean_.tolist() if horizon in self.scalers_y else None,
            "scaler_y_scales": self.scalers_y[horizon].scale_.tolist() if horizon in self.scalers_y else None
        }


# Interface simplificada para scripts
_predictor_instance = None

def get_predictor(config_path: str = "config.yaml") -> HACv6Predictor:
    """Retorna instância singleton do predictor"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HACv6Predictor(config_path)
        if not _predictor_instance.load_models():
            raise RuntimeError("Falha ao carregar modelos")
    return _predictor_instance


if __name__ == "__main__":
    # Teste básico
    print("🧪 Teste do HACv6Predictor (versão definitiva)")
    
    try:
        predictor = get_predictor()
        horizons = predictor.get_available_horizons()
        print(f"✅ Horizontes disponíveis: {horizons}")
        
        if horizons:
            print("🎯 Predictor pronto para produção!")
        else:
            print("⚠️ Nenhum horizonte disponível")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
