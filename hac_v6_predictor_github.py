#!/usr/bin/env python3
"""
hac_v6_predictor_github.py - PREDICTOR OTIMIZADO PARA GITHUB FREE

Características principais:
- Baixo consumo de memória (float32, garbage collection)
- Fallbacks automáticos para scalers ausentes
- Validação física das previsões
- Logging detalhado para debug
- Singleton pattern para evitar múltiplas instâncias
"""

import os
import re
import json
import logging
import gc
import warnings
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Configuração otimizada para GitHub Free
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia TF
warnings.filterwarnings('ignore')

# Logging otimizado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TensorFlow não disponível: {e}")
    TF_AVAILABLE = False

try:
    from hac_v6_config import HACConfig
except ImportError:
    # Fallback para import local
    import sys
    sys.path.append('.')
    from hac_v6_config import HACConfig


@dataclass
class PredictionResult:
    """Resultado estruturado de previsão"""
    values: Dict[str, float]
    horizon: int
    valid: bool
    warnings: List[str]
    physical_checks: Dict[str, bool]


class HACv6PredictorGithub:
    """Predictor otimizado para GitHub Free com fallbacks robustos"""
    
    # Estatísticas físicas realistas do vento solar
    PHYSICAL_RANGES = {
        "speed": {"min": 200, "max": 2000, "typical": 450},
        "density": {"min": 0.1, "max": 100, "typical": 7},
        "bz_gsm": {"min": -50, "max": 50, "typical": 0},
        "bt": {"min": 0, "max": 100, "typical": 5}
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = HACConfig(config_path)
        paths = self.config.get("paths")
        self.model_dir = paths["model_dir"]
        self.default_targets = self.config.get("targets")["primary"]
        
        # Estado interno otimizado
        self.models: Dict[int, Any] = {}
        self.meta: Dict[int, Dict[str, Any]] = {}
        self.scalers_y: Dict[int, Any] = {}
        self._is_loaded = False
        self._load_attempts = 0
        
        logger.info("🧠 HACv6PredictorGithub - Inicializando (GitHub Free Optimized)")
        
    def _load_model_memory_safe(self, model_path: str) -> Optional[Any]:
        """Carrega modelo com controle de memória"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow não está disponível")
            
        try:
            # Limpeza agressiva de memória antes do carregamento
            gc.collect()
            
            # Configurações para economizar memória
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')  # Desabilita GPU no Codespaces
            
            model = load_model(model_path)
            logger.info(f"✅ Modelo carregado: {os.path.basename(model_path)}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar {model_path}: {e}")
            return None

    def _create_fallback_scaler(self, horizon: int) -> Any:
        """Cria scaler de fallback baseado em estatísticas físicas realistas"""
        from sklearn.preprocessing import StandardScaler
        
        logger.warning(f"🔄 Criando scaler de fallback para H{horizon}")
        
        scaler = StandardScaler()
        n_samples = 5000  # Reduzido para economizar memória
        
        # Gera dados baseados em ranges físicos realistas
        speed_data = np.random.normal(450, 150, n_samples)  # 300-600 km/s típico
        bz_data = np.random.normal(0, 6, n_samples)         # -10 a +10 nT típico
        density_data = np.random.normal(7, 4, n_samples)    # 3-15 cm⁻³ típico
        
        synthetic_data = np.column_stack([speed_data, bz_data, density_data])
        scaler.fit(synthetic_data)
        
        return scaler

    def _load_all_memory_optimized(self) -> None:
        """Carrega modelos de forma otimizada para memória"""
        if not os.path.isdir(self.model_dir):
            logger.warning(f"📁 Diretório não encontrado: {self.model_dir}")
            os.makedirs(self.model_dir, exist_ok=True)
            return

        pattern = re.compile(r"^(?P<type>\w+)_h(?P<h>\d+)_")
        loaded_count = 0

        # Ordena por horizonte para carregar do menor para o maior
        folders = sorted(os.listdir(self.model_dir), 
                        key=lambda x: int(pattern.match(x).group("h")) if pattern.match(x) else 999)
        
        for folder in folders:
            full_path = os.path.join(self.model_dir, folder)
            if not os.path.isdir(full_path):
                continue

            match = pattern.match(folder)
            if not match:
                continue

            horizon = int(match.group("h"))
            model_path = os.path.join(full_path, "model.keras")
            meta_path = os.path.join(full_path, "metadata.json")
            scaler_y_path = os.path.join(full_path, "scaler_Y.pkl")

            if not os.path.exists(model_path):
                logger.warning(f"📦 Modelo não encontrado: {model_path}")
                continue

            # Carrega modelo com controle de memória
            model = self._load_model_memory_safe(model_path)
            if model is None:
                continue

            # Carrega metadata
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                except Exception as e:
                    logger.warning(f"📄 Erro ao carregar metadata {meta_path}: {e}")

            # Carrega ou cria scaler Y
            scaler_y = None
            if os.path.exists(scaler_y_path):
                try:
                    import joblib
                    scaler_y = joblib.load(scaler_y_path)
                    logger.info(f"✅ Scaler Y carregado para H{horizon}")
                except Exception as e:
                    logger.error(f"❌ Erro ao carregar scaler Y {scaler_y_path}: {e}")
                    scaler_y = self._create_fallback_scaler(horizon)
            else:
                logger.warning(f"⚠️ Scaler Y não encontrado para H{horizon}, criando fallback")
                scaler_y = self._create_fallback_scaler(horizon)

            self.models[horizon] = model
            self.meta[horizon] = meta
            self.scalers_y[horizon] = scaler_y
            loaded_count += 1
            
            # Limpeza de memória entre carregamentos
            if loaded_count % 2 == 0:  # A cada 2 modelos
                gc.collect()

        if loaded_count > 0:
            self._is_loaded = True
            logger.info(f"🎯 {loaded_count} modelos carregados: {sorted(self.models.keys())}")
        else:
            logger.warning("⚠️ Nenhum modelo foi carregado")

    def ensure_loaded(self) -> bool:
        """Garante que os modelos estão carregados com fallback"""
        if not self._is_loaded:
            self._load_attempts += 1
            if self._load_attempts > 3:
                logger.error("💥 Muitas tentativas de carregamento - abortando")
                return False
            self._load_all_memory_optimized()
        return self._is_loaded

    def _validate_physical_ranges(self, values: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Valida se os valores estão dentro de ranges físicos possíveis"""
        warnings = []
        valid = True
        
        for param, value in values.items():
            if param in self.PHYSICAL_RANGES:
                range_info = self.PHYSICAL_RANGES[param]
                
                if value < range_info["min"] or value > range_info["max"]:
                    warnings.append(f"{param} fora do range: {value:.1f} (esperado: {range_info['min']}-{range_info['max']})")
                    valid = False
                elif abs(value - range_info["typical"]) > 3 * (range_info["max"] - range_info["min"]) / 4:
                    warnings.append(f"{param} atípico: {value:.1f} (típico: ~{range_info['typical']})")
        
        return valid, warnings

    def _apply_scaler_safe(self, y_scaled: np.ndarray, horizon: int) -> np.ndarray:
        """Aplica scaler Y com tratamento robusto de erros"""
        try:
            if horizon in self.scalers_y and self.scalers_y[horizon] is not None:
                # Garante formato 2D para inverse_transform
                y_scaled_2d = y_scaled.reshape(1, -1)
                y_real = self.scalers_y[horizon].inverse_transform(y_scaled_2d)[0]
                
                # Log para debug
                logger.debug(f"🔄 Scaling H{horizon}: {y_scaled} → {y_real}")
                return y_real
            else:
                logger.error(f"🚨 SCALER Y NÃO DISPONÍVEL para H{horizon}")
                # Fallback: assume que já está desnormalizado (arriscado)
                return y_scaled
                
        except Exception as e:
            logger.error(f"❌ Erro no scaling H{horizon}: {e}")
            # Fallback extremo: retorna valores típicos
            return np.array([450.0, 0.0, 7.0])  # Valores típicos

    def predict_safe(self, X_window: np.ndarray, horizon: int) -> PredictionResult:
        """
        Previsão com tratamento completo de erros e validação
        
        Args:
            X_window: Array (lookback, n_features) ou (1, lookback, n_features)
            horizon: Horizonte de previsão em horas
            
        Returns:
            PredictionResult: Resultado estruturado com validações
        """
        if not self.ensure_loaded():
            raise RuntimeError("Predictor não está carregado")
            
        if horizon not in self.models:
            available = sorted(self.models.keys())
            raise ValueError(f"Modelo H{horizon} não encontrado. Disponíveis: {available}")

        try:
            # Preprocessamento seguro da entrada
            arr = np.asarray(X_window, dtype=np.float32)  # float32 para economia
            
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)
            elif arr.ndim != 3:
                raise ValueError(f"Shape inválido: {arr.shape}. Esperado: (lookback, n_features) ou (1, lookback, n_features)")

            # Previsão com batch_size=1 para economia de memória
            model = self.models[horizon]
            y_scaled = model.predict(arr, batch_size=1, verbose=0)[0]
            
            # Aplica scaler Y com fallbacks
            y_real = self._apply_scaler_safe(y_scaled, horizon)
            
            # Mapeia para dicionário
            targets = self.meta.get(horizon, {}).get("targets", self.default_targets)
            values = {targets[i]: float(y_real[i]) for i in range(len(targets))}
            
            # Validação física
            is_valid, warnings = self._validate_physical_ranges(values)
            
            # Checks individuais por parâmetro
            physical_checks = {}
            for param in values:
                if param in self.PHYSICAL_RANGES:
                    range_info = self.PHYSICAL_RANGES[param]
                    physical_checks[param] = (
                        range_info["min"] <= values[param] <= range_info["max"]
                    )
            
            # Limpeza de memória
            del arr, y_scaled
            gc.collect()
            
            return PredictionResult(
                values=values,
                horizon=horizon,
                valid=is_valid,
                warnings=warnings,
                physical_checks=physical_checks
            )
            
        except Exception as e:
            logger.error(f"💥 Erro na previsão H{horizon}: {e}")
            raise

    def predict_from_features_array(self, X_window: np.ndarray, horizon: int) -> Dict[str, float]:
        """Interface compatível com versão anterior"""
        result = self.predict_safe(X_window, horizon)
        
        # Log warnings se houver
        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"⚠️ H{horizon}: {warning}")
        
        return result.values

    def predict_from_dataframe(self, df_feat: pd.DataFrame, horizon: int, lookback: int = None) -> Dict[str, float]:
        """Previsão a partir de DataFrame"""
        if not self.ensure_loaded():
            raise RuntimeError("Predictor não está carregado")
            
        # Determina lookback
        if lookback is None:
            lookback = self.meta.get(horizon, {}).get(
                "lookback",
                self.config.get("training")["main_lookback"],
            )

        if len(df_feat) < lookback:
            raise ValueError(
                f"Dados insuficientes: precisa {lookback}, tem {len(df_feat)}"
            )

        # Converte para float32 para economia
        window = df_feat.tail(lookback).astype(np.float32).values
        return self.predict_from_features_array(window, horizon)

    def batch_predict(self, X_windows: List[np.ndarray], horizons: List[int]) -> List[PredictionResult]:
        """Previsão em lote otimizada"""
        results = []
        
        for X_window, horizon in zip(X_windows, horizons):
            try:
                result = self.predict_safe(X_window, horizon)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Erro no batch H{horizon}: {e}")
                # Resultado de fallback
                results.append(PredictionResult(
                    values={},
                    horizon=horizon,
                    valid=False,
                    warnings=[f"Erro: {e}"],
                    physical_checks={}
                ))
            
            # Limpeza periódica de memória
            if len(results) % 3 == 0:
                gc.collect()
        
        return results

    def get_status(self) -> Dict[str, Any]:
        """Retorna status completo do predictor"""
        return {
            "loaded": self._is_loaded,
            "models_loaded": len(self.models),
            "scalers_loaded": len([s for s in self.scalers_y.values() if s is not None]),
            "available_horizons": sorted(self.models.keys()),
            "load_attempts": self._load_attempts,
            "memory_optimized": True
        }


# Singleton global para otimização
_predictor_instance = None

def get_predictor_github(config_path: str = "config.yaml") -> HACv6PredictorGithub:
    """Retorna instância singleton otimizada para GitHub"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HACv6PredictorGithub(config_path)
    return _predictor_instance


# Interface de compatibilidade
def get_predictor(config_path: str = "config.yaml") -> HACv6PredictorGithub:
    """Alias para compatibilidade com scripts existentes"""
    return get_predictor_github(config_path)


if __name__ == "__main__":
    # Teste rápido
    print("🧪 Teste rápido do HACv6PredictorGithub")
    
    try:
        predictor = get_predictor_github()
        status = predictor.get_status()
        print(f"✅ Status: {status}")
        
        if status["models_loaded"] > 0:
            print("🎯 Predictor pronto para uso no GitHub Free!")
        else:
            print("⚠️ Nenhum modelo carregado - verifique a pasta models/")
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
