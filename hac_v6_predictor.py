#!/usr/bin/env python3
"""
hac_v6_predictor.py - Predictor otimizado para GitHub Free
"""

import os
import re
import json
import logging
from typing import Dict, Any, Tuple, Optional
import gc

import numpy as np
import pandas as pd

# Configuração de logging
logging.basicConfig(level=logging.INFO)
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


class HACv6Predictor:
    """Carregador e predictor otimizado para GitHub Free"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = HACConfig(config_path)
        paths = self.config.get("paths")
        self.model_dir = paths["model_dir"]
        self.default_targets = self.config.get("targets")["primary"]
        
        self.models: Dict[int, Any] = {}
        self.meta: Dict[int, Dict[str, Any]] = {}
        self._is_loaded = False
        
        logger.info("🧠 Inicializando HACv6Predictor...")
        
    def _safe_load_model(self, model_path: str) -> Optional[Any]:
        """Carrega modelo com tratamento de erro"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow não está disponível")
            
        try:
            # Otimização para GitHub Free - limpa memória antes
            gc.collect()
            model = load_model(model_path)
            logger.info(f"✅ Modelo carregado: {os.path.basename(model_path)}")
            return model
        except Exception as e:
            logger.error(f"❌ Erro ao carregar {model_path}: {e}")
            return None

    def _load_all(self) -> None:
        """Carrega modelos com fallbacks"""
        if not os.path.isdir(self.model_dir):
            logger.warning(f"Diretório não encontrado: {self.model_dir}")
            # Tenta criar ou usar fallback
            os.makedirs(self.model_dir, exist_ok=True)
            return

        pattern = re.compile(r"^(?P<type>\w+)_h(?P<h>\d+)_")
        loaded_count = 0

        for folder in sorted(os.listdir(self.model_dir)):
            full_path = os.path.join(self.model_dir, folder)
            if not os.path.isdir(full_path):
                continue

            match = pattern.match(folder)
            if not match:
                continue

            horizon = int(match.group("h"))
            model_path = os.path.join(full_path, "model.keras")
            meta_path = os.path.join(full_path, "metadata.json")

            if not os.path.exists(model_path):
                logger.warning(f"Modelo não encontrado: {model_path}")
                continue

            # Carrega modelo
            model = self._safe_load_model(model_path)
            if model is None:
                continue

            # Carrega metadata
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                except Exception as e:
                    logger.warning(f"Erro ao carregar metadata {meta_path}: {e}")

            self.models[horizon] = model
            self.meta[horizon] = meta
            loaded_count += 1

        if loaded_count > 0:
            self._is_loaded = True
            logger.info(f"✅ {loaded_count} modelos carregados: {sorted(self.models.keys())}")
        else:
            logger.warning("⚠️ Nenhum modelo foi carregado")

    def ensure_loaded(self) -> bool:
        """Garante que os modelos estão carregados"""
        if not self._is_loaded:
            self._load_all()
        return self._is_loaded

    def _ensure_model(self, horizon: int) -> Tuple[Any, Dict[str, Any]]:
        if not self.ensure_loaded():
            raise RuntimeError("Nenhum modelo disponível")
        if horizon not in self.models:
            available = sorted(self.models.keys())
            raise ValueError(f"Modelo H{horizon} não encontrado. Disponíveis: {available}")
        return self.models[horizon], self.meta.get(horizon, {})

    def predict_from_features_array(
        self,
        X_window: np.ndarray,
        horizon: int,
    ) -> Dict[str, float]:
        """Previsão a partir de array de features"""
        model, meta = self._ensure_model(horizon)

        try:
            arr = np.asarray(X_window, dtype=np.float32)  # Otimização: float32
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)
            elif arr.ndim != 3:
                raise ValueError(f"Shape inválido: {arr.shape}")

            # Previsão com batch_size=1 para otimizar memória
            y_pred = model.predict(arr, batch_size=1, verbose=0)[0]
            
            targets = meta.get("targets", self.default_targets)
            result = {targets[i]: float(y_pred[i]) for i in range(len(targets))}
            
            # Limpeza de memória
            del arr, y_pred
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na previsão H{horizon}: {e}")
            raise

    def predict_from_dataframe(
        self,
        df_feat: pd.DataFrame,
        horizon: int,
        lookback: Optional[int] = None,
    ) -> Dict[str, float]:
        """Previsão a partir de DataFrame"""
        _, meta = self._ensure_model(horizon)

        if lookback is None:
            lookback = meta.get(
                "lookback",
                self.config.get("training")["main_lookback"],
            )

        if len(df_feat) < lookback:
            raise ValueError(
                f"Dados insuficientes: precisa {lookback}, tem {len(df_feat)}"
            )

        # Usa float32 para otimização
        window = df_feat.tail(lookback).astype(np.float32).values
        return self.predict_from_features_array(window, horizon)

    def predict(self, data: Any, horizon: int, lookback: Optional[int] = None) -> Dict[str, float]:
        """Interface unificada"""
        if isinstance(data, pd.DataFrame):
            return self.predict_from_dataframe(data, horizon, lookback)
        else:
            return self.predict_from_features_array(data, horizon)

    def get_available_horizons(self) -> list:
        """Retorna horizontes disponíveis"""
        self.ensure_loaded()
        return sorted(self.models.keys())


# Singleton para otimização
_predictor_instance = None

def get_predictor(config_path: str = "config.yaml") -> HACv6Predictor:
    """Retorna instância singleton do predictor"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HACv6Predictor(config_path)
    return _predictor_instance
