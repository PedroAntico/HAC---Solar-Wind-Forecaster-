#!/usr/bin/env python3
"""
hac_v6_api.py - API Flask otimizada para GitHub Free
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, jsonify

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from hac_v6_config import HACConfig
    from hac_v6_features import HACFeatureBuilder
    from hac_v6_predictor import get_predictor
except ImportError:
    # Fallback para imports locais
    import sys
    sys.path.append('.')
    from hac_v6_config import HACConfig
    from hac_v6_features import HACFeatureBuilder
    from hac_v6_predictor import get_predictor


# Configuração global
class APIConfig:
    def __init__(self):
        self.config = HACConfig("config.yaml")
        self.feature_builder = None
        self.predictor = None
        self.thresholds = self.config.get("alerts", {}).get("thresholds", {})
        self._initialized = False
        
    def initialize(self):
        """Inicialização lazy dos componentes"""
        if self._initialized:
            return True
            
        try:
            logger.info("🧱 Inicializando feature builder...")
            self.feature_builder = HACFeatureBuilder(self.config)
            
            logger.info("🧠 Inicializando predictor...")
            self.predictor = get_predictor("config.yaml")
            
            if not self.predictor.ensure_loaded():
                logger.error("❌ Nenhum modelo carregado")
                return False
                
            self._initialized = True
            logger.info("✅ API inicializada com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            return False


# Instância global
api_config = APIConfig()


def compute_alerts(pred: Dict[str, float]) -> list:
    """Calcula alertas baseados nas previsões"""
    alerts = []
    
    try:
        speed = pred.get("speed", 0.0)
        density = pred.get("density", 0.0)
        bz = pred.get("bz_gsm", pred.get("bz", 0.0))

        thresholds = api_config.thresholds
        
        if speed > thresholds.get("speed_high", 600):
            alerts.append("HIGH_SPEED")
        if speed < thresholds.get("speed_low", 300):
            alerts.append("LOW_SPEED")
        if density > thresholds.get("density_high", 20):
            alerts.append("HIGH_DENSITY")
        if abs(bz) > thresholds.get("bz_extreme", 10):
            alerts.append("EXTREME_BZ")
            
    except Exception as e:
        logger.warning(f"Erro ao calcular alertas: {e}")
        
    return alerts


def get_latest_window(horizon: int):
    """Obtém última janela para horizonte específico"""
    if not api_config.initialized:
        raise RuntimeError("API não inicializada")
        
    try:
        datasets = api_config.feature_builder.build_all()
        if horizon not in datasets:
            available = list(datasets.keys())
            raise ValueError(f"Horizonte {horizon}h não disponível. Disponíveis: {available}")
            
        X = datasets[horizon]["X"]
        if X.size == 0:
            raise ValueError(f"Dados vazios para horizonte {horizon}h")
            
        return X[-1]  # Última janela
        
    except Exception as e:
        logger.error(f"Erro ao obter janela H{horizon}: {e}")
        raise


# Flask App
app = Flask(__name__)


@app.before_request
def before_request():
    """Inicialização antes de cada request"""
    if not api_config.initialized:
        if not api_config.initialize():
            return jsonify({"error": "Serviço não disponível"}), 503


@app.route("/")
def home():
    """Página inicial"""
    return jsonify({
        "service": "HAC v6 Solar Wind Forecast API",
        "version": "1.0",
        "status": "online",
        "available_horizons": api_config.predictor.get_available_horizons() if api_config.predictor else []
    })


@app.route("/api/v1/forecast")
def forecast():
    """Endpoint principal de previsão"""
    try:
        # Parâmetros
        model_type = request.args.get("model_type", "hybrid")
        horizon = request.args.get("horizon", "24")
        
        # Validação
        try:
            horizon = int(horizon)
        except ValueError:
            return jsonify({"error": "Horizon deve ser um número inteiro"}), 400
            
        available_horizons = api_config.predictor.get_available_horizons()
        if horizon not in available_horizons:
            return jsonify({
                "error": f"Horizonte {horizon}h não disponível",
                "available_horizons": available_horizons
            }), 400

        # Previsão
        X_last = get_latest_window(horizon)
        pred = api_config.predictor.predict_from_features_array(X_last, horizon)
        alerts = compute_alerts(pred)

        return jsonify({
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "model_type": model_type,
            "horizon": horizon,
            "predictions": pred,
            "alerts": alerts,
            "success": True
        })

    except Exception as e:
        logger.error(f"Erro no endpoint /forecast: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route("/api/v1/health")
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models_loaded": len(api_config.predictor.models) if api_config.predictor else 0
    })


@app.route("/api/v1/horizons")
def horizons():
    """Lista horizontes disponíveis"""
    available = api_config.predictor.get_available_horizons() if api_config.predictor else []
    return jsonify({
        "available_horizons": available,
        "count": len(available)
    })


if __name__ == "__main__":
    port = int(os.environ.get("HAC_API_PORT", "5000"))
    debug = os.environ.get("HAC_DEBUG", "false").lower() == "true"
    
    logger.info(f"🚀 Iniciando HAC v6 API na porta {port}")
    
    if not api_config.initialize():
        logger.error("❌ Falha na inicialização - saindo")
        exit(1)
        
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=debug,
        # Otimizado para GitHub Free
        threaded=True,
        processes=1
    )
