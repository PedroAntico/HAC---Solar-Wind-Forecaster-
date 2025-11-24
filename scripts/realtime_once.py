#!/usr/bin/env python3
"""
realtime_once.py - Previsão única otimizada
"""

import os
import sys
import json
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hac_v6_config import HACConfig
    from hac_v6_features import HACFeatureBuilder
    from hac_v6_predictor import get_predictor
except ImportError as e:
    logger.error(f"Erro de importação: {e}")
    sys.exit(1)


def main():
    """Executa uma rodada de previsão"""
    try:
        logger.info("🚀 Iniciando previsão única HAC v6")
        
        # Configuração
        config = HACConfig("config.yaml")
        
        # Inicialização
        feature_builder = HACFeatureBuilder(config)
        predictor = get_predictor("config.yaml")
        
        if not predictor.ensure_loaded():
            logger.error("❌ Nenhum modelo carregado")
            return False

        # Features
        logger.info("🧪 Gerando features...")
        datasets = feature_builder.build_all()
        horizons = config.get("horizons")
        
        # Thresholds
        thresholds = config.get("alerts", {}).get("thresholds", {})
        
        # Previsões
        predictions = {}
        success_count = 0
        
        for h in horizons:
            if h not in datasets or datasets[h]["X"].size == 0:
                logger.warning(f"⚠️ Dados insuficientes para H{h}")
                predictions[str(h)] = {"error": "Dados insuficientes", "alerts": []}
                continue
                
            try:
                X_last = datasets[h]["X"][-1]
                pred = predictor.predict_from_features_array(X_last, h)
                
                # Alertas
                alerts = []
                speed = pred.get("speed", 0)
                density = pred.get("density", 0)
                bz = pred.get("bz_gsm", pred.get("bz", 0))
                
                if speed > thresholds.get("speed_high", 600):
                    alerts.append("HIGH_SPEED")
                if speed < thresholds.get("speed_low", 300):
                    alerts.append("LOW_SPEED")
                if density > thresholds.get("density_high", 20):
                    alerts.append("HIGH_DENSITY")
                if abs(bz) > thresholds.get("bz_extreme", 10):
                    alerts.append("EXTREME_BZ")
                
                predictions[str(h)] = {
                    "values": pred,
                    "alerts": alerts
                }
                success_count += 1
                logger.info(f"  ✅ H{h}: {pred}")
                
            except Exception as e:
                logger.error(f"  ❌ Erro H{h}: {e}")
                predictions[str(h)] = {"error": str(e), "alerts": []}

        # Salva resultados
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "latest_forecast.json")
        
        output = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "predictions": predictions,
            "success_rate": f"{success_count}/{len(horizons)}"
        }
        
        with open(out_path, "w", encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📦 Previsão salva: {out_path}")
        logger.info(f"🎯 Sucesso: {success_count}/{len(horizons)} horizontes")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
