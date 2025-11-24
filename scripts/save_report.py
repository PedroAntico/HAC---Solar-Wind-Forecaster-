#!/usr/bin/env python3
"""
save_report.py - Relatório otimizado para GitHub Free
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adiciona path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hac_v6_config import HACConfig
    from hac_v6_features import HACFeatureBuilder
    from hac_v6_predictor import get_predictor
except ImportError as e:
    logger.error(f"Erro de importação: {e}")
    sys.exit(1)


def safe_build_features(feature_builder, config):
    """Construção segura de features com fallback"""
    try:
        logger.info("🧪 Gerando features...")
        datasets = feature_builder.build_all()
        
        horizons = config.get("horizons")
        latest_windows = {}
        
        for h in horizons:
            if h in datasets and datasets[h]["X"].size > 0:
                latest_windows[h] = datasets[h]["X"][-1]
            else:
                logger.warning(f"⚠️ Dados insuficientes para horizonte {h}h")
                
        return latest_windows
        
    except Exception as e:
        logger.error(f"Erro ao construir features: {e}")
        return {}


def compute_alerts(pred: Dict[str, float], thresholds: Dict[str, float]) -> list:
    """Calcula alertas de forma segura"""
    alerts = []
    
    try:
        speed = pred.get("speed", 0.0)
        density = pred.get("density", 0.0)
        bz = pred.get("bz_gsm", pred.get("bz", 0.0))

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


def main():
    """Função principal com tratamento robusto de erros"""
    try:
        logger.info("🚀 Iniciando geração de relatório HAC v6")
        
        # Configuração
        config = HACConfig("config.yaml")
        
        # Diretório de resultados
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "model_report.json")
        
        # Inicializa componentes
        feature_builder = HACFeatureBuilder(config)
        predictor = get_predictor("config.yaml")
        
        if not predictor.ensure_loaded():
            logger.error("❌ Nenhum modelo carregado - abortando")
            return False

        # Gera features
        latest_windows = safe_build_features(feature_builder, config)
        if not latest_windows:
            logger.error("❌ Não foi possível gerar features")
            return False

        # Previsões
        thresholds = config.get("alerts", {}).get("thresholds", {})
        predictions = {}
        
        logger.info("🔮 Gerando previsões...")
        for h in sorted(latest_windows.keys()):
            try:
                X_last = latest_windows[h]
                pred = predictor.predict_from_features_array(X_last, h)
                alerts = compute_alerts(pred, thresholds)
                
                predictions[str(h)] = {
                    "ok": True,
                    "values": pred,
                    "alerts": alerts,
                }
                
                logger.info(f"  ✅ H{h}: {pred} | alerts={alerts}")
                
            except Exception as e:
                logger.error(f"  ❌ Erro H{h}: {e}")
                predictions[str(h)] = {
                    "ok": False,
                    "error": str(e),
                    "alerts": [],
                }

        # Salva relatório
        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "horizons": sorted(predictions.keys(), key=int),
            "predictions": predictions,
            "success": any(p.get("ok", False) for p in predictions.values())
        }
        
        with open(out_path, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📦 Relatório salvo: {out_path}")
        
        # Status final
        success_count = sum(1 for p in predictions.values() if p.get("ok", False))
        logger.info(f"🎉 Concluído: {success_count}/{len(predictions)} previsões bem-sucedidas")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
