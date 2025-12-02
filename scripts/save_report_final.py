#!/usr/bin/env python3
"""
save_report_final.py - Gera relatório de previsões com dados reais
"""

import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from hac_v6_predictor_final import get_predictor


def main():
    print("📊 HAC v6 - Relatório de Previsões (Versão Definitiva)")
    print("=" * 60)
    
    # Configuração
    config = HACConfig()
    
    # Inicializa componentes
    feature_builder = HACFeatureBuilder(config)
    predictor = get_predictor()
    
    # Gera features
    print("🧪 Gerando features...")
    datasets = feature_builder.build_all()
    
    if not datasets:
        print("❌ Não foi possível gerar features")
        return False
    
    # Previsões para cada horizonte
    print("🔮 Gerando previsões...")
    results = {}
    
    for horizon in sorted(datasets.keys()):
        if horizon not in predictor.get_available_horizons():
            print(f"⚠️ Horizonte {horizon}h não disponível no predictor")
            continue
        
        try:
            # Última janela de dados
            X = datasets[horizon]["X"]
            if X.size == 0:
                print(f"⚠️ Dados vazios para H{horizon}")
                continue
            
            X_window = X[-1]  # Última janela
            
            # Previsão
            forecast = predictor.predict(X_window, horizon)
            
            results[horizon] = {
                "ok": True,
                "values": forecast.values,
                "alerts": forecast.alerts,
                "warnings": forecast.warnings,
                "valid": forecast.valid
            }
            
            # Log conciso
            values_str = ", ".join([f"{k}: {v:.1f}" for k, v in forecast.values.items()])
            print(f"  ✅ H{horizon}: {values_str} | alerts: {forecast.alerts}")
            
            if forecast.warnings:
                print(f"     ⚠️  {forecast.warnings}")
                
        except Exception as e:
            print(f"  ❌ Erro H{horizon}: {e}")
            results[horizon] = {
                "ok": False,
                "error": str(e),
                "alerts": []
            }
    
    # Salva relatório
    os.makedirs("results", exist_ok=True)
    report_path = "results/model_report_final.json"
    
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "horizons": list(results.keys()),
        "predictions": results,
        "success": any(r.get("ok", False) for r in results.values())
    }
    
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📦 Relatório salvo: {report_path}")
    print("🎯 Concluído!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
