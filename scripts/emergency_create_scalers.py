#!/usr/bin/env python3
"""
EMERGENCY: Cria scalers Y para modelos existentes quando o treino foi interrompido
"""

import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hac_v6_config import HACConfig
    from hac_v6_features import HACFeatureBuilder
except ImportError as e:
    logger.error(f"Erro de importação: {e}")
    sys.exit(1)


class EmergencyScalerFix:
    """Cria scalers Y de emergência baseados nas estatísticas dos dados OMNI"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = HACConfig(config_path)
        self.feature_builder = HACFeatureBuilder(config)
        
        # 🎯 ESTATÍSTICAS REAIS DO VENTO SOLAR (base OMNI)
        self.omni_stats = {
            "speed": {"mean": 400.0, "std": 100.0},     # típico: 400 ± 100 km/s
            "bz_gsm": {"mean": 0.0, "std": 5.0},       # típico: 0 ± 5 nT  
            "density": {"mean": 5.0, "std": 3.0}       # típico: 5 ± 3 cm⁻³
        }
    
    def create_dummy_scaler_y(self, horizon: int):
        """Cria um scaler Y dummy baseado em estatísticas realistas"""
        scaler = StandardScaler()
        
        # Para o StandardScaler funcionar, precisamos "enganá-lo" com dados realistas
        n_samples = 1000
        
        # Gera dados sintéticos baseados nas estatísticas OMNI
        speed_data = np.random.normal(
            self.omni_stats["speed"]["mean"], 
            self.omni_stats["speed"]["std"], 
            n_samples
        )
        bz_data = np.random.normal(
            self.omni_stats["bz_gsm"]["mean"],
            self.omni_stats["bz_gsm"]["std"], 
            n_samples
        )
        density_data = np.random.normal(
            self.omni_stats["density"]["mean"],
            self.omni_stats["density"]["std"],
            n_samples
        )
        
        # Combina em array 2D (n_samples, 3)
        dummy_data = np.column_stack([speed_data, bz_data, density_data])
        
        # Fit do scaler nos dados realistas
        scaler.fit(dummy_data)
        
        logger.info(f"✅ Scaler Y dummy criado para H{horizon}")
        logger.info(f"   Means: {scaler.mean_}")  
        logger.info(f"   Stds: {scaler.scale_}")
        
        return scaler
    
    def find_models_missing_scalers(self):
        """Encontra todos os modelos que estão sem scalers Y"""
        model_dir = self.config.get("paths")["model_dir"]
        missing = []
        
        for folder in os.listdir(model_dir):
            folder_path = os.path.join(model_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            model_path = os.path.join(folder_path, "model.keras")
            scaler_y_path = os.path.join(folder_path, "scaler_Y.pkl")
            
            if os.path.exists(model_path) and not os.path.exists(scaler_y_path):
                # Extrai horizonte do nome da pasta
                try:
                    horizon = int(folder.split("_h")[1].split("_")[0])
                    missing.append((horizon, folder_path))
                except (IndexError, ValueError):
                    continue
        
        return missing
    
    def fix_all_missing_scalers(self):
        """Corrige todos os scalers Y faltantes"""
        missing_models = self.find_models_missing_scalers()
        
        if not missing_models:
            logger.info("🎉 Nenhum scaler Y faltante encontrado!")
            return True
        
        logger.info(f"🔧 Encontrados {len(missing_models)} modelos sem scalers Y")
        
        success_count = 0
        for horizon, folder_path in missing_models:
            try:
                scaler_y = self.create_dummy_scaler_y(horizon)
                scaler_path = os.path.join(folder_path, "scaler_Y.pkl")
                
                # Salva o scaler Y
                joblib.dump(scaler_y, scaler_path)
                
                # Atualiza metadata para indicar que é um scaler de emergência
                meta_path = os.path.join(folder_path, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata["scaler_y_type"] = "emergency_dummy"
                    metadata["scaler_y_fixed_at"] = datetime.now().isoformat()
                    
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                success_count += 1
                logger.info(f"   ✅ H{horizon}: Scaler Y criado em {scaler_path}")
                
            except Exception as e:
                logger.error(f"   ❌ H{horizon}: Erro ao criar scaler - {e}")
        
        logger.info(f"🎯 {success_count}/{len(missing_models)} scalers Y criados com sucesso")
        return success_count > 0


def main():
    """Correção de emergência principal"""
    logger.info("🚨 INICIANDO CORREÇÃO DE EMERGÊNCIA - SCALERS Y")
    logger.info("📝 Criando scalers Y dummy baseados em estatísticas OMNI...")
    
    fixer = EmergencyScalerFix()
    success = fixer.fix_all_missing_scalers()
    
    if success:
        logger.info("🎉 CORREÇÃO CONCLUÍDA! Agora teste as previsões:")
        print("\n🔥 COMANDOS PARA TESTAR:")
        print("python3 -c \"")
        print("from hac_v6_predictor import HACv6Predictor")
        print("p = HACv6Predictor()")
        print("print('Scalers Y carregados:', p.has_scalers_y())")
        print("\"")
        print("\npython3 scripts/save_report.py")
        print("cat results/model_report.json | grep -A 5 '\"values\"'")
    else:
        logger.error("❌ Falha na correção")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
