#!/usr/bin/env python3
"""
SOLUÇÃO DEFINITIVA: Recria scalers Y usando as estatísticas REAIS dos dados de treino
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


class ScalerFixDefinitivo:
    """Solução definitiva - usa dados REAIS para criar scalers Y"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = HACConfig(config_path)
        self.feature_builder = HACFeatureBuilder(self.config)
    
    def get_real_training_stats(self):
        """Obtém estatísticas REAIS dos dados de treino"""
        logger.info("📊 Coletando estatísticas REAIS dos dados de treino...")
        
        datasets = self.feature_builder.build_all()
        real_stats = {}
        
        for horizon, data in datasets.items():
            if data["y"].size == 0:
                continue
                
            y_real = data["y"]  # Dados REAIS de treino
            
            # Estatísticas dos targets REAIS
            stats = {
                "speed_mean": float(np.mean(y_real[:, 0])),
                "speed_std": float(np.std(y_real[:, 0])),
                "bz_mean": float(np.mean(y_real[:, 1])), 
                "bz_std": float(np.std(y_real[:, 1])),
                "density_mean": float(np.mean(y_real[:, 2])),
                "density_std": float(np.std(y_real[:, 2])),
                "samples": len(y_real)
            }
            
            real_stats[horizon] = stats
            logger.info(f"  H{horizon}: {stats['samples']} amostras | "
                       f"Speed: {stats['speed_mean']:.1f}±{stats['speed_std']:.1f} | "
                       f"Density: {stats['density_mean']:.1f}±{stats['density_std']:.1f}")
        
        return real_stats
    
    def create_scaler_from_real_stats(self, stats):
        """Cria scaler Y baseado em estatísticas REAIS"""
        scaler = StandardScaler()
        
        # Gera dados sintéticos que replicam as estatísticas REAIS
        n_samples = 10000
        
        speed_data = np.random.normal(stats["speed_mean"], stats["speed_std"], n_samples)
        bz_data = np.random.normal(stats["bz_mean"], stats["bz_std"], n_samples)
        density_data = np.random.normal(stats["density_mean"], stats["density_std"], n_samples)
        
        # Combina e fit
        synthetic_data = np.column_stack([speed_data, bz_data, density_data])
        scaler.fit(synthetic_data)
        
        return scaler
    
    def find_all_model_folders(self):
        """Encontra todas as pastas de modelo"""
        model_dir = self.config.get("paths")["model_dir"]
        folders = []
        
        for folder in os.listdir(model_dir):
            folder_path = os.path.join(model_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            model_path = os.path.join(folder_path, "model.keras")
            if os.path.exists(model_path):
                try:
                    horizon = int(folder.split("_h")[1].split("_")[0])
                    folders.append((horizon, folder_path))
                except (IndexError, ValueError):
                    continue
        
        return folders
    
    def fix_all_scalers_definitively(self):
        """Correção DEFINITIVA de todos os scalers Y"""
        logger.info("🚀 INICIANDO CORREÇÃO DEFINITIVA DE SCALERS Y")
        
        # 1. Obtém estatísticas REAIS
        real_stats = self.get_real_training_stats()
        if not real_stats:
            logger.error("❌ Não foi possível obter estatísticas reais")
            return False
        
        # 2. Encontra todas as pastas de modelo
        model_folders = self.find_all_model_folders()
        if not model_folders:
            logger.error("❌ Nenhuma pasta de modelo encontrada")
            return False
        
        logger.info(f"🔍 Encontradas {len(model_folders)} pastas de modelo")
        
        fixed_count = 0
        
        for horizon, folder_path in model_folders:
            try:
                # Verifica se temos estatísticas para este horizonte
                if horizon not in real_stats:
                    logger.warning(f"⚠️  Sem estatísticas para H{horizon}, usando fallback")
                    # Fallback: usa estatísticas de outro horizonte ou valores padrão
                    if real_stats:  # Usa o primeiro horizonte disponível
                        fallback_horizon = list(real_stats.keys())[0]
                        stats = real_stats[fallback_horizon]
                        logger.info(f"   Usando estatísticas de H{fallback_horizon} para H{horizon}")
                    else:
                        # Fallback extremo - valores OMNI típicos
                        stats = {
                            "speed_mean": 450.0, "speed_std": 150.0,
                            "bz_mean": 0.5, "bz_std": 6.0,
                            "density_mean": 7.0, "density_std": 5.0
                        }
                else:
                    stats = real_stats[horizon]
                
                # Cria scaler baseado em estatísticas REAIS
                scaler = self.create_scaler_from_real_stats(stats)
                
                # Salva o scaler
                scaler_path = os.path.join(folder_path, "scaler_Y.pkl")
                joblib.dump(scaler, scaler_path)
                
                # Atualiza metadata
                meta_path = os.path.join(folder_path, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata["scaler_y_type"] = "definitive_real_stats"
                    metadata["scaler_y_fixed_at"] = datetime.now().isoformat()
                    metadata["training_stats_used"] = stats
                    
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                fixed_count += 1
                logger.info(f"✅ H{horizon}: Scaler criado com stats reais | "
                           f"Means: [{scaler.mean_[0]:.1f}, {scaler.mean_[1]:.1f}, {scaler.mean_[2]:.1f}]")
                
            except Exception as e:
                logger.error(f"❌ Erro em H{horizon}: {e}")
        
        logger.info(f"🎯 {fixed_count}/{len(model_folders)} scalers Y corrigidos definitivamente")
        return fixed_count > 0
    
    def verify_fix(self):
        """Verifica se a correção funcionou"""
        logger.info("🔍 Verificando correção...")
        
        try:
            from hac_v6_predictor import HACv6Predictor
            
            predictor = HACv6Predictor()
            
            print(f"\n📋 STATUS PÓS-CORREÇÃO:")
            print(f"   Modelos carregados: {len(predictor.models)}")
            print(f"   Scalers Y disponíveis: {predictor.has_scalers_y()}")
            
            # Testa uma previsão rápida
            feature_builder = HACFeatureBuilder(self.config)
            datasets = feature_builder.build_all()
            
            if datasets:
                # Testa com o primeiro horizonte disponível
                test_horizon = list(datasets.keys())[0]
                X_test = datasets[test_horizon]["X"][-1]
                
                try:
                    prediction = predictor.predict_from_features_array(X_test, test_horizon)
                    print(f"\n🧪 TESTE H{test_horizon}:")
                    for param, value in prediction.items():
                        print(f"   {param:>10}: {value:8.1f}")
                    
                    # Verificação física
                    speed = prediction.get('speed', 0)
                    density = prediction.get('density', 0)
                    
                    print(f"\n🎯 VERIFICAÇÃO FÍSICA:")
                    print(f"   Speed:    {speed:6.1f} km/s → {'✅ OK' if 200 < speed < 2000 else '❌ PROBLEMA'}")
                    print(f"   Density:  {density:6.1f} cm⁻³ → {'✅ OK' if 0.1 < density < 100 else '❌ PROBLEMA'}")
                    
                    return 200 < speed < 2000 and 0.1 < density < 100
                    
                except Exception as e:
                    print(f"❌ Erro no teste: {e}")
                    return False
            
        except Exception as e:
            logger.error(f"Erro na verificação: {e}")
            return False


def main():
    """Correção definitiva principal"""
    print("🚀 CORREÇÃO DEFINITIVA - SCALERS Y COM DADOS REAIS")
    print("=" * 60)
    
    fixer = ScalerFixDefinitivo()
    
    # Executa correção
    success = fixer.fix_all_scalers_definitively()
    
    if success:
        print(f"\n✅ CORREÇÃO APLICADA!")
        
        # Verificação
        print(f"\n🔍 VERIFICANDO RESULTADOS...")
        verification_ok = fixer.verify_fix()
        
        if verification_ok:
            print(f"\n🎉 CORREÇÃO BEM-SUCEDIDA! Valores agora devem ser realistas.")
            print(f"\n🔥 PRÓXIMOS PASSOS:")
            print("python3 scripts/save_report.py")
            print("cat results/model_report.json | grep -A 5 'values'")
        else:
            print(f"\n⚠️  Correção aplicada mas verificação falhou. Pode precisar de retreino.")
            print("python3 scripts/lightweight_retrain.py")
    else:
        print(f"\n❌ Falha na correção definitiva")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
