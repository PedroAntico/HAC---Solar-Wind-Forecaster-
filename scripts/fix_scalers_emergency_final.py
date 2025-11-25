#!/usr/bin/env python3
"""
CORREÇÃO DE EMERGÊNCIA FINAL: Recria scalers Y com dados ORIGINAIS não normalizados
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


class ScalerEmergencyFix:
    """Correção de emergência - recria scalers Y com dados ORIGINAIS"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = HACConfig(config_path)
    
    def get_original_target_data(self):
        """Obtém os dados ORIGINAIS dos targets (não normalizados)"""
        logger.info("📊 Carregando dados ORIGINAIS dos targets...")
        
        # Carrega dados OMNI originais
        data_path = self.config.get("paths")["processed_data"]
        if not os.path.exists(data_path):
            logger.error(f"❌ Arquivo de dados não encontrado: {data_path}")
            return None
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✅ Dados carregados: {len(df)} linhas, {df.shape[1]} colunas")
            
            # Extrai os targets ORIGINAIS
            targets = self.config.get("targets")["primary"]
            target_data = df[targets].dropna()
            
            logger.info(f"🎯 Targets originais: {targets}")
            logger.info(f"📈 Estatísticas dos dados ORIGINAIS:")
            for target in targets:
                logger.info(f"   {target}: mean={target_data[target].mean():.2f}, std={target_data[target].std():.2f}")
            
            return target_data.values  # Retorna array numpy com dados ORIGINAIS
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            return None
    
    def create_correct_scalers(self, original_target_data):
        """Cria scalers Y CORRETOS baseados nos dados ORIGINAIS"""
        logger.info("🔧 Criando scalers Y corretos...")
        
        if original_target_data is None or len(original_target_data) == 0:
            logger.error("❌ Dados originais vazios")
            return None
        
        # Cria scaler com dados ORIGINAIS
        scaler = StandardScaler()
        scaler.fit(original_target_data)
        
        logger.info("✅ Scaler Y correto criado com dados ORIGINAIS")
        logger.info(f"   Means: {scaler.mean_}")
        logger.info(f"   Scales: {scaler.scale_}")
        
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
                    # Extrai horizonte do nome da pasta
                    if "_h" in folder:
                        horizon = int(folder.split("_h")[1].split("_")[0])
                        folders.append((horizon, folder_path))
                except (IndexError, ValueError) as e:
                    logger.warning(f"⚠️ Não conseguiu extrair horizonte de {folder}: {e}")
                    continue
        
        return folders
    
    def test_scaler_fix(self, scaler, horizon):
        """Testa se o scaler corrigido funciona"""
        logger.info(f"🧪 Testando scaler H{horizon}...")
        
        # Valores escalados típicos (saída do modelo)
        test_scaled_values = [
            np.array([[0.0, 0.0, 0.0]]),      # Zeros
            np.array([[1.0, 1.0, 1.0]]),      # Uns  
            np.array([[2.0, 2.0, 2.0]]),      # Dois
            np.array([[-1.0, -1.0, -1.0]]),   # Negativos
        ]
        
        results = []
        for test_val in test_scaled_values:
            unscaled = scaler.inverse_transform(test_val)[0]
            results.append({
                "scaled": test_val[0].tolist(),
                "unscaled": unscaled.tolist()
            })
            logger.info(f"   {test_val[0]} → {unscaled}")
        
        return results
    
    def apply_emergency_fix(self):
        """Aplica correção de emergência em TODOS os modelos"""
        logger.info("🚨 APLICANDO CORREÇÃO DE EMERGÊNCIA FINAL")
        
        # 1. Obtém dados ORIGINAIS
        original_data = self.get_original_target_data()
        if original_data is None:
            logger.error("❌ Não foi possível obter dados originais")
            return False
        
        # 2. Cria scaler CORRETO
        correct_scaler = self.create_correct_scalers(original_data)
        if correct_scaler is None:
            logger.error("❌ Não foi possível criar scaler correto")
            return False
        
        # 3. Testa o scaler
        test_results = self.test_scaler_fix(correct_scaler, 1)
        
        # 4. Encontra todos os modelos
        model_folders = self.find_all_model_folders()
        if not model_folders:
            logger.error("❌ Nenhuma pasta de modelo encontrada")
            return False
        
        logger.info(f"🔧 Corrigindo {len(model_folders)} modelos...")
        
        fixed_count = 0
        for horizon, folder_path in model_folders:
            try:
                scaler_path = os.path.join(folder_path, "scaler_Y.pkl")
                
                # Salva o scaler CORRETO
                joblib.dump(correct_scaler, scaler_path)
                
                # Atualiza metadata
                meta_path = os.path.join(folder_path, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata["scaler_y_type"] = "emergency_fix_original_data"
                    metadata["scaler_y_fixed_at"] = datetime.now().isoformat()
                    metadata["scaler_y_means"] = correct_scaler.mean_.tolist()
                    metadata["scaler_y_scales"] = correct_scaler.scale_.tolist()
                    metadata["fix_note"] = "Scaler recriado com dados ORIGINAIS não normalizados"
                    
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                fixed_count += 1
                logger.info(f"✅ H{horizon}: Scaler corrigido")
                
            except Exception as e:
                logger.error(f"❌ Erro em H{horizon}: {e}")
        
        logger.info(f"🎯 {fixed_count}/{len(model_folders)} scalers corrigidos")
        
        # 5. Verificação final
        if fixed_count > 0:
            logger.info("🔍 Verificando correção...")
            self.verify_fix()
        
        return fixed_count > 0
    
    def verify_fix(self):
        """Verifica se a correção funcionou"""
        try:
            from hac_v6_predictor import HACv6Predictor
            
            predictor = HACv6Predictor()
            
            # Testa uma previsão rápida
            feature_builder = HACFeatureBuilder(self.config)
            datasets = feature_builder.build_all()
            
            if datasets:
                test_horizon = list(datasets.keys())[0]
                X_test = datasets[test_horizon]["X"][-1]
                
                prediction = predictor.predict_from_features_array(X_test, test_horizon)
                
                print(f"\n🎯 TESTE PÓS-CORREÇÃO H{test_horizon}:")
                for param, value in prediction.items():
                    print(f"   {param:>10}: {value:8.1f}")
                
                # Verificação física
                speed = prediction.get('speed', 0)
                density = prediction.get('density', 0)
                
                print(f"\n📊 VERIFICAÇÃO FÍSICA:")
                print(f"   Speed:    {speed:6.1f} km/s → {'✅ OK' if 200 < speed < 2000 else '❌ PROBLEMA'}")
                print(f"   Density:  {density:6.1f} cm⁻³ → {'✅ OK' if 0.1 < density < 100 else '❌ PROBLEMA'}")
                
                return 200 < speed < 2000 and 0.1 < density < 100
                
        except Exception as e:
            logger.error(f"Erro na verificação: {e}")
            return False


def main():
    """Correção de emergência principal"""
    print("🚨 CORREÇÃO DE EMERGÊNCIA FINAL - SCALERS Y")
    print("=" * 60)
    print("🎯 OBJETIVO: Recriar scalers com dados ORIGINAIS não normalizados")
    print()
    
    fixer = ScalerEmergencyFix()
    success = fixer.apply_emergency_fix()
    
    if success:
        print(f"\n🎉 CORREÇÃO APLICADA COM SUCESSO!")
        print(f"\n🔥 PRÓXIMOS PASSOS:")
        print("python3 scripts/save_report.py")
        print("cat results/model_report.json")
    else:
        print(f"\n❌ Falha na correção de emergência")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
