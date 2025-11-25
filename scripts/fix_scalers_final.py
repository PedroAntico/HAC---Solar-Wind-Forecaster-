#!/usr/bin/env python3
"""
CORREÇÃO DEFINITIVA SIMPLES: Recria scalers Y usando dados OMNI originais
"""

import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    print("🎯 CORREÇÃO DEFINITIVA - SCALERS Y")
    print("=" * 50)
    
    # 1. Carrega dados OMNI originais
    data_path = "data_real/omni_prepared.csv"  # Caminho DIRETO
    print(f"📊 Carregando dados de: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Arquivo não encontrado: {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    print(f"✅ Dados carregados: {len(df)} linhas")
    
    # 2. Extrai targets ORIGINAIS (não normalizados)
    targets = ["speed", "bz_gsm", "density"]  # Targets DIRETOS
    target_data = df[targets].dropna()
    
    print(f"🎯 Targets originais: {targets}")
    print(f"📈 Estatísticas ORIGINAIS:")
    for target in targets:
        print(f"   {target}: mean={target_data[target].mean():.2f}, std={target_data[target].std():.2f}")
    
    # 3. Cria scaler CORRETO com dados ORIGINAIS
    scaler = StandardScaler()
    scaler.fit(target_data.values)
    
    print(f"✅ Scaler correto criado:")
    print(f"   Means: {scaler.mean_}")
    print(f"   Scales: {scaler.scale_}")
    
    # 4. Aplica correção em TODOS os modelos
    model_dir = "models/hac_v6"
    fixed_count = 0
    
    print(f"\n🔧 Aplicando correção em: {model_dir}")
    
    for folder in os.listdir(model_dir):
        folder_path = os.path.join(model_dir, folder)
        if os.path.isdir(folder_path):
            scaler_path = os.path.join(folder_path, "scaler_Y.pkl")
            
            # Salva o scaler CORRETO
            joblib.dump(scaler, scaler_path)
            fixed_count += 1
            print(f"   ✅ {folder}: scaler corrigido")
    
    print(f"\n🎯 {fixed_count} scalers Y corrigidos com sucesso!")
    
    # 5. Teste rápido
    print(f"\n🧪 TESTE RÁPIDO:")
    test_values = np.array([[1.0, 1.0, 1.0]])
    unscaled = scaler.inverse_transform(test_values)[0]
    print(f"   [1.0, 1.0, 1.0] → {unscaled}")
    
    print(f"\n🔥 PRÓXIMOS PASSOS:")
    print("python3 scripts/save_report.py")
    print("cat results/model_report.json")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
