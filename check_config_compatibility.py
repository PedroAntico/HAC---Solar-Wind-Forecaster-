#!/usr/bin/env python3
"""
check_config_compatibility.py
Verifica se config.yaml é compatível com seus dados.
"""
import yaml
import pandas as pd
import os

def check_compatibility():
    # 1. Carregar config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    targets = config["targets"]["primary"]
    print(f"🎯 Targets no config: {targets}")
    
    # 2. Verificar arquivo de dados
    data_dir = config["paths"]["data_dir"]
    csv_path = os.path.join(data_dir, "omni_prepared.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ Arquivo não encontrado: {csv_path}")
        # Buscar alternativas
        for alt in ["data_real", "data", "."]:
            alt_path = os.path.join(alt, "omni_prepared.csv")
            if os.path.exists(alt_path):
                print(f"✅ Encontrado em: {alt_path}")
                csv_path = alt_path
                break
    
    # 3. Verificar colunas
    try:
        df = pd.read_csv(csv_path, nrows=5)  # Apenas cabeçalho
        available_cols = df.columns.tolist()
        
        print(f"\n📊 Colunas disponíveis no CSV (primeiras 10):")
        print(f"   {available_cols[:10]}")
        
        # Verificar correspondência
        missing = [t for t in targets if t not in available_cols]
        
        if missing:
            print(f"\n❌ FALTANDO colunas no CSV: {missing}")
            print(f"\n💡 Soluções possíveis:")
            print(f"   1. Renomeie colunas no CSV para: {targets}")
            print(f"   2. Ajuste targets no config.yaml para: {available_cols[:3]}")
        else:
            print(f"\n✅ TODAS as colunas do config estão no CSV!")
            
    except Exception as e:
        print(f"❌ Erro ao ler CSV: {e}")

if __name__ == "__main__":
    check_compatibility()
