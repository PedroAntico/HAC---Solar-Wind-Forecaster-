#!/usr/bin/env python3
"""
prepare_omni_dataset_simple.py - Apenas copia o CSV, sem processamento
"""

import pandas as pd
import os
import shutil

print("=" * 70)
print("📋 PREPARANDO DATASET OMNI (APENAS CÓPIA)")
print("=" * 70)

# Configuração
DATA_DIR = "data"
INPUT_CSV = f"{DATA_DIR}/omni_5min_1998_2024.csv"
OUTPUT_CSV = f"{DATA_DIR}/omni_prepared.csv"

print(f"\n📂 Copiando: {INPUT_CSV}")
print(f"   Para: {OUTPUT_CSV}")

# Simplesmente copiar o arquivo
try:
    shutil.copyfile(INPUT_CSV, OUTPUT_CSV)
    print("   ✅ Arquivo copiado com sucesso")
    
    # Ler para verificação
    df = pd.read_csv(OUTPUT_CSV)
    print(f"   📊 Verificação:")
    print(f"     • Linhas: {len(df)}")
    print(f"     • Colunas: {list(df.columns)}")
    
    # Verificar estatísticas básicas
    for col in ['speed', 'bz_gsm', 'density']:
        if col in df.columns:
            values = df[col].dropna()
            print(f"\n     {col}:")
            print(f"       • Média: {values.mean():.2f}")
            print(f"       • Std: {values.std():.2f}")
            print(f"       • Min: {values.min():.2f}")
            print(f"       • Max: {values.max():.2f}")
            
except Exception as e:
    print(f"❌ Erro: {e}")

print("\n" + "=" * 70)
print("✅ PRONTO! Execute agora: python hac_v6_features.py")
print("=" * 70)
