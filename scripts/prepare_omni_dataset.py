#!/usr/bin/env python3
"""
prepare_omni_dataset.py - CORRIGIDO: Preserva variabilidade da velocidade
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração
DATA_DIR = "data"
INPUT_CSV = f"{DATA_DIR}/omni_5min_1998_2024.csv"
OUTPUT_CSV = f"{DATA_DIR}/omni_prepared.csv"

print("=" * 70)
print("🔄 PREPARANDO DATASET OMNI - COM CORREÇÃO DA VELOCIDADE")
print("=" * 70)

# 1. Carregar dados
print(f"\n📂 Carregando dados de: {INPUT_CSV}")
try:
    df = pd.read_csv(INPUT_CSV, low_memory=True)
    print(f"   ✅ Carregado: {len(df)} linhas, {len(df.columns)} colunas")
except FileNotFoundError:
    print(f"❌ Arquivo não encontrado: {INPUT_CSV}")
    print("   Execute primeiro: python convert_omni_txt.py")
    exit(1)

# 2. Verificar colunas necessárias
required_columns = ['speed', 'bz_gsm', 'density']
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print(f"❌ Colunas faltando: {missing_cols}")
    print(f"   Colunas disponíveis: {list(df.columns)[:20]}...")
    exit(1)

print("\n📊 Estatísticas INICIAIS:")
for col in required_columns:
    if col in df.columns:
        values = df[col].dropna()
        print(f"   {col}:")
        print(f"     • Não-nulos: {len(values)}")
        print(f"     • Média: {values.mean():.2f}")
        print(f"     • Std: {values.std():.2f}")
        print(f"     • Min: {values.min():.1f}")
        print(f"     • Max: {values.max():.1f}")
        
        if values.std() < 0.1:
            print(f"     ⚠️  ATENÇÃO: std muito baixo ({values.std():.6f})")

# 3. Filtrar dados (remover linhas com valores inválidos)
print(f"\n🧹 Filtrando dados...")
initial_rows = len(df)

# Substituir marcadores de dados faltantes (999.99, 9999.99, etc.)
df.replace([999.99, 9999.99, 1000.00, 1e31], np.nan, inplace=True)

# Remover linhas com NaN nas colunas essenciais
df = df.dropna(subset=required_columns)

# Remover outliers extremos
def remove_extreme_outliers(series, n_std=5):
    """Remove outliers extremos mantendo a variabilidade."""
    mean = series.mean()
    std = series.std()
    
    # Limites razoáveis para dados OMNI
    if 'speed' in series.name:
        # Velocidade do vento solar: 200-1000 km/s são valores normais
        lower_bound = 200
        upper_bound = 1000
    elif 'density' in series.name:
        # Densidade: 0.1-100 cm^-3 são valores normais
        lower_bound = 0.1
        upper_bound = 100
    elif 'bz' in series.name:
        # Bz: -50 a +50 nT são valores normais
        lower_bound = -50
        upper_bound = 50
    else:
        # Método estatístico padrão
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
    
    return series.between(lower_bound, upper_bound)

print("   Removendo outliers extremos...")
for col in required_columns:
    if col in df.columns:
        mask = remove_extreme_outliers(df[col], n_std=5)
        df = df[mask]
        removed = (~mask).sum()
        if removed > 0:
            print(f"     {col}: removidos {removed} outliers")

final_rows = len(df)
print(f"   ✅ Linhas removidas: {initial_rows - final_rows}")
print(f"   ✅ Linhas restantes: {final_rows}")

# 4. Normalização CORRIGIDA - NÃO normalizar agora!
print(f"\n⚠️  ATENÇÃO: NORMALIZAÇÃO SERÁ FEITA DURANTE FEATURE ENGINEERING")
print("   Preservando valores ORIGINAIS no CSV preparado")

# Apenas salvar os dados limpos, sem normalização
df_prepared = df[required_columns].copy()

# 5. Salvar dataset
print(f"\n💾 Salvando dataset preparado: {OUTPUT_CSV}")
df_prepared.to_csv(OUTPUT_CSV, index=False)

print(f"   ✅ Dataset salvo: {len(df_prepared)} linhas")

# 6. Verificação final
print("\n" + "=" * 70)
print("✅ VERIFICAÇÃO FINAL DO DATASET PREPARADO")
print("=" * 70)

print("\n📊 Estatísticas FINAIS:")
for col in required_columns:
    if col in df_prepared.columns:
        values = df_prepared[col].dropna()
        print(f"\n   {col}:")
        print(f"     • Média: {values.mean():.2f}")
        print(f"     • Std: {values.std():.2f}")
        print(f"     • Min: {values.min():.1f}")
        print(f"     • Max: {values.max():.1f}")
        print(f"     • Únicos: {values.nunique()}")
        
        if values.std() > 1.0:
            print(f"     ✅ Variação PRESERVADA (std={values.std():.2f})")
        else:
            print(f"     ⚠️  CUIDADO: Baixa variação (std={values.std():.6f})")

# 7. Amostra dos dados
print("\n👀 Amostra dos dados (primeiras 5 linhas):")
print(df_prepared.head())

print("\n" + "=" * 70)
print("🎯 PRÓXIMOS PASSOS:")
print("=" * 70)
print("""
1. Execute: python hac_v6_features.py
2. A feature engineering fará a normalização apropriada
3. A velocidade deve manter sua variabilidade

Se a velocidade ainda estiver constante:
- Verifique o arquivo de entrada omni_5min_1998_2024.csv
- Execute o script de diagnóstico: python diagnose_speed_problem.py
""")
