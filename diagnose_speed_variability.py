#!/usr/bin/env python3
"""
diagnose_speed_variability.py

Diagnóstico passo a passo para encontrar onde a velocidade perde variabilidade.
"""

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("🔍 DIAGNÓSTICO DA VARIABILIDADE DA VELOCIDADE")
print("=" * 70)

# 1. Verificar arquivo CSV original
data_dir = "data"
csv_path = os.path.join(data_dir, "omni_prepared.csv")

print(f"\n1. Analisando {os.path.basename(csv_path)}:")
df = pd.read_csv(csv_path)

print(f"   • Shape: {df.shape}")
print(f"   • Colunas: {list(df.columns)}")

# Estatísticas da velocidade
if 'speed' in df.columns:
    speed_original = df['speed'].dropna()
    print(f"\n   📊 Velocidade no CSV original:")
    print(f"      • Média: {speed_original.mean():.2f}")
    print(f"      • Std: {speed_original.std():.6f}")
    print(f"      • Min: {speed_original.min():.2f}")
    print(f"      • Max: {speed_original.max():.2f}")
    print(f"      • Únicos: {speed_original.nunique()}")
    print(f"      • Quantiles:")
    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        print(f"        Q{q*100:2.0f}%: {speed_original.quantile(q):.2f}")
    
    # Verificar valores suspeitos
    print(f"\n   🔍 Valores suspeitos:")
    value_counts = speed_original.value_counts().head(10)
    for val, count in value_counts.items():
        print(f"      • {val}: {count} ocorrências ({count/len(speed_original)*100:.2f}%)")

# 2. Simular o que o HACFeatureBuilder faz
print(f"\n\n2. Simulando o processamento do HACFeatureBuilder...")

# Simular criação de lags
print(f"\n   🧪 Criando lags (como no _engineer_features_safe):")
for lag in [1, 3, 6]:
    lag_col = f"speed_lag{lag}"
    df[f"speed_lag{lag}"] = df['speed'].shift(lag)
    print(f"      • {lag_col}: {df[lag_col].std():.6f}")

# Simular médias móveis
print(f"\n   📈 Criando médias móveis:")
for window in [6, 12, 24]:
    ma_col = f"speed_ma_{window}"
    df[ma_col] = df['speed'].rolling(window=window, min_periods=window//2).mean()
    print(f"      • {ma_col}: {df[ma_col].std():.6f}")

# 3. Verificar se há problemas nos dados
print(f"\n\n3. Verificando problemas nos dados:")

# Verificar se há muitos valores iguais consecutivos
if 'speed' in df.columns:
    speed_diff = df['speed'].diff().abs()
    same_consecutive = (speed_diff == 0).sum()
    print(f"   • Valores consecutivos iguais: {same_consecutive} ({same_consecutive/len(df)*100:.1f}%)")
    
    # Verificar se há padrões de repetição
    print(f"\n   🔄 Padrões de repetição:")
    for window in [2, 3, 5]:
        # Verificar se há padrões repetidos
        pattern_count = 0
        for i in range(len(df) - window):
            if df['speed'].iloc[i:i+window].nunique() == 1:
                pattern_count += 1
        print(f"      • {window} valores iguais consecutivos: {pattern_count} vezes")

# 4. Verificar distribuição
print(f"\n\n4. Análise de distribuição:")
if 'speed' in df.columns:
    import matplotlib.pyplot as plt
    
    # Calcular estatísticas
    mean_speed = df['speed'].mean()
    std_speed = df['speed'].std()
    
    print(f"   • Intervalo normal: {mean_speed - std_speed:.1f} a {mean_speed + std_speed:.1f}")
    print(f"   • 68% dos dados estão nesse intervalo")
    
    # Verificar outliers
    Q1 = df['speed'].quantile(0.25)
    Q3 = df['speed'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['speed'] < lower_bound) | (df['speed'] > upper_bound)]
    print(f"   • Outliers (boxplot): {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# 5. Sugestões de correção
print(f"\n\n5. 🛠️  SUGESTÕES DE CORREÇÃO:")
print(f"   Se a velocidade tem std baixo mas valores razoáveis:")
print(f"   1. Verificar se há problemas na leitura do CSV")
print(f"   2. Testar diferentes períodos de tempo (talvez dados recentes)")
print(f"   3. Verificar se há filtros removendo muita variabilidade")
print(f"   4. Considerar usar log(speed) se a distribuição for muito assimétrica")

print(f"\n" + "=" * 70)
print("🎯 PRÓXIMOS PASSOS:")
print("=" * 70)
print("""
1. Execute este diagnóstico para ver a variabilidade real
2. Se necessário, ajuste o threshold de verificação
3. Execute: python hac_v6_features.py --debug
""")
