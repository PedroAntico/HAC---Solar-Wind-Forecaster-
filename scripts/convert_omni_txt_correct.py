#!/usr/bin/env python3
"""
convert_omni_txt_correct.py - Conversão CORRETA para o formato específico
Formato: ano, doy, hora, Bt, Bx, By, Bz, temperatura, densidade, velocidade, pressão
"""

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("📄 CONVERTENDO ARQUIVO TXT DO OMNI - FORMATO CORRETO")
print("=" * 70)

# Caminhos dos arquivos
txt_path = "data/omni2_of3LE00pQF.txt"
csv_path = "data/omni_5min_1998_2024.csv"

if not os.path.exists(txt_path):
    print(f"❌ Arquivo não encontrado: {txt_path}")
    print("   Coloque o arquivo em: data/omni2_of3LE00pQF.txt")
    exit(1)

print(f"\n📖 Lendo arquivo TXT: {txt_path}")

# Ler o arquivo - formato tem 11 colunas
try:
    df = pd.read_csv(txt_path, delim_whitespace=True, header=None, low_memory=False)
    print(f"   ✅ Arquivo lido: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Verificar número de colunas
    if df.shape[1] == 11:
        print("   ✅ Formato correto: 11 colunas detectadas")
        # Atribuir nomes CORRETOS baseado na sua descrição
        df.columns = [
            'year',      # 1. ano
            'doy',       # 2. dia do ano
            'hour',      # 3. hora
            'bt',        # 4. |B| ou Bt (intensidade do campo)
            'bx',        # 5. componente Bx
            'by',        # 6. componente By
            'bz_gsm',    # 7. componente Bz ← IMPORTANTE!
            'temperature', # 8. temperatura do plasma (K) - IGNORAR
            'density',   # 9. densidade (n/cm³) ← IMPORTANTE!
            'speed',     # 10. velocidade (km/s) ← IMPORTANTE!
            'pressure'   # 11. pressão dinâmica (nPa)
        ]
        print("   ✅ Colunas mapeadas corretamente")
    else:
        print(f"   ⚠️  Formato inesperado: {df.shape[1]} colunas (esperado: 11)")
        print("   Mostrando primeiras linhas para debug:")
        print(df.head())
        exit(1)
        
except Exception as e:
    print(f"❌ Erro ao ler arquivo: {e}")
    exit(1)

# Verificar valores dos dados importantes
print("\n🔍 VERIFICANDO DADOS CRÍTICOS:")

# 1. Bz GSM
if 'bz_gsm' in df.columns:
    print("\n🧲 Bz GSM (componente z do campo magnético):")
    # Converter para numérico
    df['bz_gsm'] = pd.to_numeric(df['bz_gsm'], errors='coerce')
    
    # Estatísticas
    bz_non_null = df['bz_gsm'].notna().sum()
    bz_mean = df['bz_gsm'].mean()
    bz_std = df['bz_gsm'].std()
    bz_min = df['bz_gsm'].min()
    bz_max = df['bz_gsm'].max()
    
    print(f"   • Não-nulos: {bz_non_null}")
    print(f"   • Média: {bz_mean:.2f} nT")
    print(f"   • Std: {bz_std:.2f} nT")
    print(f"   • Min: {bz_min:.2f} nT")
    print(f"   • Max: {bz_max:.2f} nT")
    print(f"   • Únicos: {df['bz_gsm'].nunique()}")
    
    if bz_std < 0.1:
        print(f"   ⚠️  ATENÇÃO: Bz tem std muito baixo!")
    else:
        print(f"   ✅ Variação OK")

# 2. Densidade
if 'density' in df.columns:
    print("\n⚛️  Densidade do vento solar:")
    # Converter para numérico
    df['density'] = pd.to_numeric(df['density'], errors='coerce')
    
    # Estatísticas
    n_non_null = df['density'].notna().sum()
    n_mean = df['density'].mean()
    n_std = df['density'].std()
    n_min = df['density'].min()
    n_max = df['density'].max()
    
    print(f"   • Não-nulos: {n_non_null}")
    print(f"   • Média: {n_mean:.2f} cm^-3")
    print(f"   • Std: {n_std:.2f} cm^-3")
    print(f"   • Min: {n_min:.2f} cm^-3")
    print(f"   • Max: {n_max:.2f} cm^-3")
    print(f"   • Únicos: {df['density'].nunique()}")
    
    if n_std < 0.1:
        print(f"   ⚠️  ATENÇÃO: Densidade tem std muito baixo!")
    else:
        print(f"   ✅ Variação OK")

# 3. Velocidade (SPEED) - A VARIÁVEL PROBLEMÁTICA
if 'speed' in df.columns:
    print("\n🚀 Velocidade do vento solar:")
    # Converter para numérico
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    
    # Estatísticas
    v_non_null = df['speed'].notna().sum()
    v_mean = df['speed'].mean()
    v_std = df['speed'].std()
    v_min = df['speed'].min()
    v_max = df['speed'].max()
    
    print(f"   • Não-nulos: {v_non_null}")
    print(f"   • Média: {v_mean:.2f} km/s")
    print(f"   • Std: {v_std:.2f} km/s")
    print(f"   • Min: {v_min:.2f} km/s")
    print(f"   • Max: {v_max:.2f} km/s")
    print(f"   • Únicos: {df['speed'].nunique()}")
    
    if v_std < 10:
        print(f"   ⚠️  ATENÇÃO: Velocidade tem std muito baixo! ({v_std:.2f})")
        print(f"   🔍 Verificando se há problemas nos dados...")
        
        # Verificar se há muitos valores iguais
        value_counts = df['speed'].value_counts()
        top_values = value_counts.head(5)
        print(f"   Valores mais frequentes:")
        for val, count in top_values.items():
            print(f"     {val}: {count} ocorrências ({count/len(df)*100:.1f}%)")
    else:
        print(f"   ✅ Variação OK")

# Substituir valores inválidos por NaN
print("\n🧹 Substituindo valores inválidos por NaN...")
# Valores típicos de placeholder no OMNI
placeholder_values = [999.99, 9999.99, 1000.00, 1e31, -1e31]

for col in ['speed', 'bz_gsm', 'density']:
    if col in df.columns:
        df[col] = df[col].replace(placeholder_values, np.nan)
        print(f"   ✅ {col}: placeholders substituídos")

# Selecionar apenas as colunas que precisamos
output_columns = ['year', 'doy', 'hour', 'speed', 'bz_gsm', 'density']
df_output = df[output_columns].copy()

# Remover linhas com NaN nas colunas críticas
initial_len = len(df_output)
df_output = df_output.dropna(subset=['speed', 'bz_gsm', 'density'])
final_len = len(df_output)

print(f"\n🧼 Limpeza de dados:")
print(f"   • Linhas iniciais: {initial_len}")
print(f"   • Linhas removidas (NaN): {initial_len - final_len}")
print(f"   • Linhas finais: {final_len}")

# Verificação final após limpeza
print("\n📊 VERIFICAÇÃO FINAL APÓS LIMPEZA:")

for col in ['speed', 'bz_gsm', 'density']:
    if col in df_output.columns:
        values = df_output[col]
        print(f"\n   {col}:")
        print(f"     • Média: {values.mean():.2f}")
        print(f"     • Std: {values.std():.2f}")
        print(f"     • Min: {values.min():.2f}")
        print(f"     • Max: {values.max():.2f}")
        
        if col == 'speed' and values.std() < 50:
            print(f"     ⚠️  CUIDADO: Velocidade ainda tem baixa variação!")
        elif values.std() < 0.1:
            print(f"     ⚠️  CUIDADO: {col} tem baixa variação!")

# Salvar como CSV
print(f"\n💾 Salvando como CSV: {csv_path}")
df_output.to_csv(csv_path, index=False)
print(f"   ✅ CSV salvo: {len(df_output)} linhas, {len(df_output.columns)} colunas")

# Mostrar amostra dos dados
print("\n👀 Amostra dos dados convertidos (primeiras 10 linhas):")
print(df_output.head(10))

print("\n📈 Estatísticas completas:")
print(df_output[['speed', 'bz_gsm', 'density']].describe())

print("\n" + "=" * 70)
print("✅ CONVERSÃO COMPLETA!")
print("=" * 70)
print("\n🎯 Próximos passos:")
print("""
1. Execute: python prepare_omni_dataset.py
   (Apenas para limpeza adicional, SEM normalização)

2. Execute: python hac_v6_features.py
   (Para criar as features e datasets)

3. Se a velocidade ainda estiver constante:
   - Verifique se há um bug no prepare_omni_dataset.py
   - Execute o diagnóstico: python diagnose_speed_problem.py
""")

# Adicional: Verificar se há padrões suspeitos
print("\n🔍 VERIFICAÇÃO ADICIONAL DE PADRÕES:")

if 'speed' in df_output.columns:
    # Verificar se há valores repetidos em sequência
    speed_diff = df_output['speed'].diff().abs()
    consecutive_same = (speed_diff == 0).sum()
    print(f"   • Valores consecutivos iguais na velocidade: {consecutive_same}")
    
    if consecutive_same > len(df_output) * 0.1:  # Mais de 10% dos valores
        print(f"   ⚠️  MUITOS valores consecutivos iguais!")
    
    # Verificar distribuição
    print(f"   • Distribuição dos valores de velocidade:")
    print(f"     - < 300 km/s: {(df_output['speed'] < 300).sum()} linhas")
    print(f"     - 300-500 km/s: {((df_output['speed'] >= 300) & (df_output['speed'] < 500)).sum()} linhas")
    print(f"     - 500-800 km/s: {((df_output['speed'] >= 500) & (df_output['speed'] < 800)).sum()} linhas")
    print(f"     - > 800 km/s: {(df_output['speed'] >= 800).sum()} linhas")
