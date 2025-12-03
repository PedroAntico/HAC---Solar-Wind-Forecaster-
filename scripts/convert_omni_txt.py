#!/usr/bin/env python3
"""
convert_omni_txt.py - CORRIGIDO: Leitura correta da velocidade
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("📄 CONVERTENDO ARQUIVO TXT DO OMNI PARA CSV")
print("=" * 70)

# Caminhos dos arquivos
txt_path = "data/omni_5min_1998_2024.txt"
csv_path = "data/omni_5min_1998_2024.csv"

# Verificar se o arquivo existe
import os
if not os.path.exists(txt_path):
    print(f"❌ Arquivo não encontrado: {txt_path}")
    exit(1)

print(f"\n📖 Lendo arquivo TXT: {txt_path}")

# Lê o arquivo com delimitador de espaços em branco
try:
    df = pd.read_csv(txt_path, delim_whitespace=True, header=None, low_memory=False)
    print(f"   ✅ Arquivo lido: {df.shape[0]} linhas, {df.shape[1]} colunas")
except Exception as e:
    print(f"❌ Erro ao ler arquivo: {e}")
    exit(1)

# Verificar número de colunas
n_cols = df.shape[1]
print(f"   📊 Número de colunas: {n_cols}")

# Para dados OMNI de 5 minutos, esperamos 47 colunas
if n_cols != 47:
    print(f"⚠️  Número inesperado de colunas. Esperado: 47, Encontrado: {n_cols}")

# Nomes das colunas para dados OMNI de 5 minutos (47 colunas)
# Baseado na documentação: https://omniweb.gsfc.nasa.gov/html/ow_data.html
column_names = [
    'year', 'doy', 'hour', 'minute', 'sc_id',
    'bx_gsm', 'by_gsm', 'bz_gsm',
    'theta', 'phi', 'sigma',
    'speed',        # Coluna 11: Velocidade do vento solar (km/s) ← ESTA É A IMPORTANTE!
    'density',      # Coluna 12: Densidade do vento solar (cm^-3)
    'temperature',  # Coluna 13: Temperatura (K)
    'pressure',     # Coluna 14: Pressão dinâmica (nPa)
    'e', 'beta', 'mach_num',
    'ae', 'al', 'au', 'sym_d', 'sym_h', 'asym_d', 'asym_h',
    'pc_n', 'pc_s',
    'f10.7', 'dst', 'imf_pt',
    'theta_gn', 'phi_gn',
    'bx_gse', 'by_gse', 'bz_gse',
    'theta_gse', 'phi_gse',
    'bt', 'theta_bt', 'phi_bt',
    'vx', 'vy', 'vz',
    'theta_v', 'phi_v'
]

# Usar apenas os nomes disponíveis
if n_cols <= len(column_names):
    df.columns = column_names[:n_cols]
else:
    print(f"⚠️  Arquivo tem mais colunas que o esperado")
    # Criar nomes genéricos
    df.columns = [f'col_{i}' for i in range(n_cols)]
    # Tentar mapear as colunas importantes
    if n_cols >= 13:
        # Assumir que as colunas estão nas posições padrão
        df = df.rename(columns={
            'col_11': 'speed',
            'col_12': 'density',
            'col_7': 'bz_gsm'  # Bz está na coluna 7 (índice 7)
        })

print("\n🔍 Verificando colunas importantes:")
for col in ['speed', 'density', 'bz_gsm']:
    if col in df.columns:
        # Substituir valores de placeholder por NaN
        placeholder_values = [999.99, 9999.99, 1000.00, 1e31]
        df[col] = df[col].replace(placeholder_values, np.nan)
        
        # Converter para float
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Estatísticas
        non_null = df[col].notna().sum()
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        print(f"   {col}:")
        print(f"     • Não-nulos: {non_null}")
        print(f"     • Média: {mean_val:.2f}")
        print(f"     • Std: {std_val:.2f}")
        
        if std_val < 0.1:
            print(f"     ⚠️  ATENÇÃO: {col} tem std muito baixo!")
        else:
            print(f"     ✅ OK")
    else:
        print(f"   ❌ {col} não encontrada nas colunas")

# Selecionar apenas as colunas que precisamos
output_columns = ['year', 'doy', 'hour', 'minute']
for col in ['speed', 'bz_gsm', 'density']:
    if col in df.columns:
        output_columns.append(col)

df_output = df[output_columns].copy()

# Salvar como CSV
print(f"\n💾 Salvando como CSV: {csv_path}")
df_output.to_csv(csv_path, index=False)

print(f"   ✅ CSV salvo: {len(df_output)} linhas, {len(df_output.columns)} colunas")

# Verificação final
print("\n" + "=" * 70)
print("✅ VERIFICAÇÃO FINAL")
print("=" * 70)

print("\n📊 Amostra dos dados convertidos:")
print(df_output[['speed', 'bz_gsm', 'density']].head(10))

print("\n🎯 Próximo passo:")
print("   Execute: python prepare_omni_dataset.py")
