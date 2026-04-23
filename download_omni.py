#!/usr/bin/env python3
"""
Download automático do dataset OMNI (5 min)
Pronto para HAC v6
"""

import os
import requests
import pandas as pd
from datetime import datetime

OUTPUT_PATH = "data/omni_prepared.csv"

def download_omni():
    print("=" * 60)
    print("🌌 BAIXANDO DADOS OMNI (NASA)")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)

    # Fonte OMNI (NOAA/NASA JSON)
    url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"

    print(f"📡 Baixando: {url}")
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise Exception(f"Erro download: HTTP {response.status_code}")

    data = response.json()

    # Primeira linha = header
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)

    print("🔧 Limpando dados...")

    # Converter tipos
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["density"] = pd.to_numeric(df["density"], errors="coerce")

    # Alguns datasets usam bz ou bz_gsm
    if "bz_gsm" in df.columns:
        df["bz_gsm"] = pd.to_numeric(df["bz_gsm"], errors="coerce")
    elif "bz" in df.columns:
        df["bz_gsm"] = pd.to_numeric(df["bz"], errors="coerce")
    else:
        print("⚠️ Bz não encontrado, preenchendo com 0")
        df["bz_gsm"] = 0

    # Selecionar colunas essenciais
    df = df[["time_tag", "speed", "density", "bz_gsm"]]

    # Renomear para padrão HAC
    df.rename(columns={
        "time_tag": "datetime"
    }, inplace=True)

    # Remover NaNs
    before = len(df)
    df = df.dropna()
    after = len(df)

    print(f"🧹 Removidos {before - after} valores inválidos")

    # Ordenar
    df = df.sort_values("datetime")

    # Salvar
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Dataset salvo em: {OUTPUT_PATH}")
    print(f"📊 Linhas: {len(df)}")
    print(f"🕒 Período: {df['datetime'].min()} → {df['datetime'].max()}")

    return df


if __name__ == "__main__":
    try:
        df = download_omni()

        print("\n🚀 PRÓXIMO PASSO:")
        print("python hac_v6_features.py")

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
