#!/usr/bin/env python3
"""
download_omni_1min.py

Baixa dados históricos OMNI 1-minuto diretamente da NASA CDAWeb
e salva em CSV consolidado.

Campos baixados:
- Tempo
- Bz GSM
- By GSM
- Bt
- Velocidade solar
- Densidade
- Temperatura

Requer:
pip install cdasws pandas tqdm
"""

from cdasws import CdasWs
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

START_DATE = "2000-01-01T00:00:00Z"
END_DATE   = "2020-12-31T23:59:59Z"

OUTPUT_FILE = "omni_1min_2000_2020.csv"

# Dataset OMNI 1-min
DATASET = "OMNI_HRO_1MIN"

# ============================================================
# VARIÁVEIS
# ============================================================

VARIABLES = [
    "BX_GSE",
    "BY_GSM",
    "BZ_GSM",
    "F",            # |B|
    "flow_speed",
    "proton_density",
    "T"             # temperatura
]

# ============================================================
# DOWNLOAD
# ============================================================

cdas = CdasWs()

print("=" * 70)
print("🚀 BAIXANDO OMNI 1-MIN HISTÓRICO")
print("=" * 70)

print(f"\n📥 Dataset: {DATASET}")
print(f"📅 Período: {START_DATE} → {END_DATE}")

status, data = cdas.get_data(
    DATASET,
    VARIABLES,
    START_DATE,
    END_DATE
)

if not status:
    raise RuntimeError("Falha ao baixar dados da CDAWeb.")

print("\n✅ Download concluído")

# ============================================================
# CONVERSÃO
# ============================================================

print("\n⚙️ Convertendo para DataFrame...")

df = pd.DataFrame({
    "time_tag": pd.to_datetime(data["Epoch"]),
    "bx_gse": np.array(data["BX_GSE"]),
    "by_gsm": np.array(data["BY_GSM"]),
    "bz_gsm": np.array(data["BZ_GSM"]),
    "bt": np.array(data["F"]),
    "speed": np.array(data["flow_speed"]),
    "density": np.array(data["proton_density"]),
    "temperature": np.array(data["T"])
})

# ============================================================
# LIMPEZA
# ============================================================

print("🧹 Limpando valores inválidos...")

invalid_threshold = 99999

for col in [
    "bx_gse",
    "by_gsm",
    "bz_gsm",
    "bt",
    "speed",
    "density",
    "temperature"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[np.abs(df[col]) > invalid_threshold, col] = np.nan

# Remove duplicados
df = df.drop_duplicates(subset="time_tag")

# Ordena
df = df.sort_values("time_tag").reset_index(drop=True)

print(f"✅ Pontos válidos: {len(df)}")

# ============================================================
# SAVE
# ============================================================

print(f"\n💾 Salvando: {OUTPUT_FILE}")

df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Finalizado")
print("=" * 70)
