import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ===============================
# CONFIGURAÇÕES
# ===============================
MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

OUTPUT_FIG = "hac_forecast.png"

# Constantes físicas
MU0 = 4 * np.pi * 1e-7
MP = 1.6726e-27  # massa do próton (kg)

# ===============================
# FUNÇÕES
# ===============================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_time(t):
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")

def build_dataframe(mag, plasma):
    df_mag = pd.DataFrame(mag)
    df_plasma = pd.DataFrame(plasma)

    df_mag["time"] = pd.to_datetime(df_mag["time"])
    df_plasma["time"] = pd.to_datetime(df_plasma["time"])

    df = pd.merge(df_mag, df_plasma, on="time", how="inner")

    return df

def compute_hac(df):
    """
    HAC ~ |Bz| * V * sqrt(n)
    Versão física simplificada, estável e reprodutível
    """

    Bz = df["bz_gsm"].astype(float) * 1e-9  # nT → Tesla
    V  = df["speed"].astype(float) * 1e3   # km/s → m/s
    n  = df["density"].astype(float) * 1e6 # cm^-3 → m^-3

    hac = np.abs(Bz) * V * np.sqrt(n)

    df["HAC"] = hac
    return df

def classify_storm(hac):
    if hac < 2e-5:
        return "Quiet"
    elif hac < 6e-5:
        return "G1–G2"
    elif hac < 1.2e-4:
        return "G3"
    elif hac < 2.0e-4:
        return "G4"
    else:
        return "G5"

# ===============================
# MAIN
# ===============================

print("🔹 Carregando dados OMNI...")
mag = load_json(MAG_FILE)
plasma = load_json(PLASMA_FILE)

print("🔹 Construindo dataframe...")
df = build_dataframe(mag, plasma)

print("🔹 Calculando HAC...")
df = compute_hac(df)

# Classificação final
last_hac = df["HAC"].iloc[-1]
storm_level = classify_storm(last_hac)

print(f"\n🌐 HAC atual: {last_hac:.2e}")
print(f"⚠️ Classificação prevista: {storm_level}")

# ===============================
# GRÁFICO
# ===============================

plt.figure(figsize=(12,6))

plt.plot(df["time"], df["HAC"], color="red", label="HAC")
plt.axhline(6e-5, color="orange", linestyle="--", label="G3")
plt.axhline(1.2e-4, color="darkorange", linestyle="--", label="G4")
plt.axhline(2e-4, color="darkred", linestyle="--", label="G5")

plt.title("HAC – Helio-geoeffective Accumulation")
plt.xlabel("Time (UTC)")
plt.ylabel("HAC (normalized)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=150)

print(f"\n📈 Gráfico salvo em: {OUTPUT_FIG}")
