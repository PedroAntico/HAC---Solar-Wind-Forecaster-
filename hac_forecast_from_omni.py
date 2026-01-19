import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================
# CONFIG
# =============================
MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

# =============================
# UTIL
# =============================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def find_time_column(df):
    for c in df.columns:
        if "time" in c.lower() or "epoch" in c.lower():
            return c
    raise RuntimeError("❌ Nenhuma coluna de tempo encontrada!")

def build_dataframe(mag, plasma):
    df_mag = pd.DataFrame(mag)
    df_plasma = pd.DataFrame(plasma)

    # Detecta automaticamente coluna de tempo
    tmag = find_time_column(df_mag)
    tplasma = find_time_column(df_plasma)

    df_mag["time"] = pd.to_datetime(df_mag[tmag])
    df_plasma["time"] = pd.to_datetime(df_plasma[tplasma])

    df = pd.merge(df_mag, df_plasma, on="time", how="inner")

    return df

def compute_hac(df):
    Bz = df["bz_gsm"].astype(float) * 1e-9
    V  = df["speed"].astype(float) * 1e3
    n  = df["density"].astype(float) * 1e6

    df["HAC"] = np.abs(Bz) * V * np.sqrt(n)
    return df

def classify(h):
    if h < 2e-5:
        return "Quiet"
    elif h < 6e-5:
        return "G1–G2"
    elif h < 1.2e-4:
        return "G3"
    elif h < 2e-4:
        return "G4"
    else:
        return "G5"

# =============================
# MAIN
# =============================

print("📥 Carregando dados...")
mag = load_json(MAG_FILE)
plasma = load_json(PLASMA_FILE)

print("🔧 Construindo dataframe...")
df = build_dataframe(mag, plasma)

print("⚡ Calculando HAC...")
df = compute_hac(df)

h_last = df["HAC"].iloc[-1]
storm = classify(h_last)

print("\n==============================")
print(f"📊 HAC atual: {h_last:.3e}")
print(f"🌍 Previsão: {storm}")
print("==============================\n")

# =============================
# PLOT
# =============================
plt.figure(figsize=(12,5))
plt.plot(df["time"], df["HAC"], label="HAC", color="red")

plt.axhline(6e-5, ls="--", c="orange", label="G3")
plt.axhline(1.2e-4, ls="--", c="darkorange", label="G4")
plt.axhline(2e-4, ls="--", c="darkred", label="G5")

plt.title("HAC – Helio-Geoeffective Accumulation")
plt.xlabel("UTC Time")
plt.ylabel("HAC")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("hac_forecast.png", dpi=150)

print("📈 Gráfico salvo: hac_forecast.png")
