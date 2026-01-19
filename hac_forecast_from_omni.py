import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def build_dataframe(mag, plasma):
    # OMNI formato padrão
    cols = [
        "year", "doy", "hour", "minute",
        "bt", "bz", "speed", "density"
    ]

    mag_df = pd.DataFrame(mag)
    plasma_df = pd.DataFrame(plasma)

    df = pd.concat([mag_df, plasma_df], axis=1)
    df = df.iloc[:, :8]
    df.columns = cols

    # Cria timestamp real
    df["time"] = df.apply(
        lambda r: datetime(int(r.year), 1, 1)
        + timedelta(days=int(r.doy)-1,
                    hours=int(r.hour),
                    minutes=int(r.minute)),
        axis=1
    )

    return df

def compute_hac(df):
    Bz = df["bz"].astype(float) * 1e-9
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

# ==========================
# MAIN
# ==========================

print("📥 Carregando dados OMNI...")
mag = load_json(MAG_FILE)
plasma = load_json(PLASMA_FILE)

print("🧩 Construindo dataframe...")
df = build_dataframe(mag, plasma)

print("⚡ Calculando HAC...")
df = compute_hac(df)

last = df.iloc[-1]
level = classify(last["HAC"])

print("\n==============================")
print(f"📅 {last['time']}")
print(f"⚡ HAC = {last['HAC']:.3e}")
print(f"🌍 Nível previsto = {level}")
print("==============================\n")

# Plot
plt.figure(figsize=(12,5))
plt.plot(df["time"], df["HAC"], color="red")
plt.axhline(6e-5, ls="--", c="orange", label="G3")
plt.axhline(1.2e-4, ls="--", c="darkorange", label="G4")
plt.axhline(2e-4, ls="--", c="darkred", label="G5")
plt.legend()
plt.title("HAC – Helio-Geoeffective Accumulation")
plt.xlabel("Time (UTC)")
plt.ylabel("HAC")
plt.grid()
plt.tight_layout()
plt.savefig("hac_forecast.png", dpi=150)

print("📈 Gráfico salvo: hac_forecast.png")
