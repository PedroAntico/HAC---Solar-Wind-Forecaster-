import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

# ============================
# LOADERS
# ============================

def load_json_table(path):
    with open(path, "r") as f:
        raw = json.load(f)

    header = raw[0]
    data = raw[1:]

    df = pd.DataFrame(data, columns=header)
    return df


# ============================
# PROCESSAMENTO
# ============================

def build_dataframe(mag, plasma):
    df = pd.merge(mag, plasma, on="time_tag", how="inner")

    # Conversões numéricas
    for col in df.columns:
        if col != "time_tag":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tempo
    df["time"] = pd.to_datetime(df["time_tag"])

    return df


def compute_hac(df):
    Bz = df["bz_gsm"].values * 1e-9      # nT → T
    V  = df["speed"].values * 1e3        # km/s → m/s
    n  = df["density"].values * 1e6      # cm⁻³ → m⁻³

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


# ============================
# MAIN
# ============================

print("📥 Carregando dados OMNI...")
mag = load_json_table(MAG_FILE)
plasma = load_json_table(PLASMA_FILE)

print("🔧 Construindo dataframe...")
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

# ============================
# GRÁFICO
# ============================

plt.figure(figsize=(12,5))
plt.plot(df["time"], df["HAC"], color="red", lw=2)

plt.axhline(6e-5, ls="--", c="orange", label="G3")
plt.axhline(1.2e-4, ls="--", c="darkorange", label="G4")
plt.axhline(2e-4, ls="--", c="darkred", label="G5")

plt.legend()
plt.title("HAC — Heliospheric Accumulated Coupling")
plt.xlabel("Time (UTC)")
plt.ylabel("HAC")
plt.grid(True)
plt.tight_layout()
plt.savefig("hac_forecast.png", dpi=150)

print("📈 Gráfico salvo como hac_forecast.png")
