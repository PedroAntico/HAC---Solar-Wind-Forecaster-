import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

BETA = 0.6
DT = 60  # segundos

# ============================
# LOADERS
# ============================

def load_json_table(path):
    with open(path, "r") as f:
        raw = json.load(f)

    header = raw[0]
    data = raw[1:]
    return pd.DataFrame(data, columns=header)


# ============================
# BUILD DATAFRAME
# ============================

def build_dataframe(mag, plasma):
    df = pd.merge(mag, plasma, on="time_tag", how="inner")

    for col in df.columns:
        if col != "time_tag":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["time_tag"])
    df = df.sort_values("time")

    return df


# ============================
# HAC COMPUTATION
# ============================

def compute_hac(df, beta=BETA):
    Bz = pd.to_numeric(df["bz_gsm"], errors="coerce").values * 1e-9
    V  = pd.to_numeric(df["speed"], errors="coerce").values * 1e3
    n  = pd.to_numeric(df["density"], errors="coerce").values * 1e6

    # remove NaNs
    Bz = np.nan_to_num(Bz)
    V  = np.nan_to_num(V)
    n  = np.nan_to_num(n)

    # somente Bz sul contribui
    south = np.where(Bz < 0, 1.0, 0.0)

    coupling = (n * V**2)**beta * (Bz**2) * south
    hac = np.cumsum(coupling * DT)

    df["HAC"] = hac
    return df


# ============================
# CLASSIFICAÇÃO
# ============================

def classify_hac(h):
    if h < 2e1:
        return "Quiet"
    elif h < 6e1:
        return "G1–G2"
    elif h < 1.2e2:
        return "G3"
    elif h < 2.0e2:
        return "G4"
    else:
        return "G5"


# ============================
# MAIN
# ============================

print("📥 Carregando dados...")
mag = load_json_table(MAG_FILE)
plasma = load_json_table(PLASMA_FILE)

print("🔧 Processando dataframe...")
df = build_dataframe(mag, plasma)

print("⚡ Calculando HAC...")
df = compute_hac(df)

last = df.iloc[-1]
level = classify_hac(last["HAC"])

print("\n==============================")
print(f"🕒 {last['time']}")
print(f"⚡ HAC = {last['HAC']:.2f}")
print(f"🌍 Nível previsto = {level}")
print("==============================\n")

# ============================
# PLOT
# ============================

plt.figure(figsize=(12, 5))
plt.plot(df["time"], df["HAC"], color="red", lw=2, label="HAC")

plt.axhline(60,  ls="--", color="orange", label="G3")
plt.axhline(120, ls="--", color="darkorange", label="G4")
plt.axhline(200, ls="--", color="darkred", label="G5")

plt.title("HAC — Heliospheric Accumulated Coupling")
plt.xlabel("Time (UTC)")
plt.ylabel("HAC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hac_forecast.png", dpi=150)

print("📈 Gráfico salvo como hac_forecast.png")
