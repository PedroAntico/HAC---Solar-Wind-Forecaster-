import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# ============================
# CONFIGURAÇÃO
# ============================

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

DT = 60  # segundos
BETA = 0.65

# Fator empírico calibrado (ESSENCIAL)
GAIN = 5e13

THRESHOLDS = {
    "Quiet": 0,
    "G1": 50,
    "G2": 100,
    "G3": 160,
    "G4": 220,
    "G5": 300
}

# ============================
# LOADERS
# ============================

def load_json_table(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return pd.DataFrame(raw[1:], columns=raw[0])


# ============================
# PROCESSAMENTO
# ============================

def build_dataframe(mag, plasma):
    df = pd.merge(mag, plasma, on="time_tag", how="inner")

    for col in df.columns:
        if col != "time_tag":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["time_tag"])
    return df.sort_values("time").reset_index(drop=True)


def compute_hac(df):
    Bz = df["bz_gsm"].values * 1e-9
    Bx = df["bx_gsm"].values * 1e-9
    By = df["by_gsm"].values * 1e-9
    V  = df["speed"].values * 1e3
    n  = df["density"].values * 1e6

    B = np.sqrt(Bx**2 + By**2 + Bz**2)

    south = np.where(Bz < 0, 1.0, 0.0)

    coupling = (n * V**2)**BETA * B**2
    hac = np.cumsum(coupling * south * DT)

    # 🔥 NORMALIZAÇÃO CRÍTICA
    hac = hac * GAIN

    # Saturação física
    hac = np.clip(hac, 0, 500)

    df["HAC"] = hac
    return df


def classify(h):
    if h < THRESHOLDS["G1"]:
        return "Quiet"
    elif h < THRESHOLDS["G2"]:
        return "G1"
    elif h < THRESHOLDS["G3"]:
        return "G2"
    elif h < THRESHOLDS["G4"]:
        return "G3"
    elif h < THRESHOLDS["G5"]:
        return "G4"
    else:
        return "G5"


# ============================
# PLOT
# ============================

def plot_hac(df):
    plt.figure(figsize=(14,6))

    plt.plot(df["time"], df["HAC"], color="red", lw=2, label="HAC")

    for k, v in THRESHOLDS.items():
        if k != "Quiet":
            plt.axhline(v, linestyle="--", alpha=0.6, label=k)

    plt.title("Heliospheric Accumulated Coupling (HAC)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("HAC Index")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("hac_forecast.png", dpi=300)


# ============================
# MAIN
# ============================

def main():
    print("📥 Loading data...")
    mag = load_json_table(MAG_FILE)
    plasma = load_json_table(PLASMA_FILE)

    print("⚙️ Processing...")
    df = build_dataframe(mag, plasma)
    df = compute_hac(df)

    last = df.iloc[-1]
    level = classify(last["HAC"])

    print("\n==============================")
    print(f"🕒 {last['time']}")
    print(f"⚡ HAC = {last['HAC']:.1f}")
    print(f"🌍 Storm Level = {level}")
    print("==============================\n")

    plot_hac(df)
    print("📈 Plot saved as hac_forecast.png")


if __name__ == "__main__":
    main()
