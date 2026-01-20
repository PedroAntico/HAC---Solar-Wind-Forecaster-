import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIGURAÇÃO FINAL
# ============================

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

DT = 60                     # segundos
BETA = 0.65
GAIN = 5e13                 # <<< ESSENCIAL
SATURATION = 500            # limite físico

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

def load_json(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return pd.DataFrame(raw[1:], columns=raw[0])


# ============================
# PROCESSAMENTO
# ============================

def prepare_data(mag, plasma):
    df = pd.merge(mag, plasma, on="time_tag", how="inner")

    for col in df.columns:
        if col != "time_tag":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["time_tag"])
    df = df.dropna(subset=["bz_gsm", "speed", "density"])

    return df.sort_values("time")


def compute_hac(df):
    Bz = df["bz_gsm"].values * 1e-9
    Bx = df.get("bx_gsm", 0).values * 1e-9
    By = df.get("by_gsm", 0).values * 1e-9

    V = df["speed"].values * 1e3
    n = df["density"].values * 1e6

    B = np.sqrt(Bx**2 + By**2 + Bz**2)

    south = (Bz < 0).astype(float)

    coupling = (n * V**2)**BETA * B**2
    hac = np.cumsum(coupling * south * DT)

    # 🔥 NORMALIZAÇÃO CORRETA
    hac = hac * GAIN
    hac = np.clip(hac, 0, SATURATION)

    df["HAC"] = hac
    return df


def classify(h):
    if h < 50: return "Quiet"
    if h < 100: return "G1"
    if h < 160: return "G2"
    if h < 220: return "G3"
    if h < 300: return "G4"
    return "G5"


# ============================
# PLOT
# ============================

def plot(df):
    plt.figure(figsize=(13,6))
    plt.plot(df["time"], df["HAC"], color="red", lw=2)

    for k, v in THRESHOLDS.items():
        if k != "Quiet":
            plt.axhline(v, linestyle="--", alpha=0.5, label=k)

    plt.title("Heliospheric Accumulated Coupling (HAC)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("HAC Index")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("hac_forecast.png", dpi=300)


# ============================
# MAIN
# ============================

def main():
    print("📥 Loading data...")
    mag = load_json(MAG_FILE)
    plasma = load_json(PLASMA_FILE)

    print("⚙️ Processing...")
    df = prepare_data(mag, plasma)
    df = compute_hac(df)

    last = df.iloc[-1]
    level = classify(last["HAC"])

    print("\n==============================")
    print(f"🕒 {last['time']}")
    print(f"⚡ HAC = {last['HAC']:.1f}")
    print(f"🌍 Level = {level}")
    print("==============================")

    plot(df)
    print("📈 Plot saved: hac_forecast.png")


if __name__ == "__main__":
    main()
