import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# 1. DOWNLOAD DOS DADOS DSCOVR (NOAA)
# ============================================================

PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"

def load_dscovr():
    plasma = pd.DataFrame(
        requests.get(PLASMA_URL).json()[1:],
        columns=["time", "density", "speed", "temperature"]
    )

    mag = pd.DataFrame(
        requests.get(MAG_URL).json()[1:],
        columns=["time", "bx", "by", "bz", "bt"]
    )

    plasma["time"] = pd.to_datetime(plasma["time"])
    mag["time"] = pd.to_datetime(mag["time"])

    df = pd.merge(plasma, mag, on="time", how="inner")

    for col in ["density", "speed", "temperature", "bx", "by", "bz", "bt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


# ============================================================
# 2. MODELO HAC (IGUAL AO DO PAPER)
# ============================================================

class HACModel:
    def __init__(self):
        self.alpha_L = 1.5e-5
        self.beta_L = 0.12
        self.alpha_R = 0.1
        self.beta_R = 0.2
        self.eta = 0.35
        self.theta = 2.8
        self.sigma = 0.4

        self.gamma = 0.8
        self.delta = 1.2

        self.v0 = 400
        self.n0 = 5

    def phi(self, Bz, Bt, v, n):
        return n * v**2 * Bt**2 + v * Bt**2 * (Bz < 0)

    def run(self, df):
        N = len(df)
        H_load = np.zeros(N)
        H_rel = np.zeros(N)
        HCI = np.zeros(N)

        phi = self.phi(df["bz"], df["bt"], df["speed"], df["density"])

        for i in range(1, N):
            Hsat = 1 / (1 + np.exp(-(H_load[i-1] - self.theta) / self.sigma))

            dH_load = self.alpha_L * phi[i] * (1 - Hsat) - self.beta_L * H_load[i-1]
            dH_rel = self.alpha_R * H_load[i-1] - self.beta_R * H_rel[i-1] + self.eta * Hsat

            H_load[i] = np.clip(H_load[i-1] + dH_load, 0, 5)
            H_rel[i] = np.clip(H_rel[i-1] + dH_rel, 0, 3)

            C = (np.maximum(-df["bz"][i], 0) / (df["bt"][i] + 1e-6)) \
                * (df["speed"][i] / self.v0) \
                * (df["density"][i] / self.n0)

            HCI[i] = self.gamma * C + self.delta * H_load[i]

        df["H_load"] = H_load
        df["H_rel"] = H_rel
        df["HCI"] = HCI

        return df


# ============================================================
# 3. CLASSIFICAÇÃO DE REGIME
# ============================================================

def classify_hac(hci):
    if hci < 1.2:
        return "Quiet"
    elif hci < 2.0:
        return "Active"
    elif hci < 2.8:
        return "Storm"
    else:
        return "Severe (G4–G5)"


# ============================================================
# 4. EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    print("\n📡 Baixando dados DSCOVR...")
    df = load_dscovr()

    print("Último registro:", df.iloc[-1]["time"])

    hac = HACModel()
    df = hac.run(df)

    last = df.iloc[-1]

    regime = classify_hac(last["HCI"])

    print("\n========== HAC FORECAST ==========")
    print(f"Tempo: {last['time']}")
    print(f"Velocidade: {last['speed']:.1f} km/s")
    print(f"Bz: {last['bz']:.1f} nT")
    print(f"Bt: {last['bt']:.1f} nT")
    print(f"Densidade: {last['density']:.1f} cm⁻³")
    print(f"HCI: {last['HCI']:.2f}")
    print(f"⚠️ REGIME: {regime}")
    print("=================================")

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(df["time"], df["HCI"], label="HCI")
    plt.axhline(2.8, color="r", linestyle="--", label="Severe Threshold")
    plt.axhline(2.0, color="orange", linestyle="--", label="Storm")
    plt.legend()
    plt.title("HAC – Estado Magnetosférico (Tempo Real)")
    plt.xlabel("Time")
    plt.ylabel("HCI")
    plt.grid()
    plt.tight_layout()
    plt.show()