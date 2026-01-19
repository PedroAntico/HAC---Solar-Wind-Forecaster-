# ============================================================
# HAC REAL-TIME SOLAR WIND FORECASTER
# Author: Pedro Antico
# Version: Stable / GitHub Ready
# ============================================================

import numpy as np
import pandas as pd
import requests
from datetime import datetime

# ============================================================
# 1. DOWNLOAD DOS DADOS DSCOVR (NOAA)
# ============================================================

def load_dscovr_plasma():
    url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-hour.json"
    data = requests.get(url).json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"])

    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["density"] = pd.to_numeric(df["density"], errors="coerce")

    return df[["time_tag", "speed", "density"]]


def load_dscovr_mag():
    url = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-hour.json"
    data = requests.get(url).json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"])

    df["bz_gsm"] = pd.to_numeric(df["bz_gsm"], errors="coerce")
    df["bt"] = pd.to_numeric(df["bt"], errors="coerce")

    return df[["time_tag", "bz_gsm", "bt"]]


# ============================================================
# 2. MODELO HAC (VERSÃO FINAL DO PAPER)
# ============================================================

class HAC:
    def __init__(self):
        self.alpha_L = 1.5e-5
        self.beta_L = 0.12
        self.alpha_R = 0.10
        self.beta_R = 0.20
        self.gamma = 0.8
        self.delta = 1.2
        self.theta = 2.8
        self.sigma = 0.4
        self.v0 = 400
        self.n0 = 5

    def phi(self, Bz, Bt, v, n):
        Bz_south = np.maximum(-Bz, 0)
        return n * v**2 * Bt**2 + v * Bt**2 * (Bz_south > 0)

    def step(self, H, phi):
        sat = 1 / (1 + np.exp(-(H - self.theta) / self.sigma))
        dH = self.alpha_L * phi * (1 - sat) - self.beta_L * H
        return np.clip(H + dH, 0, 10)

    def run(self, df):
        H = np.zeros(len(df))
        phi = self.phi(df.bz_gsm, df.bt, df.speed, df.density)

        for i in range(1, len(df)):
            H[i] = self.step(H[i-1], phi[i])

        HCI = self.gamma * phi + self.delta * H
        return H, HCI


# ============================================================
# 3. CLASSIFICAÇÃO DE TEMPESTADE
# ============================================================

def classify_storm(HCI):
    if HCI > 1.2e7:
        return "G5 EXTREMA"
    elif HCI > 7e6:
        return "G4 SEVERA"
    elif HCI > 4e6:
        return "G3 FORTE"
    elif HCI > 2e6:
        return "G2 MODERADA"
    elif HCI > 1e6:
        return "G1 FRACA"
    else:
        return "CALMO"


# ============================================================
# 4. EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("\n📡 Baixando dados DSCOVR...")
    plasma = load_dscovr_plasma()
    mag = load_dscovr_mag()

    df = pd.merge(plasma, mag, on="time_tag", how="inner")
    df = df.dropna()

    print(f"✅ Dados carregados: {len(df)} pontos")
    print(f"🕒 Última medição: {df.time_tag.iloc[-1]}")

    hac = HAC()
    H, HCI = hac.run(df)

    current_HCI = HCI[-1]
    level = classify_storm(current_HCI)

    print("\n===============================")
    print("🌎 HAC — ESTADO ATUAL")
    print("===============================")
    print(f"HCI atual : {current_HCI:,.2e}")
    print(f"Nível     : {level}")
    print(f"Bz atual  : {df.bz_gsm.iloc[-1]:.2f} nT")
    print(f"Vento     : {df.speed.iloc[-1]:.1f} km/s")
    print(f"Densidade : {df.density.iloc[-1]:.2f} cm⁻³")

    print("\n📌 Interpretação:")
    if "G4" in level or "G5" in level:
        print("⚠️ Condições severas — risco alto de tempestade geomagnética.")
    elif "G3" in level:
        print("⚠️ Tempestade significativa em curso.")
    else:
        print("🟢 Condições moderadas / estáveis.")
