# ============================================================
# HAC REAL-TIME SOLAR WIND FORECASTER (ROBUST VERSION)
# Author: Pedro Antico
# ============================================================

import numpy as np
import pandas as pd
import requests
import time

# ============================================================
# UTIL
# ============================================================

def safe_get_json(url, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.text.strip().startswith("["):
                return r.json()
            else:
                print(f"⚠️ Tentativa {i+1}: resposta inválida")
        except Exception as e:
            print(f"⚠️ Erro tentativa {i+1}: {e}")
        time.sleep(2)
    raise RuntimeError("❌ Falha ao obter dados da NOAA")


# ============================================================
# DOWNLOAD DADOS DSCOVR (ROBUSTO)
# ============================================================

def load_dscovr():
    plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-hour.json"
    mag_url    = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-hour.json"

    plasma = safe_get_json(plasma_url)
    mag    = safe_get_json(mag_url)

    df_p = pd.DataFrame(plasma[1:], columns=plasma[0])
    df_m = pd.DataFrame(mag[1:], columns=mag[0])

    df_p["time_tag"] = pd.to_datetime(df_p["time_tag"])
    df_m["time_tag"] = pd.to_datetime(df_m["time_tag"])

    df = pd.merge(df_p, df_m, on="time_tag", how="inner")

    df = df.rename(columns={
        "speed": "V",
        "density": "N",
        "bz_gsm": "Bz",
        "bt": "Bt"
    })

    df = df[["time_tag", "V", "N", "Bz", "Bt"]]
    df = df.apply(pd.to_numeric, errors="ignore")

    return df.dropna()


# ============================================================
# HAC MODEL
# ============================================================

class HAC:
    def __init__(self):
        self.alpha = 1.5e-5
        self.beta = 0.12
        self.gamma = 0.8
        self.delta = 1.2
        self.theta = 2.8
        self.sigma = 0.4

    def phi(self, v, n, bz, bt):
        bz_s = np.maximum(-bz, 0)
        return n * v**2 * bt**2 + v * bt**2 * (bz_s > 0)

    def step(self, H, phi):
        sat = 1 / (1 + np.exp(-(H - self.theta)/self.sigma))
        return H + self.alpha * phi * (1 - sat) - self.beta * H

    def run(self, df):
        H = np.zeros(len(df))
        phi = self.phi(df.V, df.N, df.Bz, df.Bt)

        for i in range(1, len(df)):
            H[i] = self.step(H[i-1], phi[i])

        HCI = self.gamma * phi + self.delta * H
        return H, HCI


# ============================================================
# CLASSIFICAÇÃO
# ============================================================

def classify(hci):
    if hci > 1.2e7:
        return "G5 EXTREMA"
    elif hci > 7e6:
        return "G4 SEVERA"
    elif hci > 4e6:
        return "G3 FORTE"
    elif hci > 2e6:
        return "G2 MODERADA"
    elif hci > 1e6:
        return "G1 FRACA"
    else:
        return "CALMO"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n📡 Baixando dados DSCOVR...")
    df = load_dscovr()

    print(f"✔ Dados recebidos: {len(df)} pontos")
    print(f"🕒 Último dado: {df.time_tag.iloc[-1]}")

    model = HAC()
    H, HCI = model.run(df)

    current = HCI[-1]

    print("\n==============================")
    print("🌎 HAC — ESTADO ATUAL")
    print("==============================")
    print(f"HCI: {current:,.2e}")
    print(f"Nível: {classify(current)}")
    print(f"Bz: {df.Bz.iloc[-1]:.2f} nT")
    print(f"Vento: {df.V.iloc[-1]:.1f} km/s")
    print(f"Densidade: {df.N.iloc[-1]:.2f} cm⁻³")
