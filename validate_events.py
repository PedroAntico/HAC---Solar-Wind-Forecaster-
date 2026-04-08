#!/usr/bin/env python3
"""
validate_events.py - Validação robusta com fallback automático
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# ====================== IMPORT FLEXÍVEL ======================
try:
    import hac_final as hac
except ImportError:
    raise ImportError("❌ Não foi possível importar hac_final.py")

# ====================== EVENTOS ======================
EVENTS = {
    "Halloween_2003": ("2003-10-28", "2003-11-02"),
    "St_Patrick_2015": ("2015-03-17", "2015-03-20"),
    "Carrington_like_2012": ("2012-07-22", "2012-07-25"),
}

# ====================== FALLBACK FUNCTIONS ======================
def fallback_prepare_omni(df):
    print("⚠️ Usando fallback prepare_omni")
    df = df.sort_values("time_tag").reset_index(drop=True)

    # Correção de velocidade
    if df["speed"].median() > 10000:
        df["speed"] /= 1000

    if df["speed"].median() < 300:
        factor = 420 / max(df["speed"].median(), 50)
        df["speed"] *= factor

    return df.dropna().reset_index(drop=True)


def fallback_euler_dst(df, alpha=5.5, tau=8.0):
    print("⚠️ Usando fallback euler_dst")

    Bz = df["bz_gsm"].values
    V = df["speed"].values
    VBs = V * np.maximum(-Bz, 0) / 1000.0

    dt = 1.0  # 1h

    Dst = np.zeros(len(V))
    for i in range(1, len(V)):
        dDst = -alpha * VBs[i] - Dst[i-1] / tau
        Dst[i] = Dst[i-1] + dDst * dt

    return Dst


# ====================== DETECÇÃO AUTOMÁTICA ======================
prepare_omni = getattr(hac, "prepare_omni", fallback_prepare_omni)
euler_dst = getattr(hac, "euler_dst", fallback_euler_dst)


# ====================== LOAD DATA ======================
def load_data():
    omni = pd.read_csv("data/omni_2000_2020.csv", parse_dates=["time_tag"])
    omni = prepare_omni(omni)

    dst = pd.read_csv("data/dst_kyoto_2000_2020.csv", parse_dates=["time_tag"])

    df = pd.merge_asof(
        omni, dst,
        on="time_tag",
        direction="nearest",
        tolerance=pd.Timedelta("2h")
    ).dropna(subset=["dst"])

    return df


# ====================== VALIDAÇÃO ======================
def run_event(df, name, start, end):
    mask = (df["time_tag"] >= start) & (df["time_tag"] <= end)
    event = df[mask].copy()

    if len(event) < 10:
        print(f"⚠️ {name}: poucos dados")
        return

    Dst_raw = euler_dst(event)

    # Calibração
    reg = LinearRegression().fit(Dst_raw.reshape(-1,1), event["dst"])
    pred = reg.predict(Dst_raw.reshape(-1,1))

    corr = pearsonr(event["dst"], pred)[0]
    mae = mean_absolute_error(event["dst"], pred)

    print(f"\n🌌 {name}")
    print(f"Dst real min : {event['dst'].min():.1f}")
    print(f"Dst pred min : {pred.min():.1f}")
    print(f"Corr         : {corr:.3f}")
    print(f"MAE          : {mae:.1f}")

    event["Dst_pred"] = pred
    event.to_csv(f"event_{name}.csv", index=False)


# ====================== MAIN ======================
def main():
    print("🔬 VALIDANDO EVENTOS HISTÓRICOS\n")

    df = load_data()

    for name, (start, end) in EVENTS.items():
        run_event(df, name, start, end)

    print("\n✅ Finalizado")


if __name__ == "__main__":
    main()
