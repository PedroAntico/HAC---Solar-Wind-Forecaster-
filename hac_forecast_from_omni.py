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
    return pd.DataFrame(data, columns=header)

# ============================
# PROCESSAMENTO
# ============================

def build_dataframe(mag, plasma):
    df = pd.merge(mag, plasma, on="time_tag", how="inner")

    for col in df.columns:
        if col != "time_tag":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["time_tag"])
    return df


def compute_hac(df, beta=0.6):
    """
    HAC acumulado fisicamente consistente
    """

    Bz = df["bz_gsm"].values * 1e-9      # nT → T
    V  = df["speed"].values * 1e3        # km/s → m/s
    n  = df["density"].values * 1e6      # cm⁻³ → m⁻³

    # Termo físico instantâneo
    coupling = (n * V**2)**beta * (Bz**2)

    # Apenas Bz sul contribui
    coupling[Bz > 0] = 0

    # Integração acumulada
    dt = 60  # segundos (1 min)
    hac = np.cumsum(coupling * dt)

    df["HAC"] = hac
    return df


def classify(h):
    if h < 4e1:
        return "Quiet"
    elif h < 8e1:
        return "G3"
    elif h < 1.4e2:
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
print(f"⚡ HAC = {last['HAC']:.2f}")
print(f"🌍 Nível previsto = {level}")
print("==============================\n")

# ============================
# GRÁFICO FINAL
# ============================

plt.figure(figsize=(12,5))

plt.plot(df["time"], df["HAC"], color="red", lw=2, label="HAC")

# Limiares geomagnéticos
plt.axhline(40,  ls="--", c="orange",      lw=1.5, label="G3")
plt.axhline(80,  ls="--", c="darkorange", lw=1.5, label="G4")
plt.axhline(140, ls="--", c="darkred",    lw=1.5, label="G5")

plt.title("HAC — Heliospheric Accumulated Coupling")
plt.xlabel("Time (UTC)")
plt.ylabel("HAC (acumulado)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("hac_forecast.png", dpi=150)
plt.show()

print("📈 Gráfico salvo como hac_forecast.png")
