import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import gaussian_kde

# =========================
# CONFIGURAÇÕES
# =========================
CSV_FILE = "hac_nowcast_results.csv"
OUTDIR = "./figures/"
EVENT_START = "2026-01-27"
EVENT_END   = "2026-02-03"

# Limiares operacionais HAC
H1, H2, H3 = 80, 150, 230
DHDT_ESC = -50  # nT/h

# =========================
# LOAD + PREP
# =========================
df = pd.read_csv(CSV_FILE)
df["time_tag"] = pd.to_datetime(df["time_tag"])
df = df.sort_values("time_tag")

# Derivada se não existir
if "dH_dt" not in df.columns:
    df["dH_dt"] = df["HAC_index"].diff()

# =========================
# REGIMES
# =========================
def classify_regime(row):
    if row["HAC_index"] < H1:
        return "Stable"
    elif row["HAC_index"] < H2:
        return "Loading"
    elif row["HAC_index"] < H3:
        return "Critical"
    else:
        return "Escalation"

df["regime"] = df.apply(classify_regime, axis=1)

# =========================
# EVENTO RECORTE
# =========================
event = df[(df["time_tag"] >= EVENT_START) &
           (df["time_tag"] <= EVENT_END)]

# =========================
# FIGURA 1 — TRAJETÓRIA TEMPORAL
# =========================
fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

axs[0].plot(event["time_tag"], event["HAC_index"])
axs[0].set_ylabel("H")

axs[1].plot(event["time_tag"], event["dH_dt"])
axs[1].axhline(DHDT_ESC, ls="--", c="r")
axs[1].set_ylabel("dH/dt")

axs[2].plot(event["time_tag"], event["Dst_pred"])
axs[2].set_ylabel("Dst (pred)")

axs[3].step(event["time_tag"], event["Storm_level"], where="post")
axs[3].set_ylabel("HAC Level")

axs[4].step(event["time_tag"], event["Kp_pred"], where="post")
axs[4].set_ylabel("Kp")
axs[4].set_xlabel("Time")

plt.suptitle("HAC Temporal Evolution — January 2026 Event")
plt.tight_layout()
plt.savefig(OUTDIR + "fig1_temporal_event.png", dpi=300)
plt.close()

# =========================
# FIGURA 2 — ESPAÇO DE FASE
# =========================
plt.figure(figsize=(10, 6))

colors = {
    "Stable": "#2ecc71",
    "Loading": "#f1c40f",
    "Critical": "#e67e22",
    "Escalation": "#e74c3c"
}

for reg in colors:
    subset = df[df["regime"] == reg]
    plt.scatter(
        subset["HAC_index"],
        subset["dH_dt"],
        s=8,
        alpha=0.3,
        label=reg,
        color=colors[reg]
    )

plt.axhline(50, ls="--", c="k")
plt.axhline(-50, ls="--", c="k")

plt.xlabel("HAC Index (H)")
plt.ylabel("dH/dt (nT/h)")
plt.title("HAC Phase Space — Magnetospheric Regimes")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR + "fig2_phase_space.png", dpi=300)
plt.close()

# =========================
# FIGURA 3 — DENSIDADE POR REGIME
# =========================
regime_counts = df["regime"].value_counts(normalize=True) * 100

plt.figure(figsize=(7, 5))
regime_counts.loc[
    ["Stable", "Loading", "Critical", "Escalation"]
].plot(kind="bar")

plt.ylabel("Fraction of Time (%)")
plt.title("Time Spent in Each HAC Regime")
plt.tight_layout()
plt.savefig(OUTDIR + "fig3_regime_density.png", dpi=300)
plt.close()

# =========================
# FIGURA 4 — LEAD TIME
# =========================
# Evento severo definido como Dst < -150 nT
event_flag = (df["Dst_pred"] < -150).astype(int)

# HAC alerta
hac_alert = (df["Storm_level"] >= 2).astype(int)

# Nowcast clássico (Kp ≥ 5)
nowcast_alert = (df["Kp_pred"] >= 5).astype(int)

# ROC
fpr_hac, tpr_hac, _ = roc_curve(event_flag, hac_alert)
fpr_now, tpr_now, _ = roc_curve(event_flag, nowcast_alert)

auc_hac = auc(fpr_hac, tpr_hac)
auc_now = auc(fpr_now, tpr_now)

plt.figure(figsize=(7, 6))
plt.plot(fpr_hac, tpr_hac, label=f"HAC (AUC={auc_hac:.2f})")
plt.plot(fpr_now, tpr_now, label=f"Nowcast (AUC={auc_now:.2f})")
plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Alarm Rate")
plt.ylabel("Probability of Detection")
plt.title("ROC Curve — HAC vs Classical Nowcast")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR + "fig4_roc.png", dpi=300)
plt.close()

# =========================
# FIGURA 5 — KDE DO ESPAÇO DE FASE
# =========================
x = df["HAC_index"].values
y = df["dH_dt"].values
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=z, s=6, cmap="plasma")
plt.colorbar(label="Density")
plt.xlabel("HAC Index (H)")
plt.ylabel("dH/dt")
plt.title("Density Structure of HAC Phase Space")
plt.tight_layout()
plt.savefig(OUTDIR + "fig5_phase_space_density.png", dpi=300)
plt.close()

print("✔ Todas as figuras HAC geradas com sucesso.")
