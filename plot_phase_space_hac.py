import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ============================================================
# CONFIG
# ============================================================
CSV_FILE = "hac_nowcast_results.csv"
OUT_FIG = "phase_space_regimes.png"

H_COL = "HAC_total"
DHDT_COL = "dHAC_dt"
DST_COL = "Dst_pred"

# Limiares HAC
H_G1, H_G2, H_G3, H_G4 = 50, 100, 150, 200
DHDT_CRIT = 50.0  # nT/h equivalente HAC

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(CSV_FILE)

for col in [H_COL, DHDT_COL, DST_COL]:
    if col not in df.columns:
        raise ValueError(f"Coluna ausente: {col}")

H = df[H_COL].values
dHdt = df[DHDT_COL].values
Dst = df[DST_COL].values

mask = np.isfinite(H) & np.isfinite(dHdt) & np.isfinite(Dst)
H, dHdt, Dst = H[mask], dHdt[mask], Dst[mask]

# ============================================================
# REGIMES
# ============================================================
stable = H < H_G2
loading = (H >= H_G2) & (H < H_G3)
critical = (H >= H_G3) & (H < H_G4)
escalation = H >= H_G4

# ============================================================
# KDE FUNCTION
# ============================================================
def kde2d(x, y, gridsize=150):
    kde = gaussian_kde(np.vstack([x, y]))
    xi = np.linspace(x.min(), x.max(), gridsize)
    yi = np.linspace(y.min(), y.max(), gridsize)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(14, 8))

# Regime backgrounds
plt.axvspan(0, H_G2, color="#e8f5e9", alpha=0.6)
plt.axvspan(H_G2, H_G3, color="#fffde7", alpha=0.6)
plt.axvspan(H_G3, H_G4, color="#fff3e0", alpha=0.6)
plt.axvspan(H_G4, H.max()+20, color="#fdecea", alpha=0.6)

# Thresholds
plt.axhline(DHDT_CRIT, color="k", linestyle="--", lw=1)
plt.axhline(-DHDT_CRIT, color="k", linestyle="--", lw=1)

# Scatter
sc = plt.scatter(
    H, dHdt,
    c=Dst,
    cmap="viridis_r",
    s=18,
    alpha=0.55
)

cbar = plt.colorbar(sc)
cbar.set_label("Dst previsto (nT)")

# Trajetórias temporais (memória)
stride = 6
for i in range(0, len(H) - stride, stride):
    plt.plot(
        H[i:i+stride],
        dHdt[i:i+stride],
        color="black",
        alpha=0.15,
        lw=0.7
    )

# KDE — eventos severos
severe = Dst < -150
if severe.sum() > 30:
    X, Y, Z = kde2d(H[severe], dHdt[severe])
    plt.contour(X, Y, Z, colors="purple", linewidths=1.2)

# Labels
plt.xlabel("HAC Index (H)")
plt.ylabel("dHAC/dt")
plt.title("HAC Phase Space — Regimes Magnetosféricos")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.show()

print(f"[OK] Figura gerada: {OUT_FIG}")
