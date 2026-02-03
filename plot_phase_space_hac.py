import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ============================================================
# CONFIGURAÇÕES
# ============================================================
CSV_FILE = "hac_nowcast_results.csv"
OUTPUT_FIG = "hac_phase_space_trajectories_kde.png"

# Limiares físicos (HAC)
DHDT_CRIT = 50.0     # nT/h
H_G1 = 50
H_G2 = 100
H_G3 = 150
H_G4 = 200

# ============================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================
df = pd.read_csv(CSV_FILE)

# Garantia mínima de colunas
required_cols = ["HAC_total", "dH_dt", "Dst_pred"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colunas ausentes: {missing}")

# Renomear para notação física
H = df["HAC_total"].values
dHdt = df["dH_dt"].values
Dst = df["Dst_pred"].values

# Remover NaNs
mask = np.isfinite(H) & np.isfinite(dHdt) & np.isfinite(Dst)
H, dHdt, Dst = H[mask], dHdt[mask], Dst[mask]

# ============================================================
# DEFINIÇÃO DE REGIMES (FÍSICOS)
# ============================================================
regime_stable = H < H_G2
regime_loading = (H >= H_G2) & (H < H_G3)
regime_critical = (H >= H_G3) & (H < H_G4)
regime_escalation = H >= H_G4

# ============================================================
# FUNÇÃO KDE 2D
# ============================================================
def kde_density(x, y, gridsize=150):
    values = np.vstack([x, y])
    kde = gaussian_kde(values)
    xi = np.linspace(x.min(), x.max(), gridsize)
    yi = np.linspace(y.min(), y.max(), gridsize)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z

# ============================================================
# FIGURA
# ============================================================
plt.figure(figsize=(14, 8))

# ---------- Fundos por regime ----------
plt.axvspan(0, H_G2, color="#e8f5e9", alpha=0.6, label="Estável")
plt.axvspan(H_G2, H_G3, color="#fffde7", alpha=0.6, label="Carregamento")
plt.axvspan(H_G3, H_G4, color="#fff3e0", alpha=0.6, label="Crítico")
plt.axvspan(H_G4, max(H)+20, color="#fdecea", alpha=0.6, label="Escalada")

# ---------- Limiares ----------
plt.axhline(DHDT_CRIT, color="k", linestyle="--", linewidth=1)
plt.axhline(-DHDT_CRIT, color="k", linestyle="--", linewidth=1)

# ---------- Scatter base ----------
sc = plt.scatter(
    H, dHdt,
    c=Dst,
    cmap="viridis_r",
    s=18,
    alpha=0.55,
    edgecolor="none",
    label="Eventos (1h)"
)

cbar = plt.colorbar(sc)
cbar.set_label("Dst mínimo previsto (nT)")

# ============================================================
# TRAJETÓRIAS TEMPORAIS (subamostradas)
# ============================================================
stride = 6  # 6 horas
for i in range(0, len(H) - stride, stride):
    plt.plot(
        H[i:i+stride],
        dHdt[i:i+stride],
        color="black",
        alpha=0.15,
        linewidth=0.7
    )

# ============================================================
# KDE — apenas eventos de escalada severa
# ============================================================
severe = Dst < -150
if np.sum(severe) > 50:
    X, Y, Z = kde_density(H[severe], dHdt[severe])
    plt.contour(
        X, Y, Z,
        levels=5,
        colors="purple",
        linewidths=1.2,
        alpha=0.9
    )

# ============================================================
# ANOTAÇÕES
# ============================================================
plt.text(10, 260, "Regime Estável", fontsize=10)
plt.text(110, 260, "Carregamento", fontsize=10)
plt.text(160, 260, "Crítico", fontsize=10)
plt.text(220, 260, "Escalada", fontsize=10)

# ============================================================
# FINALIZAÇÃO
# ============================================================
plt.xlabel("HAC Index (H)")
plt.ylabel("dH/dt (nT/h)")
plt.title("HAC Phase Space — Trajetórias Temporais e Densidade por Regime")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.show()

print(f"[OK] Figura gerada: {OUTPUT_FIG}")
