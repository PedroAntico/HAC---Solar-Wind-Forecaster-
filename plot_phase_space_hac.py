import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Load data
# ===============================
df = pd.read_csv("hac_nowcast_results.csv")

# ===============================
# 2. Select correct columns
# ===============================
H_col    = "HAC_total"
dHdt_col = "dHAC_dt"
Dst_col  = "Dst_pred"

required = [H_col, dHdt_col, Dst_col]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Coluna ausente no CSV: {c}")

# ===============================
# 3. Clean data
# ===============================
df = df[required].replace([np.inf, -np.inf], np.nan).dropna()

# ===============================
# 4. Color by Dst severity
# ===============================
def dst_color(dst):
    if dst <= -200:
        return "purple"      # G4–G5
    elif dst <= -150:
        return "darkblue"    # G4
    elif dst <= -100:
        return "lightblue"   # G3
    else:
        return "gray"        # Non-escalation

colors = df[Dst_col].apply(dst_color)

# ===============================
# 5. Plot
# ===============================
fig, ax = plt.subplots(figsize=(8, 6))

# Background regimes (operational)
ax.axhspan(-300, -150, color="red", alpha=0.08)     # Escalation
ax.axhspan(-150, -50, color="orange", alpha=0.08)  # Warning
ax.axhspan(-50, 50, color="yellow", alpha=0.08)    # Loading
ax.axhspan(50, 300, color="green", alpha=0.08)     # Stable

# Scatter
ax.scatter(
    df[H_col],
    df[dHdt_col],
    c=colors,
    s=18,
    alpha=0.75,
    edgecolor="none"
)

# Thresholds
ax.axhline(50, linestyle="--", color="k", linewidth=1)
ax.axhline(-50, linestyle="--", color="k", linewidth=1)

# Labels
ax.set_xlabel("HAC Index (H)")
ax.set_ylabel("dH/dt (nT/h)")
ax.set_title("HAC Phase Space — Magnetospheric Regimes")

plt.tight_layout()
plt.savefig("phase_space_regimes.png", dpi=300)
plt.show()

print("✔ Figura gerada: phase_space_regimes.png")
