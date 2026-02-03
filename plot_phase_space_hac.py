import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Carregar dados
# ===============================
FILE = "hac_nowcast_results.csv"
df = pd.read_csv(FILE)

# ===============================
# 2. Normalizar nomes de colunas
# ===============================
rename_map = {
    "HAC": "H",
    "H_index": "H",
    "dH_dt": "dHdt",
    "dH/dt": "dHdt",
    "Dst": "Dst_min",
    "Dst_equiv": "Dst_min"
}

for old, new in rename_map.items():
    if old in df.columns and new not in df.columns:
        df = df.rename(columns={old: new})

# ===============================
# 3. Verificação mínima
# ===============================
required = ["H", "dHdt"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

# ===============================
# 4. Se não houver Dst_min, inferir
# ===============================
if "Dst_min" not in df.columns:
    if "G_class" in df.columns:
        mapping = {
            "G1": -50,
            "G2": -100,
            "G3": -150,
            "G4": -200,
            "G5": -300
        }
        df["Dst_min"] = df["G_class"].map(mapping)
    else:
        raise ValueError("Não há Dst_min nem G_class para inferência.")

# ===============================
# 5. Limpeza básica
# ===============================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["H", "dHdt", "Dst_min"])

# ===============================
# 6. Classificação por severidade
# ===============================
def dst_color(dst):
    if dst < -200:
        return "purple"
    elif dst < -150:
        return "darkblue"
    elif dst < -100:
        return "lightblue"
    else:
        return "gray"

df["color"] = df["Dst_min"].apply(dst_color)

# ===============================
# 7. Criar figura
# ===============================
fig, ax = plt.subplots(figsize=(8, 6))

# Fundo por regime operacional
ax.axhspan(-300, -150, color="red", alpha=0.08)
ax.axhspan(-150, -50, color="orange", alpha=0.08)
ax.axhspan(-50, 50, color="yellow", alpha=0.08)
ax.axhspan(50, 300, color="green", alpha=0.08)

# Scatter principal
ax.scatter(
    df["H"],
    df["dHdt"],
    c=df["color"],
    s=18,
    alpha=0.7,
    edgecolor="none"
)

# ===============================
# 8. Limites e rótulos
# ===============================
ax.set_xlabel("H (HAC Index)")
ax.set_ylabel(r"$dH/dt$ (nT/h)")
ax.set_title("HAC Phase Space: Magnetospheric Regime Classification")

# Linhas de decisão
ax.axhline(50, linestyle="--", color="k", linewidth=1)
ax.axhline(-50, linestyle="--", color="k", linewidth=1)

# ===============================
# 9. Legenda customizada
# ===============================
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker="o", color="w", label="Dst > -100 nT",
           markerfacecolor="gray", markersize=7),
    Line2D([0], [0], marker="o", color="w", label="-150 < Dst ≤ -100 nT",
           markerfacecolor="lightblue", markersize=7),
    Line2D([0], [0], marker="o", color="w", label="-200 < Dst ≤ -150 nT",
           markerfacecolor="darkblue", markersize=7),
    Line2D([0], [0], marker="o", color="w", label="Dst ≤ -200 nT",
           markerfacecolor="purple", markersize=7),
]

ax.legend(handles=legend_elements, loc="upper left", frameon=True)

# ===============================
# 10. Salvar
# ===============================
plt.tight_layout()
plt.savefig("phase_space_regimes.png", dpi=300)
plt.show()

print("Figura gerada: phase_space_regimes.png")
