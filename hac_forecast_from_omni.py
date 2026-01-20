import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============================
# CONFIG - AJUSTADO
# ============================

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

DT = 60  # segundos (1 minuto)
BETA = 0.65  # Valor médio entre 0.5-0.7
NORMALIZATION = 1e-18  # Fator para valores razoáveis

# Thresholds calibrados (ajustar com validação histórica)
THRESHOLDS = {
    "Quiet": 0,
    "G1": 10,
    "G2": 50,
    "G3": 100,
    "G4": 200,
    "G5": 300
}

# ============================
# LOADERS
# ============================

def load_json_table(path):
    """Carrega dados OMNI no formato JSON"""
    with open(path, "r") as f:
        raw = json.load(f)
    return pd.DataFrame(raw[1:], columns=raw[0])

# ============================
# PROCESSAMENTO - CORRIGIDO
# ============================

def build_dataframe(mag, plasma):
    """Unifica dados magnéticos e de plasma"""
    df = pd.merge(mag, plasma, on="time_tag", how="inner")
    
    # Converter para numérico
    for col in df.columns:
        if col != "time_tag":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["time"] = pd.to_datetime(df["time_tag"])
    df = df.sort_values("time").reset_index(drop=True)
    
    return df

def compute_hac(df):
    """
    Calcula HAC conforme equação do artigo:
    HAC(t) = ∫ [(n·V²)^β · B² · Θ(-Bz)] dt
    """
    # Converter para unidades SI
    Bx = df["bx_gsm"].values * 1e-9  # nT → T
    By = df["by_gsm"].values * 1e-9
    Bz = df["bz_gsm"].values * 1e-9
    V = df["speed"].values * 1e3  # km/s → m/s
    n = df["density"].values * 1e6  # cm⁻³ → m⁻³
    
    # Magnitude total do campo B (conforme artigo)
    B = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    # Condição de IMF sul: Bz < 0
    southward = np.where(Bz < 0, 1.0, 0.0)
    
    # Termo de acoplamento: (n·V²)^β · B²
    dynamic_pressure = n * V**2  # ρ·V² (kg/m·s²)
    coupling = (np.abs(dynamic_pressure)**BETA) * (B**2)
    
    # Aplicar condição sul
    integrand = coupling * southward
    
    # Integração cumulativa com saturação
    hac_raw = np.cumsum(integrand * DT)
    
    # Normalização e suavização
    hac_normalized = hac_raw * NORMALIZATION
    
    # Saturação para evitar divergência
    saturation_level = 500  # Valor máximo físico
    hac_normalized = np.minimum(hac_normalized, saturation_level)
    
    df["HAC_raw"] = hac_raw
    df["HAC"] = hac_normalized
    
    return df

def classify_storm(hac_value):
    """Classifica tempestade baseado em thresholds calibrados"""
    if hac_value < THRESHOLDS["G1"]:
        return "Quiet", "🟢"
    elif hac_value < THRESHOLDS["G2"]:
        return "G1", "🟡"
    elif hac_value < THRESHOLDS["G3"]:
        return "G2", "🟠"
    elif hac_value < THRESHOLDS["G4"]:
        return "G3", "🔴"
    elif hac_value < THRESHOLDS["G5"]:
        return "G4", "🟣"
    else:
        return "G5", "⚫"

# ============================
# VISUALIZAÇÃO
# ============================

def plot_hac_evolution(df, save_path="hac_forecast.png"):
    """Gera gráfico profissional para o artigo"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                            gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 1. Gráfico HAC principal
    ax1 = axes[0]
    ax1.plot(df["time"], df["HAC"], color="#d62728", linewidth=2.5, 
             label="HAC")
    
    # Linhas de threshold
    colors = ["green", "yellow", "orange", "red", "purple", "black"]
    for i, (level, value) in enumerate(list(THRESHOLDS.items())[1:]):
        ax1.axhline(y=value, color=colors[i], linestyle="--", 
                   alpha=0.7, label=f"{level} threshold")
    
    ax1.fill_between(df["time"], 0, df["HAC"], alpha=0.3, color="#d62728")
    ax1.set_ylabel("HAC Index", fontsize=12, fontweight="bold")
    ax1.set_title("Heliospheric Accumulated Coupling (HAC) - January 2026 Storm", 
                  fontsize=14, fontweight="bold", pad=20)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. Parâmetros solares
    ax2 = axes[1]
    ax2.plot(df["time"], df["bz_gsm"], color="#2ca02c", linewidth=1.5, 
             label="Bz (GSM)")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax2.fill_between(df["time"], 0, df["bz_gsm"], 
                     where=(df["bz_gsm"] < 0), alpha=0.5, color="red")
    ax2.set_ylabel("Bz [nT]", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    # 3. Velocidade do vento solar
    ax3 = axes[2]
    ax3.plot(df["time"], df["speed"], color="#1f77b4", linewidth=1.5,
             label="Solar Wind Speed")
    ax3.set_ylabel("Speed [km/s]", fontsize=11)
    ax3.set_xlabel("Time (UTC)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return fig

# ============================
# MAIN
# ============================

def main():
    print("📥 Loading OMNI data...")
    try:
        mag = load_json_table(MAG_FILE)
        plasma = load_json_table(PLASMA_FILE)
        print(f"   Magnetic data: {len(mag)} records")
        print(f"   Plasma data: {len(plasma)} records")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print("🔧 Processing and merging datasets...")
    df = build_dataframe(mag, plasma)
    
    print("⚡ Computing HAC (corrected formula)...")
    df = compute_hac(df)
    
    # Análise da última janela (6h)
    window_hours = 6
    window_start = df["time"].max() - timedelta(hours=window_hours)
    recent = df[df["time"] >= window_start]
    
    if len(recent) > 0:
        current = recent.iloc[-1]
        hac_value = current["HAC"]
        storm_class, emoji = classify_storm(hac_value)
        
        print("\n" + "="*50)
        print(f"⏰ Current time: {current['time']}")
        print(f"⚡ HAC index: {hac_value:.2f}")
        print(f"🌪️  Storm level: {emoji} {storm_class}")
        print(f"📊 Bz: {current['bz_gsm']:.1f} nT")
        print(f"💨 Speed: {current['speed']:.0f} km/s")
        print("="*50)
        
        # Tendência
        if len(recent) > 1:
            hac_change = (hac_value - recent.iloc[-2]["HAC"]) / hac_value * 100
            trend = "↑↑" if hac_change > 5 else "↑" if hac_change > 0 else "↓"
            print(f"📈 Trend: {trend} ({hac_change:+.1f}% last hour)")
    
    print("\n📈 Generating publication-quality plot...")
    plot_hac_evolution(df)
    print("✅ Plot saved as 'hac_forecast.png'")
    
    # Salvar dados processados
    output_csv = "hac_processed_data.csv"
    df.to_csv(output_csv, index=False)
    print(f"💾 Processed data saved to '{output_csv}'")

if __name__ == "__main__":
    main()
