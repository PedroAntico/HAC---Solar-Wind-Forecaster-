import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============================
# CONFIG
# ============================

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

DT = 60  # segundos (1 minuto)
BETA = 0.65  # Valor entre 0.5-0.7
NORMALIZATION = 1e-18  # Fator de escala

# ============================
# LOADERS - CORRIGIDO PARA DADOS REAIS
# ============================

def load_omni_json(path):
    """Carrega dados OMNI no formato JSON com valores null"""
    with open(path, "r") as f:
        data = json.load(f)
    
    # Primeira linha são os cabeçalhos
    headers = data[0]
    df = pd.DataFrame(data[1:], columns=headers)
    
    # Converter time_tag para datetime
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")
    
    # Converter colunas numéricas, tratando "null" como NaN
    for col in headers:
        if col != "time_tag":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

# ============================
# PROCESSAMENTO - ROBUSTO PARA DADOS REAIS
# ============================

def merge_and_clean_data(mag_df, plasma_df):
    """Funde e limpa dados magnéticos e de plasma"""
    # Renomear colunas para evitar conflitos
    mag_df = mag_df.rename(columns={col: f"mag_{col}" for col in mag_df.columns if col != "time_tag"})
    plasma_df = plasma_df.rename(columns={col: f"plasma_{col}" for col in plasma_df.columns if col != "time_tag"})
    
    # Fusão dos dados
    df = pd.merge(mag_df, plasma_df, on="time_tag", how="outer")
    df = df.sort_values("time_tag").reset_index(drop=True)
    
    # Preencher valores ausentes com interpolação linear (limitada)
    numeric_cols = [col for col in df.columns if col != "time_tag"]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=5, limit_direction="both")
    
    # Remover linhas onde dados essenciais ainda estão ausentes
    essential_cols = ["mag_bz_gsm", "plasma_speed", "plasma_density"]
    df = df.dropna(subset=essential_cols)
    
    return df

def compute_hac_robust(df):
    """Calcula HAC robustamente, lidando com dados reais"""
    # Extrair arrays, garantindo que sejam float64
    Bz = df["mag_bz_gsm"].values.astype(np.float64) * 1e-9  # nT → T
    V = df["plasma_speed"].values.astype(np.float64) * 1e3  # km/s → m/s
    n = df["plasma_density"].values.astype(np.float64) * 1e6  # cm⁻³ → m⁻³
    
    # Para B total, usamos a magnitude se disponível, senão calculamos
    if "mag_bt" in df.columns:
        B = df["mag_bt"].values.astype(np.float64) * 1e-9
    else:
        # Calcular B total a partir dos componentes
        Bx = df.get("mag_bx_gsm", pd.Series(0)).values.astype(np.float64) * 1e-9
        By = df.get("mag_by_gsm", pd.Series(0)).values.astype(np.float64) * 1e-9
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    # Condição de IMF sul: Bz < 0
    southward = np.where(Bz < 0, 1.0, 0.0)
    
    # Termo de acoplamento: (n·V²)^β · B²
    # Adicionar epsilon pequeno para evitar problemas numéricos
    eps = 1e-10
    dynamic_pressure = n * V**2
    coupling = (np.abs(dynamic_pressure + eps)**BETA) * (B**2 + eps)
    
    # Aplicar condição sul
    integrand = coupling * southward
    
    # Integração cumulativa com passo de tempo
    # Calcular delta_t real entre medições
    times = pd.to_datetime(df["time_tag"]).values
    delta_t = np.zeros(len(times))
    delta_t[1:] = (times[1:] - times[:-1]).astype('timedelta64[s]').astype(np.float64)
    delta_t[0] = DT  # Assumir DT para o primeiro ponto
    
    # Integrar
    hac_raw = np.cumsum(integrand * delta_t)
    
    # Normalizar para valores razoáveis
    hac_normalized = hac_raw * NORMALIZATION
    
    # Adicionar ao DataFrame
    df["HAC"] = hac_normalized
    
    return df

def classify_storm(hac_value):
    """Classifica a tempestade baseada no valor HAC"""
    if hac_value < 10:
        return "Quiet", "🟢"
    elif hac_value < 50:
        return "G1", "🟡"
    elif hac_value < 100:
        return "G2", "🟠"
    elif hac_value < 200:
        return "G3", "🔴"
    elif hac_value < 300:
        return "G4", "🟣"
    else:
        return "G5", "⚫"

# ============================
# VISUALIZAÇÃO - GRÁFICO SIMPLIFICADO
# ============================

def create_simple_plot(df, save_path="hac_forecast_simple.png"):
    """Cria um gráfico simples e robusto"""
    plt.figure(figsize=(12, 6))
    
    # Plot HAC
    plt.plot(df["time_tag"], df["HAC"], color="red", linewidth=2, label="HAC")
    
    # Linhas de threshold
    thresholds = [10, 50, 100, 200, 300]
    labels = ["G1", "G2", "G3", "G4", "G5"]
    colors = ["yellow", "orange", "red", "purple", "black"]
    
    for thresh, label, color in zip(thresholds, labels, colors):
        plt.axhline(y=thresh, color=color, linestyle="--", alpha=0.7, label=f"{label} threshold")
    
    plt.title("Heliospheric Accumulated Coupling (HAC)", fontsize=14, fontweight="bold")
    plt.xlabel("Time (UTC)", fontsize=12)
    plt.ylabel("HAC Index", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    # Ajustar formato de data no eixo x
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path

def create_detailed_plot(df, save_path="hac_forecast_detailed.png"):
    """Cria gráfico detalhado com múltiplos painéis"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Painel 1: HAC
    ax1 = axes[0]
    ax1.plot(df["time_tag"], df["HAC"], color="#d62728", linewidth=2)
    ax1.fill_between(df["time_tag"], 0, df["HAC"], alpha=0.3, color="#d62728")
    ax1.set_ylabel("HAC Index", fontsize=12, fontweight="bold")
    ax1.set_title("Heliospheric Accumulated Coupling (HAC) Evolution", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Adicionar thresholds
    thresholds = [10, 50, 100, 200, 300]
    for thresh in thresholds:
        ax1.axhline(y=thresh, color="gray", linestyle="--", alpha=0.5)
    
    # Painel 2: Bz
    ax2 = axes[1]
    ax2.plot(df["time_tag"], df["mag_bz_gsm"], color="#2ca02c", linewidth=1.5)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax2.fill_between(df["time_tag"], 0, df["mag_bz_gsm"], 
                     where=(df["mag_bz_gsm"] < 0), alpha=0.5, color="red")
    ax2.set_ylabel("Bz (nT)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Painel 3: Velocidade
    ax3 = axes[2]
    ax3.plot(df["time_tag"], df["plasma_speed"], color="#1f77b4", linewidth=1.5)
    ax3.set_ylabel("Speed (km/s)", fontsize=11)
    ax3.set_xlabel("Time (UTC)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return save_path

# ============================
# MAIN - COM TRATAMENTO DE ERROS
# ============================

def main():
    print("📥 Loading OMNI data...")
    try:
        mag_df = load_omni_json(MAG_FILE)
        plasma_df = load_omni_json(PLASMA_FILE)
        
        print(f"   Magnetic data: {len(mag_df)} records")
        print(f"   Plasma data: {len(plasma_df)} records")
        
        # Verificar primeiras linhas
        print("\n📊 Sample of magnetic data:")
        print(mag_df.head(3))
        print("\n📊 Sample of plasma data:")
        print(plasma_df.head(3))
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print("\n🔧 Merging and cleaning data...")
    df = merge_and_clean_data(mag_df, plasma_df)
    print(f"   Merged data: {len(df)} records")
    
    if len(df) == 0:
        print("❌ No valid data after merging!")
        return
    
    print("\n⚡ Computing HAC...")
    df = compute_hac_robust(df)
    
    # Análise dos últimos pontos
    last_n = min(10, len(df))
    recent = df.tail(last_n)
    
    if len(recent) > 0:
        current = recent.iloc[-1]
        hac_value = current["HAC"]
        storm_class, emoji = classify_storm(hac_value)
        
        print("\n" + "="*60)
        print(f"⏰ Current time: {current['time_tag']}")
        print(f"⚡ HAC index: {hac_value:.2f}")
        print(f"🌪️  Storm level: {emoji} {storm_class}")
        
        # Informações adicionais se disponíveis
        if "mag_bz_gsm" in current:
            print(f"🧲 Bz: {current['mag_bz_gsm']:.2f} nT")
        if "plasma_speed" in current:
            print(f"💨 Solar wind speed: {current['plasma_speed']:.1f} km/s")
        if "plasma_density" in current:
            print(f"📊 Density: {current['plasma_density']:.2f} cm⁻³")
        
        # Tendência
        if len(recent) > 1:
            prev_hac = recent.iloc[-2]["HAC"]
            hac_change = ((hac_value - prev_hac) / hac_value * 100) if hac_value != 0 else 0
            trend = "↑↑" if hac_change > 5 else "↑" if hac_change > 0 else "↓↓" if hac_change < -5 else "↓"
            print(f"📈 Trend: {trend} ({hac_change:+.1f}% change)")
        print("="*60)
    
    print("\n📈 Generating plots...")
    try:
        # Gráfico simples
        simple_plot = create_simple_plot(df, "hac_forecast_simple.png")
        print(f"✅ Simple plot saved as '{simple_plot}'")
        
        # Gráfico detalhado
        detailed_plot = create_detailed_plot(df, "hac_forecast_detailed.png")
        print(f"✅ Detailed plot saved as '{detailed_plot}'")
        
    except Exception as e:
        print(f"⚠️  Error generating plots: {e}")
        # Tentar um gráfico mínimo
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(df["time_tag"], df["HAC"], 'r-')
            plt.title("HAC")
            plt.xlabel("Time")
            plt.ylabel("HAC Index")
            plt.grid(True)
            plt.savefig("hac_minimal.png", dpi=150)
            print("✅ Minimal plot saved as 'hac_minimal.png'")
        except:
            print("❌ Could not generate any plot")
    
    # Salvar dados processados
    try:
        output_csv = "hac_processed_data.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Processed data saved to '{output_csv}'")
        
        # Estatísticas resumidas
        print("\n📊 HAC Statistics:")
        print(f"   Min: {df['HAC'].min():.2f}")
        print(f"   Max: {df['HAC'].max():.2f}")
        print(f"   Mean: {df['HAC'].mean():.2f}")
        print(f"   Last: {df['HAC'].iloc[-1]:.2f}")
        
    except Exception as e:
        print(f"⚠️  Error saving data: {e}")
    
    print("\n✅ Processing complete!")

# ============================
# EXECUÇÃO
# ============================

if __name__ == "__main__":
    main()
