import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURAÇÃO FÍSICA
# ============================

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

# Parâmetros do modelo físico
BETA = 0.65  # Expoente de acoplamento (0.5-0.7)
DECAY_TAU = 2 * 3600  # Tempo de decaimento: 2 horas em segundos
INJECTION_GAIN = 5e13  # Fator de ganho para escala física correta
MAX_HAC = 300  # Saturação física

# Thresholds para classificação
THRESHOLDS = {
    "Quiet": 0,
    "G1": 50,
    "G2": 100,
    "G3": 150,
    "G4": 200,
    "G5": 250
}

STORM_LEVELS = [
    ("Quiet", "🟢", 0, 50),
    ("G1", "🟡", 50, 100),
    ("G2", "🟠", 100, 150),
    ("G3", "🔴", 150, 200),
    ("G4", "🟣", 200, 250),
    ("G5", "⚫", 250, 1000)
]

# ============================
# FUNÇÕES DE CARREGAMENTO
# ============================

def load_omni_data(filepath):
    """Carrega dados OMNI no formato JSON com tratamento de null."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
        
        # Converter para numérico, tratando null
        for col in headers:
            if col != 'time_tag':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"❌ Erro ao carregar {filepath}: {e}")
        return None

def prepare_and_merge_data(mag_df, plasma_df):
    """Prepara e funde dados magnéticos e de plasma."""
    # Limpeza básica
    mag_df = mag_df.dropna(subset=['time_tag']).copy()
    plasma_df = plasma_df.dropna(subset=['time_tag']).copy()
    
    # Fundir mantendo todos os timestamps
    df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
    df = df.sort_values('time_tag').reset_index(drop=True)
    
    # Colunas essenciais
    essential_cols = ['bx_gsm', 'by_gsm', 'bz_gsm', 'density', 'speed']
    
    # Interpolar apenas nas lacunas pequenas
    for col in essential_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=3, limit_direction='both')
    
    # Remover linhas onde dados essenciais ainda estão ausentes
    df = df.dropna(subset=essential_cols)
    
    return df

# ============================
# CÁLCULO FÍSICO DO HAC CORRIGIDO
# ============================

def calculate_physical_hac(df):
    """
    Calcula o HAC com o modelo físico correto:
    - Injeção quando Bz < 0
    - Decaimento exponencial quando Bz ≥ 0
    - Memória de energia acumulada
    """
    print("⚡ Calculando HAC com modelo físico...")
    
    # Converter para unidades SI
    Bz_nT = df['bz_gsm'].values
    Bz = Bz_nT * 1e-9  # nT → T
    
    V_km = df['speed'].values
    V = V_km * 1e3  # km/s → m/s
    
    n_cc = df['density'].values
    n = n_cc * 1e6  # cm⁻³ → m⁻³
    
    # Campo magnético total (B²)
    Bx = df.get('bx_gsm', pd.Series(0)).values * 1e-9
    By = df.get('by_gsm', pd.Series(0)).values * 1e-9
    B = np.sqrt(Bx**2 + By**2 + Bz**2)  # Magnitude total (T)
    
    # Calcular delta_t real
    times = pd.to_datetime(df['time_tag']).values
    dt_seconds = np.zeros(len(times))
    
    if len(times) > 1:
        time_diffs = np.diff(times)
        dt_seconds[1:] = time_diffs.astype('timedelta64[s]').astype(np.float64)
        dt_seconds[0] = dt_seconds[1] if len(dt_seconds) > 1 else 60.0
    else:
        dt_seconds[:] = 60.0
    
    # Inicializar array HAC
    hac = np.zeros(len(times))
    
    print(f"   Tempo de decaimento τ = {DECAY_TAU/3600:.1f} horas")
    print(f"   Ganho de injeção = {INJECTION_GAIN:.1e}")
    
    # Loop físico principal
    for i in range(1, len(hac)):
        # 1. Calcular injeção (apenas quando Bz < 0)
        if Bz[i] < 0:
            # Pressão dinâmica: ρV² = (n·m_p)·V²
            # Como m_p é constante, absorvemos na constante
            dynamic_term = n[i] * (V[i]**2)
            injection = (dynamic_term ** BETA) * (B[i]**2)
        else:
            injection = 0.0
        
        # 2. Calcular decaimento (sempre presente)
        # Decaimento exponencial: exp(-Δt/τ)
        decay_factor = np.exp(-dt_seconds[i] / DECAY_TAU)
        
        # 3. Atualizar HAC: decaimento + injeção
        hac[i] = hac[i-1] * decay_factor + injection * dt_seconds[i]
    
    # Aplicar ganho e saturação
    hac_scaled = hac * INJECTION_GAIN
    hac_scaled = np.clip(hac_scaled, 0, MAX_HAC)
    
    # Adicionar ao DataFrame
    df['HAC'] = hac_scaled
    df['Bz_nT'] = Bz_nT
    df['V_km'] = V_km
    df['n_cc'] = n_cc
    
    # Calcular taxa de variação
    df['HAC_rate'] = np.gradient(df['HAC'], dt_seconds)
    
    print(f"   HAC final: {df['HAC'].iloc[-1]:.2f}")
    print(f"   HAC máximo: {df['HAC'].max():.2f}")
    
    return df

def classify_storm(hac_value):
    """Classifica o nível da tempestade baseado no HAC."""
    for name, emoji, min_val, max_val in STORM_LEVELS:
        if min_val <= hac_value < max_val:
            return name, emoji
    return "G5", "⚫"

# ============================
# VISUALIZAÇÃO
# ============================

def create_hac_plot(df, filename="hac_forecast_corrected.png"):
    """Cria gráfico do HAC corrigido."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), 
                           gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Painel 1: HAC
    ax1 = axes[0]
    ax1.plot(df['time_tag'], df['HAC'], 
             color='#d62728', linewidth=3, label='HAC')
    
    # Área sob a curva
    ax1.fill_between(df['time_tag'], 0, df['HAC'], 
                     alpha=0.2, color='#d62728')
    
    # Thresholds
    for name, emoji, min_val, max_val in STORM_LEVELS:
        if min_val > 0:
            ax1.axhline(y=min_val, color='gray', linestyle='--', 
                       alpha=0.5, linewidth=0.8)
            ax1.text(df['time_tag'].iloc[0], min_val+5, 
                    f' {name}', va='bottom', fontsize=9)
    
    ax1.set_ylabel('HAC Index', fontsize=12, fontweight='bold')
    ax1.set_title('Heliospheric Accumulated Coupling (HAC) - Modelo Físico Corrigido', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 320)
    
    # Painel 2: Bz com destaque para períodos de injeção
    ax2 = axes[1]
    ax2.plot(df['time_tag'], df['Bz_nT'], 
             color='#2ca02c', linewidth=1.5, label='Bz')
    
    # Destacar períodos de IMF sul (Bz < 0)
    ax2.fill_between(df['time_tag'], 0, df['Bz_nT'], 
                     where=(df['Bz_nT'] < 0), 
                     alpha=0.5, color='red', label='IMF Sul (injeção)')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax2.set_ylabel('Bz [nT]', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Painel 3: Velocidade do vento solar
    ax3 = axes[2]
    ax3.plot(df['time_tag'], df['V_km'], 
             color='#1f77b4', linewidth=1.5, label='Velocidade')
    ax3.set_ylabel('V [km/s]', fontsize=11)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Painel 4: Densidade
    ax4 = axes[3]
    ax4.plot(df['time_tag'], df['n_cc'], 
             color='#9467bd', linewidth=1.5, label='Densidade')
    ax4.set_ylabel('n [cm⁻³]', fontsize=11)
    ax4.set_xlabel('Time (UTC)', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico salvo: {filename}")
    return filename

def create_simple_hac_plot(df, filename="hac_simple_corrected.png"):
    """Gráfico simples apenas do HAC."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot principal
    ax.plot(df['time_tag'], df['HAC'], 
            color='red', linewidth=2.5, label='HAC')
    
    # Área colorida por nível de tempestade
    colors = ['green', 'yellow', 'orange', 'red', 'purple', 'black']
    
    for i, (name, emoji, min_val, max_val) in enumerate(STORM_LEVELS):
        if i < len(STORM_LEVELS) - 1:
            ax.axhspan(min_val, max_val, alpha=0.1, color=colors[i])
            ax.text(df['time_tag'].iloc[-1] + timedelta(hours=1), 
                   (min_val + max_val) / 2, name,
                   va='center', fontsize=9, color=colors[i])
    
    ax.set_xlabel('Time (UTC)', fontsize=11)
    ax.set_ylabel('HAC Index', fontsize=11, fontweight='bold')
    ax.set_title('Heliospheric Accumulated Coupling - Modelo Corrigido', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 320)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico simples salvo: {filename}")
    return filename

# ============================
# ANÁLISE FÍSICA
# ============================

def analyze_physical_behavior(df):
    """Analisa o comportamento físico do HAC."""
    print("\n📊 ANÁLISE FÍSICA DO HAC")
    print("="*50)
    
    # Últimos valores
    last = df.iloc[-1]
    hac_value = last['HAC']
    storm_level, emoji = classify_storm(hac_value)
    
    print(f"\n⏰ Última medição: {last['time_tag']}")
    print(f"⚡ HAC atual: {hac_value:.2f}")
    print(f"🌪️  Nível: {emoji} {storm_level}")
    
    # Estatísticas de injeção
    negative_bz_mask = df['Bz_nT'] < 0
    positive_bz_mask = df['Bz_nT'] >= 0
    
    print(f"\n📈 Estatísticas de Bz:")
    print(f"   • IMF Sul (Bz < 0): {negative_bz_mask.sum()} pontos")
    print(f"   • IMF Norte (Bz ≥ 0): {positive_bz_mask.sum()} pontos")
    print(f"   • Fração de tempo com IMF Sul: {negative_bz_mask.mean()*100:.1f}%")
    
    if negative_bz_mask.any():
        print(f"   • Bz mínimo durante IMF Sul: {df.loc[negative_bz_mask, 'Bz_nT'].min():.1f} nT")
        print(f"   • Bz médio durante IMF Sul: {df.loc[negative_bz_mask, 'Bz_nT'].mean():.1f} nT")
    
    # Comportamento do HAC
    print(f"\n📊 Comportamento do HAC:")
    print(f"   • HAC mínimo: {df['HAC'].min():.2f}")
    print(f"   • HAC máximo: {df['HAC'].max():.2f}")
    print(f"   • HAC médio: {df['HAC'].mean():.2f}")
    
    # Taxa de variação
    if 'HAC_rate' in df.columns:
        positive_rate = df[df['HAC_rate'] > 0]
        negative_rate = df[df['HAC_rate'] < 0]
        
        print(f"   • Períodos de crescimento: {len(positive_rate)} pontos")
        print(f"   • Períodos de decaimento: {len(negative_rate)} pontos")
        
        if len(positive_rate) > 0:
            print(f"   • Taxa média de crescimento: {positive_rate['HAC_rate'].mean():.4f}/s")
        if len(negative_rate) > 0:
            print(f"   • Taxa média de decaimento: {negative_rate['HAC_rate'].mean():.4f}/s")
    
    # Eficiência de acoplamento
    print(f"\n🔧 Eficiência de acoplamento:")
    print(f"   • β (expoente): {BETA}")
    print(f"   • τ (tempo de decaimento): {DECAY_TAU/3600:.1f} horas")
    print(f"   • Ganho: {INJECTION_GAIN:.2e}")
    
    # Sugestões para ajuste
    print(f"\n💡 SUGESTÕES PARA AJUSTE:")
    
    if hac_value < 50 and df['Bz_nT'].min() < -10:
        print("   ⚠️  HAC baixo apesar de Bz negativo forte")
        print("   → Aumentar INJECTION_GAIN ou diminuir DECAY_TAU")
    
    if df['HAC'].max() > 250:
        print("   ⚠️  HAC atingiu saturação")
        print("   → Diminuir INJECTION_GAIN ou aumentar DECAY_TAU")
    
    if negative_bz_mask.mean() > 0.7 and hac_value < 100:
        print("   ⚠️  IMF predominantemente sul mas HAC moderado")
        print("   → Verificar velocidade e densidade do vento solar")
    
    print("\n" + "="*50)

# ============================
# FUNÇÃO PRINCIPAL
# ============================

def main():
    print("\n" + "="*70)
    print("🛰️  HAC - MODELO FÍSICO CORRIGIDO")
    print("(Com injeção Bz<0 + decaimento exponencial)")
    print("="*70)
    
    # 1. Carregar dados
    print("\n📥 Carregando dados...")
    mag_df = load_omni_data(MAG_FILE)
    plasma_df = load_omni_data(PLASMA_FILE)
    
    if mag_df is None or plasma_df is None:
        print("❌ Falha ao carregar dados")
        return
    
    print(f"   Dados magnéticos: {len(mag_df)} pontos")
    print(f"   Dados de plasma: {len(plasma_df)} pontos")
    
    # 2. Preparar dados
    print("\n🔧 Preparando dados...")
    df = prepare_and_merge_data(mag_df, plasma_df)
    
    if len(df) < 10:
        print("❌ Dados insuficientes após preparação")
        return
    
    print(f"   Dados fundidos: {len(df)} pontos")
    print(f"   Período: {df['time_tag'].min()} a {df['time_tag'].max()}")
    
    # 3. Calcular HAC com modelo físico
    df = calculate_physical_hac(df)
    
    # 4. Análise física
    analyze_physical_behavior(df)
    
    # 5. Gerar gráficos
    print("\n📈 Gerando gráficos...")
    create_hac_plot(df, "hac_forecast_corrected.png")
    create_simple_hac_plot(df, "hac_simple_corrected.png")
    
    # 6. Salvar dados processados
    try:
        output_file = "hac_data_corrected.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Dados salvos em: {output_file}")
        
        # Salvar resumo
        summary_file = "hac_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("RESUMO HAC - MODELO FÍSICO CORRIGIDO\n")
            f.write("="*40 + "\n\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parâmetros: β={BETA}, τ={DECAY_TAU/3600:.1f}h, G={INJECTION_GAIN:.2e}\n\n")
            f.write(f"Último HAC: {df['HAC'].iloc[-1]:.2f}\n")
            f.write(f"HAC máximo: {df['HAC'].max():.2f}\n")
            f.write(f"IMF Sul: {(df['Bz_nT'] < 0).mean()*100:.1f}% do tempo\n")
        
        print(f"📝 Resumo salvo em: {summary_file}")
        
    except Exception as e:
        print(f"⚠️  Erro ao salvar dados: {e}")
    
    # 7. Status final
    last_hac = df['HAC'].iloc[-1]
    storm_level, emoji = classify_storm(last_hac)
    
    print("\n" + "="*70)
    print("🎯 STATUS FINAL")
    print("="*70)
    print(f"\n{emoji} TEMPESTADE {storm_level}")
    print(f"HAC: {last_hac:.2f}")
    
    if last_hac >= 200:
        print("🚨 ALERTA: Tempestade geomagnética severa (G4/G5)")
    elif last_hac >= 150:
        print("⚠️  ALERTA: Tempestade forte (G3)")
    elif last_hac >= 100:
        print("📢 Atenção: Tempestade moderada (G2)")
    elif last_hac >= 50:
        print("📋 Monitoramento: Tempestade menor (G1)")
    else:
        print("✅ Condições quietas")
    
    print(f"\n📊 Thresholds:")
    for name, emoji, min_val, max_val in STORM_LEVELS:
        print(f"   {emoji} {name}: {min_val} ≤ HAC < {max_val}")
    
    print("\n" + "="*70)
    print("✅ PROCESSAMENTO CONCLUÍDO!")
    print("="*70)

# ============================
# EXECUÇÃO
# ============================

if __name__ == "__main__":
    main()
