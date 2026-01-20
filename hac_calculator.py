import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURAÇÃO FÍSICA CALIBRADA
# ============================

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

# Parâmetros calibrados para escala correta
BETA = 0.65
DECAY_TAU = 3.0 * 3600  # 3 horas em segundos
INJECTION_GAIN = 1e10    # Ganho calibrado para escala 0-300
MAX_HAC = 300

# Thresholds para classificação
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
    """Carrega dados OMNI no formato JSON."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
        
        for col in headers:
            if col != 'time_tag':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"❌ Erro ao carregar {filepath}: {e}")
        return None

def prepare_data(mag_df, plasma_df):
    """Prepara e funde dados magnéticos e de plasma."""
    # Limpeza básica
    mag_df = mag_df.dropna(subset=['time_tag']).copy()
    plasma_df = plasma_df.dropna(subset=['time_tag']).copy()
    
    # Fundir
    df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
    df = df.sort_values('time_tag').reset_index(drop=True)
    
    # Interpolar pequenas lacunas
    essential_cols = ['bx_gsm', 'by_gsm', 'bz_gsm', 'density', 'speed']
    for col in essential_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=3, limit_direction='both')
    
    # Remover linhas com dados essenciais ausentes
    df = df.dropna(subset=essential_cols)
    
    return df

# ============================
# CÁLCULO FÍSICO DO HAC CALIBRADO
# ============================

def calculate_physical_hac(df, beta=BETA, tau=DECAY_TAU, gain=INJECTION_GAIN):
    """
    Calcula o HAC com escala física correta.
    - Injeção: (n*V²)^β * B² quando Bz < 0
    - Decaimento: exponencial com tempo τ
    - Ganho: calibrado para escala 0-300
    """
    print("⚡ Calculando HAC com modelo calibrado...")
    print(f"   β = {beta}")
    print(f"   τ = {tau/3600:.1f} horas")
    print(f"   Ganho = {gain:.1e}")
    
    # Extrair dados e converter para SI
    Bz_nT = df['bz_gsm'].values
    Bz = Bz_nT * 1e-9  # nT → T
    
    V_km = df['speed'].values
    V = V_km * 1e3  # km/s → m/s
    
    n_cc = df['density'].values
    n = n_cc * 1e6  # cm⁻³ → m⁻³
    
    # Calcular B total
    Bx = df.get('bx_gsm', pd.Series(0)).values * 1e-9
    By = df.get('by_gsm', pd.Series(0)).values * 1e-9
    B = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    # Calcular delta_t real (em segundos)
    times = pd.to_datetime(df['time_tag']).values
    dt = np.zeros(len(times))
    if len(times) > 1:
        time_diffs = np.diff(times)
        dt[1:] = time_diffs.astype('timedelta64[s]').astype(np.float64)
        dt[0] = dt[1] if len(dt) > 1 else 60.0
    else:
        dt[:] = 60.0
    
    # Inicializar HAC
    hac = np.zeros(len(times))
    
    # Loop físico com decaimento e injeção
    for i in range(1, len(hac)):
        # 1. Aplicar decaimento exponencial (sempre)
        decay_factor = np.exp(-dt[i] / tau)
        hac[i] = hac[i-1] * decay_factor
        
        # 2. Adicionar injeção se Bz < 0
        if Bz[i] < 0:
            # Termo de pressão dinâmica (sem massa próton, conforme artigo)
            dynamic_term = n[i] * (V[i]**2)
            
            # Adicionar epsilon para estabilidade numérica
            eps = 1e-30
            
            # Calcular injeção: (nV²)^β * B²
            injection = (np.abs(dynamic_term + eps)**beta) * (B[i]**2 + eps)
            
            # Adicionar injeção com ganho
            hac[i] += injection * dt[i] * gain
    
    # Aplicar saturação física
    hac = np.clip(hac, 0, MAX_HAC)
    
    # Adicionar ao DataFrame
    df['HAC'] = hac
    df['Bz_nT'] = Bz_nT
    df['V_km'] = V_km
    df['n_cc'] = n_cc
    df['dt_seconds'] = dt
    
    # Calcular taxa de variação
    df['HAC_rate'] = np.gradient(df['HAC'], dt)
    
    print(f"   HAC final: {df['HAC'].iloc[-1]:.2f}")
    print(f"   HAC máximo: {df['HAC'].max():.2f}")
    print(f"   Fração IMF sul: {(df['Bz_nT'] < 0).mean()*100:.1f}%")
    
    return df

def classify_storm(hac_value):
    """Classifica o nível da tempestade."""
    for name, emoji, min_val, max_val in STORM_LEVELS:
        if min_val <= hac_value < max_val:
            return name, emoji
    return "G5", "⚫"

# ============================
# VISUALIZAÇÃO
# ============================

def create_combined_plot(df, filename="hac_forecast_calibrated.png"):
    """Cria gráfico combinado do HAC calibrado."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), 
                           gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Painel 1: HAC
    ax1 = axes[0]
    ax1.plot(df['time_tag'], df['HAC'], 
             color='#d62728', linewidth=2.5, label='HAC')
    
    # Área sob a curva
    ax1.fill_between(df['time_tag'], 0, df['HAC'], 
                     alpha=0.2, color='#d62728')
    
    # Thresholds
    for name, emoji, min_val, max_val in STORM_LEVELS:
        if min_val > 0:
            ax1.axhline(y=min_val, color='gray', linestyle='--', 
                       alpha=0.5, linewidth=0.8)
            ax1.text(df['time_tag'].iloc[0], min_val+2, 
                    f' {name}', va='bottom', fontsize=8)
    
    ax1.set_ylabel('HAC Index', fontsize=12, fontweight='bold')
    ax1.set_title('Heliospheric Accumulated Coupling - Modelo Calibrado', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 320)
    
    # Painel 2: Bz com destaque para injeção
    ax2 = axes[1]
    ax2.plot(df['time_tag'], df['Bz_nT'], 
             color='#2ca02c', linewidth=1.5, label='Bz')
    ax2.fill_between(df['time_tag'], 0, df['Bz_nT'], 
                     where=(df['Bz_nT'] < 0), 
                     alpha=0.4, color='red', label='IMF Sul (injeção)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax2.set_ylabel('Bz [nT]', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Painel 3: Velocidade
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

# ============================
# ANÁLISE E CALIBRAÇÃO
# ============================

def analyze_calibration(df):
    """Analisa a calibração do modelo."""
    print("\n📊 ANÁLISE DE CALIBRAÇÃO")
    print("="*50)
    
    last = df.iloc[-1]
    hac_value = last['HAC']
    storm_level, emoji = classify_storm(hac_value)
    
    print(f"\n⏰ Última medição: {last['time_tag']}")
    print(f"⚡ HAC: {hac_value:.2f}")
    print(f"🌪️  Nível: {emoji} {storm_level}")
    
    # Estatísticas básicas
    print(f"\n📈 Estatísticas:")
    print(f"   • HAC mínimo: {df['HAC'].min():.2f}")
    print(f"   • HAC máximo: {df['HAC'].max():.2f}")
    print(f"   • HAC médio: {df['HAC'].mean():.2f}")
    
    # Comportamento do Bz
    neg_bz = df[df['Bz_nT'] < 0]
    pos_bz = df[df['Bz_nT'] >= 0]
    
    print(f"\n🧲 Comportamento do Bz:")
    print(f"   • IMF Sul: {len(neg_bz)} pontos ({len(neg_bz)/len(df)*100:.1f}%)")
    print(f"   • IMF Norte: {len(pos_bz)} pontos ({len(pos_bz)/len(df)*100:.1f}%)")
    
    if len(neg_bz) > 0:
        print(f"   • Bz mínimo (sul): {neg_bz['Bz_nT'].min():.1f} nT")
        print(f"   • Bz médio (sul): {neg_bz['Bz_nT'].mean():.1f} nT")
    
    # Eficiência de injeção
    if 'HAC_rate' in df.columns:
        growth_periods = df[df['HAC_rate'] > 0]
        decay_periods = df[df['HAC_rate'] < 0]
        
        print(f"\n📊 Dinâmica do HAC:")
        print(f"   • Períodos de crescimento: {len(growth_periods)}")
        print(f"   • Períodos de decaimento: {len(decay_periods)}")
        
        if len(growth_periods) > 0:
            mean_growth = growth_periods['HAC_rate'].mean()
            print(f"   • Taxa média de crescimento: {mean_growth:.4f}/s")
        
        if len(decay_periods) > 0:
            mean_decay = decay_periods['HAC_rate'].mean()
            print(f"   • Taxa média de decaimento: {mean_decay:.4f}/s")
    
    # Sugestões de ajuste
    print(f"\n💡 SUGESTÕES PARA CALIBRAÇÃO:")
    
    if hac_value < 50 and df['Bz_nT'].min() < -20:
        print("   ⚠️  HAC baixo apesar de Bz negativo forte")
        print("   → Aumentar INJECTION_GAIN (atual: {:.1e})".format(INJECTION_GAIN))
    
    if df['HAC'].max() > 280:
        print("   ⚠️  HAC próximo à saturação")
        print("   → Diminuir INJECTION_GAIN ou aumentar DECAY_TAU")
    
    if (df['Bz_nT'] < 0).mean() > 0.5 and hac_value < 100:
        print("   ⚠️  IMF predominantemente sul mas HAC moderado")
        print("   → Aumentar β (atual: {})".format(BETA))
    
    print("\n" + "="*50)

# ============================
# FUNÇÃO PRINCIPAL COM CALIBRAÇÃO
# ============================

def calibrate_and_run():
    """Executa o cálculo com calibração interativa."""
    print("\n" + "="*70)
    print("🎯 HAC - CALIBRAÇÃO FÍSICA")
    print("="*70)
    
    # 1. Carregar dados
    print("\n📥 Carregando dados...")
    mag_df = load_omni_data(MAG_FILE)
    plasma_df = load_omni_data(PLASMA_FILE)
    
    if mag_df is None or plasma_df is None:
        print("❌ Falha ao carregar dados")
        return None
    
    # 2. Preparar dados
    print("🔧 Preparando dados...")
    df = prepare_data(mag_df, plasma_df)
    
    if len(df) < 10:
        print("❌ Dados insuficientes")
        return None
    
    print(f"   Período: {df['time_tag'].min()} a {df['time_tag'].max()}")
    print(f"   Pontos: {len(df)}")
    
    # 3. Calcular HAC com diferentes calibrações
    print("\n⚙️  Testando calibrações...")
    
    # Calibração 1: Valores padrão
    df1 = calculate_physical_hac(df.copy())
    
    # Calibração 2: Maior ganho para eventos fracos
    if df1['HAC'].max() < 100:
        print("\n🔄 Tentando calibração com maior ganho...")
        df2 = calculate_physical_hac(df.copy(), gain=INJECTION_GAIN*5)
        if df2['HAC'].max() > df1['HAC'].max():
            df1 = df2
    
    # Calibração 3: Menor tempo de decaimento para resposta mais rápida
    if df1['HAC'].max() > 280:
        print("\n🔄 Tentando calibração com menor τ...")
        df3 = calculate_physical_hac(df.copy(), tau=2.0*3600)
        df1 = df3
    
    # 4. Análise
    analyze_calibration(df1)
    
    # 5. Gráfico
    print("\n📈 Gerando gráficos...")
    create_combined_plot(df1, "hac_calibrated_final.png")
    
    # 6. Salvar resultados
    try:
        output_file = "hac_calibrated_results.csv"
        df1.to_csv(output_file, index=False)
        print(f"\n💾 Dados salvos: {output_file}")
        
        # Salvar configuração
        config_file = "hac_config.txt"
        with open(config_file, 'w') as f:
            f.write("CONFIGURAÇÃO HAC CALIBRADA\n")
            f.write("="*40 + "\n\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"β: {BETA}\n")
            f.write(f"τ: {DECAY_TAU/3600:.1f} horas\n")
            f.write(f"Ganho: {INJECTION_GAIN:.1e}\n\n")
            f.write(f"HAC final: {df1['HAC'].iloc[-1]:.2f}\n")
            f.write(f"HAC máximo: {df1['HAC'].max():.2f}\n")
            f.write(f"IMF Sul: {(df1['Bz_nT'] < 0).mean()*100:.1f}%\n")
        
        print(f"📝 Configuração salva: {config_file}")
        
    except Exception as e:
        print(f"⚠️  Erro ao salvar: {e}")
    
    return df1

def main():
    """Função principal."""
    df = calibrate_and_run()
    
    if df is not None:
        # Status final
        last_hac = df['HAC'].iloc[-1]
        storm_level, emoji = classify_storm(last_hac)
        
        print("\n" + "="*70)
        print("🎯 STATUS FINAL")
        print("="*70)
        print(f"\n{emoji} {storm_level}")
        print(f"HAC: {last_hac:.2f}")
        
        # Alertas
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
        
        print("\n📊 Thresholds aplicados:")
        for name, emoji, min_val, max_val in STORM_LEVELS:
            print(f"   {emoji} {name}: {min_val} a {max_val}")
        
        print("\n" + "="*70)
        print("✅ PROCESSAMENTO CONCLUÍDO!")
        print("="*70)

# ============================
# EXECUÇÃO
# ============================

if __name__ == "__main__":
    main()
