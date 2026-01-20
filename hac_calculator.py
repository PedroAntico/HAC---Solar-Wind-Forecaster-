import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURAÇÃO
# ============================

MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

BETA = 0.65  # Expoente de acoplamento
DT = 60  # Intervalo de tempo em segundos (1 minuto)
NORMALIZATION = 1e-15  # Fator de normalização ajustado para dados reais

# Thresholds para classificação de tempestades (ajustáveis)
THRESHOLDS = {
    "Quiet": 0,
    "G1": 50,
    "G2": 100,
    "G3": 150,
    "G4": 200,
    "G5": 250
}

# Cores para os níveis de tempestade
STORM_COLORS = {
    "Quiet": "green",
    "G1": "yellow",
    "G2": "orange",
    "G3": "red",
    "G4": "purple",
    "G5": "black"
}

# ============================
# FUNÇÕES DE CARREGAMENTO
# ============================

def load_omni_data(filepath):
    """
    Carrega dados OMNI do formato JSON.
    Lida com valores null e converte para DataFrame.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Primeira linha contém os cabeçalhos
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        
        # Converter time_tag para datetime
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
        
        # Converter colunas numéricas, tratando null como NaN
        for col in headers:
            if col != 'time_tag':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✅ {filepath}: {len(df)} registros carregados")
        return df
    
    except Exception as e:
        print(f"❌ Erro ao carregar {filepath}: {e}")
        return None

# ============================
# PROCESSAMENTO DE DADOS
# ============================

def prepare_data(mag_df, plasma_df):
    """
    Prepara e funde os dados magnéticos e de plasma.
    """
    if mag_df is None or plasma_df is None:
        return None
    
    print("\n🔧 Preparando dados...")
    
    # Garantir que temos as colunas necessárias
    required_mag_cols = ['bx_gsm', 'by_gsm', 'bz_gsm']
    required_plasma_cols = ['density', 'speed']
    
    for col in required_mag_cols:
        if col not in mag_df.columns:
            print(f"⚠️  Coluna {col} não encontrada nos dados magnéticos")
            return None
    
    for col in required_plasma_cols:
        if col not in plasma_df.columns:
            print(f"⚠️  Coluna {col} não encontrada nos dados de plasma")
            return None
    
    # Renomear colunas para evitar conflitos
    mag_df = mag_df.copy()
    plasma_df = plasma_df.copy()
    
    mag_rename = {col: f'mag_{col}' for col in mag_df.columns if col != 'time_tag'}
    plasma_rename = {col: f'plasma_{col}' for col in plasma_df.columns if col != 'time_tag'}
    
    mag_df.rename(columns=mag_rename, inplace=True)
    plasma_df.rename(columns=plasma_rename, inplace=True)
    
    # Fundir os dados
    df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
    df = df.sort_values('time_tag').reset_index(drop=True)
    
    print(f"   Dados fundidos: {len(df)} registros")
    
    # Verificar dados ausentes
    missing_before = df[['mag_bz_gsm', 'plasma_speed', 'plasma_density']].isnull().sum()
    print(f"   Valores ausentes antes da limpeza:")
    for col, count in missing_before.items():
        print(f"     {col}: {count}")
    
    # Preencher valores ausentes com interpolação limitada
    numeric_cols = [col for col in df.columns if col != 'time_tag']
    df[numeric_cols] = df[numeric_cols].interpolate(
        method='linear', 
        limit=3, 
        limit_direction='both'
    )
    
    # Remover linhas com valores essenciais ainda ausentes
    essential_cols = ['mag_bz_gsm', 'plasma_speed', 'plasma_density']
    df = df.dropna(subset=essential_cols)
    
    missing_after = df[essential_cols].isnull().sum()
    print(f"   Valores ausentes após limpeza: {missing_after.sum()}")
    
    return df

def calculate_hac(df):
    """
    Calcula o índice HAC conforme definido no artigo.
    HAC(t) = ∫ [(n·V²)^β · B² · Θ(-Bz)] dt
    """
    print("\n⚡ Calculando HAC...")
    
    # Extrair dados com tipo float64 para precisão
    Bz = df['mag_bz_gsm'].values.astype(np.float64) * 1e-9  # nT → T
    V = df['plasma_speed'].values.astype(np.float64) * 1e3  # km/s → m/s
    n = df['plasma_density'].values.astype(np.float64) * 1e6  # cm⁻³ → m⁻³
    
    # Calcular magnitude total do campo B (B² conforme artigo)
    if 'mag_bt' in df.columns:
        # Usar bt se disponível
        B = df['mag_bt'].values.astype(np.float64) * 1e-9
    else:
        # Calcular a partir dos componentes
        Bx = df.get('mag_bx_gsm', pd.Series(0)).values.astype(np.float64) * 1e-9
        By = df.get('mag_by_gsm', pd.Series(0)).values.astype(np.float64) * 1e-9
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    # Condição de IMF sul (Bz < 0)
    southward = np.where(Bz < 0, 1.0, 0.0)
    
    # Termo de pressão dinâmica: n·V²
    dynamic_pressure = n * V**2
    
    # Adicionar epsilon para evitar problemas numéricos
    eps = 1e-10
    
    # Termo de acoplamento: (n·V²)^β · B²
    coupling = (np.abs(dynamic_pressure + eps)**BETA) * (B**2 + eps)
    
    # Aplicar condição de IMF sul
    integrand = coupling * southward
    
    # Calcular delta_t real entre medições
    times = pd.to_datetime(df['time_tag']).values
    delta_t = np.zeros(len(times))
    
    if len(times) > 1:
        # Converter diferenças de tempo para segundos
        time_diffs = np.diff(times)
        delta_t[1:] = time_diffs.astype('timedelta64[s]').astype(np.float64)
        delta_t[0] = delta_t[1] if len(delta_t) > 1 else DT
    else:
        delta_t[:] = DT
    
    # Garantir que não há intervalos negativos ou zero
    delta_t = np.maximum(delta_t, 1.0)
    
    # Calcular integral cumulativa
    hac_raw = np.cumsum(integrand * delta_t)
    
    # Normalizar
    hac_normalized = hac_raw * NORMALIZATION
    
    # Adicionar ao DataFrame
    df['HAC'] = hac_normalized
    df['HAC_raw'] = hac_raw
    
    # Calcular derivada (taxa de mudança)
    df['HAC_rate'] = np.gradient(hac_normalized, delta_t)
    
    print(f"   HAC mínimo: {df['HAC'].min():.2f}")
    print(f"   HAC máximo: {df['HAC'].max():.2f}")
    print(f"   HAC final: {df['HAC'].iloc[-1]:.2f}")
    
    return df

def classify_storm_level(hac_value):
    """
    Classifica o nível da tempestade geomagnética baseado no HAC.
    """
    for level, threshold in THRESHOLDS.items():
        if hac_value < threshold:
            return level
    
    return "G5"

# ============================
# VISUALIZAÇÃO
# ============================

def create_publication_plot(df, filename="hac_forecast.png"):
    """
    Cria gráfico de qualidade para publicação.
    """
    print(f"\n📈 Criando gráfico: {filename}")
    
    # Configurar estilo do gráfico
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(14, 10))
    
    # Painel 1: HAC principal
    ax1 = plt.subplot(3, 1, 1)
    
    # Plotar HAC
    ax1.plot(df['time_tag'], df['HAC'], 
             color='#d62728', linewidth=2.5, 
             label='Heliospheric Accumulated Coupling (HAC)')
    
    # Preencher área sob a curva
    ax1.fill_between(df['time_tag'], 0, df['HAC'], 
                     alpha=0.2, color='#d62728')
    
    # Adicionar linhas de threshold
    for level, threshold in THRESHOLDS.items():
        if level != "Quiet" and threshold > 0:
            color = STORM_COLORS.get(level, 'gray')
            ax1.axhline(y=threshold, color=color, linestyle='--', 
                       alpha=0.7, linewidth=1.5,
                       label=f'{level} Threshold')
    
    # Configurar eixos
    ax1.set_ylabel('HAC Index', fontsize=12, fontweight='bold')
    ax1.set_title('Heliospheric Accumulated Coupling (HAC) - January 2026 Geomagnetic Storm', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim([df['time_tag'].min(), df['time_tag'].max()])
    
    # Painel 2: Componente Bz
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    
    ax2.plot(df['time_tag'], df['mag_bz_gsm'], 
             color='#2ca02c', linewidth=1.5, 
             label='Bz (GSM)')
    
    # Destacar períodos de IMF sul
    ax2.fill_between(df['time_tag'], 0, df['mag_bz_gsm'], 
                     where=(df['mag_bz_gsm'] < 0), 
                     alpha=0.5, color='red', label='Southward IMF')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    
    ax2.set_ylabel('Bz [nT]', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.4)
    
    # Painel 3: Velocidade do vento solar
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    
    ax3.plot(df['time_tag'], df['plasma_speed'], 
             color='#1f77b4', linewidth=1.5, 
             label='Solar Wind Speed')
    
    ax3.set_ylabel('Speed [km/s]', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time (UTC)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.4)
    
    # Formatar datas no eixo x
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    
    # Salvar em alta resolução
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Gráfico salvo como {filename}")
    return filename

def create_simple_plot(df, filename="hac_simple.png"):
    """
    Cria um gráfico simples apenas com HAC.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotar HAC
    ax.plot(df['time_tag'], df['HAC'], 
            color='red', linewidth=2, 
            label='HAC')
    
    # Adicionar thresholds
    for level, threshold in THRESHOLDS.items():
        if level != "Quiet" and threshold > 0:
            color = STORM_COLORS.get(level, 'gray')
            ax.axhline(y=threshold, color=color, linestyle='--', 
                      alpha=0.5, label=f'{level}')
    
    ax.set_xlabel('Time (UTC)', fontsize=11)
    ax.set_ylabel('HAC Index', fontsize=11, fontweight='bold')
    ax.set_title('Heliospheric Accumulated Coupling', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Formatar datas
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico simples salvo como {filename}")
    return filename

# ============================
# ANÁLISE E RELATÓRIO
# ============================

def generate_report(df):
    """
    Gera um relatório detalhado da análise.
    """
    if df is None or len(df) == 0:
        return
    
    print("\n" + "="*70)
    print("📊 RELATÓRIO DE ANÁLISE HAC")
    print("="*70)
    
    # Última leitura
    last_row = df.iloc[-1]
    current_hac = last_row['HAC']
    storm_level = classify_storm_level(current_hac)
    
    print(f"\n⏰ Última leitura: {last_row['time_tag']}")
    print(f"⚡ HAC atual: {current_hac:.2f}")
    print(f"🌪️  Nível da tempestade: {storm_level}")
    
    # Parâmetros atuais
    print(f"\n📈 Parâmetros atuais:")
    print(f"   • Bz: {last_row.get('mag_bz_gsm', 'N/A'):.2f} nT")
    print(f"   • Velocidade: {last_row.get('plasma_speed', 'N/A'):.1f} km/s")
    print(f"   • Densidade: {last_row.get('plasma_density', 'N/A'):.2f} cm⁻³")
    
    if 'mag_bt' in last_row:
        print(f"   • |B| total: {last_row['mag_bt']:.2f} nT")
    
    # Estatísticas
    print(f"\n📊 Estatísticas do período:")
    print(f"   • Período: {df['time_tag'].min()} a {df['time_tag'].max()}")
    print(f"   • Duração: {len(df)} pontos ({len(df)*DT/3600:.1f} horas)")
    print(f"   • HAC mínimo: {df['HAC'].min():.2f}")
    print(f"   • HAC máximo: {df['HAC'].max():.2f}")
    print(f"   • HAC médio: {df['HAC'].mean():.2f}")
    
    # Identificar picos
    if 'HAC_rate' in df.columns:
        positive_rate = df[df['HAC_rate'] > 0]
        if len(positive_rate) > 0:
            avg_growth = positive_rate['HAC_rate'].mean()
            print(f"   • Taxa média de crescimento: {avg_growth:.4f}/s")
    
    # Distribuição dos níveis
    print(f"\n🌍 Distribuição dos níveis de tempestade:")
    hac_values = df['HAC'].values
    for level, threshold in THRESHOLDS.items():
        if level == "Quiet":
            count = np.sum(hac_values < THRESHOLDS["G1"])
        elif level == "G5":
            count = np.sum(hac_values >= threshold)
        else:
            next_level = list(THRESHOLDS.keys())[list(THRESHOLDS.values()).index(threshold)+1]
            next_threshold = THRESHOLDS[next_level]
            count = np.sum((hac_values >= threshold) & (hac_values < next_threshold))
        
        percentage = (count / len(hac_values)) * 100
        print(f"   • {level}: {count} pontos ({percentage:.1f}%)")
    
    # Alertas
    print(f"\n⚠️  ALERTAS:")
    if current_hac >= THRESHOLDS["G4"]:
        print(f"   • ALERTA G4/G5: Tempestade geomagnética severa em progresso!")
    elif current_hac >= THRESHOLDS["G3"]:
        print(f"   • ALERTA G3: Tempestade forte em andamento")
    elif current_hac >= THRESHOLDS["G2"]:
        print(f"   • Alerta G2: Tempestade moderada")
    elif current_hac >= THRESHOLDS["G1"]:
        print(f"   • Alerta G1: Tempestade menor")
    else:
        print(f"   • Condições quietas")
    
    # Tendência
    if len(df) > 10:
        recent = df.tail(10)
        hac_trend = np.polyfit(range(len(recent)), recent['HAC'].values, 1)[0]
        
        print(f"\n📈 TENDÊNCIA:")
        if hac_trend > 0.1:
            print(f"   • ↗️  Crescimento significativo ({hac_trend:.2f}/min)")
        elif hac_trend > 0.01:
            print(f"   • ↗️  Crescimento moderado ({hac_trend:.2f}/min)")
        elif hac_trend < -0.1:
            print(f"   • ↘️  Decaimento significativo ({hac_trend:.2f}/min)")
        elif hac_trend < -0.01:
            print(f"   • ↘️  Decaimento moderado ({hac_trend:.2f}/min)")
        else:
            print(f"   • ➡️  Estável ({hac_trend:.2f}/min)")
    
    print("\n" + "="*70)
    
    # Salvar relatório em arquivo
    with open("hac_report.txt", "w") as f:
        f.write("RELATÓRIO DE ANÁLISE HAC\n")
        f.write("="*50 + "\n\n")
        f.write(f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Período dos dados: {df['time_tag'].min()} a {df['time_tag'].max()}\n")
        f.write(f"HAC final: {current_hac:.2f}\n")
        f.write(f"Nível classificado: {storm_level}\n\n")
        f.write("Estatísticas:\n")
        f.write(f"  - HAC mínimo: {df['HAC'].min():.2f}\n")
        f.write(f"  - HAC máximo: {df['HAC'].max():.2f}\n")
        f.write(f"  - HAC médio: {df['HAC'].mean():.2f}\n")
    
    print("📝 Relatório salvo como 'hac_report.txt'")

# ============================
# FUNÇÃO PRINCIPAL
# ============================

def main():
    """
    Função principal do script.
    """
    print("\n" + "="*70)
    print("🛰️  HELIOSPHERIC ACCUMULATED COUPLING (HAC) CALCULATOR")
    print("="*70)
    
    # 1. Carregar dados
    print("\n📥 Carregando dados OMNI...")
    mag_data = load_omni_data(MAG_FILE)
    plasma_data = load_omni_data(PLASMA_FILE)
    
    if mag_data is None or plasma_data is None:
        print("❌ Falha ao carregar dados. Verifique os arquivos.")
        return
    
    # 2. Preparar dados
    df = prepare_data(mag_data, plasma_data)
    
    if df is None or len(df) < 10:
        print("❌ Dados insuficientes após preparação.")
        return
    
    print(f"✅ Dados preparados: {len(df)} registros válidos")
    
    # 3. Calcular HAC
    df = calculate_hac(df)
    
    # 4. Gerar gráficos
    create_publication_plot(df, "hac_forecast_publication.png")
    create_simple_plot(df, "hac_forecast_simple.png")
    
    # 5. Gerar relatório
    generate_report(df)
    
    # 6. Salvar dados processados
    try:
        output_file = "hac_processed_data.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Dados processados salvos em '{output_file}'")
    except Exception as e:
        print(f"⚠️  Erro ao salvar dados: {e}")
    
    print("\n" + "="*70)
    print("✅ PROCESSAMENTO CONCLUÍDO!")
    print("="*70)
    print("\nArquivos gerados:")
    print("  • hac_forecast_publication.png - Gráfico para publicação")
    print("  • hac_forecast_simple.png - Gráfico simplificado")
    print("  • hac_report.txt - Relatório de análise")
    print("  • hac_processed_data.csv - Dados processados completos")
    print("\nPróximos passos:")
    print("  1. Verifique o gráfico 'hac_forecast_publication.png'")
    print("  2. Consulte o relatório 'hac_report.txt' para análise detalhada")
    print("  3. Use os thresholds no artigo: G1:50, G2:100, G3:150, G4:200, G5:250")
    print("\n")

# ============================
# EXECUÇÃO
# ============================

if __name__ == "__main__":
    main()
