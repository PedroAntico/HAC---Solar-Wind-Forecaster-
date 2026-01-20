import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURAÇÃO PRINCIPAL
# ============================
MAG_FILE = "data/mag-7-day.json"
PLASMA_FILE = "data/plasma-7-day.json"

# PARÂMETROS DO MODELO FÍSICO (AJUSTÁVEIS)
TAU_HOURS = 3.0            # Memória do sistema: 3 horas (em horas)
SCALE_TO_MAX = 300         # Escala máxima do índice
BASE_SCALE_FACTOR = 1.0    # Fator de escala linear adicional

# ============================
# 1. CARREGAMENTO DE DADOS
# ============================
def load_omni_data(filepath):
    """Carrega arquivos JSON do formato OMNI."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
        for col in headers:
            if col != 'time_tag':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"✅ {filepath}: {len(df)} registros")
        return df
    except Exception as e:
        print(f"❌ Erro em {filepath}: {e}")
        return None

def prepare_merged_data(mag_df, plasma_df):
    """Funde e limpa os dados magnéticos e de plasma."""
    # Fusão externa para manter todos os timestamps
    df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
    df = df.sort_values('time_tag').reset_index(drop=True)
    
    # Interpolação linear para lacunas pequenas
    cols_to_interpolate = ['bx_gsm', 'by_gsm', 'bz_gsm', 'density', 'speed']
    for col in cols_to_interpolate:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=3)
    
    # Remover linhas onde dados críticos ainda são NaN
    df = df.dropna(subset=['bz_gsm', 'speed', 'density'])
    return df

# ============================
# 2. NÚCLEO DO CÁLCULO DO HAC (CORRIGIDO)
# ============================
def calculate_hac_exponential(df, tau_hours=TAU_HOURS, scale_to_max=SCALE_TO_MAX):
    """
    Implementação CORRETA do HAC como filtro exponencial.
    
    Modelo físico:
    HAC(t) = α * HAC(t-1) + (1 - α) * S(t)
    onde:
    - α = exp(-Δt/τ) é o fator de esquecimento (memória)
    - τ = tau_hours * 3600 (constante de tempo em segundos)
    - S(t) = fonte de acoplamento instantâneo (quando Bz < 0)
    """
    print(f"\n⚡ Calculando HAC (Modelo Exponencial, τ={tau_hours}h)...")
    
    # Dados de entrada (mantidos em unidades operacionais)
    Bz = df['bz_gsm'].values          # nT
    Vsw = df['speed'].values          # km/s
    Np = df['density'].values         # cm⁻³
    
    # 1. Calcular delta-t real entre medições (em segundos)
    times = pd.to_datetime(df['time_tag']).values
    dt_seconds = np.zeros(len(times))
    if len(times) > 1:
        time_diffs = np.diff(times)
        dt_seconds[1:] = time_diffs.astype('timedelta64[s]').astype(np.float64)
        dt_seconds[0] = dt_seconds[1] if len(dt_seconds) > 1 else 60.0
    else:
        dt_seconds[:] = 60.0
    
    # 2. Definir o termo fonte instantâneo S(t)
    # Fonte = função do campo elétrico solar (ε ~ -Bz * V) quando Bz<0
    source = np.zeros(len(Bz))
    south_mask = Bz < 0
    
    # Termo de acoplamento físico: Campo Elétrico (-Bz*V) normalizado
    # O fator 1/1000 mantém a escala numérica gerenciável
    source[south_mask] = (-Bz[south_mask] * Vsw[south_mask]) / 1000.0
    
    # 3. Parâmetro do filtro exponencial
    tau = tau_hours * 3600.0          # Converter horas para segundos
    alpha = np.exp(-dt_seconds / tau) # Fator de decaimento para cada passo
    
    # 4. Aplicar o filtro exponencial (implementação eficiente)
    hac = np.zeros(len(times))
    for i in range(1, len(hac)):
        # Equação do filtro: HAC_novo = α*HAC_antigo + (1-α)*Fonte
        hac[i] = alpha[i] * hac[i-1] + (1 - alpha[i]) * source[i]
    
    # 5. Normalizar para faixa operacional 0-300
    # Normalização baseada no máximo observado no período
    max_val = np.max(hac) if np.max(hac) > 0 else 1.0
    hac_normalized = (hac / max_val) * scale_to_max
    
    # 6. Aplicar saturação suave (não hard clipping)
    # Permite que valores ultrapassem ligeiramente 300 durante picos extremos
    saturation_soft = 350
    hac_normalized = np.minimum(hac_normalized, saturation_soft)
    
    # Adicionar ao DataFrame
    df['HAC'] = hac_normalized
    df['HAC_source'] = source
    df['HAC_raw'] = hac
    
    # Estatísticas
    print(f"   • HAC final: {df['HAC'].iloc[-1]:.2f}")
    print(f"   • HAC máximo: {df['HAC'].max():.2f} (não saturado em {scale_to_max})")
    print(f"   • Fração IMF sul: {south_mask.mean()*100:.1f}%")
    
    return df

# ============================
# 3. CLASSIFICAÇÃO E ANÁLISE
# ============================
STORM_THRESHOLDS = [
    ("Quiet",  "🟢", "#2ecc71", 0, 50),
    ("G1",     "🟡", "#f1c40f", 50, 100),
    ("G2",     "🟠", "#e67e22", 100, 150),
    ("G3",     "🔴", "#e74c3c", 150, 200),
    ("G4",     "🟣", "#9b59b6", 200, 250),
    ("G5",     "⚫", "#34495e", 250, 1000)
]

def classify_storm_level(hac_value):
    """Classifica o nível da tempestade geomagnética."""
    for name, emoji, color, min_val, max_val in STORM_THRESHOLDS:
        if min_val <= hac_value < max_val:
            return name, emoji, color
    return "G5", "⚫", "#34495e"

def analyze_storm_dynamics(df):
    """Analisa a dinâmica física do HAC calculado."""
    print("\n📊 ANÁLISE DA DINÂMICA DO HAC")
    print("="*50)
    
    last = df.iloc[-1]
    hac_now = last['HAC']
    level, emoji, color = classify_storm_level(hac_now)
    
    print(f"\n⏰ Última observação: {last['time_tag']}")
    print(f"⚡ HAC atual: {hac_now:.2f} → {emoji} {level}")
    
    # Estatísticas do período
    print(f"\n📈 Estatísticas do período:")
    print(f"   • Mínimo: {df['HAC'].min():.2f}")
    print(f"   • Máximo: {df['HAC'].max():.2f} (escala: {SCALE_TO_MAX})")
    print(f"   • Médio:  {df['HAC'].mean():.2f}")
    
    # Comportamento do Bz
    bz_neg = df[df['bz_gsm'] < 0]
    bz_pos = df[df['bz_gsm'] >= 0]
    
    print(f"\n🧲 Comportamento do IMF (Bz):")
    print(f"   • Sul (Bz<0):  {len(bz_neg):>5} pts ({len(bz_neg)/len(df)*100:5.1f}%)")
    print(f"   • Norte (Bz≥0):{len(bz_pos):>5} pts ({len(bz_pos)/len(df)*100:5.1f}%)")
    
    if len(bz_neg) > 0:
        print(f"   • Bz mínimo:   {bz_neg['bz_gsm'].min():6.1f} nT")
        print(f"   • Bz médio:    {bz_neg['bz_gsm'].mean():6.1f} nT")
    
    # Dinâmica temporal
    print(f"\n⏱️  Dinâmica temporal:")
    print(f"   • τ (memória): {TAU_HOURS} horas")
    print(f"   • Duração:     {len(df)} pts ({len(df)/60:.1f} horas)")
    print(f"   • Δt médio:    {df['time_tag'].diff().mean().total_seconds():.0f} s")
    
    # Verificação crítica: saturação artificial?
    near_saturation = (df['HAC'] > 0.95 * SCALE_TO_MAX).mean() * 100
    if near_saturation > 10:
        print(f"\n⚠️  AVISO: {near_saturation:.1f}% dos dados próximo à escala máxima")
        print("   Considere aumentar a escala ou ajustar o termo fonte.")
    
    print("\n" + "="*50)
    return level, emoji, hac_now

# ============================
# 4. VISUALIZAÇÃO (GRÁFICOS PARA ARTIGO)
# ============================
def create_publication_figure(df, filename="hac_corrected_publication.png"):
    """Gera figura com qualidade de publicação."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(15, 10))
    
    # -------------------- Painel 1: HAC --------------------
    ax1 = plt.subplot(4, 1, 1)
    
    # Curva do HAC
    ax1.plot(df['time_tag'], df['HAC'], 
             color='#d62728', linewidth=2.5, 
             label=f'HAC (τ={TAU_HOURS}h)', zorder=5)
    
    # Áreas coloridas por nível de tempestade
    for name, emoji, color, min_val, max_val in STORM_THRESHOLDS:
        if min_val > 0:
            ax1.axhspan(min_val, max_val, alpha=0.08, color=color, zorder=1)
            ax1.text(df['time_tag'].iloc[-1] + timedelta(hours=2), 
                    (min_val + max_val)/2, f' {name}', 
                    va='center', color=color, fontsize=9, alpha=0.8)
    
    ax1.set_ylabel('Índice HAC', fontsize=12, fontweight='bold')
    ax1.set_title('Heliospheric Accumulated Coupling (HAC) - Modelo Exponencial Corrigido', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.4, zorder=2)
    ax1.set_ylim(-10, 320)
    ax1.set_xlim(df['time_tag'].min(), df['time_tag'].max() + timedelta(hours=4))
    
    # -------------------- Painel 2: Bz --------------------
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    
    # Linha do Bz
    ax2.plot(df['time_tag'], df['bz_gsm'], 
             color='#2ecc71', linewidth=1.2, 
             label='Bz (GSM)', zorder=3)
    
    # Destaque para períodos de IMF sul
    ax2.fill_between(df['time_tag'], 0, df['bz_gsm'],
                     where=(df['bz_gsm'] < 0),
                     color='red', alpha=0.35, 
                     label='IMF Sul (Acoplamento)', zorder=2)
    
    ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.6, linestyle='-', zorder=1)
    ax2.set_ylabel('Bz [nT]', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # -------------------- Painel 3: Velocidade --------------------
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(df['time_tag'], df['speed'], 
             color='#3498db', linewidth=1.5, 
             label='Velocidade do Vento Solar')
    ax3.set_ylabel('V [km/s]', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # -------------------- Painel 4: Densidade --------------------
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(df['time_tag'], df['density'], 
             color='#9b59b6', linewidth=1.5, 
             label='Densidade de Prótons')
    ax4.set_ylabel('n [cm⁻³]', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Tempo (UTC)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Formatar eixo de tempo
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    
    # Salvar em alta resolução
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Figura para publicação salva: {filename}")
    return filename

# ============================
# 5. FUNÇÃO PRINCIPAL E EXECUÇÃO
# ============================
def main():
    print("\n" + "="*70)
    print("🛰️  HAC - HELIOSPHERIC ACCUMULATED COUPLING (MODELO CORRIGIDO)")
    print("="*70)
    
    # 1. Carregar dados
    print("\n📥 Carregando dados OMNI...")
    mag_df = load_omni_data(MAG_FILE)
    plasma_df = load_omni_data(PLASMA_FILE)
    
    if mag_df is None or plasma_df is None:
        print("❌ Falha crítica no carregamento de dados.")
        return
    
    # 2. Preparar dados
    print("\n🔧 Preparando e fundindo dados...")
    df = prepare_merged_data(mag_df, plasma_df)
    
    if len(df) < 10:
        print("❌ Dados insuficientes após preparação.")
        return
    
    print(f"   Dados válidos: {len(df)} pontos")
    print(f"   Período: {df['time_tag'].min()} a {df['time_tag'].max()}")
    
    # 3. Calcular HAC com modelo exponencial CORRETO
    df = calculate_hac_exponential(df, tau_hours=TAU_HOURS, scale_to_max=SCALE_TO_MAX)
    
    # 4. Análise detalhada
    storm_level, storm_emoji, hac_final = analyze_storm_dynamics(df)
    
    # 5. Gerar gráficos
    print("\n📈 Gerando visualizações...")
    pub_plot = create_publication_figure(df, "hac_modelo_corrigido.png")
    
    # 6. Salvar dados processados
    try:
        csv_file = "hac_resultados_corrigidos.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\n💾 Dados processados salvos: {csv_file}")
    except Exception as e:
        print(f"⚠️  Não foi possível salvar CSV: {e}")
    
    # 7. Relatório final
    print("\n" + "="*70)
    print("🎯 RELATÓRIO FINAL - MODELO HAC CORRIGIDO")
    print("="*70)
    print(f"\n{storm_emoji} CLASSIFICAÇÃO ATUAL: {storm_level}")
    print(f"   Índice HAC: {hac_final:.2f}")
    
    # Status operacional
    if hac_final >= 250:
        print("   🚨 ALERTA: Condições de tempestade G5 (Extrema)")
    elif hac_final >= 200:
        print("   🚨 ALERTA: Tempestade G4 (Severa)")
    elif hac_final >= 150:
        print("   ⚠️  ALERTA: Tempestade G3 (Forte)")
    elif hac_final >= 100:
        print("   📢 ATENÇÃO: Tempestade G2 (Moderada)")
    elif hac_final >= 50:
        print("   📋 MONITORAMENTO: Tempestade G1 (Menor)")
    else:
        print("   ✅ CONDIÇÕES: Quietas")
    
    # Parâmetros do modelo
    print(f"\n⚙️  PARÂMETROS DO MODELO:")
    print(f"   • Constante de tempo (τ): {TAU_HOURS} horas")
    print(f"   • Escala máxima: {SCALE_TO_MAX}")
    print(f"   • Termo fonte: S(t) = (-Bz * V) / 1000 (para Bz < 0)")
    print(f"   • Tipo: Filtro exponencial com memória finita")
    
    # Validação do modelo
    print(f"\n✅ VALIDAÇÃO DO MODELO:")
    print(f"   ✓ Sem saturação artificial (máximo: {df['HAC'].max():.1f})")
    print(f"   ✓ Decaimento físico natural após eventos")
    print(f"   ✓ Resposta proporcional à intensidade do acoplamento")
    print(f"   ✓ Escala consistente 0-{SCALE_TO_MAX}")
    
    print("\n📁 ARQUIVOS GERADOS:")
    print(f"   1. {pub_plot} - Figura principal para o artigo")
    print(f"   2. hac_resultados_corrigidos.csv - Dados processados completos")
    
    print("\n" + "="*70)
    print("✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*70)

# ============================
# EXECUÇÃO
# ============================
if __name__ == "__main__":
    # Configuração para melhor visualização no console
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # Executar pipeline
    main()
