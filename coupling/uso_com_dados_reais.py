"""
Exemplo de uso com dados reais do DSCOVR/ACE.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hac.coupling.geometry import EventGeometry
from hac.coupling.coupling_index import compute_coupling_index

# ==============================================
# 1. CARREGAR DADOS REAIS (FORMATO CSV)
# ==============================================

def carregar_dados_reais(caminho_arquivo):
    """
    Carrega dados de vento solar de arquivo CSV.
    Espera colunas: 'timestamp', 'Bz', 'Bt' (opcional: 'By', 'Bx')
    """
    df = pd.read_csv(caminho_arquivo, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Verifica colunas necessárias
    if 'Bz' not in df.columns or 'Bt' not in df.columns:
        raise ValueError("Arquivo deve conter colunas 'Bz' e 'Bt'")
    
    # Preenche valores faltantes (interpolação linear)
    df['Bz'] = df['Bz'].interpolate(method='linear', limit=10)
    df['Bt'] = df['Bt'].interpolate(method='linear', limit=10)
    
    # Remove outliers (valores além de 3 desvios padrão)
    for col in ['Bz', 'Bt']:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(mean - 3*std, mean + 3*std)
    
    return df['Bz'], df['Bt']

# ==============================================
# 2. EXEMPLO COM DADOS DE TESTE
# ==============================================

# Se não tiver arquivo real, crie um de exemplo
def criar_arquivo_exemplo():
    """Cria arquivo CSV de exemplo com dados simulados."""
    np.random.seed(42)
    
    # Gera 24 horas de dados (1 ponto por minuto)
    n = 1440
    inicio = datetime.now() - timedelta(hours=24)
    timestamps = [inicio + timedelta(minutes=i) for i in range(n)]
    
    # Simula variação diurna
    t = np.linspace(0, 4*np.pi, n)
    
    # Bz com evento de tempestade no meio
    bz = np.random.normal(-2, 3, n)
    bz[600:900] = np.random.normal(-15, 4, 300)  # Tempestade
    
    # Bt correlacionado
    bt = np.random.normal(8, 2, n) + np.abs(bz)/3
    
    # Cria DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Bz': bz,
        'Bt': bt
    })
    
    df.to_csv('dados_vento_solar.csv', index=False)
    print("Arquivo de exemplo criado: 'dados_vento_solar.csv'")
    return df['Bz'], df['Bt']

# Tenta carregar arquivo real ou cria exemplo
try:
    bz_real, bt_real = carregar_dados_reais('dados_vento_solar.csv')
    print("✅ Dados reais carregados")
except FileNotFoundError:
    print("⚠️  Arquivo não encontrado. Criando dados de exemplo...")
    bz_real, bt_real = criar_arquivo_exemplo()

# ==============================================
# 3. DEFINE GEOMETRIA DO EVENTO
# ==============================================

# Exemplo: Evento X1.5 flare de 15/12/2023
geom_evento = EventGeometry(
    longitude_deg=-25.3,      # AR 13514
    latitude_deg=9.2,
    angular_width_deg=187.0,
    halo_flag=True,
    source_region="AR 13514"
)

# ==============================================
# 4. CALCULA ACOPLAMENTO
# ==============================================

print(f"\n📊 Analisando {len(bz_real)} pontos de dados...")

# Usa apenas as últimas 3 horas para análise
janela_analise = 180  # minutos
if len(bz_real) > janela_analise:
    bz_analise = bz_real.iloc[-janela_analise:]
    bt_analise = bt_real.iloc[-janela_analise:]
else:
    bz_analise = bz_real
    bt_analise = bt_real

print(f"   Janela de análise: {janela_analise} min")
print(f"   Período: {bz_analise.index[0]} a {bz_analise.index[-1]}")

# Calcula índice
resultado = compute_coupling_index(geom_evento, bz_analise, bt_analise)

# ==============================================
# 5. INTERPRETA RESULTADOS
# ==============================================

def interpretar_resultado(resultado):
    """Interpreta o índice de acoplamento em linguagem natural."""
    ci = resultado.coupling_index
    
    if ci >= 0.8:
        return "🌋 ACOPLAMENTO EXTREMO - Alto risco de tempestade severa"
    elif ci >= 0.6:
        return "⚠️  ACOPLAMENTO FORTE - Esperar tempestade moderada a forte"
    elif ci >= 0.4:
        return "🔶 ACOPLAMENTO MODERADO - Possível tempestade fraca a moderada"
    elif ci >= 0.2:
        return "🔸 ACOPLAMENTO FRACO - Efeitos menores esperados"
    else:
        return "✅ ACOPLAMENTO BAIXO - Impactos mínimos esperados"

print("\n" + "="*60)
print("RESULTADO DA ANÁLISE DE ACOPLAMENTO")
print("="*60)
print(f"Índice: {resultado.coupling_index:.3f}")
print(f"Interpretação: {interpretar_resultado(resultado)}")
print(f"Confiança: {resultado.confidence:.1%}")

print("\n🔍 DETALHES:")
print(f"  • Fração de tempo com Bz < -10 nT: {resultado.south_fraction:.1%}")
print(f"  • Persistência Bz sul: {resultado.south_persistence_score:.3f}")
print(f"  • Coerência do campo: {resultado.coherence_score:.3f}")
print(f"  • Dispersão angular: {resultado.angular_dispersion:.1f}°")
print(f"  • Ângulo de flanco: {resultado.flank_angle:.1f}°")
print(f"  • Densidade magnética total: {resultado.integrated_magnetic_density:.0f} nT·min")

# ==============================================
# 6. ANALISE POR JANELAS TEMPORAIS
# ==============================================

def analisar_janelas_temporais(geom, bz_series, bt_series, janela_minutos=60):
    """Analisa o acoplamento em janelas temporais móveis."""
    resultados = []
    
    n_pontos = len(bz_series)
    passo = janela_minutos  # 1 hora
    
    for i in range(0, n_pontos - passo, passo//2):  # Passo com 50% sobreposição
        bz_janela = bz_series.iloc[i:i+passo]
        bt_janela = bt_series.iloc[i:i+passo]
        
        if len(bz_janela) < passo * 0.8:  # Mínimo 80% de dados
            continue
        
        try:
            resultado = compute_coupling_index(geom, bz_janela, bt_janela)
            resultados.append({
                'inicio': bz_janela.index[0],
                'fim': bz_janela.index[-1],
                'indice': resultado.coupling_index,
                'bz_medio': bz_janela.mean(),
                'bt_medio': bt_janela.mean()
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(resultados)

# Executa análise por janelas
if len(bz_real) > 240:  # Pelo menos 4 horas de dados
    df_janelas = analisar_janelas_temporais(geom_evento, bz_real, bt_real)
    
    if not df_janelas.empty:
        print(f"\n📈 ANÁLISE TEMPORAL ({len(df_janelas)} janelas):")
        print(df_janelas[['inicio', 'indice', 'bz_medio']].to_string())
        
        # Encontra pico de acoplamento
        pico = df_janelas.loc[df_janelas['indice'].idxmax()]
        print(f"\n⏱️  PICO DE ACOPLAMENTO:")
        print(f"   Período: {pico['inicio']} a {pico['fim']}")
        print(f"   Índice: {pico['indice']:.3f}")
        print(f"   Bz médio: {pico['bz_medio']:.1f} nT")

print("\n✅ Análise concluída!")
