"""
Exemplo prático de uso do sistema HAC Coupling.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Importa os módulos
from hac.coupling.geometry import EventGeometry
from hac.coupling.coupling_index import compute_coupling_index
from hac.coupling.forecast import CouplingForecaster

# ==============================================
# 1. CRIAR GEOMETRIA DO EVENTO SOLAR
# ==============================================

geom = EventGeometry(
    longitude_deg=-15.0,      # 15 graus a leste do meridiano central
    latitude_deg=10.0,        # 10 graus norte
    angular_width_deg=220.0,  # CME larga
    halo_flag=True,           # É um halo CME
    source_region="AR 13579"  # Opcional: região ativa
)

print("✅ Geometria criada:")
print(f"   Longitude: {geom.longitude_deg}°")
print(f"   Latitude: {geom.latitude_deg}°")
print(f"   Largura: {geom.angular_width_deg}°")
print(f"   Halo: {geom.halo_flag}")

# ==============================================
# 2. SIMULAR DADOS DE VENTO SOLAR (L1)
# ==============================================

def criar_dados_simulados(n_minutos=180, bz_medio=-12, bt_medio=20):
    """Cria dados simulados de Bz e Bt."""
    np.random.seed(42)
    
    # Cria timestamps (últimos n_minutos)
    agora = datetime.now()
    timestamps = [agora - timedelta(minutes=i) for i in range(n_minutos)]
    timestamps.reverse()  # Ordem cronológica
    
    # Tempo para simulação de ondas
    t = np.linspace(0, 6*np.pi, n_minutos)
    
    # Simula Bz com componente sul forte
    bz = bz_medio + 8 * np.sin(t) + np.random.normal(0, 2, n_minutos)
    
    # Simula Bt correlacionado
    bt = bt_medio + 5 * np.sin(t + 0.1) + np.random.normal(0, 1, n_minutos)
    
    # Cria séries pandas
    bz_series = pd.Series(bz, index=timestamps, name='Bz')
    bt_series = pd.Series(bt, index=timestamps, name='Bt')
    
    return bz_series, bt_series

# Cria dados simulados
bz_series, bt_series = criar_dados_simulados(
    n_minutos=120,      # 2 horas de dados
    bz_medio=-15,       # Bz médio -15 nT (sul forte)
    bt_medio=18         # Bt médio 18 nT
)

print(f"\n✅ Dados simulados criados:")
print(f"   Período: {bz_series.index[0]} a {bz_series.index[-1]}")
print(f"   Bz médio: {bz_series.mean():.1f} nT")
print(f"   Bt médio: {bt_series.mean():.1f} nT")

# ==============================================
# 3. CALCULAR ÍNDICE DE ACOPLAMENTO
# ==============================================

resultado = compute_coupling_index(geom, bz_series, bt_series)

print("\n📊 RESULTADO DO ÍNDICE DE ACOPLAMENTO:")
print("=" * 50)
print(f"Índice: {resultado.coupling_index:.3f} (0-1)")
print(f"Confiança: {resultado.confidence:.1%}")
print(f"Score geométrico: {resultado.geom_score:.3f}")
print(f"Fração Bz sul: {resultado.south_fraction:.1%}")
print(f"Persistência: {resultado.south_persistence_score:.3f}")
print(f"Coerência: {resultado.coherence_score:.3f}")
print(f"Dispersão angular: {resultado.angular_dispersion:.1f}°")
print(f"Ângulo flanco: {resultado.flank_angle:.1f}°")
print(f"Densidade magnética: {resultado.integrated_magnetic_density:.0f} nT·min")

# ==============================================
# 4. GERAR PREVISÃO OPERACIONAL
# ==============================================

previsor = CouplingForecaster()
previsao = previsor.predict_from_coupling_index(
    resultado,
    solar_wind_speed=550.0  # km/s (opcional)
)

print("\n🔮 PREVISÃO OPERACIONAL:")
print("=" * 50)
print(f"Probabilidade Bz sul: {previsao.probability_bz_south:.1%}")
print(f"Bz mínimo esperado: {previsao.expected_min_bz:.0f} nT")
print(f"Duração esperada: {previsao.expected_duration:.0f} min")
print(f"Nível severidade: {previsao.severity_level.upper()}")
print(f"Confiança previsão: {previsao.confidence:.1%}")
print(f"Timestamp: {previsao.timestamp}")

print("\n📋 AÇÕES RECOMENDADAS:")
for i, acao in enumerate(previsao.recommended_actions, 1):
    print(f"  {i}. {acao}")

# ==============================================
# 5. GERAR MENSAGEM DE ALERTA
# ==============================================

mensagem_alerta = previsor.generate_alert_message(previsao)
print("\n🚨 MENSAGEM DE ALERTA:")
print("=" * 50)
print(mensagem_alerta)

# ==============================================
# 6. SALVAR RESULTADOS
# ==============================================

import json
from dataclasses import asdict

# Salva resultados em JSON
resultados = {
    'geometria': asdict(geom),
    'acoplamento': asdict(resultado),
    'previsao': asdict(previsao),
    'timestamp': datetime.now().isoformat()
}

with open('resultado_acoplamento.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, default=str)

print("\n💾 Resultados salvos em 'resultado_acoplamento.json'")

# ==============================================
# 7. VISUALIZAR DADOS (OPCIONAL)
# ==============================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Gráfico de Bz
axes[0].plot(bz_series.index, bz_series.values, 'b-', linewidth=1.5, label='Bz')
axes[0].axhline(y=-10, color='r', linestyle='--', alpha=0.5, label='Limiar -10 nT')
axes[0].fill_between(bz_series.index, bz_series.values, -10, 
                     where=(bz_series < -10), color='r', alpha=0.3)
axes[0].set_ylabel('Bz (nT)')
axes[0].set_title(f'Índice de Acoplamento: {resultado.coupling_index:.3f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico de Bt
axes[1].plot(bt_series.index, bt_series.values, 'g-', linewidth=1.5, label='Bt')
axes[1].set_ylabel('Bt (nT)')
axes[1].set_xlabel('Tempo')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizacao_dados.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📈 Gráfico salvo como 'visualizacao_dados.png'")
