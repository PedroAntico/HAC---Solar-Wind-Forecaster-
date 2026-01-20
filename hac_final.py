"""
HAC+ Model: Heliospheric Accumulated Coupling with Advanced Physics
Proof-of-concept implementation including:
- Nonlinear coupling saturation
- Ring current energy partitioning
- Substorm injection parameterization
- Ionospheric response nonlinearities
- Kp index saturation modeling

Author: Pedro Guilherme Antico
Repository: https://github.com/PedroAntico/HAC-Solar-Wind-Forecaster
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURAÇÃO DOS PARÂMETROS FÍSICOS
# ============================
class HACPhysicsConfig:
    """Configuração dos parâmetros físicos do modelo HAC+"""
    
    # Tempos característicos (em horas)
    TAU_RING_CURRENT = 3.0      # Tempo de decaimento da corrente de anel
    TAU_SUBSTORM = 1.5          # Tempo de injeção por subtempestades
    TAU_IONOSPHERE = 0.5        # Tempo de resposta ionosférica
    
    # Parâmetros de saturação
    E_FIELD_SATURATION = 15.0   # mV/m - Saturação do campo elétrico solar
    KP_SATURATION = 8.0         # Valor de saturação do índice Kp
    RING_CURRENT_MAX = 500.0    # nT - Saturação da corrente de anel (equivalente Dst)
    
    # Coeficientes de particionamento de energia
    ALPHA_RING = 0.4           # Fração para corrente de anel
    ALPHA_SUBSTORM = 0.3       # Fração para subtempestades
    ALPHA_IONOSPHERE = 0.3     # Fração para ionosfera
    
    # Parâmetros não lineares
    BETA_NONLINEAR = 1.5       # Expoente de resposta não linear
    COUPLING_THRESHOLD = 5.0   # mV/m - Limiar para acoplamento não linear
    
    # Escalas de normalização
    HAC_SCALE_MAX = 300.0
    KP_SCALE = 9.0

# ============================
# 1. CARREGAMENTO E PREPARAÇÃO DE DADOS
# ============================
class OMNIDataProcessor:
    """Processador de dados OMNI com validação física"""
    
    @staticmethod
    def load_and_validate(filepath):
        """Carrega e valida dados OMNI"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            headers = data[0]
            df = pd.DataFrame(data[1:], columns=headers)
            df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
            
            # Converter para numérico
            for col in headers:
                if col != 'time_tag':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar {filepath}: {e}")
            return None
    
    @staticmethod
    def compute_physical_quantities(df):
        """Calcula quantidades físicas derivadas"""
        # Campo elétrico solar (mV/m)
        df['E_field'] = df['bz_gsm'].clip(upper=0).abs() * df['speed'] * 1e-3  # -Bz*V em mV/m
        
        # Pressão dinâmica (nPa)
        df['P_dyn'] = 1.6726e-6 * df['density'] * df['speed']**2 * 1e3
        
        # Parâmetro de acoplamento de Newell
        df['Newell'] = df['speed']**(4/3) * np.abs(df['bz_gsm'])**(2/3) * np.sin(np.abs(np.arctan2(df['by_gsm'], df['bz_gsm']))/2)**(8/3)
        
        return df

# ============================
# 2. NÚCLEO DO MODELO FÍSICO HAC+
# ============================
class HACPlusModel:
    """Implementação do modelo HAC+ com física avançada"""
    
    def __init__(self, config=HACPhysicsConfig()):
        self.config = config
        self.results = {}
        
    def compute_solar_wind_coupling(self, df):
        """
        Calcula o acoplamento vento solar-magnetosfera com saturação
        
        Implementa:
        1. Campo elétrico solar com saturação
        2. Resposta não linear
        3. Múltiplas escalas de tempo
        """
        print("\n⚡ Calculando acoplamento solar-magnetosfera...")
        
        # Extrair dados
        times = pd.to_datetime(df['time_tag']).values
        E_field = df['E_field'].values  # mV/m
        Bz = df['bz_gsm'].values        # nT
        Vsw = df['speed'].values        # km/s
        Np = df['density'].values       # cm⁻³
        
        # Calcular delta-t real
        dt = np.zeros(len(times))
        if len(times) > 1:
            time_diffs = np.diff(times)
            dt[1:] = time_diffs.astype('timedelta64[s]').astype(np.float64)
            dt[0] = dt[1] if len(dt) > 1 else 60.0
        
        # 1. FUNÇÃO DE ACOPLAMENTO COM SATURAÇÃO
        # Saturação tipo tanh para campo elétrico alto
        E_field_sat = self.config.E_FIELD_SATURATION
        coupling_raw = np.tanh(E_field / E_field_sat) * E_field_sat
        
        # 2. RESPOSTA NÃO LINEAR (exponencial para altos valores)
        coupling_nonlinear = np.zeros_like(coupling_raw)
        mask_high = E_field > self.config.COUPLING_THRESHOLD
        coupling_nonlinear[mask_high] = coupling_raw[mask_high]**self.config.BETA_NONLINEAR
        coupling_nonlinear[~mask_high] = coupling_raw[~mask_high]
        
        # 3. FILTRO TEMPORAL PARA DIFERENTES PROCESSOS
        # Converter tempos característicos para segundos
        tau_rc = self.config.TAU_RING_CURRENT * 3600
        tau_sub = self.config.TAU_SUBSTORM * 3600
        tau_ion = self.config.TAU_IONOSPHERE * 3600
        
        # Inicializar estados
        hac_ring = np.zeros(len(times))
        hac_substorm = np.zeros(len(times))
        hac_ionosphere = np.zeros(len(times))
        
        # Simulação temporal com equações diferenciais acopladas
        for i in range(1, len(times)):
            # Fatores de decaimento exponencial
            alpha_rc = np.exp(-dt[i] / tau_rc)
            alpha_sub = np.exp(-dt[i] / tau_sub)
            alpha_ion = np.exp(-dt[i] / tau_ion)
            
            # Injeção condicional (apenas para Bz < 0)
            injection = coupling_nonlinear[i] if Bz[i] < 0 else 0
            
            # Sistema de equações acopladas
            hac_ring[i] = (alpha_rc * hac_ring[i-1] + 
                          self.config.ALPHA_RING * injection * dt[i])
            
            # Subtempestades: resposta mais rápida
            hac_substorm[i] = (alpha_sub * hac_substorm[i-1] + 
                              self.config.ALPHA_SUBSTORM * injection * dt[i])
            
            # Ionosfera: resposta mais rápida ainda
            hac_ionosphere[i] = (alpha_ion * hac_ionosphere[i-1] + 
                                self.config.ALPHA_IONOSPHERE * injection * dt[i])
        
        # 4. HAC TOTAL (combinação ponderada)
        hac_total = (hac_ring + hac_substorm + hac_ionosphere)
        
        # Normalizar
        if np.max(hac_total) > 0:
            hac_total = (hac_total / np.max(hac_total)) * self.config.HAC_SCALE_MAX
        
        # Armazenar resultados
        self.results['time'] = times
        self.results['HAC_total'] = hac_total
        self.results['HAC_ring'] = hac_ring
        self.results['HAC_substorm'] = hac_substorm
        self.results['HAC_ionosphere'] = hac_ionosphere
        self.results['E_field'] = E_field
        self.results['coupling'] = coupling_nonlinear
        
        print(f"   • HAC máximo: {hac_total.max():.1f}")
        print(f"   • E-field máximo: {E_field.max():.1f} mV/m")
        
        return hac_total
    
    def predict_geomagnetic_indices(self, hac_values):
        """
        Prediz índices geomagnéticos a partir do HAC
        
        Inclui:
        1. Saturação do Kp
        2. Estimativa de Dst equivalente
        3. Classificação de tempestade
        """
        print("\n🌍 Predizendo índices geomagnéticos...")
        
        # 1. MAPEAMENTO PARA Kp COM SATURAÇÃO
        # Relação não linear com saturação em ~9
        kp_pred = self.config.KP_SCALE * np.tanh(hac_values / self.config.HAC_SCALE_MAX * 2)
        
        # 2. ESTIMATIVA DE Dst EQUIVALENTE
        # Relação empírica entre HAC e corrente de anel
        dst_pred = -self.config.RING_CURRENT_MAX * (hac_values / self.config.HAC_SCALE_MAX)**1.3
        
        # 3. CLASSIFICAÇÃO NOAA G-SCALE
        storm_levels = []
        for h in hac_values:
            if h < 50:
                level = "Quiet"
            elif h < 100:
                level = "G1"
            elif h < 150:
                level = "G2"
            elif h < 200:
                level = "G3"
            elif h < 250:
                level = "G4"
            else:
                level = "G5"
            storm_levels.append(level)
        
        self.results['Kp_pred'] = kp_pred
        self.results['Dst_pred'] = dst_pred
        self.results['Storm_level'] = storm_levels
        
        # Estatísticas
        g4g5_count = sum(1 for l in storm_levels if l in ['G4', 'G5'])
        print(f"   • Kp máximo previsto: {kp_pred.max():.1f}")
        print(f"   • Dst mínimo previsto: {dst_pred.min():.1f} nT")
        print(f"   • Eventos G4/G5: {g4g5_count} pontos")
        
        return kp_pred, dst_pred, storm_levels

# ============================
# 3. VISUALIZAÇÃO AVANÇADA
# ============================
class HACVisualizer:
    """Geração de figuras para publicação científica"""
    
    @staticmethod
    def create_physics_dashboard(results, df, filename="hac_physics_dashboard.png"):
        """Cria dashboard completo da física do modelo"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(16, 14))
        
        # ========== Painel 1: Sistema de Acooplamento ==========
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(results['time'], results['E_field'], 
                color='#e74c3c', linewidth=1.5, label='E = -Bz×V')
        ax1.axhline(y=HACPhysicsConfig.E_FIELD_SATURATION, 
                   color='red', linestyle='--', alpha=0.5, 
                   label=f'Saturação ({HACPhysicsConfig.E_FIELD_SATURATION} mV/m)')
        ax1.set_ylabel('Campo Elétrico [mV/m]', fontsize=10)
        ax1.set_title('A. Vento Solar: Forçante Externa', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # ========== Painel 2: Resposta Não Linear ==========
        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(results['time'], results['coupling'], 
                color='#9b59b6', linewidth=2, label='Acoplamento efetivo')
        ax2.fill_between(results['time'], 0, results['coupling'], 
                        alpha=0.3, color='#9b59b6')
        ax2.set_ylabel('Acoplamento [mV/m]', fontsize=10)
        ax2.set_title('B. Resposta Não Linear do Sistema', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ========== Painel 3: Reservatório de Energia ==========
        ax3 = plt.subplot(4, 2, (3, 4))
        ax3.plot(results['time'], results['HAC_total'], 
                color='#d62728', linewidth=2.5, label='HAC Total')
        ax3.plot(results['time'], results['HAC_ring'], 
                color='#2ecc71', linewidth=1.5, linestyle='--', label='Corrente de Anel')
        ax3.plot(results['time'], results['HAC_substorm'], 
                color='#3498db', linewidth=1.5, linestyle='--', label='Subtempestades')
        ax3.plot(results['time'], results['HAC_ionosphere'], 
                color='#f39c12', linewidth=1.5, linestyle='--', label='Ionosfera')
        
        # Thresholds de tempestade
        colors = ['green', 'yellow', 'orange', 'red', 'purple', 'black']
        levels = [50, 100, 150, 200, 250, 300]
        labels = ['G1', 'G2', 'G3', 'G4', 'G5']
        for i, (lvl, col, lab) in enumerate(zip(levels[:5], colors, labels)):
            ax3.axhline(y=lvl, color=col, linestyle=':', alpha=0.5)
            ax3.text(results['time'][0], lvl+5, lab, color=col, fontsize=8)
        
        ax3.set_ylabel('Estado do Reservatório [HAC]', fontsize=10)
        ax3.set_title('C. Reservatório de Energia Magnetosférica', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 320)
        
        # ========== Painel 4: Índices Previstos ==========
        ax4 = plt.subplot(4, 2, (5, 6))
        
        # Kp
        ax4_kp = ax4
        ax4_kp.plot(results['time'], results['Kp_pred'], 
                   color='#e74c3c', linewidth=2, label='Kp previsto')
        ax4_kp.axhline(y=HACPhysicsConfig.KP_SATURATION, 
                      color='red', linestyle='--', alpha=0.5, 
                      label='Saturação Kp')
        ax4_kp.set_ylabel('Índice Kp', fontsize=10, color='#e74c3c')
        ax4_kp.tick_params(axis='y', labelcolor='#e74c3c')
        ax4_kp.set_ylim(0, 9)
        
        # Dst (eixo secundário)
        ax4_dst = ax4_kp.twinx()
        ax4_dst.plot(results['time'], results['Dst_pred'], 
                    color='#3498db', linewidth=2, linestyle='--', label='Dst previsto')
        ax4_dst.set_ylabel('Índice Dst [nT]', fontsize=10, color='#3498db')
        ax4_dst.tick_params(axis='y', labelcolor='#3498db')
        ax4_dst.set_ylim(-600, 50)
        
        ax4_kp.set_title('D. Índices Geomagnéticos Previstos', fontsize=11, fontweight='bold')
        ax4_kp.grid(True, alpha=0.3)
        
        # Combinar legendas
        lines_kp, labels_kp = ax4_kp.get_legend_handles_labels()
        lines_dst, labels_dst = ax4_dst.get_legend_handles_labels()
        ax4_kp.legend(lines_kp + lines_dst, labels_kp + labels_dst, loc='upper left', fontsize=8)
        
        # ========== Painel 5: Parâmetros de Entrada ==========
        ax5 = plt.subplot(4, 2, 7)
        
        # Bz
        ax5_bz = ax5
        ax5_bz.plot(df['time_tag'], df['bz_gsm'], 
                   color='#2ecc71', linewidth=1.5, label='Bz')
        ax5_bz.fill_between(df['time_tag'], 0, df['bz_gsm'], 
                           where=(df['bz_gsm'] < 0), 
                           color='red', alpha=0.3, label='IMF Sul')
        ax5_bz.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5_bz.set_ylabel('Bz [nT]', fontsize=10, color='#2ecc71')
        ax5_bz.tick_params(axis='y', labelcolor='#2ecc71')
        ax5_bz.legend(loc='upper right', fontsize=7)
        
        ax5_bz.set_title('E. Parâmetros do Vento Solar', fontsize=11, fontweight='bold')
        ax5_bz.grid(True, alpha=0.3)
        
        # ========== Painel 6: Velocidade ==========
        ax6 = plt.subplot(4, 2, 8)
        ax6.plot(df['time_tag'], df['speed'], 
                color='#3498db', linewidth=1.5, label='Velocidade')
        ax6.set_ylabel('V [km/s]', fontsize=10, color='#3498db')
        ax6.tick_params(axis='y', labelcolor='#3498db')
        ax6.legend(loc='upper right', fontsize=7)
        ax6.grid(True, alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dashboard físico salvo: {filename}")
        return fig

# ============================
# 4. ANÁLISE E RELATÓRIO
# ============================
class PhysicsAnalysis:
    """Análise física detalhada dos resultados"""
    
    @staticmethod
    def generate_physics_report(results, df):
        """Gera relatório detalhado da física do evento"""
        print("\n" + "="*70)
        print("📊 RELATÓRIO DE FÍSICA DO EVENTO")
        print("="*70)
        
        # Identificar período de tempestade
        storm_mask = np.array([l in ['G4', 'G5'] for l in results['Storm_level']])
        
        if np.any(storm_mask):
            storm_times = results['time'][storm_mask]
            storm_start = storm_times[0]
            storm_end = storm_times[-1]
            
            print(f"\n⏱️  PERÍODO DE TEMPESTADE G4/G5:")
            print(f"   • Início: {storm_start}")
            print(f"   • Fim:    {storm_end}")
            print(f"   • Duração: {len(storm_times)} pontos (~{len(storm_times)/60:.1f} horas)")
        
        # Análise do acoplamento
        max_e_field = results['E_field'].max()
        mean_coupling = results['coupling'][storm_mask].mean() if np.any(storm_mask) else 0
        
        print(f"\n⚡ ANÁLISE DO ACOPLAMENTO:")
        print(f"   • E-field máximo: {max_e_field:.1f} mV/m")
        print(f"   • E-field médio (tempestade): {mean_coupling:.1f} mV/m")
        print(f"   • Saturação configurada: {HACPhysicsConfig.E_FIELD_SATURATION} mV/m")
        print(f"   • Excedeu saturação? {'SIM' if max_e_field > HACPhysicsConfig.E_FIELD_SATURATION else 'NÃO'}")
        
        # Particionamento de energia
        hac_total_max = results['HAC_total'].max()
        ring_frac = results['HAC_ring'].max() / hac_total_max if hac_total_max > 0 else 0
        substorm_frac = results['HAC_substorm'].max() / hac_total_max if hac_total_max > 0 else 0
        ion_frac = results['HAC_ionosphere'].max() / hac_total_max if hac_total_max > 0 else 0
        
        print(f"\n⚖️  PARTICIONAMENTO DE ENERGIA:")
        print(f"   • Corrente de Anel: {ring_frac*100:.1f}% (τ={HACPhysicsConfig.TAU_RING_CURRENT}h)")
        print(f"   • Subtempestades: {substorm_frac*100:.1f}% (τ={HACPhysicsConfig.TAU_SUBSTORM}h)")
        print(f"   • Ionosfera: {ion_frac*100:.1f}% (τ={HACPhysicsConfig.TAU_IONOSPHERE}h)")
        
        # Predições
        max_kp = results['Kp_pred'].max()
        min_dst = results['Dst_pred'].min()
        
        print(f"\n🌍 PREDIÇÕES GEOMAGNÉTICAS:")
        print(f"   • Kp máximo previsto: {max_kp:.1f}")
        print(f"   • Dst mínimo previsto: {min_dst:.1f} nT")
        print(f"   • Saturação Kp: {HACPhysicsConfig.KP_SATURATION}")
        print(f"   • Alcançou saturação Kp? {'SIM' if max_kp >= HACPhysicsConfig.KP_SATURATION*0.9 else 'NÃO'}")
        
        # Eficiência do sistema
        if max_e_field > 0:
            system_efficiency = hac_total_max / (max_e_field * 10)  # Métrica adimensional
            print(f"\n🔧 EFICIÊNCIA DO SISTEMA:")
            print(f"   • Eficiência total: {system_efficiency:.3f}")
            print(f"   • β não-linear: {HACPhysicsConfig.BETA_NONLINEAR}")
        
        print("\n" + "="*70)
        
        # Salvar relatório
        with open("physics_analysis_report.txt", "w") as f:
            f.write("RELATÓRIO DE ANÁLISE FÍSICA - MODELO HAC+\n")
            f.write("="*50 + "\n\n")
            f.write(f"Evento analisado: {df['time_tag'].min()} a {df['time_tag'].max()}\n")
            f.write(f"HAC máximo: {results['HAC_total'].max():.1f}\n")
            f.write(f"E-field máximo: {max_e_field:.1f} mV/m\n")
            f.write(f"Kp máximo previsto: {max_kp:.1f}\n")
            f.write(f"Dst mínimo previsto: {min_dst:.1f} nT\n\n")
            f.write("Parâmetros do modelo:\n")
            f.write(f"  τ_ring = {HACPhysicsConfig.TAU_RING_CURRENT} h\n")
            f.write(f"  τ_substorm = {HACPhysicsConfig.TAU_SUBSTORM} h\n")
            f.write(f"  τ_ionosphere = {HACPhysicsConfig.TAU_IONOSPHERE} h\n")
            f.write(f"  E_sat = {HACPhysicsConfig.E_FIELD_SATURATION} mV/m\n")
            f.write(f"  Kp_sat = {HACPhysicsConfig.KP_SATURATION}\n")
        
        print("📝 Relatório de física salvo: physics_analysis_report.txt")

# ============================
# 5. FUNÇÃO PRINCIPAL
# ============================
def main():
    print("\n" + "="*70)
    print("🧪 HAC+ MODEL - PROOF OF CONCEPT WITH ADVANCED PHYSICS")
    print("="*70)
    print("Inclui: subtempestades, corrente de anel, resposta ionosférica, saturação")
    print("="*70)
    
    # Configurar caminhos
    MAG_FILE = "data/mag-7-day.json"
    PLASMA_FILE = "data/plasma-7-day.json"
    
    # 1. Carregar e processar dados
    print("\n📥 Carregando dados OMNI...")
    processor = OMNIDataProcessor()
    mag_df = processor.load_and_validate(MAG_FILE)
    plasma_df = processor.load_and_validate(PLASMA_FILE)
    
    if mag_df is None or plasma_df is None:
        print("❌ Falha no carregamento de dados")
        return
    
     # Preparar dados
    df = pd.merge(mag_df, plasma_df, on="time_tag", how="outer")
    df = df.sort_values("time_tag").reset_index(drop=True)
    df = processor.compute_physical_quantities(df)
    
    print(f"   Período: {df['time_tag'].min()} a {df['time_tag'].max()}")
    print(f"   Pontos: {len(df)}")
    
    # 2. Executar modelo físico
    print("\n🧮 Executando modelo HAC+ com física avançada...")
    model = HACPlusModel()
    hac_values = model.compute_solar_wind_coupling(df)
    
    # 3. Predizer índices
    kp_pred, dst_pred, storm_levels = model.predict_geomagnetic_indices(hac_values)
    
    # 4. Análise física
    analysis = PhysicsAnalysis()
    analysis.generate_physics_report(model.results, df)
    
    # 5. Visualização
    print("\n📈 Gerando visualizações avançadas...")
    viz = HACVisualizer()
    viz.create_physics_dashboard(model.results, df, "hac_physics_proof_of_concept.png")
    
    # 6. Salvar resultados
    try:
        # Adicionar resultados ao DataFrame
        df['HAC_total'] = model.results['HAC_total']
        df['HAC_ring'] = model.results['HAC_ring']
        df['HAC_substorm'] = model.results['HAC_substorm']
        df['HAC_ionosphere'] = model.results['HAC_ionosphere']
        df['Kp_pred'] = model.results['Kp_pred']
        df['Dst_pred'] = model.results['Dst_pred']
        df['Storm_level'] = model.results['Storm_level']
        df['E_field'] = model.results['E_field']
        df['Coupling'] = model.results['coupling']
        
        # Salvar
        output_file = "hac_plus_physics_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Resultados físicos salvos: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Erro ao salvar resultados: {e}")
    
    # 7. Resumo final
    print("\n" + "="*70)
    print("✅ PROOF OF CONCEPT CONCLUÍDO!")
    print("="*70)
    
    last_hac = model.results['HAC_total'][-1]
    last_level = model.results['Storm_level'][-1]
    
    print(f"\n🎯 STATUS ATUAL:")
    print(f"   HAC: {last_hac:.1f} → {last_level}")
    print(f"   Kp previsto: {model.results['Kp_pred'][-1]:.1f}")
    print(f"   Dst previsto: {model.results['Dst_pred'][-1]:.1f} nT")
    
    print(f"\n📊 FÍSICA IMPLEMENTADA:")
    print(f"   • Sistema de equações com 3 reservatórios")
    print(f"   • Saturação do campo elétrico solar")
    print(f"   • Resposta não linear (β={HACPhysicsConfig.BETA_NONLINEAR})")
    print(f"   • Múltiplas escalas de tempo (τ_ring={HACPhysicsConfig.TAU_RING_CURRENT}h)")
    print(f"   • Saturação do índice Kp")
    
    print(f"\n📁 SAÍDAS GERADAS:")
    print(f"   1. hac_physics_proof_of_concept.png - Dashboard completo")
    print(f"   2. hac_plus_physics_results.csv - Dados processados")
    print(f"   3. physics_analysis_report.txt - Análise detalhada")
    
    print("\n" + "="*70)

# ============================
# EXECUÇÃO
# ============================
if __name__ == "__main__":
    main()
