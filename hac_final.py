"""
HAC++ Model: Heliospheric Accumulated Coupling - PRODUÇÃO FINAL
Script robusto para dados OMNI reais com tratamento completo de erros
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================
# 0. NORMALIZAÇÃO DE NOMES OMNI (CRÍTICO)
# ============================
def normalize_omni_columns(df):
    """
    NORMALIZAÇÃO ROBUSTA DE NOMES OMNI
    Garante nomes consistentes independente do formato
    """
    # Mapeamento completo de nomes OMNI
    column_map = {
        # Velocidade
        'flow_speed': 'speed',
        'V': 'speed', 
        'vx': 'speed',
        'proton_speed': 'speed',
        'velocity': 'speed',
        
        # Componente Bz
        'bz': 'bz_gsm',
        'bz_gsm': 'bz_gsm',
        'bz_gse': 'bz_gsm',
        'Bz_GSM': 'bz_gsm',
        'Bz_GSE': 'bz_gsm',
        'IMF_Bz': 'bz_gsm',
        
        # Densidade
        'density': 'density',
        'np': 'density',
        'proton_density': 'density',
        'Np': 'density',
        
        # Componentes Bx, By
        'bx': 'bx_gsm',
        'bx_gsm': 'bx_gsm',
        'by': 'by_gsm',
        'by_gsm': 'by_gsm',
        
        # Magnitude B
        'bt': 'bt',
        'B': 'bt',
        'B_total': 'bt',
        'IMF_B': 'bt',
        
        # Timestamp
        'time_tag': 'time_tag',
        'Time': 'time_tag',
        'timestamp': 'time_tag'
    }
    
    # Criar novo DataFrame com nomes normalizados
    normalized_df = pd.DataFrame()
    
    # Copiar todas as colunas existentes primeiro
    for col in df.columns:
        if col in column_map:
            normalized_name = column_map[col]
            normalized_df[normalized_name] = df[col]
        else:
            normalized_df[col] = df[col]
    
    # Verificar colunas obrigatórias
    required = ['speed', 'bz_gsm', 'density', 'time_tag']
    missing = [c for c in required if c not in normalized_df.columns]
    
    if missing:
        print(f"⚠️  Colunas ausentes: {missing}")
        print(f"   Colunas disponíveis: {list(normalized_df.columns)}")
        
        # Tentar criar colunas faltantes
        if 'speed' in missing:
            # Tentar calcular de outras colunas de velocidade
            for vel_col in ['vx', 'vy', 'vz', 'Vx', 'Vy', 'Vz']:
                if vel_col in normalized_df.columns:
                    normalized_df['speed'] = np.sqrt(
                        normalized_df.get('vx', 0)**2 + 
                        normalized_df.get('vy', 0)**2 + 
                        normalized_df.get('vz', 0)**2
                    )
                    break
        
        # Verificar novamente
        missing = [c for c in required if c not in normalized_df.columns]
        if missing:
            raise ValueError(f"❌ COLUNAS OBRIGATÓRIAS AUSENTES: {missing}")
    
    return normalized_df

# ============================
# CONFIGURAÇÃO FÍSICA CALIBRADA
# ============================
class HACPhysicsConfig:
    """Configuração física validada para dados OMNI reais"""
    
    # TEMPOS CARACTERÍSTICOS (horas)
    TAU_RING_CURRENT = 3.0      # Tempo de decaimento da corrente de anel
    TAU_SUBSTORM = 1.5          # Tempo de injeção por subtempestades  
    TAU_IONOSPHERE = 0.5        # Tempo de resposta ionosférica
    
    # PARÂMETROS DE SATURAÇÃO FÍSICA
    E_FIELD_SATURATION = 15.0   # mV/m - Saturação OBSERVACIONAL
    KP_SATURATION = 8.0         # Saturação do índice Kp
    RING_CURRENT_MAX = 500.0    # nT - Saturação da corrente de anel
    
    # COEFICIENTES DE PARTICIONAMENTO (soma = 1.0)
    ALPHA_RING = 0.4           # Fração para corrente de anel
    ALPHA_SUBSTORM = 0.3       # Fração para subtempestades
    ALPHA_IONOSPHERE = 0.3     # Fração para ionosfera
    
    # PARÂMETROS NÃO LINEARES
    BETA_NONLINEAR = 1.5       # Expoente de resposta não linear
    COUPLING_THRESHOLD = 5.0   # mV/m - Limiar para não-linearidade
    
    # ESCALAS OPERACIONAIS
    HAC_SCALE_MAX = 300.0
    KP_SCALE = 9.0
    
    # LIMITES FÍSICOS
    VSW_MIN, VSW_MAX = 200, 1500      # km/s
    DENSITY_MIN, DENSITY_MAX = 0.1, 100  # cm⁻³
    BZ_MIN, BZ_MAX = -100, 100        # nT

# ============================
# 1. CARREGAMENTO ROBUSTO DE DADOS OMNI
# ============================
class RobustOMNIProcessor:
    """Processador robusto para dados OMNI reais"""
    
    @staticmethod
    def load_and_clean(filepath, max_interpolation=3):
        """
        Carrega, normaliza e limpa dados OMNI
        """
        print(f"📥 Carregando {filepath}...")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Erro JSON: {e}")
            return None
        
        # Criar DataFrame
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        
        # NORMALIZAÇÃO CRÍTICA DE NOMES (antes de qualquer coisa)
        df = normalize_omni_columns(df)
        
        # Converter timestamp
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
        df = df.sort_values('time_tag').reset_index(drop=True)
        
        # Converter para numérico
        numeric_cols = [col for col in df.columns if col != 'time_tag']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # LIMPEZA EM 4 ETAPAS
        df = RobustOMNIProcessor._clean_dataframe(df, max_interpolation)
        
        print(f"   ✅ {len(df)} pontos limpos")
        return df
    
    @staticmethod
    def _clean_dataframe(df, max_interpolation):
        """Pipeline completo de limpeza"""
        # 1. REMOVER INF/NAN EXPLÍCITOS
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. APLICAR LIMITES FÍSICOS
        config = HACPhysicsConfig()
        if 'speed' in df.columns:
            df['speed'] = df['speed'].clip(lower=config.VSW_MIN, upper=config.VSW_MAX)
        if 'density' in df.columns:
            df['density'] = df['density'].clip(lower=config.DENSITY_MIN, upper=config.DENSITY_MAX)
        if 'bz_gsm' in df.columns:
            df['bz_gsm'] = df['bz_gsm'].clip(lower=config.BZ_MIN, upper=config.BZ_MAX)
        
        # 3. INTERPOLAÇÃO INTELIGENTE
        cols_to_interpolate = ['bz_gsm', 'speed', 'density']
        for col in cols_to_interpolate:
            if col in df.columns:
                # Interpolar gaps pequenos
                df[col] = df[col].interpolate(
                    method='linear', 
                    limit=max_interpolation,
                    limit_direction='both'
                )
        
        # 4. REMOÇÃO FINAL DE NaN
        critical_cols = ['speed', 'bz_gsm', 'density']
        df_clean = df.dropna(subset=critical_cols).copy()
        
        # Estatísticas
        original_len = len(df)
        clean_len = len(df_clean)
        retention = clean_len / original_len * 100 if original_len > 0 else 0
        
        if retention < 80:
            print(f"⚠️  Retenção baixa: {retention:.1f}% ({clean_len}/{original_len})")
        
        return df_clean
    
    @staticmethod
    def merge_datasets(mag_df, plasma_df):
        """Fusão robusta de datasets"""
        if mag_df is None or plasma_df is None:
            return None
        
        # Fusão por tempo
        df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
        df = df.sort_values('time_tag').reset_index(drop=True)
        
        # Garantir colunas críticas
        critical_cols = ['speed', 'bz_gsm', 'density']
        for col in critical_cols:
            if col in df.columns:
                # Preencher NaN com valores seguros
                if col == 'speed':
                    df[col] = df[col].fillna(400)  # km/s default
                elif col == 'bz_gsm':
                    df[col] = df[col].fillna(0)    # nT default
                elif col == 'density':
                    df[col] = df[col].fillna(5)    # cm⁻³ default
        
        return df

# ============================
# 2. CÁLCULO DE CAMPOS FÍSICOS (SEMPRE SEGURO)
# ============================
class PhysicalFieldsCalculator:
    """Calcula campos físicos com proteção total contra NaN"""
    
    @staticmethod
    def compute_all_fields(df):
        """
        Calcula TODOS os campos físicos necessários
        SEM NUNCA GERAR NaN
        """
        df = df.copy()
        
        # 1. GARANTIR DADOS DE ENTRADA (CRÍTICO)
        bz = df['bz_gsm'].fillna(0).values
        v = df['speed'].fillna(400).values
        
        # 2. CAMPO ELÉTRICO BRUTO (com proteção)
        # E = -Bz * V (apenas quando Bz < 0)
        bz_negative = np.maximum(0, -bz)  # Converte Bz<0 para positivo, Bz>=0 para 0
        df['E_field_raw'] = bz_negative * v * 1e-3  # mV/m
        
        # 3. SATURAÇÃO FÍSICA (CLIPPING, não tanh!)
        config = HACPhysicsConfig()
        df['E_field_saturated'] = np.clip(
            df['E_field_raw'].values,
            0,
            config.E_FIELD_SATURATION
        )
        
        # 4. RESPOSTA NÃO LINEAR (após saturação)
        threshold = config.COUPLING_THRESHOLD
        beta = config.BETA_NONLINEAR
        
        e_saturated = df['E_field_saturated'].values
        coupling = np.zeros_like(e_saturated)
        
        # Região linear (abaixo do limiar)
        mask_linear = e_saturated <= threshold
        coupling[mask_linear] = e_saturated[mask_linear]
        
        # Região não linear (acima do limiar)
        mask_nonlinear = e_saturated > threshold
        if np.any(mask_nonlinear):
            normalized = e_saturated[mask_nonlinear] / threshold
            coupling[mask_nonlinear] = threshold * (normalized ** beta)
        
        df['coupling_nonlinear'] = coupling
        
        # 5. SINAL DE ACOPLAMENTO (0 quando Bz positivo)
        # CORREÇÃO CRÍTICA: usar bz original, não bz_negative
        coupling_signal = np.where(bz < 0, coupling, 0.0)
        df['coupling_signal'] = coupling_signal
        
        # 6. VALIDAÇÃO
        print(f"   • E-field máximo: {df['E_field_raw'].max():.1f} mV/m")
        print(f"   • E-field saturado: {df['E_field_saturated'].max():.1f} mV/m")
        print(f"   • Sinal acoplamento: {df['coupling_signal'].max():.1f}")
        
        # Garantir nenhum NaN
        for col in ['E_field_raw', 'E_field_saturated', 'coupling_signal']:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
                print(f"⚠️  NaN corrigido em {col}")
        
        return df

# ============================
# 3. MODELO HAC+ CORRIGIDO (SEM NaN, SEM SATURAÇÃO)
# ============================
class ProductionHACModel:
    """Modelo HAC+ de produção com física correta"""
    
    def __init__(self, config=None):
        self.config = config or HACPhysicsConfig()
        self.results = {}
    
    def compute_hac_system(self, df):
        """
        Sistema HAC+ completo com tratamento numérico robusto
        """
        print("\n⚡ Calculando sistema HAC+...")
        
        # Extrair dados com proteção
        times = pd.to_datetime(df['time_tag']).values
        coupling = df['coupling_signal'].fillna(0).values
        Bz = df['bz_gsm'].fillna(0).values
        
        # 1. DELTA-T SEGURO
        dt = self._safe_deltat(times)
        
        # 2. INICIALIZAR RESERVATÓRIOS
        n = len(times)
        hac_ring = np.zeros(n)
        hac_substorm = np.zeros(n)
        hac_ionosphere = np.zeros(n)
        
        # 3. CONSTANTES DE TEMPO (segundos)
        tau_rc = self.config.TAU_RING_CURRENT * 3600
        tau_sub = self.config.TAU_SUBSTORM * 3600
        tau_ion = self.config.TAU_IONOSPHERE * 3600
        
        # 4. SIMULAÇÃO TEMPORAL (PROTEGIDA)
        print("   Simulando reservatórios...")
        for i in range(1, n):
            # Fatores de decaimento
            alpha_rc = np.exp(-dt[i] / tau_rc) if dt[i] > 0 else 0
            alpha_sub = np.exp(-dt[i] / tau_sub) if dt[i] > 0 else 0
            alpha_ion = np.exp(-dt[i] / tau_ion) if dt[i] > 0 else 0
            
            # Injeção (garantida não-NaN)
            injection = coupling[i] if not np.isnan(coupling[i]) else 0
            
            # EQUAÇÕES CORRIGIDAS (com ponderação)
            hac_ring[i] = alpha_rc * hac_ring[i-1] + self.config.ALPHA_RING * injection * dt[i]
            hac_substorm[i] = alpha_sub * hac_substorm[i-1] + self.config.ALPHA_SUBSTORM * injection * dt[i]
            hac_ionosphere[i] = alpha_ion * hac_ionosphere[i-1] + self.config.ALPHA_IONOSPHERE * injection * dt[i]
        
        # 5. COMBINAÇÃO PONDERADA (CORREÇÃO CRÍTICA)
        hac_total = (
            self.config.ALPHA_RING * hac_ring +
            self.config.ALPHA_SUBSTORM * hac_substorm +
            self.config.ALPHA_IONOSPHERE * hac_ionosphere
        )
        
        # 6. NORMALIZAÇÃO SEGURA
        hac_total = self._safe_normalization(hac_total)
        
        # 7. ARMAZENAR
        self.results.update({
            'time': times,
            'HAC_total': hac_total,
            'HAC_ring': hac_ring,
            'HAC_substorm': hac_substorm,
            'HAC_ionosphere': hac_ionosphere,
            'Bz': Bz,
            'coupling_signal': coupling
        })
        
        # 8. VALIDAÇÃO
        self._validate_output(hac_total)
        
        return hac_total
    
    def _safe_deltat(self, times):
        """Calcula delta-t com proteção"""
        n = len(times)
        dt = np.full(n, 60.0)  # Default 60s
        
        if n > 1:
            for i in range(1, n):
                try:
                    delta = (times[i] - times[i-1]).total_seconds()
                    dt[i] = max(delta, 1.0)
                except:
                    dt[i] = 60.0
            
            dt[0] = dt[1] if n > 1 else 60.0
        
        return dt
    
    def _safe_normalization(self, values):
        """Normalização que NUNCA gera NaN"""
        # Encontrar máximo ignorando NaN
        max_val = np.nanmax(values) if len(values) > 0 else 1.0
        
        if max_val > 0:
            normalized = values / max_val * self.config.HAC_SCALE_MAX
        else:
            normalized = np.zeros_like(values)
        
        # Proteção final
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=self.config.HAC_SCALE_MAX, neginf=0.0)
        
        print(f"   • HAC máximo: {np.max(normalized):.1f}")
        print(f"   • HAC médio: {np.mean(normalized):.1f}")
        
        return normalized
    
    def _validate_output(self, hac_values):
        """Validação rigorosa dos resultados"""
        # Verificar NaN
        nan_count = np.sum(np.isnan(hac_values))
        if nan_count > 0:
            print(f"❌ ERRO CRÍTICO: {nan_count} NaN em HAC")
            raise ValueError("NaN detectado em HAC")
        
        # Verificar valores físicos
        if np.max(hac_values) > self.config.HAC_SCALE_MAX * 1.5:
            print(f"⚠️  AVISO: HAC excedeu escala ({np.max(hac_values):.1f})")
        
        print("   ✅ Validação passada")
    
    def predict_storm_indicators(self, hac_values):
        """
        Predição robusta de indicadores de tempestade
        """
        print("\n🌍 Predizendo indicadores...")
        
        # 1. Kp COM SATURAÇÃO
        kp_pred = self.config.KP_SCALE * np.tanh(
            hac_values / self.config.HAC_SCALE_MAX * 2
        )
        
        # 2. Dst EQUIVALENTE
        dst_pred = -self.config.RING_CURRENT_MAX * (
            hac_values / self.config.HAC_SCALE_MAX
        ) ** 1.3
        
        # 3. CLASSIFICAÇÃO NOAA
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
        
        # 4. ARMAZENAR
        self.results.update({
            'Kp_pred': kp_pred,
            'Dst_pred': dst_pred,
            'Storm_level': storm_levels
        })
        
        # 5. ESTATÍSTICAS
        g4g5_count = sum(1 for l in storm_levels if l in ['G4', 'G5'])
        print(f"   • Kp máximo: {np.max(kp_pred):.1f}")
        print(f"   • Dst mínimo: {np.min(dst_pred):.1f} nT")
        print(f"   • Eventos G4/G5: {g4g5_count}")
        
        return kp_pred, dst_pred, storm_levels

# ============================
# 4. VISUALIZAÇÃO DE PRODUÇÃO
# ============================
class ProductionVisualizer:
    """Visualização profissional para produção"""
    
    @staticmethod
    def create_final_dashboard(results, df, filename="hac_final_production.png"):
        """Cria dashboard final de produção"""
        print(f"\n📈 Criando dashboard: {filename}")
        
        # Verificar dados
        if len(results.get('HAC_total', [])) < 10:
            print("❌ Dados insuficientes")
            return None
        
        # Criar figura
        plt.style.use('default')
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle('HAC+ Model - Sistema Completo de Previsão', fontsize=14, fontweight='bold')
        
        # ===== PAINEL 1: HAC TOTAL =====
        ax1 = axes[0, 0]
        if 'HAC_total' in results:
            ax1.plot(results['time'], results['HAC_total'], 
                    color='#d62728', linewidth=2, label='HAC Total')
        
        # Thresholds
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
        levels = [50, 100, 150, 200, 250]
        for lvl, col in zip(levels, colors):
            ax1.axhline(y=lvl, color=col, linestyle=':', alpha=0.5)
        
        ax1.set_ylabel('HAC Index', fontsize=10)
        ax1.set_title('A. Estado do Reservatório', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 320)
        
        # ===== PAINEL 2: COMPONENTES =====
        ax2 = axes[0, 1]
        if all(k in results for k in ['HAC_ring', 'HAC_substorm', 'HAC_ionosphere']):
            ax2.plot(results['time'], results['HAC_ring'], label='Corrente de Anel', alpha=0.7)
            ax2.plot(results['time'], results['HAC_substorm'], label='Subtempestades', alpha=0.7)
            ax2.plot(results['time'], results['HAC_ionosphere'], label='Ionosfera', alpha=0.7)
        
        ax2.set_ylabel('Componentes HAC', fontsize=10)
        ax2.set_title('B. Particionamento de Energia', fontsize=11)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # ===== PAINEL 3: Kp PREVISTO =====
        ax3 = axes[1, 0]
        if 'Kp_pred' in results:
            ax3.plot(results['time'], results['Kp_pred'], 
                    color='#e74c3c', linewidth=1.5, label='Kp previsto')
            ax3.axhline(y=8, color='red', linestyle='--', alpha=0.5, label='Saturação')
        
        ax3.set_ylabel('Índice Kp', fontsize=10)
        ax3.set_title('C. Atividade Geomagnética Prevista', fontsize=11)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 9.5)
        
        # ===== PAINEL 4: Bz =====
        ax4 = axes[1, 1]
        if 'Bz' in results:
            ax4.plot(results['time'], results['Bz'], 
                    color='#2ecc71', linewidth=1, label='Bz')
            ax4.fill_between(results['time'], 0, results['Bz'],
                            where=(results['Bz'] < 0),
                            color='red', alpha=0.3, label='IMF Sul')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_ylabel('Bz [nT]', fontsize=10)
        ax4.set_title('D. Forçante do Vento Solar', fontsize=11)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # ===== PAINEL 5: ACOPLAMENTO =====
        ax5 = axes[2, 0]
        if 'coupling_signal' in results:
            ax5.plot(results['time'], results['coupling_signal'],
                    color='#9b59b6', linewidth=1, label='Acoplamento')
        
        ax5.set_ylabel('Acoplamento [mV/m]', fontsize=10)
        ax5.set_xlabel('Tempo (UTC)', fontsize=10)
        ax5.set_title('E. Sinal de Acoplamento Efetivo', fontsize=11)
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # ===== PAINEL 6: VELOCIDADE =====
        ax6 = axes[2, 1]
        if 'speed' in df.columns:
            ax6.plot(df['time_tag'], df['speed'],
                    color='#3498db', linewidth=1, label='Velocidade')
        
        ax6.set_ylabel('V [km/s]', fontsize=10)
        ax6.set_xlabel('Tempo (UTC)', fontsize=10)
        ax6.set_title('F. Velocidade do Vento Solar', fontsize=11)
        ax6.legend(loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dashboard salvo: {filename}")
        return fig

# ============================
# 5. RELATÓRIO FINAL
# ============================
class FinalReport:
    """Gera relatório final completo"""
    
    @staticmethod
    def generate_report(results, df, filename="hac_final_report.txt"):
        """Relatório final do sistema"""
        print("\n" + "="*70)
        print("📊 RELATÓRIO FINAL - SISTEMA HAC+")
        print("="*70)
        
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("RELATÓRIO FINAL - SISTEMA HAC+ (PRODUÇÃO)\n")
            f.write("="*60 + "\n\n")
            
            # 1. INFORMAÇÕES BÁSICAS
            f.write("1. INFORMAÇÕES DO DATASET\n")
            f.write("-"*40 + "\n")
            f.write(f"Período: {df['time_tag'].min()} a {df['time_tag'].max()}\n")
            f.write(f"Pontos totais: {len(df)}\n")
            f.write(f"Duração: {(df['time_tag'].max() - df['time_tag'].min()).total_seconds()/3600:.1f} horas\n\n")
            
            # 2. DADOS DE ENTRADA
            f.write("2. DADOS DE ENTRADA (estatísticas)\n")
            f.write("-"*40 + "\n")
            stats_cols = ['speed', 'bz_gsm', 'density']
            for col in stats_cols:
                if col in df.columns:
                    f.write(f"{col:10s}: min={df[col].min():7.2f}, "
                           f"max={df[col].max():7.2f}, "
                           f"mean={df[col].mean():7.2f}\n")
            f.write("\n")
            
            # 3. RESULTADOS HAC
            f.write("3. RESULTADOS DO MODELO HAC+\n")
            f.write("-"*40 + "\n")
            
            if 'HAC_total' in results:
                hac = results['HAC_total']
                f.write(f"HAC máximo: {np.max(hac):.2f}\n")
                f.write(f"HAC mínimo: {np.min(hac):.2f}\n")
                f.write(f"HAC médio:  {np.mean(hac):.2f}\n")
                f.write(f"Desvio padrão: {np.std(hac):.2f}\n\n")
                
                # Distribuição de níveis
                if 'Storm_level' in results:
                    levels = results['Storm_level']
                    total = len(levels)
                    f.write("Distribuição de níveis:\n")
                    for lvl in ['Quiet', 'G1', 'G2', 'G3', 'G4', 'G5']:
                        count = sum(1 for x in levels if x == lvl)
                        pct = count/total*100 if total > 0 else 0
                        f.write(f"  {lvl:6s}: {count:4d} pontos ({pct:5.1f}%)\n")
            
            # 4. PREDIÇÕES
            f.write("\n4. PREDIÇÕES GEOMAGNÉTICAS\n")
            f.write("-"*40 + "\n")
            
            if 'Kp_pred' in results:
                kp = results['Kp_pred']
                f.write(f"Kp máximo previsto: {np.max(kp):.2f}\n")
                f.write(f"Kp médio previsto:  {np.mean(kp):.2f}\n")
            
            if 'Dst_pred' in results:
                dst = results['Dst_pred']
                f.write(f"Dst mínimo previsto: {np.min(dst):.2f} nT\n")
                f.write(f"Dst médio previsto:  {np.mean(dst):.2f} nT\n")
            
            # 5. STATUS FINAL
            f.write("\n5. STATUS FINAL DO SISTEMA\n")
            f.write("-"*40 + "\n")
            
            if 'Storm_level' in results and len(results['Storm_level']) > 0:
                current_level = results['Storm_level'][-1]
                f.write(f"Nível atual: {current_level}\n")
                
                if current_level in ['G4', 'G5']:
                    f.write("🚨 ALERTA: Condições de tempestade severa\n")
                elif current_level == 'G3':
                    f.write("⚠️  ALERTA: Tempestade forte\n")
                elif current_level == 'G2':
                    f.write("📢 ATENÇÃO: Tempestade moderada\n")
                elif current_level == 'G1':
                    f.write("📋 MONITORAMENTO: Tempestade menor\n")
                else:
                    f.write("✅ Condições quietas\n")
            
            f.write(f"\nRelatório gerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n")
        
        print(f"📝 Relatório final salvo: {filename}")
        
        # Resumo no console
        if 'Storm_level' in results and len(results['Storm_level']) > 0:
            current_level = results['Storm_level'][-1]
            current_hac = results['HAC_total'][-1] if 'HAC_total' in results else 0
            
            print(f"\n🎯 STATUS ATUAL: HAC = {current_hac:.1f} → {current_level}")
            
            if current_level in ['G4', 'G5']:
                print("   🚨 ALERTA DE TEMPESTADE SEVERA")
            elif current_level == 'G3':
                print("   ⚠️  ALERTA DE TEMPESTADE FORTE")
        
        print("\n" + "="*70)

# ============================
# 6. PIPELINE PRINCIPAL (CORRIGIDO)
# ============================
def main():
    """Pipeline principal - PRODUÇÃO FINAL"""
    print("\n" + "="*70)
    print("🚀 HAC++ MODEL - SISTEMA DE PRODUÇÃO FINAL")
    print("="*70)
    print("Com normalização OMNI completa e física corrigida")
    print("="*70)
    
    # Configurar caminhos
    MAG_FILE = "data/mag-7-day.json"
    PLASMA_FILE = "data/plasma-7-day.json"
    
    # 1. CARREGAR E NORMALIZAR DADOS
    print("\n📥 CARREGANDO DADOS OMNI (com normalização)...")
    
    processor = RobustOMNIProcessor()
    mag_df = processor.load_and_clean(MAG_FILE)
    plasma_df = processor.load_and_clean(PLASMA_FILE)
    
    if mag_df is None or plasma_df is None:
        print("❌ Falha no carregamento")
        return
    
    # 2. FUSÃO
    print("\n🔧 FUNDINDO DATASETS...")
    df = processor.merge_datasets(mag_df, plasma_df)
    
    if df is None or len(df) < 10:
        print("❌ Dados insuficientes")
        return
    
    print(f"   Dataset final: {len(df)} pontos")
    
    # 3. CAMPOS FÍSICOS (SEMPRE SEGURO)
    print("\n⚡ CALCULANDO CAMPOS FÍSICOS...")
    calculator = PhysicalFieldsCalculator()
    df = calculator.compute_all_fields(df)
    
    # 4. MODELO HAC+
    print("\n🧮 EXECUTANDO MODELO HAC+ (produção)...")
    model = ProductionHACModel()
    hac_values = model.compute_hac_system(df)
    
    # 5. PREDIÇÃO
    print("\n🌍 GERANDO PREDIÇÕES...")
    kp_pred, dst_pred, storm_levels = model.predict_storm_indicators(hac_values)
    
    # 6. VISUALIZAÇÃO
    print("\n📈 CRIANDO VISUALIZAÇÕES...")
    visualizer = ProductionVisualizer()
    visualizer.create_final_dashboard(model.results, df, "hac_production_final.png")
    
    # 7. RELATÓRIO
    print("\n📊 GERANDO RELATÓRIO FINAL...")
    reporter = FinalReport()
    reporter.generate_report(model.results, df)
    
    # 8. SALVAR RESULTADOS
    try:
        # Adicionar resultados ao DataFrame
        results_df = df.copy()
        for key, value in model.results.items():
            if key != 'time':
                results_df[key] = value
        
        # Salvar
        output_file = "hac_production_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Resultados salvos: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Erro ao salvar: {e}")
    
    # 9. STATUS FINAL
    print("\n" + "="*70)
    print("✅ SISTEMA HAC++ - EXECUÇÃO CONCLUÍDA")
    print("="*70)
    
    if 'Storm_level' in model.results and len(model.results['Storm_level']) > 0:
        current_level = model.results['Storm_level'][-1]
        current_hac = model.results['HAC_total'][-1]
        
        print(f"\n🔴 STATUS OPERACIONAL:")
        print(f"   HAC: {current_hac:.1f}")
        print(f"   Nível: {current_level}")
        
        if 'Kp_pred' in model.results:
            print(f"   Kp previsto: {model.results['Kp_pred'][-1]:.1f}")
        
        print(f"\n📁 ARQUIVOS GERADOS:")
        print(f"   1. hac_production_final.png - Dashboard")
        print(f"   2. hac_production_results.csv - Dados")
        print(f"   3. hac_final_report.txt - Relatório")
    
    print("\n" + "="*70)

# ============================
# EXECUÇÃO
# ============================
if __name__ == "__main__":
    # Configurar display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    
    # Executar
    main()
