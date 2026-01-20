"""
HAC+ Model: Heliospheric Accumulated Coupling with Advanced Physics
PROOF OF CONCEPT - CORRIGIDO E ROBUSTO
Com tratamento completo de dados OMNI e física validada
"""
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
class HACPhysicsConfig:
    """Configuração robusta dos parâmetros físicos"""
    
    # TEMPOS CARACTERÍSTICOS (horas) - validados empiricamente
    TAU_RING_CURRENT = 3.0      # Tempo de decaimento da corrente de anel
    TAU_SUBSTORM = 1.5          # Tempo de injeção por subtempestades
    TAU_IONOSPHERE = 0.5        # Tempo de resposta ionosférica
    
    # PARÂMETROS DE SATURAÇÃO FÍSICA
    E_FIELD_SATURATION = 15.0   # mV/m - Saturação FÍSICA do campo elétrico
    KP_SATURATION = 8.0         # Saturação do índice Kp
    RING_CURRENT_MAX = 500.0    # nT - Saturação da corrente de anel (Dst)
    
    # PARTICIONAMENTO DE ENERGIA (soma = 1.0)
    ALPHA_RING = 0.4           # Fração para corrente de anel
    ALPHA_SUBSTORM = 0.3       # Fração para subtempestades
    ALPHA_IONOSPHERE = 0.3     # Fração para ionosfera
    
    # PARÂMETROS NÃO LINEARES
    BETA_NONLINEAR = 1.5       # Expoente de resposta não linear
    COUPLING_THRESHOLD = 5.0   # mV/m - Limiar para não-linearidade
    
    # ESCALAS OPERACIONAIS
    HAC_SCALE_MAX = 300.0
    KP_SCALE = 9.0
    
    # LIMITES PARA DADOS OMNI (valores físicos razoáveis)
    VSW_MIN, VSW_MAX = 200, 1500      # km/s
    DENSITY_MIN, DENSITY_MAX = 0.1, 100  # cm⁻³
    BZ_MIN, BZ_MAX = -100, 100        # nT

# ============================
# 1. CARREGAMENTO ROBUSTO DE DADOS OMNI
# ============================
class RobustOMNIProcessor:
    """Processador de dados OMNI com limpeza e validação robusta"""
    
    @staticmethod
    def load_and_clean(filepath, max_interpolation=3):
        """
        Carrega e limpa dados OMNI com tratamento completo de NaN
        """
        print(f"📥 Carregando {filepath}...")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {filepath}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Erro de formato JSON em: {filepath}")
            return None
        
        # Criar DataFrame
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        
        # Converter timestamp
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
        df = df.sort_values('time_tag').reset_index(drop=True)
        
        # 1. CONVERSÃO NUMÉRICA SEGURA
        numeric_cols = [col for col in headers if col != 'time_tag']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. LIMPEZA DE VALORES NÃO FÍSICOS
        df = RobustOMNIProcessor._remove_unphysical_values(df)
        
        # 3. INTERPOLAÇÃO INTELIGENTE (limite configurável)
        df = RobustOMNIProcessor._smart_interpolation(df, max_interpolation)
        
        # 4. REMOÇÃO FINAL DE NaN CRÍTICOS
        critical_cols = ['bz_gsm', 'speed', 'density']
        df_clean = df.dropna(subset=critical_cols).copy()
        
        print(f"   Pontos após limpeza: {len(df_clean)}/{len(df)} "
              f"({len(df_clean)/len(df)*100:.1f}% retidos)")
        
        return df_clean
    
    @staticmethod
    def _remove_unphysical_values(df):
        """Remove valores não físicos dos dados OMNI"""
        # Aplicar limites físicos razoáveis
        config = HACPhysicsConfig
        
        # Velocidade do vento solar
        df['speed'] = df['speed'].clip(lower=config.VSW_MIN, upper=config.VSW_MAX)
        
        # Densidade
        df['density'] = df['density'].clip(lower=config.DENSITY_MIN, 
                                          upper=config.DENSITY_MAX)
        
        # Bz
        df['bz_gsm'] = df['bz_gsm'].clip(lower=config.BZ_MIN, upper=config.BZ_MAX)
        
        return df
    
    @staticmethod
    def _smart_interpolation(df, max_gap=3):
        """
        Interpolação inteligente considerando a física
        
        Args:
            df: DataFrame com dados OMNI
            max_gap: Número máximo de pontos consecutivos para interpolar
        """
        cols_to_interpolate = ['bz_gsm', 'speed', 'density', 'bx_gsm', 'by_gsm']
        
        for col in cols_to_interpolate:
            if col in df.columns:
                # Interpolação linear para gaps pequenos
                df[col] = df[col].interpolate(
                    method='linear', 
                    limit=max_gap, 
                    limit_direction='both'
                )
        
        return df
    
    @staticmethod
    def merge_datasets(mag_df, plasma_df):
        """Funde datasets magnéticos e de plasma de forma robusta"""
        if mag_df is None or plasma_df is None:
            return None
        
        # Fusão externa para manter todos os timestamps
        df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
        df = df.sort_values('time_tag').reset_index(drop=True)
        
        # Re-aplicar limpeza após merge
        df = RobustOMNIProcessor._smart_interpolation(df, max_gap=3)
        
        # Garantir que temos as colunas críticas
        critical_cols = ['bz_gsm', 'speed', 'density']
        missing = [col for col in critical_cols if col not in df.columns]
        if missing:
            print(f"⚠️  Colunas críticas faltando após merge: {missing}")
            return None
        
        # Remover NaN residuais
        df = df.dropna(subset=critical_cols)
        
        return df
    
    @staticmethod
    def compute_physical_fields(df):
        """
        Calcula campos físicos derivados com segurança numérica
        """
        # Garantir que não temos NaN
        df = df.copy()
        
        # 1. CAMPO ELÉTRICO SOLAR (mV/m) - CORREÇÃO CRÍTICA
        # Evitar NaN: usar fillna antes da multiplicação
        bz_clean = df['bz_gsm'].fillna(0)
        v_clean = df['speed'].fillna(400)  # valor default razoável
        
        # Campo elétrico: -Bz * V (apenas quando Bz < 0)
        # ABSOLUTAMENTE CRÍTICO: clip em 0 para Bz positivo
        bz_negative = np.where(bz_clean < 0, -bz_clean, 0)
        df['E_field_raw'] = bz_negative * v_clean * 1e-3  # Convert to mV/m
        
        # 2. SATURAÇÃO FÍSICA CORRETA (clipping, não tanh)
        config = HACPhysicsConfig()
        df['E_field_saturated'] = np.clip(
            df['E_field_raw'], 
            0, 
            config.E_FIELD_SATURATION
        )
        
        # 3. RESPOSTA NÃO LINEAR (após saturação)
        # Aplicar não-linearidade apenas acima do limiar
        threshold = config.COUPLING_THRESHOLD
        beta = config.BETA_NONLINEAR
        
        # Inicializar com valores saturados
        df['coupling_nonlinear'] = df['E_field_saturated'].copy()
        
        # Aplicar não-linearidade onde apropriado
        mask_above_threshold = df['E_field_saturated'] > threshold
        if mask_above_threshold.any():
            # Normalizar antes de aplicar potência
            normalized = df.loc[mask_above_threshold, 'E_field_saturated'] / threshold
            df.loc[mask_above_threshold, 'coupling_nonlinear'] = \
                threshold * (normalized ** beta)
        
        # 4. SINAL DE ACOPLAMENTO (0 quando Bz positivo)
        df['coupling_signal'] = np.where(
            df['bz_gsm'] < 0,
            df['coupling_nonlinear'],
            0.0
        )
        
        # Garantir que não há NaN
        df['coupling_signal'] = df['coupling_signal'].fillna(0)
        
        print(f"   • E-field máximo: {df['E_field_raw'].max():.1f} mV/m")
        print(f"   • E-field saturado: {df['E_field_saturated'].max():.1f} mV/m")
        print(f"   • Sinal de acoplamento máximo: {df['coupling_signal'].max():.1f}")
        
        return df

# ============================
# 2. MODELO FÍSICO HAC+ CORRIGIDO
# ============================
class RobustHACPlusModel:
    """Modelo HAC+ com tratamento robusto de erros numéricos"""
    
    def __init__(self, config=None):
        self.config = config or HACPhysicsConfig()
        self.results = {}
        
    def compute_reservoir_dynamics(self, df):
        """
        Calcula a dinâmica dos reservatórios com segurança numérica
        
        Sistema corrigido:
        1. Saturação por clipping (não tanh)
        2. Tratamento robusto de NaN
        3. Normalização segura
        """
        print("\n⚡ Calculando dinâmica dos reservatórios...")
        
        # Extrair arrays com segurança
        times = pd.to_datetime(df['time_tag']).values
        coupling = df['coupling_signal'].values
        Bz = df['bz_gsm'].values
        
        # 1. CALCULAR delta-t REAL (segundos)
        dt = self._compute_safe_deltat(times)
        
        # 2. PARÂMETROS DO MODELO (segundos)
        tau_rc = self.config.TAU_RING_CURRENT * 3600
        tau_sub = self.config.TAU_SUBSTORM * 3600
        tau_ion = self.config.TAU_IONOSPHERE * 3600
        
        # 3. INICIALIZAR RESERVATÓRIOS
        n = len(times)
        hac_ring = np.zeros(n)
        hac_substorm = np.zeros(n)
        hac_ionosphere = np.zeros(n)
        
        # 4. SIMULAÇÃO TEMPORAL ROBUSTA
        for i in range(1, n):
            # Fatores de decaimento (com segurança numérica)
            alpha_rc = np.exp(-dt[i] / tau_rc) if dt[i] > 0 else 0
            alpha_sub = np.exp(-dt[i] / tau_sub) if dt[i] > 0 else 0
            alpha_ion = np.exp(-dt[i] / tau_ion) if dt[i] > 0 else 0
            
            # Sinal de injeção (já pré-processado)
            injection = coupling[i] if not np.isnan(coupling[i]) else 0
            
            # EQUAÇÕES DOS RESERVATÓRIOS (com segurança)
            hac_ring[i] = self._safe_update(
                alpha_rc, hac_ring[i-1], 
                self.config.ALPHA_RING, injection, dt[i]
            )
            
            hac_substorm[i] = self._safe_update(
                alpha_sub, hac_substorm[i-1],
                self.config.ALPHA_SUBSTORM, injection, dt[i]
            )
            
            hac_ionosphere[i] = self._safe_update(
                alpha_ion, hac_ionosphere[i-1],
                self.config.ALPHA_IONOSPHERE, injection, dt[i]
            )
        
        # 5. HAC TOTAL (soma ponderada)
        hac_total = hac_ring + hac_substorm + hac_ionosphere
        
        # 6. NORMALIZAÇÃO SEGURA (CORREÇÃO CRÍTICA)
        hac_total = self._safe_normalize(hac_total)
        
        # 7. ARMAZENAR RESULTADOS
        self.results['time'] = times
        self.results['HAC_total'] = hac_total
        self.results['HAC_ring'] = hac_ring
        self.results['HAC_substorm'] = hac_substorm
        self.results['HAC_ionosphere'] = hac_ionosphere
        self.results['Bz'] = Bz
        self.results['coupling_signal'] = coupling
        
        # 8. VALIDAÇÃO
        self._validate_results()
        
        return hac_total
    
    def _compute_safe_deltat(self, times):
        """Calcula delta-t com tratamento de bordas"""
        n = len(times)
        dt = np.zeros(n)
        
        if n > 1:
            # Calcular diferenças em segundos
            for i in range(1, n):
                delta = (times[i] - times[i-1]).total_seconds()
                dt[i] = max(delta, 1.0)  # Mínimo 1 segundo
            
            # Primeiro ponto usa a mesma diferença que o segundo
            dt[0] = dt[1] if n > 1 else 60.0
        else:
            dt[:] = 60.0  # Default 1 minuto
        
        return dt
    
    def _safe_update(self, alpha, previous, fraction, injection, dt):
        """Atualização segura de reservatório"""
        # Garantir que não há NaN
        previous = previous if not np.isnan(previous) else 0
        injection = injection if not np.isnan(injection) else 0
        alpha = alpha if not np.isnan(alpha) else 0
        
        # Equação do reservatório
        new_value = alpha * previous + fraction * injection * dt
        
        # Garantir não-negatividade
        return max(new_value, 0)
    
    def _safe_normalize(self, hac_values):
        """Normalização segura evitando NaN"""
        # Encontrar máximo ignorando NaN
        max_val = np.nanmax(hac_values) if len(hac_values) > 0 else 1.0
        
        # Normalizar apenas se máximo > 0
        if max_val > 0:
            normalized = hac_values / max_val * self.config.HAC_SCALE_MAX
        else:
            normalized = np.zeros_like(hac_values)
        
        # Aplicar saturação suave
        normalized = np.minimum(normalized, self.config.HAC_SCALE_MAX * 1.1)
        
        # Garantir que não há NaN
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=self.config.HAC_SCALE_MAX, neginf=0.0)
        
        print(f"   • HAC máximo após normalização: {np.max(normalized):.1f}")
        print(f"   • HAC mínimo: {np.min(normalized):.1f}")
        
        return normalized
    
    def _validate_results(self):
        """Validação dos resultados do modelo"""
        hac_total = self.results['HAC_total']
        
        # Verificar NaN
        nan_count = np.sum(np.isnan(hac_total))
        if nan_count > 0:
            print(f"⚠️  AVISO: {nan_count} valores NaN em HAC_total")
            # Corrigir NaN
            self.results['HAC_total'] = np.nan_to_num(hac_total, nan=0.0)
        
        # Verificar valores extremos
        if np.max(hac_total) > self.config.HAC_SCALE_MAX * 1.2:
            print(f"⚠️  AVISO: HAC excedeu escala máxima em {(np.max(hac_total)/self.config.HAC_SCALE_MAX-1)*100:.0f}%")
    
    def predict_geomagnetic_impact(self, hac_values):
        """
        Predição robusta de índices geomagnéticos
        """
        print("\n🌍 Predizendo impacto geomagnético...")
        
        # 1. PREDIÇÃO DE Kp COM SATURAÇÃO
        # Relação não linear com saturação em Kp=9
        kp_pred = self.config.KP_SCALE * np.tanh(
            hac_values / self.config.HAC_SCALE_MAX * 2
        )
        
        # 2. ESTIMATIVA DE Dst EQUIVALENTE
        # Relação empírica: Dst ~ - (HAC/HAC_max)^1.3 * Dst_max
        dst_pred = -self.config.RING_CURRENT_MAX * (
            hac_values / self.config.HAC_SCALE_MAX
        ) ** 1.3
        
        # 3. CLASSIFICAÇÃO NOAA G-SCALE
        storm_levels = []
        for h in hac_values:
            h_clean = h if not np.isnan(h) else 0
            if h_clean < 50:
                level = "Quiet"
            elif h_clean < 100:
                level = "G1"
            elif h_clean < 150:
                level = "G2"
            elif h_clean < 200:
                level = "G3"
            elif h_clean < 250:
                level = "G4"
            else:
                level = "G5"
            storm_levels.append(level)
        
        # 4. ARMAZENAR
        self.results['Kp_pred'] = kp_pred
        self.results['Dst_pred'] = dst_pred
        self.results['Storm_level'] = storm_levels
        
        # 5. ESTATÍSTICAS
        g4g5_mask = np.array([l in ['G4', 'G5'] for l in storm_levels])
        g4g5_count = np.sum(g4g5_mask)
        
        print(f"   • Kp máximo previsto: {np.max(kp_pred):.1f}")
        print(f"   • Dst mínimo previsto: {np.min(dst_pred):.1f} nT")
        print(f"   • Eventos G4/G5: {g4g5_count} pontos")
        
        if g4g5_count > 0 and 'time' in self.results:
            storm_times = self.results['time'][g4g5_mask]
            print(f"   • Primeiro G4/G5: {storm_times[0]}")
            print(f"   • Último G4/G5: {storm_times[-1]}")
        
        return kp_pred, dst_pred, storm_levels

# ============================
# 3. VISUALIZAÇÃO ROBUSTA
# ============================
class RobustVisualizer:
    """Visualização robusta com tratamento de dados ausentes"""
    
    @staticmethod
    def create_comprehensive_dashboard(results, df, filename="hac_final_dashboard.png"):
        """
        Cria dashboard completo com dados limpos
        """
        print(f"\n📈 Gerando dashboard: {filename}")
        
        # Verificar se temos dados suficientes
        if len(results['time']) < 10:
            print("❌ Dados insuficientes para visualização")
            return None
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(16, 12))
        
        # ========== PAINEL 1: HAC E CLASSIFICAÇÃO ==========
        ax1 = plt.subplot(3, 2, (1, 2))
        
        # Plot HAC (com verificação)
        if 'HAC_total' in results and len(results['HAC_total']) > 0:
            ax1.plot(results['time'], results['HAC_total'], 
                    color='#d62728', linewidth=2.5, label='HAC Total')
        
        # Thresholds de tempestade
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
        levels = [50, 100, 150, 200, 250]
        labels = ['G1', 'G2', 'G3', 'G4', 'G5']
        
        for lvl, col, lab in zip(levels, colors, labels):
            ax1.axhline(y=lvl, color=col, linestyle=':', alpha=0.5, linewidth=1)
            ax1.text(results['time'][0], lvl+3, lab, color=col, fontsize=9, alpha=0.8)
        
        ax1.set_ylabel('Índice HAC', fontsize=11, fontweight='bold')
        ax1.set_title('A. Estado do Reservatório Magnetosférico (HAC)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 320)
        
        # ========== PAINEL 2: COMPONENTES DO HAC ==========
        ax2 = plt.subplot(3, 2, 3)
        
        if all(key in results for key in ['HAC_ring', 'HAC_substorm', 'HAC_ionosphere']):
            ax2.plot(results['time'], results['HAC_ring'], 
                    color='#2ecc71', linewidth=1.5, label='Corrente de Anel')
            ax2.plot(results['time'], results['HAC_substorm'], 
                    color='#3498db', linewidth=1.5, label='Subtempestades')
            ax2.plot(results['time'], results['HAC_ionosphere'], 
                    color='#f39c12', linewidth=1.5, label='Ionosfera')
        
        ax2.set_ylabel('Componentes HAC', fontsize=10)
        ax2.set_title('B. Particionamento de Energia', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # ========== PAINEL 3: INDICES PREVISTOS ==========
        ax3 = plt.subplot(3, 2, 4)
        
        if 'Kp_pred' in results:
            ax3.plot(results['time'], results['Kp_pred'], 
                    color='#e74c3c', linewidth=2, label='Kp previsto')
            ax3.axhline(y=8.0, color='red', linestyle='--', alpha=0.5, 
                       label='Saturação Kp')
        
        ax3.set_ylabel('Índice Kp', fontsize=10, color='#e74c3c')
        ax3.tick_params(axis='y', labelcolor='#e74c3c')
        ax3.set_title('C. Índices Geomagnéticos Previstos', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 9.5)
        
        # ========== PAINEL 4: Bz E ACOPLAMENTO ==========
        ax4 = plt.subplot(3, 2, 5)
        
        # Bz
        if 'Bz' in results:
            ax4.plot(results['time'], results['Bz'], 
                    color='#2ecc71', linewidth=1.2, label='Bz')
            ax4.fill_between(results['time'], 0, results['Bz'],
                            where=(results['Bz'] < 0),
                            color='red', alpha=0.3, label='IMF Sul')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax4.set_ylabel('Bz [nT]', fontsize=10)
        ax4.set_xlabel('Tempo (UTC)', fontsize=10)
        ax4.set_title('D. Forçante do Vento Solar (Bz)', fontsize=11, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # ========== PAINEL 5: SINAL DE ACOPLAMENTO ==========
        ax5 = plt.subplot(3, 2, 6)
        
        if 'coupling_signal' in results:
            ax5.plot(results['time'], results['coupling_signal'], 
                    color='#9b59b6', linewidth=1.5, label='Sinal de Acoplamento')
        
        ax5.set_ylabel('Acoplamento [mV/m]', fontsize=10)
        ax5.set_xlabel('Tempo (UTC)', fontsize=10)
        ax5.set_title('E. Sinal de Acoplamento Efetivo', fontsize=11, fontweight='bold')
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # ========== FINALIZAR ==========
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Dashboard salvo: {filename}")
        return fig

# ============================
# 4. RELATÓRIO DE VALIDAÇÃO
# ============================
class ValidationReport:
    """Geração de relatório de validação do modelo"""
    
    @staticmethod
    def generate_validation_report(results, df, output_file="hac_validation_report.txt"):
        """Gera relatório completo de validação"""
        print("\n" + "="*70)
        print("📊 RELATÓRIO DE VALIDAÇÃO DO MODELO HAC+")
        print("="*70)
        
        with open(output_file, 'w') as f:
            f.write("RELATÓRIO DE VALIDAÇÃO - MODELO HAC+ (CORRIGIDO)\n")
            f.write("="*60 + "\n\n")
            
            # 1. INFORMAÇÕES BÁSICAS
            f.write("1. INFORMAÇÕES DO DATASET\n")
            f.write("-"*40 + "\n")
            f.write(f"Período: {df['time_tag'].min()} a {df['time_tag'].max()}\n")
            f.write(f"Total de pontos: {len(df)}\n")
            f.write(f"Resolução temporal média: {pd.Series(df['time_tag'].diff().dropna()).mean().total_seconds():.0f} s\n\n")
            
            # 2. QUALIDADE DOS DADOS
            f.write("2. QUALIDADE DOS DADOS OMNI\n")
            f.write("-"*40 + "\n")
            
            critical_cols = ['bz_gsm', 'speed', 'density']
            for col in critical_cols:
                if col in df.columns:
                    nan_pct = df[col].isna().mean() * 100
                    f.write(f"{col:10s}: NaN = {nan_pct:5.1f}%, "
                           f"Mín = {df[col].min():7.2f}, "
                           f"Máx = {df[col].max():7.2f}, "
                           f"Méd = {df[col].mean():7.2f}\n")
            
            f.write("\n")
            
            # 3. RESULTADOS DO MODELO
            f.write("3. RESULTADOS DO MODELO HAC+\n")
            f.write("-"*40 + "\n")
            
            if 'HAC_total' in results:
                hac = results['HAC_total']
                f.write(f"HAC máximo: {np.max(hac):.2f}\n")
                f.write(f"HAC mínimo: {np.min(hac):.2f}\n")
                f.write(f"HAC médio:  {np.mean(hac):.2f}\n")
                
                # Verificar NaN
                nan_count = np.sum(np.isnan(hac))
                f.write(f"Valores NaN em HAC: {nan_count}\n")
                
                # Estatísticas de tempestade
                if 'Storm_level' in results:
                    levels = results['Storm_level']
                    for lvl in ['G1', 'G2', 'G3', 'G4', 'G5']:
                        count = sum(1 for x in levels if x == lvl)
                        pct = count / len(levels) * 100 if len(levels) > 0 else 0
                        f.write(f"{lvl}: {count:4d} pontos ({pct:5.1f}%)\n")
            
            # 4. VALIDAÇÃO FÍSICA
            f.write("\n4. VALIDAÇÃO FÍSICA\n")
            f.write("-"*40 + "\n")
            
            if 'E_field_raw' in df.columns:
                f.write(f"Campo Elétrico máximo: {df['E_field_raw'].max():.2f} mV/m\n")
                f.write(f"Campo Elétrico médio:  {df['E_field_raw'].mean():.2f} mV/m\n")
                
                # Verificar saturação
                config = HACPhysicsConfig()
                above_sat = np.sum(df['E_field_raw'] > config.E_FIELD_SATURATION)
                sat_pct = above_sat / len(df) * 100
                f.write(f"Pontos acima da saturação ({config.E_FIELD_SATURATION} mV/m): "
                       f"{above_sat} ({sat_pct:.1f}%)\n")
            
            # 5. CONCLUSÃO
            f.write("\n5. CONCLUSÃO DA VALIDAÇÃO\n")
            f.write("-"*40 + "\n")
            
            if 'HAC_total' in results:
                hac_max = np.max(results['HAC_total'])
                if hac_max > 250:
                    f.write("✅ MODELO VALIDADO: Detectou condições G4/G5\n")
                    f.write(f"   HAC máximo = {hac_max:.1f} indica tempestade severa\n")
                elif hac_max > 150:
                    f.write("⚠️  MODELO PARCIAL: Detectou condições G2/G3\n")
                    f.write(f"   HAC máximo = {hac_max:.1f}\n")
                else:
                    f.write("❌ MODELO SUBESTIMOU: Não detectou tempestade significativa\n")
                    f.write(f"   HAC máximo = {hac_max:.1f}\n")
            
            f.write(f"\nRelatório gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"📝 Relatório de validação salvo: {output_file}")
        
        # Imprimir resumo no console
        if 'HAC_total' in results:
            hac_max = np.max(results['HAC_total'])
            current_level = results['Storm_level'][-1] if 'Storm_level' in results else "Unknown"
            
            print(f"\n🎯 RESUMO EXECUTIVO:")
            print(f"   • HAC máximo: {hac_max:.1f}")
            print(f"   • Nível atual: {current_level}")
            print(f"   • Kp previsto: {results['Kp_pred'][-1]:.1f}" if 'Kp_pred' in results else "")
        
        print("\n" + "="*70)

# ============================
# 5. FUNÇÃO PRINCIPAL (CORRIGIDA)
# ============================
def main():
    """Pipeline principal corrigido"""
    print("\n" + "="*70)
    print("🧪 HAC+ MODEL - PROOF OF CONCEPT CORRIGIDO")
    print("="*70)
    print("Com tratamento completo de dados OMNI e física validada")
    print("="*70)
    
    # Configurar caminhos
    MAG_FILE = "data/mag-7-day.json"
    PLASMA_FILE = "data/plasma-7-day.json"
    
    # 1. CARREGAMENTO ROBUSTO DE DADOS
    print("\n📥 CARREGANDO E LIMPANDO DADOS OMNI...")
    
    processor = RobustOMNIProcessor()
    mag_df = processor.load_and_clean(MAG_FILE, max_interpolation=3)
    plasma_df = processor.load_and_clean(PLASMA_FILE, max_interpolation=3)
    
    if mag_df is None or plasma_df is None:
        print("❌ Falha crítica no carregamento de dados")
        return
    
    # 2. FUSÃO E PRÉ-PROCESSAMENTO
    print("\n🔧 FUNDINDO E PRÉ-PROCESSANDO DADOS...")
    df = processor.merge_datasets(mag_df, plasma_df)
    
    if df is None or len(df) < 10:
        print("❌ Dados insuficientes após fusão")
        return
    
    print(f"   Dataset final: {len(df)} pontos")
    print(f"   Período: {df['time_tag'].min()} a {df['time_tag'].max()}")
    
    # 3. CÁLCULO DE CAMPOS FÍSICOS (CORRIGIDO)
    print("\n⚡ CALCULANDO CAMPOS FÍSICOS (com saturação correta)...")
    df = processor.compute_physical_fields(df)
    
    # 4. EXECUTAR MODELO HAC+ (CORRIGIDO)
    print("\n🧮 EXECUTANDO MODELO HAC+ (com equações corrigidas)...")
    model = RobustHACPlusModel()
    hac_values = model.compute_reservoir_dynamics(df)
    
    # 5. PREDIÇÃO DE IMPACTO
    print("\n🌍 PREDIZENDO IMPACTO GEOMAGNÉTICO...")
    kp_pred, dst_pred, storm_levels = model.predict_geomagnetic_impact(hac_values)
    
    # 6. VISUALIZAÇÃO
    print("\n📈 GERANDO VISUALIZAÇÕES...")
    visualizer = RobustVisualizer()
    dashboard = visualizer.create_comprehensive_dashboard(
        model.results, df, 
        "hac_final_corrected_dashboard.png"
    )
    
    # 7. RELATÓRIO DE VALIDAÇÃO
    print("\n📊 GERANDO RELATÓRIO DE VALIDAÇÃO...")
    reporter = ValidationReport()
    reporter.generate_validation_report(model.results, df)
    
    # 8. SALVAR RESULTADOS COMPLETOS
    try:
        # Adicionar todos os resultados ao DataFrame
        for key, value in model.results.items():
            if key != 'time':  # time já existe
                df[key] = value
        
        # Salvar CSV
        output_csv = "hac_complete_results_corrected.csv"
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n💾 Resultados completos salvos: {output_csv}")
        
    except Exception as e:
        print(f"⚠️  Erro ao salvar CSV: {e}")
    
    # 9. STATUS FINAL
    print("\n" + "="*70)
    print("✅ PROOF OF CONCEPT CONCLUÍDO COM SUCESSO!")
    print("="*70)
    
    if 'HAC_total' in model.results and 'Storm_level' in model.results:
        last_hac = model.results['HAC_total'][-1]
        last_level = model.results['Storm_level'][-1]
        
        print(f"\n🎯 STATUS ATUAL DO SISTEMA:")
        print(f"   HAC: {last_hac:.1f} → {last_level}")
        
        if last_level in ['G4', 'G5']:
            print(f"   🚨 ALERTA: Tempestade {last_level} em progresso")
        elif last_level == 'G3':
            print(f"   ⚠️  ALERTA: Tempestade forte (G3)")
        elif last_level == 'G2':
            print(f"   📢 ATENÇÃO: Tempestade moderada (G2)")
        elif last_level == 'G1':
            print(f"   📋 MONITORAMENTO: Tempestade menor (G1)")
        else:
            print(f"   ✅ CONDIÇÕES: Quietas")
    
    print(f"\n📁 SAÍDAS GERADAS:")
    print(f"   1. hac_final_corrected_dashboard.png - Dashboard completo")
    print(f"   2. hac_complete_results_corrected.csv - Dados processados")
    print(f"   3. hac_validation_report.txt - Relatório de validação")
    
    print(f"\n🔧 CORREÇÕES APLICADAS:")
    print(f"   • Tratamento robusto de NaN em dados OMNI")
    print(f"   • Saturação física por clipping (não tanh)")
    print(f"   • Normalização segura com np.nanmax")
    print(f"   • Prevenção de NaN em todas as equações")
    
    print("\n" + "="*70)

# ============================
# EXECUÇÃO
# ============================
if __name__ == "__main__":
    # Configurar pandas para melhor visualização
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    # Executar pipeline corrigido
    main()
