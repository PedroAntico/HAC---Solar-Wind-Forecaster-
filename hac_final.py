"""
HAC++ Model: Heliospheric Accumulated Coupling - PRODUÇÃO FINAL
COM NOWCAST + INÉRCIA (Previsão de Escalação)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')

# ============================
# 0. NORMALIZAÇÃO DE NOMES OMNI (CRÍTICO)
# ============================
def normalize_omni_columns(df, allow_partial=False):
    """Normalização robusta para OMNI (suporta MAG e PLASMA separados)"""

    column_map = {
        # Tempo
        'time': 'time_tag',
        'Time': 'time_tag',
        'Epoch': 'time_tag',
        'timestamp': 'time_tag',

        # Velocidade
        'V': 'speed',
        'Vsw': 'speed',
        'speed': 'speed',

        # Densidade
        'N': 'density',
        'Np': 'density',
        'density': 'density',

        # Campo magnético
        'Bz': 'bz_gsm',
        'Bz_GSM': 'bz_gsm',
        'bz_gsm': 'bz_gsm',
        'Bx': 'bx_gsm',
        'By': 'by_gsm',
        'Bt': 'bt'
    }

    df = df.copy()

    for col in df.columns:
        if col in column_map:
            df.rename(columns={col: column_map[col]}, inplace=True)

    print("\n🔍 Colunas detectadas:", list(df.columns))

    # 🔴 Só valida tudo se NÃO for parcial
    if not allow_partial:
        required = ['time_tag', 'speed', 'density', 'bz_gsm']
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"❌ COLUNAS OBRIGATÓRIAS AUSENTES: {missing}\n"
                f"➡️ Colunas disponíveis: {list(df.columns)}"
            )

    return df
# ============================
# CONFIGURAÇÃO FÍSICA CALIBRADA
# ============================
class HACPhysicsConfig:
    """Configuração física validada para dados OMNI reais"""
    
    # TEMPOS CARACTERÍSTICOS (horas)
    TAU_RING_CURRENT = 3.0      # Tempo de decaimento da corrente de anel
    TAU_SUBSTORM = 1.5          # Tempo de injeção por subtempestades  
    TAU_IONOSPHERE = 0.5        # Tempo de resposta ionosférica
    TAU_EFFECTIVE = 2.0         # τ_eff para modelo Nowcast+Inércia
    
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
    
    # NOWCAST + INÉRCIA PARAMETERS
    THETA_CRITICAL = 50.0      # nT/h - Limiar de crescimento crítico
    HG3_THRESHOLD = 150.0      # Limiar G3
    VSW_CRITICAL = 700.0       # km/s
    BZ_CRITICAL = -8.0         # nT

# ============================
# 1. CARREGAMENTO ROBUSTO DE DADOS OMNI
# ============================
class RobustOMNIProcessor:
    """Processador robusto para dados OMNI reais"""
    
    @staticmethod
    def load_and_clean(filepath, max_interpolation=3):
        """Carrega, normaliza e limpa dados OMNI"""
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
        
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        df = normalize_omni_columns(df, allow_partial=True)
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
        df = df.sort_values('time_tag').reset_index(drop=True)
        
        numeric_cols = [col for col in df.columns if col != 'time_tag']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        config = HACPhysicsConfig()
        if 'speed' in df.columns:
            df['speed'] = df['speed'].clip(lower=config.VSW_MIN, upper=config.VSW_MAX)
        if 'density' in df.columns:
            df['density'] = df['density'].clip(lower=config.DENSITY_MIN, upper=config.DENSITY_MAX)
        if 'bz_gsm' in df.columns:
            df['bz_gsm'] = df['bz_gsm'].clip(lower=config.BZ_MIN, upper=config.BZ_MAX)
        
        cols_to_interpolate = ['bz_gsm', 'speed', 'density']
        for col in cols_to_interpolate:
            if col in df.columns:
                df[col] = df[col].interpolate(
                    method='linear', 
                    limit=max_interpolation,
                    limit_direction='both'
                )
        
        # ==============================
        # DETECÇÃO AUTOMÁTICA DO TIPO
        # ==============================
        has_speed = 'speed' in df.columns
        has_density = 'density' in df.columns
        has_bz = 'bz_gsm' in df.columns

        if has_speed and has_density:
            # Arquivo PLASMA
            required = ['speed', 'density']
        elif has_bz:
            # Arquivo MAG
            required = ['bz_gsm']
        else:
            raise ValueError(
                f"❌ Arquivo inválido. Colunas encontradas: {list(df.columns)}"
            )

        df_clean = df.dropna(subset=required).copy()

        print(f"   ✅ {len(df_clean)} pontos válidos")
        return df_clean
    
    @staticmethod
    def merge_datasets(mag_df, plasma_df):
        """Fusão robusta de datasets"""
        if mag_df is None or plasma_df is None:
            return None
        
        df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
        df = df.sort_values('time_tag').reset_index(drop=True)
        
        critical_cols = ['speed', 'bz_gsm', 'density']
        for col in critical_cols:
            if col in df.columns:
                if col == 'speed':
                    df[col] = df[col].fillna(400)
                elif col == 'bz_gsm':
                    df[col] = df[col].fillna(0)
                elif col == 'density':
                    df[col] = df[col].fillna(5)
        
        return df

# ============================
# 2. CÁLCULO DE CAMPOS FÍSICOS (SEMPRE SEGURO)
# ============================
class PhysicalFieldsCalculator:
    """Calcula campos físicos com proteção total contra NaN"""
    
    @staticmethod
    def compute_all_fields(df):
        """Calcula TODOS os campos físicos necessários SEM NUNCA GERAR NaN"""
        df = df.copy()
        
        bz = df['bz_gsm'].fillna(0).values
        v = df['speed'].fillna(400).values
        
        bz_negative = np.maximum(0, -bz)
        df['E_field_raw'] = bz_negative * v * 1e-3
        
        config = HACPhysicsConfig()
        df['E_field_saturated'] = np.clip(
            df['E_field_raw'].values,
            0,
            config.E_FIELD_SATURATION
        )
        
        threshold = config.COUPLING_THRESHOLD
        beta = config.BETA_NONLINEAR
        e_saturated = df['E_field_saturated'].values
        coupling = np.zeros_like(e_saturated)
        
        mask_linear = e_saturated <= threshold
        coupling[mask_linear] = e_saturated[mask_linear]
        
        mask_nonlinear = e_saturated > threshold
        if np.any(mask_nonlinear):
            normalized = e_saturated[mask_nonlinear] / threshold
            coupling[mask_nonlinear] = threshold * (normalized ** beta)
        
        df['coupling_nonlinear'] = coupling
        coupling_signal = np.where(bz < 0, coupling, 0.0)
        df['coupling_signal'] = coupling_signal
        
        for col in ['E_field_raw', 'E_field_saturated', 'coupling_signal']:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        return df

# ============================
# 3. MODELO HAC+ COM NOWCAST + INÉRCIA
# ============================
class ProductionHACModel:
    """Modelo HAC+ de produção com física correta e Nowcast + Inércia"""
    
    def __init__(self, config=None):
        self.config = config or HACPhysicsConfig()
        self.results = {}
        self.nowcast_alerts = []
        self.escalation_triggers = []
    
    def compute_hac_system(self, df):
        """Sistema HAC+ completo com tratamento numérico robusto"""
        print("\n⚡ Calculando sistema HAC+...")
        
        times = pd.to_datetime(df['time_tag']).values
        coupling = df['coupling_signal'].fillna(0).values
        Bz = df['bz_gsm'].fillna(0).values
        Vsw = df['speed'].fillna(400).values
        
        dt = self._safe_deltat(times)
        
        n = len(times)
        hac_ring = np.zeros(n)
        hac_substorm = np.zeros(n)
        hac_ionosphere = np.zeros(n)
        
        tau_rc = self.config.TAU_RING_CURRENT * 3600
        tau_sub = self.config.TAU_SUBSTORM * 3600
        tau_ion = self.config.TAU_IONOSPHERE * 3600
        
        print("   Simulando reservatórios...")
        for i in range(1, n):
            alpha_rc = np.exp(-dt[i] / tau_rc) if dt[i] > 0 else 0
            alpha_sub = np.exp(-dt[i] / tau_sub) if dt[i] > 0 else 0
            alpha_ion = np.exp(-dt[i] / tau_ion) if dt[i] > 0 else 0
            
            injection = coupling[i] if not np.isnan(coupling[i]) else 0
            
            hac_ring[i] = alpha_rc * hac_ring[i-1] + self.config.ALPHA_RING * injection * dt[i]
            hac_substorm[i] = alpha_sub * hac_substorm[i-1] + self.config.ALPHA_SUBSTORM * injection * dt[i]
            hac_ionosphere[i] = alpha_ion * hac_ionosphere[i-1] + self.config.ALPHA_IONOSPHERE * injection * dt[i]
        
        hac_total = (
            self.config.ALPHA_RING * hac_ring +
            self.config.ALPHA_SUBSTORM * hac_substorm +
            self.config.ALPHA_IONOSPHERE * hac_ionosphere
        )
        
        hac_total = self._safe_normalization(hac_total)
        
        # NOWCAST + INÉRCIA: Calcular derivada robusta
        dHAC_dt = self._compute_robust_derivative(hac_total, times)
        
        # Detectar alertas de escalação
        escalation_flags = self._detect_escalation_triggers(
            hac_total, dHAC_dt, Bz, Vsw, times
        )
        
        self.results.update({
            'time': times,
            'HAC_total': hac_total,
            'HAC_ring': hac_ring,
            'HAC_substorm': hac_substorm,
            'HAC_ionosphere': hac_ionosphere,
            'Bz': Bz,
            'Vsw': Vsw,
            'coupling_signal': coupling,
            'dHAC_dt': dHAC_dt,
            'escalation_alert': escalation_flags,
            'nowcast_inertia_growth': self._compute_nowcast_growth(hac_total, coupling)
        })
        
        self._validate_output(hac_total)
        
        return hac_total
    
    def _safe_deltat(self, times):
        """Calcula delta-t com proteção"""
        n = len(times)
        dt = np.full(n, 60.0)
        
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
        max_val = np.nanmax(values) if len(values) > 0 else 1.0
        
        if max_val > 0:
            normalized = values / max_val * self.config.HAC_SCALE_MAX
        else:
            normalized = np.zeros_like(values)
        
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=self.config.HAC_SCALE_MAX, neginf=0.0)
        
        print(f"   • HAC máximo: {np.max(normalized):.1f}")
        print(f"   • HAC médio: {np.mean(normalized):.1f}")
        
        return normalized
    
    
    
    def _compute_nowcast_growth(self, hac_total, coupling):
        """Calcula crescimento pelo modelo Nowcast + Inércia"""
        # HAC_nowcast é proporcional ao coupling_signal normalizado
        hac_nowcast = (coupling / self.config.E_FIELD_SATURATION) * self.config.HAC_SCALE_MAX
        growth = (1.0 / self.config.TAU_EFFECTIVE) * (hac_nowcast - hac_total)
        return growth
    
    def _detect_escalation_triggers(self, hac_total, dHAC_dt, Bz, Vsw, times):
        """Detecta triggers de escalação usando regra de decisão"""
        print("   • Monitorando triggers de escalação...")
        
        n = len(hac_total)
        escalation_flags = np.zeros(n, dtype=bodef _compute_robust_derivative(self, hac_total, times):                              
    def _compute_robust_derivative(self, hac_total, times):
    """
    Derivada robusta HAC (100% compatível com numpy.datetime64)
    """

    print("   • Calculando dHAC/dt (Nowcast + Inércia)...")

    # Converter tempo para segundos
    times = np.array(times, dtype="datetime64[s]")
    dt = np.diff(times).astype("timedelta64[s]").astype(float)
    dt = np.insert(dt, 0, dt[0])
    dt[dt <= 0] = 1.0
    dt_hours = dt / 3600.0

    # Derivada
    if len(hac_total) < 7:
        dHAC_dt = np.gradient(hac_total) / dt_hours
    else:
        try:
            window = min(7, len(hac_total))
            if window % 2 == 0:
                window -= 1

            dHAC_dt = savgol_filter(
                hac_total,
                window_length=window,
                polyorder=2,
                deriv=1
            ) / np.median(dt_hours)

        except Exception as e:
            print(f"⚠️ Fallback derivada simples: {e}")
            dHAC_dt = np.gradient(hac_total) / dt_hours

    dHAC_dt = np.nan_to_num(dHAC_dt, nan=0.0)
    dHAC_dt = np.clip(dHAC_dt, -200, 200)

    print(f"     Derivada máxima: {np.max(dHAC_dt):.1f} nT/h")

    return dHAC_dt
        
        # Parâmetros críticos
        theta = self.config.THETA_CRITICAL
        h_g3 = self.config.HG3_THRESHOLD
        bz_crit = self.config.BZ_CRITICAL
        v_crit = self.config.VSW_CRITICAL
        
        # Janela temporal para condições sustentadas (15 minutos)
        window_size = 15  # pontos (1 minuto cada)
        
        for i in range(window_size, n):
            # Verificar condições simultâneas
            condition1 = hac_total[i] < h_g3  # HAC abaixo de G3
            condition2 = dHAC_dt[i] > theta   # Taxa de crescimento crítica
            
            # Condições de vento solar sustentadas (janela de 15 min)
            bz_window = Bz[max(0, i-window_size):i+1]
            v_window = Vsw[max(0, i-window_size):i+1]
            
            condition3 = np.mean(bz_window) < bz_crit  # Bz < -8 nT sustentado
            condition4 = np.mean(v_window) > v_crit    # V > 700 km/s sustentado
            
            if condition1 and condition2 and condition3 and condition4:
                escalation_flags[i] = True
                
                # Registrar alerta
                alert_info = {
                    'time': times[i],
                    'HAC': float(hac_total[i]),
                    'dHAC_dt': float(dHAC_dt[i]),
                    'Bz_avg': float(np.mean(bz_window)),
                    'V_avg': float(np.mean(v_window)),
                    'forecast_horizon_hours': self.config.TAU_EFFECTIVE * 2  # 2τ_eff para previsão
                }
                
                self.nowcast_alerts.append(alert_info)
                
                if not self.escalation_triggers or (times[i] - self.escalation_triggers[-1]['time']).total_seconds() > 3600:
                    self.escalation_triggers.append(alert_info)
                    print(f"     🚨 ALERTA DE ESCALAÇÃO em {times[i]}: "
                          f"HAC={hac_total[i]:.1f}, dH/dt={dHAC_dt[i]:.1f} nT/h")
        
        if np.any(escalation_flags):
            print(f"     ✅ {np.sum(escalation_flags)} alertas de escalação detectados")
        
        return escalation_flags
    
    def _validate_output(self, hac_values):
        """Validação rigorosa dos resultados"""
        nan_count = np.sum(np.isnan(hac_values))
        if nan_count > 0:
            print(f"❌ ERRO CRÍTICO: {nan_count} NaN em HAC")
            raise ValueError("NaN detectado em HAC")
        
        print("   ✅ Validação passada")
    
    def predict_storm_indicators(self, hac_values):
        """Predição robusta de indicadores de tempestade"""
        print("\n🌍 Predizendo indicadores...")
        
        kp_pred = self.config.KP_SCALE * np.tanh(
            hac_values / self.config.HAC_SCALE_MAX * 2
        )
        
        dst_pred = -self.config.RING_CURRENT_MAX * (
            hac_values / self.config.HAC_SCALE_MAX
        ) ** 1.3
        
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
        
        self.results.update({
            'Kp_pred': kp_pred,
            'Dst_pred': dst_pred,
            'Storm_level': storm_levels
        })
        
        g4g5_count = sum(1 for l in storm_levels if l in ['G4', 'G5'])
        print(f"   • Kp máximo: {np.max(kp_pred):.1f}")
        print(f"   • Dst mínimo: {np.min(dst_pred):.1f} nT")
        print(f"   • Eventos G4/G5: {g4g5_count}")
        
        return kp_pred, dst_pred, storm_levels
    
    def generate_nowcast_report(self):
        """Gera relatório específico do modelo Nowcast + Inércia"""
        if not self.nowcast_alerts:
            return "Nenhum alerta de escalação detectado."
        
        report = "="*70 + "\n"
        report += "🚨 RELATÓRIO NOWCAST + INÉRCIA (Escalação de Tempestades)\n"
        report += "="*70 + "\n\n"
        
        report += f"Total de alertas detectados: {len(self.nowcast_alerts)}\n"
        report += f"Triggers principais: {len(self.escalation_triggers)}\n\n"
        
        report += "PARÂMETROS CRÍTICOS:\n"
        report += f"  • τ_eff (tempo de resposta): {self.config.TAU_EFFECTIVE} horas\n"
        report += f"  • Θ (limiar crescimento): {self.config.THETA_CRITICAL} nT/h\n"
        report += f"  • H_G3 threshold: {self.config.HG3_THRESHOLD}\n"
        report += f"  • Bz crítico: < {self.config.BZ_CRITICAL} nT\n"
        report += f"  • V crítico: > {self.config.VSW_CRITICAL} km/s\n\n"
        
        if self.escalation_triggers:
            report += "ALERTAS PRINCIPAIS:\n"
            report += "-"*40 + "\n"
            for i, alert in enumerate(self.escalation_triggers, 1):
                report += f"{i}. {alert['time']}:\n"
                report += f"   HAC = {alert['HAC']:.1f} (abaixo de G3)\n"
                report += f"   dH/dt = {alert['dHAC_dt']:.1f} nT/h (acima de Θ)\n"
                report += f"   Bz médio = {alert['Bz_avg']:.1f} nT\n"
                report += f"   V médio = {alert['V_avg']:.1f} km/s\n"
                report += f"   Horizonte de previsão: {alert['forecast_horizon_hours']:.1f} horas\n\n"
        
        report += "EXPLICAÇÃO FÍSICA:\n"
        report += "O modelo Nowcast + Inércia detecta condições onde o reservatório\n"
        report += "magnetosférico está abaixo do limiar G3 mas com taxa de crescimento\n"
        report += "crítica, indicando carregamento rápido que pode evoluir para\n"
        report += "tempestades severas (G4/G5) em 2-6 horas.\n"
        report += "="*70
        
        return report

# ============================
# 4. VISUALIZAÇÃO COM NOWCAST + INÉRCIA
# ============================
class ProductionVisualizer:
    """Visualização profissional para produção com Nowcast + Inércia"""
    
    @staticmethod
    def create_final_dashboard(results, df, filename="hac_final_production.png"):
        """Cria dashboard final de produção com Nowcast + Inércia"""
        print(f"\n📈 Criando dashboard: {filename}")
        
        if len(results.get('HAC_total', [])) < 10:
            print("❌ Dados insuficientes")
            return None
        
        plt.style.use('default')
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle('HAC+ Model - Sistema Completo de Previsão (Nowcast + Inércia)', 
                    fontsize=14, fontweight='bold')
        
        times = results['time']
        
        # ===== PAINEL 1: HAC TOTAL =====
        ax1 = axes[0, 0]
        if 'HAC_total' in results:
            ax1.plot(times, results['HAC_total'], 
                    color='#d62728', linewidth=2, label='HAC Total', zorder=3)
            
            # Destacar alertas Nowcast
            if 'escalation_alert' in results:
                alert_mask = results['escalation_alert']
                if np.any(alert_mask):
                    alert_times = times[alert_mask]
                    alert_hac = results['HAC_total'][alert_mask]
                    ax1.scatter(alert_times, alert_hac, 
                              color='red', s=50, zorder=5,
                              label='Nowcast Alert')
        
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
        levels = [50, 100, 150, 200, 250]
        for lvl, col in zip(levels, colors):
            ax1.axhline(y=lvl, color=col, linestyle=':', alpha=0.5)
        
        ax1.set_ylabel('HAC Index', fontsize=10)
        ax1.set_title('A. Estado do Reservatório + Nowcast Alerts', fontsize=11)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 320)
        
        # ===== PAINEL 2: dH/dt (NOWCAST + INÉRCIA) =====
        ax2 = axes[0, 1]
        if 'dHAC_dt' in results:
            ax2.plot(times, results['dHAC_dt'], 
                    color='#3498db', linewidth=1.5, label='dH/dt (observado)', alpha=0.7)
            
            if 'nowcast_inertia_growth' in results:
                ax2.plot(times, results['nowcast_inertia_growth'],
                        color='#e74c3c', linewidth=1.5, linestyle='--',
                        label='Nowcast + Inércia', alpha=0.9)
            
            # Threshold crítico
            config = HACPhysicsConfig()
            ax2.axhline(y=config.THETA_CRITICAL, color='red', linestyle='--',
                       alpha=0.6, label=f'Θ={config.THETA_CRITICAL} nT/h')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_ylabel('dH/dt [nT/h]', fontsize=10)
        ax2.set_title('B. Taxa de Crescimento (Nowcast + Inércia)', fontsize=11)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # ===== PAINEL 3: Kp PREVISTO =====
        ax3 = axes[1, 0]
        if 'Kp_pred' in results:
            ax3.plot(times, results['Kp_pred'], 
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
            ax4.plot(times, results['Bz'], 
                    color='#2ecc71', linewidth=1, label='Bz')
            ax4.fill_between(times, 0, results['Bz'],
                            where=(results['Bz'] < 0),
                            color='red', alpha=0.3, label='IMF Sul')
            
            # Threshold crítico para Nowcast
            ax4.axhline(y=-8, color='orange', linestyle='--', alpha=0.7,
                       label='Bz crítico = -8 nT')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_ylabel('Bz [nT]', fontsize=10)
        ax4.set_title('D. Forçante do Vento Solar (Bz)', fontsize=11)
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # ===== PAINEL 5: ACOPLAMENTO =====
        ax5 = axes[2, 0]
        if 'coupling_signal' in results:
            ax5.plot(times, results['coupling_signal'],
                    color='#9b59b6', linewidth=1, label='Acoplamento')
        
        ax5.set_ylabel('Acoplamento [mV/m]', fontsize=10)
        ax5.set_title('E. Sinal de Acoplamento Efetivo', fontsize=11)
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # ===== PAINEL 6: VELOCIDADE =====
        ax6 = axes[2, 1]
        if 'Vsw' in results:
            ax6.plot(times, results['Vsw'],
                    color='#3498db', linewidth=1, label='Velocidade')
            
            # Threshold crítico para Nowcast
            ax6.axhline(y=700, color='orange', linestyle='--', alpha=0.7,
                       label='V crítico = 700 km/s')
        
        ax6.set_ylabel('V [km/s]', fontsize=10)
        ax6.set_title('F. Velocidade do Vento Solar', fontsize=11)
        ax6.legend(loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # ===== PAINEL 7: COMPONENTES HAC =====
        ax7 = axes[3, 0]
        if all(k in results for k in ['HAC_ring', 'HAC_substorm', 'HAC_ionosphere']):
            ax7.plot(times, results['HAC_ring'], label='Corrente de Anel', alpha=0.7)
            ax7.plot(times, results['HAC_substorm'], label='Subtempestades', alpha=0.7)
            ax7.plot(times, results['HAC_ionosphere'], label='Ionosfera', alpha=0.7)
        
        ax7.set_ylabel('Componentes HAC', fontsize=10)
        ax7.set_xlabel('Tempo (UTC)', fontsize=10)
        ax7.set_title('G. Particionamento de Energia', fontsize=11)
        ax7.legend(loc='upper left', fontsize=7)
        ax7.grid(True, alpha=0.3)
        
        # ===== PAINEL 8: REGRA DE DECISÃO =====
        ax8 = axes[3, 1]
        if 'HAC_total' in results and 'escalation_alert' in results:
            # Plotar condição composta
            condition_plot = np.zeros_like(results['HAC_total'])
            
            for i in range(len(condition_plot)):
                if results['HAC_total'][i] < 150:
                    condition_plot[i] = 1  # Abaixo de G3
                if results['dHAC_dt'][i] > 50:
                    condition_plot[i] = 2  # Crescimento crítico
            
            ax8.plot(times, condition_plot, 
                    color='#e67e22', linewidth=1.5, label='Regra de Decisão')
            
            ax8.axhline(y=1.5, color='red', linestyle='--', alpha=0.7,
                       label='Trigger Threshold')
            
            # Marcar alertas
            alert_mask = results['escalation_alert']
            if np.any(alert_mask):
                alert_y = [3] * np.sum(alert_mask)
                ax8.scatter(times[alert_mask], alert_y,
                          color='red', s=40, marker='^',
                          label='Escalation Watch', zorder=5)
            
            ax8.set_yticks([0, 1, 2, 3])
            ax8.set_yticklabels(['Normal', 'HAC<G3', 'dH/dt>Θ', 'ALERT'])
            ax8.set_ylabel('Estado', fontsize=10)
            ax8.set_xlabel('Tempo (UTC)', fontsize=10)
            ax8.set_title('H. Regra de Decisão Nowcast + Inércia', fontsize=11)
            ax8.legend(loc='upper left', fontsize=7)
            ax8.grid(True, alpha=0.3)
            ax8.set_ylim(-0.5, 3.5)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Dashboard salvo: {filename}")
        return fig

# ============================
# 5. RELATÓRIO FINAL COM NOWCAST
# ============================
class FinalReport:
    """Gera relatório final completo com Nowcast + Inércia"""
    
    @staticmethod
    def generate_report(results, df, model, filename="hac_final_report.txt"):
        """Relatório final do sistema com Nowcast + Inércia"""
        print("\n" + "="*70)
        print("📊 RELATÓRIO FINAL - SISTEMA HAC+ (NOWCAST + INÉRCIA)")
        print("="*70)
        
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO FINAL - SISTEMA HAC+ (NOWCAST + INÉRCIA)\n")
            f.write("="*70 + "\n\n")
            
            # 1. INFORMAÇÕES BÁSICAS
            f.write("1. INFORMAÇÕES DO DATASET\n")
            f.write("-"*40 + "\n")
            f.write(f"Período: {df['time_tag'].min()} a {df['time_tag'].max()}\n")
            f.write(f"Pontos totais: {len(df)}\n\n")
            
            # 2. PARÂMETROS NOWCAST + INÉRCIA
            f.write("2. PARÂMETROS NOWCAST + INÉRCIA\n")
            f.write("-"*40 + "\n")
            config = HACPhysicsConfig()
            f.write(f"τ_eff (tempo de resposta): {config.TAU_EFFECTIVE} horas\n")
            f.write(f"Θ (limiar crescimento crítico): {config.THETA_CRITICAL} nT/h\n")
            f.write(f"H_G3 threshold: {config.HG3_THRESHOLD}\n")
            f.write(f"Bz crítico: < {config.BZ_CRITICAL} nT\n")
            f.write(f"V crítico: > {config.VSW_CRITICAL} km/s\n\n")
            
            # 3. RESULTADOS HAC
            f.write("3. RESULTADOS DO MODELO HAC+\n")
            f.write("-"*40 + "\n")
            
            if 'HAC_total' in results:
                hac = results['HAC_total']
                f.write(f"HAC máximo: {np.max(hac):.2f}\n")
                f.write(f"HAC mínimo: {np.min(hac):.2f}\n")
                f.write(f"HAC médio:  {np.mean(hac):.2f}\n")
                
                if 'dHAC_dt' in results:
                    f.write(f"dH/dt máximo: {np.max(results['dHAC_dt']):.1f} nT/h\n")
                    f.write(f"dH/dt médio:  {np.mean(results['dHAC_dt']):.1f} nT/h\n\n")
                
                # Distribuição de níveis
                if 'Storm_level' in results:
                    levels = results['Storm_level']
                    total = len(levels)
                    f.write("Distribuição de níveis de tempestade:\n")
                    for lvl in ['Quiet', 'G1', 'G2', 'G3', 'G4', 'G5']:
                        count = sum(1 for x in levels if x == lvl)
                        pct = count/total*100 if total > 0 else 0
                        f.write(f"  {lvl:6s}: {count:4d} pontos ({pct:5.1f}%)\n")
            
            # 4. ALERTAS NOWCAST
            f.write("\n4. ALERTAS NOWCAST + INÉRCIA\n")
            f.write("-"*40 + "\n")
            
            if hasattr(model, 'escalation_triggers') and model.escalation_triggers:
                f.write(f"Total de triggers de escalação: {len(model.escalation_triggers)}\n\n")
                
                for i, alert in enumerate(model.escalation_triggers, 1):
                    f.write(f"{i}. {alert['time']}:\n")
                    f.write(f"   HAC = {alert['HAC']:.1f} (abaixo de G3)\n")
                    f.write(f"   dH/dt = {alert['dHAC_dt']:.1f} nT/h\n")
                    f.write(f"   Bz médio = {alert['Bz_avg']:.1f} nT\n")
                    f.write(f"   V médio = {alert['V_avg']:.1f} km/s\n")
                    f.write(f"   Horizonte de previsão: {alert['forecast_horizon_hours']:.1f} horas\n\n")
            else:
                f.write("Nenhum alerta de escalação detectado.\n\n")
            
            # 5. STATUS FINAL
            f.write("5. STATUS FINAL DO SISTEMA\n")
            f.write("-"*40 + "\n")
            
            if 'Storm_level' in results and len(results['Storm_level']) > 0:
                current_level = results['Storm_level'][-1]
                current_hac = results['HAC_total'][-1] if 'HAC_total' in results else 0
                
                f.write(f"Nível atual: {current_level}\n")
                f.write(f"HAC atual: {current_hac:.1f}\n")
                
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
                
                # Verificar se há alerta Nowcast ativo
                if 'escalation_alert' in results and results['escalation_alert'][-1]:
                    f.write("\n🚨 NOWCAST ALERT: ESCALAÇÃO IMINENTE!\n")
                    f.write("   Condições favoráveis para evolução para G4/G5\n")
                    f.write(f"   Horizonte: {config.TAU_EFFECTIVE * 2} horas\n")
            
            f.write(f"\nRelatório gerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n")
        
        print(f"📝 Relatório final salvo: {filename}")
        
        # Resumo no console
        if 'Storm_level' in results and len(results['Storm_level']) > 0:
            current_level = results['Storm_level'][-1]
            current_hac = results['HAC_total'][-1] if 'HAC_total' in results else 0
            
            print(f"\n🎯 STATUS ATUAL: HAC = {current_hac:.1f} → {current_level}")
            
            if 'escalation_alert' in results and results['escalation_alert'][-1]:
                print("   🚨 NOWCAST ALERT: Condições para escalação detectadas!")
                print(f"   • dH/dt = {results['dHAC_dt'][-1]:.1f} nT/h > Θ")
                print(f"   • Bz = {results['Bz'][-1]:.1f} nT < -8 nT")
                print(f"   • V = {results['Vsw'][-1]:.1f} km/s > 700 km/s")
            
            if current_level in ['G4', 'G5']:
                print("   🚨 ALERTA DE TEMPESTADE SEVERA")
            elif current_level == 'G3':
                print("   ⚠️  ALERTA DE TEMPESTADE FORTE")
        
        print("\n" + "="*70)

# ============================
# 6. PIPELINE PRINCIPAL
# ============================
def main():
    """Pipeline principal - PRODUÇÃO FINAL COM NOWCAST + INÉRCIA"""
    print("\n" + "="*70)
    print("🚀 HAC++ MODEL - SISTEMA DE PRODUÇÃO (NOWCAST + INÉRCIA)")
    print("="*70)
    
    # Configurar caminhos
    MAG_FILE = "data/mag-7-day.json"
    PLASMA_FILE = "data/plasma-7-day.json"
    
    # 1. CARREGAR E NORMALIZAR DADOS
    print("\n📥 CARREGANDO DADOS OMNI...")
    
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
    
    # 3. CAMPOS FÍSICOS
    print("\n⚡ CALCULANDO CAMPOS FÍSICOS...")
    calculator = PhysicalFieldsCalculator()
    df = calculator.compute_all_fields(df)
    
    # 4. MODELO HAC+ COM NOWCAST
    print("\n🧮 EXECUTANDO MODELO HAC+ (Nowcast + Inércia)...")
    model = ProductionHACModel()
    hac_values = model.compute_hac_system(df)
    
    # 5. PREDIÇÃO
    print("\n🌍 GERANDO PREDIÇÕES...")
    kp_pred, dst_pred, storm_levels = model.predict_storm_indicators(hac_values)
    
    # 6. RELATÓRIO NOWCAST
    print("\n🚨 GERANDO RELATÓRIO NOWCAST + INÉRCIA...")
    nowcast_report = model.generate_nowcast_report()
    print(nowcast_report)
    
    with open("nowcast_inertia_report.txt", "w") as f:
        f.write(nowcast_report)
    
    # 7. VISUALIZAÇÃO
    print("\n📈 CRIANDO VISUALIZAÇÕES...")
    visualizer = ProductionVisualizer()
    visualizer.create_final_dashboard(model.results, df, "hac_nowcast_final.png")
    
    # 8. RELATÓRIO FINAL
    print("\n📊 GERANDO RELATÓRIO FINAL...")
    reporter = FinalReport()
    reporter.generate_report(model.results, df, model)
    
    # 9. SALVAR RESULTADOS
    try:
        results_df = df.copy()
        for key, value in model.results.items():
            if key != 'time' and len(value) == len(results_df):
                results_df[key] = value
        
        output_file = "hac_nowcast_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Resultados salvos: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Erro ao salvar: {e}")
    
    # 10. STATUS FINAL
    print("\n" + "="*70)
    print("✅ SISTEMA HAC++ (NOWCAST + INÉRCIA) - EXECUÇÃO CONCLUÍDA")
    print("="*70)
    
    if 'Storm_level' in model.results and len(model.results['Storm_level']) > 0:
        current_level = model.results['Storm_level'][-1]
        current_hac = model.results['HAC_total'][-1]
        
        print(f"\n🔴 STATUS OPERACIONAL:")
        print(f"   HAC: {current_hac:.1f}")
        print(f"   Nível: {current_level}")
        
        if 'escalation_alert' in model.results and model.results['escalation_alert'][-1]:
            print("   🚨 NOWCAST ALERT: ESCALAÇÃO IMINENTE!")
        
        print(f"\n📁 ARQUIVOS GERADOS:")
        print(f"   1. hac_nowcast_final.png - Dashboard completo")
        print(f"   2. hac_nowcast_results.csv - Dados com Nowcast")
        print(f"   3. hac_final_report.txt - Relatório geral")
        print(f"   4. nowcast_inertia_report.txt - Relatório específico")
    
    print("\n" + "="*70)

# ============================
# EXECUÇÃO
# ============================
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    main()
