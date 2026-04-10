"""
HAC++ Model: Heliospheric Accumulated Coupling - PRODUÇÃO FINAL
COM NOWCAST + INÉRCIA (Previsão de Escalação Híbrida)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from scipy.signal import savgol_filter
from hac_core import HACCoreModel, HACCoreConfig
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
    TAU_RING_CURRENT = 2.0       # Tempo de decaimento da corrente do anel
    TAU_SUBSTORM = 1.0           # Tempo de injeção por subtempestades  
    TAU_IONOSPHERE = 0.3         # Tempo de resposta ionosférica
    TAU_EFFECTIVE = 2.0          # τ_eff para modelo Nowcast+Inércia
    
    #PARÂMETROS DE SATURAÇÃO FÍSICA
    E_FIELD_SATURATION = 25.0    # mV/m - Saturação OBSERVACIONAL
    KP_SATURATION = 8.0          # Saturação do índice Kp
    RING_CURRENT_MAX = 500.0     # nT - Saturação da corrente do anel
    
    # COEFICIENTES DE PARTICIONAMENTO (soma = 1,0)
    ALPHA_RING = 0.4            # Fração para corrente de anel
    ALPHA_SUBSTORM = 0.3        # Fração para subtempestades
    ALPHA_IONOSPHERE = 0.3      # Fração para ionosfera
    
    # PARÂMETROS NÃO LINEARES
    BETA_NONLINEAR = 1.5        # Expoente de resposta não linear
    COUPLING_THRESHOLD = 5.0    # mV/m - Limiar para não-linearidade
    
    # ESCALAS OPERACIONAIS
    HAC_SCALE_MAX = 500.0
    KP_SCALE = 9.0
    
    # LIMITES FÍSICOS
    VSW_MIN, VSW_MAX = 200, 1500 # km/s
    DENSITY_MIN, DENSITY_MAX = 0.1, 100 # cm⁻³0.1, 100  # cm⁻³
    BZ_MIN, BZ_MAX = -100, 100 # nT
    
    # NOWCAST + PARÂMETROS DE INÉRCIA# NOWCAST + INÉRCIA PARAMETERS
    THETA_CRITICAL = 50.0 # nT/h - Limiar de crescimento crítico50.0      # nT/h - Limiar de crescimento crítico
    HG3_THRESHOLD = 150.0 # Limiar G3150.0      # Limiar G3
    VSW_CRITICAL = 700.0       # km/s
    BZ_CRITICAL = -8.0         # nT
    
    # PARÂMETROS CLASSIFICAÇÃO HÍBRIDA
    DHDT_G5_THRESHOLD = 200.0   # nT/h para G5
    DHDT_G4_THRESHOLD = 150.0   # nT/h para G4
    DHDT_G3_THRESHOLD = 100.0   # nT/h para G3
    BZ_G5_THRESHOLD = -15.0     # nT para G5
    BZ_G4_THRESHOLD = -10.0     # nT para G4
    BZ_G3_THRESHOLD = -8.0      # nT para G3
    V_G5_THRESHOLD = 700.0      # km/s para G5
    V_G4_THRESHOLD = 650.0      # km/s para G4
    V_G3_THRESHOLD = 600.0      # km/s para G3

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
        self.core = HACCoreModel()
        self.results = {}
        self.nowcast_alerts = []
        self.escalation_triggers = []
        self.classification_logs = []
    
    def compute_hac_system(self, df):
        """Sistema HAC completo com integração ao modelo físico unificado (hac_core)"""

        print("\n⚡ Calculando sistema HAC+...")

        # ------------------------------------------------------------
        # 1. Cálculo tradicional do HAC (reservatórios)
        # ------------------------------------------------------------
        times = pd.to_datetime(df['time_tag']).values
        coupling = df['coupling_signal'].fillna(0).values
        Bz = df['bz_gsm'].fillna(0).values
        Vsw = df['speed'].fillna(400).values
        density = df['density'].fillna(5).values if 'density' in df.columns else None

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

        hac_total = hac_ring + hac_substorm + hac_ionosphere
        hac_total = self._safe_normalization(hac_total)

        # Derivada robusta
        dHAC_dt = self._compute_robust_derivative(hac_total, times)

        # Alertas
        escalation_flags = self._detect_escalation_triggers(hac_total, dHAC_dt, Bz, Vsw, times)
        nowcast_growth = self._compute_nowcast_growth(hac_total, coupling)

        # ------------------------------------------------------------
        # 2. Modelo físico (HAC CORE)
        # ------------------------------------------------------------
        self.core.config.HAC_REF = 1_250_000.0
        self.core.config.Q_FACTOR = -0.72

        core_results = self.core.process(
            time=times,
            bz=Bz,
            v=Vsw,
            density=density,
            mode='nowcast'
        )

        # ------------------------------------------------------------
        # 3. Armazenar resultados
        # ------------------------------------------------------------
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
            'nowcast_inertia_growth': nowcast_growth,

            # Core físico
            'Dst_physical': core_results['Dst_pred'],
            'Dst_min_physical': core_results['Dst_min'],
            'Dst_now': core_results['Dst_now'],
            'dDst_dt': core_results['dDst_dt'],
            'forecast': core_results['forecast'],
            'core_probabilities': core_results['probabilities'],
            'core_severity': core_results['severity']
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

            dt[0] = dt[1]

        return dt

    def _safe_normalization(self, values):
        """Normalização que NUNCA gera NaN"""
        max_val = np.nanmax(values) if len(values) > 0 else 1.0

        if max_val > 0:
            normalized = values / max_val * self.config.HAC_SCALE_MAX
        else:
            normalized = np.zeros_like(values)

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=self.config.HAC_SCALE_MAX,
            neginf=0.0
        )

        print(f"   • HAC máximo: {np.max(normalized):.1f}")
        print(f"   • HAC médio: {np.mean(normalized):.1f}")

        return normalized
    
    def _compute_robust_derivative(self, hac_total, times):
        """Calcula dHAC/dt usando Savitzky-Golay filter (suavizado)"""
        print("   • Calculando dHAC/dt (Nowcast + Inércia)...")

        # Converter tempo para segundos
        times = np.array(times, dtype="datetime64[s]")
        dt = np.diff(times).astype("timedelta64[s]").astype(float)
        dt = np.insert(dt, 0, dt[0])
        dt[dt <= 0] = 1.0
        dt_hours = np.maximum(dt / 3600.0, 1e-3)

        # Derivada
        if len(hac_total) < 7:
            # Para séries muito curtas, usa derivada simples
            dHAC_dt = np.gradient(hac_total) / dt_hours
        else:
            try:
                window = min(7, len(hac_total))
                if window % 2 == 0:
                    window -= 1

                dt_median = np.median(dt_hours)

                dHAC_dt = savgol_filter(
                  hac_total,
                  window_length=window,
                  polyorder=2,
                  deriv=1
              ) / dt_median

            except Exception as e:
                print(f"⚠️ Fallback derivada simples: {e}")
                dHAC_dt = np.gradient(hac_total) / dt_hours

        # Remove NaNs
        dHAC_dt = np.nan_to_num(dHAC_dt, nan=0.0)
        raw_dhdt = dHAC_dt.copy()

        # Log para debug
        if np.max(np.abs(raw_dhdt)) > 200:
            print(f"⚠️ Derivada bruta alta: max={np.max(raw_dhdt):.1f} nT/h")

        # Soft clipping (melhor que cortar seco)
        dHAC_dt = np.clip(raw_dhdt, -400, 400)

        print(f"     Derivada máxima: {np.max(dHAC_dt):.1f} nT/h")

        return dHAC_dt
    
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
        escalation_flags = np.zeros(n, dtype=bool)
        
        # Parâmetros críticos
        theta = self.config.THETA_CRITICAL
        h_g3 = self.config.HG3_THRESHOLD
        bz_crit = self.config.BZ_CRITICAL
        v_crit = self.config.VSW_CRITICAL
        
        # Janela temporal para condições sustentadas (15 minutos)
        window_size = 30  # pontos (1 minuto cada)
        
        for i in range(window_size, n):
            # Verificar condições simultâneas
            condition1 = hac_total[i] < h_g3  # HAC abaixo de G3
            condition2 = dHAC_dt[i] > theta   # Taxa de crescimento crítica
            
            # Condições de vento solar sustentadas (janela de 15 min)
            bz_window = Bz[max(0, i-window_size):i+1]
            v_window = Vsw[max(0, i-window_size):i+1]
            
            condition3 = np.median(bz_window) < bz_crit  # Bz < -8 nT sustentado
            condition4 = np.median(v_window) > v_crit    # V > 700 km/s sustentado
            
            if condition1 and condition2 and condition3 and condition4:
                escalation_flags[i] = True
                
                # Converter numpy.datetime64 para datetime para compatibilidade
                if isinstance(times[i], np.datetime64):
                    alert_time = pd.to_datetime(times[i])
                else:
                    alert_time = times[i]
                
                # Registrar alerta
                alert_info = {
                    'time': alert_time,
                    'HAC': float(hac_total[i]),
                    'dHAC_dt': float(dHAC_dt[i]),
                    'Bz_avg': float(np.mean(bz_window)),
                    'V_avg': float(np.mean(v_window)),
                    'forecast_horizon_hours': self.config.TAU_EFFECTIVE * 2  # 2τ_eff para previsão
                }
                
                self.nowcast_alerts.append(alert_info)
                
                if not self.escalation_triggers or (alert_time - self.escalation_triggers[-1]['time']).total_seconds() > 3600:
                    self.escalation_triggers.append(alert_info)
                    print(f"     🚨 ALERTA DE ESCALAÇÃO em {alert_time}: "
                          f"HAC={hac_total[i]:.1f}, dH/dt={dHAC_dt[i]:.1f} nT/h")
        
        if np.any(escalation_flags):
            print(f"     ✅ {np.sum(escalation_flags)} alertas de escalação detectados")
        
        return escalation_flags
    
    def _classify_storm_with_nowcast(self, hac, dhdt, bz, v):
        """Classificação híbrida: combina HAC estático com dinâmica Nowcast"""
        
        # CLASSIFICAÇÃO BASE (tradicional - baseada apenas no HAC)
        if hac < 50:
            base_level = "Quiet"
            base_severity = 0
        elif hac < 100:
            base_level = "G1"
            base_severity = 1
        elif hac < 150:
            base_level = "G2"
            base_severity = 2
        elif hac < 200:
            base_level = "G3"
            base_severity = 3
        elif hac < 250:
            base_level = "G4"
            base_severity = 4
        else:
            base_level = "G5"
            base_severity = 5
        
        # PESAGEM DINÂMICA: Calcula score Nowcast
        nowcast_score = 0
        
        # Componente taxa de crescimento (peso maior)
        if dhdt > 50: nowcast_score += 1
        if dhdt > 100: nowcast_score += 1
        if dhdt > 150: nowcast_score += 2
        if dhdt > 200: nowcast_score += 3
        
        # Componente Bz
        if bz < -5: nowcast_score += 1
        if bz < -10: nowcast_score += 2
        if bz < -15: nowcast_score += 3
        if bz < -20: nowcast_score += 4
        
        # Componente velocidade
        if v > 500: nowcast_score += 1
        if v > 600: nowcast_score += 1
        if v > 700: nowcast_score += 2
        if v > 800: nowcast_score += 3
        
        # Componente HAC (energia acumulada)
        if hac > 50: nowcast_score += 1
        if hac > 100: nowcast_score += 1
        if hac > 150: nowcast_score += 2
        if hac > 200: nowcast_score += 2
        
        # DECISÃO FINAL HÍBRIDA
        final_level = base_level
        final_severity = base_severity
        
        # Escalação baseada no score Nowcast
        if nowcast_score >= 12:
            # Condições extremas - forçar G5
            if base_severity < 5:
                final_level = "G5 (Nowcast Override)"
                final_severity = 5
        elif nowcast_score >= 9:
            # Condições muito fortes - forçar G4
            if base_severity < 4:
                final_level = "G4 (Nowcast Override)"
                final_severity = 4
        elif nowcast_score >= 7:
            # Condições fortes - forçar G3
            if base_severity < 3:
                final_level = "G3 (Nowcast Override)"
                final_severity = 3
        elif nowcast_score >= 5:
            # Condições moderadas - forçar G2
            if base_severity < 2:
                final_level = "G2 (Nowcast Enhancement)"
                final_severity = 2
        
        # REGRAS ESPECIAIS PARA CONDIÇÕES EXTREMAS
        # Mesmo com HAC baixo, se crescimento for extremo e condições favoráveis
        if dhdt > 220 and bz < -18 and v > 750 and hac > 100:
            if base_severity < 5:
                final_level = "G5 (Extreme Nowcast)"
                final_severity = 5
        elif dhdt > 150 and bz < -10 and v > 650:
            if base_severity < 4:
                final_level = "G4 (Strong Nowcast)"
                final_severity = 4
        elif dhdt > 100 and bz < -8 and v > 600 and hac > 50:
            if base_severity < 3:
                final_level = "G3 (Nowcast Trigger)"
                final_severity = 3
        
        # Log de decisão
        decision_info = {
            'hac': hac,
            'dhdt': dhdt,
            'bz': bz,
            'v': v,
            'base_level': base_level,
            'nowcast_score': nowcast_score,
            'final_level': final_level,
            'escalation': final_severity > base_severity,
            'severity': final_severity
        }
        
        return final_level, decision_info
    
    def _apply_trend_boost(self, storm_levels, hac_values, dHAC_dt):
        """Aplica boost adicional baseado em tendência de crescimento"""
        enhanced_levels = storm_levels.copy()
        n = len(storm_levels)
        
        # Janela de análise de tendência (1 hora)
        window = 60  # 60 pontos se dados são minuto a minuto
        
        for i in range(window, n):
            # Calcular tendência recente
            recent_hac = hac_values[i-window:i]
            recent_dhdt = dHAC_dt[i-window:i]
            
            # Critérios para boost
            mean_dhdt = np.mean(recent_dhdt)
            max_dhdt = np.max(recent_dhdt)
            hac_increase = hac_values[i] - hac_values[i-window]
            
            # Boost para G5 se crescimento extremo
            if max_dhdt > 220 and mean_dhdt > 70 and hac_increase > 120:
                current = storm_levels[i]
                if "G5" not in current:
                    enhanced_levels[i] = "G5 (Trend Boost)"
            
            # Boost para G4 se crescimento forte
            elif max_dhdt > 150 and mean_dhdt > 30 and hac_increase > 50:
                current = storm_levels[i]
                if "G4" not in current and "G5" not in current:
                    enhanced_levels[i] = "G4 (Trend Boost)"
        
        return enhanced_levels

    def apply_persistence_filter(self, levels, min_duration=30):
        filtered = levels.copy()
    
        current = levels[0]
        count = 1
    
        for i in range(1, len(levels)):
         if levels[i] == current:
            count += 1
         else:
              if count < min_duration:
                fill = filtered[i-count-1] if i-count-1 >= 0 else "G0"
                filtered[i-count:i] = [fill] * count
                current = levels[i]
                count = 1

        return filtered
    
    def _validate_output(self, hac_values):
        """Validação rigorosa dos resultados"""
        nan_count = np.sum(np.isnan(hac_values))
        if nan_count > 0:
            print(f"❌ ERRO CRÍTICO: {nan_count} NaN em HAC")
            raise ValueError("NaN detectado em HAC")
        
        # Verificar valores físicos
        if np.max(hac_values) > self.config.HAC_SCALE_MAX * 1.5:
            print(f"⚠️  AVISO: HAC excedeu escala ({np.max(hac_values):.1f})")
        
        print("   ✅ Validação passada")
    
    def predict_storm_indicators(self, hac_values):
        """Predição de indicadores utilizando Dst físico do core e classificação híbrida."""

        print("\n🌍 Predizendo indicadores (com Nowcast físico)...")

        kp_pred = 9 * np.tanh(hac_values / 180)

        dst_pred = self.results.get('Dst_physical', np.zeros_like(hac_values))
        dst_min = self.results.get('Dst_min_physical', np.min(dst_pred))
        dst_now = self.results.get('Dst_now', dst_pred[-1] if len(dst_pred) > 0 else 0)

        storm_levels = []
        decision_logs = []

        dHAC_dt = self.results.get('dHAC_dt', np.zeros_like(hac_values))
        Bz = self.results.get('Bz', np.zeros_like(hac_values))
        Vsw = self.results.get('Vsw', np.full_like(hac_values, 400))

        escalation_count = 0
        g4g5_nowcast_count = 0

        core_severity = self.results.get('core_severity', 0)

        for i in range(len(hac_values)):
            level, decision_info = self._classify_storm_with_nowcast(
                hac_values[i], dHAC_dt[i], Bz[i], Vsw[i]
        )

            if core_severity > 0:
                severity_map = {0: 'Quiet', 1: 'G1', 2: 'G2', 3: 'G3', 4: 'G4', 5: 'G5'}
                level = severity_map.get(core_severity, level)

            storm_levels.append(level)
            decision_logs.append(decision_info)

            if decision_info['escalation']:
            escalation_count += 1
            if "G4" in level or "G5" in level:
            g4g5_nowcast_count += 1

        enhanced_levels = self._apply_trend_boost(storm_levels, hac_values, dHAC_dt)

        self.results.update({
            'Kp_pred': kp_pred,
            'Dst_pred': dst_pred,
            'Dst_min': dst_min,
            'Dst_now': dst_now,
            'Storm_level': enhanced_levels,
            'Storm_level_base': storm_levels,
            'Decision_logs': decision_logs
    })

        self.classification_logs = decision_logs

        g4g5_final_count = sum(1 for l in enhanced_levels if "G4" in l or "G5" in l)
        g4g5_base_count = sum(1 for l in storm_levels if "G4" in l or "G5" in l)
        g4g5_traditional = sum(1 for l in storm_levels if l in ['G4', 'G5'])

        print(f"   • Kp máximo: {np.max(kp_pred):.1f}")
        print(f"   • Dst mínimo (físico): {dst_min:.1f} nT")
        print(f"   • Dst atual: {dst_now:.1f} nT")
        print(f"   • Eventos G4/G5 (tradicional): {g4g5_traditional}")
        print(f"   • Eventos G4/G5 (Nowcast base): {g4g5_base_count}")
        print(f"   • Eventos G4/G5 (com boost): {g4g5_final_count}")
        print(f"   • Escalações Nowcast: {escalation_count}")

        forecast = self.results.get('forecast', {})
        if forecast:
            print("   • Previsão Dst:")
            for h, val in forecast.items():
                print(f"       {h}: {val:.1f} nT")

        probs = self.results.get('core_probabilities', {})
        if probs:
            print("   • Probabilidades (softmax):")
            for k, v in probs.items():
                print(f"       {k}: {v*100:.1f}%")

        return kp_pred, dst_pred, enhanced_levels

    def generate_nowcast_report(self):
        """Gera relatório específico do modelo Nowcast + Inércia"""

        if not self.nowcast_alerts:
            nowcast_summary = "Nenhum alerta de escalação detectado."
        else:
            nowcast_summary = f"Total de alertas detectados: {len(self.nowcast_alerts)}\n"
            nowcast_summary += f"Triggers principais: {len(self.escalation_triggers)}\n\n"

        # Análise de classificação Nowcast
        if self.classification_logs:
            nowcast_escalations = sum(1 for log in self.classification_logs if log['escalation'])
            nowcast_g4g5 = sum(1 for log in self.classification_logs if log['severity'] >= 4)

            nowcast_summary += f"CLASSIFICAÇÃO NOWCAST:\n"
            nowcast_summary += f"• Escalações Nowcast: {nowcast_escalations}\n"
            nowcast_summary += f"• Eventos G4/G5 Nowcast: {nowcast_g4g5}\n"

            recent_escalations = [log for log in self.classification_logs[-10:] if log['escalation']]
            if recent_escalations:
                nowcast_summary += "\nÚLTIMAS ESCALAÇÕES:\n"
                for log in recent_escalations[-3:]:
                    nowcast_summary += (
                        f"• HAC={log['hac']:.1f}, dH/dt={log['dhdt']:.1f} nT/h, "
                        f"Bz={log['bz']:.1f} nT: {log['base_level']} → {log['final_level']}\n"
                    )

        report = "=" * 70 + "\n"
        report += "🚨 RELATÓRIO NOWCAST + INÉRCIA (Escalação de Tempestades)\n"
        report += "=" * 70 + "\n\n"

        report += nowcast_summary + "\n"

        report += "PARÂMETROS CRÍTICOS:\n"
        report += f"  • τ_eff (tempo de resposta): {self.config.TAU_EFFECTIVE} horas\n"
        report += f"  • Θ (limiar crescimento): {self.config.THETA_CRITICAL} nT/h\n"
        report += f"  • H_G3 threshold: {self.config.HG3_THRESHOLD}\n"
        report += f"  • Bz crítico: < {self.config.BZ_CRITICAL} nT\n"
        report += f"  • V crítico: > {self.config.VSW_CRITICAL} km/s\n\n"

        if self.escalation_triggers:
            report += "ALERTAS PRINCIPAIS:\n"
            report += "-" * 40 + "\n"
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
        report += "=" * 70

        return report
    
    def get_current_assessment(self):
        """Retorna avaliação detalhada do momento atual"""
        if not self.results or 'Storm_level' not in self.results:
            return None
        
        idx = -1  # Último ponto
        assessment = {
            'time': self.results['time'][idx],
            'HAC': float(self.results['HAC_total'][idx]),
            'dHAC_dt': float(self.results['dHAC_dt'][idx]),
            'Bz': float(self.results['Bz'][idx]),
            'Vsw': float(self.results['Vsw'][idx]),
            'classification': self.results['Storm_level'][idx],
            'base_classification': self.results['Storm_level_base'][idx] if 'Storm_level_base' in self.results else self.results['Storm_level'][idx],
            'coupling': float(self.results['coupling_signal'][idx]),
            'Kp_pred': float(self.results['Kp_pred'][idx]),
            'Dst_pred': float(self.results['Dst_pred'][idx]),
            'escalation_risk': 'HIGH' if self.results['escalation_alert'][idx] else 'LOW'
        }
        
        # Adicionar explicação
        if "Nowcast" in assessment['classification'] or "Boost" in assessment['classification']:
            assessment['explanation'] = "Classificação elevada devido à dinâmica rápida de crescimento"
        elif assessment['escalation_risk'] == 'HIGH':
            assessment['explanation'] = "Condições favoráveis para escalação iminente"
        else:
            assessment['explanation'] = "Classificação baseada no estado atual do reservatório"
        
        return assessment

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
        fig, axes = plt.subplots(4, 2, figsize=(16, 14))
        fig.suptitle('HAC++ Model - Sistema Completo de Previsão (Nowcast + Inércia Híbrido)', 
                    fontsize=16, fontweight='bold')
        
        times = results['time']
        
        # ===== PAINEL 1: HAC TOTAL COM CLASSIFICAÇÃO =====
        ax1 = axes[0, 0]
        if 'HAC_total' in results:
            # Plotar HAC com cores baseadas na classificação
            hac = results['HAC_total']
            
            # Definir cores baseadas na classificação final
            colors = []
            if 'Storm_level' in results:
                for level in results['Storm_level']:
                    if "G5" in level:
                        colors.append('#8B0000')  # Vermelho escuro
                    elif "G4" in level:
                        colors.append('#FF4500')  # Laranja vermelho
                    elif "G3" in level:
                        colors.append('#FF8C00')  # Laranja escuro
                    elif "G2" in level:
                        colors.append('#FFD700')  # Amarelo ouro
                    elif "G1" in level:
                        colors.append('#ADFF2F')  # Verde amarelado
                    elif "Nowcast" in level:
                        colors.append('#9370DB')  # Roxo médio
                    else:
                        colors.append('#1E90FF')  # Azul dodger
            else:
                colors = ['#d62728'] * len(hac)
            
            # Plotar com cores variadas
            for i in range(len(hac)-1):
                ax1.plot(times[i:i+2], hac[i:i+2], 
                        color=colors[i], linewidth=2, alpha=0.8)
            
            # Destacar alertas Nowcast
            if 'escalation_alert' in results:
                alert_mask = results['escalation_alert']
                if np.any(alert_mask):
                    alert_times = times[alert_mask]
                    alert_hac = results['HAC_total'][alert_mask]
                    ax1.scatter(alert_times, alert_hac, 
                              color='red', s=60, zorder=5,
                              label='Nowcast Alert', marker='^')
        
        # Thresholds NOAA
        colors_thresh = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
        levels = [50, 100, 150, 200, 250]
        labels = ['G1', 'G2', 'G3', 'G4', 'G5']
        for lvl, col, lbl in zip(levels, colors_thresh, labels):
            ax1.axhline(y=lvl, color=col, linestyle=':', alpha=0.7, 
                       label=f'{lbl} ({lvl})')
        
        ax1.set_ylabel('HAC Index', fontsize=11)
        ax1.set_title('A. Estado do Reservatório + Classificação Nowcast', fontsize=12)
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
            
            # Thresholds de classificação
            config = HACPhysicsConfig()
            ax2.axhline(y=config.THETA_CRITICAL, color='orange', linestyle='--',
                       alpha=0.8, label=f'Θ={config.THETA_CRITICAL} nT/h (G3)')
            ax2.axhline(y=config.DHDT_G4_THRESHOLD, color='red', linestyle='--',
                       alpha=0.8, label=f'G4={config.DHDT_G4_THRESHOLD} nT/h')
            ax2.axhline(y=config.DHDT_G5_THRESHOLD, color='darkred', linestyle='--',
                       alpha=0.8, label=f'G5={config.DHDT_G5_THRESHOLD} nT/h')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_ylabel('dH/dt [nT/h]', fontsize=11)
        ax2.set_title('B. Taxa de Crescimento (Nowcast + Inércia)', fontsize=12)
        ax2.legend(loc='upper left', fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        # ===== PAINEL 3: CLASSIFICAÇÃO HÍBRIDA =====
        ax3 = axes[1, 0]
        if 'Storm_level' in results and 'Storm_level_base' in results:
            # Converter classificações para valores numéricos
            severity_map = {
                'Quiet': 0, 'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4, 'G5': 5,
                'G2 (Nowcast Enhancement)': 2.5,
                'G3 (Nowcast Trigger)': 3.3, 'G3 (Nowcast Override)': 3.7,
                'G4 (Strong Nowcast)': 4.3, 'G4 (Nowcast Override)': 4.7,
                'G4 (Trend Boost)': 4.5,
                'G5 (Extreme Nowcast)': 5.3, 'G5 (Nowcast Override)': 5.7,
                'G5 (Trend Boost)': 5.5
            }
            
            base_severity = np.array([severity_map.get(l, 0) for l in results['Storm_level_base']])
            final_severity = np.array([severity_map.get(l, 0) for l in results['Storm_level']])
            
            ax3.plot(times, base_severity, 
                    color='#7f8c8d', linewidth=1, linestyle='--',
                    label='Classificação Base', alpha=0.6)
            ax3.plot(times, final_severity,
                    color='#9b59b6', linewidth=2,
                    label='Classificação Híbrida', alpha=0.9)
            
            # Sombrear áreas de escalação
            escalation_mask = final_severity > base_severity
            if np.any(escalation_mask):
                ax3.fill_between(times, 0, final_severity,
                               where=escalation_mask,
                               color='#ff9999', alpha=0.4,
                               label='Escalação Nowcast')
        
        ax3.set_yticks([0, 1, 2, 3, 4, 5])
        ax3.set_yticklabels(['Quiet', 'G1', 'G2', 'G3', 'G4', 'G5'])
        ax3.set_ylabel('Classificação', fontsize=11)
        ax3.set_title('C. Classificação Híbrida (Base vs Nowcast)', fontsize=12)
        ax3.legend(loc='upper left', fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.5, 5.5)
        
        # ===== PAINEL 4: Bz COM THRESHOLDS =====
        ax4 = axes[1, 1]
        if 'Bz' in results:
            ax4.plot(times, results['Bz'], 
                    color='#2ecc71', linewidth=1, label='Bz')
            ax4.fill_between(times, 0, results['Bz'],
                            where=(results['Bz'] < 0),
                            color='red', alpha=0.3, label='IMF Sul')
            
            # Thresholds de classificação
            config = HACPhysicsConfig()
            ax4.axhline(y=-5, color='yellow', linestyle=':', alpha=0.7)
            ax4.axhline(y=config.BZ_G3_THRESHOLD, color='orange', linestyle='--',
                       alpha=0.8, label=f'G3={config.BZ_G3_THRESHOLD} nT')
            ax4.axhline(y=config.BZ_G4_THRESHOLD, color='red', linestyle='--',
                       alpha=0.8, label=f'G4={config.BZ_G4_THRESHOLD} nT')
            ax4.axhline(y=config.BZ_G5_THRESHOLD, color='darkred', linestyle='--',
                       alpha=0.8, label=f'G5={config.BZ_G5_THRESHOLD} nT')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_ylabel('Bz [nT]', fontsize=11)
        ax4.set_title('D. Forçante do Vento Solar (Bz)', fontsize=12)
        ax4.legend(loc='upper right', fontsize=7)
        ax4.grid(True, alpha=0.3)
        
        # ===== PAINEL 5: Kp PREVISTO =====
        ax5 = axes[2, 0]
        if 'Kp_pred' in results:
            ax5.plot(times, results['Kp_pred'], 
                    color='#e74c3c', linewidth=1.5, label='Kp previsto')
            ax5.axhline(y=5, color='orange', linestyle=':', alpha=0.7, label='Kp=5 (G1)')
            ax5.axhline(y=6, color='yellow', linestyle=':', alpha=0.7, label='Kp=6 (G2)')
            ax5.axhline(y=7, color='red', linestyle=':', alpha=0.7, label='Kp=7 (G3)')
            ax5.axhline(y=8, color='darkred', linestyle='--', alpha=0.8, label='Kp=8 (G4)')
            ax5.axhline(y=9, color='purple', linestyle='--', alpha=0.8, label='Kp=9 (G5)')
        
        ax5.set_ylabel('Índice Kp', fontsize=11)
        ax5.set_title('E. Atividade Geomagnética Prevista', fontsize=12)
        ax5.legend(loc='upper left', fontsize=7)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 9.5)
        
        # ===== PAINEL 6: VELOCIDADE =====
        ax6 = axes[2, 1]
        if 'Vsw' in results:
            ax6.plot(times, results['Vsw'],
                    color='#3498db', linewidth=1, label='Velocidade')
            
            # Thresholds de classificação
            config = HACPhysicsConfig()
            ax6.axhline(y=500, color='lightblue', linestyle=':', alpha=0.7)
            ax6.axhline(y=config.V_G3_THRESHOLD, color='orange', linestyle='--',
                       alpha=0.8, label=f'G3={config.V_G3_THRESHOLD} km/s')
            ax6.axhline(y=config.V_G4_THRESHOLD, color='red', linestyle='--',
                       alpha=0.8, label=f'G4={config.V_G4_THRESHOLD} km/s')
            ax6.axhline(y=config.V_G5_THRESHOLD, color='darkred', linestyle='--',
                       alpha=0.8, label=f'G5={config.V_G5_THRESHOLD} km/s')
        
        ax6.set_ylabel('V [km/s]', fontsize=11)
        ax6.set_title('F. Velocidade do Vento Solar', fontsize=12)
        ax6.legend(loc='upper left', fontsize=7)
        ax6.grid(True, alpha=0.3)
        
        # ===== PAINEL 7: ACOPLAMENTO =====
        ax7 = axes[3, 0]
        if 'coupling_signal' in results:
            ax7.plot(times, results['coupling_signal'],
                    color='#9b59b6', linewidth=1, label='Acoplamento')
        
        ax7.set_ylabel('Acoplamento [mV/m]', fontsize=11)
        ax7.set_xlabel('Tempo (UTC)', fontsize=11)
        ax7.set_title('G. Sinal de Acoplamento Efetivo', fontsize=12)
        ax7.legend(loc='upper left')
        ax7.grid(True, alpha=0.3)
        
        # ===== PAINEL 8: Dst PREVISTO =====
        ax8 = axes[3, 1]
        if 'Dst_pred' in results:
            ax8.plot(times, results['Dst_pred'],
                    color='#e67e22', linewidth=1.5, label='Dst previsto')
            
            # Thresholds Dst
            ax8.axhline(y=-50, color='lightgreen', linestyle=':', alpha=0.7)
            ax8.axhline(y=-100, color='orange', linestyle=':', alpha=0.7)
            ax8.axhline(y=-200, color='red', linestyle=':', alpha=0.7)
            ax8.axhline(y=-350, color='darkred', linestyle=':', alpha=0.7)
        
        ax8.set_ylabel('Dst [nT]', fontsize=11)
        ax8.set_xlabel('Tempo (UTC)', fontsize=11)
        ax8.set_title('H. Dst Equivalente Previsto', fontsize=12)
        ax8.legend(loc='lower left')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(-550, 50)
        
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
        print("📊 RELATÓRIO FINAL - SISTEMA HAC+ (NOWCAST + INÉRCIA HÍBRIDO)")
        print("="*70)
        
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO FINAL - SISTEMA HAC+ (NOWCAST + INÉRCIA HÍBRIDO)\n")
            f.write("="*80 + "\n\n")
            
            # 1. INFORMAÇÕES DO DATASET
            f.write("1. INFORMAÇÕES DO DATASET\n")
            f.write("-"*50 + "\n")
            f.write(f"Período: {df['time_tag'].min()} a {df['time_tag'].max()}\n")
            f.write(f"Pontos totais: {len(df)}\n")
            f.write(f"Duração: {(df['time_tag'].max() - df['time_tag'].min()).total_seconds()/3600:.1f} horas\n\n")
            
            # 2. PARÂMETROS NOWCAST + INÉRCIA
            f.write("2. PARÂMETROS NOWCAST + INÉRCIA\n")
            f.write("-"*50 + "\n")
            config = HACPhysicsConfig()
            f.write(f"τ_eff (tempo de resposta): {config.TAU_EFFECTIVE} horas\n")
            f.write(f"Θ (limiar crescimento crítico): {config.THETA_CRITICAL} nT/h\n")
            f.write(f"H_G3 threshold: {config.HG3_THRESHOLD}\n")
            f.write(f"Bz crítico: < {config.BZ_CRITICAL} nT\n")
            f.write(f"V crítico: > {config.VSW_CRITICAL} km/s\n\n")
            
            # 3. RESULTADOS HAC
            f.write("3. RESULTADOS DO MODELO HAC+\n")
            f.write("-"*50 + "\n")
            
            if 'HAC_total' in results:
                hac = results['HAC_total']
                f.write(f"HAC máximo: {np.max(hac):.2f}\n")
                f.write(f"HAC mínimo: {np.min(hac):.2f}\n")
                f.write(f"HAC médio:  {np.mean(hac):.2f}\n")
                
                if 'dHAC_dt' in results:
                    f.write(f"dH/dt máximo: {np.max(results['dHAC_dt']):.1f} nT/h\n")
                    f.write(f"dH/dt médio:  {np.mean(results['dHAC_dt']):.1f} nT/h\n")
                    f.write(f"dH/dt > 100 nT/h: {np.sum(results['dHAC_dt'] > 100)} pontos\n")
                    f.write(f"dH/dt > 150 nT/h: {np.sum(results['dHAC_dt'] > 150)} pontos\n\n")
                
                # Distribuição de níveis TRADICIONAL
                if 'Storm_level_base' in results:
                    levels_base = results['Storm_level_base']
                    total = len(levels_base)
                    
                    # Contagem tradicional (base)
                    counts_base = {
                        'Quiet': sum(1 for x in levels_base if 'Quiet' in x),
                        'G1': sum(1 for x in levels_base if 'G1' in x and 'G2' not in x),
                        'G2': sum(1 for x in levels_base if 'G2' in x and 'G3' not in x),
                        'G3': sum(1 for x in levels_base if 'G3' in x and 'G4' not in x),
                        'G4': sum(1 for x in levels_base if 'G4' in x and 'G5' not in x),
                        'G5': sum(1 for x in levels_base if 'G5' in x)
                    }
                    
                    # Contagem híbrida (final)
                    if 'Storm_level' in results:
                        levels_final = results['Storm_level']
                        counts_final = {
                            'Quiet': sum(1 for x in levels_final if 'Quiet' in x),
                            'G1': sum(1 for x in levels_final if 'G1' in x and 'G2' not in x),
                            'G2': sum(1 for x in levels_final if 'G2' in x and 'G3' not in x),
                            'G3': sum(1 for x in levels_final if 'G3' in x and 'G4' not in x),
                            'G4': sum(1 for x in levels_final if 'G4' in x and 'G5' not in x),
                            'G5': sum(1 for x in levels_final if 'G5' in x),
                            'Nowcast Override': sum(1 for x in levels_final if 'Nowcast' in x or 'Boost' in x)
                        }
                    
                    f.write("Distribuição de níveis de tempestade (BASE):\n")
                    for lvl in ['Quiet', 'G1', 'G2', 'G3', 'G4', 'G5']:
                        count = counts_base[lvl]
                        pct = count/total*100 if total > 0 else 0
                        f.write(f"  {lvl:6s}: {count:5d} pontos ({pct:5.1f}%)\n")
                    
                    f.write("\nDistribuição de níveis (HÍBRIDO):\n")
                    for lvl in ['Quiet', 'G1', 'G2', 'G3', 'G4', 'G5']:
                        count = counts_final[lvl]
                        pct = count/total*100 if total > 0 else 0
                        f.write(f"  {lvl:6s}: {count:5d} pontos ({pct:5.1f}%)\n")
                    
                    f.write(f"\n  Escalações Nowcast: {counts_final['Nowcast Override']} pontos\n")
            
            # 4. ALERTAS NOWCAST
            f.write("\n4. ALERTAS NOWCAST + INÉRCIA\n")
            f.write("-"*50 + "\n")
            
            if hasattr(model, 'escalation_triggers') and model.escalation_triggers:
                f.write(f"Total de triggers de escalação: {len(model.escalation_triggers)}\n")
                
                # Análise de classificação Nowcast
                if hasattr(model, 'classification_logs') and model.classification_logs:
                    nowcast_escalations = sum(1 for log in model.classification_logs if log['escalation'])
                    nowcast_g4g5 = sum(1 for log in model.classification_logs if log['severity'] >= 4)
                    f.write(f"Escalações Nowcast detectadas: {nowcast_escalations}\n")
                    f.write(f"Eventos G4/G5 Nowcast: {nowcast_g4g5}\n\n")
                
                f.write("\nALERTAS PRINCIPAIS:\n")
                f.write("-"*40 + "\n")
                for i, alert in enumerate(model.escalation_triggers, 1):
                    f.write(f"{i}. {alert['time']}:\n")
                    f.write(f"   HAC = {alert['HAC']:.1f} (abaixo de G3)\n")
                    f.write(f"   dH/dt = {alert['dHAC_dt']:.1f} nT/h\n")
                    f.write(f"   Bz médio = {alert['Bz_avg']:.1f} nT\n")
                    f.write(f"   V médio = {alert['V_avg']:.1f} km/s\n")
                    f.write(f"   Horizonte de previsão: {alert['forecast_horizon_hours']:.1f} horas\n\n")
            else:
                f.write("Nenhum alerta de escalação detectado.\n\n")
            
            # 5. STATUS FINAL COM ANÁLISE HÍBRIDA
            f.write("5. STATUS FINAL DO SISTEMA\n")
            f.write("-"*50 + "\n")
            
            if 'Storm_level' in results and len(results['Storm_level']) > 0:
                current_level = results['Storm_level'][-1]
                current_level_base = results['Storm_level_base'][-1] if 'Storm_level_base' in results else current_level
                current_hac = results['HAC_total'][-1] if 'HAC_total' in results else 0
                current_dhdt = results['dHAC_dt'][-1] if 'dHAC_dt' in results else 0
                current_bz = results['Bz'][-1] if 'Bz' in results else 0
                current_v = results['Vsw'][-1] if 'Vsw' in results else 0
                
                f.write(f"ESTADO ATUAL:\n")
                f.write(f"  • HAC: {current_hac:.1f}\n")
                f.write(f"  • dH/dt: {current_dhdt:.1f} nT/h\n")
                f.write(f"  • Bz: {current_bz:.1f} nT\n")
                f.write(f"  • V: {current_v:.1f} km/s\n")
                f.write(f"  • Classificação Base: {current_level_base}\n")
                f.write(f"  • Classificação Final: {current_level}\n")
                
                if current_level in ['G4', 'G5', 'G4 (Nowcast Override)', 'G5 (Nowcast Override)', 
                                   'G4 (Strong Nowcast)', 'G5 (Extreme Nowcast)',
                                   'G4 (Trend Boost)', 'G5 (Trend Boost)']:
                    f.write("\n🚨 ALERTA: Condições de tempestade severa\n")
                elif "G3" in current_level:
                    f.write("\n⚠️  ALERTA: Tempestade forte\n")
                elif "G2" in current_level:
                    f.write("\n📢 ATENÇÃO: Tempestade moderada\n")
                elif "G1" in current_level:
                    f.write("\n📋 MONITORAMENTO: Tempestade menor\n")
                else:
                    f.write("\n✅ Condições quietas\n")
                
                # Verificar se há alerta Nowcast ativo
                if 'escalation_alert' in results and results['escalation_alert'][-1]:
                    f.write("\n🚨 NOWCAST ALERT: ESCALAÇÃO IMINENTE!\n")
                    f.write("   Condições favoráveis para evolução para G4/G5\n")
                    f.write(f"   Horizonte: {config.TAU_EFFECTIVE * 2} horas\n")
                
                # Análise de risco
                f.write("\nANÁLISE DE RISCO:\n")
                risk_factors = []
                if current_dhdt > 100: risk_factors.append(f"Alta taxa de crescimento ({current_dhdt:.1f} nT/h)")
                if current_bz < -10: risk_factors.append(f"Bz fortemente negativo ({current_bz:.1f} nT)")
                if current_v > 700: risk_factors.append(f"Vento solar rápido ({current_v:.1f} km/s)")
                if current_hac > 150: risk_factors.append(f"Energia acumulada elevada (HAC={current_hac:.1f})")
                
                if risk_factors:
                    f.write("   Fatores de risco ativos:\n")
                    for factor in risk_factors:
                        f.write(f"   • {factor}\n")
                else:
                    f.write("   Nenhum fator de risco crítico ativo\n")
            
            f.write(f"\nRelatório gerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
        print(f"📝 Relatório final salvo: {filename}")
        
        # Resumo no console
        if 'Storm_level' in results and len(results['Storm_level']) > 0:
            current_level = results['Storm_level'][-1]
            current_hac = results['HAC_total'][-1] if 'HAC_total' in results else 0
            current_dhdt = results['dHAC_dt'][-1] if 'dHAC_dt' in results else 0
            
            print(f"\n🎯 STATUS ATUAL: HAC = {current_hac:.1f}, dH/dt = {current_dhdt:.1f} nT/h → {current_level}")
            
            if 'escalation_alert' in results and results['escalation_alert'][-1]:
                print("   🚨 NOWCAST ALERT: Condições para escalação detectadas!")
                print(f"   • dH/dt = {current_dhdt:.1f} nT/h > Θ")
                print(f"   • Bz = {results['Bz'][-1]:.1f} nT < -8 nT")
                print(f"   • V = {results['Vsw'][-1]:.1f} km/s > 700 km/s")
            
            if "G5" in current_level:
                print("   🚨🚨 ALERTA DE TEMPESTADE G5 SEVERA")
            elif "G4" in current_level:
                print("   🚨 ALERTA DE TEMPESTADE G4 SEVERA")
            elif "G3" in current_level:
                print("   ⚠️  ALERTA DE TEMPESTADE G3 FORTE")
            elif "G2" in current_level:
                print("   📢 ATENÇÃO: TEMPESTADE G2 MODERADA")
        
        print("\n" + "="*70)

# ============================
# 6. PIPELINE PRINCIPAL
# ============================
def main():
    """Pipeline principal - PRODUÇÃO FINAL COM NOWCAST + INÉRCIA"""
    print("\n" + "="*70)
    print("🚀 HAC++ MODEL - SISTEMA DE PRODUÇÃO (NOWCAST + INÉRCIA HÍBRIDO)")
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
    print("\n🧮 EXECUTANDO MODELO HAC+ (Nowcast + Inércia Híbrido)...")
    model = ProductionHACModel()
    hac_values = model.compute_hac_system(df)
    
    # 5. PREDIÇÃO COM CLASSIFICAÇÃO HÍBRIDA
    print("\n🌍 GERANDO PREDIÇÕES (Classificação Híbrida)...")
    kp_pred, dst_pred, storm_levels = model.predict_storm_indicators(hac_values)
    
    # 6. OBTER AVALIAÇÃO ATUAL
    print("\n🔍 AVALIAÇÃO DO ESTADO ATUAL:")
    current_assessment = model.get_current_assessment()
    if current_assessment:
        print(f"   • Hora: {current_assessment['time']}")
        print(f"   • HAC: {current_assessment['HAC']:.1f}")
        print(f"   • dH/dt: {current_assessment['dHAC_dt']:.1f} nT/h")
        print(f"   • Bz: {current_assessment['Bz']:.1f} nT")
        print(f"   • Classificação: {current_assessment['classification']}")
        print(f"   • Risco de Escalação: {current_assessment['escalation_risk']}")
    
    # 7. RELATÓRIO NOWCAST
    print("\n🚨 GERANDO RELATÓRIO NOWCAST + INÉRCIA...")
    nowcast_report = model.generate_nowcast_report()
    print(nowcast_report)
    
    with open("nowcast_inertia_report.txt", "w") as f:
        f.write(nowcast_report)
    
    # 8. VISUALIZAÇÃO
    print("\n📈 CRIANDO VISUALIZAÇÕES...")
    visualizer = ProductionVisualizer()
    visualizer.create_final_dashboard(model.results, df, "hac_nowcast_final.png")
    
    # 9. RELATÓRIO FINAL
    print("\n📊 GERANDO RELATÓRIO FINAL...")
    reporter = FinalReport()
    reporter.generate_report(model.results, df, model)
    
    # 10. SALVAR RESULTADOS
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
    
    # 11. STATUS FINAL
    print("\n" + "="*70)
    print("✅ SISTEMA HAC++ (NOWCAST + INÉRCIA HÍBRIDO) - EXECUÇÃO CONCLUÍDA")
    print("="*70)
    
    if 'Storm_level' in model.results and len(model.results['Storm_level']) > 0:
        current_level = model.results['Storm_level'][-1]
        current_hac = model.results['HAC_total'][-1]
        
        print(f"\n🔴 STATUS OPERACIONAL:")
        print(f"   HAC: {current_hac:.1f}")
        print(f"   dH/dt: {model.results['dHAC_dt'][-1]:.1f} nT/h")
        print(f"   Nível: {current_level}")
        
        if 'escalation_alert' in model.results and model.results['escalation_alert'][-1]:
            print("   🚨 NOWCAST ALERT: ESCALAÇÃO IMINENTE!")
        
        print(f"\n📁 ARQUIVOS GERADOS:")
        print(f"   1. hac_nowcast_final.png - Dashboard completo")
        print(f"   2. hac_nowcast_results.csv - Dados com Nowcast")
        print(f"   3. hac_final_report.txt - Relatório geral")
        print(f"   4. nowcast_inertia_report.txt - Relatório específico")
    
    print("\n" + "="*70)
    
# ==========================================================
# 🔌 WRAPPERS PARA VALIDAÇÃO (compatível com validate_events)
# ==========================================================

def prepare_omni(df):
    """
    Adapter para validação.
    Normaliza + calcula campos físicos.
    """
    df = normalize_omni_columns(df)
    df = PhysicalFieldsCalculator.compute_all_fields(df)
    return df


def euler_dst(df, alpha=5.5, tau=8.0):
    """
    Adapter: usa HAC no lugar do Burton.
    Retorna proxy de Dst baseado no HAC.
    """
    model = ProductionHACModel()
    hac = model.compute_hac_system(df)

    # 🔴 Conversão HAC → Dst (proxy físico)
    dst_proxy = -350 * (hac / 300) ** 1.2

    return dst_proxy


# ============================
# EXECUÇÃO
# ============================
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    main()
