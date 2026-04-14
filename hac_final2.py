"""
HAC++ Model: Heliospheric Accumulated Coupling - PRODUÇÃO FINAL
COM NOWCAST + INÉRCIA (Previsão de Escalação Híbrida)
Versão Final Corrigida:
- Atualização física separada: decay → injeção → perda
- Perda dependente do estado (loss_factor variável)
- Saturação suave do HAC (evita clipping brusco)
- Fallback de coupling reduzido e com aviso
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
# 0. NORMALIZAÇÃO DE NOMES OMNI
# ============================
def normalize_omni_columns(df, allow_partial=False):
    column_map = {
        'time': 'time_tag', 'Time': 'time_tag', 'Epoch': 'time_tag', 'timestamp': 'time_tag',
        'V': 'speed', 'Vsw': 'speed', 'speed': 'speed',
        'N': 'density', 'Np': 'density', 'density': 'density',
        'Bz': 'bz_gsm', 'Bz_GSM': 'bz_gsm', 'bz_gsm': 'bz_gsm',
        'Bx': 'bx_gsm', 'By': 'by_gsm', 'Bt': 'bt'
    }
    df = df.copy()
    for col in df.columns:
        if col in column_map:
            df.rename(columns={col: column_map[col]}, inplace=True)
    print("\n🔍 Colunas detectadas:", list(df.columns))
    if not allow_partial:
        required = ['time_tag', 'speed', 'density', 'bz_gsm']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"❌ COLUNAS OBRIGATÓRIAS AUSENTES: {missing}\n➡️ Colunas disponíveis: {list(df.columns)}")
    return df

# ============================
# CONFIGURAÇÃO FÍSICA
# ============================
class HACPhysicsConfig:
    # Tempos característicos (horas)
    TAU_RING_CURRENT = 1.2
    TAU_SUBSTORM = 0.6
    TAU_IONOSPHERE = 0.2
    TAU_EFFECTIVE = 2.0

    # Saturação
    E_FIELD_SATURATION = 35.0    # mV/m
    KP_SATURATION = 8.0
    RING_CURRENT_MAX = 800.0

    # Particionamento
    ALPHA_RING = 0.7
    ALPHA_SUBSTORM = 0.2
    ALPHA_IONOSPHERE = 0.1

    # Não linearidade
    BETA_NONLINEAR = 2.2
    COUPLING_THRESHOLD = 3.0
    HAC_REF = 300.0

    # Escalas
    HAC_SCALE_MAX = 800.0
    KP_SCALE = 9.0

    # Limites físicos
    VSW_MIN, VSW_MAX = 200, 1500
    DENSITY_MIN, DENSITY_MAX = 0.1, 100
    BZ_MIN, BZ_MAX = -100, 100

    # Nowcast + Inércia
    THETA_CRITICAL = 50.0
    HG3_THRESHOLD = 150.0
    VSW_CRITICAL = 700.0
    BZ_CRITICAL = -8.0

    # Classificação híbrida
    DHDT_G5_THRESHOLD = 150.0
    DHDT_G4_THRESHOLD = 100.0
    DHDT_G3_THRESHOLD = 80.0
    BZ_G5_THRESHOLD = -15.0
    BZ_G4_THRESHOLD = -10.0
    BZ_G3_THRESHOLD = -8.0
    V_G5_THRESHOLD = 700.0
    V_G4_THRESHOLD = 650.0
    V_G3_THRESHOLD = 600.0

# ============================
# 1. CARREGAMENTO ROBUSTO OMNI
# ============================
class RobustOMNIProcessor:
    @staticmethod
    def load_and_clean(filepath, max_interpolation=3):
        print(f"📥 Carregando {filepath}...")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Erro: {e}")
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
            mean_v = df['speed'].mean()
            if mean_v > 10000:
                print("   ⚠️ Vsw em m/s → convertendo para km/s")
                df['speed'] = df['speed'] / 1000.0
            elif mean_v > 2000:
                print("   ⚠️ Vsw fora de escala → ajustando")
                df['speed'] = df['speed'] / 10.0
            df['speed'] = df['speed'].clip(lower=config.VSW_MIN, upper=config.VSW_MAX)

        if 'density' in df.columns:
            df['density'] = df['density'].clip(lower=config.DENSITY_MIN, upper=config.DENSITY_MAX)
        if 'bz_gsm' in df.columns:
            df['bz_gsm'] = df['bz_gsm'].clip(lower=config.BZ_MIN, upper=config.BZ_MAX)

        cols_to_interpolate = ['bz_gsm', 'speed', 'density']
        for col in cols_to_interpolate:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit=max_interpolation, limit_direction='both')

        has_speed = 'speed' in df.columns
        has_density = 'density' in df.columns
        has_bz = 'bz_gsm' in df.columns
        if has_speed and has_density:
            required = ['speed', 'density']
        elif has_bz:
            required = ['bz_gsm']
        else:
            raise ValueError(f"❌ Arquivo inválido. Colunas: {list(df.columns)}")
        df_clean = df.dropna(subset=required).copy()
        print(f"   ✅ {len(df_clean)} pontos válidos")
        return df_clean

    @staticmethod
    def merge_datasets(mag_df, plasma_df):
        if mag_df is None or plasma_df is None:
            return None
        df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
        df = df.sort_values('time_tag').reset_index(drop=True)
        critical_cols = ['speed', 'bz_gsm', 'density']
        for col in critical_cols:
            if col in df.columns:
                if col == 'speed': df[col] = df[col].fillna(400)
                elif col == 'bz_gsm': df[col] = df[col].fillna(0)
                elif col == 'density': df[col] = df[col].fillna(5)
        return df

# ============================
# 2. CAMPOS FÍSICOS
# ============================
class PhysicalFieldsCalculator:
    @staticmethod
    def compute_all_fields(df):
        df = df.copy()
        config = HACPhysicsConfig()
        bz = df['bz_gsm'].fillna(0).values
        by = df['by_gsm'].fillna(0).values if 'by_gsm' in df.columns else np.zeros_like(bz)
        v = df['speed'].fillna(400).values

        bz_negative = np.maximum(0, -bz)
        E_field_raw = bz_negative * v * 1e-3
        df['E_field_raw'] = E_field_raw
        df['E_field_saturated'] = np.clip(E_field_raw, 0, config.E_FIELD_SATURATION)

        Bt = np.sqrt(by**2 + bz**2)
        theta = np.arctan2(by, bz)
        theta_factor = np.sin(theta / 2.0) ** 4.0
        coupling_newell = (v ** (4/3)) * (Bt ** (2/3)) * theta_factor * 1e-4

        threshold = config.COUPLING_THRESHOLD
        beta = config.BETA_NONLINEAR
        coupling_nl = np.zeros_like(coupling_newell)
        mask_linear = coupling_newell <= threshold
        coupling_nl[mask_linear] = coupling_newell[mask_linear]
        mask_nl = coupling_newell > threshold
        if np.any(mask_nl):
            normalized = coupling_newell[mask_nl] / threshold
            coupling_nl[mask_nl] = threshold * (normalized ** beta)

        coupling_signal = np.where(bz < 0, coupling_nl, 0.0)
        if np.max(coupling_signal) < 1e-3:
            print("   ⚠️ Coupling extremamente baixo → usando fallback (10% do E_field)")
            coupling_signal = df['E_field_saturated'].values * 0.1
        coupling_signal = np.clip(coupling_signal, 0, 100)

        df['Bt'] = Bt
        df['theta'] = theta
        df['coupling_newell'] = coupling_newell
        df['coupling_nonlinear'] = coupling_nl
        df['coupling_signal'] = coupling_signal

        for col in ['E_field_raw', 'E_field_saturated', 'coupling_signal', 'coupling_newell', 'coupling_nonlinear']:
            df[col] = np.nan_to_num(df[col], nan=0.0, posinf=100, neginf=0.0)
        return df

# ============================
# 3. MODELO HAC+ PRINCIPAL
# ============================
class ProductionHACModel:
    def __init__(self, config=None):
        self.config = config or HACPhysicsConfig()
        self.core = HACCoreModel()
        self.results = {}
        self.nowcast_alerts = []
        self.escalation_triggers = []
        self.classification_logs = []

    def compute_hac_system(self, df):
        print("\n⚡ Calculando sistema HAC+...")
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
            # Frações de decaimento linear
            decay_rc = min(dt[i] / tau_rc, 1.0) if dt[i] > 0 else 0.0
            decay_sub = min(dt[i] / tau_sub, 1.0) if dt[i] > 0 else 0.0
            decay_ion = min(dt[i] / tau_ion, 1.0) if dt[i] > 0 else 0.0

            # --- INJECTION ---
            injection = max(0.0, coupling[i])
            if Bz[i] < 0:
                e_field = -Bz[i] * Vsw[i] * 1e-3
                e_field_clipped = np.clip(e_field, 0, self.config.E_FIELD_SATURATION)
                injection += 5.0 * e_field_clipped * (1.0 + 0.1 * abs(bz[i]))
                #if Bz[i] < -5:
                    #injection *= 1.5
                #if Bz[i] < -10:
                    #injection *= 2.5
            else:
                injection = 0.0

            dt_hours = dt[i] / 3600.0
            injection_eff = injection * dt_hours
            injection_eff = np.clip(injection_eff, 0, 80)

            # --- LOSS DEPENDENTE DO ESTADO ---
            loss_factor_ring = 0.02 + 0.0002 * np.sqrt(hac_ring[i-1])
            loss_ring = loss_factor_ring * hac_ring[i-1]

            loss_factor_sub = 0.02 + 0.0002 * np.sqrt(hac_substorm[i-1])
            loss_sub = loss_factor_sub * hac_substorm[i-1]

            loss_factor_ion = 0.02 + 0.0002 * np.sqrt(hac_ionosphere[i-1])]
            loss_ion = loss_factor_ion * hac_ionosphere[i-1]

            # --- ATUALIZAÇÃO COM DECAIMENTO LINEAR ---
            hac_ring[i] = hac_ring[i-1] * (1.0 - decay_rc)
            hac_ring[i] += self.config.ALPHA_RING * injection_eff
            hac_ring[i] -= loss_ring

            hac_substorm[i] = hac_substorm[i-1] * (1.0 - decay_sub)
            hac_substorm[i] += self.config.ALPHA_SUBSTORM * injection_eff
            hac_substorm[i] -= loss_sub

            hac_ionosphere[i] = hac_ionosphere[i-1] * (1.0 - decay_ion)
            hac_ionosphere[i] += self.config.ALPHA_IONOSPHERE * injection_eff
            hac_ionosphere[i] -= loss_ion

            # Garantir não-negatividade
            hac_ring[i] = max(0.0, hac_ring[i])
            hac_substorm[i] = max(0.0, hac_substorm[i])
            hac_ionosphere[i] = max(0.0, hac_ionosphere[i])

        hac_total = hac_ring + hac_substorm + hac_ionosphere
        hac_total = hac_total / (1.0 + hac_total / 800.0)   # saturação suave melhorada
        hac_total = np.clip(hac_total, 0, self.config.RING_CURRENT_MAX)

        dHAC_dt = self._compute_robust_derivative(hac_total, times)
        escalation_flags = self._detect_escalation_triggers(hac_total, dHAC_dt, Bz, Vsw, times)
        nowcast_growth = self._compute_nowcast_growth(hac_total, coupling)

        core_results = self.core.process(time=times, bz=Bz, v=Vsw, density=density, mode='nowcast')

        # Conversão HAC → Dst (recalibrada)
        dst_from_hac = -4.5 * (hac_total ** 1.25) - 20
        dst_from_hac = np.clip(dst_from_hac, -600, 50)
        dst_hybrid = dst_from_hac.copy()
        core_results['Dst_pred'] = dst_hybrid
        core_results['Dst_min'] = np.min(dst_hybrid)
        core_results['Dst_now'] = dst_hybrid[-1]

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

    def _compute_robust_derivative(self, hac_total, times):
        times = np.array(times, dtype="datetime64[s]")
        dt = np.diff(times).astype("timedelta64[s]").astype(float)
        dt = np.insert(dt, 0, np.median(dt))
        dt[dt <= 0] = 1.0
        dt_hours = np.maximum(dt / 3600.0, 1e-3)

        if hac_total.size < 7:
            dHAC_dt = np.gradient(hac_total) / dt_hours
        else:
            window = min(7, len(hac_total))
            if window < 3:
                return np.gradient(hac_total) / dt_hours
            if window % 2 == 0:
                window -= 1
            dt_median = np.median(dt_hours)
            try:
                dHAC_dt = savgol_filter(hac_total, window_length=window, polyorder=2, deriv=1, delta=dt_median)
            except:
                dHAC_dt = np.gradient(hac_total) / dt_hours
        dHAC_dt = np.nan_to_num(dHAC_dt, nan=0.0)
        limit = 300
        dHAC_dt = limit * np.tanh(dHAC_dt / limit)
        return dHAC_dt

    def _compute_nowcast_growth(self, hac_total, coupling):
        hac_nowcast = (coupling / self.config.E_FIELD_SATURATION) * self.config.HAC_SCALE_MAX
        growth = (1.0 / self.config.TAU_EFFECTIVE) * (hac_nowcast - hac_total)
        return growth

    def _detect_escalation_triggers(self, hac_total, dHAC_dt, Bz, Vsw, times):
        n = len(hac_total)
        escalation_flags = np.zeros(n, dtype=bool)
        theta = self.config.THETA_CRITICAL
        h_g3 = self.config.HG3_THRESHOLD
        bz_crit = self.config.BZ_CRITICAL
        v_crit = self.config.VSW_CRITICAL
        window_size = 30
        for i in range(window_size, n):
            condition1 = hac_total[i] < h_g3
            condition2 = dHAC_dt[i] > theta
            bz_window = Bz[max(0, i-window_size):i+1]
            v_window = Vsw[max(0, i-window_size):i+1]
            condition3 = np.median(bz_window) < bz_crit
            condition4 = np.median(v_window) > v_crit
            if condition1 and condition2 and condition3 and condition4:
                escalation_flags[i] = True
                alert_time = pd.to_datetime(times[i]) if isinstance(times[i], np.datetime64) else times[i]
                alert_info = {
                    'time': alert_time,
                    'HAC': float(hac_total[i]),
                    'dHAC_dt': float(dHAC_dt[i]),
                    'Bz_avg': float(np.mean(bz_window)),
                    'V_avg': float(np.mean(v_window)),
                    'forecast_horizon_hours': self.config.TAU_EFFECTIVE * 2
                }
                self.nowcast_alerts.append(alert_info)
                if not self.escalation_triggers or (alert_time - self.escalation_triggers[-1]['time']).total_seconds() > 3600:
                    self.escalation_triggers.append(alert_info)
        return escalation_flags

    def _classify_storm_with_nowcast(self, hac, dhdt, bz, v):
        if hac < 50: base_level, base_severity = "Quiet", 0
        elif hac < 100: base_level, base_severity = "G1", 1
        elif hac < 200: base_level, base_severity = "G2", 2
        elif hac < 350: base_level, base_severity = "G3", 3
        elif hac < 550: base_level, base_severity = "G4", 4
        else: base_level, base_severity = "G5", 5

        nowcast_score = 0
        if dhdt > 50: nowcast_score += 1
        if dhdt > 100: nowcast_score += 1
        if dhdt > 150: nowcast_score += 2
        if dhdt > 200: nowcast_score += 3
        if bz < -8: nowcast_score += 1
        if bz < -10: nowcast_score += 2
        if bz < -15: nowcast_score += 3
        if bz < -20: nowcast_score += 4
        if v > 600: nowcast_score += 1
        if v > 700: nowcast_score += 2
        if v > 800: nowcast_score += 3
        if hac > 50: nowcast_score += 1
        if hac > 100: nowcast_score += 1
        if hac > 150: nowcast_score += 2
        if hac > 200: nowcast_score += 2

        # Quiet override
        if hac < 20 and abs(dhdt) < 20 and bz > -2 and v < 450:
            return {'final_level': "G0", 'final_severity': 0, 'escalation': False,
                    'hac': hac, 'dhdt': dhdt, 'bz': bz, 'v': v,
                    'base_level': base_level, 'nowcast_score': nowcast_score}

        final_level, final_severity = base_level, base_severity
        if nowcast_score >= 16:
            if base_severity < 5: final_level, final_severity = "G5 (Nowcast Override)", 5
        elif nowcast_score >= 13:
            if base_severity < 4: final_level, final_severity = "G4 (Nowcast Override)", 4
        elif nowcast_score >= 10:
            if base_severity < 3: final_level, final_severity = "G3 (Nowcast Override)", 3
        elif nowcast_score >= 6:
            if base_severity < 2: final_level, final_severity = "G2 (Nowcast Enhancement)", 2

        if dhdt > 220 and bz < -10 and v > 750 and hac > 100:
            if base_severity < 5: final_level, final_severity = "G5 (Extreme Nowcast)", 5
        elif dhdt > 150 and bz < -8 and v > 650:
            if base_severity < 4: final_level, final_severity = "G4 (Strong Nowcast)", 4
        elif dhdt > 100 and bz < -5 and v > 600 and hac > 50:
            if base_severity < 3: final_level, final_severity = "G3 (Nowcast Trigger)", 3

        return {'final_level': final_level, 'final_severity': final_severity,
                'escalation': final_severity > base_severity,
                'hac': hac, 'dhdt': dhdt, 'bz': bz, 'v': v,
                'base_level': base_level, 'nowcast_score': nowcast_score}

    def _apply_trend_boost(self, storm_levels, hac_values, dHAC_dt):
        enhanced = storm_levels.copy()
        n = len(storm_levels)
        window = 60
        for i in range(window, n):
            recent_hac = hac_values[i-window:i]
            recent_dhdt = dHAC_dt[i-window:i]
            mean_dhdt = np.mean(recent_dhdt)
            max_dhdt = np.max(recent_dhdt)
            hac_increase = hac_values[i] - hac_values[i-window]
            if max_dhdt > 220 and mean_dhdt > 100 and hac_increase > 150:
                if "G5" not in storm_levels[i]:
                    enhanced[i] = "G5 (Trend Boost)"
            elif max_dhdt > 200 and mean_dhdt > 80 and hac_increase > 120:
                if "G4" not in storm_levels[i] and "G5" not in storm_levels[i]:
                    enhanced[i] = "G4 (Trend Boost)"
        return enhanced

    def _validate_output(self, hac_values):
        if np.any(np.isnan(hac_values)):
            raise ValueError("NaN detectado em HAC")
        print("   ✅ Validação passada")

    def predict_storm_indicators(self, hac_values):
        print("\n🌍 Predizendo indicadores (com Nowcast físico)...")
        kp_pred = 9 * np.tanh(hac_values / 180)
        dst_pred = self.results.get('Dst_physical', np.zeros_like(hac_values))
        dst_min = self.results.get('Dst_min_physical', np.min(dst_pred))
        dst_now = self.results.get('Dst_now', dst_pred[-1] if dst_pred.size > 0 else 0)

        storm_levels = []
        decision_logs = []
        dHAC_dt = self.results.get('dHAC_dt', np.zeros_like(hac_values))
        Bz = self.results.get('Bz', np.zeros_like(hac_values))
        Vsw = self.results.get('Vsw', np.full_like(hac_values, 400))

        for i in range(len(hac_values)):
            result = self._classify_storm_with_nowcast(hac_values[i], dHAC_dt[i], Bz[i], Vsw[i])
            level = result['final_level']

            # Blindagem adicional (evita G2/G3 com HAC muito baixo)
            if hac_values[i] < 20 and abs(dHAC_dt[i]) < 20:
                level = "G0"
                result['final_level'] = level
                result['final_severity'] = 0

            if Bz[i] < -8 and Vsw[i] > 600:
                if dst_pred[i] <= -300: level = "G5 (Dst Override)"
                elif dst_pred[i] <= -200: level = "G4 (Dst Override)"
                elif dst_pred[i] <= -150: level = "G3 (Dst Override)"
                elif dst_pred[i] <= -100: level = "G2 (Dst Override)"
                elif dst_pred[i] <= -50: level = "G1 (Dst Override)"
                result['final_level'] = level

            storm_levels.append(level)
            decision_logs.append(result)

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

        g4g5_final = sum(1 for l in enhanced_levels if "G4" in l or "G5" in l)
        g4g5_base = sum(1 for l in storm_levels if "G4" in l or "G5" in l)
        print(f"   • Kp máximo: {np.max(kp_pred):.1f}")
        print(f"   • Dst mínimo (físico): {dst_min:.1f} nT")
        print(f"   • Dst atual: {dst_now:.1f} nT")
        print(f"   • Eventos G4/G5 (base): {g4g5_base}")
        print(f"   • Eventos G4/G5 (boost): {g4g5_final}")

        forecast = self.results.get('forecast', {})
        if forecast:
            print("   • Previsão Dst:")
            for h, val in forecast.items():
                val_clipped = np.clip(val, -500, 50)
                print(f"       {h}: {val_clipped:.1f} nT")
        probs = self.results.get('core_probabilities', {})
        if probs:
            print("   • Probabilidades:")
            for k, v in probs.items():
                print(f"       {k}: {v*100:.1f}%")

        return kp_pred, dst_pred, enhanced_levels

    def generate_nowcast_report(self):
        safe_logs = [{'final_level': log.get('final_level','Unknown'),
                      'severity': log.get('final_severity', log.get('severity',0)),
                      'escalation': log.get('escalation', False),
                      'hac': log.get('hac',0.0), 'dhdt': log.get('dhdt',0.0),
                      'bz': log.get('bz',0.0), 'v': log.get('v',0.0),
                      'base_level': log.get('base_level','Unknown')}
                     for log in self.classification_logs]
        nowcast_escalations = sum(1 for log in safe_logs if log['escalation'])
        nowcast_g4g5 = sum(1 for log in safe_logs if log['severity'] >= 4)

        report = "="*70 + "\n🚨 RELATÓRIO NOWCAST + INÉRCIA\n" + "="*70 + "\n\n"
        report += f"Total de alertas: {len(self.nowcast_alerts)}\n"
        report += f"Triggers principais: {len(self.escalation_triggers)}\n"
        report += f"Escalações Nowcast: {nowcast_escalations}\n"
        report += f"Eventos G4/G5 Nowcast: {nowcast_g4g5}\n"

        if self.escalation_triggers:
            report += "\nALERTAS PRINCIPAIS:\n"
            for i, alert in enumerate(self.escalation_triggers[-3:], 1):
                report += f"{i}. {alert['time']}: HAC={alert['HAC']:.1f}, dH/dt={alert['dHAC_dt']:.1f} nT/h\n"
        return report

    def get_current_assessment(self):
        if not self.results or 'Storm_level' not in self.results:
            return None
        idx = -1
        return {
            'time': self.results['time'][idx],
            'HAC': float(self.results['HAC_total'][idx]),
            'dHAC_dt': float(self.results['dHAC_dt'][idx]),
            'Bz': float(self.results['Bz'][idx]),
            'Vsw': float(self.results['Vsw'][idx]),
            'classification': self.results['Storm_level'][idx],
            'escalation_risk': 'HIGH' if self.results['escalation_alert'][idx] else 'LOW'
        }

# ============================
# 4. VISUALIZAÇÃO
# ============================
class ProductionVisualizer:
    @staticmethod
    def create_final_dashboard(results, df, filename="hac_nowcast_final.png"):
        print(f"\n📈 Criando dashboard: {filename}")
        if len(results.get('HAC_total', [])) < 10:
            print("❌ Dados insuficientes")
            return None

        plt.style.use('default')
        fig, axes = plt.subplots(4, 2, figsize=(16, 14))
        fig.suptitle('HAC++ Model - Sistema Completo de Previsão (Nowcast + Inércia Híbrido)', fontsize=16, fontweight='bold')
        times = results['time']

        # Painel 1: HAC
        ax1 = axes[0, 0]
        hac = results['HAC_total']
        colors = []
        if 'Storm_level' in results:
            for level in results['Storm_level']:
                if "G5" in level: colors.append('#8B0000')
                elif "G4" in level: colors.append('#FF4500')
                elif "G3" in level: colors.append('#FF8C00')
                elif "G2" in level: colors.append('#FFD700')
                elif "G1" in level: colors.append('#ADFF2F')
                elif "Nowcast" in level: colors.append('#9370DB')
                else: colors.append('#1E90FF')
        else:
            colors = ['#d62728'] * len(hac)
        for i in range(len(hac)-1):
            ax1.plot(times[i:i+2], hac[i:i+2], color=colors[i], linewidth=2, alpha=0.8)
        for lvl, col, lbl in zip([50,100,150,200,250], ['#2ecc71','#f1c40f','#e67e22','#e74c3c','#9b59b6'], ['G1','G2','G3','G4','G5']):
            ax1.axhline(y=lvl, color=col, linestyle=':', alpha=0.7, label=f'{lbl} ({lvl})')
        ax1.set_ylabel('HAC Index')
        ax1.set_title('A. Estado do Reservatório')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 320)

        # Painel 2: dH/dt
        ax2 = axes[0, 1]
        ax2.plot(times, results['dHAC_dt'], color='#3498db', linewidth=1.5, label='dH/dt')
        ax2.axhline(y=HACPhysicsConfig.THETA_CRITICAL, color='orange', linestyle='--', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('dH/dt [nT/h]')
        ax2.set_title('B. Taxa de Crescimento')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Painel 3: Classificação
        ax3 = axes[1, 0]
        if 'Storm_level' in results and 'Storm_level_base' in results:
            severity_map = {'Quiet':0,'G1':1,'G2':2,'G3':3,'G4':4,'G5':5,
                            'G2 (Nowcast Enhancement)':2.5, 'G3 (Nowcast Trigger)':3.3, 'G3 (Nowcast Override)':3.7,
                            'G4 (Strong Nowcast)':4.3, 'G4 (Nowcast Override)':4.7, 'G4 (Trend Boost)':4.5,
                            'G5 (Extreme Nowcast)':5.3, 'G5 (Nowcast Override)':5.7, 'G5 (Trend Boost)':5.5}
            base_sev = np.array([severity_map.get(l,0) for l in results['Storm_level_base']])
            final_sev = np.array([severity_map.get(l,0) for l in results['Storm_level']])
            ax3.plot(times, base_sev, color='#7f8c8d', linestyle='--', label='Base')
            ax3.plot(times, final_sev, color='#9b59b6', linewidth=2, label='Híbrida')
            ax3.fill_between(times, 0, final_sev, where=(final_sev > base_sev), color='#ff9999', alpha=0.4, label='Escalação')
        ax3.set_yticks([0,1,2,3,4,5])
        ax3.set_yticklabels(['Quiet','G1','G2','G3','G4','G5'])
        ax3.set_ylabel('Classificação')
        ax3.set_title('C. Classificação Híbrida')
        ax3.legend(loc='upper left', fontsize=7)
        ax3.grid(True, alpha=0.3)

        # Painel 4: Bz
        ax4 = axes[1, 1]
        ax4.plot(times, results['Bz'], color='#2ecc71')
        ax4.fill_between(times, 0, results['Bz'], where=(results['Bz']<0), color='red', alpha=0.3)
        ax4.axhline(y=-5, color='yellow', linestyle=':')
        ax4.axhline(y=HACPhysicsConfig.BZ_G3_THRESHOLD, color='orange', linestyle='--')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_ylabel('Bz [nT]')
        ax4.set_title('D. Forçante do Vento Solar')
        ax4.grid(True, alpha=0.3)

        # Painel 5: Kp
        ax5 = axes[2, 0]
        ax5.plot(times, results['Kp_pred'], color='#e74c3c')
        for kp, col, lbl in zip([5,6,7,8,9], ['orange','yellow','red','darkred','purple'], ['G1','G2','G3','G4','G5']):
            ax5.axhline(y=kp, color=col, linestyle=':', alpha=0.7, label=f'Kp={kp} ({lbl})')
        ax5.set_ylabel('Kp previsto')
        ax5.set_title('E. Atividade Geomagnética')
        ax5.legend(loc='upper left', fontsize=7)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 9.5)

        # Painel 6: Velocidade
        ax6 = axes[2, 1]
        ax6.plot(times, results['Vsw'], color='#3498db')
        ax6.axhline(y=500, color='lightblue', linestyle=':')
        ax6.axhline(y=HACPhysicsConfig.V_G3_THRESHOLD, color='orange', linestyle='--')
        ax6.set_ylabel('V [km/s]')
        ax6.set_title('F. Velocidade do Vento Solar')
        ax6.grid(True, alpha=0.3)

        # Painel 7: Acoplamento
        ax7 = axes[3, 0]
        ax7.plot(times, results['coupling_signal'], color='#9b59b6')
        ax7.set_ylabel('Acoplamento')
        ax7.set_xlabel('Tempo (UTC)')
        ax7.set_title('G. Sinal de Acoplamento')
        ax7.grid(True, alpha=0.3)

        # Painel 8: Dst
        ax8 = axes[3, 1]
        ax8.plot(times, results['Dst_physical'], color='#e67e22', linewidth=1.5)
        for dst, col in zip([-50,-100,-200,-350], ['lightgreen','orange','red','darkred']):
            ax8.axhline(y=dst, color=col, linestyle=':', alpha=0.7)
        ax8.set_ylabel('Dst [nT]')
        ax8.set_xlabel('Tempo (UTC)')
        ax8.set_title('H. Dst Previsto (Físico)')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(-550, 50)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Dashboard salvo: {filename}")
        return fig

# ============================
# 5. RELATÓRIO FINAL
# ============================
class FinalReport:
    @staticmethod
    def generate_report(results, df, model, filename="hac_final_report.txt"):
        print("\n📊 Gerando relatório final...")
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO FINAL - SISTEMA HAC+ (NOWCAST + INÉRCIA HÍBRIDO)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Período: {df['time_tag'].min()} a {df['time_tag'].max()}\n")
            f.write(f"Pontos totais: {len(df)}\n\n")

            if 'HAC_total' in results:
                hac = results['HAC_total']
                f.write(f"HAC máximo: {np.max(hac):.2f}\n")
                f.write(f"HAC mínimo: {np.min(hac):.2f}\n")
                f.write(f"HAC médio:  {np.mean(hac):.2f}\n")
                if 'dHAC_dt' in results:
                    f.write(f"dH/dt máximo: {np.max(results['dHAC_dt']):.1f} nT/h\n")
                    f.write(f"dH/dt > 100 nT/h: {np.sum(results['dHAC_dt'] > 100)} pontos\n\n")

                if 'Storm_level' in results:
                    levels = results['Storm_level']
                    total = len(levels)
                    counts = {'G0':0,'G1':0,'G2':0,'G3':0,'G4':0,'G5':0}
                    for l in levels:
                        for k in counts:
                            if k in l: counts[k] += 1
                    f.write("Distribuição de níveis:\n")
                    for k,v in counts.items():
                        f.write(f"  {k}: {v} ({v/total*100:.1f}%)\n")

            if hasattr(model, 'escalation_triggers') and model.escalation_triggers:
                f.write(f"\nTriggers de escalação: {len(model.escalation_triggers)}\n")
            f.write(f"\nRelatório gerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"📝 Relatório salvo: {filename}")

# ============================
# 6. PIPELINE PRINCIPAL
# ============================
def main():
    print("\n" + "="*70)
    print("🚀 HAC++ MODEL - SISTEMA DE PRODUÇÃO (NOWCAST + INÉRCIA HÍBRIDO)")
    print("="*70)

    MAG_FILE = "data/mag-7-day.json"
    PLASMA_FILE = "data/plasma-7-day.json"

    processor = RobustOMNIProcessor()
    mag_df = processor.load_and_clean(MAG_FILE)
    plasma_df = processor.load_and_clean(PLASMA_FILE)

    if mag_df is None or plasma_df is None:
        print("❌ Falha no carregamento")
        return

    df = processor.merge_datasets(mag_df, plasma_df)
    if df is None or len(df) < 10:
        print("❌ Dados insuficientes")
        return

    calculator = PhysicalFieldsCalculator()
    df = calculator.compute_all_fields(df)

    model = ProductionHACModel()
    hac_values = model.compute_hac_system(df)
    model.predict_storm_indicators(hac_values)

    print("\n🚨 Relatório Nowcast:")
    print(model.generate_nowcast_report())
    with open("nowcast_inertia_report.txt", "w") as f:
        f.write(model.generate_nowcast_report())

    visualizer = ProductionVisualizer()
    visualizer.create_final_dashboard(model.results, df)

    reporter = FinalReport()
    reporter.generate_report(model.results, df, model)

    try:
        results_df = df.copy()
        for key, value in model.results.items():
            if key != 'time' and hasattr(value, "__len__") and len(value) == len(results_df):
                results_df[key] = value
        results_df.to_csv("hac_nowcast_results.csv", index=False)
        print("💾 Resultados salvos em hac_nowcast_results.csv")
    except Exception as e:
        print(f"⚠️ Erro ao salvar CSV: {e}")

    print("\n✅ EXECUÇÃO CONCLUÍDA")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    main()
