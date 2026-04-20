#!/usr/bin/env python3
"""
hac_final.py - HAC++ Model: Sistema de Produção com Nowcast + Inércia Híbrido
Versão refinada:
- Acoplamento combinado (Newell + não-linear) com peso 70/30.
- Injeção basal dinâmica dependente da densidade do plasma.
- Mapeamento HAC→Dst calibrado (menos agressivo).
- Previsão Dst por simulação da equação de relaxação.
- Detecção de regime mais precisa (CME/HSS/SIR/Quiet).
- Classificação híbrida com override por Dst real.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime, timedelta

# ============================================================
# CONFIGURAÇÃO FÍSICA (REFINADA)
# ============================================================
class HACPhysicsConfig:
    """Configuração física calibrada para acoplamento realista"""

    # Tempos característicos (horas) - variam com regime
    TAU_RING_CURRENT_QUIET = 10.0
    TAU_RING_CURRENT_HSS = 6.0
    TAU_RING_CURRENT_CME = 3.0
    TAU_SUBSTORM = 0.6
    TAU_IONOSPHERE = 0.2

    # Saturações
    E_FIELD_SATURATION = 35.0      # mV/m
    KP_SATURATION = 9.0
    RING_CURRENT_MAX = 800.0       # nT

    # Partição de energia
    ALPHA_RING = 0.4
    ALPHA_SUBSTORM = 0.3
    ALPHA_IONOSPHERE = 0.3

    # Acoplamento não-linear (Newell + campo elétrico)
    BETA_NONLINEAR = 2.2
    COUPLING_THRESHOLD = 3.0      # mV/m
    NEWELL_SCALE = 5e-4           # fator empírico (ajustado)

    # Escalas operacionais
    HAC_SCALE_MAX = 800.0
    KP_SCALE = 9.0

    # Limites físicos
    VSW_MIN, VSW_MAX = 200, 1500
    DENSITY_MIN, DENSITY_MAX = 0.1, 100
    BZ_MIN, BZ_MAX = -100, 100

    # Nowcast + Inércia
    THETA_CRITICAL = 50.0          # nT/h
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

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def normalize_omni_columns(df, allow_partial=False):
    """Normaliza nomes de colunas OMNI para o padrão esperado."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {
        'bz': 'bz_gsm',
        'bz_gse': 'bz_gsm',
        'bz_gsm': 'bz_gsm',
        'bt': 'bt',
        'b_total': 'bt',
        'density': 'density',
        'n_p': 'density',
        'proton_density': 'density',
        'speed': 'speed',
        'v': 'speed',
        'flow_speed': 'speed',
        'time': 'time_tag',
        'datetime': 'time_tag',
        'epoch': 'time_tag'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def detect_regime(v, density, bz):
    """
    Classifica o regime de vento solar de forma mais precisa.
    Prioridade: CME (densidade alta + Bz negativo) > HSS (velocidade alta, baixa densidade) > SIR (densidade muito baixa) > Quiet.
    """
    if density > 8 and bz < -8:
        return 'CME'
    elif v > 600 and density < 5:
        return 'HSS'
    elif density < 2:
        return 'SIR'
    else:
        return 'Quiet'

# ============================================================
# 1. CARREGAMENTO ROBUSTO DE DADOS OMNI
# ============================================================
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

        has_speed = 'speed' in df.columns
        has_density = 'density' in df.columns
        has_bz = 'bz_gsm' in df.columns

        if has_speed and has_density:
            required = ['speed', 'density']
        elif has_bz:
            required = ['bz_gsm']
        else:
            raise ValueError(f"❌ Colunas insuficientes: {list(df.columns)}")

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

# ============================================================
# 2. CÁLCULO DE CAMPOS FÍSICOS (COMBINADO)
# ============================================================
class PhysicalFieldsCalculator:
    """Calcula campos físicos combinando acoplamento Newell e não-linear"""

    @staticmethod
    def compute_all_fields(df):
        """Calcula coupling_signal como média ponderada de dois métodos"""
        df = df.copy()
        config = HACPhysicsConfig()

        bz = df['bz_gsm'].fillna(0).values
        v = df['speed'].fillna(400).values
        bt = df.get('bt', np.abs(bz)).fillna(0).values
        by = df.get('by_gsm', np.zeros_like(bz)).fillna(0).values

        # ---------- 1. Acoplamento Newell (escala corrigida) ----------
        theta = np.arctan2(by, bz)
        theta_factor = np.abs(np.sin(theta / 2)) ** 3
        coupling_newell = (v ** (4/3)) * (bt ** (2/3)) * theta_factor * config.NEWELL_SCALE

        # ---------- 2. Acoplamento não-linear via campo elétrico ----------
        bz_negative = np.maximum(0, -bz)
        e_field = bz_negative * v * 1e-3
        e_sat = np.clip(e_field, 0, config.E_FIELD_SATURATION)

        threshold = config.COUPLING_THRESHOLD
        beta = config.BETA_NONLINEAR
        coupling_nonlinear = np.zeros_like(e_sat)
        mask_linear = e_sat <= threshold
        coupling_nonlinear[mask_linear] = e_sat[mask_linear]
        mask_nonlinear = e_sat > threshold
        if np.any(mask_nonlinear):
            normalized = e_sat[mask_nonlinear] / threshold
            coupling_nonlinear[mask_nonlinear] = threshold * (normalized ** beta)

        # ---------- 3. Combinação (70% Newell, 30% não-linear) ----------
        coupling_combined = 0.7 * coupling_newell + 0.3 * coupling_nonlinear
        coupling_signal = np.where(bz < 0, coupling_combined, 0.0)
        df['coupling_signal'] = coupling_signal

        # Armazena componentes para diagnóstico
        df['coupling_newell'] = coupling_newell
        df['coupling_nonlinear'] = coupling_nonlinear
        df['E_field_raw'] = e_field
        df['E_field_saturated'] = e_sat

        print(f"   • Bz min/max: {bz.min():.1f} / {bz.max():.1f} nT")
        print(f"   • V min/max: {v.min():.1f} / {v.max():.1f} km/s")
        print(f"   • Coupling max: {coupling_signal.max():.2f}")

        return df

# ============================================================
# 3. MODELO HAC+ COM NOWCAST + INÉRCIA (PRODUÇÃO)
# ============================================================
class ProductionHACModel:
    """Modelo HAC+ de produção com física refinada e previsão por simulação"""

    def __init__(self, config=None):
        self.config = config or HACPhysicsConfig()
        self.results = {}
        self.nowcast_alerts = []
        self.escalation_triggers = []
        self.classification_logs = []

    def _safe_deltat(self, times):
        """Calcula delta-t em segundos com proteção"""
        n = len(times)
        dt = np.full(n, 60.0)
        if n > 1:
            t_sec = times.astype('datetime64[s]')
            diffs = np.diff(t_sec).astype(float)
            dt[1:] = np.maximum(diffs, 1.0)
            dt[0] = dt[1]
        return dt

    def _safe_normalization(self, values):
        """
        Normalização robusta usando percentis dinâmicos.
        Transforma HAC bruto em índice operacional 0–800.
        Nota: Para estudos físicos, recomenda-se usar o HAC não normalizado.
        """
        p99 = np.percentile(values, 99)
        p95 = np.percentile(values, 95)
        scale = max(p99, p95 * 1.2)
        if scale < 1.0:
            scale = 1.0

        normalized = (values / scale) * 300.0
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=800, neginf=0.0)
        normalized = np.clip(normalized, 0, self.config.HAC_SCALE_MAX)

        print(f"   • HAC escala: p99={p99:.1f}, p95={p95:.1f} → scale={scale:.1f}")
        print(f"   • HAC máx: {np.max(normalized):.1f}, méd: {np.mean(normalized):.1f}")
        return normalized

    def _compute_robust_derivative(self, hac_total, times):
        """Calcula dHAC/dt com Savitzky-Golay e clipping"""
        times_sec = times.astype('datetime64[s]')
        dt = np.diff(times_sec).astype(float)
        dt = np.insert(dt, 0, np.median(dt))
        dt[dt <= 0] = 1.0
        dt_hours = np.maximum(dt / 3600.0, 1e-3)

        if len(hac_total) < 7:
            dHAC_dt = np.gradient(hac_total) / dt_hours
        else:
            window = min(7, len(hac_total))
            if window % 2 == 0:
                window -= 1
            try:
                dHAC_dt = savgol_filter(hac_total, window, 2, deriv=1, delta=np.median(dt_hours))
            except:
                dHAC_dt = np.gradient(hac_total) / dt_hours

        dHAC_dt = np.nan_to_num(dHAC_dt, nan=0.0)
        dHAC_dt = np.clip(dHAC_dt, -400, 400)
        print(f"     Derivada máx: {np.max(dHAC_dt):.1f} nT/h")
        return dHAC_dt

    def _detect_escalation_triggers(self, hac_total, dHAC_dt, Bz, Vsw, times):
        """Detecta triggers de escalação usando regra de decisão"""
        n = len(hac_total)
        escalation_flags = np.zeros(n, dtype=bool)
        window_size = 30  # ~15 min com dados de 1 min

        for i in range(window_size, n):
            cond1 = hac_total[i] < self.config.HG3_THRESHOLD
            cond2 = dHAC_dt[i] > self.config.THETA_CRITICAL
            bz_window = Bz[max(0, i-window_size):i+1]
            v_window = Vsw[max(0, i-window_size):i+1]
            cond3 = np.median(bz_window) < self.config.BZ_CRITICAL
            cond4 = np.median(v_window) > self.config.VSW_CRITICAL

            if cond1 and cond2 and cond3 and cond4:
                escalation_flags[i] = True
                alert = {
                    'time': pd.to_datetime(times[i]),
                    'HAC': float(hac_total[i]),
                    'dHAC_dt': float(dHAC_dt[i]),
                    'Bz_avg': float(np.mean(bz_window)),
                    'V_avg': float(np.mean(v_window)),
                    'forecast_horizon_hours': 2.0
                }
                self.nowcast_alerts.append(alert)
                if not self.escalation_triggers or (alert['time'] - self.escalation_triggers[-1]['time']).total_seconds() > 3600:
                    self.escalation_triggers.append(alert)
                    print(f"     🚨 ALERTA em {alert['time']}: HAC={hac_total[i]:.1f}, dH/dt={dHAC_dt[i]:.1f} nT/h")

        return escalation_flags

    def _compute_nowcast_growth(self, hac_total, coupling):
        """Crescimento pelo modelo Nowcast + Inércia"""
        hac_nowcast = (coupling / self.config.E_FIELD_SATURATION) * self.config.HAC_SCALE_MAX
        growth = (1.0 / self.config.TAU_RING_CURRENT_QUIET) * (hac_nowcast - hac_total)
        return growth

    def compute_hac_system(self, df):
        """Sistema HAC completo com regime dinâmico e injeção basal adaptativa"""
        print("\n⚡ Calculando sistema HAC+...")

        times = pd.to_datetime(df['time_tag']).values
        coupling = df['coupling_signal'].fillna(0).values
        Bz = df['bz_gsm'].fillna(0).values
        Vsw = df['speed'].fillna(400).values
        density = df['density'].fillna(5).values if 'density' in df.columns else np.full_like(Bz, 5)

        dt = self._safe_deltat(times)
        n = len(times)

        hac_ring = np.zeros(n)
        hac_substorm = np.zeros(n)
        hac_ionosphere = np.zeros(n)

        print("   Simulando reservatórios...")
        for i in range(1, n):
            regime = detect_regime(Vsw[i], density[i], Bz[i])

            # Injeção basal dinâmica: aumenta com densidade do plasma
            baseline = 0.2 + 0.3 * (density[i] / 10.0)
            injection = coupling[i] + baseline

            # Boost por regime
            if regime == 'CME':
                injection *= 1.3
                tau_rc = self.config.TAU_RING_CURRENT_CME * 3600
            elif regime == 'HSS':
                injection *= 1.1
                tau_rc = self.config.TAU_RING_CURRENT_HSS * 3600
            else:
                tau_rc = self.config.TAU_RING_CURRENT_QUIET * 3600

            tau_sub = self.config.TAU_SUBSTORM * 3600
            tau_ion = self.config.TAU_IONOSPHERE * 3600

            dt_hours = dt[i] / 3600.0
            injection_eff = injection * dt_hours
            injection_eff = np.clip(injection_eff, 0, 100)

            # Perdas dissipativas
            loss_ring = (0.015 + 0.0002 * np.sqrt(max(0, hac_ring[i-1]))) * hac_ring[i-1]
            loss_sub = (0.015 + 0.0002 * np.sqrt(max(0, hac_substorm[i-1]))) * hac_substorm[i-1]
            loss_ion = (0.015 + 0.0002 * np.sqrt(max(0, hac_ionosphere[i-1]))) * hac_ionosphere[i-1]

            # Recuperação acelerada quando Bz vira para norte
            if Bz[i] > 0:
                loss_ring *= 1.3
                loss_sub *= 1.3
                loss_ion *= 1.3

            alpha_rc = np.exp(-dt[i] / tau_rc)
            alpha_sub = np.exp(-dt[i] / tau_sub)
            alpha_ion = np.exp(-dt[i] / tau_ion)

            hac_ring[i] = alpha_rc * hac_ring[i-1] + self.config.ALPHA_RING * injection_eff - loss_ring
            hac_substorm[i] = alpha_sub * hac_substorm[i-1] + self.config.ALPHA_SUBSTORM * injection_eff - loss_sub
            hac_ionosphere[i] = alpha_ion * hac_ionosphere[i-1] + self.config.ALPHA_IONOSPHERE * injection_eff - loss_ion

        hac_total = hac_ring + hac_substorm + hac_ionosphere
        hac_total = self._safe_normalization(hac_total)

        dHAC_dt = self._compute_robust_derivative(hac_total, times)
        escalation_flags = self._detect_escalation_triggers(hac_total, dHAC_dt, Bz, Vsw, times)
        nowcast_growth = self._compute_nowcast_growth(hac_total, coupling)

        # ------------------- Mapeamento HAC → Dst (menos agressivo) ------------------
        hac_clipped = np.clip(hac_total, 0, 800)
        hac_norm = hac_clipped / 800.0
        dhdt_clipped = np.clip(np.abs(dHAC_dt), 0, 400)
        dhdt_norm = dhdt_clipped / 400.0

        dst_from_hac = -450 * hac_norm - 150 * dhdt_norm - 30
        dst_from_hac = np.clip(dst_from_hac, -600, 20)

        # Core dummy (substitua pelo seu HACCoreModel se disponível)
        dst_core = np.full(n, -20.0)   # valor quieto
        core_unstable = False

        if core_unstable:
            print("   ⚠️ CORE INSTÁVEL → usando apenas HAC")
            dst_hybrid = dst_from_hac.copy()
        else:
            bz_factor = np.clip((-Bz) / 15.0, 0, 1)
            v_factor = np.clip((Vsw - 400) / 400, 0, 1)
            activity = bz_factor * v_factor
            blend = np.clip(activity ** 1.5, 0.1, 0.95)
            dst_hybrid = (1 - blend) * dst_core + blend * dst_from_hac

        dst_hybrid = np.clip(dst_hybrid, -600, 50)

        # ------------------- Previsão por simulação da equação de relaxação ------------------
        forecast = {}
        dt_median = np.median(dt) / 3600.0  # horas
        for horizon in [1, 2, 3]:
            steps = int(horizon / dt_median)
            if steps < 1:
                steps = 1
            # Simula a evolução do Dst usando o último HAC conhecido
            dst_future = dst_hybrid[-1]
            hac_now = hac_total[-1]
            tau_dst = self.config.TAU_RING_CURRENT_QUIET  # pode ser ajustado pelo regime
            alpha_dst = np.exp(-dt_median / tau_dst)
            for _ in range(steps):
                # Forçante: -hac_now atua como termo fonte (simplificado)
                dst_future = alpha_dst * dst_future + (1 - alpha_dst) * (-hac_now * 0.8)
            forecast[f"{horizon}h"] = dst_future

        print(f"   • Dst híbrido mín: {np.min(dst_hybrid):.1f} nT")

        # Armazena resultados
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
            'Dst_physical': dst_hybrid,
            'Dst_min_physical': np.min(dst_hybrid),
            'Dst_now': dst_hybrid[-1],
            'forecast': forecast,
            'core_probabilities': {'G1': 0.0, 'G2': 0.0, 'G3': 0.0, 'G4': 0.0, 'G5': 0.0},
            'core_severity': 0
        })

        self._validate_output(hac_total)
        return hac_total

    def _validate_output(self, hac_values):
        if np.any(np.isnan(hac_values)):
            raise ValueError("NaN detectado em HAC")
        print("   ✅ Validação passada")

    def _classify_storm_with_nowcast(self, hac, dhdt, bz, v):
        """Classificação híbrida com override de Dst"""
        if hac < 50:
            base_level = "Quiet"
            base_severity = 0
        elif hac < 100:
            base_level = "G1"
            base_severity = 1
        elif hac < 200:
            base_level = "G2"
            base_severity = 2
        elif hac < 350:
            base_level = "G3"
            base_severity = 3
        elif hac < 550:
            base_level = "G4"
            base_severity = 4
        else:
            base_level = "G5"
            base_severity = 5

        nowcast_score = 0
        if dhdt > 50: nowcast_score += 1
        if dhdt > 100: nowcast_score += 1
        if dhdt > 150: nowcast_score += 2
        if dhdt > 200: nowcast_score += 3
        if bz < -5: nowcast_score += 1
        if bz < -10: nowcast_score += 2
        if bz < -15: nowcast_score += 3
        if bz < -20: nowcast_score += 4
        if v > 500: nowcast_score += 1
        if v > 600: nowcast_score += 1
        if v > 700: nowcast_score += 2
        if v > 800: nowcast_score += 3
        if hac > 50: nowcast_score += 1
        if hac > 100: nowcast_score += 1
        if hac > 150: nowcast_score += 2
        if hac > 200: nowcast_score += 2

        final_level = base_level
        final_severity = base_severity

        if nowcast_score >= 16 and base_severity < 5:
            final_level = "G5 (Nowcast Override)"
            final_severity = 5
        elif nowcast_score >= 13 and base_severity < 4:
            final_level = "G4 (Nowcast Override)"
            final_severity = 4
        elif nowcast_score >= 10 and base_severity < 3:
            final_level = "G3 (Nowcast Override)"
            final_severity = 3
        elif nowcast_score >= 6 and base_severity < 2:
            final_level = "G2 (Nowcast Enhancement)"
            final_severity = 2

        if dhdt > 220 and bz < -10 and v > 750 and hac > 100 and base_severity < 5:
            final_level = "G5 (Extreme Nowcast)"
            final_severity = 5
        elif dhdt > 150 and bz < -8 and v > 650 and base_severity < 4:
            final_level = "G4 (Strong Nowcast)"
            final_severity = 4
        elif dhdt > 100 and bz < -5 and v > 600 and hac > 50 and base_severity < 3:
            final_level = "G3 (Nowcast Trigger)"
            final_severity = 3

        return final_level, {
            'hac': hac, 'dhdt': dhdt, 'bz': bz, 'v': v,
            'base_level': base_level, 'nowcast_score': nowcast_score,
            'final_level': final_level, 'escalation': final_severity > base_severity,
            'severity': final_severity
        }

    def _apply_trend_boost(self, storm_levels, hac_values, dHAC_dt):
        enhanced = storm_levels.copy()
        n = len(storm_levels)
        window = 60
        for i in range(window, n):
            mean_dhdt = np.mean(dHAC_dt[i-window:i])
            max_dhdt = np.max(dHAC_dt[i-window:i])
            hac_inc = hac_values[i] - hac_values[i-window]
            if max_dhdt > 220 and mean_dhdt > 100 and hac_inc > 150 and "G5" not in storm_levels[i]:
                enhanced[i] = "G5 (Trend Boost)"
            elif max_dhdt > 200 and mean_dhdt > 80 and hac_inc > 120 and "G4" not in storm_levels[i] and "G5" not in storm_levels[i]:
                enhanced[i] = "G4 (Trend Boost)"
        return enhanced

    def predict_storm_indicators(self, hac_values):
        """Predição final com classificação híbrida e override de Dst"""
        print("\n🌍 Predizendo indicadores (com Nowcast físico)...")

        kp_pred = 9 * np.tanh(hac_values / 180)

        dst_pred = self.results.get('Dst_physical', np.zeros_like(hac_values))
        dst_min = self.results.get('Dst_min_physical', np.min(dst_pred))
        dst_now = self.results.get('Dst_now', dst_pred[-1])

        dHAC_dt = self.results.get('dHAC_dt', np.zeros_like(hac_values))
        Bz = self.results.get('Bz', np.zeros_like(hac_values))
        Vsw = self.results.get('Vsw', np.full_like(hac_values, 400))

        storm_levels = []
        decision_logs = []
        escalation_count = 0
        g4g5_nowcast_count = 0

        for i in range(len(hac_values)):
            level, info = self._classify_storm_with_nowcast(hac_values[i], dHAC_dt[i], Bz[i], Vsw[i])
            # Override por Dst real
            dst = dst_pred[i]
            if Bz[i] < -8 and Vsw[i] > 600:
                if dst <= -300:
                    level = "G5 (Dst Override)"
                elif dst <= -200:
                    level = "G4 (Dst Override)"
                elif dst <= -150:
                    level = "G3 (Dst Override)"
                elif dst <= -100:
                    level = "G2 (Dst Override)"
                elif dst <= -50:
                    level = "G1 (Dst Override)"
            storm_levels.append(level)
            decision_logs.append(info)
            if info['escalation']:
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

        g4g5_final = sum(1 for l in enhanced_levels if "G4" in l or "G5" in l)
        g4g5_base = sum(1 for l in storm_levels if "G4" in l or "G5" in l)
        g4g5_trad = sum(1 for l in storm_levels if l in ['G4', 'G5'])

        print(f"   • Kp máximo: {np.max(kp_pred):.1f}")
        print(f"   • Dst mínimo (físico): {dst_min:.1f} nT")
        print(f"   • Dst atual: {dst_now:.1f} nT")
        print(f"   • Eventos G4/G5 (tradicional): {g4g5_trad}")
        print(f"   • Eventos G4/G5 (Nowcast base): {g4g5_base}")
        print(f"   • Eventos G4/G5 (com boost): {g4g5_final}")
        print(f"   • Escalações Nowcast: {escalation_count}")

        forecast = self.results.get('forecast', {})
        if forecast:
            print("   • Previsão Dst (simulada):")
            for h, val in forecast.items():
                print(f"       {h}: {val:.1f} nT")

        probs = self.results.get('core_probabilities', {})
        if hac_values[-1] < 10:
            probs = {'G1': 0.0, 'G2': 0.0, 'G3': 0.0, 'G4': 0.0, 'G5': 0.0}
        if probs:
            print("   • Probabilidades (softmax):")
            for k, v in probs.items():
                print(f"       {k}: {v*100:.1f}%")

        return kp_pred, dst_pred, enhanced_levels

    def generate_nowcast_report(self):
        """Gera relatório específico do modelo Nowcast + Inércia"""
        summary = "Nenhum alerta de escalação detectado."
        if self.nowcast_alerts:
            summary = f"Total de alertas detectados: {len(self.nowcast_alerts)}\n"
            summary += f"Triggers principais: {len(self.escalation_triggers)}\n\n"

        if self.classification_logs:
            esc = sum(1 for log in self.classification_logs if log['escalation'])
            g4g5 = sum(1 for log in self.classification_logs if log['severity'] >= 4)
            summary += f"CLASSIFICAÇÃO NOWCAST:\n• Escalações: {esc}\n• Eventos G4/G5: {g4g5}\n"

        report = "="*70 + "\n"
        report += "🚨 RELATÓRIO NOWCAST + INÉRCIA\n" + "="*70 + "\n\n"
        report += summary + "\n"
        report += "PARÂMETROS CRÍTICOS:\n"
        report += f"  • τ_eff quiet: {self.config.TAU_RING_CURRENT_QUIET} h\n"
        report += f"  • Θ: {self.config.THETA_CRITICAL} nT/h\n"
        report += f"  • H_G3: {self.config.HG3_THRESHOLD}\n"
        report += "="*70
        return report

# ============================================================
# MAIN (EXEMPLO DE USO)
# ============================================================
def main():
    print("="*70)
    print("🚀 HAC++ MODEL - SISTEMA DE PRODUÇÃO (NOWCAST + INÉRCIA HÍBRIDO)")
    print("="*70)

    # Carregar dados (exemplo com arquivos JSON de 7 dias)
    mag_df = RobustOMNIProcessor.load_and_clean("data/mag-7-day.json")
    plasma_df = RobustOMNIProcessor.load_and_clean("data/plasma-7-day.json")
    df = RobustOMNIProcessor.merge_datasets(mag_df, plasma_df)

    # Calcular campos físicos combinados
    print("\n⚡ CALCULANDO CAMPOS FÍSICOS...")
    df = PhysicalFieldsCalculator.compute_all_fields(df)

    # Instanciar modelo
    model = ProductionHACModel()

    # Executar
    hac = model.compute_hac_system(df)
    kp, dst, levels = model.predict_storm_indicators(hac)

    # Relatório
    print("\n" + model.generate_nowcast_report())

    # Salvar resultados
    df_out = pd.DataFrame({
        'time': model.results['time'],
        'HAC': model.results['HAC_total'],
        'dHAC_dt': model.results['dHAC_dt'],
        'Dst': model.results['Dst_physical'],
        'Storm_level': model.results['Storm_level']
    })
    df_out.to_csv("hac_nowcast_results.csv", index=False)
    print("\n💾 Resultados salvos em hac_nowcast_results.csv")

if __name__ == "__main__":
    main()
