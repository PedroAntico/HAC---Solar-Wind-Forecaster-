#!/usr/bin/env python3
"""
hac_final.py - HAC++ Model: Sistema de Produção com Nowcast + Inércia Híbrido
Versão final com parâmetros calibráveis e ganho de memória dependente do regime
(julho/2026)

Melhorias finais:
- Q_SATURATION como parâmetro configurável (substitui o divisor fixo 6.0).
- Ganho de memória dependente do regime (CME, HSS, Quiet).
- Q_SCALE ajustado para -60.0.
- Memória exponencial para eventos sustentados.
- Derivada 100% causal.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from collections import deque
import os

# ============================================================
# CONFIGURAÇÃO FÍSICA (CALIBRÁVEL)
# ============================================================
class HACPhysicsConfig:
    # Tempos característicos (horas)
    TAU_RING_CURRENT_QUIET = 10.0
    TAU_RING_CURRENT_HSS = 6.0
    TAU_RING_CURRENT_CME = 4.5
    TAU_SUBSTORM = 0.6
    TAU_IONOSPHERE = 0.2

    # Persistência do Bz (dinâmica)
    TAU_BZ_QUIET = 1.0
    TAU_BZ_HSS = 2.5
    TAU_BZ_CME = 0.25

    # Escalas físicas fixas
    E_FIELD_REF = 5.0
    NEWELL_REF = 1e4
    PRESSURE_REF = 3.0

    # Saturações
    E_FIELD_SATURATION = 35.0
    KP_SATURATION = 9.0
    RING_CURRENT_MAX = 800.0

    # Parâmetros do modelo Burton (calibrados)
    VBs_THRESHOLD = 0.5
    Q_SCALE = -60.0             # ganho principal da injeção
    Q_SATURATION = 8.0          # controle da suavidade da saturação
    TAU_DST = 12.0
    VBS_SAT = 28.0

    # Memória de reconexão
    TAU_RECONNECTION = 6.0
    RECONNECTION_SAT = 120.0
    RECONNECTION_K = 22.0

    # Magnetotail Storage Physics
    TAU_TAIL_LOADING = 3.5
    TAU_TAIL_UNLOADING = 7.0
    TAIL_ENERGY_MAX = 250.0
    TAIL_TO_RING = 0.66
    TAIL_TO_RING_GAIN = 1.0      # fator calibrável
    TAIL_TO_SUBSTORM = 0.25
    SUBSTORM_TRIGGER = 18.0
    TAIL_DISSIPATION = 0.013

    # Explosive unloading thresholds
    EXPLOSIVE_TAIL_THRESHOLD = 55.0
    EXPLOSIVE_VBS_THRESHOLD = 12.0

    # Memória exponencial para eventos sustentados
    TAU_ENERGY_MEMORY = 4.0      # horas
    ENERGY_MEMORY_GAIN_CME = 0.50
    ENERGY_MEMORY_GAIN_HSS = 0.25
    ENERGY_MEMORY_GAIN_QUIET = 0.15

    # Partição de energia (reservatórios HAC)
    ALPHA_RING = 0.4
    ALPHA_SUBSTORM = 0.3
    ALPHA_IONOSPHERE = 0.3

    # Acoplamento não‑linear (para HAC)
    BETA_NONLINEAR = 2.2
    COUPLING_THRESHOLD = 2.0
    NEWELL_SCALE = 5e-4

    # Escalas operacionais do HAC
    HAC_SCALE_MAX = 200.0
    HAC_NORM_FACTOR = 150.0

    # Limiares do HAC (provisórios)
    HAC_G1 = 20
    HAC_G2 = 40
    HAC_G3 = 60
    HAC_G4 = 85
    HAC_G5 = 105

    # Limites físicos
    VSW_MIN, VSW_MAX = 200, 1500
    DENSITY_MIN, DENSITY_MAX = 0.1, 100
    BZ_MIN, BZ_MAX = -100, 100

    # Nowcast + Inércia
    THETA_CRITICAL = 50.0
    HG3_THRESHOLD = 150.0
    VSW_CRITICAL = 700.0
    BZ_CRITICAL = -8.0

    # Flags para boosts (desabilitados)
    USE_BZ_BOOST = False
    USE_REGIME_BOOST = False


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def normalize_omni_columns(df, allow_partial=False):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        'bz': 'bz_gsm', 'bz_gse': 'bz_gsm', 'bz_gsm': 'bz_gsm',
        'bt': 'bt', 'b_total': 'bt',
        'density': 'density', 'n_p': 'density', 'proton_density': 'density',
        'speed': 'speed', 'v': 'speed', 'flow_speed': 'speed',
        'time': 'time_tag', 'datetime': 'time_tag', 'epoch': 'time_tag'
    }
    df.rename(columns=rename_map, inplace=True)
    return df


def _detect_regime_scalar(v, density, bz):
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
    @staticmethod
    def load_and_clean(filepath, max_interpolation=3):
        print(f"📥 Carregando {filepath}...")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Erro: {e}")
            return None
        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        df = normalize_omni_columns(df)
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
        df = df.sort_values('time_tag').reset_index(drop=True)
        numeric_cols = [c for c in df.columns if c != 'time_tag']
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        config = HACPhysicsConfig()
        if 'speed' in df.columns:
            df['speed'] = df['speed'].clip(config.VSW_MIN, config.VSW_MAX)
        if 'density' in df.columns:
            df['density'] = df['density'].clip(config.DENSITY_MIN, config.DENSITY_MAX)
        if 'bz_gsm' in df.columns:
            df['bz_gsm'] = df['bz_gsm'].clip(config.BZ_MIN, config.BZ_MAX)
        for col in ['bz_gsm', 'speed', 'density']:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit=max_interpolation, limit_direction='both')
        required = ['speed', 'density'] if ('speed' in df.columns and 'density' in df.columns) else ['bz_gsm']
        df = df.dropna(subset=required).copy()
        print(f"   ✅ {len(df)} pontos válidos")
        return df

    @staticmethod
    def merge_datasets(mag_df, plasma_df):
        if mag_df is None or plasma_df is None:
            return None
        df = pd.merge(mag_df, plasma_df, on='time_tag', how='outer')
        df = df.sort_values('time_tag').reset_index(drop=True)
        for col, fill in [('speed', 400), ('bz_gsm', 0), ('density', 5)]:
            if col in df.columns:
                df[col] = df[col].fillna(fill)
        return df


# ============================================================
# 2. CÁLCULO DE CAMPOS FÍSICOS
# ============================================================
class PhysicalFieldsCalculator:
    @staticmethod
    def compute_all_fields(df):
        df = df.copy()
        config = HACPhysicsConfig()
        bz = df['bz_gsm'].fillna(0).values
        v = df['speed'].fillna(400).values
        density = df['density'].fillna(5).values

        time_sec = pd.to_datetime(df['time_tag']).values.astype('datetime64[s]')
        dt_sec = np.diff(time_sec).astype(float)
        dt_sec = np.insert(dt_sec, 0, np.median(dt_sec))
        dt_hours = np.maximum(dt_sec / 3600.0, 1e-6)

        bz_neg = np.minimum(0, bz)
        bz_eff = np.zeros_like(bz)
        bz_eff[0] = bz_neg[0]
        regimes = np.array([_detect_regime_scalar(v[i], density[i], bz[i]) for i in range(len(bz))])
        for i in range(1, len(bz)):
            if regimes[i] == 'CME':
                tau_bz = config.TAU_BZ_CME
            elif regimes[i] == 'HSS':
                tau_bz = config.TAU_BZ_HSS
            else:
                tau_bz = config.TAU_BZ_QUIET
            alpha = np.exp(-dt_hours[i] / tau_bz)
            bz_eff[i] = alpha * bz_eff[i-1] + (1 - alpha) * bz_neg[i]
        bz_eff = np.minimum(bz_eff, 0.0)

        bz_south_real = np.maximum(0, -bz)
        vbs_real = v * bz_south_real * 1e-3
        vbs_real = np.clip(vbs_real, 0, config.E_FIELD_SATURATION)
        df['VBs_real'] = vbs_real

        bz_south_eff = np.maximum(0, -bz_eff)
        vbs_eff = v * bz_south_eff * 1e-3
        vbs_eff = np.clip(vbs_eff, 0, config.E_FIELD_SATURATION)
        df['VBs_eff'] = vbs_eff

        pdyn = 1.6726e-6 * density * (v ** 2)
        df['Pdyn'] = pdyn

        bt = df.get('bt', pd.Series(np.abs(bz), index=df.index)).fillna(0).values
        by = df.get('by_gsm', pd.Series(np.zeros_like(bz), index=df.index)).fillna(0).values
        theta = np.arctan2(by, bz)
        theta_factor = np.abs(np.sin(theta / 2)) ** 3
        coupling_newell = (v ** (4/3)) * (bt ** (2/3)) * theta_factor * config.NEWELL_SCALE

        e_field = (-bz_eff) * v * 1e-3
        e_sat = np.clip(e_field, 0, config.E_FIELD_SATURATION)
        thr = config.COUPLING_THRESHOLD
        beta = config.BETA_NONLINEAR
        coupling_nl = np.where(e_sat <= thr, e_sat, thr * ((e_sat / thr) ** beta))

        coupling_comb = 0.6 * coupling_newell + 0.4 * coupling_nl
        coupling_signal = np.where(bz_eff < 0, coupling_comb, 0.0)
        coupling_signal = 45 * np.tanh(coupling_signal / 26)
        df['coupling_signal'] = coupling_signal
        df['bz_eff'] = bz_eff

        print(f"   • Bz min/max: {bz.min():.1f} / {bz.max():.1f} nT")
        print(f"   • Bz eff min: {bz_eff.min():.1f} nT")
        print(f"   • V min/max: {v.min():.1f} / {v.max():.1f} km/s")
        print(f"   • VBs real max: {vbs_real.max():.2f} mV/m")
        print(f"   • Pdyn max: {pdyn.max():.1f} nPa")

        return df


# ============================================================
# 3. MODELO HAC+ (COM SATURAÇÃO E MEMÓRIA DEPENDENTE DO REGIME)
# ============================================================
class ProductionHACModel:
    def __init__(self, config=None, ml_corrector=None):
        self.config = config or HACPhysicsConfig()
        self.ml_corrector = ml_corrector
        self.results = {}
        self.nowcast_alerts = []
        self.escalation_triggers = []
        self.classification_logs = []
        self._online_residual_state = None
        self._online_ema_state = None
        self._energy_memory = 0.0

    def _safe_deltat(self, times):
        n = len(times)
        dt = np.full(n, 60.0)
        if n > 1:
            t_sec = times.astype('datetime64[s]')
            diffs = np.diff(t_sec).astype(float)
            dt[1:] = np.maximum(diffs, 1.0)
            dt[0] = dt[1]
        return dt

    def _safe_normalization(self, values):
        norm = np.clip(values, 0, self.config.HAC_SCALE_MAX)
        print(f"   • HAC máx: {np.max(norm):.1f}, méd: {np.mean(norm):.1f}")
        return norm

    def _compute_robust_derivative(self, hac_total, times):
        t_sec = times.astype('datetime64[s]')
        dt = np.diff(t_sec).astype(float)
        dt = np.insert(dt, 0, np.median(dt))
        dt[dt <= 0] = 1.0
        dt_h = np.maximum(dt / 3600.0, 1e-3)
        dH = np.zeros_like(hac_total)
        dH[1:] = np.diff(hac_total) / dt_h[1:]
        dH = np.nan_to_num(dH, nan=0.0)
        dH = np.clip(dH, -150, 150)
        print(f"     Derivada máx: {np.max(dH):.1f} nT/h")
        return dH

    def _detect_escalation_triggers(self, hac_total, dHAC_dt, Bz, Vsw, times):
        n = len(hac_total)
        flags = np.zeros(n, dtype=bool)
        window = 30
        for i in range(window, n):
            if (hac_total[i] < self.config.HG3_THRESHOLD and
                dHAC_dt[i] > self.config.THETA_CRITICAL and
                np.median(Bz[max(0, i-window):i+1]) < self.config.BZ_CRITICAL and
                np.median(Vsw[max(0, i-window):i+1]) > self.config.VSW_CRITICAL):
                flags[i] = True
                alert = {
                    'time': pd.to_datetime(times[i]),
                    'HAC': float(hac_total[i]),
                    'dHAC_dt': float(dHAC_dt[i]),
                    'Bz_avg': float(np.mean(Bz[max(0, i-window):i+1])),
                    'V_avg': float(np.mean(Vsw[max(0, i-window):i+1])),
                    'forecast_horizon_hours': 2.0
                }
                self.nowcast_alerts.append(alert)
                if (not self.escalation_triggers or
                    (alert['time'] - self.escalation_triggers[-1]['time']).total_seconds() > 3600):
                    self.escalation_triggers.append(alert)
        return flags

    def compute_hac_system(self, df, calibration_mode=False):
        print("\n⚡ Calculando sistema HAC+...")

        times = pd.to_datetime(df['time_tag']).values
        coupling = df['coupling_signal'].fillna(0).values
        Bz = df['bz_gsm'].fillna(0).values
        Vsw = df['speed'].fillna(400).values
        vbs_real = df.get('VBs_real', pd.Series(np.zeros_like(Bz))).values
        pdyn = df.get('Pdyn', pd.Series(np.full_like(Bz, 3.0))).values
        density = df['density'].fillna(5).values if 'density' in df.columns else np.full_like(Bz, 5)

        dt = self._safe_deltat(times)
        n = len(times)

        hac_ring = np.zeros(n)
        hac_substorm = np.zeros(n)
        hac_ionosphere = np.zeros(n)

        print("   Simulando reservatórios...")
        for i in range(1, n):
            regime = _detect_regime_scalar(Vsw[i], density[i], Bz[i])
            bz_eff_i = df['bz_eff'].iloc[i] if 'bz_eff' in df.columns else Bz[i]

            if coupling[i] < 0.02 and bz_eff_i > -3.0:
                baseline = 0.05 + 0.1 * (density[i] / 10.0)
            else:
                baseline = 0.15 + 0.25 * (density[i] / 10.0)
            injection = coupling[i] + baseline

            if self.config.USE_BZ_BOOST and Bz[i] < -5:
                injection *= (1.0 + 0.2 * min(abs(Bz[i]) / 20.0, 1.0))

            if self.config.USE_REGIME_BOOST:
                if regime == 'CME':
                    injection *= 1.3
                elif regime == 'HSS':
                    injection *= 1.1

            if regime == 'CME':
                tau_rc = self.config.TAU_RING_CURRENT_CME * 3600
            elif regime == 'HSS':
                tau_rc = self.config.TAU_RING_CURRENT_HSS * 3600
            else:
                tau_rc = self.config.TAU_RING_CURRENT_QUIET * 3600

            tau_sub = self.config.TAU_SUBSTORM * 3600
            tau_ion = self.config.TAU_IONOSPHERE * 3600

            dt_hours = dt[i] / 3600.0
            injection_eff = np.clip(injection * (dt_hours ** 0.82), 0, 100)

            base_loss = 0.010
            if regime == 'CME':
                base_loss *= 0.72
            elif regime == 'HSS':
                base_loss *= 0.88

            loss_ring = (base_loss + 0.00003 * np.sqrt(max(0, hac_ring[i-1]))) * hac_ring[i-1]
            loss_sub = (0.015 + 0.00003 * np.sqrt(max(0, hac_substorm[i-1]))) * hac_substorm[i-1]
            loss_ion = (0.015 + 0.00003 * np.sqrt(max(0, hac_ionosphere[i-1]))) * hac_ionosphere[i-1]

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

        hac_total = np.clip(hac_ring + hac_substorm + hac_ionosphere, 0, self.config.HAC_SCALE_MAX)
        print(f"   • HAC máx: {np.max(hac_total):.1f}, méd: {np.mean(hac_total):.1f}")

        dHAC_dt = self._compute_robust_derivative(hac_total, times)
        _ = self._detect_escalation_triggers(hac_total, dHAC_dt, Bz, Vsw, times)

        tau_rec = self.config.TAU_RECONNECTION
        rec_sat = self.config.RECONNECTION_SAT
        rec_k = self.config.RECONNECTION_K
        q_scale = self.config.Q_SCALE
        vbs_thr = self.config.VBs_THRESHOLD
        vbs_sat = self.config.VBS_SAT

        dst_ring = np.zeros(n)
        dst_pressure = np.zeros(n)
        dst_physical = np.zeros(n)
        dst_star = np.zeros(n)

        dst_ring[0] = -20.0
        dst_pressure[0] = np.clip(7.26 * np.sqrt(max(0.0, pdyn[0])) - 11.0, -20, 35)
        dst_physical[0] = dst_ring[0] + dst_pressure[0]
        dst_star[0] = dst_ring[0]

        injection_buffer = np.zeros(n)
        injection_buffer[0] = 0.0

        tail_energy = np.zeros(n)
        tail_release = np.zeros(n)
        tail_energy[0] = 0.0

        energy_memory = 0.0
        tau_memory = self.config.TAU_ENERGY_MEMORY

        mean_dt_hours = np.median(dt) / 3600.0
        transport_delay_h = np.clip(2.4 - (Vsw[0] - 400) / 500, 0.45, 2.5)
        delay_steps = max(1, int(transport_delay_h / mean_dt_hours))
        vbs_buffer = deque([0.0] * delay_steps, maxlen=delay_steps)

        for i in range(1, n):
            dt_hours = dt[i] / 3600.0

            dst_pressure[i] = np.clip(7.26 * np.sqrt(max(0.0, pdyn[i])) - 11.0, -20, 35)

            vbs_eff_val = max(0.0, vbs_real[i] - vbs_thr)

            # Memória exponencial: amplifica eventos sustentados
            alpha_mem = np.exp(-dt_hours / tau_memory)
            energy_memory = alpha_mem * energy_memory + (1.0 - alpha_mem) * vbs_eff_val

            # Ganho de memória dependente do regime
            regime_i = _detect_regime_scalar(Vsw[i], density[i], Bz[i])
            if regime_i == 'CME':
                mem_gain = self.config.ENERGY_MEMORY_GAIN_CME
            elif regime_i == 'HSS':
                mem_gain = self.config.ENERGY_MEMORY_GAIN_HSS
            else:
                mem_gain = self.config.ENERGY_MEMORY_GAIN_QUIET

            memory_gain = 1.0 + mem_gain * np.tanh(energy_memory / 8.0)
            vbs_eff_val *= memory_gain

            vbs_nl = vbs_sat * np.tanh(vbs_eff_val / 40.0)

            vbs_buffer.append(vbs_nl)
            vbs_delayed = vbs_buffer[0]

            if regime_i == 'CME':
                tau_inj_rise = 0.55
                tau_inj_decay = 4.5
            elif regime_i == 'HSS':
                tau_inj_rise = 2.5
                tau_inj_decay = 6.0
            else:
                tau_inj_rise = 0.8
                tau_inj_decay = 2.2

            if vbs_delayed > injection_buffer[i-1]:
                tau_inj = tau_inj_rise
            else:
                tau_inj = tau_inj_decay

            alpha_inj = np.exp(-dt_hours / tau_inj)
            injection_buffer[i] = alpha_inj * injection_buffer[i-1] + (1.0 - alpha_inj) * vbs_delayed
            injection_buffer[i] = np.clip(injection_buffer[i], 0, vbs_sat)

            tail_loading = 0.28 * injection_buffer[i] * dt_hours

            substorm_factor = np.clip(tail_energy[i-1] / self.config.SUBSTORM_TRIGGER, 0, 3)
            tail_unloading = ((0.11 + 0.07 * np.tanh(substorm_factor)) * tail_energy[i-1] * dt_hours)

            tail_loss = self.config.TAIL_DISSIPATION * tail_energy[i-1] * dt_hours

            tail_energy[i] = tail_energy[i-1] + tail_loading - tail_unloading - tail_loss
            tail_energy[i] = np.clip(tail_energy[i], 0, self.config.TAIL_ENERGY_MAX)

            tail_release[i] = tail_unloading

            if (tail_energy[i] > self.config.EXPLOSIVE_TAIL_THRESHOLD and
                injection_buffer[i] > self.config.EXPLOSIVE_VBS_THRESHOLD and
                Bz[i] < -12):
                explosive_release = 0.22 * tail_energy[i] * dt_hours
                explosive_release = np.clip(explosive_release, 0, 65)
                tail_release[i] += explosive_release

            explosive_factor = 1.0
            if (tail_energy[i] > self.config.EXPLOSIVE_TAIL_THRESHOLD and
                injection_buffer[i] > self.config.EXPLOSIVE_VBS_THRESHOLD and
                Bz[i] < -12):
                explosive_factor = 1.0 + 0.6 * np.tanh(tail_energy[i] / 90.0)
            tail_release[i] = tail_unloading * explosive_factor + (tail_release[i] - tail_unloading)

            # ================================================================
            # INJEÇÃO NO ANEL DE CORRENTE (SATURAÇÃO FÍSICA)
            # ================================================================
            ring_driver = (self.config.TAIL_TO_RING * self.config.TAIL_TO_RING_GAIN * tail_release[i] +
                          0.15 * injection_buffer[i])

            Q_linear = ring_driver ** 1.22

            regime_gain = 1.0
            if regime_i == 'CME':
                regime_gain += 0.16 * np.tanh(injection_buffer[i] / 10.0)
            elif regime_i == 'HSS':
                regime_gain -= 0.04 * np.tanh(injection_buffer[i] / 14.0)
            Q_linear *= regime_gain

            hac_feedback = np.clip(hac_total[i-1] / 120.0, 0.7, 1.5)
            Q_linear *= hac_feedback

            if regime_i == 'CME':
                extreme_factor = np.clip(injection_buffer[i] / 18.0, 0.0, 2.0)
                Q_linear *= (1.0 + 0.12 * extreme_factor ** 1.25)

            # Saturação suave e escala final
            Q_raw = q_scale * np.tanh(Q_linear / self.config.Q_SATURATION)

            # ================================================================
            # TAU DINÂMICO
            # ================================================================
            effective_bz = Bz[i]
            if effective_bz < 0:
                tau_dynamic = 12.0 + 6.0 * injection_buffer[i] / 15.0
            else:
                tau_dynamic = 6.0 + 3.0 * injection_buffer[i] / 15.0

            depth_factor = np.clip(abs(dst_ring[i-1]) / 180.0, 0, 2.5)
            tau_dynamic *= (1.0 - 0.15 * np.tanh(depth_factor))
            tau_dynamic = np.clip(tau_dynamic, 4.0, 42.0)

            dst_ring[i] = dst_ring[i-1] + (Q_raw - dst_ring[i-1] / tau_dynamic) * dt_hours
            dst_ring[i] = np.clip(dst_ring[i], -450, 40)

            dst_physical[i] = dst_ring[i] + dst_pressure[i]
            dst_physical[i] = np.clip(dst_physical[i], -500, 50)

            dst_star[i] = dst_ring[i]

        print(f"   • Dst físico mín: {np.min(dst_physical):.1f} nT")

        # Previsão (forecast) simplificada
        forecast = {}
        dt_median = np.median(dt) / 3600.0
        window_persist = min(120, n)
        weights = np.exp(-np.linspace(0, 3, window_persist))
        weights /= weights.sum()
        vbs_persist = np.sum(vbs_real[-window_persist:] * weights)
        pdyn_persist = pdyn[-1]

        for h in [1, 2, 3]:
            steps = max(1, int(h / dt_median))
            dst_fut = dst_physical[-1]
            dst_ring_fut = dst_ring[-1]
            inj_future = injection_buffer[-1]
            tail_fut = tail_energy[-1]

            for step in range(steps):
                tau_dyn = np.clip(14.0 + 12.0 * np.tanh(abs(dst_ring_fut) / 180.0), 6.0, 40.0)
                alpha = np.exp(-dt_median / tau_dyn)
                time_elapsed = step * dt_median
                decay = np.exp(-time_elapsed / 4.5)
                recent_window = vbs_real[-18:]
                if len(recent_window) >= 7:
                    smooth_recent = savgol_filter(recent_window, 7, 2)
                    recent_trend = np.mean(np.diff(smooth_recent))
                else:
                    recent_trend = 0.0
                vbs_future = max(0, vbs_persist * decay + recent_trend * 0.25)

                alpha_inj_f = np.exp(-dt_median / 2.5)
                inj_future = alpha_inj_f * inj_future + (1.0 - alpha_inj_f) * vbs_future

                tail_loading_f = 0.22 * inj_future * dt_median
                tail_unloading_f = (0.14 + 0.12 * np.clip(tail_fut / self.config.SUBSTORM_TRIGGER, 0, 3)) * tail_fut * dt_median
                tail_fut = np.clip(tail_fut + tail_loading_f - tail_unloading_f, 0, self.config.TAIL_ENERGY_MAX)

                ring_driver_f = 0.88 * tail_unloading_f + 0.12 * injection_buffer[-1]
                q_fut = q_scale * np.tanh(ring_driver_f / self.config.Q_SATURATION)

                dst_ring_fut = dst_ring_fut * alpha + q_fut * tau_dyn * (1.0 - alpha)

                pdyn_future = pdyn_persist * np.exp(-time_elapsed / 2.0)
                dst_pressure_future = np.clip(7.26 * np.sqrt(max(0.0, pdyn_future)) - 11.0, -20, 35)
                dst_fut = dst_ring_fut + dst_pressure_future

            forecast[f"{h}h"] = np.clip(dst_fut, -500, 50)

        self.results.update({
            'time': times,
            'HAC_total': hac_total,
            'dHAC_dt': dHAC_dt,
            'Bz': Bz,
            'Vsw': Vsw,
            'coupling_signal': coupling,
            'Dst_physical': dst_physical,
            'Dst_ring': dst_ring,
            'Dst_pressure': dst_pressure,
            'Dst_star': dst_star,
            'injection_buffer': injection_buffer,
            'tail_energy': tail_energy,
            'tail_release': tail_release,
            'Pdyn': pdyn,
            'VBs_eff': df.get('VBs_eff', pd.Series(np.zeros_like(Bz))).values,
            'density': density,
            'Dst_min_physical': np.min(dst_physical),
            'Dst_now': dst_physical[-1],
            'forecast': forecast
        })

        self._validate_output(hac_total)
        return hac_total

    def _validate_output(self, hac_values):
        if np.any(np.isnan(hac_values)):
            raise ValueError("NaN em HAC")
        print("   ✅ Validação passada")

    def predict_storm_indicators(self, hac_values):
        print("\n🌍 Predizendo indicadores (com Nowcast físico)...")
        dst_physics = self.results.get('Dst_physical', np.zeros_like(hac_values))
        pdyn = self.results.get('Pdyn', np.full_like(hac_values, 3.0))
        vbs_eff = self.results.get('VBs_eff', np.zeros_like(hac_values))
        dst_star = dst_physics - 7.26 * np.sqrt(np.maximum(0, pdyn)) + 11.0

        if self.ml_corrector is not None:
            features = self.ml_corrector.build_features(self.results, hac_values, dst_star)
            n = len(dst_star)
            residual_corrected = np.zeros(n)

            if self._online_residual_state is None:
                self._online_residual_state = 0.0

            DECAY = 0.001
            for t in range(n):
                x_t = features.iloc[t:t+1]
                delta_t = self.ml_corrector.predict_delta_residual(x_t)[0]
                self._online_residual_state *= (1.0 - DECAY)
                self._online_residual_state += delta_t

                clamp_scale = 0.5 * abs(hac_values[t])
                self._online_residual_state = np.clip(
                    self._online_residual_state, -50 - clamp_scale, 50 + clamp_scale
                )
                residual_corrected[t] = self._online_residual_state

            dst_star_pred = dst_star + residual_corrected
        else:
            dst_star_pred = dst_star

        dst_pred = dst_star_pred + 7.26 * np.sqrt(np.maximum(0, pdyn)) - 11.0

        if self._online_ema_state is None:
            self._online_ema_state = float(dst_pred[0])

        dst_causal = np.copy(dst_pred)
        dDst_dt_raw = np.zeros_like(dst_pred)
        dDst_dt_raw[1:] = np.diff(dst_pred)

        grad_smooth = np.copy(dDst_dt_raw)
        for i in range(1, len(grad_smooth)):
            grad_smooth[i] = 0.2 * dDst_dt_raw[i] + 0.8 * grad_smooth[i-1]

        for i in range(len(dst_pred)):
            if vbs_eff[i] > 5.0:
                alpha = 0.85
            elif dst_star[i] < -30:
                alpha = 0.2
            else:
                alpha = 0.05

            if i == 0:
                self._online_ema_state = float(dst_pred[0])
            else:
                self._online_ema_state = alpha * dst_pred[i] + (1 - alpha) * self._online_ema_state
            dst_causal[i] = self._online_ema_state

        smooth_mask = np.abs(grad_smooth) < 8.0
        dst_pred[smooth_mask] = dst_causal[smooth_mask]

        dst_min = np.min(dst_pred)
        dst_now = dst_pred[-1]
        self.results.update({
            'Dst_physical': dst_pred,
            'Dst_min_physical': dst_min,
            'Dst_now': dst_now
        })

        dHAC_dt = self.results.get('dHAC_dt', np.zeros_like(hac_values))
        Bz_arr = self.results.get('Bz', np.zeros_like(hac_values))
        Vsw_arr = self.results.get('Vsw', np.full_like(hac_values, 400))

        storm_levels, logs = [], []
        esc_cnt = g4g5_cnt = 0
        for i in range(len(hac_values)):
            level, info = self._classify_storm_with_nowcast(hac_values[i], dHAC_dt[i], Bz_arr[i], Vsw_arr[i])
            dst = dst_pred[i]
            if dst > -150 and ("G4" in level or "G5" in level):
                level = "G2 (Dst Clipped)"
            elif dst > -100 and ("G3" in level or "G4" in level or "G5" in level):
                level = "G1 (Dst Clipped)"
            storm_levels.append(level)
            logs.append(info)
            if info['escalation']:
                esc_cnt += 1
            if "G4" in level or "G5" in level:
                g4g5_cnt += 1

        enhanced = self._apply_trend_boost(storm_levels, hac_values, dHAC_dt)
        self.results.update({
            'Storm_level': enhanced,
            'Storm_level_base': storm_levels,
            'Decision_logs': logs
        })
        self.classification_logs = logs

        kp_from_dst = np.clip(0.072 * abs(dst_min)**0.8 + 0.7, 0, 9)
        kp_pred = np.full_like(hac_values, kp_from_dst)
        self.results['Kp_pred'] = kp_pred

        g4g5_final = sum(1 for l in enhanced if "G4" in l or "G5" in l)
        g4g5_base = sum(1 for l in storm_levels if "G4" in l or "G5" in l)
        g4g5_trad = sum(1 for l in storm_levels if l in ['G4', 'G5'])

        print(f"   • Kp máximo estimado: {np.max(kp_pred):.1f}")
        print(f"   • Dst mínimo (físico): {dst_min:.1f} nT")
        print(f"   • Dst atual: {dst_now:.1f} nT")
        print(f"   • Eventos G4/G5 (tradicional): {g4g5_trad}")
        print(f"   • Eventos G4/G5 (Nowcast base): {g4g5_base}")
        print(f"   • Eventos G4/G5 (com boost): {g4g5_final}")
        print(f"   • Escalações Nowcast: {esc_cnt}")

        forecast = self.results.get('forecast', {})
        if forecast:
            print("   • Previsão Dst (simulada):")
            for h, val in forecast.items():
                print(f"       {h}: {val:.1f} nT")

        recent_hac = np.max(hac_values[-48:])
        probs = {'G1': 0., 'G2': 0., 'G3': 0., 'G4': 0., 'G5': 0.}
        storm_score = max(recent_hac, 0.32 * abs(dst_min))
        if storm_score >= self.config.HAC_G5:
            probs['G5'] = 1.0
        elif storm_score >= self.config.HAC_G4:
            probs['G4'] = 1.0
        elif storm_score >= self.config.HAC_G3:
            probs['G3'] = 1.0
        elif storm_score >= self.config.HAC_G2:
            probs['G2'] = 1.0
        elif storm_score >= self.config.HAC_G1:
            probs['G1'] = 1.0
        print("   • Probabilidades (baseadas no HAC):")
        for k, v in probs.items():
            print(f"       {k}: {v*100:.1f}%")

        if self.ml_corrector is not None:
            try:
                p10, p50, p90 = self.ml_corrector.predict_interval(features)
                self.results['Dst_p10'] = dst_pred + (p10 - p50)
                self.results['Dst_p90'] = dst_pred + (p90 - p50)
            except:
                pass

        return kp_pred, dst_pred, enhanced

    def _classify_storm_with_nowcast(self, hac, dhdt, bz, v):
        cfg = self.config
        if hac < cfg.HAC_G1:
            base, sev = "Quiet", 0
        elif hac < cfg.HAC_G2:
            base, sev = "G1", 1
        elif hac < cfg.HAC_G3:
            base, sev = "G2", 2
        elif hac < cfg.HAC_G4:
            base, sev = "G3", 3
        elif hac < cfg.HAC_G5:
            base, sev = "G4", 4
        else:
            base, sev = "G5", 5

        score = 0
        if dhdt > 50: score += 1
        if dhdt > 100: score += 1
        if dhdt > 150: score += 2
        if bz < -5: score += 1
        if bz < -10: score += 2
        if bz < -15: score += 3
        if v > 500: score += 1
        if v > 600: score += 1
        if v > 700: score += 2
        if hac > 50: score += 1
        if hac > 100: score += 1
        if hac > 150: score += 2

        final, sev_f = base, sev
        if score >= 14 and sev < 5:
            final, sev_f = "G5 (Nowcast Override)", 5
        elif score >= 13 and sev < 4:
            final, sev_f = "G4 (Nowcast Override)", 4
        elif score >= 10 and sev < 3:
            final, sev_f = "G3 (Nowcast Override)", 3
        elif score >= 5 and sev < 2:
            final, sev_f = "G2 (Nowcast Enhancement)", 2

        if dhdt > 180 and bz < -10 and v > 700 and hac > 100 and sev < 5:
            final, sev_f = "G5 (Extreme Nowcast)", 5
        elif dhdt > 120 and bz < -8 and v > 600 and sev < 4:
            final, sev_f = "G4 (Strong Nowcast)", 4

        confidence = min(1.0, score / 20.0)

        return final, {'hac': hac, 'dhdt': dhdt, 'bz': bz, 'v': v,
                       'base_level': base, 'nowcast_score': score,
                       'final_level': final, 'escalation': sev_f > sev,
                       'severity': sev_f, 'confidence': confidence}

    def _apply_trend_boost(self, storm_levels, hac_values, dHAC_dt):
        enhanced = storm_levels.copy()
        n = len(storm_levels)
        window = 60
        for i in range(window, n):
            mean_dhdt = np.mean(dHAC_dt[i-window:i])
            max_dhdt = np.max(dHAC_dt[i-window:i])
            hac_inc = hac_values[i] - hac_values[i-window]
            cur = storm_levels[i]
            if max_dhdt > 180 and mean_dhdt > 70 and hac_inc > 120 and "G5" not in cur:
                enhanced[i] = "G5 (Trend Boost)"
            elif max_dhdt > 140 and mean_dhdt > 50 and hac_inc > 90 and "G4" not in cur and "G5" not in cur:
                enhanced[i] = "G4 (Trend Boost)"
        return enhanced

    def generate_nowcast_report(self):
        alerts = self.nowcast_alerts
        if not alerts:
            summary = "Nenhum alerta de escalação detectado."
        else:
            summary = f"Total de alertas detectados: {len(alerts)}\n"
            summary += f"Triggers principais: {len(self.escalation_triggers)}\n\n"
        if self.classification_logs:
            esc = sum(1 for log in self.classification_logs if log['escalation'])
            g4g5 = sum(1 for log in self.classification_logs if log['severity'] >= 4)
            summary += f"CLASSIFICAÇÃO NOWCAST:\n• Escalações: {esc}\n• Eventos G4/G5: {g4g5}\n"
        report = "=" * 70 + "\n🚨 RELATÓRIO NOWCAST + INÉRCIA\n" + "=" * 70 + "\n\n"
        report += summary + "\n"
        report += f"PARÂMETROS CRÍTICOS:\n  • τ_eff quiet: {self.config.TAU_RING_CURRENT_QUIET} h\n"
        report += f"  • Θ: {self.config.THETA_CRITICAL} nT/h\n  • H_G3: {self.config.HG3_THRESHOLD}\n"
        report += "=" * 70
        return report


# ============================================================
# MAIN (EXEMPLO DE USO)
# ============================================================
def main():
    print("=" * 70)
    print("🚀 HAC++ MODEL - SISTEMA DE PRODUÇÃO (NOWCAST + INÉRCIA HÍBRIDO)")
    print("=" * 70)
    mag_df = RobustOMNIProcessor.load_and_clean("data/mag-7-day.json")
    plasma_df = RobustOMNIProcessor.load_and_clean("data/plasma-7-day.json")
    df = RobustOMNIProcessor.merge_datasets(mag_df, plasma_df)
    print("\n⚡ CALCULANDO CAMPOS FÍSICOS...")
    df = PhysicalFieldsCalculator.compute_all_fields(df)
    model = ProductionHACModel()
    hac = model.compute_hac_system(df)
    kp, dst, levels = model.predict_storm_indicators(hac)
    print("\n" + model.generate_nowcast_report())
    df_out = pd.DataFrame({
        'time': model.results['time'],
        'HAC': model.results['HAC_total'],
        'dHAC_dt': model.results['dHAC_dt'],
        'Dst': model.results['Dst_physical'],
        'Dst_ring': model.results['Dst_ring'],
        'Dst_pressure': model.results['Dst_pressure'],
        'tail_energy': model.results['tail_energy'],
        'Storm_level': model.results['Storm_level']
    })
    df_out.to_csv("hac_nowcast_results.csv", index=False)
    print("\n💾 Resultados salvos em hac_nowcast_results.csv")


if __name__ == "__main__":
    main()
