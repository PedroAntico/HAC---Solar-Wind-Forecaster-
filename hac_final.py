#!/usr/bin/env python3
"""
hac_final.py - HAC++ Model: Sistema de Produção com Nowcast + Inércia Híbrido
Versão com memória de reconexão invariante, recuperação alongada e métricas refinadas
(maio/2026)

Melhorias aplicadas:
- Memória de reconexão com decaimento exponencial (tau=18h) em vez de fator fixo.
- Saturação da memória de reconexão (clip em 120) para evitar acúmulo espúrio.
- memory_factor mais gradual (expoente 0.55 em vez de tanh/14).
- Tendência recente suavizada via Savitzky‑Golay no forecast.
- Recuperação extrema: tau_dynamic = 30h para Vsw>850 e Bz<-18.
- Compressão Burton‑like reduzida (q_comp = -0.18*sqrt(Pdyn)).
- Métrica adicional sugerida: área integrada do erro (implementar no validate_events.py).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# CONFIGURAÇÃO FÍSICA (CALIBRADA)
# ============================================================
class HACPhysicsConfig:
    # Tempos característicos (horas)
    TAU_RING_CURRENT_QUIET = 10.0
    TAU_RING_CURRENT_HSS = 6.0
    TAU_RING_CURRENT_CME = 3.0
    TAU_SUBSTORM = 0.6
    TAU_IONOSPHERE = 0.2

    # Persistência do Bz (dinâmica)
    TAU_BZ_QUIET = 1.0
    TAU_BZ_HSS = 2.5
    TAU_BZ_CME = 0.2            # resposta quase instantânea a choques

    # Escalas físicas fixas
    E_FIELD_REF = 5.0           # mV/m
    NEWELL_REF = 1e4
    PRESSURE_REF = 3.0          # nPa

    # Saturações
    E_FIELD_SATURATION = 35.0   # mV/m
    KP_SATURATION = 9.0
    RING_CURRENT_MAX = 800.0    # nT

    # Parâmetros do modelo Burton (calibrados)
    VBs_THRESHOLD = 0.5         # mV/m
    Q_SCALE = -2.8              # nT/h por mV/m
    TAU_DST = 12.0              # horas
    VBS_SAT = 28.0              # mV/m – saturação não‑linear do acoplamento

    # Memória de reconexão (NOVO)
    TAU_RECONNECTION = 18.0     # horas – decaimento da memória
    RECONNECTION_SAT = 120.0    # saturação da memória

    # Partição de energia (reservatórios HAC)
    ALPHA_RING = 0.4
    ALPHA_SUBSTORM = 0.3
    ALPHA_IONOSPHERE = 0.3

    # Acoplamento não‑linear (para HAC)
    BETA_NONLINEAR = 2.2
    COUPLING_THRESHOLD = 2.0    # mV/m
    NEWELL_SCALE = 5e-4

    # Escalas operacionais do HAC (provisórias)
    HAC_SCALE_MAX = 200.0       # ajustado para valores brutos
    HAC_NORM_FACTOR = 150.0     # não aplicado agora

    # Limiares do HAC (provisórios – calibrar com dados)
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
    THETA_CRITICAL = 50.0       # nT/h
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

        # ------------------------------------------------------------
        # Bz efetivo com TAU_BZ DINÂMICO (para HAC/Nowcast)
        # ------------------------------------------------------------
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

        bz_eff = np.minimum(bz_eff, 0.0)   # garante não positivo

        # ------------------------------------------------------------
        # Driver Burton: VBs com Bz REAL (sem memória)
        # ------------------------------------------------------------
        bz_south_real = np.maximum(0, -bz)
        vbs_real = v * bz_south_real * 1e-3
        vbs_real = np.clip(vbs_real, 0, config.E_FIELD_SATURATION)
        df['VBs_real'] = vbs_real

        # VBs efetivo (com bz_eff) para HAC/Nowcast
        bz_south_eff = np.maximum(0, -bz_eff)
        vbs_eff = v * bz_south_eff * 1e-3
        vbs_eff = np.clip(vbs_eff, 0, config.E_FIELD_SATURATION)
        df['VBs_eff'] = vbs_eff

        # ------------------------------------------------------------
        # Pressão dinâmica CORRIGIDA (nPa)
        # ------------------------------------------------------------
        pdyn = 1.6726e-6 * density * (v ** 2)
        df['Pdyn'] = pdyn

        # ------------------------------------------------------------
        # Acoplamento Newell (mantido para HAC)
        # ------------------------------------------------------------
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
        coupling_signal = 32 * np.tanh(coupling_signal / 18)

        df['coupling_signal'] = coupling_signal
        df['bz_eff'] = bz_eff

        print(f"   • Bz min/max: {bz.min():.1f} / {bz.max():.1f} nT")
        print(f"   • Bz eff min: {bz_eff.min():.1f} nT")
        print(f"   • V min/max: {v.min():.1f} / {v.max():.1f} km/s")
        print(f"   • VBs real max: {vbs_real.max():.2f} mV/m")
        print(f"   • Pdyn max: {pdyn.max():.1f} nPa")

        return df


# ============================================================
# 3. MODELO HAC+
# ============================================================
class ProductionHACModel:
    def __init__(self, config=None):
        self.config = config or HACPhysicsConfig()
        self.results = {}
        self.nowcast_alerts = []
        self.escalation_triggers = []
        self.classification_logs = []

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
        """Normalização removida – usamos valores brutos."""
        norm = np.clip(values, 0, self.config.HAC_SCALE_MAX)
        print(f"   • HAC máx: {np.max(norm):.1f}, méd: {np.mean(norm):.1f}")
        return norm

    def _compute_robust_derivative(self, hac_total, times):
        t_sec = times.astype('datetime64[s]')
        dt = np.diff(t_sec).astype(float)
        dt = np.insert(dt, 0, np.median(dt))
        dt[dt <= 0] = 1.0
        dt_h = np.maximum(dt / 3600.0, 1e-3)

        # Adaptativo: para tempestades fortes, usa gradiente simples (preserva picos)
        if np.max(hac_total) > 40:
            dH = np.gradient(hac_total) / dt_h
        else:
            if len(hac_total) < 7:
                dH = np.gradient(hac_total) / dt_h
            else:
                w = min(7, len(hac_total))
                if w % 2 == 0:
                    w -= 1
                try:
                    dH = savgol_filter(hac_total, w, 2, deriv=1, delta=np.median(dt_h))
                except:
                    dH = np.gradient(hac_total) / dt_h

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

        # Driver Burton (Bz REAL) – array NumPy
        vbs_real = df.get('VBs_real', pd.Series(np.zeros_like(Bz))).values
        # Pressão dinâmica – array NumPy
        pdyn = df.get('Pdyn', pd.Series(np.full_like(Bz, 3.0))).values

        density = df['density'].fillna(5).values if 'density' in df.columns else np.full_like(Bz, 5)

        dt = self._safe_deltat(times)
        n = len(times)

        # ========================================================
        # Reservatórios HAC (indicador auxiliar)
        # ========================================================
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
            injection_eff = np.clip(injection * dt_hours, 0, 100)

            loss_ring = (0.015 + 0.00008 * np.sqrt(max(0, hac_ring[i-1]))) * hac_ring[i-1]
            loss_sub = (0.015 + 0.00008 * np.sqrt(max(0, hac_substorm[i-1]))) * hac_substorm[i-1]
            loss_ion = (0.015 + 0.00008 * np.sqrt(max(0, hac_ionosphere[i-1]))) * hac_ionosphere[i-1]

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

        # HAC total (bruto, sem normalização)
        hac_total = np.clip(hac_ring + hac_substorm + hac_ionosphere, 0, self.config.HAC_SCALE_MAX)
        print(f"   • HAC máx: {np.max(hac_total):.1f}, méd: {np.mean(hac_total):.1f}")

        dHAC_dt = self._compute_robust_derivative(hac_total, times)
        _ = self._detect_escalation_triggers(hac_total, dHAC_dt, Bz, Vsw, times)

        # ========================================================
        # Dst Burton (VBs REAL, com memória de reconexão e regime)
        # ========================================================
        tau_dst_base = self.config.TAU_DST
        q_scale = self.config.Q_SCALE
        vbs_thr = self.config.VBs_THRESHOLD
        vbs_sat = self.config.VBS_SAT
        tau_rec = self.config.TAU_RECONNECTION
        rec_sat = self.config.RECONNECTION_SAT

        dst_physical = np.zeros(n)
        dst_physical[0] = -20.0

        # Memória de reconexão (acumula VBs não‑linear ao longo do tempo)
        reconnection_memory = np.zeros(n)

        for i in range(1, n):
            dt_hours = dt[i] / 3600.0

            # VBs efetivo acima do limiar
            vbs_eff_val = max(0.0, vbs_real[i] - vbs_thr)

            # SATURAÇÃO NÃO‑LINEAR
            vbs_nl = vbs_eff_val / (1.0 + vbs_eff_val / vbs_sat)

            # Memória de reconexão (decaimento temporal, invariante à resolução)
            alpha_rec = np.exp(-dt_hours / tau_rec)
            reconnection_memory[i] = alpha_rec * reconnection_memory[i-1] + vbs_nl * dt_hours
            reconnection_memory[i] = np.clip(reconnection_memory[i], 0, rec_sat)

            # Fator de memória mais gradual (expoente 0.55)
            memory_factor = np.minimum(reconnection_memory[i], 120.0) ** 0.55

            # Q instantâneo + contribuição da memória
            Q_raw = q_scale * (0.65 * vbs_nl + 0.35 * memory_factor)

            # Compressão Burton‑like (reduzida)
            q_comp = -0.18 * np.sqrt(max(0.0, pdyn[i]))
            Q_injection = Q_raw + q_comp

            # Diferenciação de regime no próprio Burton
            regime_i = _detect_regime_scalar(Vsw[i], density[i], Bz[i])
            if regime_i == 'CME':
                Q_injection *= 1.18
            elif regime_i == 'HSS':
                Q_injection *= 0.92

            # Recuperação alongada para tempestades profundas
            if Vsw[i] > 850 and Bz[i] < -18:
                tau_dynamic = 30.0          # eventos extremos têm recuperação lenta
            elif dst_physical[i-1] < -250:
                tau_dynamic = 36.0
            elif dst_physical[i-1] < -150:
                tau_dynamic = 28.0
            elif dst_physical[i-1] < -80:
                tau_dynamic = 18.0
            else:
                tau_dynamic = 12.0

            alpha = np.exp(-dt_hours / tau_dynamic)

            dst_physical[i] = (dst_physical[i-1] * alpha
                               + Q_injection * tau_dynamic * (1.0 - alpha))
            dst_physical[i] = np.clip(dst_physical[i], -500, 50)

        print(f"   • Dst físico mín: {np.min(dst_physical):.1f} nT")

        # ========================================================
        # Forecast com persistência suave e decaimento lento
        # ========================================================
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
            for step in range(steps):
                tau_dyn = tau_dst_base * (1.0 + 0.28 * abs(dst_fut) / 140.0)
                alpha = np.exp(-dt_median / tau_dyn)
                time_elapsed = step * dt_median
                decay = np.exp(-time_elapsed / 4.5)          # decaimento mais lento
                # Tendência recente suavizada com Savitzky‑Golay
                recent_window = vbs_real[-18:]
                if len(recent_window) >= 7:
                    smooth_recent = savgol_filter(recent_window, 7, 2)
                    recent_trend = np.mean(np.diff(smooth_recent))
                else:
                    recent_trend = 0.0
                vbs_future = max(0, vbs_persist * decay + recent_trend * 0.25)
                vbs_future_eff = max(0.0, vbs_future - vbs_thr)
                vbs_future_nl = vbs_future_eff / (1.0 + vbs_future_eff / vbs_sat)
                q_fut = q_scale * vbs_future_nl
                q_comp_fut = -0.10 * np.sqrt(max(0.0, pdyn_persist))
                Q_fut = q_fut + q_comp_fut
                dst_fut = dst_fut * alpha + Q_fut * tau_dyn * (1.0 - alpha)
            forecast[f"{h}h"] = np.clip(dst_fut, -500, 50)

        self.results.update({
            'time': times,
            'HAC_total': hac_total,
            'dHAC_dt': dHAC_dt,
            'Bz': Bz,
            'Vsw': Vsw,
            'coupling_signal': coupling,
            'Dst_physical': dst_physical,
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
                       'final_level': final, 'escalation': sev_f > sev, 'severity': sev_f,
                       'confidence': confidence}

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

    def predict_storm_indicators(self, hac_values):
        print("\n🌍 Predizendo indicadores (com Nowcast físico)...")
        dst_pred = self.results.get('Dst_physical', np.zeros_like(hac_values))
        dst_min = self.results.get('Dst_min_physical', np.min(dst_pred))
        dst_now = self.results.get('Dst_now', dst_pred[-1])
        dHAC_dt = self.results.get('dHAC_dt', np.zeros_like(hac_values))
        Bz = self.results.get('Bz', np.zeros_like(hac_values))
        Vsw = self.results.get('Vsw', np.full_like(hac_values, 400))

        # Estimativa de Kp melhorada
        kp_from_dst = np.clip( 0.072 * abs(dst_min)**0.8 + 0.7, 0, 9)
        kp_pred = np.full_like(hac_values, kp_from_dst)

        storm_levels, logs = [], []
        esc_cnt = g4g5_cnt = 0
        for i in range(len(hac_values)):
            level, info = self._classify_storm_with_nowcast(hac_values[i], dHAC_dt[i], Bz[i], Vsw[i])
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
        self.results.update({'Kp_pred': kp_pred, 'Dst_pred': dst_pred,
                             'Dst_min': dst_min, 'Dst_now': dst_now,
                             'Storm_level': enhanced, 'Storm_level_base': storm_levels,
                             'Decision_logs': logs})
        self.classification_logs = logs

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

        # Probabilidades com thresholds provisórios
        recent_hac = np.max(hac_values[-48:])
        probs = {'G1': 0., 'G2': 0., 'G3': 0., 'G4': 0., 'G5': 0.}
        storm_score = max(recent_hac, 0.32 * abs(dst_min))
        if storm_score < self.config.HAC_G1:
            pass
        elif storm_score < self.config.HAC_G2:
            probs['G1'] = 1.0
        elif storm_score < self.config.HAC_G3:
            probs['G2'] = 1.0
        elif storm_score < self.config.HAC_G4:
            probs['G3'] = 1.0
        elif storm_score < self.config.HAC_G5:
            probs['G4'] = 1.0
        else:
            probs['G5'] = 1.0
        print("   • Probabilidades (baseadas no HAC):")
        for k, v in probs.items():
            print(f"       {k}: {v*100:.1f}%")

        return kp_pred, dst_pred, enhanced

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
        'Storm_level': model.results['Storm_level']
    })
    df_out.to_csv("hac_nowcast_results.csv", index=False)
    print("\n💾 Resultados salvos em hac_nowcast_results.csv")


if __name__ == "__main__":
    main()
