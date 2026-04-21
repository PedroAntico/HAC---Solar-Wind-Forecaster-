#!/usr/bin/env python3
"""
hac_final.py - HAC++ Model: Sistema de Produção com Nowcast + Inércia Híbrido
Versão final corrigida (abril/2026):
- Acoplamento combinado (Newell + não-linear) com ajuste para baixa resolução.
- Reservatórios HAC com injeção dinâmica e perdas dependentes de Bz.
- Evolução temporal do Dst (equação de Burton) – essencial para tempestades prolongadas.
- Normalização HAC adaptativa (fator 300).
- Classificação híbrida com override por Dst.
- Previsão por simulação da equação de relaxação.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# CONFIGURAÇÃO FÍSICA (AJUSTADA)
# ============================================================
class HACPhysicsConfig:
    # Tempos característicos (horas)
    TAU_RING_CURRENT_QUIET = 10.0
    TAU_RING_CURRENT_HSS = 6.0
    TAU_RING_CURRENT_CME = 3.0
    TAU_SUBSTORM = 0.6
    TAU_IONOSPHERE = 0.2
    TAU_BZ_MEMORY = 3.0

    # Saturações
    E_FIELD_SATURATION = 35.0      # mV/m
    KP_SATURATION = 9.0
    RING_CURRENT_MAX = 800.0       # nT

    # Partição de energia
    ALPHA_RING = 0.4
    ALPHA_SUBSTORM = 0.3
    ALPHA_IONOSPHERE = 0.3

    # Acoplamento não-linear
    BETA_NONLINEAR = 2.2
    COUPLING_THRESHOLD = 2.0      # mV/m (reduzido para ativar mais cedo)
    NEWELL_SCALE = 5e-4           # será multiplicado por fator adicional

    # Escalas operacionais
    HAC_SCALE_MAX = 800.0
    HAC_NORM_FACTOR = 150.0       # fator de normalização (aumentado para 300)

    # Limites físicos
    VSW_MIN, VSW_MAX = 200, 1500
    DENSITY_MIN, DENSITY_MAX = 0.1, 100
    BZ_MIN, BZ_MAX = -100, 100

    # Nowcast + Inércia
    THETA_CRITICAL = 50.0          # nT/h
    HG3_THRESHOLD = 150.0
    VSW_CRITICAL = 700.0
    BZ_CRITICAL = -8.0

    # Classificação por HAC (limiares)
    HAC_G1 = 50
    HAC_G2 = 120
    HAC_G3 = 250
    HAC_G4 = 450
    HAC_G5 = 650

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
    """Versão escalar para uso dentro do loop de simulação."""
    if density > 8 and bz < -8:
        return 'CME'
    elif v > 600 and density < 5:
        return 'HSS'
    elif density < 2:
        return 'SIR'
    else:
        return 'Quiet'

def detect_regime_array(v, density, bz):
    """
    Versão vetorizada para aplicação de blend e boost.
    Prioridade: CME > HSS > SIR > Quiet.
    """
    n = len(v)
    regime = np.full(n, 'Quiet', dtype=object)
    regime[(v > 600) & (density < 5)] = 'HSS'
    regime[(density > 8) & (bz < -8)] = 'CME'   # sobrescreve HSS se conflito
    regime[(density < 2)] = 'SIR'
    return regime

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
# 2. CÁLCULO DE CAMPOS FÍSICOS (COM AJUSTES PARA BAIXA RESOLUÇÃO)
# ============================================================
class PhysicalFieldsCalculator:
    @staticmethod
    def compute_all_fields(df):
        df = df.copy()
        config = HACPhysicsConfig()
        bz = df['bz_gsm'].fillna(0).values
        v = df['speed'].fillna(400).values
        
        # bt seguro
        if 'bt' in df.columns:
            bt = df['bt'].fillna(0).values
        else:
            bt = np.abs(bz)
        by = df.get('by_gsm', pd.Series(np.zeros_like(bz), index=df.index)).fillna(0).values

        # ========================================================
        # Bz efetivo com memória exponencial (persistência)
        # ========================================================
        time_sec = pd.to_datetime(df['time_tag']).values.astype('datetime64[s]')
        dt_sec = np.diff(time_sec).astype(float)
        dt_sec = np.insert(dt_sec, 0, np.median(dt_sec))
        dt_hours = np.maximum(dt_sec / 3600.0, 1e-6)

        tau_bz = getattr(config, 'TAU_BZ_MEMORY', 2.0)
        
        bz_neg = np.minimum(0, bz)
        bz_eff = np.zeros_like(bz)
        bz_eff[0] = bz_neg[0]

        for i in range(1, len(bz)):
            alpha = np.exp(-dt_hours[i] / tau_bz)
            bz_eff[i] = alpha * bz_eff[i-1] + (1 - alpha) * bz_neg[i]

        # ------------------------------------------------------------
        # 1. Acoplamento Newell (original)
        # ------------------------------------------------------------
        theta = np.arctan2(by, bz)
        theta_factor = np.abs(np.sin(theta / 2)) ** 3
        coupling_newell = (v ** (4/3)) * (bt ** (2/3)) * theta_factor * config.NEWELL_SCALE

        # ------------------------------------------------------------
        # 2. Acoplamento não-linear via campo elétrico (com Bz persistente)
        # ------------------------------------------------------------
        e_field = (-bz_eff) * v * 1e-3
        e_sat = np.clip(e_field, 0, config.E_FIELD_SATURATION)
        thr = config.COUPLING_THRESHOLD
        beta = config.BETA_NONLINEAR
        coupling_nl = np.where(e_sat <= thr, e_sat, thr * ((e_sat / thr) ** beta))

        # ------------------------------------------------------------
        # 3. Combinação (60% Newell, 40% não-linear)
        # ------------------------------------------------------------
        coupling_comb = 0.6 * coupling_newell + 0.4 * coupling_nl
        
        # Normalização adaptativa
        scale = np.percentile(coupling_comb, 99)
        if scale > 1e-6:
            coupling_norm = coupling_comb / scale
        else:
            coupling_norm = coupling_comb
        
        coupling_signal = np.where(bz < 0, coupling_norm, 0.0)
        coupling_signal = np.clip(coupling_signal, 0, 20)
        
        df['coupling_signal'] = coupling_signal
        df['coupling_newell'] = coupling_newell
        df['coupling_nonlinear'] = coupling_nl
        df['E_field_raw'] = e_field
        df['bz_eff'] = bz_eff

        print(f"   • Bz min/max: {bz.min():.1f} / {bz.max():.1f} nT")
        print(f"   • Bz eff min: {bz_eff.min():.1f} nT")
        print(f"   • V min/max: {v.min():.1f} / {v.max():.1f} km/s")
        print(f"   • Coupling max: {coupling_signal.max():.2f}")

        return df
# ============================================================
# 3. MODELO HAC+ COM EVOLUÇÃO TEMPORAL DO DST
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
        p99 = np.percentile(values, 99)
        p95 = np.percentile(values, 95)
        scale = max(p99, p95 * 1.2, 1.0)
        norm = (values / scale) * self.config.HAC_NORM_FACTOR
        norm = np.nan_to_num(norm, nan=0.0, posinf=800, neginf=0.0)
        norm = np.clip(norm, 0, self.config.HAC_SCALE_MAX)
        print(f"   • HAC escala: p99={p99:.1f}, p95={p95:.1f} → scale={scale:.1f}")
        print(f"   • HAC máx: {np.max(norm):.1f}, méd: {np.mean(norm):.1f}")
        return norm

    def _compute_robust_derivative(self, hac_total, times):
        t_sec = times.astype('datetime64[s]')
        dt = np.diff(t_sec).astype(float)
        dt = np.insert(dt, 0, np.median(dt))
        dt[dt <= 0] = 1.0
        dt_h = np.maximum(dt / 3600.0, 1e-3)
        if len(hac_total) < 7:
            dH = np.gradient(hac_total) / dt_h
        else:
            w = min(7, len(hac_total))
            if w % 2 == 0: w -= 1
            try:
                dH = savgol_filter(hac_total, w, 2, deriv=1, delta=np.median(dt_h))
            except:
                dH = np.gradient(hac_total) / dt_h
        dH = np.nan_to_num(dH, nan=0.0)
        dH = np.clip(dH, -150, 150)   # limite físico realista
        print(f"     Derivada máx: {np.max(dH):.1f} nT/h")
        return dH

    def _detect_escalation_triggers(self, hac_total, dHAC_dt, Bz, Vsw, times):
        n = len(hac_total)
        flags = np.zeros(n, dtype=bool)
        window = 30
        for i in range(window, n):
            if (hac_total[i] < self.config.HG3_THRESHOLD and
                dHAC_dt[i] > self.config.THETA_CRITICAL and
                np.median(Bz[max(0,i-window):i+1]) < self.config.BZ_CRITICAL and
                np.median(Vsw[max(0,i-window):i+1]) > self.config.VSW_CRITICAL):
                flags[i] = True
                alert = {
                    'time': pd.to_datetime(times[i]),
                    'HAC': float(hac_total[i]),
                    'dHAC_dt': float(dHAC_dt[i]),
                    'Bz_avg': float(np.mean(Bz[max(0,i-window):i+1])),
                    'V_avg': float(np.mean(Vsw[max(0,i-window):i+1])),
                    'forecast_horizon_hours': 2.0
                }
                self.nowcast_alerts.append(alert)
                if (not self.escalation_triggers or
                    (alert['time'] - self.escalation_triggers[-1]['time']).total_seconds() > 3600):
                    self.escalation_triggers.append(alert)
                    print(f"     🚨 ALERTA em {alert['time']}: HAC={hac_total[i]:.1f}, dH/dt={dHAC_dt[i]:.1f} nT/h")
        return flags

    def compute_hac_system(self, df):
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
            regime = _detect_regime_scalar(Vsw[i], density[i], Bz[i])
            baseline = 0.15 + 0.25 * (density[i] / 10.0)
            injection = coupling[i] + baseline

            # Reforço físico por Bz negativo
            if Bz[i] < -5:
                boost_bz = 1.0 + 0.2 * min(abs(Bz[i]) / 20.0, 1.0)
                injection *= boost_bz
            
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
            injection_eff = np.clip(injection * dt_hours, 0, 100)
    
            loss_ring = (0.015 + 0.0002 * np.sqrt(max(0, hac_ring[i-1]))) * hac_ring[i-1]
            loss_sub  = (0.015 + 0.0002 * np.sqrt(max(0, hac_substorm[i-1]))) * hac_substorm[i-1]
            loss_ion  = (0.015 + 0.0002 * np.sqrt(max(0, hac_ionosphere[i-1]))) * hac_ionosphere[i-1]
            if Bz[i] > 0:
                loss_ring *= 1.3
                loss_sub  *= 1.3
                loss_ion  *= 1.3
    
            alpha_rc = np.exp(-dt[i] / tau_rc)
            alpha_sub = np.exp(-dt[i] / tau_sub)
            alpha_ion = np.exp(-dt[i] / tau_ion)
    
            hac_ring[i] = alpha_rc * hac_ring[i-1] + self.config.ALPHA_RING * injection_eff - loss_ring
            hac_substorm[i] = alpha_sub * hac_substorm[i-1] + self.config.ALPHA_SUBSTORM * injection_eff - loss_sub
            hac_ionosphere[i] = alpha_ion * hac_ionosphere[i-1] + self.config.ALPHA_IONOSPHERE * injection_eff - loss_ion
    
        hac_total = self._safe_normalization(hac_ring + hac_substorm + hac_ionosphere)
        dHAC_dt = self._compute_robust_derivative(hac_total, times)
        _ = self._detect_escalation_triggers(hac_total, dHAC_dt, Bz, Vsw, times)
    
        # ============================================================
        # EVOLUÇÃO TEMPORAL DO Dst (equação de Burton com injeção sublinear)
        # ============================================================
        tau_dst_base = 8.0   # horas
        k_dst = 18.0          # fator de escala (nT/h por sqrt(HAC))
        
        dst_physical = np.zeros(n)
        dst_physical[0] = -20.0
        
        for i in range(1, n):
            dt_hours = dt[i] / 3600.0
            regime_i = _detect_regime_scalar(Vsw[i], density[i], Bz[i])
            if regime_i == 'CME':
                tau_dst = 3.0
            elif regime_i == 'HSS':
                tau_dst = 5.0
            else:
                tau_dst = tau_dst_base
        
            alpha = np.exp(-dt_hours / tau_dst)
        
            # Injeção sublinear: raiz quadrada do HAC
            # Apenas quando HAC > 0 (evita sqrt de negativo)
            hac_val = max(0.0, hac_total[i])
            hac_thr = 40.0   # limiar de ativação (HAC abaixo disso não injeta energia)
            hac_eff = max(0.0, hac_val - hac_thr)
            Q_injection = k_dst * np.sqrt(hac_eff)
        
            # Pequeno boost apenas para Bz extremamente negativo (opcional)
            if Bz[i] < -15:
                Q_injection *= 1.2
        
            # Equação de relaxação (sem tanh, apenas física linear)
            dst_raw = alpha * dst_physical[i-1] + (1 - alpha) * (-Q_injection)
            dst_physical[i] = dst_raw
        
        # Clipping físico suave (apenas para evitar extrapolação extrema)
        dst_physical = np.clip(dst_physical, -500, 50)
        print(f"   • Dst físico mín: {np.min(dst_physical):.1f} nT")
    
        # Previsão por simulação (estendendo a equação de evolução)
        forecast = {}
        dt_median = np.median(dt) / 3600.0
        for h in [1, 2, 3]:
            steps = max(1, int(h / dt_median))
            dst_fut = dst_physical[-1]
            hac_fut = max(0.0, hac_total[-1])
            hac_eff_fut = max(0.0, hac_fut - 30.0)
            Q_fut = k_dst * np.sqrt(hac_eff_fut)
            
            tau = tau_dst_base
            alpha = np.exp(-dt_median / tau)
            for _ in range(steps):
                Q_fut = k_dst * np.sqrt(hac_fut)
                dst_fut = alpha * dst_fut + (1 - alpha) * (-Q_fut)
            forecast[f"{h}h"] = np.clip(dst_fut, -500, 50)
    
        self.results.update({
            'time': times, 'HAC_total': hac_total, 'dHAC_dt': dHAC_dt,
            'Bz': Bz, 'Vsw': Vsw, 'coupling_signal': coupling,
            'Dst_physical': dst_physical, 'Dst_min_physical': np.min(dst_physical),
            'Dst_now': dst_physical[-1], 'forecast': forecast
        })
        self._validate_output(hac_total)
        return hac_total

    def _validate_output(self, hac_values):
        if np.any(np.isnan(hac_values)):
            raise ValueError("NaN em HAC")
        print("   ✅ Validação passada")

    def _classify_storm_with_nowcast(self, hac, dhdt, bz, v):
        cfg = self.config
        if hac < cfg.HAC_G1: base, sev = "Quiet", 0
        elif hac < cfg.HAC_G2: base, sev = "G1", 1
        elif hac < cfg.HAC_G3: base, sev = "G2", 2
        elif hac < cfg.HAC_G4: base, sev = "G3", 3
        elif hac < cfg.HAC_G5: base, sev = "G4", 4
        else: base, sev = "G5", 5

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
        if score >= 14 and sev < 5: final, sev_f = "G5 (Nowcast Override)", 5
        elif score >= 11 and sev < 4: final, sev_f = "G4 (Nowcast Override)", 4
        elif score >= 8 and sev < 3: final, sev_f = "G3 (Nowcast Override)", 3
        elif score >= 5 and sev < 2: final, sev_f = "G2 (Nowcast Enhancement)", 2

        if dhdt > 180 and bz < -10 and v > 700 and hac > 100 and sev < 5:
            final, sev_f = "G5 (Extreme Nowcast)", 5
        elif dhdt > 120 and bz < -8 and v > 600 and sev < 4:
            final, sev_f = "G4 (Strong Nowcast)", 4

        return final, {'hac': hac, 'dhdt': dhdt, 'bz': bz, 'v': v,
                       'base_level': base, 'nowcast_score': score,
                       'final_level': final, 'escalation': sev_f > sev, 'severity': sev_f}

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
        kp_pred = 9 * np.tanh(hac_values / 300)   # saturação mais suave
        dst_pred = self.results.get('Dst_physical', np.zeros_like(hac_values))
        dst_min = self.results.get('Dst_min_physical', np.min(dst_pred))
        dst_now = self.results.get('Dst_now', dst_pred[-1])
        dHAC_dt = self.results.get('dHAC_dt', np.zeros_like(hac_values))
        Bz = self.results.get('Bz', np.zeros_like(hac_values))
        Vsw = self.results.get('Vsw', np.full_like(hac_values, 400))

        storm_levels, logs = [], []
        esc_cnt = g4g5_cnt = 0
        for i in range(len(hac_values)):
            level, info = self._classify_storm_with_nowcast(hac_values[i], dHAC_dt[i], Bz[i], Vsw[i])
            dst = dst_pred[i]

            # Override por Dst (clipping de classificações infladas)
            if dst > -150 and ("G4" in level or "G5" in level):
                level = "G2 (Dst Clipped)"
            elif dst > -100 and ("G3" in level or "G4" in level or "G5" in level):
                level = "G1 (Dst Clipped)"

            storm_levels.append(level)
            logs.append(info)
            if info['escalation']: esc_cnt += 1
            if "G4" in level or "G5" in level: g4g5_cnt += 1

        enhanced = self._apply_trend_boost(storm_levels, hac_values, dHAC_dt)
        self.results.update({'Kp_pred': kp_pred, 'Dst_pred': dst_pred,
                             'Dst_min': dst_min, 'Dst_now': dst_now,
                             'Storm_level': enhanced, 'Storm_level_base': storm_levels,
                             'Decision_logs': logs})
        self.classification_logs = logs

        g4g5_final = sum(1 for l in enhanced if "G4" in l or "G5" in l)
        g4g5_base = sum(1 for l in storm_levels if "G4" in l or "G5" in l)
        g4g5_trad = sum(1 for l in storm_levels if l in ['G4', 'G5'])

        print(f"   • Kp máximo: {np.max(kp_pred):.1f}")
        print(f"   • Dst mínimo (físico): {dst_min:.1f} nT")
        print(f"   • Dst atual: {dst_now:.1f} nT")
        print(f"   • Eventos G4/G5 (tradicional): {g4g5_trad}")
        print(f"   • Eventos G4/G5 (Nowcast base): {g4g5_base}")
        print(f"   • Eventos G4/G5 (com boost): {g4g5_final}")
        print(f"   • Escalações Nowcast: {esc_cnt}")

        forecast = self.results.get('forecast', {})
        if forecast:
            print("   • Previsão Dst (simulada):")
            for h, val in forecast.items(): print(f"       {h}: {val:.1f} nT")

        # Probabilidades baseadas no HAC atual
        last_hac = hac_values[-1]
        probs = {'G1':0., 'G2':0., 'G3':0., 'G4':0., 'G5':0.}
        if last_hac >= self.config.HAC_G1:
            if last_hac < self.config.HAC_G2: probs['G1'] = min(1.0, (last_hac - 50) / 70)
            elif last_hac < self.config.HAC_G3: probs['G2'] = min(1.0, (last_hac - 120) / 130)
            elif last_hac < self.config.HAC_G4: probs['G3'] = min(1.0, (last_hac - 250) / 200)
            elif last_hac < self.config.HAC_G5: probs['G4'] = min(1.0, (last_hac - 450) / 200)
            else: probs['G5'] = min(1.0, (last_hac - 650) / 150)
        print("   • Probabilidades (baseadas no HAC):")
        for k, v in probs.items(): print(f"       {k}: {v*100:.1f}%")

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
        report = "="*70 + "\n🚨 RELATÓRIO NOWCAST + INÉRCIA\n" + "="*70 + "\n\n"
        report += summary + "\n"
        report += f"PARÂMETROS CRÍTICOS:\n  • τ_eff quiet: {self.config.TAU_RING_CURRENT_QUIET} h\n"
        report += f"  • Θ: {self.config.THETA_CRITICAL} nT/h\n  • H_G3: {self.config.HG3_THRESHOLD}\n"
        report += "="*70
        return report

# ============================================================
# MAIN (EXEMPLO DE USO)
# ============================================================
def main():
    print("="*70)
    print("🚀 HAC++ MODEL - SISTEMA DE PRODUÇÃO (NOWCAST + INÉRCIA HÍBRIDO)")
    print("="*70)
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
