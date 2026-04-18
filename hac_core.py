"""
hac_core.py – Núcleo unificado do modelo HAC++ (VERSÃO FINAL CONSOLIDADA)

Aperfeiçoamentos aplicados:
- Acoplamento unificado: pode receber coupling_override (escala 0–100).
- Previsão Dst com Q_effective limitado e sem multiplicação extra por tau.
- Probabilidades baseadas em limiares físicos (ReLU-like) com corte para quietude.
- Injeção contínua dependente de Bz (residual 10% para Bz norte).
- Perda proporcional à raiz quadrada do HAC (mais realista).
- Decaimento linear (1 - dt/tau) para evitar dupla dissipação.
- Saturação suave do HAC: HAC / (1 + HAC/800).
- Calibração de HAC_REF via percentil 99 e Q_FACTOR via regressão linear.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple, Optional, List

# ------------------------------------------------------------
# CONFIGURAÇÃO FÍSICA (VALORES CALIBRÁVEIS)
# ------------------------------------------------------------
class HACCoreConfig:
    # Referência para normalização do HAC (calibrada via dados)
    HAC_REF = 1_000_000.0          # placeholder
    HAC_SCALE_MAX = 500.0          # escala de saída (0-500)

    # Parâmetros do modelo físico de Dst
    TAU_DST = 7 * 3600             # 7 horas (decaimento em segundos)
    DST_Q = -20.0                  # baseline de quiet time (nT)
    Q_FACTOR = -0.7                # fator de injeção (calibrado globalmente)

    # Saturação do campo elétrico
    E_FIELD_SATURATION = 25.0      # mV/m

    # Limiares Dst para classificação
    DST_G1 = -50
    DST_G2 = -100
    DST_G3 = -150
    DST_G4 = -200
    DST_G5 = -300

    # Freios físicos para classificação
    BZ_FREIO_MEDIAN = -5.0         # se mediana recente > -5, reduz 2 níveis
    BZ_PERSISTENCE_THRESH = -6.0   # se média recente > -6 e severidade>=3, reduz 1 nível

    # Pesos para probabilidade (soma = 1.0)
    W_HAC = 0.25
    W_BZ = 0.35
    W_V = 0.20
    W_DHDT = 0.20

    # Limiares de ativação (abaixo destes valores a contribuição é zero)
    LIM_HAC = 30.0
    LIM_BZ = -4.0
    LIM_V = 400.0
    LIM_DHDT = 20.0

    # Fatores de escala para normalização das contribuições
    SCALE_HAC = 120.0
    SCALE_BZ = 12.0
    SCALE_V = 350.0
    SCALE_DHDT = 100.0

    # Limites físicos para dH/dt
    DHDT_MIN = -500.0
    DHDT_MAX = 500.0

    # Janela para médias recentes (minutos)
    RECENT_WINDOW_MINUTES = 30


# ------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax estável numericamente."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ------------------------------------------------------------
# 1. CÁLCULO DO HAC (COM PRESSÃO DINÂMICA E INJEÇÃO CONTÍNUA)
# ------------------------------------------------------------
def compute_hac(
    time: np.ndarray,
    bz: np.ndarray,
    v: np.ndarray,
    density: Optional[np.ndarray] = None,
    config: Optional[HACCoreConfig] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Calcula o HAC total usando acoplamento Newell-like:
    coupling = (E_sat ** 1.5) * sqrt(density) * (v / 400)
    A injeção é contínua em função de Bz, com residual de 10% para Bz norte.
    """
    if config is None:
        config = HACCoreConfig()

    bz = np.nan_to_num(bz, nan=0.0)
    v = np.nan_to_num(v, nan=400.0)
    if density is None:
        density = np.full_like(bz, 5.0)
    else:
        density = np.nan_to_num(density, nan=5.0)
    density = np.maximum(density, 0.1)

    # Campo elétrico (apenas Bz sul)
    bz_south = np.maximum(0, -bz)
    e_field = bz_south * v * 1e-3                     # mV/m
    e_field_sat = np.clip(e_field, 0, config.E_FIELD_SATURATION)

    # Acoplamento base
    coupling = (e_field_sat ** 1.5) * np.sqrt(density) * (v / 400.0)

    # Tempos característicos (segundos)
    tau_ring = 2.0 * 3600
    tau_sub = 1.0 * 3600
    tau_ion = 0.3 * 3600

    alpha_ring = 0.4
    alpha_sub = 0.3
    alpha_ion = 0.3

    # dt em segundos
    if isinstance(time, (pd.DatetimeIndex, pd.Series)):
        time_arr = pd.to_datetime(time).values
    else:
        time_arr = time
    dt = np.zeros_like(coupling)
    dt[1:] = (time_arr[1:] - time_arr[:-1]).astype('timedelta64[s]').astype(float)
    dt[0] = dt[1] if len(dt) > 1 else 60.0
    dt = np.maximum(dt, 1.0)

    n = len(coupling)
    hac_ring = np.zeros(n)
    hac_sub = np.zeros(n)
    hac_ion = np.zeros(n)

    # Loop de atualização com injeção contínua e perda sqrt
    for i in range(1, n):
        # Decaimento linear (evita dupla dissipação)
        decay_rc = min(dt[i] / tau_ring, 1.0) if dt[i] > 0 else 0.0
        decay_sb = min(dt[i] / tau_sub, 1.0)   if dt[i] > 0 else 0.0
        decay_io = min(dt[i] / tau_ion, 1.0)   if dt[i] > 0 else 0.0

        # Injeção contínua dependente de Bz
        injection = coupling[i]
        if bz[i] < 0:
            e_field_inst = -bz[i] * v[i] * 1e-3
            e_field_clipped = np.clip(e_field_inst, 0, config.E_FIELD_SATURATION)
            injection += 6.0 * e_field_clipped * (1.0 + 0.1 * abs(bz[i]))
        else:
            injection *= 0.1   # residual para Bz norte (não zera)

        dt_hours = dt[i] / 3600.0
        injection_eff = injection * dt_hours
        injection_eff = np.clip(injection_eff, 0, 80)

        # Perda dependente da raiz quadrada do HAC
        loss_factor_ring = 0.015 + 0.0002 * np.sqrt(hac_ring[i-1])
        loss_ring = loss_factor_ring * hac_ring[i-1]

        loss_factor_sub = 0.015 + 0.0002 * np.sqrt(hac_sub[i-1])
        loss_sub = loss_factor_sub * hac_sub[i-1]

        loss_factor_ion = 0.015 + 0.0002 * np.sqrt(hac_ion[i-1])
        loss_ion = loss_factor_ion * hac_ion[i-1]

        # Atualização sequencial: decaimento → injeção → perda
        hac_ring[i] = hac_ring[i-1] * (1.0 - decay_rc)
        hac_ring[i] += alpha_ring * injection_eff
        hac_ring[i] -= loss_ring

        hac_sub[i] = hac_sub[i-1] * (1.0 - decay_sb)
        hac_sub[i] += alpha_sub * injection_eff
        hac_sub[i] -= loss_sub

        hac_ion[i] = hac_ion[i-1] * (1.0 - decay_io)
        hac_ion[i] += alpha_ion * injection_eff
        hac_ion[i] -= loss_ion

        # Garantir não-negatividade
        hac_ring[i] = max(0.0, hac_ring[i])
        hac_sub[i]  = max(0.0, hac_sub[i])
        hac_ion[i]  = max(0.0, hac_ion[i])

    hac_total_raw = hac_ring + hac_sub + hac_ion

    # Saturação suave (preserva extremos)
    hac_total = hac_total_raw / (1.0 + hac_total_raw / 2000.0)

    # Normalização com HAC_REF (fixo após calibração)
    hac_total = hac_total / config.HAC_REF * config.HAC_SCALE_MAX
    hac_total = np.clip(hac_total, 0, config.HAC_SCALE_MAX)

    components = {
        'ring': hac_ring,
        'substorm': hac_sub,
        'ionosphere': hac_ion,
        'coupling': coupling,
        'raw': hac_total_raw,
        'e_field_sat': e_field_sat
    }
    return hac_total, components


# ------------------------------------------------------------
# 2. DERIVADA dH/dt (SUAVIZADA E COM CLIPPING)
# ------------------------------------------------------------
def compute_dHdt(hac: np.ndarray, time: np.ndarray, config: Optional[HACCoreConfig] = None) -> np.ndarray:
    if config is None:
        config = HACCoreConfig()

    if isinstance(time, (pd.DatetimeIndex, pd.Series)):
        time_arr = pd.to_datetime(time).values
    else:
        time_arr = time

    dt_hours = np.zeros_like(hac)
    dt_hours[1:] = (time_arr[1:] - time_arr[:-1]).astype('timedelta64[s]').astype(float) / 3600.0
    dt_hours[0] = dt_hours[1] if len(dt_hours) > 1 else 1/60.0
    dt_hours = np.maximum(dt_hours, 1e-3)

    if len(hac) >= 5:
        window = min(5, len(hac))
        if window % 2 == 0:
            window -= 1
        hac_smooth = savgol_filter(hac, window_length=window, polyorder=2)
    else:
        hac_smooth = hac

    dhdt = np.gradient(hac_smooth) / dt_hours
    dhdt = np.clip(dhdt, config.DHDT_MIN, config.DHDT_MAX)
    return dhdt


# ------------------------------------------------------------
# 3. MODELO FÍSICO DE Dst (COM BASELINE DST_Q)
# ------------------------------------------------------------
def compute_dst_physical(
    coupling: np.ndarray,
    dt: np.ndarray,
    tau: float = 7 * 3600,
    Q_factor: float = -0.7,
    Dst_q: float = -20.0
) -> np.ndarray:
    dst = np.full_like(coupling, Dst_q)
    for i in range(1, len(coupling)):
        Q = Q_factor * coupling[i]
        dst[i] = dst[i-1] + (Q - (dst[i-1] - Dst_q)/tau) * dt[i]
    return dst


class DstPhysicalCalibrator:
    def __init__(self, tau: float = 7*3600, Dst_q: float = -20.0):
        self.tau = tau
        self.Dst_q = Dst_q
        self.Q_factor = None
        self.intercept = None

    def fit(self, coupling: np.ndarray, dt: np.ndarray, dst_obs: np.ndarray):
        integrated = np.full_like(coupling, self.Dst_q)
        for i in range(1, len(coupling)):
            integrated[i] = integrated[i-1] + coupling[i] * dt[i] - (integrated[i-1] - self.Dst_q)/self.tau * dt[i]

        mask = ~(np.isnan(integrated) | np.isnan(dst_obs))
        if np.sum(mask) < 10:
            raise ValueError("Dados insuficientes para calibração")

        reg = LinearRegression(fit_intercept=True)
        reg.fit(integrated[mask].reshape(-1, 1), dst_obs[mask])
        self.Q_factor = reg.coef_[0]
        self.intercept = reg.intercept_
        return self

    def predict(self, coupling: np.ndarray, dt: np.ndarray) -> np.ndarray:
        if self.Q_factor is None:
            raise ValueError("Modelo não calibrado. Execute .fit() primeiro.")
        dst = compute_dst_physical(coupling, dt, tau=self.tau,
                                   Q_factor=self.Q_factor, Dst_q=self.Dst_q)
        if self.intercept is not None:
            dst += self.intercept
        return dst


# ------------------------------------------------------------
# 4. CLASSIFICAÇÃO DE SEVERIDADE (COM FREIOS REFORÇADOS)
# ------------------------------------------------------------
def classify_storm_severity(
    dst_value: float,
    bz_recent: np.ndarray,
    config: Optional[HACCoreConfig] = None
) -> int:
    if config is None:
        config = HACCoreConfig()

    severity = 0
    if dst_value <= config.DST_G1: severity = 1
    if dst_value <= config.DST_G2: severity = 2
    if dst_value <= config.DST_G3: severity = 3
    if dst_value <= config.DST_G4: severity = 4
    if dst_value <= config.DST_G5: severity = 5

    if severity > 0 and len(bz_recent) > 0:
        # Freio principal: mediana > -5 → reduz 2 níveis
        if np.median(bz_recent) > config.BZ_FREIO_MEDIAN:
            severity = max(0, severity - 2)
        # Freio adicional: persistência de Bz não tão negativo
        if severity >= 3 and np.mean(bz_recent) > config.BZ_PERSISTENCE_THRESH:
            severity -= 1

    return severity


# ------------------------------------------------------------
# 5. PROBABILIDADES G1–G5 (ATIVAÇÃO POR LIMIAR + CORTE QUIETUDE)
# ------------------------------------------------------------
def storm_probability(
    hac: float,
    dhdt: float,
    bz: float,
    v: float,
    config: Optional[HACCoreConfig] = None
) -> Dict[str, float]:
    """
    Calcula probabilidades usando contribuições que só ativam acima de limiares físicos.
    Em condições muito quietas (score < 0.8), retorna 100% G1.
    """
    if config is None:
        config = HACCoreConfig()

    # Contribuições individuais (ReLU-like)
    contrib_hac  = config.W_HAC  * np.maximum(0, hac - config.LIM_HAC) / config.SCALE_HAC
    contrib_bz   = config.W_BZ   * np.maximum(0, -bz - abs(config.LIM_BZ)) / config.SCALE_BZ
    contrib_v    = config.W_V    * np.maximum(0, v - config.LIM_V) / config.SCALE_V
    contrib_dhdt = config.W_DHDT * np.maximum(0, dhdt - config.LIM_DHDT) / config.SCALE_DHDT

    score = contrib_hac + contrib_bz + contrib_v + contrib_dhdt
    score = np.clip(score, 0.0, 4.0)

    # Corte para quietude: se score < 0.8, retorna G1 dominante
    if score < 0.3:
        return {"G1": 1.0, "G2": 0.0, "G3": 0.0, "G4": 0.0, "G5": 0.0}

    # Escores para cada nível (não normalizados)
    level_scores = np.array([
        1.0 - score/4.0,          # G1
        score/4.0,                # G2
        (score - 1.0) / 3.0,      # G3
        (score - 2.0) / 2.0,      # G4
        (score - 3.0) / 1.0       # G5
    ])
    level_scores = np.maximum(level_scores, 0.0)

    probs = softmax(level_scores)

    return {
        "G1": probs[0],
        "G2": probs[1],
        "G3": probs[2],
        "G4": probs[3],
        "G5": probs[4]
    }


# ------------------------------------------------------------
# 6. MÉTRICAS CIENTÍFICAS
# ------------------------------------------------------------
def evaluate_event(
    time: np.ndarray,
    dst_obs: np.ndarray,
    dst_pred: np.ndarray
) -> Dict[str, float]:
    mask = ~(np.isnan(dst_obs) | np.isnan(dst_pred))
    if np.sum(mask) < 2:
        return {"correlation": np.nan, "MAE": np.nan,
                "peak_time_error_min": np.nan, "min_Dst_error_nT": np.nan}

    obs = dst_obs[mask]
    pred = dst_pred[mask]
    t = time[mask]

    corr, _ = stats.pearsonr(obs, pred)
    mae = np.mean(np.abs(obs - pred))

    idx_obs_min = np.argmin(obs)
    idx_pred_min = np.argmin(pred)

    if isinstance(t[0], np.datetime64):
        peak_time_error = (t[idx_pred_min] - t[idx_obs_min]).astype('timedelta64[s]').astype(float) / 60.0
    else:
        peak_time_error = 0.0

    min_dst_error = pred[idx_pred_min] - obs[idx_obs_min]

    return {
        "correlation": corr,
        "MAE": mae,
        "peak_time_error_min": peak_time_error,
        "min_Dst_error_nT": min_dst_error
    }


# ------------------------------------------------------------
# 7. CLASSE UNIFICADA (COM PREVISÃO FÍSICA CORRIGIDA)
# ------------------------------------------------------------
class HACCoreModel:
    def __init__(self, config: Optional[HACCoreConfig] = None):
        self.config = config or HACCoreConfig()
        self.calibrator = DstPhysicalCalibrator(tau=self.config.TAU_DST,
                                                Dst_q=self.config.DST_Q)
        self.results = {}

    def calibrate_hac_ref(self, time, bz, v, density=None):
        """
        Calibra HAC_REF usando percentil 99 do HAC bruto.
        """
        _, comp = compute_hac(time, bz, v, density, self.config)
        hac_raw = comp['raw']
        self.config.HAC_REF = np.percentile(hac_raw, 99)
        print(f"HAC_REF calibrado: {self.config.HAC_REF:.2f}")
        return self.config.HAC_REF

    def fit_calibration(self, coupling: np.ndarray, dt: np.ndarray, dst_obs: np.ndarray):
        self.calibrator.fit(coupling, dt, dst_obs)
        self.config.Q_FACTOR = self.calibrator.Q_factor
        print(f"Q_factor calibrado: {self.config.Q_FACTOR:.4f}")

    def process(self, time, bz, v, density=None, coupling_override=None, mode='event') -> Dict:
        # 1. HAC
        hac, components = compute_hac(time, bz, v, density, self.config)

        # 2. dt
        if isinstance(time, (pd.Series, pd.DatetimeIndex)):
            time_arr = pd.to_datetime(time).values
        else:
            time_arr = time
        dt = np.zeros_like(hac)
        dt[1:] = (time_arr[1:] - time_arr[:-1]).astype('timedelta64[s]').astype(float)
        dt[0] = dt[1] if len(dt) > 1 else 60.0
        dt = np.maximum(dt, 1.0)

        # 3. dH/dt
        dhdt = compute_dHdt(hac, time_arr, self.config)

        # 4. Acoplamento unificado
        if coupling_override is not None:
            coupling = np.clip(coupling_override, 0, 100)
        else:
            coupling = components['coupling']
            coupling = np.clip(coupling, 0, 100)

        # 5. Dst físico
        if self.calibrator.Q_factor is not None:
            dst_pred = self.calibrator.predict(coupling, dt)
        else:
            dst_pred = compute_dst_physical(coupling, dt, self.config.TAU_DST,
                                            self.config.Q_FACTOR, self.config.DST_Q)

        # 6. Janela recente (últimos 30 min)
        window_minutes = self.config.RECENT_WINDOW_MINUTES
        if isinstance(time_arr[-1], np.datetime64):
            recent_mask = (time_arr[-1] - time_arr) <= np.timedelta64(window_minutes, 'm')
        else:
            n_recent = min(window_minutes, len(time_arr))
            recent_mask = np.zeros(len(time_arr), dtype=bool)
            recent_mask[-n_recent:] = True

        # 7. Classificação de severidade
        if mode == 'nowcast':
            dst_for_class = dst_pred[-1]
            bz_recent = bz[recent_mask] if np.any(recent_mask) else bz[-30:] if len(bz)>=30 else bz
        else:
            dst_for_class = np.min(dst_pred)
            bz_recent = bz

        severity = classify_storm_severity(dst_for_class, bz_recent, self.config)

        # 8. Probabilidades (usando médias recentes para nowcast)
        if mode == 'nowcast' and np.any(recent_mask):
            hac_val = np.mean(hac[recent_mask])
            dhdt_val = np.mean(dhdt[recent_mask])
            bz_val = np.mean(bz[recent_mask])
            v_val = np.mean(v[recent_mask])
        else:
            idx_prob = -1 if mode == 'nowcast' else np.argmin(dst_pred)
            hac_val = hac[idx_prob]
            dhdt_val = dhdt[idx_prob]
            bz_val = bz[idx_prob]
            v_val = v[idx_prob]

        probs = storm_probability(hac_val, dhdt_val, bz_val, v_val, self.config)

        # 9. Previsão de Dst (solução analítica com Q_effective limitado)
        tau = float(self.config.TAU_DST)
        Dst_q = self.config.DST_Q
        coupling_last = np.clip(coupling[-1], 0, 200)
        Q_effective = self.config.Q_FACTOR * coupling_last
        Q_effective = np.clip(Q_effective, -200, 0)   # limite suave

        Dst_now = dst_pred[-1]
        decay_term = Dst_now - Dst_q

        forecast = {}
        for h in [1, 2, 3]:
            dt_forecast = h * 3600.0
            exp_term = np.exp(-dt_forecast / tau)
            dst_future = Dst_q + decay_term * exp_term + Q_effective * (1 - exp_term)
            forecast[f'{h}h'] = np.clip(dst_future, -500, 50)

        self.results = {
            'time': time_arr,
            'HAC': hac,
            'dHdt': dhdt,
            'Dst_pred': dst_pred,
            'Dst_min': np.min(dst_pred),
            'Dst_now': Dst_now,
            'dDst_dt': (Q_effective - (Dst_now - Dst_q)/tau) * 3600.0,
            'forecast': forecast,
            'severity': severity,
            'probabilities': probs,
            'components': components
        }
        return self.results
