#!/usr/bin/env python3
"""
validate_events.py - Validação HAC++ com alinhamento causal e calibração robusta
                    (otimizado para dados OMNI de 1 minuto)

Alterações para alta resolução:
- Lag de propagação fixo (20 min) para preservar extremos.
- Interpolação estendida (linear + preenchimento) para lidar com NaNs.
- Derivada do Dst calculada com dt real (não assume 1 hora).
- Reamostragem opcional para 5 min para acelerar processamento.
- Diagnóstico de extremos após alinhamento.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from hac_final import (
    ProductionHACModel,
    PhysicalFieldsCalculator,
    HACPhysicsConfig,
    normalize_omni_columns
)
from hac_core import HACCoreModel, HACCoreConfig, evaluate_event, compute_hac

# =========================
# CONFIGURAÇÕES
# =========================
EVENTS = {
    "Halloween_2003": ("2003-10-28", "2003-11-02"),
    "St_Patrick_2015": ("2015-03-17", "2015-03-20"),
    "Carrington_like_2012": ("2012-07-22", "2012-07-25"),
}

TRAIN_END_DATE = "2014-12-31"

OMNI_FILE = "omni_1min_2000_2020.csv"   # arquivo de alta resolução
DST_FILE = "data/dst_kyoto_2000_2020.csv"
CALIBRATION_FILE = "hac_calibration.json"

# Parâmetros para dados de 1 minuto
LAG_MINUTES = 20                        # lag fixo de propagação
WINDOW_FORECAST_MIN = 120               # 2h para persistência do forecast
WINDOW_HAC_MEMORY = 720                 # 12h para memória longa do HAC
RESAMPLE_MINUTES = 5                    # reamostragem para acelerar (None para desabilitar)

# =========================
# CARREGAMENTO DOS DADOS (COM INTERPOLAÇÃO ROBUSTA)
# =========================
def load_dst_kyoto(filepath):
    """Carrega Dst do Kyoto, garante datetime e filtra valores absurdos."""
    print(f"\n📥 Carregando DST: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    # Garantir time_tag como datetime
    if 'time_tag' not in df.columns:
        if 'date' in df.columns and 'time' in df.columns:
            df['time_tag'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        else:
            df['time_tag'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    else:
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')

    # Identificar coluna DST
    dst_col = [c for c in df.columns if 'dst' in c][0]
    df['dst'] = pd.to_numeric(df[dst_col], errors='coerce')

    # Filtrar valores fisicamente impossíveis
    df = df[(df['dst'] > -1000) & (df['dst'] < 500)]
    df = df.dropna(subset=['time_tag', 'dst']).sort_values('time_tag')

    print(f"   ✅ {len(df)} pontos, Dst: {df['dst'].min():.1f} a {df['dst'].max():.1f} nT")
    return df[['time_tag', 'dst']]

def load_omni_clean(filepath, resample_minutes=None):
    """
    Carrega OMNI, remove flags inválidas, interpola gaps longos,
    converte unidades e opcionalmente reamostra para acelerar.
    """
    print(f"\n📥 Carregando OMNI: {filepath}")
    df = pd.read_csv(filepath)
    df = normalize_omni_columns(df, allow_partial=True)

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df = df.dropna(subset=['time_tag']).sort_values('time_tag')

    if 'dst' in df.columns:
        df = df.drop(columns=['dst'])

    if 'speed' in df.columns and df['speed'].median() > 2000:
        print("   ⚠️ Convertendo m/s → km/s")
        df['speed'] /= 1000.0

    # Remover flags inválidas (valores sentinelas)
    invalid_flags = [999, 9999, 99999, 999.9, 9999.9]
    for col in ['speed', 'density', 'bz_gsm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(invalid_flags, np.nan)

    # Filtrar valores fisicamente absurdos (menos restritivo para capturar extremos)
    df = df[
        (df['speed'] > 100) & (df['speed'] < 3000) &
        (df['density'] > 0.1) & (df['density'] < 300) &
        (df['bz_gsm'] > -100) & (df['bz_gsm'] < 100)
    ].copy()

    # Interpolação linear + preenchimento forward/backward
    for col in ['speed', 'density', 'bz_gsm']:
        df[col] = (
            df[col]
            .interpolate(method='linear', limit=10)
            .ffill()
            .bfill()
        )

    # Reamostragem opcional para reduzir tempo de processamento
    if resample_minutes is not None:
        print(f"   ⏱️ Reamostrando para {resample_minutes} minutos...")
        df = df.set_index('time_tag')
        df = df.resample(f'{resample_minutes}min').mean()
        df = df.interpolate(method='linear').reset_index()
        print(f"   ✅ Após reamostragem: {len(df)} pontos")

    print(f"   ✅ {len(df)} pontos limpos (com interpolação)")
    return df

# =========================
# ALINHAMENTO COM LAG FIXO (PRESERVA EXTREMOS)
# =========================
def align_omni_to_dst_fixed(omni_df, dst_df, lag_minutes=LAG_MINUTES):
    """
    Alinha OMNI ao Dst usando um lag fixo simples.
    Preserva os extremos do vento solar (essencial para tempestades).
    """
    dst_sorted = dst_df.sort_values('time_tag').copy()
    omni_sorted = omni_df.sort_values('time_tag').copy()

    # Renomear coluna de tempo do OMNI para evitar conflito no merge
    omni_sorted = omni_sorted.rename(columns={'time_tag': 'time_tag_omni'})

    # Desloca o tempo do OMNI pelo lag de propagação
    omni_sorted['time_shifted'] = omni_sorted['time_tag_omni'] + pd.Timedelta(minutes=lag_minutes)

    # Merge asof: para cada Dst, pega o OMNI mais recente antes do tempo deslocado
    aligned = pd.merge_asof(
        dst_sorted,
        omni_sorted,
        left_on='time_tag',
        right_on='time_shifted',
        direction='backward'
    )

    # Limpeza das colunas
    aligned = aligned.drop(columns=['time_shifted', 'time_tag_omni'])
    aligned = aligned.dropna(subset=['speed', 'bz_gsm'])

    print(f"   ✅ Alinhamento com lag fixo de {lag_minutes} min: {len(aligned)} pontos")
    return aligned.reset_index(drop=True)

# =========================
# CALIBRAÇÃO GLOBAL (CORE)
# =========================
def global_calibration(df_aligned):
    """Calibra HAC_REF e Q_FACTOR usando dados já alinhados."""
    print("\n📊 Calibração global com dados alinhados...")

    core = HACCoreModel(HACCoreConfig())

    _, comp = compute_hac(
        time=df_aligned['time_tag'].values,
        bz=df_aligned['bz_gsm'].values,
        v=df_aligned['speed'].values,
        density=df_aligned['density'].values,
        config=core.config
    )
    coupling = comp['coupling']
    corr = np.corrcoef(coupling, df_aligned['dst'])[0, 1]
    print(f"   • Corr(coupling, Dst): {corr:.3f}")

    # dt real
    times = pd.to_datetime(df_aligned['time_tag']).values
    dt = np.diff(times).astype('timedelta64[s]').astype(float)
    dt = np.insert(dt, 0, np.median(dt))
    dt = np.maximum(dt, 1.0)

    core.fit_calibration(coupling, dt, df_aligned['dst'].values)

    # Proteção contra Q_FACTOR nulo
    if abs(core.config.Q_FACTOR) < 1e-5:
        print("   ⚠️ Q_FACTOR muito baixo → ajustando para -0.01")
        core.config.Q_FACTOR = -0.01

    hac_raw = comp['raw']
    hac_ref = np.percentile(hac_raw, 99)
    core.config.HAC_REF = max(hac_ref, 50.0)

    print(f"   • HAC_REF: {core.config.HAC_REF:.1f}")
    print(f"   • Q_FACTOR: {core.config.Q_FACTOR:.5f}")

    with open(CALIBRATION_FILE, "w") as f:
        json.dump({
            "HAC_REF": core.config.HAC_REF,
            "Q_FACTOR": core.config.Q_FACTOR,
            "TAU_DST": core.config.TAU_DST,
            "DST_Q": core.config.DST_Q
        }, f, indent=4)

    return core.config

def auto_calibrate_parameters(df_train, filter_quiet=True, dst_threshold=-20):
    """
    Calibra automaticamente k_dst, HAC_Q_SCALE e hac_thr usando regressão
    ponderada pela intensidade do Dst. Usa dt REAL dos timestamps.
    """
    from scipy.optimize import differential_evolution

    print("\n🔧 AUTO-CALIBRAÇÃO DE PARÂMETROS (HAC → Dst)\n")

    # Filtrar períodos muito calmos
    if filter_quiet:
        mask_quiet = df_train['dst'] < dst_threshold
        df_calib = df_train[mask_quiet].copy()
        print(f"   • Filtrando Dst < {dst_threshold} nT: {len(df_calib)} pontos restantes")
    else:
        df_calib = df_train.copy()

    dst = df_calib['dst'].values
    hac = df_calib['HAC_total'].values
    times = pd.to_datetime(df_calib['time_tag']).values

    # Calcular dt REAL (horas)
    dt_sec = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec = np.insert(dt_sec, 0, np.median(dt_sec))
    dt_hours = dt_sec / 3600.0
    dt_hours[dt_hours <= 0] = 1.0 / 60.0   # mínimo 1 minuto

    dDst_dt = np.gradient(dst) / dt_hours

    # Remover NaNs
    mask = np.isfinite(dst) & np.isfinite(hac) & np.isfinite(dDst_dt)
    dst = dst[mask]
    hac = hac[mask]
    dDst_dt = dDst_dt[mask]

    # Pesos proporcionais à intensidade do Dst
    weights = np.clip(np.abs(dst) / 50.0, 1.0, 10.0)

    def objective(params):
        k, scale, thr = params
        if k <= 0 or scale <= 0 or thr < 0:
            return 1e9
        hac_eff = np.maximum(0, hac - thr)
        X = np.sqrt(hac_eff / scale)
        valid = X > 0
        if np.sum(valid) < 50:
            return 1e9
        Y = -dDst_dt[valid]
        Xv = X[valid]
        w = weights[valid]
        Q_pred = k * Xv
        mse = np.average((Q_pred - Y)**2, weights=w)
        return mse

    bounds = [(0.5, 20.0), (10.0, 500.0), (5.0, 80.0)]
    result = differential_evolution(objective, bounds, maxiter=100, disp=False, seed=42)

    k_dst, HAC_Q_SCALE, hac_thr = result.x
    best_score = result.fun

    print(f"✅ k_dst ótimo:        {k_dst:.3f}")
    print(f"✅ HAC_Q_SCALE ótimo:  {HAC_Q_SCALE:.1f}")
    print(f"✅ hac_thr ótimo:      {hac_thr:.1f}")
    print(f"📉 Erro quadrático médio (ponderado): {best_score:.3f}")

    return {
        'k_dst': k_dst,
        'HAC_Q_SCALE': HAC_Q_SCALE,
        'hac_thr': hac_thr
    }

# =========================
# VALIDAÇÃO DE UM EVENTO
# =========================
def validate_event(config_core, df_aligned, name, start, end, physics_config):
    print(f"\n🌌 {name}")

    mask = (df_aligned['time_tag'] >= start) & (df_aligned['time_tag'] <= end)
    event = df_aligned[mask].copy()

    if len(event) < 10:
        print("   ⚠️ Dados insuficientes")
        return None

    print(f"   • Pontos: {len(event)}")
    print(f"   • Dst obs mín: {event['dst'].min():.1f} nT")
    print(f"   • Bz min: {event['bz_gsm'].min():.1f} nT")
    print(f"   • V max: {event['speed'].max():.1f} km/s")

    # Diagnóstico de extremos
    print("   🔍 Extremos do evento:")
    print(event.nsmallest(5, 'bz_gsm')[['time_tag', 'bz_gsm', 'speed', 'dst']])

    # Calcular campos físicos
    event = PhysicalFieldsCalculator.compute_all_fields(event)

    print(f"   • Coupling max: {event['coupling_signal'].max():.2f}")
    print(f"   • VBs real max: {event['VBs_real'].max():.2f} mV/m")

    # Aplicar HAC_REF do core
    physics_config.HAC_REF = config_core.HAC_REF

    # Instanciar modelo de produção
    model = ProductionHACModel(config=physics_config)

    # Garantir que os parâmetros estão no config interno
    model.config.HAC_Q_SCALE = physics_config.HAC_Q_SCALE
    model.config.K_DST = physics_config.K_DST
    model.config.HAC_THR = physics_config.HAC_THR
    if hasattr(model, 'core'):
        model.core.config.HAC_Q_SCALE = physics_config.HAC_Q_SCALE
        model.core.config.K_DST = physics_config.K_DST
        model.core.config.HAC_THR = physics_config.HAC_THR

    hac = model.compute_hac_system(event)
    _, dst_pred, _ = model.predict_storm_indicators(hac)
    dst_pred = np.clip(dst_pred, -500, 50)

    metrics = evaluate_event(
        time=event['time_tag'].values,
        dst_obs=event['dst'].values,
        dst_pred=dst_pred
    )

    print(f"   • Dst pred mín: {dst_pred.min():.1f} nT")
    print(f"   • Correlação: {metrics['correlation']:.3f}")
    print(f"   • MAE: {metrics['MAE']:.1f} nT")
    print(f"   • Erro mín Dst: {metrics['min_Dst_error_nT']:.1f} nT")

    timing = phase_metrics(
    event['time_tag'].values,
    event['dst'].values,
    dst_pred)
    
    print(f"   • Timing erro mín: {timing['min_timing_error_min']:.1f} min")
    print(f"   • Área integrada erro: {timing['integrated_area_error_nT_h']:.1f}")
    print(f"   • Recovery obs: {timing['obs_recovery_slope_nT_h']:.2f}")
    print(f"   • Recovery pred: {timing['pred_recovery_slope_nT_h']:.2f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(event['time_tag'], event['dst'], 'b-', label='Dst Kyoto')
    plt.plot(event['time_tag'], dst_pred, 'r--', label='HAC++')
    plt.title(f'{name} - Corr={metrics["correlation"]:.2f}, MAE={metrics["MAE"]:.0f} nT')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"event_{name}_validation.png", dpi=150)
    plt.close()

    return metrics
  
def integrated_area_error(time, obs, pred):
    """Integral temporal do erro absoluto (nT·h)."""

    dt = np.diff(time).astype('timedelta64[s]').astype(float) / 3600.0

    area = np.sum( 0.5 * ( np.abs(obs[:-1] - pred[:-1]) + np.abs(obs[1:] - pred[1:]) ) * dt)

    return area

def phase_metrics(time, obs, pred):
    """Calcula métricas de fase temporal."""
    # Timing do mínimo
    idx_obs = np.argmin(obs)
    idx_pred = np.argmin(pred)
    dt_min = (time[idx_pred] - time[idx_obs]).astype('timedelta64[m]').astype(float)

    # Área integrada do erro (nT·h)
    dt = np.diff(time).astype('timedelta64[s]').astype(float) / 3600.0
    iae = np.sum( 0.5 * ( np.abs(obs[:-1] - pred[:-1]) + np.abs(obs[1:] - pred[1:])) * dt)
  
    # Recovery slope (nT/h) – entre 6h e 12h após o mínimo
    t_min_obs = time[idx_obs]
    mask_rec = (time >= t_min_obs + np.timedelta64(6, 'h')) & (time <= t_min_obs + np.timedelta64(12, 'h'))
    if np.sum(mask_rec) >= 2:
        slope_obs = np.polyfit(np.arange(np.sum(mask_rec)), obs[mask_rec], 1)[0]
        slope_pred = np.polyfit(np.arange(np.sum(mask_rec)), pred[mask_rec], 1)[0]
    else:
        slope_obs = slope_pred = np.nan

    return {
        'min_timing_error_min': dt_min,
        'integrated_area_error_nT_h': iae,
        'obs_recovery_slope_nT_h': slope_obs,
        'pred_recovery_slope_nT_h': slope_pred}
  
# =========================
# MAIN
# =========================
def main():
    print("=" * 70)
    print("🚀 HAC++ VALIDAÇÃO COM ALINHAMENTO FIXO E INTERPOLAÇÃO ROBUSTA")
    print("=" * 70)

    omni = load_omni_clean(OMNI_FILE, resample_minutes=RESAMPLE_MINUTES)
    dst = load_dst_kyoto(DST_FILE)

    # Alinhar OMNI ao Dst com lag fixo
    print("\n🔗 Alinhando OMNI → Dst (lag fixo de 20 min)...")
    df_aligned = align_omni_to_dst_fixed(omni, dst)
    print(f"   ✅ Dataset alinhado: {len(df_aligned)} pontos")

    # Diagnóstico geral
    print("\n🔍 Estatísticas gerais após alinhamento:")
    print(df_aligned[['bz_gsm', 'speed', 'dst']].describe())

    # Divisão treino/teste
    df_train = df_aligned[df_aligned['time_tag'] <= TRAIN_END_DATE].copy()
    df_test  = df_aligned[df_aligned['time_tag'] > TRAIN_END_DATE].copy()
    print(f"\n📊 Treino: {len(df_train)} | Teste: {len(df_test)}")

    # ========== AUTO-CALIBRAÇÃO NO CONJUNTO DE TREINO ==========
    print("\n⚙️ Preparando conjunto de treino para auto-calibração...")
    df_train = PhysicalFieldsCalculator.compute_all_fields(df_train)
    model_temp = ProductionHACModel()
    hac_train = model_temp.compute_hac_system(df_train, calibration_mode=True)
    df_train['HAC_total'] = hac_train

    params_raw = auto_calibrate_parameters(df_train, filter_quiet=True, dst_threshold=-20)

    # Fatores de correção empíricos
    CORR_SCALE = 0.3
    CORR_K = 12.0
    CORR_THR = 10.0

    physics_config = HACPhysicsConfig()
    physics_config.HAC_Q_SCALE = params_raw['HAC_Q_SCALE'] * CORR_SCALE
    physics_config.K_DST = params_raw['k_dst'] * CORR_K
    physics_config.HAC_THR = params_raw['hac_thr'] + CORR_THR

    print(f"\n🔧 Parâmetros auto‑calibrados (brutos):")
    print(f"   • k_dst = {params_raw['k_dst']:.3f}")
    print(f"   • HAC_Q_SCALE = {params_raw['HAC_Q_SCALE']:.1f}")
    print(f"   • hac_thr = {params_raw['hac_thr']:.1f}")

    print(f"\n🔧 Parâmetros APÓS correção empírica:")
    print(f"   • HAC_Q_SCALE: {physics_config.HAC_Q_SCALE:.1f}")
    print(f"   • K_DST: {physics_config.K_DST:.3f}")
    print(f"   • HAC_THR: {physics_config.HAC_THR:.1f}")

    config_core = global_calibration(df_train)
        
    # Validação
    results = []
    for name, (start, end) in EVENTS.items():
        m = validate_event(config_core, df_aligned, name, start, end, physics_config)
        if m:
            results.append(m)

    if results:
        print("\n" + "=" * 70)
        print("📈 MÉTRICAS MÉDIAS")
        print("=" * 70)
        avg_corr = np.mean([m['correlation'] for m in results])
        avg_mae = np.mean([m['MAE'] for m in results])
        avg_min_err = np.mean([m['min_Dst_error_nT'] for m in results])
        print(f"   • Correlação: {avg_corr:.3f}")
        print(f"   • MAE: {avg_mae:.1f} nT")
        print(f"   • Erro mín Dst: {avg_min_err:.1f} nT")

    print("\n✅ Validação concluída.")

if __name__ == "__main__":
    main()
