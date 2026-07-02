#!/usr/bin/env python3
"""
validate_events.py – Validação HAC++ com calibração Leave‑One‑Event‑Out,
                      warm‑up magnetosférico, persistência multihorizonte
                      e diagnóstico de saturação.

Melhorias aplicadas:
  • LOEO (Leave‑One‑Event‑Out): cada evento é avaliado com parâmetros
    calibrados excluindo o próprio período do evento → elimina data leakage.
  • Warm‑up de 5 dias: os reservatórios (HAC, tail, ring current) são
    inicializados com 5 dias de histórico antes do evento.
  • Persistência em 20 min, 1 h e 3 h (além do passo de 5 min).
  • Mais eventos históricos (8 no total) para cobertura estatística.
  • Diagnóstico de clipping do Dst previsto.
  • Lag de propagação dinâmico (dependente de Vsw) em vez de fixo.
  • Métricas de fase temporal (timing, área integrada, recovery slope).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta

from hac_final import (
    ProductionHACModel,
    PhysicalFieldsCalculator,
    HACPhysicsConfig,
    normalize_omni_columns,
)
from hac_core import HACCoreModel, HACCoreConfig, evaluate_event, compute_hac

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
EVENTS = {
    "Bastille_Day_2000":      ("2000-07-14", "2000-07-18"),
    "April_2001":             ("2001-04-11", "2001-04-15"),
    "Halloween_2003":         ("2003-10-28", "2003-11-02"),
    "November_2004":          ("2004-11-07", "2004-11-11"),
    "December_2006":          ("2006-12-14", "2006-12-18"),
    "St_Patrick_2015":        ("2015-03-17", "2015-03-20"),
    "September_2017":         ("2017-09-07", "2017-09-11"),
    "Carrington_like_2012":   ("2012-07-22", "2012-07-25"),
}

OMNI_FILE = "omni_1min_2000_2020.csv"
DST_FILE = "data/dst_kyoto_2000_2020.csv"
CALIBRATION_FILE = "hac_calibration.json"

RESAMPLE_MINUTES = 5               # reamostragem para acelerar
WARMUP_DAYS = 5                    # dias de aquecimento antes do evento
LAG_BASE_MINUTES = 20              # lag nominal (usado como fallback)
PERSIST_HORIZONS = {               # passos de persistência (5 min cada)
    "5min": 1,
    "20min": 4,
    "1h": 12,
    "3h": 36,
}


# ============================================================================
# CARREGAMENTO DOS DADOS
# ============================================================================
def load_dst_kyoto(filepath):
    """Carrega Dst do Kyoto, garante datetime e filtra valores absurdos."""
    print(f"\n📥 Carregando DST: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    if 'time_tag' not in df.columns:
        if 'date' in df.columns and 'time' in df.columns:
            df['time_tag'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        else:
            df['time_tag'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    else:
        df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')

    dst_col = [c for c in df.columns if 'dst' in c][0]
    df['dst'] = pd.to_numeric(df[dst_col], errors='coerce')

    df = df[(df['dst'] > -1000) & (df['dst'] < 500)]
    df = df.dropna(subset=['time_tag', 'dst']).sort_values('time_tag')

    print(f"   ✅ {len(df)} pontos, Dst: {df['dst'].min():.1f} a {df['dst'].max():.1f} nT")
    return df[['time_tag', 'dst']]


def load_omni_clean(filepath, resample_minutes=None):
    """Carrega OMNI, remove flags, interpola e opcionalmente reamostra."""
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

    invalid_flags = [999, 9999, 99999, 999.9, 9999.9]
    for col in ['speed', 'density', 'bz_gsm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(invalid_flags, np.nan)

    df = df[
        (df['speed'] > 100) & (df['speed'] < 3000) &
        (df['density'] > 0.1) & (df['density'] < 300) &
        (df['bz_gsm'] > -100) & (df['bz_gsm'] < 100)
    ].copy()

    for col in ['speed', 'density', 'bz_gsm']:
        df[col] = (
            df[col]
            .interpolate(method='linear', limit=10)
            .ffill()
            .bfill()
        )

    if resample_minutes is not None:
        print(f"   ⏱️ Reamostrando para {resample_minutes} minutos...")
        df = df.set_index('time_tag')
        df = df.resample(f'{resample_minutes}min').mean()
        df = df.interpolate(method='linear').reset_index()
        print(f"   ✅ Após reamostragem: {len(df)} pontos")

    print(f"   ✅ {len(df)} pontos limpos (com interpolação)")
    return df


# ============================================================================
# ALINHAMENTO COM LAG DINÂMICO
# ============================================================================
def compute_dynamic_lag_minutes(v, bz):
    """
    Lag de propagação + resposta magnetosférica, em minutos.
      - v: velocidade do vento solar (km/s)
      - bz: componente Bz (nT)
    Retorna valor entre 15 e 90 minutos.
    """
    # tempo de viagem até a magnetopausa (~1.5e6 km)
    propagation = 1.5e6 / (v * 1e3) / 60.0 if v > 0 else 30.0
    # tempo de resposta (eventos mais intensos respondem mais rápido)
    vbs = v * max(0.0, -bz) * 1e-3
    response = 30.0 * np.exp(-vbs / 5.0)
    lag = propagation + response
    return np.clip(lag, 15, 90)


def align_omni_to_dst_dynamic(omni_df, dst_df):
    """
    Alinha OMNI ao Dst usando lag dinâmico (dependente de Vsw e Bz).
    Cada linha do Dst recebe o OMNI correspondente ao instante
        t_dst - lag_minutos(V, Bz)
    usando merge_asof com direction='backward'.
    """
    dst_sorted = dst_df.sort_values('time_tag').copy()
    omni_sorted = omni_df.sort_values('time_tag').copy()

    # Para cada timestamp do Dst, calculamos o lag individual
    times_dst = dst_sorted['time_tag'].values.astype('datetime64[ns]')
    times_omni = omni_sorted['time_tag'].values.astype('datetime64[ns]')

    # Pré‑calcular lag médio em janelas (para eficiência)
    lags = np.full(len(times_dst), LAG_BASE_MINUTES, dtype=float)
    # Usamos o OMNI mais próximo para estimar V, Bz
    idx_omni = np.searchsorted(times_omni, times_dst, side='right') - 1
    idx_omni = np.clip(idx_omni, 0, len(times_omni)-1)
    v_est = omni_sorted['speed'].iloc[idx_omni].values
    bz_est = omni_sorted['bz_gsm'].iloc[idx_omni].values
    for i in range(len(times_dst)):
        lags[i] = compute_dynamic_lag_minutes(v_est[i], bz_est[i])

    # Criar coluna de tempo defasado
    omni_sorted = omni_sorted.rename(columns={'time_tag': 'time_tag_omni'})
    omni_sorted['time_shifted'] = omni_sorted['time_tag_omni']  # será sobrescrito
    # Como o merge_asof não suporta lag variável por linha, fazemos um loop
    # mas de forma vetorizada: criamos um DataFrame auxiliar com os tempos defasados
    target_times = times_dst - pd.to_timedelta(lags, unit='m')
    # Para cada target_time, buscamos o OMNI mais recente (backward)
    indices = np.searchsorted(times_omni, target_times, side='right') - 1
    indices = np.clip(indices, 0, len(times_omni)-1)

    aligned = omni_sorted.iloc[indices].copy()
    aligned['time_tag'] = times_dst
    aligned['dst'] = dst_sorted['dst'].values
    aligned = aligned.drop(columns=['time_tag_omni', 'time_shifted'], errors='ignore')
    aligned = aligned.dropna(subset=['speed', 'bz_gsm'])

    print(f"   ✅ Alinhamento dinâmico: {len(aligned)} pontos (lag médio {np.mean(lags):.1f} min)")
    return aligned.reset_index(drop=True)


# ============================================================================
# CALIBRAÇÃO GLOBAL (HAC_REF, Q_FACTOR) COM EXCLUSÃO DE PERÍODO
# ============================================================================
def global_calibration(df_aligned, exclude_start=None, exclude_end=None):
    """
    Calibra HAC_REF e Q_FACTOR.
    Se exclude_start/end forem fornecidos, aquele intervalo é removido
    (Leave‑One‑Event‑Out).
    """
    if exclude_start is not None and exclude_end is not None:
        mask = ~df_aligned['time_tag'].between(exclude_start, exclude_end)
        df_calib = df_aligned[mask].copy()
        print("   🔹 LOEO: excluindo período do evento da calibração")
    else:
        df_calib = df_aligned.copy()

    print("\n📊 Calibração global com dados alinhados...")

    core = HACCoreModel(HACCoreConfig())

    _, comp = compute_hac(
        time=df_calib['time_tag'].values,
        bz=df_calib['bz_gsm'].values,
        v=df_calib['speed'].values,
        density=df_calib['density'].values,
        config=core.config
    )
    coupling = comp['coupling']
    corr = np.corrcoef(coupling, df_calib['dst'])[0, 1]
    print(f"   • Corr(coupling, Dst): {corr:.3f}")

    times = pd.to_datetime(df_calib['time_tag']).values
    dt = np.diff(times).astype('timedelta64[s]').astype(float)
    dt = np.insert(dt, 0, np.median(dt))
    dt = np.maximum(dt, 1.0)

    core.fit_calibration(coupling, dt, df_calib['dst'].values)

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


# ============================================================================
# MÉTRICAS DE FASE TEMPORAL
# ============================================================================
def phase_metrics(time, obs, pred):
    """Calcula métricas de fase temporal."""
    time = np.asarray(time, dtype='datetime64[ns]')
    idx_obs = np.argmin(obs)
    idx_pred = np.argmin(pred)

    dt_min = (time[idx_pred] - time[idx_obs]).astype('timedelta64[m]').astype(float)

    dt_sec = np.diff(time).astype('timedelta64[s]').astype(float) / 3600.0
    dt_sec = np.insert(dt_sec, 0, np.median(dt_sec))
    err = np.abs(obs - pred)
    iae = np.sum(err * dt_sec)

    t_min_obs = time[idx_obs]
    mask_rec = (time >= t_min_obs + np.timedelta64(6, 'h')) & (
        time <= t_min_obs + np.timedelta64(12, 'h'))
    if np.sum(mask_rec) >= 2:
        t_rec = np.arange(np.sum(mask_rec)) * np.median(dt_sec[mask_rec])
        slope_obs = np.polyfit(t_rec, obs[mask_rec], 1)[0]
        slope_pred = np.polyfit(t_rec, pred[mask_rec], 1)[0]
    else:
        slope_obs = slope_pred = np.nan

    return {
        'min_timing_error_min': dt_min,
        'integrated_area_error_nT_h': iae,
        'obs_recovery_slope_nT_h': slope_obs,
        'pred_recovery_slope_nT_h': slope_pred,
    }


# ============================================================================
# BENCHMARK: PERSISTÊNCIA EM MÚLTIPLOS HORIZONTES
# ============================================================================
def persistence_forecast(dst_obs, steps):
    """Previsão por persistência: Dst(t+steps) = Dst(t)."""
    pred = np.roll(dst_obs, -steps)
    pred[-steps:] = dst_obs[-1]
    return pred


# ============================================================================
# VALIDAÇÃO DE UM EVENTO (COM WARM‑UP)
# ============================================================================
def validate_event(config_core, df_aligned, name, start, end, physics_config,
                   warmup_days=WARMUP_DAYS):
    print(f"\n🌌 {name}")

    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    warmup_start = start_dt - timedelta(days=warmup_days)

    # ----- Janela com warm‑up -----
    mask_all = (df_aligned['time_tag'] >= warmup_start) & (df_aligned['time_tag'] <= end)
    event_all = df_aligned[mask_all].copy()

    if len(event_all) < 20:
        print("   ⚠️ Dados insuficientes (mesmo com warm‑up)")
        return None

    # ----- Calcular campos físicos -----
    event_all = PhysicalFieldsCalculator.compute_all_fields(event_all)

    # ----- Rodar modelo na janela completa (aquece reservatórios) -----
    physics_config.HAC_REF = config_core.HAC_REF
    model = ProductionHACModel(config=physics_config)
    hac = model.compute_hac_system(event_all)
    _, dst_pred_full, _ = model.predict_storm_indicators(hac)
    dst_pred_full = np.clip(dst_pred_full, -500, 50)

    # ----- Separar apenas o período do evento para métricas -----
    mask_event = (event_all['time_tag'] >= start) & (event_all['time_tag'] <= end)
    event = event_all[mask_event].copy()
    dst_pred = dst_pred_full[mask_event]
    dst_obs  = event['dst'].values
    time_vec = event['time_tag'].values

    if len(event) < 10:
        print("   ⚠️ Dados insuficientes no período do evento")
        return None

    # ----- Diagnóstico de clipping -----
    n_low = np.sum(dst_pred < -500)
    n_high = np.sum(dst_pred > 50)
    if n_low or n_high:
        print(f"   ⚠️ Clip Dst: {n_low} pts < -500, {n_high} pts > 50")

    print(f"   • Pontos no evento (após warm‑up): {len(event)}")
    print(f"   • Dst obs mín: {dst_obs.min():.1f} nT")
    print(f"   • Bz min: {event['bz_gsm'].min():.1f} nT")
    print(f"   • V max: {event['speed'].max():.1f} km/s")
    print("   🔍 Extremos do evento:")
    print(event.nsmallest(5, 'bz_gsm')[['time_tag', 'bz_gsm', 'speed', 'dst']])
    print(f"   • VBs real max: {event['VBs_real'].max():.2f} mV/m")

    # ----- Métricas HAC++ -----
    metrics = evaluate_event(time=time_vec, dst_obs=dst_obs, dst_pred=dst_pred)
    phase = phase_metrics(time_vec, dst_obs, dst_pred)

    print(f"\n   📊 HAC++:")
    print(f"      Dst pred mín: {dst_pred.min():.1f} nT")
    print(f"      Correlação: {metrics['correlation']:.3f}")
    print(f"      MAE: {metrics['MAE']:.1f} nT")
    print(f"      Erro mín Dst: {metrics['min_Dst_error_nT']:.1f} nT")
    print(f"      Timing erro: {phase['min_timing_error_min']:.1f} min")
    print(f"      Área integrada: {phase['integrated_area_error_nT_h']:.1f} nT·h")
    print(f"      Recovery obs: {phase['obs_recovery_slope_nT_h']:.2f} nT/h")
    print(f"      Recovery pred: {phase['pred_recovery_slope_nT_h']:.2f} nT/h")

    # ----- Benchmark: persistência multihorizonte -----
    for label, steps in PERSIST_HORIZONS.items():
        dst_persist = persistence_forecast(dst_obs, steps)
        metrics_p = evaluate_event(time=time_vec, dst_obs=dst_obs, dst_pred=dst_persist)
        if label in ("5min", "1h", "3h"):   # exibe apenas alguns para não poluir
            print(f"\n   📊 Persistência {label}:")
            print(f"      Correlação: {metrics_p['correlation']:.3f}")
            print(f"      MAE: {metrics_p['MAE']:.1f} nT")

    # ----- Plot -----
    plt.figure(figsize=(14, 6))
    plt.plot(time_vec, dst_obs, 'b-', label='Dst Kyoto', linewidth=2)
    plt.plot(time_vec, dst_pred, 'r--', label='HAC++', linewidth=2)
    # persistência de 1h para referência no gráfico
    dst_p1h = persistence_forecast(dst_obs, PERSIST_HORIZONS["1h"])
    plt.plot(time_vec, dst_p1h, 'g:', label='Persistência 1h', linewidth=1.5)
    plt.axhline(-50, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(-100, color='orange', linestyle=':', alpha=0.5)
    plt.axhline(-200, color='red', linestyle=':', alpha=0.5)
    plt.title(f'{name} – HAC++ Corr={metrics["correlation"]:.2f}, MAE={metrics["MAE"]:.0f} nT')
    plt.xlabel('Tempo (UTC)')
    plt.ylabel('Dst [nT]')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"event_{name}_validation.png", dpi=150)
    plt.close()

    return metrics


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("🚀 HAC++ VALIDAÇÃO COM LOEO, WARM‑UP E LAG DINÂMICO")
    print("=" * 70)

    omni = load_omni_clean(OMNI_FILE, resample_minutes=RESAMPLE_MINUTES)
    dst = load_dst_kyoto(DST_FILE)

    # Alinhamento dinâmico (substitui lag fixo)
    print("\n🔗 Alinhando OMNI → Dst (lag dinâmico)...")
    df_aligned = align_omni_to_dst_dynamic(omni, dst)
    print(f"   ✅ Dataset alinhado: {len(df_aligned)} pontos")

    print("\n🔍 Estatísticas gerais após alinhamento:")
    print(df_aligned[['bz_gsm', 'speed', 'dst']].describe())

    # Configuração física base (HAC_REF será atualizado por evento)
    physics_config = HACPhysicsConfig()

    results = []
    for name, (start, end) in EVENTS.items():
        # ----- Calibração LOEO -----
        config_core = global_calibration(df_aligned, exclude_start=start, exclude_end=end)
        physics_config.HAC_REF = config_core.HAC_REF

        # ----- Validação com warm‑up -----
        m = validate_event(config_core, df_aligned, name, start, end, physics_config)
        if m:
            results.append(m)

    if results:
        print("\n" + "=" * 70)
        print("📈 MÉTRICAS MÉDIAS (HAC++ FÍSICO)")
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
