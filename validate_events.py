#!/usr/bin/env python3
"""
validate_events.py - Validação HAC++ com alinhamento causal e calibração robusta.
Versão corrigida para modelo sem atributo 'core'.
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

OMNI_FILE = "data/omni_2000_2020.csv"
DST_FILE = "data/dst_kyoto_2000_2020.csv"
CALIBRATION_FILE = "hac_calibration.json"

# =========================
# CARREGAMENTO DOS DADOS
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

def load_omni_clean(filepath):
    """Carrega OMNI, remove flags inválidas e converte unidades."""
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
        (df['speed'] > 100) & (df['speed'] < 2000) &
        (df['density'] > 0.1) & (df['density'] < 200) &
        (df['bz_gsm'] > -100) & (df['bz_gsm'] < 100)
    ].copy()

    df['speed'] = df['speed'].ffill(limit=3).fillna(400)
    df['density'] = df['density'].ffill(limit=3).fillna(5)
    df['bz_gsm'] = df['bz_gsm'].fillna(0)

    print(f"   ✅ {len(df)} pontos limpos")
    return df

# =========================
# LAG DINÂMICO
# =========================
def compute_dynamic_lag(speed):
    """Lag em horas: distância ~1.5e6 km / velocidade."""
    lag = 1.5e6 / (speed * 3600.0)
    return np.clip(lag, 0.3, 1.5)

def align_omni_to_dst(omni_df, dst_df):
    """
    Para cada timestamp do Dst, busca o OMNI mais recente
    antes de (t_dst - lag), com lag baseado no percentil 75 da velocidade recente.
    """
    dst_sorted = dst_df.sort_values('time_tag').copy()
    times_dst = dst_sorted['time_tag'].values.astype('datetime64[ns]')

    omni_sorted = omni_df.sort_values('time_tag').copy()
    times_omni = omni_sorted['time_tag'].values.astype('datetime64[ns]')

    indices = []
    for t_dst in times_dst:
        # Índice do OMNI mais recente até t_dst (para estimar velocidade)
        idx_now = np.searchsorted(times_omni, t_dst, side='right') - 1
        idx_now = max(0, min(idx_now, len(omni_sorted)-1))
        # Usar percentil 75 para capturar rajadas de velocidade (CME)
        w_start = max(0, idx_now - 3)
        w_end = min(len(omni_sorted), idx_now + 1)
        speed_vals = omni_sorted.iloc[w_start:w_end]['speed']
        speed_eff = np.percentile(speed_vals, 75) if len(speed_vals) > 1 else speed_vals.mean()
        lag_h = compute_dynamic_lag(speed_eff)

        target_time = t_dst - np.timedelta64(int(lag_h * 3600), 's')
        idx_lag = np.searchsorted(times_omni, target_time, side='right') - 1
        idx_lag = max(0, min(idx_lag, len(omni_sorted)-1))
        indices.append(idx_lag)

    aligned = omni_sorted.iloc[indices].copy()
    aligned['time_tag'] = times_dst
    aligned['dst'] = dst_sorted['dst'].values
    return aligned.reset_index(drop=True)

# =========================
# CALIBRAÇÃO GLOBAL
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
    
def auto_calibrate_parameters(df_train):
    """
    Calibra automaticamente k_dst, HAC_Q_SCALE e hac_thr
    usando regressão linear entre √(HAC_eff / scale) e -dDst/dt.
    """
    import numpy as np
    from scipy.optimize import minimize

    print("\n🔧 AUTO-CALIBRAÇÃO DE PARÂMETROS (HAC → Dst)\n")

    dst = df_train['dst'].values
    hac = df_train['HAC_total'].values

    # Derivada do Dst (nT/h) – assumindo dados horários
    dt_hours = 1.0
    dDst_dt = np.gradient(dst, dt_hours)

    # Remover NaNs
    mask = np.isfinite(dst) & np.isfinite(hac) & np.isfinite(dDst_dt)
    dst = dst[mask]
    hac = hac[mask]
    dDst_dt = dDst_dt[mask]

    # Função objetivo para otimização
    def objective(params):
        k, scale, thr = params
        if k <= 0 or scale <= 0 or thr < 0:
            return 1e9
        hac_eff = np.maximum(0, hac - thr)
        X = np.sqrt(hac_eff / scale)
        valid = X > 0
        if np.sum(valid) < 100:
            return 1e9
        Y = -dDst_dt[valid]
        Xv = X[valid]
        # Regressão linear forçando o coeficiente k
        Q_pred = k * Xv
        mse = np.mean((Q_pred - Y)**2)
        return mse

    # Otimização com limites razoáveis
    from scipy.optimize import differential_evolution
    bounds = [(1.0, 20.0), (10.0, 200.0), (10.0, 80.0)]
    result = differential_evolution(objective, bounds, maxiter=50, disp=False)

    k_dst, HAC_Q_SCALE, hac_thr = result.x
    best_score = result.fun

    print(f"✅ k_dst ótimo:        {k_dst:.3f}")
    print(f"✅ HAC_Q_SCALE ótimo:  {HAC_Q_SCALE:.1f}")
    print(f"✅ hac_thr ótimo:      {hac_thr:.1f}")
    print(f"📉 Erro quadrático médio: {best_score:.3f}")

    return {
        'k_dst': k_dst,
        'HAC_Q_SCALE': HAC_Q_SCALE,
        'hac_thr': hac_thr
    }
    
# =========================
# VALIDAÇÃO DE UM EVENTO
# =========================
def validate_event(config_core, df_aligned, name, start, end):
    print(f"\n🌌 {name}")

    mask = (df_aligned['time_tag'] >= start) & (df_aligned['time_tag'] <= end)
    event = df_aligned[mask].copy()

    if len(event) < 10:
        print("   ⚠️ Dados insuficientes")
        return None

    print(f"   • Pontos: {len(event)}")
    print(f"   • Dst obs mín: {event['dst'].min():.1f} nT")

    # Calcular campos físicos (coupling etc.)
    event = PhysicalFieldsCalculator.compute_all_fields(event)
    # TESTE DIAGNÓSTICO: amplificar artificialmente o coupling
    #event['coupling_signal'] = event['coupling_signal'] * 5.0
    #print("   ⚠️ TESTE: coupling_signal multiplicado por 5")
    print(f"   • Coupling max: {event['coupling_signal'].max():.2f}")
    print(f"   • Coupling mean: {event['coupling_signal'].mean():.2f}")

    # Criar configuração física a partir da calibração do core
    physics_config = HACPhysicsConfig()
    physics_config.HAC_REF = config_core.HAC_REF
    # Os parâmetros Q_FACTOR, TAU_DST, DST_Q podem ser usados se o modelo de produção os incorporar
    # Caso contrário, serão ignorados

    # Modelo de produção
    model = ProductionHACModel(config=physics_config)

    hac = model.compute_hac_system(event)
    _, dst_pred, _ = model.predict_storm_indicators(hac)
    dst_pred = np.clip(dst_pred, -500, 50)

    metrics = evaluate_event(
        time=event['time_tag'].values,
        dst_obs=event['dst'].values,
        dst_pred=dst_pred
    )

    results = []
    for name, (start, end) in EVENTS.items():
        m = validate_event(config_core, df_aligned, name, start, end, physics_config)
        if m:
            results.append(m)
    
    print(f"   • Dst pred mín: {dst_pred.min():.1f} nT")
    print(f"   • Correlação: {metrics['correlation']:.3f}")
    print(f"   • MAE: {metrics['MAE']:.1f} nT")
    print(f"   • Erro mín Dst: {metrics['min_Dst_error_nT']:.1f} nT")

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

# =========================
# MAIN
# =========================
def main():
    print("=" * 70)
    print("🚀 HAC++ VALIDAÇÃO COM ALINHAMENTO CAUSAL (CORRIGIDO)")
    print("=" * 70)

    omni = load_omni_clean(OMNI_FILE)
    dst = load_dst_kyoto(DST_FILE)

    # Alinhar OMNI ao Dst (lag dinâmico com percentil 75)
    print("\n🔗 Alinhando OMNI → Dst (lag dinâmico)...")
    df_aligned = align_omni_to_dst(omni, dst)
    print(f"   ✅ Dataset alinhado: {len(df_aligned)} pontos")


    # ========== AUTO-CALIBRAÇÃO NO CONJUNTO DE TREINO ==========
    print("\n⚙️ Preparando conjunto de treino para auto-calibração...")
    # Calcular campos físicos no treino
    df_train = PhysicalFieldsCalculator.compute_all_fields(df_train)
    model_temp = ProductionHACModel()
    hac_train = model_temp.compute_hac_system(df_train)
    df_train['HAC_total'] = hac_train
    
    # Auto-calibração dos parâmetros HAC → Dst
    params = auto_calibrate_parameters(df_train)
    
    # Atualizar configuração física com os parâmetros calibrados
    physics_config = HACPhysicsConfig()
    physics_config.HAC_Q_SCALE = params['HAC_Q_SCALE']
    physics_config.K_DST = params['k_dst']
    physics_config.HAC_THR = params['hac_thr']
    print(f"\n📊 Treino: {len(df_train)} | Teste: {len(df_test)}")

    # Calibração
    config = global_calibration(df_train)

    # Validação
    results = []
    for name, (start, end) in EVENTS.items():
        m = validate_event(config, df_aligned, name, start, end)
        if m:
            results.append(m)

    if results:
        print("\n" + "=" * 70)
        print("📈 MÉTRICAS MÉDIAS")
        print("=" * 70)
        print(f"   • Correlação: {np.mean([m['correlation'] for m in results]):.3f}")
        print(f"   • MAE: {np.mean([m['MAE'] for m in results]):.1f} nT")
        print(f"   • Erro mín Dst: {np.mean([m['min_Dst_error_nT'] for m in results]):.1f} nT")

    print("\n✅ Validação concluída.")

if __name__ == "__main__":
    main()
