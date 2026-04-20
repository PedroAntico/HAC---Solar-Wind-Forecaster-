#!/usr/bin/env python3
"""
validate_events.py - HAC++ VALIDAÇÃO FINAL (LAG TEMPORAL REAL + GANHO DINÂMICO)

Correções aplicadas:
- Reforço de injeção por Bz negativo DENTRO do modelo (não pós-processado).
- Lag físico usando busca temporal exata (searchsorted), robusto a gaps.
- Mapeamento Dst dinâmico: ganho varia com Bz e Vsw.
- Tratamento seguro de 'bt'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from hac_final import (
    ProductionHACModel,
    normalize_omni_columns,
    PhysicalFieldsCalculator,
    HACPhysicsConfig
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

EXPECTED_DST_MIN = {
    "Halloween_2003": -350,
    "St_Patrick_2015": -200,
    "Carrington_like_2012": -50,
}

TRAIN_END_DATE = "2014-12-31"

OMNI_FILE = "data/omni_2000_2020.csv"
DST_FILE = "data/dst_kyoto_2000_2020.csv"

DST_MIN_PHYSICAL = -500
DST_MAX_PHYSICAL = 50

# =========================
# CARREGAMENTO OMNI
# =========================
def load_omni(filepath):
    print(f"\n📥 Carregando OMNI: {filepath}")

    df = pd.read_csv(filepath)
    df = normalize_omni_columns(df, allow_partial=True)

    if 'time_tag' not in df.columns:
        raise ValueError("❌ OMNI sem coluna time_tag")

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df = df.dropna(subset=['time_tag']).sort_values('time_tag')

    for col, default in [('speed', 400), ('density', 5), ('bz_gsm', 0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = default

    # Remover flags OMNI
    invalid_flags = [999, 9999, 99999, 999999, 9999999,
                     999.9, 9999.9, 99999.9]
    for col in ['speed', 'density', 'bz_gsm']:
        df[col] = df[col].replace(invalid_flags, np.nan)

    # Converter velocidade se necessário
    if df['speed'].median() > 2000:
        print("   ⚠️ Convertendo m/s → km/s")
        df['speed'] /= 1000.0

    # Preenchimento forward seguro
    df['speed'] = df['speed'].ffill(limit=3).fillna(400)
    df['density'] = df['density'].ffill(limit=3).fillna(5)
    df['bz_gsm'] = df['bz_gsm'].fillna(0)

    # Limites físicos
    df = df[
        (df['speed'] > 150) & (df['speed'] < 2000) &
        (df['density'] > 0.05) & (df['density'] < 200) &
        (df['bz_gsm'] > -100) & (df['bz_gsm'] < 100)
    ].copy()

    print(f"   ✅ OMNI carregado: {len(df)} pontos")
    print(f"   • V range: {df['speed'].min():.1f} – {df['speed'].max():.1f} km/s")
    print(f"   • Bz range: {df['bz_gsm'].min():.1f} – {df['bz_gsm'].max():.1f} nT")
    return df.reset_index(drop=True)


# =========================
# CARREGAMENTO DST
# =========================
def load_dst(filepath):
    print(f"\n📥 Carregando DST: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    if 'time_tag' not in df.columns:
        raise ValueError("❌ DST sem coluna time_tag")

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    dst_col = df.columns[1] if len(df.columns) > 1 else 'dst'
    df['dst'] = pd.to_numeric(df[dst_col], errors='coerce')

    df = df.dropna(subset=['time_tag', 'dst'])
    df = df[(df['dst'] > -1000) & (df['dst'] < 500)]

    print(f"   ✅ DST carregado: {len(df)} pontos")
    print(f"   • Dst range: {df['dst'].min():.1f} – {df['dst'].max():.1f} nT")
    return df.sort_values('time_tag').reset_index(drop=True)


# =========================
# MERGE CAUSAL
# =========================
def merge_data(omni, dst):
    print("\n🔗 Merge OMNI + DST (backward, tolerância 30min)...")

    df = pd.merge_asof(
        omni.sort_values('time_tag'),
        dst.sort_values('time_tag'),
        on='time_tag',
        direction='backward',
        tolerance=pd.Timedelta("30min"),
        suffixes=('', '_dst')
    )

    if 'dst' not in df.columns:
        dst_cols = [c for c in df.columns if 'dst' in c.lower()]
        if dst_cols:
            df['dst'] = df[dst_cols[0]]
        else:
            raise ValueError("❌ Coluna Dst não encontrada após merge")

    before = len(df)
    df = df.dropna(subset=['dst'])
    print(f"   ✅ Merge concluído: {len(df)} pontos (removidos {before - len(df)} sem Dst)")
    return df


# =========================
# VERIFICAÇÃO DE INTEGRIDADE DO DST
# =========================
def check_event_integrity(df, name, start, end):
    mask = (df['time_tag'] >= start) & (df['time_tag'] <= end)
    event = df[mask]

    if len(event) == 0:
        print(f"   ❌ NENHUM dado encontrado para {name}")
        return False, event

    dst_min = event['dst'].min()
    expected = EXPECTED_DST_MIN.get(name, -50)

    if dst_min > -100 or dst_min > expected * 0.7:
        print(f"   ⚠️ ALERTA: Dst mínimo observado ({dst_min:.1f} nT) muito acima do esperado (~{expected} nT)")
        print(f"      → Dados Dst não confiáveis. Validação ignorada.")
        return False, event

    print(f"   ✅ Dst do evento íntegro: mínimo {dst_min:.1f} nT")
    return True, event


# =========================
# CALIBRAÇÃO GLOBAL
# =========================
def global_calibration(df_train):
    print("\n📊 Calibração global com dt real...")

    core = HACCoreModel(HACCoreConfig())

    _, comp = compute_hac(
        time=df_train['time_tag'].values,
        bz=df_train['bz_gsm'].values,
        v=df_train['speed'].values,
        density=df_train['density'].values,
        config=core.config
    )

    coupling = comp['coupling']
    corr = np.corrcoef(coupling, df_train['dst'])[0, 1]
    print(f"   • Corr(coupling, Dst): {corr:.3f}")

    if abs(corr) < 0.2:
        print("   ⚠️ Correlação muito baixa – calibração pode não ser confiável.")

    times = pd.to_datetime(df_train['time_tag']).values
    dt = np.diff(times).astype('timedelta64[s]').astype(float)
    dt = np.insert(dt, 0, np.median(dt))
    dt = np.maximum(dt, 1.0)

    core.fit_calibration(coupling, dt, df_train['dst'].values)

    hac_raw = comp['raw']
    hac_ref = np.percentile(hac_raw, 99.5)
    core.config.HAC_REF = np.clip(hac_ref, 100, 5000)

    print(f"   • HAC_REF: {core.config.HAC_REF:.1f}")
    print(f"   • Q_FACTOR: {core.config.Q_FACTOR:.5f}")

    return core.config


# =========================
# VALIDAÇÃO DE UM EVENTO (LAG TEMPORAL REAL)
# =========================
def validate_event(config_core, df, name, start, end, is_test=True):
    print(f"\n🌌 {name} {'[TESTE]' if is_test else '[TREINO]'}")

    is_valid, event = check_event_integrity(df, name, start, end)
    if not is_valid:
        return None

    if len(event) < 20:
        print("   ⚠️ Poucos pontos no evento")
        return None

    print(f"   • Pontos: {len(event)}")
    print(f"   • Período: {event['time_tag'].min()} a {event['time_tag'].max()}")

    # Calcular campos físicos (coupling ajustado)
    event = PhysicalFieldsCalculator.compute_all_fields(event)

    # ========== LAG FÍSICO BASEADO EM TEMPO REAL ==========
    times = event['time_tag'].values.astype('datetime64[s]')
    lag_hours = 1.5e6 / (event['speed'] * 3600.0)
    lag_hours = np.clip(lag_hours, 0.3, 1.5)

    shifted = np.zeros(len(event))
    for i in range(len(event)):
        lag_sec = lag_hours.iloc[i] * 3600.0
        target_time = times[i] - np.timedelta64(int(lag_sec), 's')
        j = np.searchsorted(times, target_time)
        if j < len(event):
            shifted[i] = event['coupling_signal'].iloc[j]
        else:
            shifted[i] = 0.0

    event['coupling_signal'] = shifted

    # Configurar modelo com parâmetros calibrados
    physics_config = HACPhysicsConfig()
    physics_config.HAC_REF = config_core.HAC_REF

    model = ProductionHACModel(config=physics_config)

    hac = model.compute_hac_system(event)
    _, dst_pred, levels = model.predict_storm_indicators(hac)

    # NENHUM ajuste pós-predição – toda a física está dentro do modelo
    dst_pred = np.clip(dst_pred, DST_MIN_PHYSICAL, DST_MAX_PHYSICAL)

    metrics = evaluate_event(
        time=event['time_tag'].values,
        dst_obs=event['dst'].values,
        dst_pred=dst_pred
    )

    print(f"   • Dst predito mínimo: {dst_pred.min():.1f} nT")
    print(f"   • Correlação: {metrics['correlation']:.3f}")
    print(f"   • MAE: {metrics['MAE']:.1f} nT")
    print(f"   • Erro do mínimo Dst: {metrics['min_Dst_error_nT']:.1f} nT")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(event['time_tag'], event['dst'], 'b-', label='Dst observado')
    plt.plot(event['time_tag'], dst_pred, 'r--', label='Dst previsto')
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
    print("🚀 HAC++ VALIDAÇÃO FINAL (LAG TEMPORAL REAL + GANHO DINÂMICO)")
    print("=" * 70)

    omni = load_omni(OMNI_FILE)
    dst = load_dst(DST_FILE)
    df = merge_data(omni, dst)

    # Verificação direta do Halloween no DST original
    print("\n🔍 VERIFICAÇÃO DIRETA DO ARQUIVO DST (Halloween 2003):")
    halloween_dst = dst[(dst['time_tag'] >= "2003-10-28") & (dst['time_tag'] <= "2003-11-02")]
    if len(halloween_dst) > 0:
        print(f"   • Mínimo Dst no arquivo: {halloween_dst['dst'].min():.1f} nT")
        print(f"   • Número de registros: {len(halloween_dst)}")
        if halloween_dst['dst'].min() > -100:
            print("   ❌ ARQUIVO DST NÃO CONTÉM A TEMPESTADE REAL! Obtenha um arquivo íntegro.")
    else:
        print("   ❌ Nenhum dado DST encontrado para o período!")

    # Divisão treino/teste
    df_train = df[df['time_tag'] <= TRAIN_END_DATE].copy()
    df_test = df[df['time_tag'] > TRAIN_END_DATE].copy()

    print(f"\n📊 Divisão:")
    print(f"   • Treino: {len(df_train)} pontos")
    print(f"   • Teste:  {len(df_test)} pontos")

    config_core = global_calibration(df_train)

    results_test = []
    for name, (start, end) in EVENTS.items():
        is_test = pd.to_datetime(start) > pd.to_datetime(TRAIN_END_DATE)
        metrics = validate_event(config_core, df, name, start, end, is_test)
        if metrics and is_test:
            results_test.append(metrics)

    if results_test:
        print("\n" + "=" * 70)
        print("📈 MÉTRICAS MÉDIAS (EVENTOS DE TESTE ÍNTEGROS)")
        print("=" * 70)
        avg_corr = np.mean([m['correlation'] for m in results_test])
        avg_mae = np.mean([m['MAE'] for m in results_test])
        avg_min_err = np.mean([m['min_Dst_error_nT'] for m in results_test])
        print(f"   • Correlação média: {avg_corr:.3f}")
        print(f"   • MAE média: {avg_mae:.1f} nT")
        print(f"   • Erro médio do mínimo Dst: {avg_min_err:.1f} nT")
    else:
        print("\n⚠️ Nenhum evento de teste pôde ser validado (dados não íntegros).")

    print("\n✅ Validação concluída.")


if __name__ == "__main__":
    main()
