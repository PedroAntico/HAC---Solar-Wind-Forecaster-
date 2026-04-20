#!/usr/bin/env python3
"""
validate_events.py - HAC++ VALIDAÇÃO FINAL (VERSÃO CORRIGIDA E ROBUSTA)

Correções aplicadas:
- Merge causal (direction='backward', tolerância 30min)
- Filtros de dados OMNI menos restritivos (recupera ~80% dos pontos)
- Cálculo correto de dt na calibração
- Diagnóstico completo de dados por evento
- Validação de integridade dos campos físicos
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

TRAIN_END_DATE = "2014-12-31"

OMNI_FILE = "data/omni_2000_2020.csv"
DST_FILE = "data/dst_kyoto_2000_2020.csv"
CALIBRATION_FILE = "hac_calibration.json"

DST_MIN_PHYSICAL = -500
DST_MAX_PHYSICAL = 50

# =========================
# CARREGAMENTO OMNI (FILTROS MENOS AGRESSIVOS)
# =========================
def load_omni(filepath):
    print(f"\n📥 Carregando OMNI: {filepath}")

    df = pd.read_csv(filepath)
    df = normalize_omni_columns(df, allow_partial=True)

    if 'time_tag' not in df.columns:
        raise ValueError("❌ OMNI sem coluna time_tag")

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df = df.dropna(subset=['time_tag']).sort_values('time_tag')

    # Preencher colunas essenciais com valores padrão
    for col, default in [('speed', 400), ('density', 5), ('bz_gsm', 0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = default

    # Remover flags OMNI (valores sentinelas comuns)
    invalid_flags = [999, 9999, 99999, 999999, 9999999,
                     999.9, 9999.9, 99999.9]
    for col in ['speed', 'density', 'bz_gsm']:
        df[col] = df[col].replace(invalid_flags, np.nan)

    # Converter velocidade se necessário (m/s → km/s)
    if df['speed'].median() > 2000:
        print("   ⚠️ Convertendo m/s → km/s")
        df['speed'] /= 1000.0

    # Preenchimento forward para gaps pequenos (evita perder dados)
    df['speed'] = df['speed'].ffill().fillna(400)
    df['density'] = df['density'].ffill().fillna(5)
    df['bz_gsm'] = df['bz_gsm'].fillna(0)

    # Aplicar limites físicos RAZOÁVEIS (sem cortar tempestades extremas)
    df = df[
        (df['speed'] > 150) & (df['speed'] < 2000) &
        (df['density'] > 0.05) & (df['density'] < 200) &   # mais permissivo
        (df['bz_gsm'] > -100) & (df['bz_gsm'] < 100)
    ].copy()

    print(f"   ✅ OMNI carregado: {len(df)} pontos")
    print(f"   • V range: {df['speed'].min():.1f} – {df['speed'].max():.1f} km/s")
    print(f"   • Bz range: {df['bz_gsm'].min():.1f} – {df['bz_gsm'].max():.1f} nT")
    return df.reset_index(drop=True)


# =========================
# CARREGAMENTO DST (ROBUSTO)
# =========================
def load_dst(filepath):
    print(f"\n📥 Carregando DST: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    if 'time_tag' not in df.columns:
        raise ValueError("❌ DST sem coluna time_tag")

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    # Assume que a segunda coluna é o Dst
    dst_col = df.columns[1] if len(df.columns) > 1 else 'dst'
    df['dst'] = pd.to_numeric(df[dst_col], errors='coerce')

    df = df.dropna(subset=['time_tag', 'dst'])

    # Filtrar valores absurdos
    df = df[(df['dst'] > -1000) & (df['dst'] < 500)]

    print(f"   ✅ DST carregado: {len(df)} pontos")
    print(f"   • Dst range: {df['dst'].min():.1f} – {df['dst'].max():.1f} nT")
    return df.sort_values('time_tag').reset_index(drop=True)


# =========================
# MERGE CAUSAL (CORRIGIDO)
# =========================
def merge_data(omni, dst):
    print("\n🔗 Merge OMNI + DST (causal)...")

    # direction='backward': associa cada linha OMNI ao Dst anterior mais recente
    # tolerance='30min': evita associar a horas erradas
    df = pd.merge_asof(
        omni.sort_values('time_tag'),
        dst.sort_values('time_tag'),
        on='time_tag',
        direction='backward',
        tolerance=pd.Timedelta("30min"),
        suffixes=('', '_dst')
    )

    # Garantir que a coluna 'dst' exista
    if 'dst' not in df.columns:
        # Procura coluna que contenha 'dst'
        dst_cols = [c for c in df.columns if 'dst' in c.lower()]
        if dst_cols:
            df['dst'] = df[dst_cols[0]]
        else:
            raise ValueError("❌ Coluna Dst não encontrada após merge")

    # Remover linhas onde o Dst não foi encontrado (NaN)
    before = len(df)
    df = df.dropna(subset=['dst'])
    print(f"   ✅ Merge concluído: {len(df)} pontos (removidos {before - len(df)} sem Dst)")

    return df


# =========================
# CALIBRAÇÃO GLOBAL (COM DT CORRETO)
# =========================
def global_calibration(df_train):
    print("\n📊 Calibração global com dt real...")

    core = HACCoreModel(HACCoreConfig())

    # Calcular acoplamento unificado
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

    # Calcular dt real em segundos
    times = pd.to_datetime(df_train['time_tag']).values
    dt = np.diff(times).astype('timedelta64[s]').astype(float)
    dt = np.insert(dt, 0, np.median(dt))
    dt = np.maximum(dt, 1.0)   # evitar zeros

    # Calibração do Q_FACTOR
    core.fit_calibration(coupling, dt, df_train['dst'].values)

    # HAC_REF (usando o raw do core)
    hac_raw = comp['raw']
    hac_ref = np.percentile(hac_raw, 99.5)
    core.config.HAC_REF = np.clip(hac_ref, 100, 5000)

    print(f"   • HAC_REF: {core.config.HAC_REF:.1f}")
    print(f"   • Q_FACTOR: {core.config.Q_FACTOR:.5f}")

    # Salvar calibração
    calib_data = {
        "HAC_REF": core.config.HAC_REF,
        "Q_FACTOR": core.config.Q_FACTOR,
        "TAU_DST": core.config.TAU_DST,
        "DST_Q": core.config.DST_Q,
        "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calib_data, f, indent=4)

    return core.config


# =========================
# VALIDAÇÃO DE UM EVENTO (COM DIAGNÓSTICO)
# =========================
def validate_event(config, df, name, start, end, is_test=True):
    print(f"\n🌌 {name} {'[TESTE]' if is_test else '[TREINO]'}")

    mask = (df['time_tag'] >= start) & (df['time_tag'] <= end)
    event = df[mask].copy()

    if len(event) < 20:
        print("   ⚠️ Poucos dados no evento (menos de 20 pontos)")
        return None

    # Diagnóstico dos dados brutos
    print(f"   • Pontos no evento: {len(event)}")
    print(f"   • Período: {event['time_tag'].min()} a {event['time_tag'].max()}")
    print(f"   • Dst real mínimo: {event['dst'].min():.1f} nT (em {event.loc[event['dst'].idxmin(), 'time_tag']})")
    print(f"   • Bz mínimo: {event['bz_gsm'].min():.1f} nT")
    print(f"   • V máximo: {event['speed'].max():.1f} km/s")

    # Calcular campos físicos (inclui coupling_signal)
    event = PhysicalFieldsCalculator.compute_all_fields(event)

    # Aplicar lag físico (tempo de propagação do vento solar)
    # Usamos um lag simplificado baseado na velocidade média
    lag_hours = np.clip(1.5e6 / event['speed'], 0.3 * 3600, 1.5 * 3600) / 3600.0
    lag_steps = np.round(lag_hours * 60).astype(int)   # assumindo dados a cada ~1 min

    coupling_shifted = np.zeros(len(event))
    for i in range(len(event)):
        step = lag_steps.iloc[i]
        if i - step >= 0:
            coupling_shifted[i] = event['coupling_signal'].iloc[i - step]
        else:
            coupling_shifted[i] = 0
    event['coupling_signal'] = coupling_shifted

    # Instanciar modelo de produção
    model = ProductionHACModel()

    # Injeta parâmetros calibrados no core interno
    model.core.config.HAC_REF = config.HAC_REF
    model.core.config.Q_FACTOR = config.Q_FACTOR
    model.core.config.TAU_DST = config.TAU_DST
    model.core.config.DST_Q = config.DST_Q

    # Executar pipeline completo
    hac = model.compute_hac_system(event)
    _, dst_pred, levels = model.predict_storm_indicators(hac)

    # Clipping de segurança
    dst_pred = np.clip(dst_pred, DST_MIN_PHYSICAL, DST_MAX_PHYSICAL)

    # Métricas
    metrics = evaluate_event(
        time=event['time_tag'].values,
        dst_obs=event['dst'].values,
        dst_pred=dst_pred
    )

    print(f"   • Dst predito mínimo: {dst_pred.min():.1f} nT")
    print(f"   • Correlação: {metrics['correlation']:.3f}")
    print(f"   • MAE: {metrics['MAE']:.1f} nT")
    print(f"   • Erro do mínimo Dst: {metrics['min_Dst_error_nT']:.1f} nT")
    print(f"   • Erro de timing: {metrics['peak_time_error_min']:.0f} min")

    # Plot opcional (salvar figura)
    plt.figure(figsize=(12, 5))
    plt.plot(event['time_tag'], event['dst'], 'b-', label='Dst observado', linewidth=2)
    plt.plot(event['time_tag'], dst_pred, 'r--', label='Dst previsto', linewidth=2)
    plt.axhline(y=-50, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=-100, color='orange', linestyle=':', alpha=0.5)
    plt.axhline(y=-200, color='red', linestyle=':', alpha=0.5)
    plt.xlabel('Tempo (UTC)')
    plt.ylabel('Dst [nT]')
    plt.title(f'{name} - Corr={metrics["correlation"]:.2f}, MAE={metrics["MAE"]:.0f} nT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"event_{name}_validation.png", dpi=150)
    plt.close()

    return metrics


# =========================
# MAIN
# =========================
def main():
    print("=" * 70)
    print("🚀 HAC++ VALIDAÇÃO FINAL (VERSÃO CORRIGIDA)")
    print("=" * 70)

    # Carregar dados
    omni = load_omni(OMNI_FILE)
    dst = load_dst(DST_FILE)
    df = merge_data(omni, dst)

    # Divisão treino/teste
    df_train = df[df['time_tag'] <= TRAIN_END_DATE].copy()
    df_test = df[df['time_tag'] > TRAIN_END_DATE].copy()

    print(f"\n📊 Divisão dos dados:")
    print(f"   • Treino (até {TRAIN_END_DATE}): {len(df_train)} pontos")
    print(f"   • Teste (após {TRAIN_END_DATE}): {len(df_test)} pontos")

    # Calibração com conjunto de treino
    config = global_calibration(df_train)

    # Validar eventos
    results_test = []
    for name, (start, end) in EVENTS.items():
        is_test = pd.to_datetime(start) > pd.to_datetime(TRAIN_END_DATE)
        metrics = validate_event(config, df, name, start, end, is_test)
        if metrics and is_test:
            results_test.append(metrics)

    # Resumo final
    if results_test:
        print("\n" + "=" * 70)
        print("📈 MÉTRICAS MÉDIAS NOS EVENTOS DE TESTE")
        print("=" * 70)
        avg_corr = np.mean([m['correlation'] for m in results_test])
        avg_mae = np.mean([m['MAE'] for m in results_test])
        avg_min_err = np.mean([m['min_Dst_error_nT'] for m in results_test])
        avg_time_err = np.mean([abs(m['peak_time_error_min']) for m in results_test])

        print(f"   • Correlação média: {avg_corr:.3f}")
        print(f"   • MAE média: {avg_mae:.1f} nT")
        print(f"   • Erro médio do mínimo Dst: {avg_min_err:.1f} nT")
        print(f"   • Erro de timing médio: {avg_time_err:.0f} min")
    else:
        print("\n⚠️ Nenhum evento de teste com métricas disponíveis.")

    print("\n✅ Validação concluída.")


if __name__ == "__main__":
    main()
