#!/usr/bin/env python3
"""
validate_events.py - HAC++ VALIDAÇÃO FINAL (VERSÃO ESTÁVEL E FÍSICA)
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
# CONFIG
# =========================
EVENTS = {
    "Halloween_2003": ("2003-10-28", "2003-11-02"),
    "St_Patrick_2015": ("2015-03-17", "2015-03-20"),
    "Carrington_like_2012": ("2012-07-22", "2012-07-25"),
}

TRAIN_END_DATE = "2014-12-31"

OMNI_FILE = "data/omni_2000_2020.csv"
DST_FILE = "data/dst_kyoto_2000_2020.csv"

DST_MIN_PHYSICAL = -500
DST_MAX_PHYSICAL = 50


# =========================
# LOAD OMNI (ROBUSTO)
# =========================
def load_omni(filepath):
    print(f"\n📥 Carregando OMNI: {filepath}")

    df = pd.read_csv(filepath)
    df = normalize_omni_columns(df, allow_partial=True)

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df = df.dropna(subset=['time_tag']).sort_values('time_tag')

    for col, default in [('speed', 400), ('density', 5), ('bz_gsm', 0)]:
        df[col] = pd.to_numeric(df.get(col, default), errors='coerce')

    # remover flags OMNI
    invalid = [999, 9999, 99999, 999999, 9999999,
               999.9, 9999.9, 99999.9]

    for col in ['speed', 'density', 'bz_gsm']:
        df[col] = df[col].replace(invalid, np.nan)

    # corrigir unidade
    if df['speed'].median() > 2000:
        print("   ⚠️ Convertendo m/s → km/s")
        df['speed'] /= 1000.0

    # preencher
    df['speed'] = df['speed'].ffill().fillna(400)
    df['density'] = df['density'].ffill().fillna(5)
    df['bz_gsm'] = df['bz_gsm'].fillna(0)

    # filtro físico (menos agressivo!)
    df = df[
        (df['speed'] > 150) & (df['speed'] < 2000) &
        (df['density'] > 0.05) & (df['density'] < 100) &
        (df['bz_gsm'] > -100) & (df['bz_gsm'] < 100)
    ]

    print(f"   ✅ {len(df)} pontos válidos")
    print(f"   • V range: {df['speed'].min():.1f} – {df['speed'].max():.1f}")

    return df.reset_index(drop=True)


# =========================
# LOAD DST (ROBUSTO)
# =========================
def load_dst(filepath):
    print(f"\n📥 Carregando DST: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df['dst'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

    df = df.dropna()

    # remover lixo absurdo
    df = df[(df['dst'] > -1000) & (df['dst'] < 500)]

    print(f"   ✅ {len(df)} pontos DST")
    print(f"   • Dst range: {df['dst'].min():.1f} – {df['dst'].max():.1f}")

    return df.sort_values('time_tag')


# =========================
# MERGE
# =========================
def merge_data(omni, dst):
    print("\n🔗 Merge OMNI + DST...")

    df = pd.merge_asof(
        omni.sort_values('time_tag'),
        dst.sort_values('time_tag'),
        on='time_tag',
        tolerance=pd.Timedelta("1h"),
        direction='nearest',
        suffixes=('', '_dst')
    )

    # =========================
    # 🔥 GARANTIR COLUNA DST
    # =========================
    possible_dst_cols = [c for c in df.columns if 'dst' in c.lower()]

    if not possible_dst_cols:
        print("❌ Colunas disponíveis:")
        print(df.columns)
        raise ValueError("DST não encontrado após merge")

    # pega a coluna correta automaticamente
    for col in possible_dst_cols:
        if col == 'dst':
            df['dst'] = df[col]
            break
        elif 'dst' in col:
            df['dst'] = df[col]
            break

    # =========================
    # LIMPEZA FINAL
    # =========================
    df = df.dropna(subset=['dst'])

    print(f"   ✅ final: {len(df)} pontos")
    return df


# =========================
# CALIBRAÇÃO
# =========================
def global_calibration(df_train):

    print("\n📊 Calibração...")

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

    print(f"   Corr(coupling, Dst): {corr:.3f}")

    core.fit_calibration(coupling, np.ones(len(coupling)), df_train['dst'].values)

    print(f"   Q_FACTOR: {core.config.Q_FACTOR:.5f}")

    return core.config


# =========================
# EVENTO
# =========================
def validate_event(config, df, name, start, end, is_test=True):

    print(f"\n🌌 {name} {'[TESTE]' if is_test else '[TREINO]'}")

    event = df[(df['time_tag'] >= start) & (df['time_tag'] <= end)].copy()

    if len(event) < 20:
        print("⚠️ poucos dados")
        return None

    event = PhysicalFieldsCalculator.compute_all_fields(event)

    # lag físico
    lag_h = np.clip((1500000 / event['speed']) / 3600, 0.3, 1.5)
    lag_steps = lag_h.round().astype(int)

    shifted = np.zeros(len(event))
    for i in range(len(event)):
        step = lag_steps.iloc[i]
        shifted[i] = event['coupling_signal'].iloc[i-step] if i-step >= 0 else 0

    event['coupling_signal'] = shifted

    # 🔥 NÃO USAR CORE CONFIG DIRETO
    model = ProductionHACModel()

    hac = model.compute_hac_system(event)
    _, dst_pred, _ = model.predict_storm_indicators(hac)

    dst_pred = np.clip(dst_pred, DST_MIN_PHYSICAL, DST_MAX_PHYSICAL)

    metrics = evaluate_event(
        time=event['time_tag'].values,
        dst_obs=event['dst'].values,
        dst_pred=dst_pred
    )

    print(f"   Dst real: {event['dst'].min():.1f}")
    print(f"   Dst pred: {dst_pred.min():.1f}")
    print(f"   Corr: {metrics['correlation']:.3f}")
    print(f"   MAE: {metrics['MAE']:.1f}")

    return metrics


# =========================
# MAIN
# =========================
def main():

    print("🚀 HAC++ VALIDAÇÃO FINAL")

    omni = load_omni(OMNI_FILE)
    dst = load_dst(DST_FILE)
    df = merge_data(omni, dst)

    df_train = df[df['time_tag'] <= TRAIN_END_DATE]
    df_test = df[df['time_tag'] > TRAIN_END_DATE]

    print(f"\nTreino: {len(df_train)}")
    print(f"Teste: {len(df_test)}")

    config = global_calibration(df_train)

    results = []

    for name, (start, end) in EVENTS.items():
        is_test = pd.to_datetime(start) > pd.to_datetime(TRAIN_END_DATE)
        m = validate_event(config, df, name, start, end, is_test)

        if m and is_test:
            results.append(m)

    if results:
        print("\n📊 MÉDIA TESTE")
        print("Corr:", np.mean([m['correlation'] for m in results]))
        print("MAE:", np.mean([m['MAE'] for m in results]))

    print("\n✅ FINALIZADO")


if __name__ == "__main__":
    main()
