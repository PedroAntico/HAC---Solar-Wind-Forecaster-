#!/usr/bin/env python3
"""
validate_events.py - HAC++ VALIDAÇÃO FINAL (VERSÃO FÍSICA CORRIGIDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from hac_final import (
    ProductionHACModel,
    normalize_omni_columns,
    PhysicalFieldsCalculator
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
CALIBRATION_FILE = "hac_calibration.json"

DST_MIN_PHYSICAL = -500
DST_MAX_PHYSICAL = 50


# =========================
# LOAD
# =========================
def load_omni(filepath):
    print(f"\n📥 Carregando OMNI: {filepath}")

    df = pd.read_csv(filepath)
    df = normalize_omni_columns(df, allow_partial=True)

    # =========================
    # Tempo
    # =========================
    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df = df.dropna(subset=['time_tag']).sort_values('time_tag')

    # =========================
    # Conversão numérica
    # =========================
    for col, default in [('speed', 400), ('density', 5), ('bz_gsm', 0)]:
        df[col] = pd.to_numeric(df.get(col, default), errors='coerce')

    # =========================
    # Remover valores OMNI inválidos
    # =========================
    invalid_values = [
        999, 9999, 99999, 999999, 9999999,
        999.9, 9999.9, 99999.9
    ]

    for col in ['speed', 'density', 'bz_gsm']:
        df[col] = df[col].replace(invalid_values, np.nan)

    # =========================
    # 🔥 CORREÇÃO AUTOMÁTICA DE UNIDADE (CRÍTICO)
    # =========================
    if df['speed'].median() > 2000:
        print("   ⚠️ Detectado speed em m/s → convertendo para km/s")
        df['speed'] = df['speed'] / 1000.0

    # =========================
    # Preencher faltantes (após correção)
    # =========================
    df['speed'] = df['speed'].fillna(method='ffill').fillna(400)
    df['density'] = df['density'].fillna(method='ffill').fillna(5)
    df['bz_gsm'] = df['bz_gsm'].fillna(0)

    # =========================
    # 🔥 FILTRO FÍSICO INTELIGENTE
    # =========================
    mask = (
        (df['speed'] > 250) & (df['speed'] < 1200) &
        (df['density'] > 0.1) & (df['density'] < 50) &
        (df['bz_gsm'] > -80) & (df['bz_gsm'] < 80)
    )

    removed = len(df) - mask.sum()
    df = df[mask]

    df = df.reset_index(drop=True)

    # =========================
    # Diagnóstico final
    # =========================
    print(f"   ✅ {len(df)} pontos válidos após limpeza")
    print(f"   🧹 Removidos: {removed}")
    print(f"   • V range: {df['speed'].min():.1f} – {df['speed'].max():.1f} km/s")
    print(f"   • Bz range: {df['bz_gsm'].min():.1f} – {df['bz_gsm'].max():.1f} nT")
    print(f"   • Density range: {df['density'].min():.1f} – {df['density'].max():.1f}")

    # =========================
    # ⚠️ ALERTAS DE QUALIDADE
    # =========================
    if df['speed'].std() < 20:
        print("   ⚠️ ALERTA: velocidade quase constante → dados suspeitos")

    if len(df) < 100000:
        print("   ⚠️ ALERTA: dataset pequeno demais para validação robusta")

    return df

def load_dst(filepath):
    print(f"\n📥 Carregando DST: {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    # =========================
    # Garantir colunas
    # =========================
    if 'time_tag' not in df.columns:
        raise ValueError("❌ Coluna time_tag não encontrada no DST")

    # Detectar coluna DST automaticamente
    if 'dst' not in df.columns:
        for col in df.columns:
            if 'dst' in col:
                df.rename(columns={col: 'dst'}, inplace=True)
                break

    if 'dst' not in df.columns:
        raise ValueError(f"❌ Coluna DST não encontrada. Colunas: {df.columns}")

    # =========================
    # Conversão
    # =========================
    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df['dst'] = pd.to_numeric(df['dst'], errors='coerce')

    # =========================
    # Limpeza
    # =========================
    df = df.dropna(subset=['time_tag', 'dst'])
    df = df.sort_values('time_tag').reset_index(drop=True)

    print(f"   ✅ {len(df)} pontos DST válidos")
    print(f"   • Dst range: {df['dst'].min():.1f} – {df['dst'].max():.1f}")

    return df

def merge_data(omni, dst):
    print("\n🔗 Fazendo merge OMNI + DST...")

    df = pd.merge_asof(
        omni.sort_values('time_tag'),
        dst.sort_values('time_tag'),
        on='time_tag',
        direction='nearest',
        tolerance=pd.Timedelta("1h"),
        suffixes=('_omni', '_dst')
    )

    # 🔥 GARANTIR DST
    if 'dst' not in df.columns:
        if 'dst_dst' in df.columns:
            df['dst'] = df['dst_dst']
        elif 'dst_y' in df.columns:
            df['dst'] = df['dst_y']
        else:
            print("❌ Colunas disponíveis após merge:")
            print(df.columns)
            raise ValueError("DST não encontrado após merge")

    # remover lixo
    df = df.dropna(subset=['dst'])

    print(f"   ✅ final: {len(df)} pontos")
    return df


# =========================
# CALIBRAÇÃO CORRIGIDA
# =========================
def global_calibration(df_train, core_model):

    print("\n📊 Calibração física...")

    _, comp = compute_hac(
        time=df_train['time_tag'].values,
        bz=df_train['bz_gsm'].values,
        v=df_train['speed'].values,
        density=df_train['density'].values,
        config=core_model.config
    )

    coupling = comp['coupling']

    # 🔥 NORMALIZAÇÃO FÍSICA CRÍTICA
    scale = np.percentile(np.abs(coupling), 95)
    coupling = coupling / max(scale, 1e-6)

    # 🔍 Diagnóstico físico
    corr = np.corrcoef(coupling, df_train['dst'])[0,1]
    print(f"   Correlação coupling vs Dst: {corr:.3f}")

    # HAC_REF mais robusto
    hac_raw = comp['raw']
    hac_ref = np.percentile(hac_raw, 99.5)
    core_model.config.HAC_REF = np.clip(hac_ref, 100, 5000)

    print(f"   HAC_REF: {core_model.config.HAC_REF:.1f}")

    # dt
    time_arr = pd.to_datetime(df_train['time_tag']).values
    dt = np.diff(time_arr).astype('timedelta64[s]').astype(float)
    dt = np.insert(dt, 0, dt[0])
    dt = np.maximum(dt, 1.0)

    # Fit real
    core_model.fit_calibration(coupling, dt, df_train['dst'].values)

    print(f"   Q_FACTOR: {core_model.config.Q_FACTOR:.5f}")

    # salvar
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({
            "HAC_REF": core_model.config.HAC_REF,
            "Q_FACTOR": core_model.config.Q_FACTOR,
            "TAU_DST": core_model.config.TAU_DST,
            "DST_Q": core_model.config.DST_Q
        }, f, indent=4)

    return core_model.config


# =========================
# EVENTO
# =========================
def validate_event(config, df, name, start, end, is_test=True):

    print(f"\n🌌 {name} {'[TESTE]' if is_test else '[TREINO]'}")

    # =========================
    # Corte do evento
    # =========================
    event = df[(df['time_tag'] >= start) & (df['time_tag'] <= end)].copy()

    if len(event) < 20:
        print("⚠️ poucos dados")
        return None

    # =========================
    # Campos físicos
    # =========================
    event = PhysicalFieldsCalculator.compute_all_fields(event)

    # =========================
    # 🔥 Lag físico (ESSENCIAL)
    # =========================
    lag_hours = (1500000 / event['speed']) / 3600.0
    lag_hours = np.clip(lag_hours, 0.3, 1.5)
    lag_steps = lag_hours.round().astype(int)

    shifted = np.zeros(len(event))
    for i in range(len(event)):
        step = lag_steps.iloc[i]
        if i - step >= 0:
            shifted[i] = event['coupling_signal'].iloc[i - step]
        else:
            shifted[i] = 0

    event['coupling_signal'] = shifted
    event = event.dropna()

    # =========================
    # Modelo HAC++
    # =========================
    model = ProductionHACModel(config=config)

    hac = model.compute_hac_system(event)
    _, dst_pred, levels = model.predict_storm_indicators(hac)

    dst_pred = np.clip(dst_pred, DST_MIN_PHYSICAL, DST_MAX_PHYSICAL)

    # =========================
    # Métricas
    # =========================
    metrics = evaluate_event(
        time=event['time_tag'].values,
        dst_obs=event['dst'].values,
        dst_pred=dst_pred
    )

    # =========================
    # Diagnóstico
    # =========================
    print(f"   Dst real mín: {event['dst'].min():.1f} nT")
    print(f"   Dst pred mín: {dst_pred.min():.1f} nT")
    print(f"   Correlação:   {metrics['correlation']:.3f}")
    print(f"   MAE:          {metrics['MAE']:.1f} nT")
    print(f"   Timing erro:  {metrics.get('peak_time_error_min', 0):.0f} min")

    # =========================
    # Sanidade física
    # =========================
    if np.max(event['speed']) > 2000:
        print("⚠️ ALERTA: velocidades irreais detectadas (dados OMNI sujos)")

    if np.max(event['coupling_signal']) > 1e5:
        print("⚠️ ALERTA: coupling explosivo (possível erro de escala)")

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

    core = HACCoreModel(HACCoreConfig())

    config = global_calibration(df_train, core)

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
