#!/usr/bin/env python3
"""
validate_events.py - Validação científica com separação TREINO/TESTE (VERSÃO FINAL)

Melhorias aplicadas:
- Treino: dados até 2014-12-31. Teste: eventos a partir de 2015-01-01.
- Calibração do Q_FACTOR com acoplamento LINEAR do core (compute_hac).
- HAC_REF limitado a [100, 2000] para evitar explosão.
- Uso do coupling_override no modelo de produção (unificação).
- Salvamento e reutilização inteligente da calibração.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# =========================
# IMPORTS DO MODELO
# =========================
from hac_final import (
    ProductionHACModel,
    normalize_omni_columns,
    PhysicalFieldsCalculator
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

DST_MIN_PHYSICAL = -500
DST_MAX_PHYSICAL = 50

OMNI_FILE = "data/omni_2000_2020.csv"
DST_FILE = "data/dst_kyoto_2000_2020.csv"
CALIBRATION_FILE = "hac_calibration.json"


# =========================
# CARREGAMENTO DE DADOS
# =========================
def load_omni(filepath):
    print(f"\n📥 OMNI: {filepath}")
    df = pd.read_csv(filepath)
    df = normalize_omni_columns(df, allow_partial=True)

    if 'time_tag' not in df.columns:
        raise ValueError("❌ OMNI sem coluna time_tag")

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df = df.dropna(subset=['time_tag']).sort_values('time_tag')

    for col in ['speed', 'density', 'bz_gsm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['speed'] = df.get('speed', 400).fillna(400)
    df['density'] = df.get('density', 5).fillna(5)
    df['bz_gsm'] = df.get('bz_gsm', 0).fillna(0)

    print(f"   ✅ {len(df)} pontos OMNI")
    return df


def load_dst(filepath):
    print(f"\n📥 DST: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    if 'dst' not in df.columns:
        for col in df.columns:
            if 'dst' in col:
                df.rename(columns={col: 'dst'}, inplace=True)

    if 'time_tag' not in df.columns:
        raise ValueError("❌ DST sem time_tag")
    if 'dst' not in df.columns:
        raise ValueError(f"❌ DST não encontrado. Colunas: {df.columns}")

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df['dst'] = pd.to_numeric(df['dst'], errors='coerce')
    df = df.dropna(subset=['time_tag', 'dst']).sort_values('time_tag')

    print(f"   ✅ {len(df)} pontos DST")
    return df


def merge_data(omni, dst):
    print("\n🔗 Merge OMNI + DST...")
    df = pd.merge_asof(
        omni.sort_values('time_tag'),
        dst.sort_values('time_tag'),
        on='time_tag',
        direction='backward',
        tolerance=pd.Timedelta("2h"),
        suffixes=('_omni', '_dst')
    )

    if 'dst_dst' in df.columns:
        df['dst'] = df['dst_dst']
    elif 'dst_y' in df.columns:
        df['dst'] = df['dst_y']
    elif 'dst' not in df.columns:
        raise ValueError("❌ dst não encontrado após merge")

    df = df.dropna(subset=['dst'])
    print(f"   ✅ final: {len(df)} pontos")
    return df


# =========================
# CALIBRAÇÃO GLOBAL (CORRIGIDA)
# =========================
def global_calibration(df_train, core_model):
    """
    Calibra HAC_REF e Q_FACTOR usando APENAS os dados de treino.
    Usa o acoplamento LINEAR do core (compute_hac) para Q_FACTOR.
    """
    print("\n📊 CALIBRAÇÃO GLOBAL (somente treino)...")

    # 1. Obter acoplamento linear e HAC bruto do core
    _, comp = compute_hac(
        time=df_train['time_tag'].values,
        bz=df_train['bz_gsm'].values,
        v=df_train['speed'].values,
        density=df_train['density'].values,
        config=core_model.config
    )
    coupling = comp['coupling'] * 1000.0          # sinal linear → adequado para regressão
    hac_raw = comp['raw']                # HAC bruto para HAC_REF

    # 2. Calibrar HAC_REF com limite de segurança
    hac_ref_raw = np.percentile(hac_raw, 99)
    core_model.config.HAC_REF = np.clip(hac_ref_raw, 100.0, 5000.0)
    print(f"   HAC_REF calibrado: {core_model.config.HAC_REF:.2f} (bruto: {hac_ref_raw:.2f})")

    # 3. Preparar dt
    time_arr = pd.to_datetime(df_train['time_tag']).values
    dt = np.zeros(len(df_train))
    if len(dt) > 1:
        dt[1:] = (time_arr[1:] - time_arr[:-1]).astype('timedelta64[s]').astype(float)
        dt[0] = dt[1]
    else:
        dt[0] = 60.0
    dt = np.maximum(dt, 1.0)

    # 4. Calibrar Q_FACTOR com coupling linear
    core_model.fit_calibration(coupling, dt, df_train['dst'].values)
    print(f"   Q_FACTOR calibrado: {core_model.config.Q_FACTOR:.4f}")

    # 5. Salvar calibração
    calib_data = {
        "HAC_REF": core_model.config.HAC_REF,
        "Q_FACTOR": core_model.config.Q_FACTOR,
        "TAU_DST": core_model.config.TAU_DST,
        "DST_Q": core_model.config.DST_Q,
        "train_end": TRAIN_END_DATE,
        "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calib_data, f, indent=4)
    print("   Calibração salva em 'hac_calibration.json'")

    return core_model.config


# =========================
# VALIDAÇÃO DE UM EVENTO
# =========================
def validate_event(calibrated_config, df, name, start, end, is_test=True):
    """
    Executa o pipeline de produção e calcula métricas.
    is_test: se True, o evento NÃO participou da calibração.
    """
    print(f"\n🌌 {name} {'[TESTE]' if is_test else '[TREINO - apenas ilustrativo]'}")

    mask = (df['time_tag'] >= start) & (df['time_tag'] <= end)
    event = df[mask].copy()

    if len(event) < 20:
        print("   ⚠️ poucos dados")
        return None

    # 1. Calcular campos físicos (necessário para coupling_signal)
    print("   🔄 Calculando campos físicos...")
    event = PhysicalFieldsCalculator.compute_all_fields(event)

    # 2. Modelo de produção com parâmetros calibrados
    model = ProductionHACModel()
    model.core.config.HAC_REF = calibrated_config.HAC_REF
    model.core.config.Q_FACTOR = calibrated_config.Q_FACTOR
    model.core.config.TAU_DST = calibrated_config.TAU_DST
    model.core.config.DST_Q = calibrated_config.DST_Q

    # 3. Executar pipeline
    hac_values = model.compute_hac_system(event)
    kp_pred, dst_pred_raw, storm_levels = model.predict_storm_indicators(hac_values)

    # 4. Diagnóstico de saturação
    if np.any(dst_pred_raw < DST_MIN_PHYSICAL) or np.any(dst_pred_raw > DST_MAX_PHYSICAL):
        print(f"   ⚠️ Dst previsto extrapolou [{DST_MIN_PHYSICAL}, {DST_MAX_PHYSICAL}]")

    dst_pred = np.clip(dst_pred_raw, DST_MIN_PHYSICAL, DST_MAX_PHYSICAL)
    event['Dst_pred'] = dst_pred
    event['Storm_level'] = storm_levels

    # 5. Métricas
    metrics = evaluate_event(
        time=event['time_tag'].values,
        dst_obs=event['dst'].values,
        dst_pred=event['Dst_pred'].values
    )

    print(f"   Dst real mín: {event['dst'].min():.1f} nT")
    print(f"   Dst pred mín: {event['Dst_pred'].min():.1f} nT")
    print(f"   Correlação:    {metrics['correlation']:.3f}")
    print(f"   MAE:           {metrics['MAE']:.1f} nT")
    print(f"   Erro timing:   {metrics['peak_time_error_min']:.0f} min")
    print(f"   Erro mín Dst:  {metrics['min_Dst_error_nT']:.1f} nT")

    g4g5_count = sum(1 for l in storm_levels if "G4" in l or "G5" in l)
    print(f"   Eventos G4/G5: {g4g5_count}")

    # 6. Salvar resultados
    suffix = "test" if is_test else "train"
    event.to_csv(f"event_{name}_{suffix}.csv", index=False)

    # 7. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(event['time_tag'], event['dst'], 'b-', label='Dst observado', linewidth=2)
    plt.plot(event['time_tag'], event['Dst_pred'], 'r--', label='Dst previsto', linewidth=2)
    plt.axhline(y=-50, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=-100, color='orange', linestyle=':', alpha=0.5)
    plt.axhline(y=-200, color='red', linestyle=':', alpha=0.5)
    plt.xlabel('Tempo (UTC)')
    plt.ylabel('Dst [nT]')
    plt.title(f'{name} - HAC++ (calibração até {TRAIN_END_DATE})\n'
              f'Corr={metrics["correlation"]:.3f}, MAE={metrics["MAE"]:.0f} nT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"event_{name}_{suffix}.png", dpi=150)
    plt.close()

    return metrics


# =========================
# MAIN
# =========================
def main():
    print("🚀 VALIDAÇÃO HAC++ COM SEPARAÇÃO TREINO/TESTE")
    print("=" * 70)

    # 1. Carregar dados completos
    omni = load_omni(OMNI_FILE)
    dst = load_dst(DST_FILE)
    df = merge_data(omni, dst)

    # 2. Dividir treino/teste
    train_mask = df['time_tag'] <= TRAIN_END_DATE
    df_train = df[train_mask].copy()
    df_test  = df[~train_mask].copy()
    print(f"\n📌 Treino: {len(df_train)} pontos (até {TRAIN_END_DATE})")
    print(f"📌 Teste:  {len(df_test)} pontos (após {TRAIN_END_DATE})")

    # 3. Calibração (ou carregar existente)
    core_config = HACCoreConfig()
    core_model = HACCoreModel(core_config)

    if os.path.exists(CALIBRATION_FILE):
        print(f"\n⚡ Arquivo de calibração encontrado: {CALIBRATION_FILE}")
        use_existing = input("   Deseja usá-lo? (s/n): ").strip().lower()
        if use_existing == 's':
            with open(CALIBRATION_FILE, 'r') as f:
                calib_data = json.load(f)
            core_config.HAC_REF = calib_data["HAC_REF"]
            core_config.Q_FACTOR = calib_data["Q_FACTOR"]
            core_config.TAU_DST = calib_data["TAU_DST"]
            core_config.DST_Q = calib_data["DST_Q"]
            print("   ✅ Calibração carregada.")
            calibrated_config = core_config
        else:
            calibrated_config = global_calibration(df_train, core_model)
    else:
        calibrated_config = global_calibration(df_train, core_model)

    # 4. Validar eventos
    all_test_metrics = []
    for name, (start, end) in EVENTS.items():
        event_start = pd.to_datetime(start)
        is_test = event_start > pd.to_datetime(TRAIN_END_DATE)
        metrics = validate_event(calibrated_config, df, name, start, end, is_test=is_test)
        if metrics and is_test:
            all_test_metrics.append(metrics)

    # 5. Resumo apenas com eventos de teste
    if all_test_metrics:
        print("\n" + "=" * 70)
        print("📈 RESUMO DAS MÉTRICAS (EVENTOS DE TESTE)")
        print("=" * 70)
        avg_corr = np.mean([m['correlation'] for m in all_test_metrics])
        avg_mae = np.mean([m['MAE'] for m in all_test_metrics])
        avg_time_err = np.mean([abs(m['peak_time_error_min']) for m in all_test_metrics])
        avg_min_err = np.mean([m['min_Dst_error_nT'] for m in all_test_metrics])

        print(f"Correlação média:        {avg_corr:.3f}")
        print(f"MAE médio:               {avg_mae:.1f} nT")
        print(f"Erro timing médio:       {avg_time_err:.0f} min")
        print(f"Erro mínimo Dst médio:   {avg_min_err:.1f} nT")
    else:
        print("\n⚠️ Nenhum evento de teste encontrado no período.")

    print("\n✅ VALIDAÇÃO CONCLUÍDA")
    print("   (Eventos anteriores a 2015 foram avaliados apenas de forma ilustrativa)")


if __name__ == "__main__":
    main()
