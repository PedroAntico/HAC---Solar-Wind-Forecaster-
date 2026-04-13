#!/usr/bin/env python3
"""
validate_events.py - Validação científica usando o MODELO DE PRODUÇÃO HAC++

Arquitetura correta:
- Calibração global com HACCoreModel (parâmetros físicos).
- Cálculo de campos físicos (PhysicalFieldsCalculator) antes da predição.
- Injeção dos parâmetros calibrados no ProductionHACModel.
- Validação de eventos com o mesmo pipeline usado em produção.
- Proteção contra valores não físicos de Dst (clipping).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# =========================
# IMPORTS DO MODELO DE PRODUÇÃO E CORE
# =========================
from hac_final import (
    ProductionHACModel,
    normalize_omni_columns,
    PhysicalFieldsCalculator
)
from hac_core import HACCoreModel, HACCoreConfig, evaluate_event, compute_hac

# =========================
# EVENTOS PARA VALIDAÇÃO
# =========================
EVENTS = {
    "Halloween_2003": ("2003-10-28", "2003-11-02"),
    "St_Patrick_2015": ("2015-03-17", "2015-03-20"),
    "Carrington_like_2012": ("2012-07-22", "2012-07-25"),
}

# Limites físicos para Dst (segurança contra valores absurdos)
DST_MIN_PHYSICAL = -500   # nT (pode ser ajustado conforme necessidade)
DST_MAX_PHYSICAL = 50     # nT

# =========================
# CARREGAMENTO OMNI
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

    # Fallback físico
    df['speed'] = df.get('speed', 400).fillna(400)
    df['density'] = df.get('density', 5).fillna(5)
    df['bz_gsm'] = df.get('bz_gsm', 0).fillna(0)

    print(f"   ✅ {len(df)} pontos OMNI")
    return df

# =========================
# CARREGAMENTO DST
# =========================
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

# =========================
# MERGE OMNI + DST
# =========================
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
# CALIBRAÇÃO GLOBAL (com HACCoreModel)
# =========================
def global_calibration(df, core_model):
    """
    Realiza a calibração global de HAC_REF e Q_FACTOR usando todo o dataset.
    Utiliza o acoplamento unificado do core.
    Retorna o objeto HACCoreConfig calibrado.
    """
    print("\n📊 CALIBRAÇÃO GLOBAL DO MODELO...")

    # 1. Calcular acoplamento unificado
    _, comp = compute_hac(
        time=df['time_tag'].values,
        bz=df['bz_gsm'].values,
        v=df['speed'].values,
        density=df['density'].values,
        config=core_model.config
    )
    coupling = comp['coupling']

    # 2. Calibrar HAC_REF (com remoção de outliers extremos)
    hac_raw = comp['raw']
    threshold = np.percentile(hac_raw, 90)
    high_values = hac_raw[hac_raw > threshold]

    if len(high_values) > 0:
        upper_limit = np.percentile(high_values, 99.5)
        high_values = high_values[high_values < upper_limit]
        core_model.config.HAC_REF = np.median(high_values) * 3.0
    else:
        core_model.config.HAC_REF = np.percentile(hac_raw, 99)

    print(f"   HAC_REF calibrado: {core_model.config.HAC_REF:.2f}")

    # 3. Preparar dt robusto
    time_arr = pd.to_datetime(df['time_tag']).values
    dt = np.zeros(len(df))
    if len(dt) > 1:
        dt[1:] = (time_arr[1:] - time_arr[:-1]).astype('timedelta64[s]').astype(float)
        dt[0] = dt[1]
    else:
        dt[0] = 60.0
    dt = np.maximum(dt, 1.0)   # evitar zeros

    # 4. Calibrar Q_FACTOR
    core_model.fit_calibration(coupling, dt, df['dst'].values)
    print(f"   Q_FACTOR calibrado: {core_model.config.Q_FACTOR:.4f}")

    # Salvar calibração em JSON (para uso no modelo de produção)
    calib_data = {
        "HAC_REF": core_model.config.HAC_REF,
        "Q_FACTOR": core_model.config.Q_FACTOR,
        "TAU_DST": core_model.config.TAU_DST,
        "DST_Q": core_model.config.DST_Q,
        "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("hac_calibration.json", "w") as f:
        json.dump(calib_data, f, indent=4)
    print("   Calibração salva em 'hac_calibration.json'")

    return core_model.config

# =========================
# VALIDAÇÃO DE UM EVENTO (USANDO MODELO DE PRODUÇÃO)
# =========================
def validate_event(calibrated_config, df, name, start, end):
    """
    Valida um evento usando o ProductionHACModel (modelo completo de produção).
    calibrated_config: objeto HACCoreConfig calibrado (do core).
    """
    print(f"\n🌌 {name}")

    mask = (df['time_tag'] >= start) & (df['time_tag'] <= end)
    event = df[mask].copy()

    if len(event) < 20:
        print("   ⚠️ poucos dados")
        return None

    # 1. Calcular campos físicos (necessários para o modelo de produção)
    print("   🔄 Calculando campos físicos...")
    event = PhysicalFieldsCalculator.compute_all_fields(event)

    # 2. Instanciar modelo de produção
    model = ProductionHACModel()

    # 3. Injeta os parâmetros calibrados no core físico interno
    model.core.config.HAC_REF = calibrated_config.HAC_REF
    model.core.config.Q_FACTOR = calibrated_config.Q_FACTOR
    model.core.config.TAU_DST = calibrated_config.TAU_DST
    model.core.config.DST_Q = calibrated_config.DST_Q

    # 4. Executa a cadeia completa de produção
    hac_values = model.compute_hac_system(event)
    kp_pred, dst_pred_raw, storm_levels = model.predict_storm_indicators(hac_values)

    # 5. Proteção contra valores não físicos de Dst (clipping)
    dst_pred = np.clip(dst_pred_raw, DST_MIN_PHYSICAL, DST_MAX_PHYSICAL)
    if not np.array_equal(dst_pred, dst_pred_raw):
        print(f"   ⚠️ Dst previsto continha valores fora do intervalo físico [{DST_MIN_PHYSICAL}, {DST_MAX_PHYSICAL}] e foi clipado.")

    # Adiciona previsão ao DataFrame
    event['Dst_pred'] = dst_pred
    event['Storm_level'] = storm_levels

    # 6. Métricas científicas
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

    # Contagem de níveis G4/G5 (para referência)
    g4g5_count = sum(1 for l in storm_levels if "G4" in l or "G5" in l)
    print(f"   Eventos G4/G5: {g4g5_count}")

    # 7. Salvar resultados completos
    event.to_csv(f"event_{name}_production.csv", index=False)

    # 8. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(event['time_tag'], event['dst'], 'b-', label='Dst observado', linewidth=2)
    plt.plot(event['time_tag'], event['Dst_pred'], 'r--', label='Dst previsto (modelo produção)', linewidth=2)
    plt.axhline(y=-50, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=-100, color='orange', linestyle=':', alpha=0.5)
    plt.axhline(y=-200, color='red', linestyle=':', alpha=0.5)
    plt.xlabel('Tempo (UTC)')
    plt.ylabel('Dst [nT]')
    plt.title(f'{name} - Modelo de Produção HAC++\n'
              f'Corr={metrics["correlation"]:.3f}, MAE={metrics["MAE"]:.0f} nT, '
              f'Erro mín={metrics["min_Dst_error_nT"]:.0f} nT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"event_{name}_production.png", dpi=150)
    plt.close()

    return metrics

# =========================
# MAIN
# =========================
def main():
    print("🚀 VALIDAÇÃO HAC++ COM MODELO DE PRODUÇÃO")
    print("=" * 70)

    # 1. Carregar dados
    omni = load_omni("data/omni_2000_2020.csv")
    dst = load_dst("data/dst_kyoto_2000_2020.csv")
    df = merge_data(omni, dst)

    # 2. Calibração global com HACCoreModel
    core_config = HACCoreConfig()
    core_model = HACCoreModel(core_config)
    calibrated_config = global_calibration(df, core_model)

    # 3. Validar cada evento usando a configuração calibrada
    all_metrics = []
    for name, (start, end) in EVENTS.items():
        metrics = validate_event(calibrated_config, df, name, start, end)
        if metrics:
            all_metrics.append(metrics)

    # 4. Resumo geral
    if all_metrics:
        print("\n" + "=" * 70)
        print("📈 RESUMO GERAL DAS MÉTRICAS")
        print("=" * 70)
        avg_corr = np.mean([m['correlation'] for m in all_metrics])
        avg_mae = np.mean([m['MAE'] for m in all_metrics])
        avg_time_err = np.mean([abs(m['peak_time_error_min']) for m in all_metrics])
        avg_min_err = np.mean([m['min_Dst_error_nT'] for m in all_metrics])

        print(f"Correlação média:        {avg_corr:.3f}")
        print(f"MAE médio:               {avg_mae:.1f} nT")
        print(f"Erro timing médio:       {avg_time_err:.0f} min")
        print(f"Erro mínimo Dst médio:   {avg_min_err:.1f} nT")

    print("\n✅ VALIDAÇÃO CONCLUÍDA")
    print("   Resultados salvos em event_*_production.csv e event_*_production.png")
    print("   Calibração salva em hac_calibration.json")

if __name__ == "__main__":
    main()
