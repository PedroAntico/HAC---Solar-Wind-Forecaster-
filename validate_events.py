#!/usr/bin/env python3
"""
validate_events.py - Validação robusta HAC++ (VERSÃO FINAL ESTÁVEL)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# =========================
# IMPORT DO MODELO
# =========================
from hac_final import (
    normalize_omni_columns,
    PhysicalFieldsCalculator,
    ProductionHACModel,
    HACPhysicsConfig
)

# =========================
# EVENTOS
# =========================
EVENTS = {
    "Halloween_2003": ("2003-10-28", "2003-11-02"),
    "St_Patrick_2015": ("2015-03-17", "2015-03-20"),
    "Carrington_like_2012": ("2012-07-22", "2012-07-25"),
}

# =========================
# CARREGAMENTO OMNI (CSV REAL)
# =========================
def load_omni(filepath):
    print(f"\n📥 OMNI: {filepath}")

    df = pd.read_csv(filepath)
    df = normalize_omni_columns(df, allow_partial=True)

    if 'time_tag' not in df.columns:
        raise ValueError("❌ OMNI sem coluna time_tag")

    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df = df.dropna(subset=['time_tag']).sort_values('time_tag')

    # numéricos
    for col in ['speed', 'density', 'bz_gsm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # fallback físico
    df['speed'] = df.get('speed', 400).fillna(400)
    df['density'] = df.get('density', 5).fillna(5)
    df['bz_gsm'] = df.get('bz_gsm', 0).fillna(0)

    print(f"   ✅ {len(df)} pontos OMNI")
    return df


# =========================
# CARREGAMENTO DST (ROBUSTO)
# =========================
def load_dst(filepath):
    print(f"\n📥 DST: {filepath}")

    df = pd.read_csv(filepath)

    # normalizar nomes
    df.columns = [c.strip().lower() for c in df.columns]

    print("   🔍 colunas DST:", df.columns.tolist())

    # detectar coluna dst automaticamente
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
# MERGE
# =========================
def merge_data(omni, dst):
    print("\n🔗 Merge OMNI + DST...")

    df = pd.merge_asof(
        omni.sort_values('time_tag'),
        dst.sort_values('time_tag'),
        on='time_tag',
        direction='backward',
        tolerance=pd.Timedelta("2h")
    )

    print("   🔍 NaNs em DST após merge:", df['dst'].isna().sum())

    df = df.dropna(subset=['dst'])

    print(f"   ✅ final: {len(df)} pontos")

    if len(df) < 100:
        raise ValueError("❌ Merge falhou (dados insuficientes)")

    return df


# =========================
# MODELO HAC++
# =========================
def run_model(df):
    print("\n⚙️ Rodando HAC++...")

    df = PhysicalFieldsCalculator.compute_all_fields(df)

    model = ProductionHACModel(HACPhysicsConfig())
    hac = model.compute_hac_system(df)

    for k, v in model.results.items():
        if k != 'time':
            df[k] = v

    return df


# =========================
# CALIBRAÇÃO
# =========================
def calibrate(hac, dst):
    mask = ~(np.isnan(hac) | np.isnan(dst))

    X = hac[mask].reshape(-1, 1)
    y = dst[mask]

    if len(X) < 20:
        raise ValueError("Poucos dados")

    reg = LinearRegression().fit(X, y)

    pred = reg.predict(hac.reshape(-1, 1))

    return pred, reg.coef_[0], reg.intercept_


# =========================
# VALIDA EVENTO
# =========================
def validate_event(df, name, start, end):
    print(f"\n🌌 {name}")

    event = df[(df['time_tag'] >= start) & (df['time_tag'] <= end)].copy()

    if len(event) < 20:
        print("⚠️ poucos dados")
        return

    pred, a, b = calibrate(event['HAC_total'].values, event['dst'].values)
    event['Dst_pred'] = pred

    corr = pearsonr(event['dst'], event['Dst_pred'])[0]
    mae = mean_absolute_error(event['dst'], event['Dst_pred'])

    print(f"   min real: {event['dst'].min():.1f}")
    print(f"   min pred: {event['Dst_pred'].min():.1f}")
    print(f"   corr: {corr:.3f}")
    print(f"   mae: {mae:.1f}")
    print(f"   fit: Dst = {a:.2f} * HAC + {b:.1f}")

    # salvar
    event.to_csv(f"event_{name}.csv", index=False)

    # plot
    plt.figure(figsize=(10,5))
    plt.plot(event['time_tag'], event['dst'], label="Dst real")
    plt.plot(event['time_tag'], event['Dst_pred'], '--', label="HAC")
    plt.legend()
    plt.grid()
    plt.savefig(f"event_{name}.png")
    plt.close()


# =========================
# MAIN
# =========================
def main():
    print("🚀 VALIDAÇÃO HAC++\n")

    omni = load_omni("data/omni_2000_2020.csv")
    dst = load_dst("data/dst_kyoto_2000_2020.csv")

    df = merge_data(omni, dst)
    df = run_model(df)

    for name, (start, end) in EVENTS.items():
        validate_event(df, name, start, end)

    print("\n✅ FINALIZADO")


if __name__ == "__main__":
    main()
