#!/usr/bin/env python3
"""
validate_events.py - Validação robusta do modelo HAC em eventos históricos
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# =========================
# IMPORT DO SEU MODELO
# =========================
try:
    from hac_final import normalize_omni_columns
except ImportError:
    raise ImportError("❌ Não foi possível importar hac_final.py")

# =========================
# EVENTOS HISTÓRICOS
# =========================
EVENTS = {
    "Halloween_2003": ("2003-10-28", "2003-11-02"),
    "St_Patrick_2015": ("2015-03-17", "2015-03-20"),
    "Carrington_like_2012": ("2012-07-22", "2012-07-25"),
    "May_2024": ("2024-05-10", "2024-05-12"),
}

# =========================
# LOAD DE DADOS ROBUSTO
# =========================
def load_data(
    omni_path="data/omni_data.csv",
    dst_path="data/dst_kyoto_2000_2020.csv"
):
    print("📥 Carregando dados...")

    omni = pd.read_csv(omni_path)
    dst = pd.read_csv(dst_path)

    # Normalizar nomes
    omni = normalize_omni_columns(omni, allow_partial=True)

    # Converter tempo
    omni['time_tag'] = pd.to_datetime(omni['time_tag'], errors='coerce')
    dst['time_tag'] = pd.to_datetime(dst['time_tag'], errors='coerce')

    omni = omni.sort_values('time_tag').dropna(subset=['time_tag'])
    dst = dst.sort_values('time_tag').dropna(subset=['time_tag'])

    print(f"   OMNI: {len(omni)} pontos")
    print(f"   DST : {len(dst)} pontos")

    # =========================
    # MERGE ROBUSTO
    # =========================
    print("🔗 Fazendo merge temporal...")

    df = pd.merge_asof(
        omni,
        dst,
        on="time_tag",
        direction="nearest",
        tolerance=pd.Timedelta("10min")  # 🔥 crítico
    )

    # Garantir coluna
    if 'dst' not in df.columns:
        raise ValueError(f"❌ Coluna 'dst' não encontrada: {df.columns}")

    # Remover NaN após merge
    df = df.dropna(subset=['dst'])

    print(f"   ✅ Merge final: {len(df)} pontos válidos")

    if len(df) < 100:
        raise ValueError("❌ Poucos dados após merge")

    return df


# =========================
# MODELO SIMPLES DST (BASELINE)
# =========================
def simple_dst_model(df):
    """
    Modelo simplificado tipo Burton (baseline físico)
    """
    bz = df['bz_gsm'].fillna(0).values
    v = df['speed'].fillna(400).values

    bz_south = np.maximum(0, -bz)
    Ey = bz_south * v * 1e-3

    # Integração simples
    dst = np.zeros(len(Ey))

    for i in range(1, len(Ey)):
        dst[i] = dst[i-1] + (Ey[i] - dst[i-1] / 7)

    return -dst * 20


# =========================
# VALIDAÇÃO POR EVENTO
# =========================
def validate_event(df, name, start, end):
    print(f"\n🌌 EVENTO: {name}")

    mask = (df['time_tag'] >= start) & (df['time_tag'] <= end)
    event_df = df[mask].copy()

    if len(event_df) < 20:
        print("⚠️ Poucos dados, pulando...")
        return

    # Modelo
    pred = simple_dst_model(event_df)

    # Calibração linear
    reg = LinearRegression().fit(pred.reshape(-1,1), event_df['dst'])
    pred_cal = reg.predict(pred.reshape(-1,1))

    # Métricas
    corr = pearsonr(event_df['dst'], pred_cal)[0]
    mae = mean_absolute_error(event_df['dst'], pred_cal)

    print(f"   Dst real min : {event_df['dst'].min():.1f}")
    print(f"   Dst pred min : {pred_cal.min():.1f}")
    print(f"   Correlação   : {corr:.3f}")
    print(f"   MAE          : {mae:.1f} nT")
    print(f"   Pontos       : {len(event_df)}")

    # Salvar
    event_df['Dst_pred'] = pred_cal
    event_df.to_csv(f"event_{name}.csv", index=False)


# =========================
# MAIN
# =========================
def main():
    print("🔬 VALIDANDO EVENTOS HISTÓRICOS\n")

    df = load_data()

    for name, (start, end) in EVENTS.items():
        validate_event(df, name, start, end)

    print("\n✅ Validação concluída")


if __name__ == "__main__":
    main()
