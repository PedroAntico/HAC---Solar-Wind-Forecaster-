#!/usr/bin/env python3
"""
validate_events.py - Validação HAC++ final (lag suavizado, sem clipping agressivo)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hac_final import (
    ProductionHACModel,
    PhysicalFieldsCalculator,
    HACPhysicsConfig,
    normalize_omni_columns
)
from hac_core import evaluate_event

# =========================
# CONFIGURAÇÕES
# =========================
EVENTS = {
    "Halloween_2003": ("2003-10-28", "2003-11-02"),
    "St_Patrick_2015": ("2015-03-17", "2015-03-20"),
    "Carrington_like_2012": ("2012-07-22", "2012-07-25"),
}

OMNI_FILE = "data/omni_2000_2020.csv"
DST_FILE = "data/dst_kyoto_2000_2020.csv"

# =========================
# CARREGAMENTO DST (IAGA-2002)
# =========================
def load_dst_kyoto(filepath):
    print(f"\n📥 Carregando DST Kyoto: {filepath}")
    data_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('|') or line.startswith('DATE'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                data_lines.append(parts)
    if not data_lines:
        raise ValueError("❌ Nenhum dado DST")
    df = pd.DataFrame(data_lines, columns=['date', 'time', 'doy', 'dst'])
    df['time_tag'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df['dst'] = pd.to_numeric(df['dst'], errors='coerce')
    df = df.dropna(subset=['time_tag', 'dst']).sort_values('time_tag')
    df = df[(df['dst'] > -1000) & (df['dst'] < 500)]
    print(f"   ✅ {len(df)} pontos, Dst range: {df['dst'].min():.1f} – {df['dst'].max():.1f} nT")
    return df[['time_tag', 'dst']]

# =========================
# CARREGAMENTO OMNI (SEM DST)
# =========================
def load_omni_clean(filepath):
    print(f"\n📥 Carregando OMNI: {filepath}")
    df = pd.read_csv(filepath)
    df = normalize_omni_columns(df, allow_partial=True)
    if 'time_tag' not in df.columns:
        raise ValueError("❌ OMNI sem time_tag")
    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df = df.dropna(subset=['time_tag']).sort_values('time_tag')
    if 'dst' in df.columns:
        print("   ⚠️ Removendo DST do OMNI")
        df = df.drop(columns=['dst'])
    if 'speed' in df.columns and df['speed'].median() > 2000:
        print("   ⚠️ Convertendo m/s → km/s")
        df['speed'] = df['speed'] / 1000.0
    for col, default in [('speed', 400), ('density', 5), ('bz_gsm', 0)]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
    print(f"   ✅ {len(df)} pontos, V: {df['speed'].min():.0f}–{df['speed'].max():.0f} km/s, Bz: {df['bz_gsm'].min():.1f}–{df['bz_gsm'].max():.1f} nT")
    return df

# =========================
# LAG DINÂMICO (MÉDIA LOCAL)
# =========================
def compute_dynamic_lag(speed):
    lag = 1.5e6 / (speed * 3600.0)
    return np.clip(lag, 0.3, 1.5)

# =========================
# PREPARA EVENTO (LAG SUAVIZADO)
# =========================
def prepare_event_data(omni_df, dst_df, start, end):
    dst_event = dst_df[(dst_df['time_tag'] >= start) & (dst_df['time_tag'] <= end)].copy()
    if len(dst_event) == 0:
        return None
    dst_event = dst_event.sort_values('time_tag')
    times_dst = dst_event['time_tag'].values

    omni_hist = omni_df[omni_df['time_tag'] <= end].copy().sort_values('time_tag')
    times_omni = omni_hist['time_tag'].values

    indices = []
    for t_dst in times_dst:
        idx_now = np.searchsorted(times_omni, t_dst, side='right') - 1
        idx_now = max(0, min(idx_now, len(omni_hist)-1))
        # Média local da velocidade (janela de 4 pontos)
        w_start = max(0, idx_now - 3)
        w_end = min(len(omni_hist), idx_now + 1)
        speed_avg = omni_hist.iloc[w_start:w_end]['speed'].mean()
        lag_h = compute_dynamic_lag(speed_avg)
        target_time = t_dst - pd.Timedelta(hours=lag_h)
        idx_lag = np.searchsorted(times_omni, target_time, side='right') - 1
        idx_lag = max(0, min(idx_lag, len(omni_hist)-1))
        indices.append(idx_lag)

    event_data = omni_hist.iloc[indices].copy()
    event_data['time_tag'] = times_dst
    event_data['dst'] = dst_event['dst'].values
    event_data = event_data.reset_index(drop=True)

    # Reuso de índices OMNI
    reuse_ratio = 1 - (len(set(indices)) / len(indices))
    if reuse_ratio > 0.3:
        print(f"   ⚠️ Alto reuso de índices OMNI ({reuse_ratio*100:.1f}%)")
    return event_data

# =========================
# VALIDAÇÃO
# =========================
def validate_event(omni_df, dst_df, name, start, end):
    print(f"\n🌌 {name}")
    event = prepare_event_data(omni_df, dst_df, start, end)
    if event is None or len(event) < 10:
        print("   ⚠️ Dados insuficientes")
        return None

    print(f"   • Pontos: {len(event)}")
    print(f"   • Dst obs mín: {event['dst'].min():.1f} nT")

    # Amostra
    print("\n   🔍 Amostra:")
    print(event[['time_tag', 'dst', 'bz_gsm', 'speed']].head(2).to_string(index=False))

    # Campos físicos
    event = PhysicalFieldsCalculator.compute_all_fields(event)

    # Diagnóstico do coupling (após remoção do clipping agressivo)
    print(f"   • Coupling max: {event['coupling_signal'].max():.2f}")
    print(f"   • Coupling mean: {event['coupling_signal'].mean():.2f}")

    # Modelo
    model = ProductionHACModel()
    hac = model.compute_hac_system(event)
    _, dst_pred, _ = model.predict_storm_indicators(hac)
    dst_pred = np.clip(dst_pred, -500, 50)

    metrics = evaluate_event(event['time_tag'].values, event['dst'].values, dst_pred)

    print(f"   • Dst pred mín: {dst_pred.min():.1f} nT")
    print(f"   • Corr: {metrics['correlation']:.3f}")
    print(f"   • MAE: {metrics['MAE']:.1f} nT")
    print(f"   • Erro mín: {metrics['min_Dst_error_nT']:.1f} nT")

    # Plot
    plt.figure(figsize=(12,5))
    plt.plot(event['time_tag'], event['dst'], 'b-', label='Kyoto')
    plt.plot(event['time_tag'], dst_pred, 'r--', label='HAC++')
    plt.axhline(-50, c='gray', ls=':', alpha=0.5)
    plt.axhline(-100, c='orange', ls=':', alpha=0.5)
    plt.axhline(-200, c='red', ls=':', alpha=0.5)
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
    print("="*70)
    print("🚀 HAC++ VALIDAÇÃO FINAL (SEM CLIPPING, LAG SUAVIZADO)")
    print("="*70)

    omni = load_omni_clean(OMNI_FILE)
    dst = load_dst_kyoto(DST_FILE)

    print("\n🔍 Verificação Halloween 2003:")
    h = dst[(dst['time_tag'] >= "2003-10-28") & (dst['time_tag'] <= "2003-11-02")]
    if len(h) > 0:
        print(f"   • Dst mín Kyoto: {h['dst'].min():.1f} nT")
    else:
        print("   ❌ Sem dados")

    results = []
    for name, (start, end) in EVENTS.items():
        m = validate_event(omni, dst, name, start, end)
        if m:
            results.append(m)

    if results:
        print("\n" + "="*70)
        print("📈 MÉTRICAS MÉDIAS")
        print("="*70)
        print(f"   • Correlação: {np.mean([m['correlation'] for m in results]):.3f}")
        print(f"   • MAE: {np.mean([m['MAE'] for m in results]):.1f} nT")
        print(f"   • Erro mín Dst: {np.mean([m['min_Dst_error_nT'] for m in results]):.1f} nT")
    else:
        print("\n⚠️ Nenhum evento validado.")

    print("\n✅ Concluído.")

if __name__ == "__main__":
    main()
