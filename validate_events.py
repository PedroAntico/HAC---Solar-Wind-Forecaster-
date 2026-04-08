#!/usr/bin/env python3
"""
validate_events.py - Validação robusta do modelo HAC++ em eventos históricos
Corrigido para leitura de CSV (não JSON)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# =========================
# IMPORTA FUNÇÕES DO MODELO
# =========================
try:
    from hac_final import (
        normalize_omni_columns,
        PhysicalFieldsCalculator,
        ProductionHACModel,
        HACPhysicsConfig
    )
except ImportError as e:
    raise ImportError("❌ Não foi possível importar hac_final.py.") from e

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
# LEITURA E PRÉ-PROCESSAMENTO DO OMNI (CSV)
# =========================
def load_and_clean_omni_csv(filepath):
    """
    Carrega arquivo CSV do OMNI, normaliza colunas, converte tipos e interpola.
    Substitui o método JSON-based do RobustOMNIProcessor.
    """
    print(f"📥 Carregando OMNI CSV: {filepath}")
    df = pd.read_csv(filepath)
    
    # Normaliza nomes de colunas
    df = normalize_omni_columns(df, allow_partial=True)
    
    # Converte time_tag para datetime
    df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
    df = df.dropna(subset=['time_tag']).sort_values('time_tag').reset_index(drop=True)
    
    # Converte colunas numéricas
    numeric_cols = ['speed', 'density', 'bz_gsm', 'bx_gsm', 'by_gsm', 'bt']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Aplica limites físicos (opcional)
    config = HACPhysicsConfig()
    if 'speed' in df.columns:
        df['speed'] = df['speed'].clip(lower=config.VSW_MIN, upper=config.VSW_MAX)
    if 'density' in df.columns:
        df['density'] = df['density'].clip(lower=config.DENSITY_MIN, upper=config.DENSITY_MAX)
    if 'bz_gsm' in df.columns:
        df['bz_gsm'] = df['bz_gsm'].clip(lower=config.BZ_MIN, upper=config.BZ_MAX)
    
    # Interpola valores ausentes (máximo 3 gaps consecutivos)
    cols_to_interp = ['bz_gsm', 'speed', 'density']
    for col in cols_to_interp:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=3, limit_direction='both')
    
    # Preenche eventuais NaNs restantes com valores padrão
    if 'speed' in df.columns:
        df['speed'] = df['speed'].fillna(400)
    if 'density' in df.columns:
        df['density'] = df['density'].fillna(5)
    if 'bz_gsm' in df.columns:
        df['bz_gsm'] = df['bz_gsm'].fillna(0)
    
    print(f"   ✅ OMNI processado: {len(df)} pontos")
    return df

def load_data(omni_path="data/omni_2000_2020.csv",
              dst_path="data/dst_kyoto_2000_2020.csv"):
    """
    Carrega OMNI (CSV) e DST (CSV), faz merge temporal.
    """
    # ----- OMNI -----
    omni = load_and_clean_omni_csv(omni_path)
    
    # ----- DST -----
    print(f"📥 Carregando DST CSV: {dst_path}")
    dst = pd.read_csv(dst_path)
    dst.columns = [c.strip().lower() for c in dst.columns]
    
    if 'time_tag' not in dst.columns or 'dst' not in dst.columns:
        raise KeyError(f"DST precisa das colunas 'time_tag' e 'dst'. Encontradas: {dst.columns}")
    
    dst['time_tag'] = pd.to_datetime(dst['time_tag'], errors='coerce')
    dst = dst.dropna(subset=['time_tag', 'dst']).sort_values('time_tag')
    print(f"   ✅ DST processado: {len(dst)} pontos")
    
    # ----- MERGE -----
    print("🔗 Merge OMNI + DST...")
    df = pd.merge_asof(
        omni.sort_values('time_tag'),
        dst.sort_values('time_tag'),
        on='time_tag',
        direction='backward',
        tolerance=pd.Timedelta("2h")
    )
    df = df.dropna(subset=['dst'])
    print(f"   ✅ Merge final: {len(df)} pontos com DST")
    
    if len(df) < 100:
        raise ValueError("❌ Poucos dados após merge. Verifique timestamps.")
    
    return df

# =========================
# EXECUTA MODELO HAC++
# =========================
def run_hac_model(df):
    """Aplica o modelo HAC++ completo e adiciona colunas de resultado."""
    print("⚙️ Executando modelo HAC++...")
    df = PhysicalFieldsCalculator.compute_all_fields(df)
    config = HACPhysicsConfig()
    model = ProductionHACModel(config)
    hac_total = model.compute_hac_system(df)
    
    for key, values in model.results.items():
        if key != 'time':
            df[key] = values
    return df, model

# =========================
# CALIBRAÇÃO LINEAR
# =========================
def calibrate_hac_to_dst(hac_values, dst_values):
    mask = ~(np.isnan(hac_values) | np.isnan(dst_values))
    X = hac_values[mask].reshape(-1, 1)
    y = dst_values[mask]
    if len(X) < 10:
        raise ValueError("Poucos pontos para calibração.")
    reg = LinearRegression().fit(X, y)
    slope, intercept = reg.coef_[0], reg.intercept_
    dst_pred = slope * hac_values + intercept
    return dst_pred, slope, intercept

# =========================
# VALIDAÇÃO DE EVENTO
# =========================
def validate_event(df, name, start, end, output_dir="."):
    print(f"\n🌌 EVENTO: {name}")
    mask = (df['time_tag'] >= start) & (df['time_tag'] <= end)
    event_df = df[mask].copy()
    
    if len(event_df) < 20:
        print("⚠️ Poucos dados, pulando...")
        return
    
    if 'HAC_total' not in event_df.columns:
        print("❌ HAC_total não encontrado.")
        return
    
    try:
        dst_pred, slope, intercept = calibrate_hac_to_dst(
            event_df['HAC_total'].values,
            event_df['dst'].values
        )
        event_df['Dst_pred'] = dst_pred
    except ValueError as e:
        print(f"⚠️ Falha na calibração: {e}")
        return
    
    corr, _ = pearsonr(event_df['dst'], event_df['Dst_pred'])
    mae = mean_absolute_error(event_df['dst'], event_df['Dst_pred'])
    
    print(f"   Dst real min : {event_df['dst'].min():.1f} nT")
    print(f"   Dst pred min : {event_df['Dst_pred'].min():.1f} nT")
    print(f"   Correlação   : {corr:.3f}")
    print(f"   MAE          : {mae:.1f} nT")
    print(f"   Pontos       : {len(event_df)}")
    print(f"   Calibração   : Dst = {slope:.2f} * HAC + {intercept:.1f}")
    
    # Salva CSV
    event_df.to_csv(f"{output_dir}/event_{name}.csv", index=False)
    
    # Gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(event_df['time_tag'], event_df['dst'], 'k-', label='Dst Real', linewidth=2)
    plt.plot(event_df['time_tag'], event_df['Dst_pred'], 'r--', label='HAC++ (calibrado)', linewidth=2)
    plt.axhline(-50, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(-100, color='orange', linestyle=':', alpha=0.7)
    plt.xlabel('Tempo')
    plt.ylabel('Dst (nT)')
    plt.title(f'Validação HAC++ - Evento {name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/event_{name}.png", dpi=150)
    plt.close()
    
    return event_df

# =========================
# MAIN
# =========================
def main():
    print("🔬 VALIDAÇÃO DO MODELO HAC++ EM EVENTOS HISTÓRICOS\n")
    
    # Ajuste os caminhos conforme necessário
    df = load_data(
        omni_path="data/omni_2000_2020.csv",
        dst_path="data/dst_kyoto_2000_2020.csv"
    )
    
    df, model = run_hac_model(df)
    
    for name, (start, end) in EVENTS.items():
        validate_event(df, name, start, end, output_dir=".")
    
    print("\n✅ Validação concluída.")

if __name__ == "__main__":
    main()
