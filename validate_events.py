#!/usr/bin/env python3
"""
validate_events.py - Validação robusta do modelo HAC++ em eventos históricos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# =========================
# IMPORTA O MODELO HAC++
# =========================
try:
    from hac_final import (
        RobustOMNIProcessor,
        PhysicalFieldsCalculator,
        ProductionHACModel,
        HACPhysicsConfig
    )
except ImportError as e:
    raise ImportError("❌ Não foi possível importar hac_final.py. Verifique o nome do arquivo.") from e

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
# CARREGAMENTO ROBUSTO (OMNI + DST)
# =========================
def load_data(omni_path="data/omni_2000_2020.csv",
              dst_path="data/dst_kyoto_2000_2020.csv"):
    """
    Carrega OMNI e DST, faz merge temporal e retorna DataFrame unificado.
    """
    print("📥 Carregando dados...")

    # ----- OMNI -----
    processor = RobustOMNIProcessor()
    omni = processor.load_and_clean(omni_path)
    if omni is None:
        raise FileNotFoundError(f"Arquivo OMNI não encontrado: {omni_path}")
    print(f"   OMNI: {len(omni)} pontos")

    # ----- DST -----
    dst = pd.read_csv(dst_path)
    dst.columns = [c.strip().lower() for c in dst.columns]

    # Garantir nomes das colunas
    if 'time_tag' not in dst.columns:
        raise KeyError(f"Coluna 'time_tag' não encontrada no DST. Colunas: {dst.columns}")
    if 'dst' not in dst.columns:
        raise KeyError(f"Coluna 'dst' não encontrada no DST. Colunas: {dst.columns}")

    dst['time_tag'] = pd.to_datetime(dst['time_tag'], errors='coerce')
    dst = dst.dropna(subset=['time_tag', 'dst']).sort_values('time_tag')
    print(f"   DST : {len(dst)} pontos")

    # ----- MERGE TEMPORAL -----
    print("🔗 Fazendo merge OMNI + DST...")
    # Merge asof: para cada timestamp OMNI, pega o DST mais próximo (atrás ou igual)
    df = pd.merge_asof(
        omni.sort_values('time_tag'),
        dst.sort_values('time_tag'),
        on='time_tag',
        direction='backward',            # Usa o valor de DST imediatamente anterior
        tolerance=pd.Timedelta("2h")     # Até 2 horas de diferença
    )

    # Remove linhas sem DST
    df = df.dropna(subset=['dst'])
    print(f"   ✅ Merge final: {len(df)} pontos com DST")

    if len(df) < 100:
        raise ValueError("❌ Poucos pontos após merge. Verifique os arquivos de entrada.")

    return df

# =========================
# EXECUTA MODELO HAC++ NO DATAFRAME
# =========================
def run_hac_model(df):
    """
    Aplica o modelo HAC++ completo ao DataFrame.
    Retorna o DataFrame original acrescido das colunas HAC_total, dHAC_dt, etc.
    """
    print("⚙️ Executando modelo HAC++...")

    # Calcula campos físicos (E_field, coupling)
    df = PhysicalFieldsCalculator.compute_all_fields(df)

    # Instancia e executa modelo
    config = HACPhysicsConfig()
    model = ProductionHACModel(config)
    hac_total = model.compute_hac_system(df)

    # Adiciona resultados ao DataFrame
    for key, values in model.results.items():
        # 'time' já existe, vamos pular
        if key != 'time':
            df[key] = values

    # Ajuste de escala: HAC_total está normalizado para 0-300
    # Podemos manter assim ou calibrar para DST posteriormente
    return df, model

# =========================
# CALIBRAÇÃO LINEAR HAC → DST
# =========================
def calibrate_hac_to_dst(hac_values, dst_values):
    """
    Ajusta regressão linear para converter HAC_total em estimativa de DST.
    Retorna coeficientes (slope, intercept) e o DST predito.
    """
    # Remove NaNs
    mask = ~(np.isnan(hac_values) | np.isnan(dst_values))
    X = hac_values[mask].reshape(-1, 1)
    y = dst_values[mask]

    if len(X) < 10:
        raise ValueError("Poucos pontos para calibração.")

    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    # Aplica a todos os pontos (inclusive NaNs, mas depois tratamos)
    dst_pred = slope * hac_values + intercept
    return dst_pred, slope, intercept

# =========================
# VALIDAÇÃO DE UM EVENTO
# =========================
def validate_event(df, name, start, end, output_dir="."):
    print(f"\n🌌 EVENTO: {name}")

    # Filtra período do evento
    mask = (df['time_tag'] >= start) & (df['time_tag'] <= end)
    event_df = df[mask].copy()

    if len(event_df) < 20:
        print("⚠️ Poucos dados, pulando...")
        return

    # HAC já está calculado para todo o DataFrame
    if 'HAC_total' not in event_df.columns:
        print("❌ Coluna HAC_total não encontrada. Execute run_hac_model primeiro.")
        return

    # Calibração HAC -> DST usando APENAS dados do evento
    try:
        dst_pred, slope, intercept = calibrate_hac_to_dst(
            event_df['HAC_total'].values,
            event_df['dst'].values
        )
        event_df['Dst_pred'] = dst_pred
    except ValueError as e:
        print(f"⚠️ Falha na calibração: {e}")
        return

    # Métricas
    corr, p_val = pearsonr(event_df['dst'], event_df['Dst_pred'])
    mae = mean_absolute_error(event_df['dst'], event_df['Dst_pred'])

    # Resultados
    print(f"   Dst real min : {event_df['dst'].min():.1f} nT")
    print(f"   Dst pred min : {event_df['Dst_pred'].min():.1f} nT")
    print(f"   Correlação   : {corr:.3f}")
    print(f"   MAE          : {mae:.1f} nT")
    print(f"   Pontos       : {len(event_df)}")
    print(f"   Calibração   : Dst = {slope:.2f} * HAC + {intercept:.1f}")

    # Salva CSV
    event_df.to_csv(f"{output_dir}/event_{name}.csv", index=False)

    # Gera gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(event_df['time_tag'], event_df['dst'], 'k-', label='Dst Real', linewidth=2)
    plt.plot(event_df['time_tag'], event_df['Dst_pred'], 'r--', label='HAC++ (calibrado)', linewidth=2)
    plt.axhline(y=-50, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(y=-100, color='orange', linestyle=':', alpha=0.7)
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

    # 1. Carrega todos os dados (OMNI + DST)
    df = load_data(
        omni_path="data/omni_2000_2020.csv",      # ajuste para seu caminho
        dst_path="data/dst_kyoto_2000_2020.csv"   # ajuste para seu caminho
    )

    # 2. Executa modelo HAC++ uma única vez (para todos os dados)
    df, model = run_hac_model(df)

    # 3. Valida cada evento
    for name, (start, end) in EVENTS.items():
        validate_event(df, name, start, end, output_dir=".")

    print("\n✅ Validação concluída. Arquivos CSV e gráficos salvos.")

if __name__ == "__main__":
    main()
