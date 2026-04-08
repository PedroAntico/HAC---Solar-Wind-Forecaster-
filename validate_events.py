#!/usr/bin/env python3
"""
validate_events.py - Validação em eventos históricos reais
Reutiliza as funções do seu modelo principal
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# ====================== IMPORTAR SUAS FUNÇÕES ======================
# Ajuste o nome do arquivo principal se for diferente
from hac_final import prepare_omni, euler_dst   # ← mude para o nome correto do seu arquivo

# ====================== EVENTOS HISTÓRICOS ======================
EVENTS = {
    "Halloween_2003": ("2003-10-28", "2003-11-02"),
    "St_Patrick_2015": ("2015-03-17", "2015-03-20"),
    "Carrington_like_2012": ("2012-07-22", "2012-07-25"),
    "May_2024": ("2024-05-10", "2024-05-12"),   # se seus dados chegarem até aqui
}

def load_and_prepare_data(omni_path="data/omni_2000_2020.csv", 
                         dst_path="data/dst_kyoto_2000_2020.csv"):
    omni = pd.read_csv(omni_path, parse_dates=['time_tag'])
    omni = prepare_omni(omni)

    dst = pd.read_csv(dst_path, parse_dates=['time_tag'])
    df = pd.merge_asof(omni, dst, on='time_tag', direction='nearest', 
                       tolerance=pd.Timedelta('2h')).dropna(subset=['dst'])
    return df

def run_event_validation(df, event_name, start, end, alpha=5.5, tau=8.0):
    mask = (df['time_tag'] >= start) & (df['time_tag'] <= end)
    event_df = df[mask].copy().reset_index(drop=True)

    if len(event_df) < 10:
        print(f"⚠️ Evento {event_name} tem poucos dados.")
        return

    # Rodar o modelo (Burton otimizado)
    Dst_raw = euler_dst(event_df, alpha=alpha, tau=tau)

    # Calibração linear (usando todo o treino, mas aplicando aqui)
    # Se quiser calibração global, carregue do treino uma vez só
    reg = LinearRegression().fit(Dst_raw.reshape(-1,1), event_df['dst'])
    Dst_calib = reg.predict(Dst_raw.reshape(-1,1))

    # Métricas
    corr = pearsonr(event_df['dst'], Dst_calib)[0]
    mae = mean_absolute_error(event_df['dst'], Dst_calib)
    dst_min_real = event_df['dst'].min()
    dst_min_pred = Dst_calib.min()

    print(f"\n🌌 EVENTO: {event_name} ({start} → {end})")
    print(f"   Dst mínimo real : {dst_min_real:.1f} nT")
    print(f"   Dst mínimo pred : {dst_min_pred:.1f} nT")
    print(f"   Correlação      : {corr:.3f}")
    print(f"   MAE             : {mae:.1f} nT")
    print(f"   Nº de pontos    : {len(event_df)}")

    # Salvar para plot (opcional)
    event_df['Dst_pred'] = Dst_calib
    event_df.to_csv(f"event_{event_name.lower()}.csv", index=False)

def main():
    print("🔬 Iniciando validação em eventos históricos...\n")
    
    df = load_and_prepare_data()

    for name, (start, end) in EVENTS.items():
        run_event_validation(df, name, start, end, alpha=5.5, tau=8.0)

    print("\n✅ Validação concluída. Arquivos .csv gerados para cada evento.")

if __name__ == "__main__":
    main()
