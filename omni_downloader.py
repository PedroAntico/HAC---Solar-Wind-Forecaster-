"""
OMNI Data Downloader (2000-2020)
Baixa dados de vento solar e Dst da NASA para o período 2000-2020,
gerando um arquivo CSV limpo para uso no modelo HAC.

Uso:
    python omni_downloader.py
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from io import StringIO
from datetime import datetime, timedelta

def download_omni_month(year, month, resolution="hour"):
    """
    Baixa dados OMNI para um mês específico.
    Retorna DataFrame ou None em caso de falha.
    """
    start_date = f"{year}{month:02d}01"
    # Último dia do mês
    if month == 12:
        end_date = f"{year+1}0101"
    else:
        end_date = f"{year}{month+1:02d}01"
    end = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=1)
    end_date = end.strftime("%Y%m%d")
    
    url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
    payload = {
        "activity": "retrieve",
        "res": resolution,
        "spacecraft": "omni2",
        "start_date": start_date,
        "end_date": end_date,
        "vars": ["12", "13", "14", "21", "22", "26"],  # Bx, By, Bz, dens, vel, Dst
        "scale": "Linear",
        "view": "0",
        "table": "0"
    }
    print(f"📡 Baixando {start_date} → {end_date}...")
    for attempt in range(3):
        try:
            response = requests.post(url, data=payload, timeout=60)
            response.raise_for_status()
            break
        except Exception as e:
            print(f"   Tentativa {attempt+1} falhou: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                return None
    text = response.text
    lines = text.split("\n")
    data_lines = [l for l in lines if l.strip() and l[0].isdigit()]
    if not data_lines:
        print("   ⚠️ Nenhum dado encontrado.")
        return None
    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        delim_whitespace=True,
        header=None,
        na_values=[9999, 9999.99, 99999.0, 99999.99]
    )
    df.columns = [
        "year", "doy", "hour",
        "bx_gsm", "by_gsm", "bz_gsm",
        "density", "speed", "dst"
    ]
    # Criar timestamp
    df["time_tag"] = pd.to_datetime(df["year"], format="%Y") \
        + pd.to_timedelta(df["doy"] - 1, unit="D") \
        + pd.to_timedelta(df["hour"], unit="h")
    df = df[["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "density", "speed", "dst"]]
    df = df.dropna()
    return df

def download_omni_range(start_year, end_year):
    """
    Baixa dados de um intervalo de anos, mês a mês.
    Retorna DataFrame concatenado.
    """
    all_dfs = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            print(f"\n--- Ano {year}, mês {month:02d} ---")
            df_month = download_omni_month(year, month)
            if df_month is not None and len(df_month) > 0:
                all_dfs.append(df_month)
                print(f"   ✅ {len(df_month)} pontos")
            else:
                print("   ⚠️ Sem dados para este mês")
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)

def save_data(df, filename="data/omni_2000_2020.csv"):
    """Salva DataFrame em CSV, criando diretório se necessário."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"\n💾 Dados salvos em {filename}")

def main():
    print("="*70)
    print("🛰️  OMNI Data Downloader (2000-2020)")
    print("="*70)
    
    # Baixar dados
    df = download_omni_range(2000, 2020)
    if df is None:
        print("❌ Falha no download.")
        return
    
    print(f"\n✅ Total de pontos baixados: {len(df)}")
    print(f"   Período: {df['time_tag'].min()} a {df['time_tag'].max()}")
    
    # Estatísticas rápidas
    dst_min = df['dst'].min()
    print(f"   Dst mínimo: {dst_min:.1f} nT")
    if dst_min < -50:
        print("   ✅ O período contém tempestades geomagnéticas significativas.")
    else:
        print("   ⚠️ O período não contém tempestades fortes (Dst < -50 nT).")
    
    # Salvar
    save_data(df)
    
    print("\n✅ Download concluído.")

if __name__ == "__main__":
    main()
