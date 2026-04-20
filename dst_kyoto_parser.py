#!/usr/bin/env python3
"""
Parser de arquivo Kyoto Dst (IAGA-2002) → CSV limpo

Entrada:
    data/WWW_dstae00870436.dat.txt

Saída:
    data/dst_kyoto_2000_2020.csv

Uso:
    python dst_kyoto_parser.py
"""

import pandas as pd
import os

INPUT_FILE = "data/WWW_dstae00870436.dat.txt"
OUTPUT_FILE = "data/dst_kyoto_2000_2020.csv"


def parse_kyoto_dst(filepath):
    data = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Ignora cabeçalho e linhas inválidas
            if not line or not line[0].isdigit():
                continue

            parts = line.split()

            # Esperado: DATE TIME DOY DST
            if len(parts) < 4:
                continue

            try:
                date_str = parts[0]
                time_str = parts[1]
                dst_val = float(parts[3])

                timestamp = pd.to_datetime(f"{date_str} {time_str}")

                data.append([timestamp, dst_val])

            except Exception:
                continue

    df = pd.DataFrame(data, columns=["time_tag", "dst_kyoto"])

    # Ordenar e remover duplicatas
    df = df.sort_values("time_tag").drop_duplicates("time_tag")

    return df


def main():
    print("="*60)
    print("🌍 Kyoto Dst Parser (2000–2020)")
    print("="*60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Arquivo não encontrado: {INPUT_FILE}")
        return

    df = parse_kyoto_dst(INPUT_FILE)

    if df.empty:
        print("❌ Nenhum dado válido encontrado.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Total de pontos: {len(df)}")
    print(f"📅 Período: {df['time_tag'].min()} → {df['time_tag'].max()}")
    print(f"📉 Dst mínimo: {df['dst_kyoto'].min():.1f} nT")

    print(f"\n💾 Salvo em: {OUTPUT_FILE}")
    print("🚀 Pronto para validação com HAC!")


if __name__ == "__main__":
    main()
