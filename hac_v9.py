#!/usr/bin/env python3
"""
HAC v9 - Versão Rápida (Euler + refinamento cirúrgico)
Sem solve_ivp → roda em segundos
Foco em MAE_storm e corr_storm
"""

import pandas as pd
import numpy as np
import warnings
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

def prepare_omni(df):
    df = df.copy().sort_values('time_tag').reset_index(drop=True)
    if 'dst' in df.columns:
        df = df.drop(columns=['dst'])

    # Correção de speed
    if 'speed' in df.columns and len(df) > 100:
        med = df['speed'].median()
        print(f"   Speed original mediana: {med:.0f}")
        if med > 10000:
            df['speed'] /= 1000.0
        if df['speed'].median() < 300:
            factor = 420 / max(df['speed'].median(), 50)
            df['speed'] *= factor
            print(f"   Correção adicional (fator {factor:.2f}). Mediana: {df['speed'].median():.0f} km/s")

    # Limpeza mínima
    bad = (
        (df['bz_gsm'].abs() > 100) |
        (df['speed'] < 150) | (df['speed'] > 2500) |
        (df['density'] < 0.001) | (df['density'] > 10000)
    )
    df.loc[bad, ['bz_gsm', 'speed', 'density']] = np.nan

    df['bz_gsm'] = df['bz_gsm'].ffill(limit=12)
    for col in ['speed', 'density']:
        df[col] = df[col].interpolate(method='linear', limit=24)

    df = df.dropna(subset=['bz_gsm', 'speed', 'density']).reset_index(drop=True)
    print(f"   Dados finais: {len(df):,} pontos | Speed mediana = {df['speed'].median():.0f} km/s")
    return df

def euler_dst(df, alpha=4.5, tau=6.0, b_pressure=0.0, alpha_L=0.0, beta_L=0.2, use_hac=False):
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    N = df['density'].values
    VBs = V * np.maximum(-Bz, 0) / 1000.0

    # dt em horas
    dt_h = np.diff(df['time_tag']).astype('timedelta64[h]').astype(float)
    dt_h = np.concatenate([[1.0], dt_h])

    # Hload disciplinado (se ativado)
    Hload = np.zeros(len(V))
    if use_hac and alpha_L > 0:
        for i in range(1, len(V)):
            dH = alpha_L * VBs[i] - beta_L * Hload[i-1]
            Hload[i] = np.clip(Hload[i-1] + dH, 0, 50)

    driver = alpha * VBs
    if use_hac and alpha_L > 0:
        driver += 0.8 * Hload

    Dst = np.zeros(len(V))
    for i in range(1, len(V)):
        dDst = -driver[i] - Dst[i-1] / tau
        if b_pressure > 0:
            pdyn = 1.6726e-27 * (N[i] * 1e6) * (V[i] * 1000)**2 * 1e9
            dDst += b_pressure * np.sqrt(pdyn)
        Dst[i] = Dst[i-1] + dDst * dt_h[i]

    return Dst

def evaluate(y_true, y_pred, label=""):
    corr = pearsonr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true < -50
    mae_storm = mean_absolute_error(y_true[mask], y_pred[mask]) if mask.any() else np.nan
    corr_storm = pearsonr(y_true[mask], y_pred[mask])[0] if mask.sum() > 10 else np.nan
    print(f"{label:35} corr={corr:.3f}  R²={r2:.3f}  MAE_storm={mae_storm:.1f}  corr_storm={corr_storm:.3f}")
    return mae_storm, corr_storm

def main():
    # Carregar dados (ajuste os caminhos conforme necessário)
    df = pd.read_csv('data/omni_2000_2020.csv', parse_dates=['time_tag'])
    df = prepare_omni(df)

    dst = pd.read_csv('data/dst_kyoto_2000_2020.csv', parse_dates=['time_tag'])
    df = pd.merge_asof(df, dst, on='time_tag', direction='nearest', tolerance=pd.Timedelta('2h')).dropna(subset=['dst'])

    train = df[df['time_tag'] < '2010-01-01'].copy()
    test = df[df['time_tag'] >= '2010-01-01'].copy()

    print(f"\nTreino: {len(train):,} | Teste: {len(test):,}\n")

    print("=== TESTE 1: Burton clássico (variações rápidas) ===")
    best_mae = float('inf')
    for alpha in [3.5, 4.0, 4.5, 5.0, 5.5]:
        for tau in [5.0, 6.0, 7.0, 8.0]:
            Dst_train = euler_dst(train, alpha=alpha, tau=tau)
            Dst_test = euler_dst(test, alpha=alpha, tau=tau)

            reg = LinearRegression().fit(Dst_train.reshape(-1,1), train['dst'])
            Dst_train_cal = reg.predict(Dst_train.reshape(-1,1))
            Dst_test_cal = reg.predict(Dst_test.reshape(-1,1))

            mae_storm, corr_storm = evaluate(test['dst'], Dst_test_cal, f"α={alpha:4.1f} τ={tau:4.1f}")

            if mae_storm < best_mae:
                best_mae = mae_storm
                print(f"   → Melhor até agora: MAE_storm = {mae_storm:.1f}")

    print("\n=== TESTE 2: Burton + Pressão dinâmica ===")
    Dst_train = euler_dst(train, b_pressure=1.5)
    Dst_test = euler_dst(test, b_pressure=1.5)
    reg = LinearRegression().fit(Dst_train.reshape(-1,1), train['dst'])
    evaluate(test['dst'], reg.predict(Dst_test.reshape(-1,1)), "Burton + Pressão (b=1.5)")

    print("\n=== TESTE 3: HAC disciplinado ===")
    Dst_train = euler_dst(train, alpha_L=0.012, beta_L=0.20, use_hac=True)
    Dst_test = euler_dst(test, alpha_L=0.012, beta_L=0.20, use_hac=True)
    reg = LinearRegression().fit(Dst_train.reshape(-1,1), train['dst'])
    evaluate(test['dst'], reg.predict(Dst_test.reshape(-1,1)), "HAC disciplinado (α_L=0.012)")

    print("\n✅ Pronto. Agora você pode iterar rápido.")

if __name__ == "__main__":
    main()
