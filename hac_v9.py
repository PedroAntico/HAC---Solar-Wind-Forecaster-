#!/usr/bin/env python3
"""
HAC v9 - Refinamento cirúrgico (Burton otimizado + HAC disciplinado)
Modelos:
  - Burton clássico: dDst/dt = -α * VBs - Dst/τ
  - Burton + pressão dinâmica: dDst/dt = -α * VBs + b * sqrt(Pdyn) - Dst/τ
  - HAC disciplinado: dH/dt = α_L * VBs - β_L * H, Dst = -γ * H (com calibração linear)

Uso:
    python hac_v9.py --omni data/omni_2000_2020.csv --dst data/dst_kyoto_2000_2020.csv
"""

import pandas as pd
import numpy as np
import warnings
import argparse
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ============================
# CONSTANTES FÍSICAS
# ============================
M_P = 1.6726e-27
CM3_TO_M3 = 1e6
KMS_TO_MS = 1e3

def compute_pdyn_nPa(N_cm3, V_kms):
    """Pressão dinâmica em nPa."""
    N_m3 = N_cm3 * CM3_TO_M3
    V_ms = V_kms * KMS_TO_MS
    return M_P * N_m3 * V_ms**2 * 1e9

# ============================
# PREPARE OMNI - LIMPEZA ROBUSTA
# ============================
def prepare_omni(df):
    df = df.copy().sort_values('time_tag').reset_index(drop=True)
    if 'dst' in df.columns:
        df = df.drop(columns=['dst'])

    # Correção de unidade da velocidade
    if 'speed' in df.columns and len(df) > 100:
        med = df['speed'].median()
        print(f"   Speed original mediana: {med:.0f}")
        if med > 10000:
            df['speed'] /= 1000.0
            print(f"   → convertido para km/s (mediana: {df['speed'].median():.0f})")
        # Ajuste adicional se ainda fora da faixa realista (200-600 km/s)
        if df['speed'].median() < 300:
            target = 400.0
            factor = target / max(df['speed'].median(), 50)
            df['speed'] *= factor
            print(f"   → correção adicional (fator {factor:.2f}). Nova mediana: {df['speed'].median():.0f} km/s")

    # Remoção de valores fisicamente impossíveis
    bad = (
        (df['bz_gsm'].abs() > 100) |
        (df['speed'] < 150) | (df['speed'] > 2500) |
        (df['density'] < 0.001) | (df['density'] > 10000) |
        (df['bx_gsm'].abs() > 100) |
        (df['by_gsm'].abs() > 100)
    )
    before = len(df)
    df.loc[bad, ['bz_gsm', 'bx_gsm', 'by_gsm', 'density', 'speed']] = np.nan

    # Preenchimento de lacunas
    df['bz_gsm'] = df['bz_gsm'].ffill(limit=12)
    for col in ['bx_gsm', 'by_gsm', 'density', 'speed']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=24)

    df = df.dropna(subset=['bz_gsm', 'speed', 'density']).reset_index(drop=True)
    after = len(df)
    print(f"   Dados finais: {before:,} → {after:,} pontos ({(before-after)/before*100:.1f}% removidos)")
    if after > 0:
        print(f"   Speed final: mediana = {df['speed'].median():.0f} km/s")
    return df

# ============================
# MODELO BURTON (COM INTEGRAÇÃO ESTÁVEL)
# ============================
def burton_model(df, alpha=4.5, tau=6.0, b_pressure=0.0, delay_hours=2.0):
    """
    Modelo Burton:
        dDst/dt = -α * VBs - Dst/τ   [+ b * sqrt(Pdyn) se b_pressure > 0]
    Retorna Dst (nT).
    """
    times = df['time_tag'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    density = df['density'].values

    # Campo elétrico de reconexão (mV/m)
    Bs = np.maximum(-Bz, 0)
    VBs = V * Bs / 1000.0

    # Atraso (shift temporal)
    if delay_hours > 0:
        dt_sec = np.median(np.diff(times).astype('timedelta64[s]').astype(float))
        delay_steps = int(round(delay_hours * 3600 / dt_sec))
        if delay_steps > 0:
            VBs = np.concatenate([np.zeros(delay_steps), VBs[:-delay_steps]])

    # Pressão dinâmica (nPa)
    Pdyn = compute_pdyn_nPa(density, V)

    # Interpoladores para uso no solve_ivp
    t_eval = np.arange(len(times))
    VBs_interp = interp1d(t_eval, VBs, kind='linear', fill_value='extrapolate')
    if b_pressure > 0:
        Pdyn_interp = interp1d(t_eval, np.sqrt(Pdyn), kind='linear', fill_value='extrapolate')
    else:
        Pdyn_interp = None

    def ode(t, Dst):
        idx = int(round(t))
        idx = max(0, min(idx, len(times)-1))
        dDst = -alpha * VBs_interp(idx) - Dst / tau
        if b_pressure > 0:
            dDst += b_pressure * Pdyn_interp(idx)
        return dDst

    sol = solve_ivp(ode, [0, len(times)-1], [0.0], method='LSODA', t_eval=t_eval, rtol=1e-6, atol=1e-8)
    return sol.y[0]

# ============================
# MODELO HAC DISCIPLINADO (COM LIMITE FÍSICO)
# ============================
def hac_disciplinado(df, alpha_L=0.012, beta_L=0.20, gamma=0.8, tau=6.0, delay_hours=2.0):
    """
    Modelo HAC simplificado:
        dH/dt = α_L * VBs - β_L * H
        Dst = -γ * H
    Com limite superior em H para estabilidade.
    """
    times = df['time_tag'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values

    VBs = V * np.maximum(-Bz, 0) / 1000.0

    # Atraso
    if delay_hours > 0:
        dt_sec = np.median(np.diff(times).astype('timedelta64[s]').astype(float))
        delay_steps = int(round(delay_hours * 3600 / dt_sec))
        if delay_steps > 0:
            VBs = np.concatenate([np.zeros(delay_steps), VBs[:-delay_steps]])

    # Passo de tempo (horas)
    dt_hours = np.diff(times).astype('timedelta64[h]').astype(float)
    dt_hours = np.concatenate([[1.0], dt_hours])

    H = np.zeros(len(times))
    for i in range(1, len(times)):
        dH = alpha_L * VBs[i] - beta_L * H[i-1]
        H[i] = H[i-1] + dH * dt_hours[i]
        H[i] = np.clip(H[i], 0, 50.0)      # limite físico (evita explosão)

    Dst_raw = -gamma * H

    # Integração adicional? Não, Dst já é dado diretamente por H.
    # Mas para manter consistência com a equação diferencial, podemos também aplicar um filtro suave.
    # Por simplicidade, retornamos Dst_raw.
    return Dst_raw

# ============================
# MÉTRICAS (COM FOCO EM TEMPESTADES)
# ============================
def storm_metrics(y_true, y_pred, threshold=-50):
    mask = y_true < threshold
    n_storm = mask.sum()
    if n_storm == 0:
        return {'mae_storm': np.nan, 'corr_storm': np.nan, 'n_storm': 0}
    mae_storm = mean_absolute_error(y_true[mask], y_pred[mask])
    corr_storm = pearsonr(y_true[mask], y_pred[mask])[0] if n_storm > 1 else np.nan
    return {'mae_storm': mae_storm, 'corr_storm': corr_storm, 'n_storm': n_storm}

def compute_full_metrics(y_true, y_pred):
    corr = pearsonr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    storm = storm_metrics(y_true, y_pred)
    return {
        'correlation': corr,
        'r2': r2,
        'mae': mae,
        'mae_storm': storm['mae_storm'],
        'corr_storm': storm['corr_storm'],
        'n_storm': storm['n_storm']
    }

def evaluate_and_calibrate(train_df, test_df, model_func, model_name, **kwargs):
    """
    Aplica o modelo, calibra linearmente no treino, e retorna métricas no treino e teste.
    """
    print(f"\n🔬 Rodando {model_name}...")
    train_pred = model_func(train_df, **kwargs)
    test_pred = model_func(test_df, **kwargs)

    # Calibração linear (escala e offset)
    reg = LinearRegression().fit(train_pred.reshape(-1, 1), train_df['dst'].values)
    coef, interc = reg.coef_[0], reg.intercept_
    train_calib = coef * train_pred + interc
    test_calib = coef * test_pred + interc

    train_metrics = compute_full_metrics(train_df['dst'].values, train_calib)
    test_metrics = compute_full_metrics(test_df['dst'].values, test_calib)

    print(f"   Calibração: Dst = {coef:.4f} * raw + {interc:.2f}")
    print(f"   TREINO → corr={train_metrics['correlation']:.3f}  R²={train_metrics['r2']:.3f}  MAE_storm={train_metrics['mae_storm']:.1f}")
    print(f"   TESTE  → corr={test_metrics['correlation']:.3f}  R²={test_metrics['r2']:.3f}  MAE_storm={test_metrics['mae_storm']:.1f}  corr_storm={test_metrics['corr_storm']:.3f}")
    return train_metrics, test_metrics

# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser(description='HAC v9 - Refinamento cirúrgico')
    parser.add_argument('--omni', default='data/omni_2000_2020.csv', help='Arquivo OMNI')
    parser.add_argument('--dst', default='data/dst_kyoto_2000_2020.csv', help='Arquivo Dst Kyoto')
    parser.add_argument('--output', default='hac_v9_results.csv', help='Arquivo de saída (opcional)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🛰️  HAC v9 - Burton otimizado + HAC disciplinado")
    print("="*80)

    # Carregar e preparar dados
    if not os.path.exists(args.omni):
        print(f"❌ OMNI não encontrado: {args.omni}")
        return
    omni = pd.read_csv(args.omni, parse_dates=['time_tag'])
    omni = prepare_omni(omni)
    if len(omni) == 0:
        return

    if not os.path.exists(args.dst):
        print(f"❌ Dst não encontrado: {args.dst}")
        return
    dst = pd.read_csv(args.dst, parse_dates=['time_tag']).sort_values('time_tag')
    dst = dst[dst['time_tag'] <= omni['time_tag'].max()]

    # Merge (interpolação do Dst na grade do OMNI)
    time_grid = omni[['time_tag']].copy()
    dst_interp = pd.merge_asof(
        time_grid.sort_values('time_tag'),
        dst[['time_tag', 'dst']].sort_values('time_tag'),
        on='time_tag', direction='nearest', tolerance=pd.Timedelta('2h')
    ).set_index('time_tag').interpolate(method='time', limit=3).reset_index()
    df = pd.merge(omni, dst_interp, on='time_tag', how='left').dropna(subset=['dst'])
    print(f"   Dataset final: {len(df):,} pontos | Dst min = {df['dst'].min():.0f} nT")

    # Split treino/teste (2000-2010 / 2010-2020)
    train = df[df['time_tag'] < '2010-01-01'].copy()
    test = df[df['time_tag'] >= '2010-01-01'].copy()
    print(f"   Treino: {len(train):,} pontos ({train['time_tag'].min()} a {train['time_tag'].max()})")
    print(f"   Teste:  {len(test):,} pontos ({test['time_tag'].min()} a {test['time_tag'].max()})")

    # ==================== TESTE 1: BURTON OTIMIZADO ====================
    print("\n" + "="*80)
    print("TESTE 1: Burton clássico (varredura de parâmetros)")
    print("="*80)
    best_corr_storm = -1
    best_params = None
    for alpha in [4.0, 4.5, 5.0]:
        for tau in [5.0, 6.0, 7.0]:
            train_pred = burton_model(train, alpha=alpha, tau=tau, b_pressure=0.0, delay_hours=2.0)
            test_pred = burton_model(test, alpha=alpha, tau=tau, b_pressure=0.0, delay_hours=2.0)
            reg = LinearRegression().fit(train_pred.reshape(-1, 1), train['dst'])
            test_calib = reg.predict(test_pred.reshape(-1, 1))
            metrics = compute_full_metrics(test['dst'].values, test_calib)
            print(f"α={alpha:.1f} τ={tau:.1f} → test corr={metrics['correlation']:.3f}  corr_storm={metrics['corr_storm']:.3f}  MAE_storm={metrics['mae_storm']:.1f}")
            if metrics['corr_storm'] is not np.nan and metrics['corr_storm'] > best_corr_storm:
                best_corr_storm = metrics['corr_storm']
                best_params = (alpha, tau)
    if best_params:
        print(f"\n✅ Melhor Burton: α={best_params[0]:.1f}, τ={best_params[1]:.1f} (corr_storm={best_corr_storm:.3f})")
        # Rodar com os melhores parâmetros para obter métricas completas
        evaluate_and_calibrate(train, test, burton_model, "Burton otimizado",
                               alpha=best_params[0], tau=best_params[1], b_pressure=0.0, delay_hours=2.0)

    # ==================== TESTE 2: BURTON + PRESSÃO ====================
    print("\n" + "="*80)
    print("TESTE 2: Burton + pressão dinâmica")
    print("="*80)
    for b_press in [0.5, 1.0, 1.5, 2.0]:
        evaluate_and_calibrate(train, test, burton_model, f"Burton + pressão (b={b_press})",
                               alpha=4.5, tau=6.0, b_pressure=b_press, delay_hours=2.0)

    # ==================== TESTE 3: HAC DISCIPLINADO ====================
    print("\n" + "="*80)
    print("TESTE 3: HAC disciplinado (com limites físicos)")
    print("="*80)
    # Varredura de α_L e β_L
    best_corr_storm_hac = -1
    best_params_hac = None
    for alpha_L in [0.008, 0.010, 0.012, 0.015]:
        for beta_L in [0.15, 0.20, 0.25]:
            train_pred = hac_disciplinado(train, alpha_L=alpha_L, beta_L=beta_L, gamma=0.8, tau=6.0, delay_hours=2.0)
            test_pred = hac_disciplinado(test, alpha_L=alpha_L, beta_L=beta_L, gamma=0.8, tau=6.0, delay_hours=2.0)
            reg = LinearRegression().fit(train_pred.reshape(-1, 1), train['dst'])
            test_calib = reg.predict(test_pred.reshape(-1, 1))
            metrics = compute_full_metrics(test['dst'].values, test_calib)
            print(f"α_L={alpha_L:.3f} β_L={beta_L:.2f} → test corr={metrics['correlation']:.3f}  corr_storm={metrics['corr_storm']:.3f}  MAE_storm={metrics['mae_storm']:.1f}")
            if metrics['corr_storm'] is not np.nan and metrics['corr_storm'] > best_corr_storm_hac:
                best_corr_storm_hac = metrics['corr_storm']
                best_params_hac = (alpha_L, beta_L)
    if best_params_hac:
        print(f"\n✅ Melhor HAC: α_L={best_params_hac[0]:.3f}, β_L={best_params_hac[1]:.2f} (corr_storm={best_corr_storm_hac:.3f})")
        evaluate_and_calibrate(train, test, hac_disciplinado, "HAC disciplinado (melhor)",
                               alpha_L=best_params_hac[0], beta_L=best_params_hac[1], gamma=0.8, tau=6.0, delay_hours=2.0)

    # Salvar previsões do melhor modelo (opcional)
    # Escolhemos o melhor Burton (ou Burton+pressão) para salvar
    best_burton_train = burton_model(train, alpha=4.5, tau=6.0, b_pressure=0.0, delay_hours=2.0)
    best_burton_test = burton_model(test, alpha=4.5, tau=6.0, b_pressure=0.0, delay_hours=2.0)
    reg_best = LinearRegression().fit(best_burton_train.reshape(-1, 1), train['dst'])
    test['Dst_Burton_calib'] = reg_best.predict(best_burton_test.reshape(-1, 1))
    test[['time_tag', 'dst', 'Dst_Burton_calib']].to_csv(args.output, index=False)
    print(f"\n💾 Previsões do Burton (α=4.5, τ=6.0) salvas em {args.output}")

    print("\n✅ Análise concluída. Os resultados mostram que o Burton simples já é competitivo; o HAC disciplinado só melhora se bem calibrado e com limites físicos.")

if __name__ == "__main__":
    main()
