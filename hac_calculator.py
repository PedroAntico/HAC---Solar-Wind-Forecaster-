#!/usr/bin/env python3
"""
HAC - Heliospheric Accumulated Coupling (Versão Profissional)

Modelo vetorial de acumulação de energia (Hload, Hrel, Hsat) com:
- Driver normalizado por percentil 90 (estável a outliers)
- Parâmetros calibrados (r=0.503 em 2000-2010)
- Validação treino/teste temporal (2000-2010 / 2010-2020)
- Opção de salvar/carregar parâmetros
- Comparação com Burton (opcional)

Parâmetros baseline (obtidos via grid search):
    alpha = 1.5
    tau = 12.0
    gamma1 = 0.8
    gamma2 = 0.1
    delay_internal = 0.0
    tau_driver = 3.0
    b_pressure = 0.0

Uso:
    # Executar com parâmetros baseline
    python hac_calculator.py

    # Validação treino/teste
    python hac_calculator.py --train_test

    # Grid search (para recalibrar)
    python hac_calculator.py --grid_search
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import os
import itertools
import json
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================
# CONSTANTES FÍSICAS
# ============================
M_P = 1.6726e-27      # kg
CM3_TO_M3 = 1e6
KMS_TO_MS = 1e3

def compute_pdyn_nPa(N_cm3, V_kms):
    N_m3 = N_cm3 * CM3_TO_M3
    V_ms = V_kms * KMS_TO_MS
    return M_P * N_m3 * V_ms**2 * 1e9

# ============================
# FUNÇÕES AUXILIARES
# ============================
def prepare_omni(df):
    df = df.copy().sort_values('time_tag').reset_index(drop=True)
    if 'dst' in df.columns:
        df = df.drop(columns=['dst'])
    df['bz_gsm'] = df['bz_gsm'].ffill(limit=2)
    for col in ['bx_gsm', 'by_gsm', 'density', 'speed']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=3)
    return df.dropna(subset=['bz_gsm', 'speed', 'density'])

def apply_time_shift(df, hours):
    df = df.copy()
    df['time_tag'] = df['time_tag'] - pd.Timedelta(hours=hours)
    return df

def split_data(df, split_date='2010-01-01'):
    """Divide DataFrame em treino (antes da data) e teste (após)."""
    train = df[df['time_tag'] < split_date].copy()
    test = df[df['time_tag'] >= split_date].copy()
    return train, test

# ============================
# HAC VETORIAL (NORMALIZAÇÃO POR PERCENTIL 90)
# ============================
def integrate_hac_vectorial(df, driver_type='akasofu'):
    """
    Calcula os estados Hload, Hrel, Hsat e HCI.
    driver_type: 'akasofu' (normalizado por percentil 90) ou 'hybrid' (antigo)
    """
    times = df['time_tag'].values
    Bz = df['bz_gsm'].values
    V = df['speed'].values
    N = df['density'].values
    Bx = df['bx_gsm'].values
    By = df['by_gsm'].values

    # ----- Driver φ(t) -----
    Btot = np.sqrt(Bx**2 + By**2 + Bz**2)
    theta = np.arctan2(By, Bz)
    sin4 = np.sin(theta / 2)**4

    if driver_type == 'akasofu':
        # Akasofu puro
        phi_raw = V * (Btot**2) * sin4
        # Normalização por percentil 90 (robusto a outliers)
        pos = phi_raw[phi_raw > 0]
        if len(pos) > 0:
            scale = np.percentile(pos, 90)
        else:
            scale = 1.0
        phi = phi_raw / (scale + 1e-6)
    else:
        # Híbrido antigo (preservado para comparação)
        Pdyn = compute_pdyn_nPa(N, V)
        phi = 0.7 * (Pdyn * Btot) + 0.3 * (V * Btot**2 * sin4)
        phi = phi / 1000.0

    # ----- Tempo -----
    dt_sec = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec[1:] = diffs
    dt_sec[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    dt_h = dt_sec / 3600.0

    # ----- Estados -----
    Hload = np.zeros(len(times))
    Hrel  = np.zeros(len(times))
    Hsat  = np.zeros(len(times))

    # ----- Parâmetros fixos (do paper) -----
    alpha_L = 1.2e-5
    beta_L  = 0.15
    alpha_R = 0.08
    beta_R  = 0.25
    eta     = 0.3
    theta_sat = 2.5
    sigma_sat = 0.5

    for i in range(1, len(times)):
        dt = dt_h[i]

        # saturação sigmoidal (baseada em Hload anterior)
        Hsat[i-1] = 1 / (1 + np.exp(-(Hload[i-1] - theta_sat) / sigma_sat))

        # LOADING
        dHload = alpha_L * phi[i] * (1 - Hsat[i-1]) - beta_L * Hload[i-1]

        # RELEASE
        dHrel = alpha_R * Hload[i-1] - beta_R * Hrel[i-1] + eta * Hsat[i-1] * Hload[i-1]

        # integração Euler
        Hload[i] = Hload[i-1] + dHload * dt
        Hrel[i]  = Hrel[i-1]  + dHrel  * dt

        # limites físicos
        Hload[i] = max(0, Hload[i])
        Hrel[i]  = max(0, Hrel[i])

    # último Hsat
    Hsat[-1] = 1 / (1 + np.exp(-(Hload[-1] - theta_sat) / sigma_sat))

    # ----- HCI (acoplamento complementar) -----
    C = np.maximum(-Bz, 0) * (V / 400.0)
    HCI = C + Hload

    out = df.copy()
    out['Hload'] = Hload
    out['Hrel']  = Hrel
    out['Hsat']  = Hsat
    out['HCI']   = HCI
    out['phi']   = phi
    out['C']     = C
    return out

# ============================
# MODELO DST (COM PARÂMETROS BASELINE)
# ============================
def compute_dst_hac(df, alpha=1.5, tau=12.0, gamma1=0.8, gamma2=0.1,
                    delay_internal=0.0, tau_driver=3.0, b_pressure=0.0,
                    debug=False):
    """
    Dst a partir dos estados HAC.
    Parâmetros padrão: os que deram r=0.503 (alpha=1.5, tau=12, gamma1=0.8, gamma2=0.1,
                      delay_internal=0, tau_driver=3, b_pressure=0)
    """
    times = df['time_tag'].values
    Hload = df['Hload'].values
    Hrel  = df['Hrel'].values
    C     = df['C'].values
    Pdyn  = compute_pdyn_nPa(df['density'].values, df['speed'].values)

    # Combinação linear dos estados
    gamma3 = 1.0 - gamma1 - gamma2
    driver_raw = gamma1 * Hload + gamma2 * Hrel + gamma3 * C

    # Atraso interno (shift)
    dt_sec_arr = np.zeros(len(times))
    diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    dt_sec_arr[1:] = diffs
    dt_sec_arr[0] = np.median(diffs) if len(diffs) > 0 else 60.0
    dt_h = dt_sec_arr / 3600.0

    if delay_internal > 0:
        median_dt = np.median(dt_sec_arr[dt_sec_arr > 0])
        delay_steps = int(round(delay_internal * 3600 / median_dt))
        if delay_steps > 0:
            driver_delayed = np.zeros_like(driver_raw)
            if delay_steps < len(driver_raw):
                driver_delayed[delay_steps:] = driver_raw[:-delay_steps]
        else:
            driver_delayed = driver_raw
    else:
        driver_delayed = driver_raw

    # Memória exponencial no driver
    if tau_driver > 0:
        driver_eff = np.zeros_like(driver_delayed)
        for i in range(1, len(driver_eff)):
            decay = np.exp(-dt_h[i] / tau_driver)
            driver_eff[i] = driver_eff[i-1] * decay + driver_delayed[i] * (1 - decay)
        driver = driver_eff
    else:
        driver = driver_delayed

    if debug:
        print(f"DEBUG: driver médio = {np.mean(driver):.3f}, max = {np.max(driver):.3f}")
        print(f"DEBUG: Hload médio = {np.mean(Hload):.3f}, max = {np.max(Hload):.3f}")

    # Integração do Dst
    Dst = np.zeros(len(times))
    for i in range(1, len(times)):
        dt = dt_h[i]
        if b_pressure > 0:
            Dst_star = Dst[i-1] - b_pressure * np.sqrt(Pdyn[i])
        else:
            Dst_star = Dst[i-1]
        injection = -alpha * driver[i]
        decay = Dst_star / tau
        dDst = injection - decay
        Dst[i] = Dst[i-1] + dDst * dt
        Dst[i] = max(-500, min(50, Dst[i]))
    return Dst

# ============================
# VALIDAÇÃO E MÉTRICAS
# ============================
def smooth_dst(merged, hours):
    if hours <= 0:
        merged = merged.copy()
        merged['dst_smooth'] = merged['dst']
        return merged
    merged = merged.set_index('time_tag')
    if len(merged) > 1:
        med_dt = merged.index.to_series().diff().median().total_seconds()
        win = max(3, int(hours * 3600 / med_dt))
        win = win + 1 if win % 2 == 0 else win
        merged['dst_smooth'] = merged['dst'].rolling(win, center=True, min_periods=1).mean()
    else:
        merged['dst_smooth'] = merged['dst']
    merged = merged.dropna(subset=['dst_smooth'])
    return merged.reset_index()

def compute_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return None
    corr = pearsonr(y_true, y_pred)[0] if np.std(y_true) * np.std(y_pred) > 0 else np.nan
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'correlation': corr, 'r2': r2, 'rmse': rmse, 'mae': mae}

def validation_metrics(df_model, dst_series, smooth_hours):
    if dst_series is None or len(dst_series) == 0:
        return None
    df_model = df_model.copy()
    dst = dst_series.copy()
    df_model['time_tag'] = pd.to_datetime(df_model['time_tag'])
    dst['time_tag'] = pd.to_datetime(dst['time_tag'])
    df_model = df_model.dropna(subset=['time_tag', 'Dst_model'])
    dst = dst.dropna(subset=['time_tag', 'dst'])
    if len(df_model) < 10 or len(dst) < 10:
        return None
    merged = pd.merge_asof(df_model.sort_values('time_tag'),
                           dst.sort_values('time_tag'),
                           on='time_tag', direction='backward')
    merged = merged.dropna(subset=['dst', 'Dst_model'])
    merged = smooth_dst(merged, smooth_hours)
    y_true = -merged['dst_smooth'].values
    y_pred = -merged['Dst_model'].values
    return compute_metrics(y_true, y_pred)

def find_optimal_delay(df_model, dst_series, max_delay, step, smooth_hours):
    best_corr = -1
    best_delay = 0
    print(f"\n🔍 Buscando atraso ótimo (Dst_model) até {max_delay}h...")
    delays = np.arange(0, max_delay + step, step)
    for delay in delays:
        shifted = apply_time_shift(dst_series, delay)
        m = validation_metrics(df_model, shifted, smooth_hours)
        if m and not np.isnan(m['correlation']):
            corr = m['correlation']
            print(f"   Delay {delay:3.0f}h → r = {corr:.3f}")
            if corr > best_corr:
                best_corr = corr
                best_delay = delay
    if best_corr < 0.1:
        print("⚠️ Correlação máxima muito baixa. Usando delay=0.")
        best_delay = 0
        best_corr = 0.0
    print(f"🔥 Melhor delay: {best_delay:.0f}h (r = {best_corr:.3f})")
    return best_delay, best_corr

# ============================
# GRÁFICO
# ============================
def plot_comparison(df, df_model, dst_df, delay, smooth_hours, fname):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    if 'Hload' in df.columns:
        axes[0].plot(df['time_tag'], df['Hload'], label='Hload', color='#d62728', lw=2)
        axes[0].plot(df['time_tag'], df['Hrel'], label='Hrel', color='#1f77b4', lw=1.5, alpha=0.7)
        axes[0].plot(df['time_tag'], df['Hsat'], label='Hsat', color='#2ca02c', lw=1.5, alpha=0.7)
        axes[0].set_ylabel('HAC')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Estados HAC (Hload, Hrel, Hsat)')
    else:
        axes[0].set_visible(False)
    if dst_df is not None:
        if delay > 0:
            dst_shifted = apply_time_shift(dst_df, delay)
        else:
            dst_shifted = dst_df.copy()
        dst_shifted = smooth_dst(dst_shifted, smooth_hours)
        axes[1].plot(df['time_tag'], df_model['Dst_model'], label='Dst_model', color='blue', lw=1.5)
        axes[1].plot(dst_shifted['time_tag'], dst_shifted['dst_smooth'], 'k', lw=1.5, label='Dst real suavizado')
        axes[1].axhline(-50, color='red', ls='--', label='Limiar tempestade')
        axes[1].set_ylabel('Dst [nT]')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title(f'Comparação Dst_model vs Dst real (delay {delay}h)')
    else:
        axes[1].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figura salva: {fname}")

# ============================
# GRID SEARCH SIMPLES (opcional)
# ============================
def grid_search(df, dst_df, smooth_hours, param_grid):
    best_corr = -1
    best_params = None
    results = []
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        Dst_model = compute_dst_hac(df, **params)
        df_model = df[['time_tag']].copy()
        df_model['Dst_model'] = Dst_model
        metrics = validation_metrics(df_model, dst_df, smooth_hours)
        if metrics and metrics['correlation'] > best_corr:
            best_corr = metrics['correlation']
            best_params = params
        results.append((params, metrics['correlation'] if metrics else -np.inf))
        print(f"Testado {params} -> r = {metrics['correlation']:.3f}" if metrics else f"Testado {params} -> falhou")
    print(f"\n🏆 Melhor: {best_params} com r = {best_corr:.3f}")
    return best_params

# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser(description='HAC Calculator - Versão Profissional')
    parser.add_argument('--omni', default='data/omni_2000_2020.csv')
    parser.add_argument('--dst', default='data/dst_kyoto_2000_2020.csv')
    parser.add_argument('--output', default='hac_results.csv')
    parser.add_argument('--driver_type', default='akasofu', choices=['akasofu', 'hybrid'])
    # Parâmetros do modelo (valores padrão = melhores encontrados)
    parser.add_argument('--alpha', type=float, default=1.5, help='Coeficiente de injeção')
    parser.add_argument('--tau', type=float, default=12.0, help='Constante de decaimento (horas)')
    parser.add_argument('--gamma1', type=float, default=0.8, help='Peso de Hload')
    parser.add_argument('--gamma2', type=float, default=0.1, help='Peso de Hrel')
    parser.add_argument('--delay_internal', type=float, default=0.0, help='Atraso interno (horas)')
    parser.add_argument('--tau_driver', type=float, default=3.0, help='Memória do driver (horas)')
    parser.add_argument('--b_pressure', type=float, default=0.0, help='Correção de pressão')
    parser.add_argument('--smooth_hours', type=float, default=1, help='Suavização do Dst')
    parser.add_argument('--find_delay', action='store_true', help='Busca atraso ótimo')
    parser.add_argument('--max_delay', type=int, default=24, help='Máximo delay para busca')
    parser.add_argument('--debug', action='store_true', help='Imprime estatísticas dos estados')
    parser.add_argument('--train_test', action='store_true', help='Validação treino/teste temporal')
    parser.add_argument('--grid_search', action='store_true', help='Executa grid search (substitui execução normal)')
    parser.add_argument('--save_params', type=str, default=None, help='Salva parâmetros em arquivo JSON')
    parser.add_argument('--load_params', type=str, default=None, help='Carrega parâmetros de arquivo JSON')
    args = parser.parse_args()

    # Carregar parâmetros de arquivo, se fornecido
    if args.load_params:
        with open(args.load_params, 'r') as f:
            params = json.load(f)
        for key, val in params.items():
            if hasattr(args, key):
                setattr(args, key, val)
        print(f"📂 Parâmetros carregados de {args.load_params}")

    print("\n" + "="*70)
    print("🛰️  HAC - Modelo Vetorial Profissional")
    print(f"   Driver: {args.driver_type.upper()}")
    print(f"   Parâmetros: alpha={args.alpha}, tau={args.tau}, gamma1={args.gamma1}, gamma2={args.gamma2}")
    print(f"   delay_internal={args.delay_internal}h, tau_driver={args.tau_driver}h, b_pressure={args.b_pressure}")
    print("="*70)

    # Carregar OMNI
    if not os.path.exists(args.omni):
        print(f"❌ OMNI não encontrado: {args.omni}")
        return
    omni = pd.read_csv(args.omni, parse_dates=['time_tag'])
    omni = prepare_omni(omni)
    print(f"✅ OMNI: {len(omni)} pontos, {omni['time_tag'].min()} a {omni['time_tag'].max()}")

    # Carregar Dst Kyoto
    if not os.path.exists(args.dst):
        print(f"❌ Dst não encontrado: {args.dst}")
        return
    dst = pd.read_csv(args.dst, parse_dates=['time_tag']).sort_values('time_tag')
    print(f"✅ Dst Kyoto: {len(dst)} pontos, {dst['time_tag'].min()} a {dst['time_tag'].max()}")

    # Cortar Dst para período do OMNI
    dst = dst[dst['time_tag'] <= omni['time_tag'].max()]
    print(f"   Dst cortado: {len(dst)} pontos")

    # Interpolação robusta
    time_grid = omni[['time_tag']].copy()
    dst_interp = pd.merge_asof(
        time_grid.sort_values('time_tag'),
        dst[['time_tag', 'dst']].sort_values('time_tag'),
        on='time_tag',
        direction='nearest',
        tolerance=pd.Timedelta('2h')
    )
    dst_interp = dst_interp.set_index('time_tag').interpolate(method='time', limit=3).reset_index()
    assert 'dst' in dst_interp.columns, "ERRO: coluna dst sumiu"

    # Merge final
    if 'dst' in omni.columns:
        omni = omni.drop(columns=['dst'])
    df = pd.merge(omni, dst_interp, on='time_tag', how='left', suffixes=('', '_dst'))
    if 'dst_dst' in df.columns:
        df = df.rename(columns={'dst_dst': 'dst'})
    df = df.dropna(subset=['dst'])
    print(f"   Merge final: {len(df)} pontos, Dst min = {df['dst'].min():.1f} nT")
    if df['dst'].min() > -50:
        print("⚠️ ALERTA: Período sem tempestades fortes (Dst > -50 nT).")

    # HAC vetorial (com normalização por percentil 90)
    df = integrate_hac_vectorial(df, driver_type=args.driver_type)

    # Grid search (substitui execução normal)
    if args.grid_search:
        param_grid = {
            'alpha': [1.5, 2.0, 2.5],
            'tau': [10, 12, 14],
            'gamma1': [0.7, 0.8, 0.9],
            'gamma2': [0.0, 0.1, 0.2],
            'delay_internal': [0, 1, 2],
            'tau_driver': [0, 1.5, 3.0],
            'b_pressure': [0, 3, 6]
        }
        best_params = grid_search(df, dst[['time_tag', 'dst']], args.smooth_hours, param_grid)
        if best_params:
            # Atualizar args com os melhores parâmetros
            for key, val in best_params.items():
                setattr(args, key, val)
            print(f"\n✅ Melhores parâmetros encontrados: {best_params}")
            if args.save_params:
                with open(args.save_params, 'w') as f:
                    json.dump(best_params, f, indent=2)
                print(f"💾 Parâmetros salvos em {args.save_params}")
        else:
            print("❌ Grid search falhou, usando parâmetros atuais")

    # Validação treino/teste (se solicitado)
    if args.train_test:
        print("\n📊 VALIDAÇÃO TREINO/TESTE (2000-2010 / 2010-2020)")
        # Dividir os dados originais (antes de integrar HAC? Melhor integrar separado)
        # Mas como o HAC depende de todo o período (por causa dos estados), devemos recalcular para cada parte?
        # Para simular um cenário real, dividimos os dados antes de integrar HAC.
        # Então vamos recarregar os dados e fazer a divisão antes da integração.
        # Para simplificar, recarregamos os dados crus e dividimos.
        omni_raw = pd.read_csv(args.omni, parse_dates=['time_tag'])
        omni_raw = prepare_omni(omni_raw)
        dst_raw = pd.read_csv(args.dst, parse_dates=['time_tag']).sort_values('time_tag')
        dst_raw = dst_raw[dst_raw['time_tag'] <= omni_raw['time_tag'].max()]
        # Interpolação
        dst_interp_raw = pd.merge_asof(
            omni_raw[['time_tag']].sort_values('time_tag'),
            dst_raw[['time_tag', 'dst']].sort_values('time_tag'),
            on='time_tag', direction='nearest', tolerance=pd.Timedelta('2h')
        ).set_index('time_tag').interpolate(method='time', limit=3).reset_index()
        df_raw = pd.merge(omni_raw, dst_interp_raw, on='time_tag', how='left').dropna(subset=['dst'])
        # Dividir
        train_df, test_df = split_data(df_raw, split_date='2010-01-01')
        print(f"Treino: {len(train_df)} pontos ({train_df['time_tag'].min()} a {train_df['time_tag'].max()})")
        print(f"Teste: {len(test_df)} pontos ({test_df['time_tag'].min()} a {test_df['time_tag'].max()})")

        # Integrar HAC e calcular Dst para cada conjunto
        train_df = integrate_hac_vectorial(train_df, driver_type=args.driver_type)
        test_df = integrate_hac_vectorial(test_df, driver_type=args.driver_type)

        train_df['Dst_model'] = compute_dst_hac(train_df, alpha=args.alpha, tau=args.tau,
                                               gamma1=args.gamma1, gamma2=args.gamma2,
                                               delay_internal=args.delay_internal,
                                               tau_driver=args.tau_driver,
                                               b_pressure=args.b_pressure,
                                               debug=args.debug)
        test_df['Dst_model'] = compute_dst_hac(test_df, alpha=args.alpha, tau=args.tau,
                                              gamma1=args.gamma1, gamma2=args.gamma2,
                                              delay_internal=args.delay_internal,
                                              tau_driver=args.tau_driver,
                                              b_pressure=args.b_pressure,
                                              debug=args.debug)

        # Avaliar métricas
        train_metrics = validation_metrics(train_df[['time_tag', 'Dst_model']],
                                           train_df[['time_tag', 'dst']],
                                           args.smooth_hours)
        test_metrics = validation_metrics(test_df[['time_tag', 'Dst_model']],
                                          test_df[['time_tag', 'dst']],
                                          args.smooth_hours)

        print("\n📊 TREINO (2000-2010):")
        if train_metrics:
            print(f"   Correlação: {train_metrics['correlation']:.3f}")
            print(f"   R²: {train_metrics['r2']:.3f}")
            print(f"   RMSE: {train_metrics['rmse']:.2f}")
        print("\n📊 TESTE (2010-2020):")
        if test_metrics:
            print(f"   Correlação: {test_metrics['correlation']:.3f}")
            print(f"   R²: {test_metrics['r2']:.3f}")
            print(f"   RMSE: {test_metrics['rmse']:.2f}")
        # Salvar resultados
        test_df.to_csv(args.output.replace('.csv', '_test.csv'), index=False)
        print(f"\n💾 Resultados do teste salvos em {args.output.replace('.csv', '_test.csv')}")
        return  # Encerra após validação treino/teste

    # Execução normal (sem treino/teste)
    df['Dst_model'] = compute_dst_hac(df, alpha=args.alpha, tau=args.tau,
                                      gamma1=args.gamma1, gamma2=args.gamma2,
                                      delay_internal=args.delay_internal,
                                      tau_driver=args.tau_driver,
                                      b_pressure=args.b_pressure,
                                      debug=args.debug)

    # Validação
    df_model = df[['time_tag', 'Dst_model']].copy()
    dst_df = df[['time_tag', 'dst']].copy()
    smooth_hours = args.smooth_hours

    best_delay = 0
    if args.find_delay:
        best_delay, best_corr = find_optimal_delay(df_model, dst_df, args.max_delay, 1, smooth_hours)
    else:
        best_delay = 0
        shifted = apply_time_shift(dst_df, best_delay)
        metrics = validation_metrics(df_model, shifted, smooth_hours)
        if metrics:
            best_corr = metrics['correlation']
            print(f"\n📊 Correlação sem delay externo: r = {best_corr:.3f}")

    print("\n📊 VALIDAÇÃO DO MODELO Dst")
    print("="*50)
    shifted = apply_time_shift(dst_df, best_delay)
    metrics = validation_metrics(df_model, shifted, smooth_hours)
    if metrics:
        print(f"✅ Dst_model vs Dst real (delay={best_delay}h):")
        print(f"   Correlação: {metrics['correlation']:.3f}")
        print(f"   R²: {metrics['r2']:.3f}")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   MAE: {metrics['mae']:.2f}")

    # Salvar resultados e parâmetros
    df.to_csv(args.output, index=False)
    print(f"\n💾 Resultados salvos em {args.output}")

    if args.save_params:
        params_to_save = {
            'alpha': args.alpha,
            'tau': args.tau,
            'gamma1': args.gamma1,
            'gamma2': args.gamma2,
            'delay_internal': args.delay_internal,
            'tau_driver': args.tau_driver,
            'b_pressure': args.b_pressure
        }
        with open(args.save_params, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        print(f"💾 Parâmetros salvos em {args.save_params}")

    # Gráfico
    plot_comparison(df, df_model, dst_df, best_delay, smooth_hours, "hac_comparison.png")

if __name__ == "__main__":
    main()
