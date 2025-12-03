#!/usr/bin/env python3
"""
hac_v6_features.py - COM DENSIDADE RECUPERADA E NORMALIZADA
Feature Engineering & Dataset Builder para HAC v6.

RECUPERAÇÃO DE DENSIDADE:
1. Densidade foi recuperada dos dados OMNI originais e normalizada
2. Agora usa variáveis: 'speed', 'bz_gsm', 'density' (todas presentes)
3. Densidade NÃO é mais constante - foi normalizada corretamente
"""

import os
import gc
import json
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from hac_v6_config import HACConfig


class HACFeatureBuilder:
    """Gera features e datasets supervisionados para HAC v6."""

    def __init__(self, config: HACConfig):
        """Inicializa o builder com configuração e scalers vazios."""
        self.config = config
        
        # Scaler para features (X) - apenas UM para todas as features
        self.scaler_X = StandardScaler()
        
        # Scalers para targets (Y) - UM POR VARIÁVEL FÍSICA
        # Formato: {horizon: {target_name: StandardScaler}}
        self.scalers_y: Dict[int, Dict[str, StandardScaler]] = {}
        
        # Carregar dados
        data_dir = self.config.get("paths")["data_dir"]
        csv_path = os.path.join(data_dir, "omni_prepared.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Arquivo não encontrado: {csv_path}")
        
        print(f"📂 Carregando dataset: {csv_path}")
        self.raw_df = pd.read_csv(csv_path, low_memory=True)
        
        # Processar datetime se existir
        if "datetime" in self.raw_df.columns:
            self.raw_df["datetime"] = pd.to_datetime(self.raw_df["datetime"])
        
        # Validar colunas necessárias
        self._validate_required_columns()
        
        print(f"✅ Dataset carregado: {len(self.raw_df)} linhas, {len(self.raw_df.columns)} colunas")
        
        # ✅ ANÁLISE INICIAL DAS VARIÁVEIS
        self._analyze_variables()

    # ------------------------------------------------------------
    def _validate_required_columns(self):
        """Garante que as colunas-alvo existem no CSV."""
        targets = self.config.get("targets")["primary"]
        missing = [c for c in targets if c not in self.raw_df.columns]
        
        if missing:
            available = list(self.raw_df.columns)[:20]
            raise ValueError(
                f"❌ Colunas target ausentes no CSV: {missing}\n"
                f"   Colunas disponíveis (primeiras 20): {available}"
            )
        
        print(f"✅ Targets validados: {targets}")
        
        # Verificar especificamente a densidade
        if 'density' in targets:
            print(f"   ✅ Densidade presente e será usada como target")
            print(f"   📊 Estatísticas da densidade:")
            print(f"      • Média: {self.raw_df['density'].mean():.2f}")
            print(f"      • Std: {self.raw_df['density'].std():.2f}")
            print(f"      • Min-Max: {self.raw_df['density'].min():.1f}-{self.raw_df['density'].max():.1f}")

    # ------------------------------------------------------------
    def _analyze_variables(self):
        """Analisa as variáveis target para detectar problemas."""
        print("\n🔍 Análise inicial das variáveis target:")
        
        targets = self.config.get("targets")["primary"]
        
        for target in targets:
            if target in self.raw_df.columns:
                values = self.raw_df[target].dropna()
                mean_val = values.mean()
                std_val = values.std()
                min_val = values.min()
                max_val = values.max()
                unique_count = values.nunique()
                nan_count = self.raw_df[target].isna().sum()
                
                print(f"   {target}:")
                print(f"     • Média: {mean_val:.2f}")
                print(f"     • Desvio padrão: {std_val:.2f}")
                print(f"     • Range: {min_val:.1f} - {max_val:.1f}")
                print(f"     • Valores únicos: {unique_count}")
                print(f"     • NaNs: {nan_count} ({nan_count/len(self.raw_df)*100:.1f}%)")
                
                if std_val < 0.1:
                    print(f"     ⚠️  VARIÁVEL QUASE CONSTANTE (std={std_val:.3f})")
                elif unique_count < 10:
                    print(f"     ⚠️  POUCOS VALORES ÚNICOS ({unique_count})")
                else:
                    print(f"     ✅ Variação OK")

    # ------------------------------------------------------------
    def build_all(self) -> Dict[int, Dict[str, Any]]:
        """Pipeline principal: engineer → scale → windows."""
        print("\n🔧 Iniciando pipeline de construção de features...")
        
        # 1. Engenharia de features (usando nomes CORRETOS do CSV)
        print("  1. Criando features físicas e estatísticas...")
        df_feat = self._engineer_features(self.raw_df.copy())
        
        # 2. Separar features (X) e targets (Y) antes de escalonar
        target_cols = self.config.get("targets")["primary"]
        feature_cols = [col for col in df_feat.columns if col not in target_cols]
        
        print(f"  2. Separando {len(feature_cols)} features e {len(target_cols)} targets")
        
        # 3. Escalonar APENAS as features (X)
        print(f"  3. Escalonando features (X)...")
        if feature_cols:
            df_feat[feature_cols] = self.scaler_X.fit_transform(df_feat[feature_cols])
            print(f"     ✅ {len(feature_cols)} colunas de features escalonadas")
        
        # 4. Targets (Y) NÃO são escalonados aqui - serão escalonados por variável no windowing
        print(f"  4. Targets (Y) serão escalonados POR VARIÁVEL durante a criação das janelas")
        
        # 5. Criar janelas temporais (com escalonamento Y correto)
        print("  5. Criando janelas temporais para cada horizonte...")
        datasets = self._make_all_horizon_windows(df_feat, target_cols)
        
        # 6. Limpar memória
        gc.collect()
        
        print(f"\n✅ Pipeline completo. {len(datasets)} horizontes processados.")
        return datasets

    # ------------------------------------------------------------
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features: físicas + lags + estatísticas móveis.
        
        USA OS NOMES CORRETOS DO CSV:
        - 'speed' (velocidade do vento solar)
        - 'bz_gsm' (componente Bz)
        - 'density' (densidade do vento solar - RECUPERADA E NORMALIZADA)
        """
        print(f"    Formato inicial: {df.shape}")
        
        # Remover colunas temporais não-numéricas
        non_numeric = ["datetime", "timestamp", "year", "doy", "hour"]
        for col in non_numeric:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # -----------------------------------------------------------------
        # ENGINEERING FÍSICO DO Bz (usando 'bz_gsm')
        # -----------------------------------------------------------------
        if 'bz_gsm' in df.columns:
            print("    Criando features físicas do Bz...")
            
            # 1. Componente SUL do Bz (fisicamente relevante para reconexão)
            df['bz_sul'] = df['bz_gsm'].clip(upper=0)
            
            # 2. Valor absoluto do Bz sul (para funções de acoplamento)
            df['bz_sul_abs'] = np.abs(df['bz_sul'])
            
            # 3. Bz negativo acumulado (proxy para injeção de energia na magnetosfera)
            window_1h = 12  # Supondo dados a 5-min: 12 pontos = 1 hora
            window_3h = 36  # 36 pontos = 3 horas
            
            df['bz_neg_acum_1h'] = df['bz_sul'].rolling(window=window_1h, min_periods=1).sum()
            df['bz_neg_acum_3h'] = df['bz_sul'].rolling(window=window_3h, min_periods=1).sum()
            
            print(f"      ✅ Features físicas do Bz criadas: bz_sul, acumulados")
        
        # -----------------------------------------------------------------
        # ENGINEERING FÍSICO DA DENSIDADE (agora NORMALIZADA)
        # -----------------------------------------------------------------
        if 'density' in df.columns:
            print("    Criando features físicas da densidade...")
            
            # 1. Logaritmo da densidade (comum em física de plasma)
            df['density_log'] = np.log1p(df['density'])
            
            # 2. Variação percentual da densidade
            df['density_pct_change'] = df['density'].pct_change().fillna(0)
            
            # 3. Média móvel da densidade (para suavizar)
            df['density_smoothed_1h'] = df['density'].rolling(window=12, min_periods=6).mean()
            df['density_smoothed_3h'] = df['density'].rolling(window=36, min_periods=18).mean()
            
            print(f"      ✅ Features físicas da densidade criadas: log, pct_change, smoothed")
        
        # -----------------------------------------------------------------
        # ENGINEERING FÍSICO DA VELOCIDADE (speed)
        # -----------------------------------------------------------------
        if 'speed' in df.columns:
            print("    Criando features físicas da velocidade...")
            
            # 1. Aceleração (derivada da velocidade)
            df['speed_acceleration'] = df['speed'].diff().fillna(0)
            
            # 2. Média móvel da velocidade
            df['speed_smoothed_1h'] = df['speed'].rolling(window=12, min_periods=6).mean()
            df['speed_smoothed_3h'] = df['speed'].rolling(window=36, min_periods=18).mean()
            
            print(f"      ✅ Features físicas da velocidade criadas: acceleration, smoothed")
        
        # -----------------------------------------------------------------
        # INTERAÇÕES FÍSICAS ENTRE VARIÁVEIS
        # -----------------------------------------------------------------
        print("    Criando interações físicas entre variáveis...")
        
        # 1. Pressão dinâmica: P_dyn = density * speed^2 (proxy para pressão solar)
        if 'density' in df.columns and 'speed' in df.columns:
            df['pressure_dynamic'] = df['density'] * (df['speed'] ** 2)
            print(f"      ✅ Pressão dinâmica criada: P_dyn = density * speed^2")
        
        # 2. Fluxo de momento: n * V
        if 'density' in df.columns and 'speed' in df.columns:
            df['momentum_flux'] = df['density'] * df['speed']
            print(f"      ✅ Fluxo de momento criado: n * V")
        
        # 3. Função de acoplamento de Newell: V * |Bz_sul|^2
        if 'speed' in df.columns and 'bz_sul_abs' in df.columns:
            df['newell_coupling'] = df['speed'] * (df['bz_sul_abs'] ** 2)
            print(f"      ✅ Função de acoplamento de Newell criada: V * |Bz_sul|^2")
        
        # 4. Interação Bz-densidade (para eventos de CME com alta densidade)
        if 'bz_gsm' in df.columns and 'density' in df.columns:
            df['bz_density_interaction'] = df['bz_gsm'] * df['density']
            print(f"      ✅ Interação Bz-densidade criada: Bz * n")
        
        # -----------------------------------------------------------------
        # Features base para engenharia (targets primários + secundários)
        # -----------------------------------------------------------------
        targets_cfg = self.config.get("targets")
        primary = targets_cfg.get("primary", [])
        secondary = targets_cfg.get("secondary", [])
        
        # Incluir todas as features físicas criadas nas secundárias
        physical_features = [
            'bz_sul', 'bz_sul_abs', 'bz_neg_acum_1h', 'bz_neg_acum_3h',
            'density_log', 'density_pct_change', 'density_smoothed_1h', 'density_smoothed_3h',
            'speed_acceleration', 'speed_smoothed_1h', 'speed_smoothed_3h',
            'pressure_dynamic', 'momentum_flux', 'newell_coupling', 'bz_density_interaction'
        ]
        
        # Combinar todas as colunas base para criar lags
        base_cols = []
        all_potential_cols = primary + secondary + physical_features
        
        for col in all_potential_cols:
            if col in df.columns and col not in base_cols:
                base_cols.append(col)
        
        print(f"    Criando lags e estatísticas para {len(base_cols)} colunas base...")
        
        # -----------------------------------------------------------------
        # Lags e estatísticas móveis (para todas as colunas base)
        # -----------------------------------------------------------------
        for col in base_cols:
            # Lags (1, 3, 6 passos atrás - 5, 15, 30 minutos)
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag3"] = df[col].shift(3)
            df[f"{col}_lag6"] = df[col].shift(6)
            
            # Médias móveis (6, 12, 24 passos - 30 min, 1h, 2h)
            df[f"{col}_rm6"] = df[col].rolling(window=6, min_periods=3).mean()
            df[f"{col}_rm12"] = df[col].rolling(window=12, min_periods=6).mean()
            df[f"{col}_rm24"] = df[col].rolling(window=24, min_periods=12).mean()
            
            # Desvio padrão móvel (para variabilidade)
            df[f"{col}_std6"] = df[col].rolling(window=6, min_periods=3).std()
            df[f"{col}_std12"] = df[col].rolling(window=12, min_periods=6).std()
            
            # Range móvel (max-min)
            df[f"{col}_range12"] = df[col].rolling(window=12, min_periods=6).max() - \
                                 df[col].rolling(window=12, min_periods=6).min()
        
        # -----------------------------------------------------------------
        # Limpar NaNs e retornar
        # -----------------------------------------------------------------
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        final_len = len(df)
        
        removed = initial_len - final_len
        print(f"    Removidos {removed} NaNs ({removed/initial_len*100:.1f}% dos dados)")
        print(f"    Formato final: {df.shape}")
        print(f"    Total de features criadas: {len(df.columns)}")
        
        # Mostrar algumas features criadas
        print(f"    Exemplos de features criadas: {list(df.columns[:15])}...")
        
        return df

    # ------------------------------------------------------------
    def _make_all_horizon_windows(self, df_feat: pd.DataFrame, 
                                  target_cols: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Cria janelas temporais para todos os horizontes.
        
        IMPORTANTE: Densidade NÃO é mais constante - foi normalizada corretamente
        """
        lookback = self.config.get("training")["main_lookback"]
        horizons = self.config.get("horizons")
        
        print(f"    Lookback: {lookback}, Horizons: {horizons}")
        
        # Separar features (X) e targets (Y)
        feature_data_cols = [col for col in df_feat.columns if col not in target_cols]
        X_data = df_feat[feature_data_cols].values.astype(np.float32)
        
        # Targets: valores ORIGINAIS (não escalonados)
        y_data_raw = df_feat[target_cols].values.astype(np.float32)
        
        print(f"    X shape: {X_data.shape}, y shape: {y_data_raw.shape}")
        print(f"    Feature columns: {len(feature_data_cols)}")
        
        # ✅ VERIFICAÇÃO DE VARIÁVEIS CONSTANTES (para debug)
        print(f"\n    🔍 Verificando variáveis nos targets...")
        for idx, var_name in enumerate(target_cols):
            var_data = y_data_raw[:, idx]
            std_val = np.std(var_data)
            range_val = np.ptp(var_data)
            nan_count = np.isnan(var_data).sum()
            
            print(f"      {var_name}:")
            print(f"        • std={std_val:.4f}, range={range_val:.2f}, NaNs={nan_count}")
            
            if std_val < 0.01:  # Threshold muito baixo para constante
                print(f"        ⚠️  VARIÁVEL QUASE CONSTANTE: {var_name} (std={std_val:.6f})")
            else:
                print(f"        ✅ OK para treinamento (std={std_val:.4f})")
        
        datasets: Dict[int, Dict[str, Any]] = {}
        
        for horizon in horizons:
            print(f"\n    🪟 Horizonte {horizon}h...")
            
            # Inicializar scalers Y para ESTE horizonte
            self.scalers_y[horizon] = {}
            
            # Criar janelas temporais
            X_windows, y_windows_raw = [], []
            
            # Limitar tamanho para GitHub Free
            max_samples = len(X_data) - lookback - horizon
            max_samples = min(max_samples, 50000)
            
            print(f"      Criando até {max_samples} janelas...")
            
            for i in range(max_samples):
                X_windows.append(X_data[i:i + lookback])
                y_windows_raw.append(y_data_raw[i + lookback + horizon])
            
            # Converter para arrays numpy
            X_arr = np.array(X_windows, dtype=np.float32)
            y_arr_raw = np.array(y_windows_raw, dtype=np.float32)
            
            print(f"      X shape: {X_arr.shape}, y raw shape: {y_arr_raw.shape}")
            
            # -----------------------------------------------------------------
            # ESCALONAMENTO Y POR VARIÁVEL FÍSICA
            # -----------------------------------------------------------------
            y_arr_scaled = np.zeros_like(y_arr_raw, dtype=np.float32)
            
            for idx, var_name in enumerate(target_cols):
                # Criar scaler INDEPENDENTE para esta variável física
                scaler = StandardScaler()
                
                # Extrair apenas ESTA coluna do y
                y_single_var = y_arr_raw[:, idx].reshape(-1, 1)
                
                # Verificar se ainda é constante (por segurança)
                var_std = np.std(y_single_var)
                
                if var_std < 1e-6:  # Ainda constante mesmo após filtro
                    print(f"      ⚠️  ATENÇÃO: {var_name} ainda constante (std={var_std:.6f})")
                    print(f"        Usando StandardScaler(with_std=False)")
                    scaler = StandardScaler(with_std=False)
                elif var_std < 0.1:  # Baixa variância
                    print(f"      ℹ️  {var_name} tem baixa variância (std={var_std:.4f})")
                    scaler = StandardScaler()
                else:
                    scaler = StandardScaler()
                
                # Escalonar APENAS esta variável
                y_scaled_single = scaler.fit_transform(y_single_var)
                
                # Guardar no array escalonado
                y_arr_scaled[:, idx] = y_scaled_single.flatten()
                
                # Salvar scaler para esta variável
                self.scalers_y[horizon][var_name] = scaler
                
                # Log para debug (apenas primeira variável e densidade)
                if idx == 0 or var_name == 'density':
                    print(f"      {var_name}: mean={scaler.mean_[0]:.4f}, scale={scaler.scale_[0]:.4f}")
            
            # -----------------------------------------------------------------
            # Montar dataset para este horizonte
            # -----------------------------------------------------------------
            datasets[horizon] = {
                "X": X_arr,                    # Features escalonadas
                "y_scaled": y_arr_scaled,      # Targets escalonados (PARA TREINO)
                "y_raw": y_arr_raw,            # Targets originais (PARA MÉTRICAS)
                "target_names": target_cols,   # Nomes das variáveis
                "feature_names": feature_data_cols,
                "horizon": horizon,
                "lookback": lookback
            }
            
            # Limpar memória entre horizontes
            del X_windows, y_windows_raw
            gc.collect()
        
        return datasets

    # ------------------------------------------------------------
    def get_y_scalers(self, horizon: int) -> Dict[str, StandardScaler]:
        """Retorna os scalers Y para um horizonte específico."""
        return self.scalers_y.get(horizon, {})


# ------------------------------------------------------------
# Execução direta (para teste)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 HAC v6 FEATURE BUILDER - COM DENSIDADE RECUPERADA")
    print("=" * 70)
    
    try:
        # 1. Carregar configuração
        config = HACConfig("config.yaml")
        
        # 2. Criar feature builder
        builder = HACFeatureBuilder(config)
        
        # 3. Executar pipeline completo
        datasets = builder.build_all()
        
        # 4. Salvar resultados
        out_dir = os.path.join(config.get("paths")["data_dir"], "prepared")
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n💾 Salvando datasets em: {out_dir}")
        
        # Salvar cada horizonte em arquivo NPZ compactado
        for horizon, data in datasets.items():
            npz_path = os.path.join(out_dir, f"dataset_h{horizon}.npz")
            np.savez_compressed(
                npz_path,
                X=data["X"],
                y_scaled=data["y_scaled"],
                y_raw=data["y_raw"],
                target_names=data["target_names"],
                feature_names=data["feature_names"],
                horizon=data["horizon"],
                lookback=data["lookback"]
            )
            print(f"   ✅ Horizonte {horizon}h: {npz_path}")
            print(f"      • Amostras: {data['X'].shape[0]}")
            print(f"      • Lookback: {data['lookback']}")
            print(f"      • Features: {data['X'].shape[2]}")
            print(f"      • Targets: {data['target_names']}")
        
        # Salvar scalers X
        scaler_x_path = os.path.join(out_dir, "scaler_X.pkl")
        joblib.dump(builder.scaler_X, scaler_x_path)
        print(f"   ✅ Scaler X: {scaler_x_path}")
        
        # Salvar scalers Y (por horizonte e variável)
        y_scalers_info = {}
        for h, scaler_dict in builder.scalers_y.items():
            y_scalers_info[str(h)] = {}
            for var_name, scaler in scaler_dict.items():
                y_scalers_info[str(h)][var_name] = {
                    "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [],
                    "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [],
                    "n_samples": int(scaler.n_samples_seen_) if hasattr(scaler, 'n_samples_seen_') else 0
                }
        
        scalers_y_path = os.path.join(out_dir, "y_scalers_info.json")
        with open(scalers_y_path, "w") as f:
            json.dump(y_scalers_info, f, indent=2)
        
        print(f"   ✅ Scalers Y: {scalers_y_path}")
        
        # 5. Resumo final
        print("\n" + "=" * 70)
        print("✅ FEATURE BUILD COMPLETADO COM SUCESSO!")
        print("=" * 70)
        
        # Mostrar estatísticas dos datasets
        for horizon in datasets.keys():
            data = datasets[horizon]
            print(f"\n📊 Dataset {horizon}h:")
            print(f"   • Amostras: {data['X'].shape[0]}")
            print(f"   • Lookback: {data['lookback']}")
            print(f"   • Features: {data['X'].shape[2]} colunas")
            print(f"   • Targets: {', '.join(data['target_names'])}")
            
            # Estatísticas dos targets originais
            y_raw = data["y_raw"]
            for idx, name in enumerate(data["target_names"]):
                print(f"     - {name}:")
                print(f"       • média={y_raw[:, idx].mean():.4f}")
                print(f"       • std={y_raw[:, idx].std():.4f}")
                print(f"       • min={y_raw[:, idx].min():.4f}")
                print(f"       • max={y_raw[:, idx].max():.4f}")
                
                # Obter scaler para esta variável
                scaler = builder.scalers_y[horizon][name]
                print(f"       • escalonado: mean={scaler.mean_[0]:.4f}, scale={scaler.scale_[0]:.4f}")
        
        print("\n🎯 DENSIDADE RECUPERADA E INCLUÍDA NO MODELO!")
        print("📈 Pronto para treinar HAC v6 com features físicas completas!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRO: {e}")
        print("   Execute primeiro: python prepare_omni_dataset.py")
        
    except Exception as e:
        print(f"\n❌ ERRO inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
