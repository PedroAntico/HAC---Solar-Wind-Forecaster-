#!/usr/bin/env python3
"""
hac_v6_features.py - CORREÇÃO DEFINITIVA DOS NOMES DE VARIÁVEIS
Feature Engineering & Dataset Builder para HAC v6.

CORREÇÕES CRÍTICAS APLICADAS:
1. Usa nomes CORRETOS das variáveis do CSV: 'speed', 'bz_gsm', 'density'
2. Trata densidade constante removendo-a ANTES do escalonamento
3. Engineering de features físicas baseado nos nomes reais
4. Escalonamento Y por variável com tratamento de constantes
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
                
                print(f"   {target}:")
                print(f"     • Média: {mean_val:.2f}")
                print(f"     • Desvio padrão: {std_val:.2f}")
                print(f"     • Range: {min_val:.1f} - {max_val:.1f}")
                print(f"     • Valores únicos: {unique_count}")
                
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
        - 'speed' (não 'V')
        - 'bz_gsm' (não 'Bz')
        - 'density' (não 'n')
        """
        print(f"    Formato inicial: {df.shape}")
        
        # Remover colunas temporais não-numéricas
        non_numeric = ["datetime", "timestamp", "year", "doy", "hour"]
        for col in non_numeric:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # -----------------------------------------------------------------
        # ENGINEERING FÍSICO DO Bz (usando 'bz_gsm' que existe no CSV)
        # -----------------------------------------------------------------
        if 'bz_gsm' in df.columns:  # ✅ CORREÇÃO: 'bz_gsm' não 'Bz'
            print("    Criando features físicas do Bz...")
            
            # 1. Componente SUL do Bz (fisicamente relevante para reconexão)
            #    Bz_sul = min(bz_gsm, 0) → apenas valores negativas, zero se positivo
            df['bz_sul'] = df['bz_gsm'].clip(upper=0)  # ✅ CORREÇÃO: coluna 'bz_gsm'
            
            # 2. Valor absoluto do Bz sul (para funções de acoplamento)
            df['bz_sul_abs'] = np.abs(df['bz_sul'])
            
            # 3. Bz negativo acumulado (proxy para injeção de energia na magnetosfera)
            window_1h = 12  # Supondo dados a 5-min: 12 pontos = 1 hora
            window_3h = 36  # 36 pontos = 3 horas
            
            df['bz_neg_acum_1h'] = df['bz_sul'].rolling(window=window_1h, min_periods=1).sum()
            df['bz_neg_acum_3h'] = df['bz_sul'].rolling(window=window_3h, min_periods=1).sum()
            
            # 4. Função de acoplamento de Newell (aproximada)
            if 'speed' in df.columns:  # ✅ CORREÇÃO: 'speed' não 'V'
                df['newell_coupling'] = df['speed'] * (df['bz_sul_abs'] ** 2)
                print(f"      ✅ Função de acoplamento de Newell criada")
            
            print(f"      ✅ Features físicas do Bz criadas: bz_sul, acumulados, newell")
        
        # -----------------------------------------------------------------
        # Features base para engenharia (targets primários + secundários)
        # -----------------------------------------------------------------
        targets_cfg = self.config.get("targets")
        primary = targets_cfg.get("primary", [])
        secondary = targets_cfg.get("secondary", [])
        
        # Incluir as features físicas do Bz nas secundárias para criar lags
        bz_physical_features = ['bz_sul', 'bz_sul_abs', 'bz_neg_acum_1h', 'bz_neg_acum_3h']
        if 'newell_coupling' in df.columns:
            bz_physical_features.append('newell_coupling')
        
        # Combinar todas as colunas base para criar lags
        base_cols = []
        for col in primary + secondary + bz_physical_features:
            if col in df.columns and col not in base_cols:
                base_cols.append(col)
        
        print(f"    Criando lags e estatísticas para {len(base_cols)} colunas base...")
        
        # -----------------------------------------------------------------
        # Lags e estatísticas móveis (para todas as colunas base)
        # -----------------------------------------------------------------
        for col in base_cols:
            # Lags (1 e 3 passos atrás)
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag3"] = df[col].shift(3)
            
            # Médias móveis (6 e 12 passos)
            df[f"{col}_rm6"] = df[col].rolling(window=6, min_periods=3).mean()
            df[f"{col}_rm12"] = df[col].rolling(window=12, min_periods=6).mean()
            
            # Desvio padrão móvel
            df[f"{col}_std6"] = df[col].rolling(window=6, min_periods=3).std()
        
        # -----------------------------------------------------------------
        # Limpar NaNs e retornar
        # -----------------------------------------------------------------
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        final_len = len(df)
        
        print(f"    Removidos {initial_len - final_len} NaNs")
        print(f"    Formato final: {df.shape}")
        
        return df

    # ------------------------------------------------------------
    def _make_all_horizon_windows(self, df_feat: pd.DataFrame, 
                                  target_cols: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Cria janelas temporais para todos os horizontes.
        
        IMPORTANTE: DETECTA E REMOVE VARIÁVEIS CONSTANTES ANTES DO ESCALONAMENTO!
        Se 'density' for constante (sempre 50), será removida automaticamente.
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
        
        # ✅ CORREÇÃO CRÍTICA: Verificar quais targets são constantes
        constant_vars = []
        valid_target_indices = []
        valid_target_names = []
        
        print(f"\n    🔍 Verificando variáveis constantes nos targets...")
        for idx, var_name in enumerate(target_cols):
            var_data = y_data_raw[:, idx]
            std_val = np.std(var_data)
            range_val = np.ptp(var_data)  # peak-to-peak (max-min)
            
            print(f"      {var_name}: std={std_val:.6f}, range={range_val:.2f}")
            
            if std_val < 0.1 or range_val < 1.0:  # Threshold para constante
                constant_vars.append(var_name)
                print(f"      ⚠️  VARIÁVEL CONSTANTE: {var_name} (std={std_val:.3f}, range={range_val:.1f})")
            else:
                valid_target_indices.append(idx)
                valid_target_names.append(var_name)
                print(f"      ✅ {var_name} OK para treinamento")
        
        # ✅ Se houver variáveis constantes, criar dataset apenas com as variáveis válidas
        if constant_vars:
            print(f"\n    🛠️  Removendo variáveis constantes: {constant_vars}")
            print(f"    ✅ Targets válidos para treinamento: {valid_target_names}")
            
            if not valid_target_names:
                raise ValueError(f"❌ TODAS as variáveis target são constantes! Verifique o dataset.")
            
            # Manter apenas as colunas válidas
            y_data_raw = y_data_raw[:, valid_target_indices]
            target_cols = valid_target_names
            print(f"    ✅ Novo shape de y: {y_data_raw.shape}")
        else:
            print(f"    ✅ Nenhuma variável constante detectada")
        
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
                    print(f"      ⚠️  ATENÇÃO: {var_name} ainda constante após filtro")
                    print(f"        Usando StandardScaler(with_std=False)")
                    scaler = StandardScaler(with_std=False)
                
                # Escalonar APENAS esta variável
                y_scaled_single = scaler.fit_transform(y_single_var)
                
                # Guardar no array escalonado
                y_arr_scaled[:, idx] = y_scaled_single.flatten()
                
                # Salvar scaler para esta variável
                self.scalers_y[horizon][var_name] = scaler
                
                # Log para debug (apenas primeira variável)
                if idx == 0 and len(target_cols) > 0:
                    print(f"      Scalers Y criados para {len(target_cols)} variáveis")
                    print(f"        {var_name}: mean={scaler.mean_[0]:.2f}, scale={scaler.scale_[0]:.2f}")
            
            # -----------------------------------------------------------------
            # Montar dataset para este horizonte
            # -----------------------------------------------------------------
            datasets[horizon] = {
                "X": X_arr,                    # Features escalonadas
                "y_scaled": y_arr_scaled,      # Targets escalonados (PARA TREINO)
                "y_raw": y_arr_raw,            # Targets originais (PARA MÉTRICAS)
                "target_names": target_cols,   # Nomes DAS VARIÁVEIS VÁLIDAS
                "feature_names": feature_data_cols,
                "horizon": horizon,
                "lookback": lookback,
                "constant_variables_removed": constant_vars  # Para debug
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
    print("🚀 HAC v6 FEATURE BUILDER - CORREÇÃO DEFINITIVA DOS NOMES")
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
                lookback=data["lookback"],
                constant_variables_removed=data.get("constant_variables_removed", [])
            )
            print(f"   ✅ Horizonte {horizon}h: {npz_path}")
            print(f"      • Amostras: {data['X'].shape[0]}")
            print(f"      • Targets: {data['target_names']}")
            print(f"      • Variáveis constantes removidas: {data.get('constant_variables_removed', [])}")
        
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
            print(f"   • Targets válidos: {', '.join(data['target_names'])}")
            
            if data.get("constant_variables_removed"):
                print(f"   • Removidos (constantes): {', '.join(data['constant_variables_removed'])}")
            
            # Estatísticas dos targets originais
            y_raw = data["y_raw"]
            for idx, name in enumerate(data["target_names"]):
                print(f"     - {name}: média={y_raw[:, idx].mean():.2f}, "
                      f"std={y_raw[:, idx].std():.2f}, "
                      f"range={y_raw[:, idx].min():.1f}-{y_raw[:, idx].max():.1f}")
        
        print("\n🎯 Pronto para treino com VARIÁVEIS CORRETAS!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRO: {e}")
        print("   Execute primeiro: python prepare_omni_dataset.py")
        
    except Exception as e:
        print(f"\n❌ ERRO inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
