#!/usr/bin/env python3
"""
hac_v6_features.py - CORRIGIDO
Feature Engineering & Dataset Builder para HAC v6 com FÍSICA CORRETA.

CORREÇÕES PRINCIPAIS:
1. Escalonamento Y SEPARADO por variável física (V, Bz, n)
2. Engineering de features físicas do Bz (Bz_sul, acumulados, função de Newell)
3. Otimizado para GitHub Free (controle de memória, float32)

Fluxo:
- Carrega omni_prepared.csv
- Gera features físicas do Bz
- Faz lags + estatísticas móveis
- Escala X (features) com UM scaler
- Escala Y (targets) com SCALERS SEPARADOS (um por variável física)
- Cria janelas temporais para todos os horizontes
"""

import os
import gc
import json
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from hac_v6_config import HACConfig


class HACFeatureBuilder:
    """Gera features e datasets supervisionados para HAC v6 com física correta."""

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
    def build_all(self) -> Dict[int, Dict[str, Any]]:
        """Pipeline principal: engineer → scale → windows."""
        print("\n🔧 Iniciando pipeline de construção de features...")
        
        # 1. Engenharia de features (incluindo features físicas do Bz)
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
        Cria features: físicas do Bz + lags + estatísticas móveis.
        
        FEATURES FÍSICAS DO Bz (novas):
        - Bz_sul: apenas componente sul (negativa) do Bz
        - Bz_neg_acum: Bz negativo acumulado (proxy para injeção de energia)
        - Newell_coupling: função de acoplamento V * |Bz_sul| (aproximada)
        """
        print(f"    Formato inicial: {df.shape}")
        
        # Remover colunas temporais não-numéricas
        non_numeric = ["datetime", "timestamp", "year", "doy", "hour"]
        for col in non_numeric:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # -----------------------------------------------------------------
        # NOVO: ENGINEERING FÍSICO DO Bz (ANTES de criar lags)
        # -----------------------------------------------------------------
        if 'Bz' in df.columns:
            print("    Criando features físicas do Bz...")
            
            # 1. Componente SUL do Bz (fisicamente relevante para reconexão)
            #    Bz_sul = min(Bz, 0) → apenas valores negativos, zero se positivo
            df['Bz_sul'] = df['Bz'].clip(upper=0)
            
            # 2. Valor absoluto do Bz sul (para funções de acoplamento)
            df['Bz_sul_abs'] = np.abs(df['Bz_sul'])
            
            # 3. Bz negativo acumulado (proxy para injeção de energia na magnetosfera)
            #    Dados OMNI tipicamente em 1-min ou 5-min, ajustar para horas
            window_1h = 12  # Supondo dados a 5-min: 12 pontos = 1 hora
            window_3h = 36  # 36 pontos = 3 horas
            
            df['Bz_neg_acum_1h'] = df['Bz_sul'].rolling(window=window_1h, min_periods=1).sum()
            df['Bz_neg_acum_3h'] = df['Bz_sul'].rolling(window=window_3h, min_periods=1).sum()
            
            # 4. Função de acoplamento de Newell (aproximada: ε = V * B_T² * sin⁴(θ/2))
            #    Simplificação: ε ≈ V * |Bz_sul|² (para vento solar radial)
            if 'V' in df.columns:
                df['Newell_coupling'] = df['V'] * (df['Bz_sul_abs'] ** 2)
                print(f"      ✅ Função de acoplamento de Newell criada")
            
            print(f"      ✅ Features físicas do Bz criadas: Bz_sul, acumulados, Newell")
        
        # -----------------------------------------------------------------
        # Features base para engenharia (targets primários + secundários)
        # -----------------------------------------------------------------
        targets_cfg = self.config.get("targets")
        primary = targets_cfg.get("primary", [])
        secondary = targets_cfg.get("secondary", [])
        
        # Incluir as NOVAS features físicas do Bz nas secundárias para criar lags
        bz_physical_features = ['Bz_sul', 'Bz_sul_abs', 'Bz_neg_acum_1h', 'Bz_neg_acum_3h']
        if 'Newell_coupling' in df.columns:
            bz_physical_features.append('Newell_coupling')
        
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
        
        CORREÇÃO CRÍTICA: Escalonamento Y SEPARADO por variável física.
        Cada target (V, Bz, n) tem seu próprio scaler, preservando unidades físicas.
        
        Args:
            df_feat: DataFrame com features (X escalonado) e targets (Y original)
            target_cols: Lista de targets primários (ex: ["V", "Bz", "n"])
            
        Returns:
            Dicionário com datasets por horizonte
        """
        lookback = self.config.get("training")["main_lookback"]
        horizons = self.config.get("horizons")
        
        print(f"    Lookback: {lookback}, Horizons: {horizons}")
        
        # Separar features (X) e targets (Y)
        # Features: todas as colunas EXCETO os targets
        feature_data_cols = [col for col in df_feat.columns if col not in target_cols]
        X_data = df_feat[feature_data_cols].values.astype(np.float32)
        
        # Targets: valores ORIGINAIS (não escalonados)
        y_data_raw = df_feat[target_cols].values.astype(np.float32)
        
        print(f"    X shape: {X_data.shape}, y shape: {y_data_raw.shape}")
        print(f"    Feature columns: {len(feature_data_cols)}")
        
        datasets: Dict[int, Dict[str, Any]] = {}
        
        for horizon in horizons:
            print(f"\n    🪟 Horizonte {horizon}h...")
            
            # Inicializar scalers Y para ESTE horizonte
            # Um scaler INDEPENDENTE para cada variável física
            self.scalers_y[horizon] = {}
            
            # Criar janelas temporais
            X_windows, y_windows_raw = [], []
            
            # Limitar tamanho para GitHub Free
            max_samples = len(X_data) - lookback - horizon
            max_samples = min(max_samples, 50000)  # Limite seguro
            
            print(f"      Criando até {max_samples} janelas...")
            
            for i in range(max_samples):
                # Janela de features (lookback steps)
                X_windows.append(X_data[i:i + lookback])
                # Target (horizon steps à frente)
                y_windows_raw.append(y_data_raw[i + lookback + horizon])
            
            # Converter para arrays numpy
            X_arr = np.array(X_windows, dtype=np.float32)
            y_arr_raw = np.array(y_windows_raw, dtype=np.float32)
            
            print(f"      X shape: {X_arr.shape}, y raw shape: {y_arr_raw.shape}")
            
            # -----------------------------------------------------------------
            # CORREÇÃO: ESCALONAMENTO Y POR VARIÁVEL FÍSICA
            # -----------------------------------------------------------------
            y_arr_scaled = np.zeros_like(y_arr_raw, dtype=np.float32)
            
            for idx, var_name in enumerate(target_cols):
                # Criar scaler INDEPENDENTE para esta variável física
                scaler = StandardScaler()
                
                # Extrair apenas ESTA coluna do y
                y_single_var = y_arr_raw[:, idx].reshape(-1, 1)
                
                # Escalonar APENAS esta variável
                y_scaled_single = scaler.fit_transform(y_single_var)
                
                # Guardar no array escalonado
                y_arr_scaled[:, idx] = y_scaled_single.flatten()
                
                # Salvar scaler para esta variável
                self.scalers_y[horizon][var_name] = scaler
                
                # Log para debug
                if idx == 0:  # Apenas para primeira variável para não poluir output
                    print(f"      Scalers Y criados para {len(target_cols)} variáveis")
                    print(f"        {var_name}: mean={scaler.mean_[0]:.2f}, scale={scaler.scale_[0]:.2f}")
            
            # -----------------------------------------------------------------
            # Montar dataset para este horizonte
            # -----------------------------------------------------------------
            datasets[horizon] = {
                "X": X_arr,                    # Features escalonadas
                "y_scaled": y_arr_scaled,      # Targets escalonados (PARA TREINO)
                "y_raw": y_arr_raw,            # Targets originais (PARA MÉTRICAS)
                "target_names": target_cols,   # Nomes das variáveis
                "feature_names": feature_data_cols,  # Nomes das features
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
    
    def get_feature_names(self) -> List[str]:
        """Retorna os nomes das features (após engineering)."""
        # Isso seria implementado se necessário
        return []


# ------------------------------------------------------------
# Execução direta (para teste)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 HAC v6 FEATURE BUILDER - COM FÍSICA CORRIGIDA")
    print("=" * 60)
    
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
                feature_names=data["feature_names"]
            )
            print(f"   ✅ Horizonte {horizon}h: {npz_path} ({data['X'].shape[0]} amostras)")
        
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
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_]
            "n_samples": scaler.n_samples_seen_
                }
        
        scalers_y_path = os.path.join(out_dir, "y_scalers_info.json")
        with open(scalers_y_path, "w") as f:
            json.dump(y_scalers_info, f, indent=2)
        
        print(f"   ✅ Scalers Y: {scalers_y_path}")
        
        # 5. Resumo final
        print("\n" + "=" * 60)
        print("✅ FEATURE BUILD COMPLETADO COM SUCESSO!")
        print("=" * 60)
        
        # Mostrar estatísticas
        for horizon in datasets.keys():
            data = datasets[horizon]
            print(f"\n📊 Horizonte {horizon}h:")
            print(f"   • Amostras: {data['X'].shape[0]}")
            print(f"   • Lookback: {data['lookback']}")
            print(f"   • Features: {data['X'].shape[2]} colunas")
            print(f"   • Targets: {', '.join(data['target_names'])}")
            
            # Mostrar estatísticas dos targets originais
            y_raw = data["y_raw"]
            for idx, name in enumerate(data["target_names"]):
                print(f"     - {name}: {y_raw[:, idx].mean():.2f} ± {y_raw[:, idx].std():.2f}")
        
        print("\n🎯 Pronto para treino com FÍSICA CORRETA!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRO: {e}")
        print("   Certifique-se de que o arquivo 'data/omni_prepared.csv' existe.")
        print("   Dica: Verifique o caminho no config.yaml")
        
    except Exception as e:
        print(f"\n❌ ERRO inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
