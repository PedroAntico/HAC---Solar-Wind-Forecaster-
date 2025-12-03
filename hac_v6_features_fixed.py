#!/usr/bin/env python3
"""
hac_v6_features_fixed.py - Versão corrigida com diagnóstico melhorado
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
        
        # Validar colunas necessárias COM THRESHOLDS ADEQUADOS
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
        
        # Verificar estatísticas das variáveis com thresholds realistas
        print("\n📊 Estatísticas iniciais dos targets:")
        for target in targets:
            if target in self.raw_df.columns:
                values = self.raw_df[target].dropna()
                
                print(f"   {target}:")
                print(f"     • Média: {values.mean():.2f}")
                print(f"     • Std: {values.std():.6f}")  # Mostrar mais decimais
                print(f"     • Min: {values.min():.2f}")
                print(f"     • Max: {values.max():.2f}")
                print(f"     • Únicos: {values.nunique()}")
                
                # Thresholds mais realistas por variável
                if target == 'speed':
                    # Velocidade: std esperado > 50 km/s
                    if values.std() < 10:
                        print(f"     ⚠️  VELOCIDADE COM BAIXA VARIABILIDADE (std={values.std():.6f})")
                        print(f"     💡 Valores típicos: 300-800 km/s com std ~100-200")
                elif target == 'density':
                    # Densidade: std esperado > 1 cm^-3
                    if values.std() < 0.5:
                        print(f"     ⚠️  Densidade com baixa variabilidade")
                elif target == 'bz_gsm':
                    # Bz: std esperado > 2 nT
                    if values.std() < 0.5:
                        print(f"     ⚠️  Bz com baixa variabilidade")

    # ------------------------------------------------------------
    def _analyze_variables(self):
        """Analisa as variáveis target para detectar problemas."""
        print("\n🔍 Análise detalhada das variáveis target:")
        
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
                print(f"     • Desvio padrão: {std_val:.6f}")  # Mais precisão
                print(f"     • Range: {min_val:.1f} - {max_val:.1f}")
                print(f"     • Valores únicos: {unique_count}")
                print(f"     • Coeficiente de variação (CV): {std_val/mean_val*100:.1f}%")
                
                # Thresholds específicos por variável
                if target == 'speed':
                    if std_val < 10:
                        print(f"     ⚠️⚠️⚠️ ATENÇÃO: Velocidade quase constante (std={std_val:.6f})")
                        print(f"     💡 O problema pode estar na fonte de dados")
                    elif std_val < 50:
                        print(f"     ⚠️  Velocidade com variabilidade moderada (std={std_val:.2f})")
                    else:
                        print(f"     ✅ Velocidade OK")
                elif target == 'bz_gsm':
                    if std_val < 0.5:
                        print(f"     ⚠️  Bz com baixa variabilidade")
                    else:
                        print(f"     ✅ Bz OK")
                elif target == 'density':
                    if std_val < 0.5:
                        print(f"     ⚠️  Densidade com baixa variabilidade")
                    else:
                        print(f"     ✅ Densidade OK")

    # ------------------------------------------------------------
    def build_all(self) -> Dict[int, Dict[str, Any]]:
        """Pipeline principal com separação clara entre features e targets."""
        print("\n🔧 Iniciando pipeline de construção de features...")
        
        # 0. Verificação EXTRA antes de qualquer processamento
        print("  0. Verificação EXTRA pré-processamento...")
        self._extra_preprocessing_check()
        
        # 1. Engenharia de features (preservando targets originais)
        print("  1. Criando features físicas e estatísticas...")
        df_with_features, targets_original = self._engineer_features_safe()
        
        # 2. Separar features (X) e targets (Y) 
        print("  2. Separando features e targets...")
        
        # Obter lista de todas as colunas exceto os targets
        feature_cols = [col for col in df_with_features.columns 
                       if col not in targets_original.columns]
        
        print(f"    • {len(feature_cols)} features")
        print(f"    • {len(targets_original.columns)} targets")
        
        # 3. Escalonar APENAS as features (X)
        print("  3. Escalonando features (X)...")
        X_data = self.scaler_X.fit_transform(df_with_features[feature_cols])
        
        # Targets (Y) permanecem ORIGINAIS por enquanto
        y_data = targets_original.values.astype(np.float32)
        
        print(f"    ✅ Features escalonadas: {X_data.shape}")
        print(f"    ✅ Targets preservados: {y_data.shape}")
        
        # Verificação EXTRA após separação
        print("\n    🔍 Verificação EXTRA após separação:")
        for idx, name in enumerate(targets_original.columns):
            std_val = np.std(y_data[:, idx])
            print(f"      {name}: std={std_val:.6f}")
        
        # 4. Criar janelas temporais
        print("  4. Criando janelas temporais para cada horizonte...")
        datasets = self._make_all_horizon_windows_safe(
            X_data, y_data, feature_cols, targets_original.columns.tolist()
        )
        
        # 5. Limpar memória
        gc.collect()
        
        print(f"\n✅ Pipeline completo. {len(datasets)} horizontes processados.")
        return datasets

    # ------------------------------------------------------------
    def _extra_preprocessing_check(self):
        """Verificação extra para detectar problemas antes do processamento."""
        print("    🔍 Verificando dados antes do processamento...")
        
        targets = self.config.get("targets")["primary"]
        
        for target in targets:
            if target in self.raw_df.columns:
                values = self.raw_df[target].dropna()
                
                # Verificar se há muitos valores iguais
                value_counts = values.value_counts()
                most_common = value_counts.iloc[0] if len(value_counts) > 0 else 0
                
                print(f"      {target}:")
                print(f"        • Total: {len(values)}")
                print(f"        • Valor mais comum: {value_counts.index[0] if len(value_counts) > 0 else 'N/A'}")
                print(f"        • Frequência do mais comum: {most_common} ({most_common/len(values)*100:.1f}%)")
                
                # Se mais de 30% dos valores forem iguais, alerta
                if most_common / len(values) > 0.3:
                    print(f"        ⚠️⚠️⚠️ MUITOS VALORES REPETIDOS!")

    # ------------------------------------------------------------
    def _engineer_features_safe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cria features SEM modificar os targets originais.
        Retorna: (features_df, targets_df)
        """
        print(f"    Formato inicial do raw: {self.raw_df.shape}")
        
        # Fazer uma cópia dos targets ORIGINAIS
        target_cols = self.config.get("targets")["primary"]
        targets_df = self.raw_df[target_cols].copy()
        
        # Criar DataFrame para features (iniciar vazio)
        features_df = pd.DataFrame(index=self.raw_df.index)
        
        # -----------------------------------------------------------------
        # 1. Features temporais básicas (se existirem)
        # -----------------------------------------------------------------
        temporal_cols = ["year", "doy", "hour"]
        for col in temporal_cols:
            if col in self.raw_df.columns:
                features_df[col] = self.raw_df[col]
        
        # -----------------------------------------------------------------
        # 2. Criar features derivadas dos targets (mas NÃO os próprios targets!)
        # -----------------------------------------------------------------
        print("    Criando features derivadas dos targets...")
        
        # Para cada target, criar lags e estatísticas
        for target in target_cols:
            if target in self.raw_df.columns:
                # Usar valores ORIGINAIS para criar features
                original_series = self.raw_df[target]
                
                print(f"      Criando features para {target} (std original: {original_series.std():.6f})")
                
                # Lags (1, 3, 6 passos)
                features_df[f"{target}_lag1"] = original_series.shift(1)
                features_df[f"{target}_lag3"] = original_series.shift(3)
                features_df[f"{target}_lag6"] = original_series.shift(6)
                
                # Médias móveis
                features_df[f"{target}_ma_6"] = original_series.rolling(window=6, min_periods=3).mean()
                features_df[f"{target}_ma_12"] = original_series.rolling(window=12, min_periods=6).mean()
                features_df[f"{target}_ma_24"] = original_series.rolling(window=24, min_periods=12).mean()
                
                # Desvios padrão móveis
                features_df[f"{target}_std_6"] = original_series.rolling(window=6, min_periods=3).std()
                features_df[f"{target}_std_12"] = original_series.rolling(window=12, min_periods=6).std()
                
                # Range móvel
                features_df[f"{target}_range_12"] = (
                    original_series.rolling(window=12, min_periods=6).max() -
                    original_series.rolling(window=12, min_periods=6).min()
                )
        
        # -----------------------------------------------------------------
        # 3. Features físicas específicas
        # -----------------------------------------------------------------
        print("    Criando features físicas específicas...")
        
        # Features do Bz
        if 'bz_gsm' in self.raw_df.columns:
            bz = self.raw_df['bz_gsm']
            features_df['bz_south'] = bz.clip(upper=0)
            features_df['bz_south_abs'] = np.abs(features_df['bz_south'])
            
            # Acumulados de Bz sul
            features_df['bz_south_cum1h'] = features_df['bz_south'].rolling(window=12, min_periods=1).sum()
            features_df['bz_south_cum3h'] = features_df['bz_south'].rolling(window=36, min_periods=1).sum()
        
        # Features da densidade
        if 'density' in self.raw_df.columns:
            density = self.raw_df['density']
            features_df['density_log'] = np.log1p(density)
            features_df['density_pct_change'] = density.pct_change().fillna(0)
        
        # Features da velocidade
        if 'speed' in self.raw_df.columns:
            speed = self.raw_df['speed']
            features_df['speed_acceleration'] = speed.diff().fillna(0)
            features_df['speed_smoothed'] = speed.rolling(window=12, min_periods=6).mean()
        
        # -----------------------------------------------------------------
        # 4. Interações físicas
        # -----------------------------------------------------------------
        if all(col in self.raw_df.columns for col in ['density', 'speed']):
            density = self.raw_df['density']
            speed = self.raw_df['speed']
            features_df['dynamic_pressure'] = density * (speed ** 2)
            features_df['momentum_flux'] = density * speed
        
        if all(col in self.raw_df.columns for col in ['speed', 'bz_gsm']):
            speed = self.raw_df['speed']
            bz = self.raw_df['bz_gsm']
            bz_south = bz.clip(upper=0)
            features_df['newell_coupling'] = speed * (np.abs(bz_south) ** 2)
        
        # -----------------------------------------------------------------
        # 5. Remover NaNs e alinhar índices
        # -----------------------------------------------------------------
        print("    Removendo NaNs e alinhando dados...")
        
        # Juntar features e targets temporariamente para remover NaNs
        combined = pd.concat([features_df, targets_df], axis=1)
        initial_len = len(combined)
        combined_clean = combined.dropna()
        final_len = len(combined_clean)
        
        # Separar novamente
        features_clean = combined_clean[features_df.columns].reset_index(drop=True)
        targets_clean = combined_clean[targets_df.columns].reset_index(drop=True)
        
        print(f"    Removidos {initial_len - final_len} NaNs")
        print(f"    Features shape: {features_clean.shape}")
        print(f"    Targets shape: {targets_clean.shape}")
        
        # Verificação EXTRA após limpeza
        print("\n    🔍 Verificação EXTRA após limpeza:")
        for target in target_cols:
            if target in targets_clean.columns:
                std_val = targets_clean[target].std()
                print(f"      {target}: std={std_val:.6f}")
                
                # Verificar se a limpeza não destruiu a variabilidade
                std_original = self.raw_df[target].std()
                if std_val < std_original * 0.1:  # Se perdeu mais de 90% da variabilidade
                    print(f"      ⚠️⚠️⚠️ ATENÇÃO: {target} perdeu muita variabilidade!")
                    print(f"        Original: std={std_original:.6f}")
                    print(f"        Após limpeza: std={std_val:.6f}")
        
        return features_clean, targets_clean

    # ------------------------------------------------------------
    def _make_all_horizon_windows_safe(self, X_data: np.ndarray, y_data: np.ndarray,
                                      feature_names: List[str], target_names: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Cria janelas temporais de forma segura, garantindo que os targets
        mantenham sua variabilidade original.
        """
        lookback = self.config.get("training")["main_lookback"]
        horizons = self.config.get("horizons")
        
        print(f"\n    Configuração de janelas:")
        print(f"      • Lookback: {lookback}")
        print(f"      • Horizons: {horizons}")
        print(f"      • X shape: {X_data.shape}")
        print(f"      • y shape: {y_data.shape}")
        
        # Verificar variabilidade dos targets ANTES de criar janelas
        print(f"\n    🔍 Variabilidade dos targets ANTES do windowing:")
        variability_issues = []
        for idx, name in enumerate(target_names):
            target_values = y_data[:, idx]
            std_val = np.std(target_values)
            range_val = np.ptp(target_values)
            
            print(f"      {name}:")
            print(f"        • std={std_val:.6f}")
            print(f"        • range={range_val:.4f}")
            print(f"        • min={np.min(target_values):.4f}, max={np.max(target_values):.4f}")
            
            # Thresholds específicos por variável
            if name == 'speed' and std_val < 10:
                print(f"        ❌ PROBLEMA CRÍTICO: speed está CONSTANTE (std={std_val:.6f})!")
                variability_issues.append((name, std_val))
            elif name == 'bz_gsm' and std_val < 0.5:
                print(f"        ⚠️  Bz com baixa variabilidade (std={std_val:.6f})")
            elif name == 'density' and std_val < 0.5:
                print(f"        ⚠️  Densidade com baixa variabilidade (std={std_val:.6f})")
            else:
                print(f"        ✅ OK")
        
        # Se houver problemas críticos, continuar mas com aviso
        if variability_issues:
            print(f"\n    ⚠️⚠️⚠️ ATENÇÃO: Variáveis com baixa variabilidade:")
            for name, std_val in variability_issues:
                print(f"        • {name}: std={std_val:.6f}")
            print(f"    ⚠️  Continuando, mas os resultados podem não ser confiáveis.")
        
        datasets: Dict[int, Dict[str, Any]] = {}
        
        for horizon in horizons:
            print(f"\n    🪟 Criando janelas para horizonte {horizon}h...")
            
            # Inicializar scalers Y para este horizonte
            self.scalers_y[horizon] = {}
            
            # Calcular número de janelas
            max_samples = len(X_data) - lookback - horizon
            max_samples = min(max_samples, 50000)  # Limite para GitHub
            
            print(f"      Máximo de amostras: {max_samples}")
            
            # Criar janelas
            X_windows, y_windows = [], []
            
            for i in range(max_samples):
                X_windows.append(X_data[i:i + lookback])
                y_windows.append(y_data[i + lookback + horizon])
            
            # Converter para arrays
            X_arr = np.array(X_windows, dtype=np.float32)
            y_arr = np.array(y_windows, dtype=np.float32)
            
            print(f"      Janelas criadas:")
            print(f"        • X shape: {X_arr.shape}")
            print(f"        • y shape: {y_arr.shape}")
            
            # Verificar variabilidade dos targets NAS JANELAS
            print(f"      📊 Estatísticas dos targets nas janelas:")
            for idx, name in enumerate(target_names):
                target_vals = y_arr[:, idx]
                std_val = np.std(target_vals)
                range_val = np.ptp(target_vals)
                print(f"        {name}: std={std_val:.6f}, range={range_val:.4f}")
                
                if name == 'speed' and std_val < 10:
                    print(f"        ⚠️⚠️⚠️ ALERTA CRÍTICO: {name} constante nas janelas!")
                elif std_val < 0.001:
                    print(f"        ⚠️  {name} quase constante nas janelas!")
            
            # -----------------------------------------------------------------
            # ESCALONAMENTO DOS TARGETS (Y) - POR VARIÁVEL
            # -----------------------------------------------------------------
            print(f"      🔧 Escalonando targets por variável...")
            y_scaled = np.zeros_like(y_arr, dtype=np.float32)
            
            for idx, name in enumerate(target_names):
                # Extrair esta variável
                y_single = y_arr[:, idx].reshape(-1, 1)
                
                # Verificar se não é constante
                if np.std(y_single) < 1e-10:
                    print(f"        ❌ {name}: Constante - usando StandardScaler(with_std=False)")
                    # Usar StandardScaler sem escala
                    scaler = StandardScaler(with_std=False)
                    y_scaled_single = scaler.fit_transform(y_single)
                    y_scaled[:, idx] = y_scaled_single.flatten()
                    self.scalers_y[horizon][name] = scaler
                else:
                    # Escalonar normalmente
                    scaler = StandardScaler()
                    y_scaled_single = scaler.fit_transform(y_single)
                    y_scaled[:, idx] = y_scaled_single.flatten()
                    self.scalers_y[horizon][name] = scaler
                    
                    print(f"        ✅ {name}: mean={scaler.mean_[0]:.4f}, scale={scaler.scale_[0]:.4f}")
            
            # -----------------------------------------------------------------
            # Montar dataset
            # -----------------------------------------------------------------
            datasets[horizon] = {
                "X": X_arr,                    # Features escalonadas
                "y_scaled": y_scaled,          # Targets escalonados (para treino)
                "y_raw": y_arr.copy(),         # Targets originais (para métricas)
                "target_names": target_names,
                "feature_names": feature_names,
                "horizon": horizon,
                "lookback": lookback
            }
            
            # Limpar memória
            del X_windows, y_windows
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
    print("🚀 HAC v6 FEATURE BUILDER - VERSÃO CORRIGIDA")
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
            
            # Converter listas para arrays de bytes
            target_names_bytes = np.array(data["target_names"], dtype=object)
            feature_names_bytes = np.array(data["feature_names"], dtype=object)
            
            np.savez_compressed(
                npz_path,
                X=data["X"],
                y_scaled=data["y_scaled"],
                y_raw=data["y_raw"],
                target_names=target_names_bytes,
                feature_names=feature_names_bytes,
                horizon=data["horizon"],
                lookback=data["lookback"]
            )
            
            print(f"   ✅ Horizonte {horizon}h: {npz_path}")
            print(f"      • Amostras: {data['X'].shape[0]}")
            print(f"      • Targets: {data['target_names']}")
        
        # Salvar scalers X
        scaler_x_path = os.path.join(out_dir, "scaler_X.pkl")
        joblib.dump(builder.scaler_X, scaler_x_path)
        print(f"   ✅ Scaler X: {scaler_x_path}")
        
        # Salvar scalers Y
        y_scalers_info = {}
        for h, scaler_dict in builder.scalers_y.items():
            y_scalers_info[str(h)] = {}
            for var_name, scaler in scaler_dict.items():
                y_scalers_info[str(h)][var_name] = {
                    "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [0.0],
                    "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [1.0],
                    "n_samples": int(scaler.n_samples_seen_) if hasattr(scaler, 'n_samples_seen_') else 0
                }
        
        scalers_y_path = os.path.join(out_dir, "y_scalers_info.json")
        with open(scalers_y_path, "w") as f:
            json.dump(y_scalers_info, f, indent=2)
        
        print(f"   ✅ Scalers Y: {scalers_y_path}")
        
        # 5. Resumo final
        print("\n" + "=" * 70)
        print("✅ FEATURE BUILD COMPLETADO!")
        print("=" * 70)
        
        # Verificação final
        print("\n🔍 RESUMO FINAL:")
        
        all_good = True
        for horizon in datasets.keys():
            data = datasets[horizon]
            print(f"\n📊 Dataset {horizon}h:")
            
            for idx, name in enumerate(data["target_names"]):
                raw_std = data["y_raw"][:, idx].std()
                scaled_std = data["y_scaled"][:, idx].std()
                
                print(f"   {name}:")
                print(f"     • Raw std: {raw_std:.4f}")
                print(f"     • Scaled std: {scaled_std:.4f}")
                
                # Verificar se é aceitável
                if name == 'speed' and raw_std < 10:
                    print(f"     ⚠️  VELOCIDADE COM BAIXA VARIABILIDADE (std={raw_std:.4f})")
                    all_good = False
                elif raw_std < 0.01:
                    print(f"     ⚠️  {name} com baixa variabilidade")
                    all_good = False
        
        if all_good:
            print("\n🎯 TODAS AS VARIÁVEIS OK!")
            print("   Pronto para treinamento!")
        else:
            print("\n⚠️  ALGUMAS VARIÁVEIS COM BAIXA VARIABILIDADE")
            print("   Considere verificar os dados originais.")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRO: {e}")
        print("   Execute primeiro: python prepare_omni_dataset.py")
        
    except Exception as e:
        print(f"\n❌ ERRO inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
