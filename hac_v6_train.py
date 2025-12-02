#!/usr/bin/env python3
"""
hac_v6_train.py - CORRIGIDO E ADAPTADO
Training pipeline para HAC v6 com nomes de variáveis corretos.

CORREÇÕES APLICADAS:
1. Criar diretórios antes de salvar modelos (FileNotFoundError)
2. Remover save_format depreciado do model.save()
3. Mapear nomes de variáveis para limites físicos corretos
"""

import os
import json
import gc
import psutil
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from hac_v6_models import create_model_builder


class HACTrainer:
    """Pipeline de treino completo para HAC v6."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Inicializa o trainer com configuração e componentes."""
        print("=" * 60)
        print("🚀 HAC v6 PHYSICAL TRAINER - COM NOMES CORRETOS")
        print("=" * 60)
        
        # 1. Carregar configuração
        print("📡 Loading configuration...")
        self.config = HACConfig(config_path)
        
        # 2. Configurar ambiente para GitHub Free
        self._setup_github_free_environment()
        
        # 3. Verificar se os dados já foram processados
        self._check_prepared_data()
        
        # 4. Inicializar componentes (sem rebuild se já existir)
        print("🔧 Initializing feature builder...")
        self.feature_builder = HACFeatureBuilder(self.config)
        
        print("🧠 Initializing model builder...")
        self.model_builder = create_model_builder(self.config)
        
        # 5. Configurar diretórios
        self._setup_directories()
        
        # 6. Mapeamento de nomes de variáveis (speed → V, bz_gsm → Bz, density → n)
        self.variable_mapping = self._create_variable_mapping()
        
        # 7. Inicializar relatório
        self.train_report = self._initialize_report()
        
        print("✅ Trainer initialized successfully")
        self._log_memory("After initialization")
    
    def _create_variable_mapping(self) -> Dict:
        """Cria mapeamento entre nomes de variáveis físicas e nomes no dataset."""
        targets = self.config.get("targets")["primary"]
        
        # Mapeamento padrão: nomes do dataset -> nomes físicos para limites
        mapping = {}
        for target in targets:
            if target == "speed":
                mapping[target] = "V"
            elif target == "bz_gsm":
                mapping[target] = "Bz"
            elif target == "density":
                mapping[target] = "n"
            elif target == "bx_gsm":
                mapping[target] = "Bx"
            elif target == "by_gsm":
                mapping[target] = "By"
            elif target == "bt":
                mapping[target] = "Bt"
            else:
                # Se não mapeado, usar o próprio nome
                mapping[target] = target
        
        print(f"📋 Variable mapping: {mapping}")
        return mapping
    
    def _get_physical_variable_name(self, dataset_name: str) -> str:
        """Converte nome do dataset para nome físico para limites."""
        return self.variable_mapping.get(dataset_name, dataset_name)
    
    def _setup_github_free_environment(self):
        """Configura otimizações específicas para GitHub Free."""
        tf.config.set_visible_devices([], 'GPU')
        tf.keras.backend.set_floatx('float32')
        
        # Limitar threads
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        print("⚙️  Configured for GitHub Free environment")
    
    def _check_prepared_data(self):
        """Verifica se os dados já foram processados."""
        data_dir = self.config.get("paths")["data_dir"]
        prepared_dir = os.path.join(data_dir, "prepared")
        
        if os.path.exists(prepared_dir):
            # Verificar se há datasets salvos
            npz_files = [f for f in os.listdir(prepared_dir) if f.endswith('.npz')]
            if npz_files:
                print(f"✅ Found {len(npz_files)} prepared datasets in {prepared_dir}")
                print("   Using pre-processed data (fast mode)")
            else:
                print(f"⚠️  Prepared directory exists but no .npz files found")
        else:
            print(f"ℹ️  No prepared data found. Will process from scratch.")
    
    def _setup_directories(self):
        """Cria diretórios necessários para o treino."""
        paths = self.config.get("paths")
        self.model_dir = paths["model_dir"]
        self.results_dir = paths["results_dir"]
        
        # Criar diretórios
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"📁 Model dir: {self.model_dir}")
        print(f"📁 Results dir: {self.results_dir}")
    
    def _initialize_report(self) -> Dict:
        """Inicializa o relatório de treino."""
        targets = self.config.get("targets")["primary"]
        physical_targets = [self._get_physical_variable_name(t) for t in targets]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "horizons": self.config.get("horizons"),
                "dataset_targets": self.config.get("targets")["primary"],
                "physical_targets": physical_targets,
                "lookback": self.config.get("training")["main_lookback"],
                "batch_size": self.config.get("training")["batch_size"],
                "max_epochs": self.config.get("training")["max_epochs"]
            },
            "physical_constraints": True,
            "github_free_optimized": True,
            "variable_mapping": self.variable_mapping,
            "horizons": {},
            "resource_usage": {}
        }
    
    def _log_memory(self, stage: str):
        """Registra uso de memória para monitoramento."""
        mem = psutil.virtual_memory()
        self.train_report["resource_usage"][stage] = {
            "percent": float(mem.percent),
            "available_gb": float(mem.available / 1e9)
        }
        
        if mem.percent > 70 or "horizon" in stage:
            print(f"   🧠 {stage} - Mem: {mem.percent:.1f}% | "
                  f"Disponível: {mem.available / 1e9:.1f} GB")
    
    def run(self):
        """Executa o pipeline completo de treino."""
        print("\n" + "=" * 60)
        print("🔥 STARTING TRAINING PIPELINE")
        print("=" * 60)
        
        try:
            # 1. Construir ou carregar datasets
            print("\n📊 STEP 1: Loading datasets...")
            datasets = self.feature_builder.build_all()
            
            if not datasets:
                raise ValueError("❌ No datasets generated!")
            
            print(f"✅ Loaded {len(datasets)} horizons")
            
            # 2. Configurações de treino
            training_config = self.config.get("training")
            lookback = training_config["main_lookback"]
            horizons = self.config.get("horizons")[:3]  # Limitar a 3 para GitHub Free
            dataset_targets = self.config.get("targets")["primary"]  # Nomes no dataset
            physical_targets = [self._get_physical_variable_name(t) for t in dataset_targets]
            
            print(f"📋 Target mapping:")
            for i, (d_name, p_name) in enumerate(zip(dataset_targets, physical_targets)):
                print(f"   • {d_name} → {p_name} (para limites físicos)")
            
            # Ajustes para GitHub Free
            batch_size = min(training_config["batch_size"], 32)
            max_epochs = min(training_config["max_epochs"], 50)
            
            print(f"\n⚙️  Training configuration:")
            print(f"   • Horizons: {horizons}")
            print(f"   • Dataset targets: {dataset_targets}")
            print(f"   • Physical targets: {physical_targets}")
            print(f"   • Lookback: {lookback}")
            print(f"   • Batch size: {batch_size}")
            print(f"   • Max epochs: {max_epochs}")
            
            # 3. Treinar para cada horizonte
            trained_horizons = 0
            for horizon in horizons:
                if horizon in datasets:
                    print(f"\n{'='*50}")
                    print(f"🎯 TRAINING HORIZON {horizon}h")
                    print(f"{'='*50}")
                    
                    success = self._train_single_horizon(
                        horizon=horizon,
                        dataset=datasets[horizon],
                        lookback=lookback,
                        batch_size=batch_size,
                        max_epochs=max_epochs,
                        dataset_targets=dataset_targets,
                        physical_targets=physical_targets
                    )
                    
                    if success:
                        trained_horizons += 1
                    
                    # Limpar memória entre horizontes
                    self._cleanup_between_horizons()
                else:
                    print(f"⚠️  No data for horizon {horizon}h")
            
            # 4. Salvar relatório final
            if trained_horizons > 0:
                self._save_final_report()
                print(f"\n🎉 TRAINING COMPLETED! {trained_horizons} horizons trained")
                
                # Imprimir resumo
                self._print_training_summary()
            else:
                print("\n❌ No horizons were successfully trained")
            
            # 5. Resumo de recursos
            self._print_resource_summary()
            
        except MemoryError as e:
            print(f"\n💥 OUT OF MEMORY ERROR: {str(e)}")
            print("\n🔧 Recommendations for GitHub Free:")
            print("   1. Reduce horizons in config.yaml")
            print("   2. Reduce batch_size to 8")
            print("   3. Reduce lookback to 24")
            raise
        
        except Exception as e:
            print(f"\n💥 TRAINING FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _train_single_horizon(self, horizon: int, dataset: Dict,
                            lookback: int, batch_size: int,
                            max_epochs: int, dataset_targets: List[str],
                            physical_targets: List[str]) -> bool:
        """Treina modelo para um único horizonte."""
        try:
            print(f"\n📦 Preparing data for horizon {horizon}h...")
            X = dataset["X"]
            y_scaled = dataset["y_scaled"]
            y_raw = dataset["y_raw"]
            
            print(f"   Dataset shape: X={X.shape}, y={y_scaled.shape}")
            print(f"   Dataset targets: {dataset_targets}")
            print(f"   Physical targets: {physical_targets}")
            
            # Split dos dados
            print("📊 Splitting data...")
            X_train, y_train_scaled, X_val, y_val_scaled, X_test, y_test_scaled, y_test_raw = \
                self._split_data(X, y_scaled, y_raw)
            
            # Construir modelo com nomes físicos para limites
            print("🔨 Building physical model...")
            model = self._build_model_for_horizon(
                input_shape=(lookback, X.shape[2]),
                dataset_targets=dataset_targets,
                physical_targets=physical_targets,
                horizon=horizon
            )
            
            # Criar callbacks
            callbacks = self._create_callbacks_for_horizon(horizon)
            
            # Treinar modelo - CORREÇÃO: sem workers/use_multiprocessing
            print(f"\n🔥 Training model for horizon {horizon}h...")
            history = model.fit(
                X_train, y_train_scaled,
                validation_data=(X_val, y_val_scaled),
                epochs=max_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Avaliar modelo
            print("\n📈 Evaluating model...")
            metrics = self._evaluate_model(
                model=model,
                X_test=X_test,
                y_test_scaled=y_test_scaled,
                y_test_raw=y_test_raw,
                horizon=horizon,
                dataset_targets=dataset_targets,
                physical_targets=physical_targets
            )
            
            # Salvar artefatos
            self._save_horizon_artifacts(
                model=model,
                horizon=horizon,
                history=history.history,
                metrics=metrics,
                test_size=len(X_test),
                dataset_targets=dataset_targets,
                physical_targets=physical_targets
            )
            
            # Atualizar relatório
            self._update_horizon_report(horizon, metrics, len(X_train))
            
            return True
            
        except Exception as e:
            print(f"\n❌ Failed to train horizon {horizon}h: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _split_data(self, X: np.ndarray, y_scaled: np.ndarray,
                   y_raw: np.ndarray) -> Tuple:
        """Divide os dados em treino, validação e teste."""
        n_total = len(X)
        
        val_split = min(self.config.get("training")["val_split"], 0.2)
        test_split = min(self.config.get("training")["test_split"], 0.2)
        
        n_val = int(n_total * val_split)
        n_test = int(n_total * test_split)
        n_train = n_total - n_val - n_test
        
        print(f"   Split: Train={n_train}, Val={n_val}, Test={n_test}")
        
        # Split dos dados
        X_train = X[:n_train]
        y_train_scaled = y_scaled[:n_train]
        
        X_val = X[n_train:n_train + n_val]
        y_val_scaled = y_scaled[n_train:n_train + n_val]
        
        X_test = X[n_train + n_val:n_train + n_val + n_test]
        y_test_scaled = y_scaled[n_train + n_val:n_train + n_val + n_test]
        y_test_raw = y_raw[n_train + n_val:n_train + n_val + n_test]
        
        return X_train, y_train_scaled, X_val, y_val_scaled, X_test, y_test_scaled, y_test_raw
    
    def _build_model_for_horizon(self, input_shape: Tuple[int, int],
                               dataset_targets: List[str], 
                               physical_targets: List[str],
                               horizon: int) -> tf.keras.Model:
        """Constrói modelo físico para um horizonte específico."""
        model_type = self.config.get("model", {}).get("type", "lstm")
        
        print(f"   Building {model_type} model")
        print(f"   Dataset targets: {dataset_targets}")
        print(f"   Physical targets (para limites): {physical_targets}")
        
        # O modelo deve ser construído com os nomes físicos para que os limites funcionem
        if model_type == "hybrid":
            model = self.model_builder.build_hybrid_model(input_shape, physical_targets)
        elif model_type == "lightweight":
            model = self.model_builder.build_lightweight_model(input_shape, physical_targets)
        else:  # Default: LSTM
            model = self.model_builder.build_lstm_model(input_shape, physical_targets)
        
        # Resumo do modelo
        print(f"   Model: {model.name}")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Output shape: {model.output_shape}")
        
        return model
    
    def _create_callbacks_for_horizon(self, horizon: int) -> List:
        """Cria callbacks para o treino de um horizonte."""
        model_type = self.config.get("model", {}).get("type", "lstm")
        callbacks = self.model_builder.create_callbacks(horizon, model_type)
        
        # Adicionar callback para monitoramento
        class ProgressMonitor(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0 or epoch == 0:
                    mem = psutil.virtual_memory()
                    loss = logs.get('loss', 0)
                    val_loss = logs.get('val_loss', 0)
                    print(f"      Epoch {epoch:3d} | Loss: {loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Mem: {mem.percent:.1f}%")
        
        callbacks.append(ProgressMonitor())
        return callbacks
    
    def _evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray,
                       y_test_scaled: np.ndarray, y_test_raw: np.ndarray,
                       horizon: int, dataset_targets: List[str],
                       physical_targets: List[str]) -> Dict:
        """Avalia o modelo e retorna métricas."""
        # 1. Avaliar no conjunto de teste
        test_loss = model.evaluate(X_test, y_test_scaled, verbose=0)[0]
        
        # 2. Fazer previsões
        y_pred_scaled = model.predict(X_test, verbose=0, batch_size=32)
        
        # 3. Dessecalizar previsões
        y_pred_raw = self._inverse_scale_predictions(
            y_pred_scaled, horizon, dataset_targets
        )
        
        # 4. Calcular métricas nos valores reais
        metrics = self._calculate_metrics(y_test_raw, y_pred_raw, dataset_targets)
        
        # 5. Verificar limites físicos (usar nomes físicos para verificação)
        violations = self._check_physical_violations(y_pred_raw, dataset_targets)
        
        # 6. Log dos resultados
        print(f"\n📊 Evaluation results for horizon {horizon}h:")
        print(f"   • Test Loss: {test_loss:.4f}")
        print(f"   • MAE:  {metrics['mae']:.4f}")
        print(f"   • RMSE: {metrics['rmse']:.4f}")
        
        # Métricas por variável
        for var_name, var_metrics in metrics['per_variable'].items():
            print(f"   • {var_name}: MAE={var_metrics['mae']:.4f}, RMSE={var_metrics['rmse']:.4f}")
        
        if violations > 0:
            print(f"   ⚠️  Physical violations: {violations}")
        else:
            print(f"   ✅ All predictions respect physical limits")
        
        # Adicionar métricas
        metrics.update({
            "test_loss": float(test_loss),
            "physical_violations": violations,
            "test_samples": len(X_test)
        })
        
        return metrics
    
    def _inverse_scale_predictions(self, y_pred_scaled: np.ndarray,
                                 horizon: int, targets: List[str]) -> np.ndarray:
        """Dessecalona previsões usando os scalers Y por variável."""
        y_pred_raw = np.zeros_like(y_pred_scaled)
        
        # Obter scalers Y para este horizonte
        y_scalers = self.feature_builder.get_y_scalers(horizon)
        
        if not y_scalers:
            print(f"⚠️  No Y scalers found for horizon {horizon}h")
            return y_pred_scaled
        
        # Dessecalonar cada variável separadamente
        for idx, var_name in enumerate(targets):
            if var_name in y_scalers:
                scaler = y_scalers[var_name]
                y_single = y_pred_scaled[:, idx].reshape(-1, 1)
                y_descaled = scaler.inverse_transform(y_single)
                y_pred_raw[:, idx] = y_descaled.flatten()
                # Debug: mostrar transformação
                if idx == 0:
                    print(f"   Descaled sample: {y_pred_scaled[0, idx]:.3f} → {y_pred_raw[0, idx]:.1f}")
            else:
                print(f"⚠️  No scaler found for {var_name} at horizon {horizon}h")
                y_pred_raw[:, idx] = y_pred_scaled[:, idx]
        
        return y_pred_raw
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          targets: List[str]) -> Dict:
        """Calcula métricas de avaliação."""
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))
        }
        
        # Métricas por variável
        per_variable_metrics = {}
        for idx, var_name in enumerate(targets):
            y_true_var = y_true[:, idx]
            y_pred_var = y_pred[:, idx]
            
            per_variable_metrics[var_name] = {
                "mae": float(mean_absolute_error(y_true_var, y_pred_var)),
                "rmse": float(np.sqrt(mean_squared_error(y_true_var, y_pred_var))),
                "mean_true": float(y_true_var.mean()),
                "std_true": float(y_true_var.std()),
                "mean_pred": float(y_pred_var.mean()),
                "std_pred": float(y_pred_var.std())
            }
        
        metrics["per_variable"] = per_variable_metrics
        return metrics
    
    def _check_physical_violations(self, y_pred: np.ndarray,
                                 dataset_targets: List[str]) -> int:
        """Conta violações de limites físicos nas previsões."""
        violations = 0
        
        # Limites físicos baseados no model builder
        # Usamos os nomes físicos mapeados para verificação
        physical_limits = {
            "V": (250, 1650),       # km/s
            "Bz": (-40, 40),        # nT
            "n": (0, 100),          # cm⁻³
            "Bx": (-50, 50),        # nT
            "By": (-50, 50),        # nT
            "Bt": (0, 80),          # nT
            # Aliases para nomes do dataset
            "speed": (250, 1650),
            "bz_gsm": (-40, 40),
            "density": (0, 100),
            "bx_gsm": (-50, 50),
            "by_gsm": (-50, 50),
            "bt": (0, 80)
        }
        
        for idx, dataset_name in enumerate(dataset_targets):
            # Obter o nome físico para verificação de limites
            physical_name = self._get_physical_variable_name(dataset_name)
            
            if physical_name in physical_limits:
                v_min, v_max = physical_limits[physical_name]
                var_violations = np.sum((y_pred[:, idx] < v_min) | (y_pred[:, idx] > v_max))
                violations += int(var_violations)
                
                if var_violations > 0:
                    print(f"      ⚠️  {dataset_name} ({physical_name}): {int(var_violations)} violations "
                          f"({y_pred[:, idx].min():.1f} to {y_pred[:, idx].max():.1f})")
        
        return violations
    
    def _save_horizon_artifacts(self, model: tf.keras.Model, horizon: int,
                              history: Dict, metrics: Dict, test_size: int,
                              dataset_targets: List[str], physical_targets: List[str]):
        """Salva todos os artefatos de um horizonte."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        model_name = f"hac_physical_h{horizon}_{timestamp}"
        out_dir = os.path.join(self.model_dir, model_name)
        
        # ✅ CORREÇÃO CRÍTICA: Criar diretório ANTES de salvar qualquer coisa
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n💾 Saving artifacts for horizon {horizon}h...")
        print(f"   Directory: {out_dir} (criado)")
        
        # 1. Salvar modelo - ✅ CORREÇÃO: sem save_format depreciado
        model_path = os.path.join(out_dir, "model.keras")
        model.save(model_path)  # ✅ CORRIGIDO: Removido save_format="keras"
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   ✅ Model saved ({model_size_mb:.1f} MB)")
        
        # 2. Salvar scalers
        self._save_scalers(out_dir, horizon, dataset_targets)
        
        # 3. Salvar história do treino
        history_path = os.path.join(out_dir, "training_history.json")
        with open(history_path, "w") as f:
            clean_history = {}
            for key, values in history.items():
                clean_history[key] = [float(v) for v in values]
            json.dump(clean_history, f, indent=2)
        
        # 4. Salvar métricas
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # 5. Salvar configuração
        config_summary = {
            "horizon": horizon,
            "dataset_targets": dataset_targets,
            "physical_targets": physical_targets,
            "variable_mapping": {d: p for d, p in zip(dataset_targets, physical_targets)},
            "lookback": self.config.get("training")["main_lookback"],
            "batch_size": self.config.get("training")["batch_size"],
            "max_epochs": self.config.get("training")["max_epochs"],
            "test_samples": test_size,
            "timestamp": timestamp,
            "model_name": model.name,
            "model_parameters": model.count_params()
        }
        
        config_path = os.path.join(out_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config_summary, f, indent=2)
        
        print(f"   ✅ All artifacts saved to {out_dir}")
        
        # Atualizar relatório
        self.train_report["horizons"][str(horizon)] = {
            "model_dir": out_dir,
            "model_size_mb": model_size_mb,
            "test_samples": test_size,
            "metrics": {
                "mae": metrics.get("mae", 0),
                "rmse": metrics.get("rmse", 0)
            },
            "dataset_targets": dataset_targets,
            "physical_targets": physical_targets
        }
    
    def _save_scalers(self, out_dir: str, horizon: int, targets: List[str]):
        """Salva os scalers X e Y."""
        # ✅ CORREÇÃO: Garantir que o diretório existe
        os.makedirs(out_dir, exist_ok=True)
        
        # Salvar scaler X
        scaler_x_path = os.path.join(out_dir, "scaler_X.pkl")
        joblib.dump(self.feature_builder.scaler_X, scaler_x_path)
        print(f"   ✅ Scaler X saved")
        
        # Salvar scalers Y (por variável)
        y_scalers = self.feature_builder.get_y_scalers(horizon)
        if y_scalers:
            scaler_count = 0
            for var_name, scaler in y_scalers.items():
                if var_name in targets:  # Salvar apenas os targets usados
                    scaler_path = os.path.join(out_dir, f"scaler_y_{var_name}.pkl")
                    joblib.dump(scaler, scaler_path)
                    scaler_count += 1
            
            print(f"   ✅ {scaler_count} Y scalers saved for {targets}")
    
    def _update_horizon_report(self, horizon: int, metrics: Dict, train_size: int):
        """Atualiza o relatório com resultados do horizonte."""
        if str(horizon) not in self.train_report["horizons"]:
            self.train_report["horizons"][str(horizon)] = {}
        
        self.train_report["horizons"][str(horizon)].update({
            "train_samples": train_size,
            "evaluation_metrics": metrics,
            "completed_at": datetime.utcnow().isoformat()
        })
    
    def _cleanup_between_horizons(self):
        """Limpa memória entre o treino de diferentes horizontes."""
        gc.collect()
        tf.keras.backend.clear_session()
        self._log_memory("After horizon cleanup")
    
    def _save_final_report(self):
        """Salva o relatório final de treino."""
        # Adicionar uso final de recursos
        final_mem = psutil.virtual_memory()
        self.train_report["final_resource_usage"] = {
            "memory_percent": float(final_mem.percent),
            "memory_available_gb": float(final_mem.available / 1e9),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Adicionar resumo
        self.train_report["summary"] = {
            "total_horizons_trained": len(self.train_report["horizons"]),
            "successful_horizons": list(self.train_report["horizons"].keys()),
            "total_training_time": datetime.utcnow().isoformat()
        }
        
        # ✅ CORREÇÃO: Garantir que o diretório de resultados existe
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Salvar relatório
        report_path = os.path.join(self.results_dir, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(self.train_report, f, indent=2)
        
        print(f"\n📘 Training report saved: {report_path}")
    
    def _print_training_summary(self):
        """Imprime resumo do treino."""
        print("\n" + "=" * 60)
        print("📈 TRAINING SUMMARY")
        print("=" * 60)
        
        for horizon, data in self.train_report["horizons"].items():
            metrics = data.get("metrics", {})
            print(f"\nHorizon {horizon}h:")
            print(f"   • MAE:  {metrics.get('mae', 'N/A'):.4f}")
            print(f"   • RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"   • Model: {data.get('model_size_mb', 'N/A'):.1f} MB")
            print(f"   • Samples: Train={data.get('train_samples', 'N/A')}, "
                  f"Test={data.get('test_samples', 'N/A')}")
            print(f"   • Dataset targets: {data.get('dataset_targets', [])}")
            print(f"   • Physical targets: {data.get('physical_targets', [])}")
    
    def _print_resource_summary(self):
        """Imprime resumo do uso de recursos."""
        print("\n" + "=" * 60)
        print("📊 RESOURCE USAGE SUMMARY")
        print("=" * 60)
        
        final_mem = psutil.virtual_memory()
        print(f"Final memory usage: {final_mem.percent:.1f}%")
        print(f"Available: {final_mem.available / 1e9:.1f} GB")
        
        # Verificar uso de memória
        if final_mem.percent > 85:
            print(f"⚠️  High memory usage ({final_mem.percent:.1f}%)")
        else:
            print(f"✅ Memory management: OK")
        
        print("=" * 60)


# ------------------------------------------------------------
# Execução principal
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 HAC v6 TRAINING SCRIPT - TODAS CORREÇÕES APLICADAS")
    print("=" * 60)
    
    try:
        # Verificar se os dados preparados existem
        if not os.path.exists("data_real/prepared/"):
            print("⚠️  No prepared data found. Run hac_v6_features.py first!")
            response = input("Run feature builder first? (y/n): ")
            if response.lower() == 'y':
                print("\nRunning feature builder...")
                import subprocess
                result = subprocess.run(["python", "hac_v6_features.py"], 
                                      capture_output=True, text=True)
                print(result.stdout)
                if result.returncode != 0:
                    print(f"❌ Feature builder failed: {result.stderr}")
                    exit(1)
        
        # Criar e executar trainer
        trainer = HACTrainer()
        trainer.run()
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n✨ Next steps:")
        print("   1. Check models/hac_v6/ directory for trained models")
        print("   2. Check results/ directory for training reports")
        print("   3. Test models with inference script")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
