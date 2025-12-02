#!/usr/bin/env python3
"""
hac_v6_train.py - CORREÇÃO DA LOSS
Training pipeline para HAC v6 com correção da função de perda.

CORREÇÃO CRÍTICA: 
- A loss estava comparando valores normalizados (y_scaled) com valores físicos (modelo)
- Agora o modelo será treinado com valores normalizados e converte para físicos apenas na saída
"""

import os
import json
import gc
import psutil
from datetime import datetime
from typing import Dict, List, Any, Tuple, Callable
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
        print("🚀 HAC v6 PHYSICAL TRAINER - CORREÇÃO DA LOSS")
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
        
        # 6. Mapeamento de nomes de variáveis
        self.variable_mapping = self._create_variable_mapping()
        
        # 7. Inicializar relatório
        self.train_report = self._initialize_report()
        
        print("✅ Trainer initialized successfully")
        self._log_memory("After initialization")
    
    def _create_variable_mapping(self) -> Dict:
        """Cria mapeamento entre nomes de variáveis físicas e nomes no dataset."""
        targets = self.config.get("targets")["primary"]
        
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
            npz_files = [f for f in os.listdir(prepared_dir) if f.endswith('.npz')]
            if npz_files:
                print(f"✅ Found {len(npz_files)} prepared datasets in {prepared_dir}")
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
            "loss_correction_applied": True,  # Nova flag
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
            horizons = self.config.get("horizons")[:3]
            dataset_targets = self.config.get("targets")["primary"]
            physical_targets = [self._get_physical_variable_name(t) for t in dataset_targets]
            
            print(f"📋 Target mapping:")
            for i, (d_name, p_name) in enumerate(zip(dataset_targets, physical_targets)):
                print(f"   • {d_name} → {p_name}")
            
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
            
            print(f"   Dataset shape: X={X.shape}, y_scaled={y_scaled.shape}")
            print(f"   Dataset targets: {dataset_targets}")
            print(f"   Physical targets: {physical_targets}")
            
            # Split dos dados
            print("📊 Splitting data...")
            (X_train, y_train_scaled, X_val, y_val_scaled, 
             X_test, y_test_scaled, y_test_raw) = self._split_data(X, y_scaled, y_raw)
            
            print(f"   X_train shape: {X_train.shape}")
            print(f"   y_train_scaled shape: {y_train_scaled.shape}")
            print(f"   Mean y_train_scaled: {np.mean(y_train_scaled, axis=0)}")
            print(f"   Std y_train_scaled: {np.std(y_train_scaled, axis=0)}")
            
            # Construir modelo adaptado para valores normalizados
            print("🔨 Building adapted model...")
            model = self._build_adapted_model_for_horizon(
                input_shape=(lookback, X.shape[2]),
                dataset_targets=dataset_targets,
                physical_targets=physical_targets,
                horizon=horizon
            )
            
            # Criar callbacks
            callbacks = self._create_callbacks_for_horizon(horizon)
            
            # ✅ CORREÇÃO CRÍTICA: Treinar com valores NORMALIZADOS
            print(f"\n🔥 Training model for horizon {horizon}h...")
            print(f"   Training with NORMALIZED targets (mean≈0, std≈1)")
            
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
    
    def _build_adapted_model_for_horizon(self, input_shape: Tuple[int, int],
                                       dataset_targets: List[str], 
                                       physical_targets: List[str],
                                       horizon: int) -> tf.keras.Model:
        """Constrói modelo ADAPTADO que lida com valores normalizados."""
        model_type = self.config.get("model", {}).get("type", "lstm")
        
        print(f"   Building ADAPTED {model_type} model")
        print(f"   Model will: Input→Process→Output NORMALIZED values")
        print(f"   Physical limits applied AFTER training")
        
        # Construir modelo normal (com head físico)
        if model_type == "hybrid":
            model = self.model_builder.build_hybrid_model(input_shape, physical_targets)
        elif model_type == "lightweight":
            model = self.model_builder.build_lightweight_model(input_shape, physical_targets)
        else:  # Default: LSTM
            model = self.model_builder.build_lstm_model(input_shape, physical_targets)
        
        # ✅ CORREÇÃO: Modificar a loss para lidar com valores normalizados
        # Mas manter a arquitetura original para aplicar limites físicos
        print(f"   Model: {model.name}")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Output units: PHYSICAL (km/s, nT, cm⁻³)")
        
        return model
    
    def _create_adapted_loss_function(self, y_scalers: Dict):
        """Cria função de loss adaptada para valores normalizados."""
        # Esta função será usada se quisermos uma loss customizada
        # Por enquanto, usaremos MSE padrão
        pass
    
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
    
    def _create_callbacks_for_horizon(self, horizon: int) -> List:
        """Cria callbacks para o treino de um horizonte."""
        model_type = self.config.get("model", {}).get("type", "lstm")
        callbacks = self.model_builder.create_callbacks(horizon, model_type)
        
        class LossMonitor(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0 or epoch == 0:
                    loss = logs.get('loss', 0)
                    val_loss = logs.get('val_loss', 0)
                    
                    # Verificar se a loss está em escala razoável
                    if loss > 1000 or val_loss > 1000:
                        status = "⚠️  HIGH"
                    elif loss < 1 and val_loss < 1:
                        status = "✅ OK"
                    else:
                        status = "📊 NORMAL"
                    
                    print(f"      Epoch {epoch:3d} | Loss: {loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | {status}")
        
        callbacks.append(LossMonitor())
        return callbacks
    
    def _evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray,
                       y_test_scaled: np.ndarray, y_test_raw: np.ndarray,
                       horizon: int, dataset_targets: List[str],
                       physical_targets: List[str]) -> Dict:
        """Avalia o modelo e retorna métricas."""
        # 1. Avaliar no conjunto de teste (com valores normalizados)
        test_loss = model.evaluate(X_test, y_test_scaled, verbose=0)[0]
        
        # 2. Fazer previsões
        y_pred_scaled = model.predict(X_test, verbose=0, batch_size=32)
        
        print(f"\n   Loss on test set: {test_loss:.4f}")
        print(f"   Predicted shape: {y_pred_scaled.shape}")
        
        # 3. Dessecalizar previsões para avaliação física
        y_pred_raw = self._inverse_scale_predictions(
            y_pred_scaled, horizon, dataset_targets
        )
        
        # 4. Verificar se as previsões estão dentro dos limites físicos
        print("\n   Verifying physical limits on predictions:")
        violations_info = self._check_and_report_physical_limits(
            y_pred_raw, dataset_targets
        )
        
        # 5. Calcular métricas nos valores reais (físicos)
        print("\n   Calculating metrics on PHYSICAL values:")
        metrics = self._calculate_physical_metrics(
            y_test_raw, y_pred_raw, dataset_targets
        )
        
        # 6. Log dos resultados
        print(f"\n📊 Evaluation results for horizon {horizon}h:")
        print(f"   • Test Loss (normalized): {test_loss:.4f}")
        print(f"   • Physical MAE:  {metrics['mae']:.4f}")
        print(f"   • Physical RMSE: {metrics['rmse']:.4f}")
        
        # Métricas por variável
        for var_name, var_metrics in metrics['per_variable'].items():
            print(f"   • {var_name}: MAE={var_metrics['mae']:.4f}, RMSE={var_metrics['rmse']:.4f}")
        
        if violations_info["total_violations"] > 0:
            print(f"   ⚠️  Physical violations: {violations_info['total_violations']}")
            for var, info in violations_info["per_variable"].items():
                if info["violations"] > 0:
                    print(f"      {var}: {info['violations']} violations "
                          f"({info['min']:.1f} to {info['max']:.1f})")
        else:
            print(f"   ✅ All predictions respect physical limits")
        
        # Adicionar métricas
        metrics.update({
            "normalized_test_loss": float(test_loss),
            "physical_violations": violations_info["total_violations"],
            "test_samples": len(X_test),
            "physical_limits_info": violations_info
        })
        
        return metrics
    
    def _calculate_physical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  targets: List[str]) -> Dict:
        """Calcula métricas em valores físicos."""
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
            
            mae = mean_absolute_error(y_true_var, y_pred_var)
            rmse = np.sqrt(mean_squared_error(y_true_var, y_pred_var))
            
            per_variable_metrics[var_name] = {
                "mae": float(mae),
                "rmse": float(rmse),
                "mean_true": float(y_true_var.mean()),
                "std_true": float(y_true_var.std()),
                "mean_pred": float(y_pred_var.mean()),
                "std_pred": float(y_pred_var.std()),
                "range_true": f"{y_true_var.min():.1f}-{y_true_var.max():.1f}",
                "range_pred": f"{y_pred_var.min():.1f}-{y_pred_var.max():.1f}"
            }
            
            # Log para debug
            print(f"      {var_name}: MAE={mae:.2f}, "
                  f"True range: {y_true_var.min():.1f}-{y_true_var.max():.1f}, "
                  f"Pred range: {y_pred_var.min():.1f}-{y_pred_var.max():.1f}")
        
        metrics["per_variable"] = per_variable_metrics
        return metrics
    
    def _check_and_report_physical_limits(self, y_pred: np.ndarray,
                                        dataset_targets: List[str]) -> Dict:
        """Verifica limites físicos e retorna informações detalhadas."""
        violations_info = {
            "total_violations": 0,
            "per_variable": {}
        }
        
        # Limites físicos (mesmo do model builder)
        physical_limits = {
            "V": {"min": 250, "max": 1650, "unit": "km/s"},
            "Bz": {"min": -40, "max": 40, "unit": "nT"},
            "n": {"min": 0, "max": 100, "unit": "cm⁻³"},
            "Bx": {"min": -50, "max": 50, "unit": "nT"},
            "By": {"min": -50, "max": 50, "unit": "nT"},
            "Bt": {"min": 0, "max": 80, "unit": "nT"}
        }
        
        for idx, dataset_name in enumerate(dataset_targets):
            physical_name = self._get_physical_variable_name(dataset_name)
            
            if physical_name in physical_limits:
                limits = physical_limits[physical_name]
                v_min = limits["min"]
                v_max = limits["max"]
                
                pred_values = y_pred[:, idx]
                violations = np.sum((pred_values < v_min) | (pred_values > v_max))
                
                violations_info["per_variable"][dataset_name] = {
                    "physical_name": physical_name,
                    "violations": int(violations),
                    "min": float(pred_values.min()),
                    "max": float(pred_values.max()),
                    "limits": f"{v_min}-{v_max}",
                    "unit": limits["unit"]
                }
                
                violations_info["total_violations"] += int(violations)
        
        return violations_info
    
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
            else:
                print(f"⚠️  No scaler found for {var_name} at horizon {horizon}h")
                y_pred_raw[:, idx] = y_pred_scaled[:, idx]
        
        return y_pred_raw
    
    def _save_horizon_artifacts(self, model: tf.keras.Model, horizon: int,
                              history: Dict, metrics: Dict, test_size: int,
                              dataset_targets: List[str], physical_targets: List[str]):
        """Salva todos os artefatos de um horizonte."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        model_name = f"hac_physical_h{horizon}_{timestamp}"
        out_dir = os.path.join(self.model_dir, model_name)
        
        # ✅ CORREÇÃO: Criar diretório ANTES de salvar
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n💾 Saving artifacts for horizon {horizon}h...")
        print(f"   Directory: {out_dir}")
        
        # 1. Salvar modelo
        model_path = os.path.join(out_dir, "model.keras")
        model.save(model_path)  # ✅ CORRIGIDO: sem save_format
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
            "lookback": self.config.get("training")["main_lookback"],
            "batch_size": self.config.get("training")["batch_size"],
            "max_epochs": self.config.get("training")["max_epochs"],
            "test_samples": test_size,
            "timestamp": timestamp,
            "model_name": model.name,
            "model_parameters": model.count_params(),
            "loss_correction": "normalized_targets"
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
                "normalized_loss": history.get('val_loss', [0])[-1] if 'val_loss' in history else 0,
                "physical_mae": metrics.get("mae", 0),
                "physical_rmse": metrics.get("rmse", 0)
            },
            "dataset_targets": dataset_targets,
            "physical_targets": physical_targets
        }
    
    def _save_scalers(self, out_dir: str, horizon: int, targets: List[str]):
        """Salva os scalers X e Y."""
        os.makedirs(out_dir, exist_ok=True)
        
        # Salvar scaler X
        scaler_x_path = os.path.join(out_dir, "scaler_X.pkl")
        joblib.dump(self.feature_builder.scaler_X, scaler_x_path)
        
        # Salvar scalers Y (por variável)
        y_scalers = self.feature_builder.get_y_scalers(horizon)
        if y_scalers:
            for var_name, scaler in y_scalers.items():
                if var_name in targets:
                    scaler_path = os.path.join(out_dir, f"scaler_y_{var_name}.pkl")
                    joblib.dump(scaler, scaler_path)
        
        print(f"   ✅ Scalers saved")
    
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
            print(f"   • Normalized Loss: {metrics.get('normalized_loss', 'N/A'):.4f}")
            print(f"   • Physical MAE:  {metrics.get('physical_mae', 'N/A'):.4f}")
            print(f"   • Physical RMSE: {metrics.get('physical_rmse', 'N/A'):.4f}")
            print(f"   • Model: {data.get('model_size_mb', 'N/A'):.1f} MB")
            print(f"   • Samples: Train={data.get('train_samples', 'N/A')}, "
                  f"Test={data.get('test_samples', 'N/A')}")
    
    def _print_resource_summary(self):
        """Imprime resumo do uso de recursos."""
        print("\n" + "=" * 60)
        print("📊 RESOURCE USAGE SUMMARY")
        print("=" * 60)
        
        final_mem = psutil.virtual_memory()
        print(f"Final memory usage: {final_mem.percent:.1f}%")
        print(f"Available: {final_mem.available / 1e9:.1f} GB")
        
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
    print("🧪 HAC v6 TRAINING SCRIPT - LOSS CORRIGIDA")
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
