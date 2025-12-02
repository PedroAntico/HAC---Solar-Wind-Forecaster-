#!/usr/bin/env python3
"""
hac_v6_train.py - CORRIGIDO
Training pipeline para HAC v6 Solar Wind Forecaster.

INTEGRAÇÃO COMPLETA:
1. HACFeatureBuilder (com escalonamento Y separado por variável)
2. PhysicalModelBuilder (com head físico e limites)
3. Pipeline de treino robusto para GitHub Free
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
    """Pipeline de treino completo para HAC v6 com física correta."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Inicializa o trainer com configuração e componentes."""
        print("=" * 60)
        print("🚀 HAC v6 PHYSICAL TRAINER - GitHub Free Optimized")
        print("=" * 60)
        
        # 1. Carregar configuração
        print("📡 Loading configuration...")
        self.config = HACConfig(config_path)
        
        # 2. Configurar ambiente para GitHub Free
        self._setup_github_free_environment()
        
        # 3. Inicializar componentes
        print("🔧 Initializing feature builder...")
        self.feature_builder = HACFeatureBuilder(self.config)
        
        print("🧠 Initializing model builder...")
        self.model_builder = create_model_builder(self.config.get_all())
        
        # 4. Configurar diretórios
        self._setup_directories()
        
        # 5. Inicializar relatório
        self.train_report = self._initialize_report()
        
        print("✅ Trainer initialized successfully")
        self._log_memory("After initialization")
    
    def _setup_github_free_environment(self):
        """Configura otimizações específicas para GitHub Free."""
        # Desativar GPU completamente
        tf.config.set_visible_devices([], 'GPU')
        
        # Configurar threads do TensorFlow
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        # Usar float32 para economia de memória
        tf.keras.backend.set_floatx('float32')
        
        # Configurar alocação de memória do TF
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Limitar uso de GPU se disponível (para testes locais)
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"⚠️  GPU config warning: {e}")
        
        print("⚙️  Configured for GitHub Free environment")
    
    def _setup_directories(self):
        """Cria diretórios necessários para o treino."""
        paths = self.config.get("paths")
        self.model_dir = paths["model_dir"]
        self.results_dir = paths["results_dir"]
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        
        # Criar diretórios
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        print(f"📁 Model dir: {self.model_dir}")
        print(f"📁 Results dir: {self.results_dir}")
    
    def _initialize_report(self) -> Dict:
        """Inicializa o relatório de treino."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "horizons": self.config.get("horizons"),
                "targets": self.config.get("targets")["primary"],
                "lookback": self.config.get("training")["main_lookback"],
                "batch_size": self.config.get("training")["batch_size"],
                "max_epochs": self.config.get("training")["max_epochs"]
            },
            "physical_constraints": True,
            "github_free_optimized": True,
            "horizons": {},
            "resource_usage": {}
        }
    
    def _log_memory(self, stage: str):
        """Registra uso de memória para monitoramento."""
        mem = psutil.virtual_memory()
        self.train_report["resource_usage"][stage] = {
            "percent": float(mem.percent),
            "available_gb": float(mem.available / 1e9),
            "used_gb": float(mem.used / 1e9)
        }
        
        # Log apenas se uso > 70% ou em pontos críticos
        if mem.percent > 70 or stage in ["After horizon", "After training"]:
            print(f"   🧠 {stage} - Mem: {mem.percent:.1f}% | "
                  f"Disponível: {mem.available / 1e9:.1f} GB")
    
    def run(self):
        """Executa o pipeline completo de treino."""
        print("\n" + "=" * 60)
        print("🔥 STARTING COMPLETE TRAINING PIPELINE")
        print("=" * 60)
        
        try:
            # 1. Construir datasets
            print("\n📊 STEP 1: Building datasets...")
            datasets = self.feature_builder.build_all()
            
            if not datasets:
                raise ValueError("❌ No datasets generated!")
            
            # 2. Configurações de treino
            training_config = self.config.get("training")
            lookback = training_config["main_lookback"]
            horizons = self.config.get("horizons")
            targets = self.config.get("targets")["primary"]
            
            # Ajustes para GitHub Free
            batch_size = min(training_config["batch_size"], 32)
            max_epochs = min(training_config["max_epochs"], 50)
            
            print(f"\n⚙️  Training configuration:")
            print(f"   • Horizons: {horizons}")
            print(f"   • Targets: {targets}")
            print(f"   • Lookback: {lookback}")
            print(f"   • Batch size: {batch_size} (adjusted for GitHub Free)")
            print(f"   • Max epochs: {max_epochs} (adjusted for GitHub Free)")
            
            # 3. Treinar para cada horizonte
            trained_horizons = 0
            for horizon in horizons[:3]:  # Limitar a 3 horizontes no GitHub Free
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
                        targets=targets
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
            else:
                print("\n❌ No horizons were successfully trained")
            
            # 5. Resumo de recursos
            self._print_resource_summary()
            
        except MemoryError as e:
            print(f"\n💥 OUT OF MEMORY ERROR: {str(e)}")
            print("\n🔧 Recommendations for GitHub Free:")
            print("   1. Reduce batch_size in config.yaml")
            print("   2. Reduce max_epochs")
            print("   3. Use fewer horizons")
            print("   4. Reduce lookback window")
            raise
        
        except Exception as e:
            print(f"\n💥 TRAINING FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _train_single_horizon(self, horizon: int, dataset: Dict,
                            lookback: int, batch_size: int,
                            max_epochs: int, targets: List[str]) -> bool:
        """Treina modelo para um único horizonte."""
        try:
            # 1. Preparar dados
            print(f"\n📦 Preparing data for horizon {horizon}h...")
            X = dataset["X"]
            y_scaled = dataset["y_scaled"]  # y já escalonado por variável
            y_raw = dataset["y_raw"]        # y original para métricas
            
            print(f"   Dataset shape: X={X.shape}, y={y_scaled.shape}")
            
            # 2. Split dos dados
            print("📊 Splitting data...")
            X_train, y_train_scaled, X_val, y_val_scaled, X_test, y_test_scaled, y_test_raw = \
                self._split_data(X, y_scaled, y_raw)
            
            # 3. Construir modelo
            print("🔨 Building physical model...")
            model = self._build_model_for_horizon(
                input_shape=(lookback, X.shape[2]),
                targets=targets,
                horizon=horizon
            )
            
            # 4. Criar callbacks
            callbacks = self._create_callbacks_for_horizon(horizon)
            
            # 5. Treinar modelo
            print(f"\n🔥 Training model for horizon {horizon}h...")
            history = model.fit(
                X_train, y_train_scaled,
                validation_data=(X_val, y_val_scaled),
                epochs=max_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                workers=1,  # Reduzir para GitHub Free
                use_multiprocessing=False
            )
            
            # 6. Avaliar modelo
            print("\n📈 Evaluating model...")
            metrics = self._evaluate_model(
                model=model,
                X_test=X_test,
                y_test_scaled=y_test_scaled,
                y_test_raw=y_test_raw,
                horizon=horizon,
                targets=targets
            )
            
            # 7. Salvar artefatos
            self._save_horizon_artifacts(
                model=model,
                horizon=horizon,
                history=history.history,
                metrics=metrics,
                test_size=len(X_test)
            )
            
            # 8. Atualizar relatório
            self._update_horizon_report(horizon, metrics, len(X_train))
            
            return True
            
        except Exception as e:
            print(f"\n❌ Failed to train horizon {horizon}h: {str(e)}")
            return False
    
    def _split_data(self, X: np.ndarray, y_scaled: np.ndarray,
                   y_raw: np.ndarray) -> Tuple:
        """Divide os dados em treino, validação e teste."""
        n_total = len(X)
        
        # Usar splits do config
        val_split = min(self.config.get("training")["val_split"], 0.2)
        test_split = min(self.config.get("training")["test_split"], 0.2)
        
        n_val = int(n_total * val_split)
        n_test = int(n_total * test_split)
        n_train = n_total - n_val - n_test
        
        # Verificar tamanhos mínimos
        if n_train < 100:
            print(f"⚠️  Warning: Small training set ({n_train} samples)")
        
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
                               targets: List[str], horizon: int) -> tf.keras.Model:
        """Constrói modelo físico para um horizonte específico."""
        model_type = self.config.get("model", {}).get("type", "lstm")
        
        print(f"   Building {model_type} model for {targets}...")
        
        if model_type == "hybrid":
            model = self.model_builder.build_hybrid_model(input_shape, targets)
        elif model_type == "lightweight":
            model = self.model_builder.build_lightweight_model(input_shape, targets)
        else:  # Default: LSTM
            model = self.model_builder.build_lstm_model(input_shape, targets)
        
        # Resumo do modelo
        print(f"   Model built: {model.name}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parameters: {model.count_params():,}")
        
        return model
    
    def _create_callbacks_for_horizon(self, horizon: int) -> List:
        """Cria callbacks para o treino de um horizonte."""
        model_type = self.config.get("model", {}).get("type", "lstm")
        callbacks = self.model_builder.create_callbacks(horizon, model_type)
        
        # Adicionar callback customizado para monitoramento
        class ResourceMonitor(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 5 == 0:
                    mem = psutil.virtual_memory()
                    print(f"      Epoch {epoch}: Mem {mem.percent:.1f}% | "
                          f"Loss: {logs.get('loss', 0):.4f} | "
                          f"Val Loss: {logs.get('val_loss', 0):.4f}")
        
        callbacks.append(ResourceMonitor())
        
        return callbacks
    
    def _evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray,
                       y_test_scaled: np.ndarray, y_test_raw: np.ndarray,
                       horizon: int, targets: List[str]) -> Dict:
        """Avalia o modelo e retorna métricas."""
        # 1. Avaliar no conjunto de teste (escalonado)
        test_metrics = model.evaluate(X_test, y_test_scaled, verbose=0)
        
        # 2. Fazer previsões
        y_pred_scaled = model.predict(X_test, verbose=0, batch_size=32)
        
        # 3. Dessecalizar previsões
        y_pred_raw = self._inverse_scale_predictions(
            y_pred_scaled, horizon, targets
        )
        
        # 4. Calcular métricas nos valores reais
        metrics = self._calculate_metrics(y_test_raw, y_pred_raw, targets)
        
        # 5. Verificar limites físicos
        violations = self._check_physical_violations(y_pred_raw, targets)
        
        # 6. Log dos resultados
        print(f"\n📊 Evaluation results for horizon {horizon}h:")
        print(f"   • Test Loss: {test_metrics[0]:.4f}")
        print(f"   • MAE: {metrics['mae']:.4f}")
        print(f"   • RMSE: {metrics['rmse']:.4f}")
        
        if violations > 0:
            print(f"   ⚠️  Physical violations: {violations}")
        else:
            print(f"   ✅ All predictions respect physical limits")
        
        # Adicionar métricas de teste
        metrics.update({
            "test_loss": float(test_metrics[0]),
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
                # Extrair apenas esta coluna
                y_single = y_pred_scaled[:, idx].reshape(-1, 1)
                # Dessecalonar
                y_descaled = scaler.inverse_transform(y_single)
                y_pred_raw[:, idx] = y_descaled.flatten()
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
                "rmse": float(np.sqrt(mean_squared_error(y_true_var, y_pred_var)))
            }
        
        metrics["per_variable"] = per_variable_metrics
        
        return metrics
    
    def _check_physical_violations(self, y_pred: np.ndarray,
                                 targets: List[str]) -> int:
        """Conta violações de limites físicos nas previsões."""
        violations = 0
        
        # Limites físicos baseados no model builder
        physical_limits = {
            "V": (250, 1650),
            "Bz": (-40, 40),
            "n": (0, 100),
            "Bx": (-50, 50),
            "By": (-50, 50),
            "Bt": (0, 80)
        }
        
        for idx, var_name in enumerate(targets):
            if var_name in physical_limits:
                v_min, v_max = physical_limits[var_name]
                var_violations = np.sum((y_pred[:, idx] < v_min) | (y_pred[:, idx] > v_max))
                violations += int(var_violations)
                
                if var_violations > 0:
                    print(f"      ⚠️  {var_name}: {int(var_violations)} violations")
        
        return violations
    
    def _save_horizon_artifacts(self, model: tf.keras.Model, horizon: int,
                              history: Dict, metrics: Dict, test_size: int):
        """Salva todos os artefatos de um horizonte."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        model_name = f"hac_physical_h{horizon}_{timestamp}"
        out_dir = os.path.join(self.model_dir, model_name)
        
        print(f"\n💾 Saving artifacts for horizon {horizon}h...")
        print(f"   Directory: {out_dir}")
        
        # 1. Salvar modelo
        model_path = os.path.join(out_dir, "model.keras")
        model.save(model_path, save_format="keras")
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   ✅ Model saved ({model_size_mb:.1f} MB)")
        
        # 2. Salvar scalers
        self._save_scalers(out_dir, horizon)
        
        # 3. Salvar história do treino
        history_path = os.path.join(out_dir, "training_history.json")
        with open(history_path, "w") as f:
            # Converter arrays numpy para listas
            clean_history = {}
            for key, values in history.items():
                clean_history[key] = [float(v) for v in values]
            json.dump(clean_history, f, indent=2)
        
        # 4. Salvar métricas
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # 5. Salvar configuração
        config_path = os.path.join(out_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(self.train_report["config"], f, indent=2)
        
        print(f"   ✅ All artifacts saved to {out_dir}")
        
        # Atualizar relatório
        self.train_report["horizons"][str(horizon)] = {
            "model_dir": out_dir,
            "model_size_mb": model_size_mb,
            "test_samples": test_size,
            "metrics": {
                "mae": metrics.get("mae", 0),
                "rmse": metrics.get("rmse", 0)
            }
        }
    
    def _save_scalers(self, out_dir: str, horizon: int):
        """Salva os scalers X e Y."""
        os.makedirs(out_dir, exist_ok=True)
        
        # Salvar scaler X
        scaler_x_path = os.path.join(out_dir, "scaler_X.pkl")
        joblib.dump(self.feature_builder.scaler_X, scaler_x_path)
        
        # Salvar scalers Y (por variável)
        y_scalers = self.feature_builder.get_y_scalers(horizon)
        if y_scalers:
            for var_name, scaler in y_scalers.items():
                scaler_path = os.path.join(out_dir, f"scaler_y_{var_name}.pkl")
                joblib.dump(scaler, scaler_path)
            print(f"   ✅ Y scalers saved for {list(y_scalers.keys())}")
    
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
            "successful_horizons": [
                h for h in self.train_report["horizons"].keys()
            ]
        }
        
        # Salvar relatório
        report_path = os.path.join(self.results_dir, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(self.train_report, f, indent=2)
        
        print(f"\n📘 Training report saved: {report_path}")
    
    def _print_resource_summary(self):
        """Imprime resumo do uso de recursos."""
        print("\n" + "=" * 60)
        print("📊 RESOURCE USAGE SUMMARY")
        print("=" * 60)
        
        final_mem = psutil.virtual_memory()
        print(f"Final memory usage: {final_mem.percent:.1f}%")
        print(f"Available: {final_mem.available / 1e9:.1f} GB")
        
        # Verificar se há vazamentos de memória
        initial_usage = self.train_report["resource_usage"].get("After initialization", {})
        final_usage = self.train_report["resource_usage"].get("After horizon cleanup", {})
        
        if initial_usage and final_usage:
            mem_increase = final_usage.get("percent", 0) - initial_usage.get("percent", 0)
            if mem_increase > 10:
                print(f"⚠️  Memory increase: {mem_increase:.1f}% (possible leak)")
            else:
                print(f"✅ Memory management: OK ({mem_increase:.1f}% change)")
        
        print("=" * 60)


# ------------------------------------------------------------
# Funções auxiliares para execução direta
# ------------------------------------------------------------

def validate_training_environment():
    """Valida se o ambiente está pronto para treino."""
    print("🔍 Validating training environment...")
    
    issues = []
    
    # Verificar TensorFlow
    try:
        tf_version = tf.__version__
        print(f"   ✅ TensorFlow {tf_version}")
    except Exception as e:
        issues.append(f"TensorFlow error: {e}")
    
    # Verificar memória
    try:
        mem = psutil.virtual_memory()
        if mem.available < 1e9:  # Menos de 1GB disponível
            issues.append(f"Low memory: {mem.available / 1e9:.1f} GB available")
        else:
            print(f"   ✅ Memory available: {mem.available / 1e9:.1f} GB")
    except Exception as e:
        issues.append(f"Memory check error: {e}")
    
    # Verificar diretórios
    required_dirs = ["data", "models", "results"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ Directory exists: {dir_name}")
        else:
            issues.append(f"Missing directory: {dir_name}")
    
    if issues:
        print("\n⚠️  Validation issues found:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print("✅ Environment validation passed")
        return True


def create_github_free_config():
    """Cria configuração otimizada para GitHub Free."""
    config = {
        "paths": {
            "data_dir": "data",
            "model_dir": "models/github_free",
            "results_dir": "results/github_free"
        },
        "targets": {
            "primary": ["V", "Bz", "n"],
            "secondary": ["Bx", "By", "Bt"]
        },
        "horizons": [1, 3, 6],
        "training": {
            "main_lookback": 12,  # Reduzido
            "batch_size": 16,     # Reduzido
            "max_epochs": 30,     # Reduzido
            "val_split": 0.15,
            "test_split": 0.15
        },
        "model": {
            "type": "lightweight",  # Modelo leve
            "lstm_units": [32],
            "dense_units": [16],
            "dropout_rate": 0.2
        }
    }
    
    # Salvar config temporária
    import yaml
    config_path = "config_github_free.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"✅ GitHub Free config created: {config_path}")
    return config_path


# ------------------------------------------------------------
# Execução principal
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 HAC v6 TRAINING SCRIPT - CORRECTED")
    print("=" * 60)
    
    try:
        # Validar ambiente
        if not validate_training_environment():
            print("\n⚠️  Environment validation failed. Creating GitHub Free config...")
            config_path = create_github_free_config()
        else:
            config_path = "config.yaml"
        
        # Criar e executar trainer
        trainer = HACTrainer(config_path)
        trainer.run()
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Imprimir resumo final
        if trainer.train_report.get("horizons"):
            print("\n📈 TRAINING SUMMARY:")
            for horizon, data in trainer.train_report["horizons"].items():
                metrics = data.get("metrics", {})
                print(f"   Horizon {horizon}h:")
                print(f"     • MAE:  {metrics.get('mae', 'N/A'):.4f}")
                print(f"     • RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                print(f"     • Model: {data.get('model_size_mb', 'N/A'):.1f} MB")
        
        print("\n✨ Next steps:")
        print("   1. Check models/ directory for trained models")
        print("   2. Check results/ directory for training reports")
        print("   3. Use models for inference with correct scaling")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
