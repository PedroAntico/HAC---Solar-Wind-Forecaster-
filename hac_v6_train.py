#!/usr/bin/env python3
"""
hac_v6_train.py
Training pipeline for HAC v6 Solar Wind Forecaster

Fluxo:
- Carrega config.yaml
- Gera features e janelas (HACFeatureBuilder)
- Constroi modelo HYBRID (HACModelBuilder)
- Treina para cada horizonte
- Salva:
    • modelo (.keras)
    • scaler_X.pkl
    • metadata.json
- Gera relatório geral em results/training_report.json
"""

import os
import json
from datetime import datetime

import numpy as np
import joblib
import tensorflow as tf

from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from hac_v6_models import create_model_builder


class HACTainer:
    """Roda todo o pipeline de treino do HAC v6."""

    def __init__(self, config_path="config.yaml"):
        print("📡 Loading configuration...")
        self.config = HACConfig(config_path)

        print("🔧 Loading feature builder...")
        self.feature_builder = HACFeatureBuilder(self.config)

        print("🧠 Initializing model builder...")
        self.model_builder = create_model_builder(self.config)

        paths = self.config.get("paths")
        self.model_dir = paths["model_dir"]
        self.results_dir = paths["results_dir"]
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.train_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "horizons": {}
        }

    # ------------------------------------------------------------
    def run(self):
        print("\n🚀 Starting HAC v6 training pipeline...")
        print("---------------------------------------------------")

        datasets = self.feature_builder.build_all()

        lookback = self.config.get("training")["main_lookback"]
        horizons = self.config.get("horizons")
        targets = self.config.get("targets")["primary"]

        # deduz feature_count de qualquer horizonte
        first_h = horizons[0]
        feature_count = datasets[first_h]["X"].shape[2]
        output_dim = len(targets)

        model_type = self.config.get("model", {}).get("type", "hybrid")
        print(f"\n🧠 Model type (from config): {model_type.upper()}")

        for horizon in horizons:
            print(f"\n========== TRAINING HORIZON {horizon}h ==========")

            X = datasets[horizon]["X"]
            y = datasets[horizon]["y"]

            n = len(X)
            val_split = self.config.get("training")["val_split"]
            test_split = self.config.get("training")["test_split"]

            n_train = int(n * (1 - val_split - test_split))
            n_val = int(n * val_split)

            X_train = X[:n_train]
            y_train = y[:n_train]

            X_val = X[n_train:n_train + n_val]
            y_val = y[n_train:n_train + n_val]

            X_test = X[n_train + n_val:]
            y_test = y[n_train + n_val:]

            print(f"→ Dataset: {X.shape}")
            print(f"   train = {len(X_train)}, val = {len(X_val)}, test = {len(X_test)}")

            model = self.model_builder.build_model(
                model_type=model_type,
                input_shape=(lookback, feature_count),
                output_dim=output_dim
            )

            callbacks = self.model_builder.create_advanced_callbacks(
                model_type,
                horizon
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.get("training")["max_epochs"],
                batch_size=self.config.get("training")["batch_size"],
                callbacks=callbacks,
                verbose=1
            )

            print("\n📊 Evaluating...")
            eval_results = model.evaluate(X_test, y_test, verbose=0)
            # [loss, mae, mse, rmse]
            loss, mae, mse, rmse = eval_results

            print(f"🎯 Horizon {horizon}h — "
                  f"LOSS={loss:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

            self._save_artifacts(model, model_type, horizon, history, loss, mae, mse, rmse)

        self._save_report()
        print("\n\n🎉 ALL TRAINING COMPLETE!")

    # ------------------------------------------------------------
    def _save_artifacts(self, model, model_type, horizon, history, loss, mae, mse, rmse):

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            self.model_dir,
            f"{model_type}_h{horizon}_{timestamp}"
        )
        os.makedirs(out_dir, exist_ok=True)

        model_path = os.path.join(out_dir, "model.keras")
        model.save(model_path)
        print(f"💾 Saved model → {model_path}")

        # scaler_X
        scaler_X = self.feature_builder.scalers["X"]
        joblib.dump(scaler_X, os.path.join(out_dir, "scaler_X.pkl"))

        metadata = {
            "model_type": model_type,
            "horizon": horizon,
            "timestamp": timestamp,
            "lookback": self.config.get("training")["main_lookback"],
            "feature_count": self.feature_builder.raw_df.select_dtypes(include=["number"]).shape[1],
            "targets": self.config.get("targets")["primary"],
            "metrics": {
                "loss": float(loss),
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse)
            },
            "history": {k: [float(vv) for vv in v] for k, v in history.history.items()}
        }

        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"📄 Saved metadata → {out_dir}/metadata.json\n")

        # Atualiza relatório
        self.train_report["horizons"][str(horizon)] = {
            "model_dir": out_dir,
            "metrics": metadata["metrics"]
        }

    # ------------------------------------------------------------
    def _save_report(self):
        report_path = os.path.join(self.results_dir, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(self.train_report, f, indent=2)
        print(f"📘 Training report saved → {report_path}")


# ------------------------------------------------------------
if __name__ == "__main__":
    trainer = HACTainer()
    trainer.run()
