#!/usr/bin/env python3
"""
hac_v6_train.py
Training pipeline for HAC v6 Solar Wind Forecaster

- Loads engineered datasets (hac_v6_features)
- Builds models (hac_v6_models)
- Trains per horizon (1–48 hours)
- Saves models + scalers + metadata
"""

import os
import json
import numpy as np
from datetime import datetime
from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from hac_v6_models import create_model_builder

import tensorflow as tf


class HACTainer:
    """Runs the full HAC training pipeline"""

    def __init__(self, config_path="config.yaml"):
        print("📡 Loading configuration...")
        self.config = HACConfig(config_path)

        print("🔧 Loading feature builder...")
        self.feature_builder = HACFeatureBuilder(self.config)

        print("🧠 Initializing model builder...")
        self.model_builder = create_model_builder(self.config)

        self.model_dir = self.config.get("paths")["model_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Main training routine
    # ------------------------------------------------------------
    def run(self):
        print("\n🚀 Starting HAC v6 training...")
        print("---------------------------------------")

        # Build all datasets
        datasets = self.feature_builder.build_all()

        lookback = self.config.get("training")["main_lookback"]
        feature_count = len(datasets[self.config.get("horizons")[0]]["X"][0][0])
        output_dim = len(self.config.get("targets")["primary"])

        # Train for each horizon
        for horizon in self.config.get("horizons"):
            print(f"\n===== HORIZON {horizon}h =====")

            X = datasets[horizon]["X"]
            y = datasets[horizon]["y"]

            # Split
            n = len(X)
            val_split = self.config.get("training")["val_split"]
            test_split = self.config.get("training")["test_split"]

            n_train = int(n * (1 - val_split - test_split))
            n_val = int(n * val_split)

            X_train, y_train = X[:n_train], y[:n_train]
            X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
            X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

            print(f"Dataset: {X.shape} → train:{len(X_train)}, val:{len(X_val)}, test:{len(X_test)}")

            # Train each model type
            for model_type in self.config.get("models")["default"]:
                print(f"\n🧠 Training model: {model_type.upper()} (h={horizon})")

                model = self.model_builder.build_model(
                    model_type=model_type,
                    input_shape=(lookback, feature_count),
                    output_dim=output_dim
                )

                callbacks = self.model_builder.create_advanced_callbacks(
                    model_type, horizon
                )

                # Train
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config.get("training")["max_epochs"],
                    batch_size=self.config.get("training")["batch_size"],
                    verbose=1,
                    callbacks=callbacks
                )

                # Evaluate
                loss, mae, mse, rmse, dir_acc = model.evaluate(X_test, y_test, verbose=0)

                print(f"🎯 Test RMSE: {rmse:.3f} | MAE: {mae:.3f} | Directional: {dir_acc:.3f}")

                # Save ALL artifacts
                self.save_artifacts(model, model_type, horizon, history)

        print("\n✅ ALL TRAINING COMPLETE!")

    # ------------------------------------------------------------
    # Artifact saver
    # ------------------------------------------------------------
    def save_artifacts(self, model, model_type, horizon, history):
        """Save model, scaler, metadata"""

        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.model_dir, f"{model_type}_h{horizon}_{stamp}")
        os.makedirs(out_dir, exist_ok=True)

        # Save model
        model.save(os.path.join(out_dir, "model.h5"))

        # Save scalers used in the feature builder
        import joblib
        scaler = self.feature_builder.scalers["main"]
        joblib.dump(scaler, os.path.join(out_dir, "scaler_X.pkl"))
        joblib.dump(scaler, os.path.join(out_dir, "scaler_Y.pkl"))

        # Metadata
        metadata = {
            "model_type": model_type,
            "horizon": horizon,
            "timestamp": stamp,
            "feature_count": len(self.feature_builder.df.columns),
            "targets": self.config.get("targets")["primary"],
            "history": {k: list(v) for k, v in history.history.items()}
        }

        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Saved: {out_dir}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    trainer = HACTainer()
    trainer.run()
