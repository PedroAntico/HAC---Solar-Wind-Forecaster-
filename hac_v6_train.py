#!/usr/bin/env python3
"""
hac_v6_train.py
Training pipeline for HAC v6 Solar Wind Forecaster (2025)

- Loads engineered datasets
- Selects model types from config.yaml (model.type OR ensemble)
- Trains per forecast horizon (1–48h)
- Saves complete artifacts: model, scaler, metadata
"""

import os
import json
import numpy as np
from datetime import datetime

import tensorflow as tf

from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from hac_v6_models import create_model_builder


class HACTainer:
    """Runs the full HAC v6 training pipeline"""

    def __init__(self, config_path="config.yaml"):
        print("📡 Loading configuration...")
        self.config = HACConfig(config_path)

        print("🔧 Loading feature builder...")
        self.feature_builder = HACFeatureBuilder(self.config)

        print("🧠 Initializing model builder...")
        self.model_builder = create_model_builder(self.config)

        # Output directory
        self.model_dir = self.config.get("paths")["model_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Model selection logic (NEW)
    # ------------------------------------------------------------
    def _resolve_model_types(self):
        """Decides which model(s) to train based on config.yaml"""

        mcfg = self.config.get("model", {})
        main_type = mcfg.get("type", "hybrid")

        if main_type == "ensemble":
            print("🧩 Ensemble mode enabled")
            return mcfg.get("ensemble_types", ["lstm", "gru", "hybrid"])

        return [main_type]

    # ------------------------------------------------------------
    # Main training routine
    # ------------------------------------------------------------
    def run(self):
        print("\n🚀 Starting HAC v6 training...")
        print("---------------------------------------")

        # Build full dataset
        datasets = self.feature_builder.build_all()

        horizons = self.config.get("horizons")
        lookback = self.config.get("training")["main_lookback"]
        target_dim = len(self.config.get("targets")["primary"])

        # Infer feature dimension
        any_h = horizons[0]
        feature_dim = datasets[any_h]["X"][0].shape[-1]

        # Model types to train
        model_types = self._resolve_model_types()
        print(f"📌 Training model types: {model_types}")

        # ------------------------------------------------------------
        # Train for each forecast horizon
        # ------------------------------------------------------------
        for horizon in horizons:
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

            print(f"📊 Dataset: {X.shape} → train:{len(X_train)}, val:{len(X_val)}, test:{len(X_test)}")

            # --------------------------------------------------------
            # Train each model type
            # --------------------------------------------------------
            for model_type in model_types:
                print(f"\n🧠 Training model: {model_type.upper()} (h={horizon})")

                # Build model from factory
                model = self.model_builder(
                    input_shape=(lookback, feature_dim),
                    output_dim=target_dim
                )

                # Compile
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.config.get("training")["learning_rate"]
                    ),
                    loss="mse",
                    metrics=["mae", "mse", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
                )

                # Callbacks simples (compatível com Codespaces)
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        patience=15, restore_best_weights=True, monitor="val_loss"
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        patience=5, factor=0.5, monitor="val_loss"
                    ),
                ]

                # Train
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config.get("training")["max_epochs"],
                    batch_size=self.config.get("training")["batch_size"],
                    verbose=1,
                    callbacks=callbacks,
                )

                # Evaluate
                ev = model.evaluate(X_test, y_test, verbose=0)
                rmse = ev[-1]
                print(f"🎯 Horizon {horizon}h → Test RMSE={rmse:.3f}")

                # Save everything
                self.save_artifacts(model, model_type, horizon, history)

        print("\n✅ ALL TRAINING COMPLETE!")

    # ------------------------------------------------------------
    # Artifact saver
    # ------------------------------------------------------------
    def save_artifacts(self, model, model_type, horizon, history):
        """Save model, scalers, metadata"""

        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.model_dir, f"{model_type}_h{horizon}_{stamp}")
        os.makedirs(out_dir, exist_ok=True)

        # Save TF model
        model.save(os.path.join(out_dir, "model.h5"))

        # Save scalers
        import joblib
        scaler = self.feature_builder.scalers["main"]
        joblib.dump(scaler, os.path.join(out_dir, "scaler_X.pkl"))
        joblib.dump(scaler, os.path.join(out_dir, "scaler_Y.pkl"))

        # Metadata
        meta = {
            "model_type": model_type,
            "horizon": horizon,
            "timestamp": stamp,
            "feature_count": len(self.feature_builder.df.columns),
            "targets": self.config.get("targets")["primary"],
            "history": {k: list(v) for k, v in history.history.items()},
        }

        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"💾 Saved: {out_dir}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    trainer = HACTainer()
    trainer.run()
