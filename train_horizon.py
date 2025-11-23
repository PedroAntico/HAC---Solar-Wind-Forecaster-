#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import tensorflow as tf

from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from hac_v6_models import create_model_builder


def train_single_horizon(h):

    config = HACConfig("config.yaml")

    feature_builder = HACFeatureBuilder(config)
    datasets = feature_builder.build_all()

    if h not in datasets:
        raise RuntimeError(f"Horizon {h} not available in datasets")

    X = datasets[h]["X"]
    y = datasets[h]["y"]

    if len(X) < 100:
        raise RuntimeError(f"Not enough data to train horizon {h}h ({len(X)} samples).")

    lookback = config.get("training")["main_lookback"]
    feature_count = X.shape[2]
    output_dim = len(config.get("targets")["primary"])

    builder = create_model_builder(config)

    model = builder.build_model(
        model_type="hybrid",
        input_shape=(lookback, feature_count),
        output_dim=output_dim
    )

    # Train/Val/Test split
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

    callbacks = builder.create_advanced_callbacks("hybrid", h)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get("training")["max_epochs"],
        batch_size=config.get("training")["batch_size"],
        verbose=1,
        callbacks=callbacks
    )

    loss, mae, mse, rmse = model.evaluate(X_test, y_test, verbose=0)

    stamp = tf.timestamp().numpy().astype(int)
    out_dir = os.path.join(config.get("paths")["model_dir"], f"hybrid_h{h}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    model.save(os.path.join(out_dir, "model.keras"))

    import joblib
    joblib.dump(feature_builder.scalers["main"], os.path.join(out_dir, "scaler.pkl"))

    meta = {"horizon": h, "lookback": lookback, "targets": config.get("targets")["primary"]}
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✔ Saved model for horizon {h}h in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, required=True)
    args = parser.parse_args()
    train_single_horizon(args.h)
