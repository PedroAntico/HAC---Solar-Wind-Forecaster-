#!/usr/bin/env python3
"""
hac_v6_models.py
Neural architectures for HAC v6 Solar Wind Forecaster

Includes:
- LSTM
- GRU
- Hybrid CNN+LSTM
- Transformer-lite
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class HACModelBuilder:
    """Builds deep learning models based on model_type string."""

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------
    # LSTM Model
    # ------------------------------------------------------------
    def build_lstm(self, input_shape, output_dim):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(32, activation="relu"),
            layers.Dense(output_dim)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                self.config.get("training", {}).get("learning_rate", 0.001)
            ),
            loss="mse"
        )
        return model

    # ------------------------------------------------------------
    # GRU Model
    # ------------------------------------------------------------
    def build_gru(self, input_shape, output_dim):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.GRU(64, return_sequences=True),
            layers.GRU(32),
            layers.Dense(32, activation="relu"),
            layers.Dense(output_dim)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                self.config.get("training", {}).get("learning_rate", 0.001)
            ),
            loss="mse"
        )
        return model

    # ------------------------------------------------------------
    # Hybrid Model (CNN + LSTM)
    # ------------------------------------------------------------
    def build_hybrid(self, input_shape, output_dim):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(64),
            layers.Dense(32, activation="relu"),
            layers.Dense(output_dim)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                self.config.get("training", {}).get("learning_rate", 0.001)
            ),
            loss="mse"
        )
        return model

    # ------------------------------------------------------------
    # Transformer-lite Model
    # ------------------------------------------------------------
    def build_transformer(self, input_shape, output_dim):
        inputs = layers.Input(shape=input_shape)

        x = layers.LayerNormalization()(inputs)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        x = layers.Dense(64, activation="relu")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(output_dim)(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                self.config.get("training", {}).get("learning_rate", 0.001)
            ),
            loss="mse"
        )
        return model

    # ------------------------------------------------------------
    # Model Router
    # ------------------------------------------------------------
    def build_model(self, model_type, input_shape, output_dim):
        model_type = model_type.lower()

        if model_type == "lstm":
            return self.build_lstm(input_shape, output_dim)
        elif model_type == "gru":
            return self.build_gru(input_shape, output_dim)
        elif model_type == "hybrid":
            return self.build_hybrid(input_shape, output_dim)
        elif model_type == "transformer":
            return self.build_transformer(input_shape, output_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    def create_callbacks(self, model_type, horizon, ensemble_id=None):
        name = f"{model_type}_h{horizon}"
        if ensemble_id is not None:
            name += f"_e{ensemble_id}"

        cb = [
            callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor="val_loss"
            ),
            callbacks.ReduceLROnPlateau(
                patience=5,
                monitor="val_loss"
            )
        ]
        return cb
