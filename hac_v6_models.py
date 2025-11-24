#!/usr/bin/env python3
"""
hac_v6_models.py
Model builder for HAC v6 Solar Wind Forecasting System

• HYBRID (CNN + LSTM) ultraleve
• Compatível com hac_v6_train.py e hac_v6_predictor.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class HACModelBuilder:

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------
    def build_model(self, model_type, input_shape, output_dim):
        if model_type == "hybrid":
            return self._build_hybrid(input_shape, output_dim)
        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------
    def _build_hybrid(self, input_shape, output_dim):

        inputs = layers.Input(shape=input_shape)

        # CNN leve
        x = layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation="relu",
            padding="same"
        )(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # LSTM leve
        x = layers.LSTM(32, return_sequences=False)(x)

        # Saída final
        outputs = layers.Dense(output_dim)(x)

        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="mse",
            metrics=[
                "mae",
                "mse",
                tf.keras.metrics.RootMeanSquaredError(name="rmse")
            ]
        )

        return model

    # ------------------------------------------------------------
    def create_advanced_callbacks(self, model_type, horizon):

        return [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]


def create_model_builder(config):
    return HACModelBuilder(config)
