#!/usr/bin/env python3
"""
hac_v6_models.py
Model builder for HAC v6 Solar Wind Forecasting System

• Supports HYBRID (CNN + LSTM ultraleve)
• Clean, stable, compatible with hac_v6_train.py and hac_v6_predictor.py
• Provides:
      create_model_builder(config)
      HACModelBuilder.build_model(...)
      HACModelBuilder.create_advanced_callbacks(...)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class HACModelBuilder:

    def __init__(self, config):
        self.config = config

    # ======================================================================
    # FACTORY
    # ======================================================================
    def build_model(self, model_type, input_shape, output_dim):
        """
        Returns a compiled Keras model.
        """
        if model_type == "hybrid":
            return self._build_hybrid(input_shape, output_dim)

        raise ValueError(f"Unknown model type: {model_type}")

    # ======================================================================
    # HYBRID MODEL (CNN + LSTM)
    # Ultraleve, rápido, estável e compatível com seu predictor
    # ======================================================================
    def _build_hybrid(self, input_shape, output_dim):

        inputs = layers.Input(shape=input_shape)

        # 1) CNN leve para extrair padrões locais
        x = layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation="relu",
            padding="same"
        )(inputs)

        x = layers.MaxPooling1D(pool_size=2)(x)

        # 2) LSTM leve para capturar dependências temporais
        x = layers.LSTM(32, return_sequences=False)(x)

        # 3) Saída final de 3 variáveis (speed, bz_gse, density)
        outputs = layers.Dense(output_dim)(x)

        model = models.Model(inputs, outputs)

        # ------------------------------------------------------------------
        # 🔥 COMPILAÇÃO CONSISTENTE COM TRAIN + PREDICTOR
        # IMPORTANTE → métricas devolvem APENAS:
        #   loss, mae, mse, rmse
        # ------------------------------------------------------------------
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

    # ======================================================================
    # CALLBACKS PROFISSIONAIS
    # ======================================================================
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


# ======================================================================
# FACTORY WRAPPER
# ======================================================================
def create_model_builder(config):
    return HACModelBuilder(config)
