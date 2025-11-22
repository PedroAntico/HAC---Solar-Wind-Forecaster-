#!/usr/bin/env python3
"""
hac_v6_models.py
Neural architectures for HAC v6 Solar Wind Forecaster
Compatible with TensorFlow 2.17
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------------------------------------------
# Base Builders
# ------------------------------------------------------------

def build_lstm(input_shape, output_dim):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(output_dim)
    ])
    return model


def build_gru(input_shape, output_dim):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64, return_sequences=True),
        layers.GRU(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(output_dim)
    ])
    return model


def build_hybrid(input_shape, output_dim):
    """
    Hybrid CNN + LSTM architecture
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D()(x)

    x = layers.LSTM(64)(x)
    x = layers.Dense(32, activation="relu")(x)

    out = layers.Dense(output_dim)(x)

    model = models.Model(inputs=inp, outputs=out)
    return model


def build_transformer(input_shape, output_dim, num_heads=4, ff_dim=64):
    """
    Minimal Transformer encoder block for time series
    """
    inp = layers.Input(shape=input_shape)

    x = layers.LayerNormalization()(inp)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    x = layers.Dropout(0.1)(x)
    x = layers.Add()([x, inp])  # residual

    y = layers.LayerNormalization()(x)
    y = layers.Dense(ff_dim, activation="relu")(y)
    y = layers.Dense(input_shape[-1])(y)
    y = layers.Add()([y, x])  # residual

    y = layers.Flatten()(y)
    y = layers.Dense(64, activation="relu")(y)
    out = layers.Dense(output_dim)(y)

    return models.Model(inputs=inp, outputs=out)

# ------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------

def create_model_builder(model_type):
    """
    Returns a function that builds the requested model.
    """
    builders = {
        "lstm": build_lstm,
        "gru": build_gru,
        "hybrid": build_hybrid,
        "transformer": build_transformer
    }

    if model_type not in builders:
        raise ValueError(f"Unknown model type: {model_type}")

    return builders[model_type]
