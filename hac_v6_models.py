#!/usr/bin/env python3
"""
hac_v6_models.py
Neural architectures for HAC v6 Solar Wind Forecaster
Compatible with TensorFlow 2.17

- LSTM, GRU, Hybrid (CNN+LSTM) e Transformer
- Suporte a ensembles (vários modelos ao mesmo tempo)
- Integração com config.yaml
"""

from typing import Callable, Dict, Any, List, Union
import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
#  MODEL BLOCKS
# ============================================================

def build_lstm(input_shape, output_dim, units=64, num_layers=2, dropout=0.1):
    inp = layers.Input(shape=input_shape, name="lstm_input")
    x = inp

    for i in range(num_layers):
        x = layers.LSTM(
            units,
            return_sequences=(i < num_layers - 1),
            name=f"lstm_{i}",
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    x = layers.Dense(units, activation="relu")(x)
    out = layers.Dense(output_dim)(x)

    return models.Model(inp, out, name="LSTMModel")


def build_gru(input_shape, output_dim, units=64, num_layers=2, dropout=0.1):
    inp = layers.Input(shape=input_shape, name="gru_input")
    x = inp

    for i in range(num_layers):
        x = layers.GRU(
            units,
            return_sequences=(i < num_layers - 1),
            name=f"gru_{i}",
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    x = layers.Dense(units, activation="relu")(x)
    out = layers.Dense(output_dim)(x)

    return models.Model(inp, out, name="GRUModel")


def build_hybrid(input_shape, output_dim, units=64, dropout=0.1):
    inp = layers.Input(shape=input_shape, name="hybrid_input")

    x = layers.Conv1D(32, 3, activation="relu", padding="same")(inp)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D()(x)

    x = layers.LSTM(units)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(units, activation="relu")(x)
    out = layers.Dense(output_dim)(x)

    return models.Model(inp, out, name="HybridCNNLSTM")


def build_transformer(
    input_shape,
    output_dim,
    num_heads=4,
    ff_dim=64,
    depth=2,
    dropout=0.1,
):
    inp = layers.Input(shape=input_shape, name="transformer_input")
    x = inp

    for i in range(depth):
        att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[-1],
            name=f"mha_{i}",
        )(x, x)
        att = layers.Dropout(dropout)(att)
        x = layers.Add()([x, att])
        x = layers.LayerNormalization()(x)

        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dense(input_shape[-1])(ff)
        ff = layers.Dropout(dropout)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(output_dim)(x)

    return models.Model(inp, out, name="TransformerModel")


# ============================================================
#  ENSEMBLE BUILDER
# ============================================================
def build_ensemble(input_shape, output_dim, model_types, cfg):
    inp = layers.Input(shape=input_shape, name="ensemble_input")
    outs = []

    for mt in model_types:
        mt = mt.lower()

        if mt == "lstm":
            m = build_lstm(
                input_shape,
                output_dim,
                units=cfg.get("units", 64),
                num_layers=cfg.get("num_layers", 2),
                dropout=cfg.get("dropout", 0.1),
            )

        elif mt == "gru":
            m = build_gru(
                input_shape,
                output_dim,
                units=cfg.get("units", 64),
                num_layers=cfg.get("num_layers", 2),
                dropout=cfg.get("dropout", 0.1),
            )

        elif mt == "hybrid":
            m = build_hybrid(
                input_shape,
                output_dim,
                units=cfg.get("units", 64),
                dropout=cfg.get("dropout", 0.1),
            )

        elif mt == "transformer":
            tcfg = cfg.get("transformer", {})
            m = build_transformer(
                input_shape,
                output_dim,
                num_heads=tcfg.get("heads", 4),
                ff_dim=tcfg.get("ff_dim", 64),
                depth=tcfg.get("depth", 2),
                dropout=cfg.get("dropout", 0.1),
            )

        else:
            raise ValueError(f"Unknown model in ensemble: {mt}")

        outs.append(m(inp))

    if len(outs) == 1:
        out = outs[0]
    else:
        out = layers.Average(name="ensemble_average")(outs)

    name = "Ensemble_" + "_".join(model_types)
    return models.Model(inputs=inp, outputs=out, name=name)


# ============================================================
#  MODEL FACTORY (MAIN ENTRY POINT)
# ============================================================

def create_model_builder(config: Dict[str, Any]) -> Callable:
    """
    Lê APENAS o bloco `model:` do config.yaml
    """
    cfg = config.get("model", {})
    model_type = str(cfg.get("type", "hybrid")).lower()

    def builder(input_shape, output_dim):
        # ENSEMBLE
        if model_type == "ensemble":
            model_list = cfg.get("ensemble_types", ["lstm", "gru", "hybrid"])
            return build_ensemble(input_shape, output_dim, model_list, cfg)

        # INDIVIDUAL MODELS
        if model_type == "lstm":
            return build_lstm(
                input_shape,
                output_dim,
                units=cfg.get("units", 64),
                num_layers=cfg.get("num_layers", 2),
                dropout=cfg.get("dropout", 0.1),
            )

        if model_type == "gru":
            return build_gru(
                input_shape,
                output_dim,
                units=cfg.get("units", 64),
                num_layers=cfg.get("num_layers", 2),
                dropout=cfg.get("dropout", 0.1),
            )

        if model_type == "hybrid":
            return build_hybrid(
                input_shape,
                output_dim,
                units=cfg.get("units", 64),
                dropout=cfg.get("dropout", 0.1),
            )

        if model_type == "transformer":
            tcfg = cfg.get("transformer", {})
            return build_transformer(
                input_shape,
                output_dim,
                num_heads=tcfg.get("heads", 4),
                ff_dim=tcfg.get("ff_dim", 64),
                depth=tcfg.get("depth", 2),
                dropout=cfg.get("dropout", 0.1),
            )

        raise ValueError(f"Unknown model type: {model_type}")

    return builder
