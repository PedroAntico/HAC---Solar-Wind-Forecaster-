#!/usr/bin/env python3
"""
hac_v6_models.py
Neural architectures for HAC v6 Solar Wind Forecaster
Compatible with TensorFlow 2.17

- LSTM, GRU, Hybrid (CNN+LSTM) e Transformer
- Suporte a ensembles (vários modelos ao mesmo tempo)
- Integração com config.yaml (models.model_type, units, num_layers, dropout, etc.)
"""

from typing import Callable, Dict, Any, List, Union

import tensorflow as tf
from tensorflow.keras import layers, models


# ------------------------------------------------------------
# Model blocks
# ------------------------------------------------------------

def build_lstm(
    input_shape,
    output_dim: int,
    units: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """LSTM empilhado simples, mas já decente."""
    inp = layers.Input(shape=input_shape, name="lstm_input")
    x = inp

    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            name=f"lstm_{i}",
        )(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"lstm_dropout_{i}")(x)

    x = layers.Dense(units, activation="relu", name="lstm_dense")(x)
    out = layers.Dense(output_dim, name="lstm_output")(x)

    return models.Model(inputs=inp, outputs=out, name="LSTMModel")


def build_gru(
    input_shape,
    output_dim: int,
    units: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """GRU empilhado simples."""
    inp = layers.Input(shape=input_shape, name="gru_input")
    x = inp

    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        x = layers.GRU(
            units,
            return_sequences=return_sequences,
            name=f"gru_{i}",
        )(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"gru_dropout_{i}")(x)

    x = layers.Dense(units, activation="relu", name="gru_dense")(x)
    out = layers.Dense(output_dim, name="gru_output")(x)

    return models.Model(inputs=inp, outputs=out, name="GRUModel")


def build_hybrid(
    input_shape,
    output_dim: int,
    units: int = 64,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """
    Hybrid CNN + LSTM: convolução temporal + LSTM em cima.
    """
    inp = layers.Input(shape=input_shape, name="hybrid_input")

    x = layers.Conv1D(32, 3, padding="same", activation="relu", name="conv1")(inp)
    x = layers.Conv1D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling1D(name="pool")(x)

    x = layers.LSTM(units, name="hybrid_lstm")(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout, name="hybrid_dropout")(x)

    x = layers.Dense(units, activation="relu", name="hybrid_dense")(x)
    out = layers.Dense(output_dim, name="hybrid_output")(x)

    return models.Model(inputs=inp, outputs=out, name="HybridCNNLSTM")


def build_transformer(
    input_shape,
    output_dim: int,
    num_heads: int = 4,
    ff_dim: int = 64,
    depth: int = 2,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """
    Transformer encoder simples para séries temporais.
    """
    inp = layers.Input(shape=input_shape, name="transformer_input")
    x = inp

    for i in range(depth):
        # Multi-head self-attention
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[-1],
            name=f"mha_{i}",
        )(x, x)
        attn_out = layers.Dropout(dropout, name=f"attn_dropout_{i}")(attn_out)
        x = layers.Add(name=f"attn_residual_{i}")([x, attn_out])
        x = layers.LayerNormalization(name=f"attn_ln_{i}")(x)

        # Feed-forward
        ff = layers.Dense(ff_dim, activation="relu", name=f"ff_{i}_1")(x)
        ff = layers.Dense(input_shape[-1], name=f"ff_{i}_2")(ff)
        ff = layers.Dropout(dropout, name=f"ff_dropout_{i}")(ff)
        x = layers.Add(name=f"ff_residual_{i}")([x, ff])
        x = layers.LayerNormalization(name=f"ff_ln_{i}")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="dense_final")(x)
    out = layers.Dense(output_dim, name="transformer_output")(x)

    return models.Model(inputs=inp, outputs=out, name="TransformerModel")


# ------------------------------------------------------------
# Ensemble builder
# ------------------------------------------------------------

def build_ensemble(
    input_shape,
    output_dim: int,
    model_types: List[str],
    base_cfg: Dict[str, Any],
) -> tf.keras.Model:
    """
    Constrói um ensemble simples: mesma entrada, vários modelos,
    saída = média das previsões.
    """
    # Normaliza lista
    model_types = [m.lower() for m in model_types]

    inp = layers.Input(shape=input_shape, name="ensemble_input")
    outputs = []

    for mt in model_types:
        if mt == "lstm":
            sub = build_lstm(
                input_shape,
                output_dim,
                units=base_cfg.get("units", 64),
                num_layers=base_cfg.get("num_layers", 2),
                dropout=base_cfg.get("dropout", 0.1),
            )
        elif mt == "gru":
            sub = build_gru(
                input_shape,
                output_dim,
                units=base_cfg.get("units", 64),
                num_layers=base_cfg.get("num_layers", 2),
                dropout=base_cfg.get("dropout", 0.1),
            )
        elif mt == "hybrid":
            sub = build_hybrid(
                input_shape,
                output_dim,
                units=base_cfg.get("units", 64),
                dropout=base_cfg.get("dropout", 0.1),
            )
        elif mt == "transformer":
            tcfg = base_cfg.get("transformer", {})
            sub = build_transformer(
                input_shape,
                output_dim,
                num_heads=tcfg.get("heads", 4),
                ff_dim=tcfg.get("ff_dim", 64),
                depth=tcfg.get("depth", 2),
                dropout=base_cfg.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown model type in ensemble: {mt}")

        outputs.append(sub(inp))

    if len(outputs) == 1:
        out = outputs[0]
    else:
        out = layers.Average(name="ensemble_average")(outputs)

    name = "Ensemble_" + "_".join(model_types)
    return models.Model(inputs=inp, outputs=out, name=name)


# ------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------

def _extract_models_config(config_or_type: Union[Dict[str, Any], Any, str]) -> Dict[str, Any]:
    """
    Lê o bloco 'models' do config, mesmo se vier:
      - como string ("lstm")
      - como HACConfig
      - como dict bruto
    """
    # Caso simples: usuário passou só "lstm"
    if isinstance(config_or_type, str):
        return {"model_type": config_or_type.lower()}

    # HACConfig ou dict
    if hasattr(config_or_type, "get"):
        # HACConfig: get("models") devolve o dict que já vimos
        models_cfg = config_or_type.get("models", {})
    elif isinstance(config_or_type, dict):
        models_cfg = config_or_type.get("models", config_or_type)
    else:
        models_cfg = {}

    # Defaults
    if "model_type" not in models_cfg:
        models_cfg["model_type"] = "hybrid"

    return models_cfg


def create_model_builder(config_or_type: Union[Dict[str, Any], Any, str]) -> Callable:
    """
    Fábrica de modelos.

    Uso 1 (compatível com antes):
        builder = create_model_builder("lstm")
        model = builder(input_shape, output_dim)

    Uso 2 (via HACConfig + config.yaml):
        config = HACConfig("config.yaml")
        builder = create_model_builder(config)
        model = builder(input_shape, output_dim)

    No config.yaml você pode usar, por exemplo:

    models:
      model_type: "hybrid"      # ou "lstm", "gru", "transformer", "ensemble"
      units: 64
      num_layers: 3
      dropout: 0.15
      ensemble_types: ["lstm", "gru", "hybrid"]
    """
    models_cfg = _extract_models_config(config_or_type)
    model_type = models_cfg.get("model_type", "hybrid")

    # Se model_type já for lista, tratamos como ensemble implícito
    if isinstance(model_type, (list, tuple)):
        models_cfg.setdefault("ensemble_types", list(model_type))
        model_type = "ensemble"
        models_cfg["model_type"] = "ensemble"

    model_type = str(model_type).lower()

    def builder(input_shape, output_dim: int) -> tf.keras.Model:
        # Ensemble explícito
        if model_type == "ensemble":
            ensemble_types = models_cfg.get(
                "ensemble_types",
                ["lstm", "gru", "hybrid"],
            )
            return build_ensemble(input_shape, output_dim, ensemble_types, models_cfg)

        # Modelos individuais
        if model_type == "lstm":
            return build_lstm(
                input_shape,
                output_dim,
                units=models_cfg.get("units", 64),
                num_layers=models_cfg.get("num_layers", 2),
                dropout=models_cfg.get("dropout", 0.1),
            )
        elif model_type == "gru":
            return build_gru(
                input_shape,
                output_dim,
                units=models_cfg.get("units", 64),
                num_layers=models_cfg.get("num_layers", 2),
                dropout=models_cfg.get("dropout", 0.1),
            )
        elif model_type == "hybrid":
            return build_hybrid(
                input_shape,
                output_dim,
                units=models_cfg.get("units", 64),
                dropout=models_cfg.get("dropout", 0.1),
            )
        elif model_type == "transformer":
            tcfg = models_cfg.get("transformer", {})
            return build_transformer(
                input_shape,
                output_dim,
                num_heads=tcfg.get("heads", 4),
                ff_dim=tcfg.get("ff_dim", 64),
                depth=tcfg.get("depth", 2),
                dropout=models_cfg.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return builder


# ------------------------------------------------------------
# Teste rápido (opcional)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Mock simples pra testar sem depender do HACConfig
    class MockConfig:
        def __init__(self):
            self._cfg = {
                "models": {
                    "model_type": "ensemble",
                    "units": 64,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "ensemble_types": ["lstm", "gru", "hybrid"],
                    "transformer": {"heads": 4, "depth": 2, "ff_dim": 64},
                }
            }

        def get(self, key, default=None):
            return self._cfg.get(key, default)

    cfg = MockConfig()
    builder = create_model_builder(cfg)
    m = builder(input_shape=(168, 16), output_dim=3)
    m.summary()
