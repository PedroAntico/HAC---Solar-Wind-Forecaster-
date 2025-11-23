import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class HACModelBuilder:

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------
    # Main factory
    # ------------------------------------------------------------
    def build_model(self, model_type, input_shape, output_dim):

        if model_type == "hybrid":
            return self._build_hybrid(input_shape, output_dim)

        raise ValueError(f"Unknown model type: {model_type}")

    # ------------------------------------------------------------
    # Hybrid LSTM + CNN – ULTRA-LEVE
    # ------------------------------------------------------------
    def _build_hybrid(self, input_shape, output_dim):

        inputs = layers.Input(shape=input_shape)

        # Light 1D CNN
        x = layers.Conv1D(16, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling1D(2)(x)

        # Light LSTM
        x = layers.LSTM(32, return_sequences=False)(x)

        # Output
        outputs = layers.Dense(output_dim)(x)

        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="mse",
            metrics=["mae", "mse", tf.keras.metrics.RootMeanSquaredError()]
        )

        return model

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    def create_advanced_callbacks(self, model_type, horizon):

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6
            )
        ]

        return cb


def create_model_builder(config):
    return HACModelBuilder(config)
