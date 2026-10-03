from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import tensorflow as tf

Task = Literal["regression", "binary", "multiclass"]


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for the feed-forward neural network."""

    hidden_units: Sequence[int] = (32, 16)
    activation: str = "relu"
    learning_rate: float = 1e-3
    dropout: float = 0.0
    epochs: int = 100
    batch_size: int = 128
    validation_split: float = 0.2
    patience: int = 10
    verbose: int = 0


def build_tabular_network(
    n_features: int,
    task: Task,
    config: NetworkConfig | None = None,
    n_classes: int | None = None,
    seed: int = 42,
) -> tf.keras.Model:
    """Build a feed-forward neural network for tabular data.

    Parameters
    ----------
    n_features:
        Number of input features.
    task:
        ``"regression"``, ``"binary"``, or ``"multiclass"``.
    config:
        Network architecture and training settings.
    n_classes:
        Number of classes for multiclass classification.
    seed:
        TensorFlow random seed.
    """

    cfg = config or NetworkConfig()
    tf.keras.utils.set_random_seed(seed)

    inputs = tf.keras.Input(shape=(n_features,), name="features")
    x = inputs
    for i, units in enumerate(cfg.hidden_units):
        x = tf.keras.layers.Dense(units, activation=cfg.activation, name=f"dense_{i + 1}")(x)
        if cfg.dropout > 0:
            x = tf.keras.layers.Dropout(cfg.dropout, name=f"dropout_{i + 1}")(x)

    if task == "regression":
        outputs = tf.keras.layers.Dense(1, activation="linear", name="output")(x)
        loss = "mse"
        metrics = [tf.keras.metrics.MeanSquaredError(name="mse")]
    elif task == "binary":
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
        loss = "binary_crossentropy"
        metrics = [tf.keras.metrics.AUC(name="auc")]
    elif task == "multiclass":
        if n_classes is None or n_classes < 2:
            raise ValueError("n_classes must be provided for multiclass classification.")
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="output")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    else:
        raise ValueError(f"Unknown task: {task}")

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


def fit_network(
    model: tf.keras.Model,
    X,
    y,
    config: NetworkConfig | None = None,
) -> tf.keras.callbacks.History:
    """Fit a neural network using the supplied training configuration."""

    cfg = config or NetworkConfig()
    callbacks = []
    if cfg.patience > 0 and cfg.validation_split > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.patience,
                restore_best_weights=True,
            )
        )

    return model.fit(
        X,
        y,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        callbacks=callbacks,
        verbose=cfg.verbose,
        shuffle=True,
    )
