from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import tensorflow as tf

from .models import NetworkConfig, Task, build_tabular_network, fit_network


@dataclass
class NeuralSignificanceResult:
    """Results returned by the neural target-permutation test."""

    feature_names: list[str]
    observed_statistics: np.ndarray
    null_statistics: np.ndarray
    p_values: np.ndarray

    def summary(self) -> pd.DataFrame:
        """Return one row per feature, ordered by empirical p-value."""

        frame = pd.DataFrame(
            {
                "feature": self.feature_names,
                "test_statistic": self.observed_statistics,
                "p_value": self.p_values,
            }
        )
        return frame.sort_values(["p_value", "test_statistic"], ascending=[True, False]).reset_index(
            drop=True
        )

    def significant_features(self, alpha: float = 0.05) -> list[str]:
        """Return features with empirical p-value at or below ``alpha``."""

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        return [name for name, p in zip(self.feature_names, self.p_values) if p <= alpha]


def mean_absolute_input_gradient(
    model: tf.keras.Model,
    X: np.ndarray,
) -> np.ndarray:
    """Compute the paper's global feature test statistic.

    For a scalar model output, this is the mean absolute input gradient

        tau_j = mean_i |d y_hat_i / d x_ij|.

    For a multiclass output, the same quantity is calculated for every output
    class and averaged across classes so that each input feature has one global
    statistic. This follows the paper's statement that multi-output settings can
    be handled output by output.
    """

    X_tensor = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))

    output_dim = int(model.output_shape[-1])
    if output_dim == 1:
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor, training=False)
        gradients = tape.gradient(predictions, X_tensor)
        if gradients is None:
            raise RuntimeError("Could not compute input gradients for the supplied model.")
        return tf.reduce_mean(tf.abs(gradients), axis=0).numpy()

    class_statistics = []
    for class_index in range(output_dim):
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor, training=False)[:, class_index]
        gradients = tape.gradient(predictions, X_tensor)
        if gradients is None:
            raise RuntimeError("Could not compute class-specific input gradients.")
        class_statistics.append(tf.reduce_mean(tf.abs(gradients), axis=0).numpy())

    return np.mean(np.stack(class_statistics, axis=0), axis=0)


def empirical_p_values(
    observed_statistics: np.ndarray,
    null_statistics: np.ndarray,
) -> np.ndarray:
    """Compute empirical p-values using the formula reported in the paper.

    p_j = count(tau_j^(b) > tau_j) / B

    where B is the number of target permutations.
    """

    observed = np.asarray(observed_statistics, dtype=float)
    null = np.asarray(null_statistics, dtype=float)

    if null.ndim != 2:
        raise ValueError("null_statistics must have shape (n_permutations, n_features).")
    if null.shape[1] != observed.shape[0]:
        raise ValueError("Observed and null statistics must contain the same number of features.")
    if null.shape[0] == 0:
        raise ValueError("At least one permutation is required.")

    return np.mean(null > observed[None, :], axis=0)


class NeuralTargetPermutationTest:
    """Target-permutation feature significance test for neural networks.

    This class implements the experiment described in the published neural-network
    paper. The feature matrix is kept fixed, the target is repeatedly permuted,
    and the same neural-network architecture/training settings are used for the
    observed and permuted datasets.
    """

    def __init__(
        self,
        task: Task,
        n_permutations: int = 100,
        network_config: NetworkConfig | None = None,
        n_classes: int | None = None,
        random_state: int = 42,
        model_builder: Callable[..., tf.keras.Model] | None = None,
    ) -> None:
        if n_permutations < 1:
            raise ValueError("n_permutations must be at least 1.")

        self.task = task
        self.n_permutations = n_permutations
        self.network_config = network_config or NetworkConfig()
        self.n_classes = n_classes
        self.random_state = random_state
        self.model_builder = model_builder or build_tabular_network

    def _make_model(self, n_features: int, seed: int) -> tf.keras.Model:
        return self.model_builder(
            n_features=n_features,
            task=self.task,
            config=self.network_config,
            n_classes=self.n_classes,
            seed=seed,
        )

    def fit(
        self,
        X,
        y,
        feature_names: Iterable[str] | None = None,
    ) -> NeuralSignificanceResult:
        """Fit the observed model and all target-permuted models."""

        if isinstance(X, pd.DataFrame):
            inferred_names = list(X.columns)
            X_array = X.to_numpy(dtype=np.float32)
        else:
            X_array = np.asarray(X, dtype=np.float32)
            inferred_names = [f"X{i + 1}" for i in range(X_array.shape[1])]

        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        names = list(feature_names) if feature_names is not None else inferred_names
        if len(names) != X_array.shape[1]:
            raise ValueError("feature_names must match the number of columns in X.")

        rng = np.random.default_rng(self.random_state)

        tf.keras.backend.clear_session()
        observed_model = self._make_model(X_array.shape[1], self.random_state)
        fit_network(observed_model, X_array, y_array, self.network_config)
        observed_statistics = mean_absolute_input_gradient(observed_model, X_array)

        null_statistics = np.empty((self.n_permutations, X_array.shape[1]), dtype=float)

        for permutation_index in range(self.n_permutations):
            permuted_y = rng.permutation(y_array)
            seed = self.random_state + permutation_index + 1

            tf.keras.backend.clear_session()
            permuted_model = self._make_model(X_array.shape[1], seed)
            fit_network(permuted_model, X_array, permuted_y, self.network_config)
            null_statistics[permutation_index] = mean_absolute_input_gradient(
                permuted_model,
                X_array,
            )

        p_values = empirical_p_values(observed_statistics, null_statistics)

        return NeuralSignificanceResult(
            feature_names=names,
            observed_statistics=observed_statistics,
            null_statistics=null_statistics,
            p_values=p_values,
        )
