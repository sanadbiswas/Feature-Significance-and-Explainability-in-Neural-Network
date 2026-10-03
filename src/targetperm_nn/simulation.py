from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class SimulationData:
    X: pd.DataFrame
    y: np.ndarray
    true_significant: tuple[str, ...] = ("X1", "X2", "X3")


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, columns=[f"X{i + 1}" for i in range(values.shape[1])])


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    """Return a positive-semidefinite correlation matrix close to ``matrix``."""

    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(psd))
    return psd / np.outer(scale, scale)


def linear_regression(
    n_samples: int = 2000,
    random_state: int = 42,
) -> SimulationData:
    """Generate independent predictors with a linear continuous target."""

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 5))
    error = rng.normal(size=n_samples)
    y = 0.8 * X[:, 0] + 0.6 * X[:, 1] + 0.7 * X[:, 2] + error
    return SimulationData(_frame(X), y)


def nonlinear_regression(
    n_samples: int = 2000,
    random_state: int = 42,
) -> SimulationData:
    """Generate a continuous target with polynomial, logarithmic, and periodic effects."""

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 5))
    X[:, 1] = np.abs(X[:, 1]) + 1e-6
    error = rng.normal(size=n_samples)
    y = 0.8 * X[:, 0] ** 2 + 0.6 * np.log(X[:, 1]) + 0.7 * np.sin(X[:, 2]) + error
    return SimulationData(_frame(X), y)


def linear_classification(
    n_samples: int = 2000,
    random_state: int = 42,
) -> SimulationData:
    """Generate a binary target from a linear predictor."""

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 5))
    error = rng.normal(size=n_samples)
    linear_term = 0.8 * X[:, 0] + 0.6 * X[:, 1] + 0.7 * X[:, 2] + error
    probability = 1.0 / (1.0 + np.exp(-linear_term))
    y = (probability > 0.5).astype(int)
    return SimulationData(_frame(X), y)


def nonlinear_classification(
    n_samples: int = 2000,
    random_state: int = 42,
) -> SimulationData:
    """Generate a binary target with polynomial, logarithmic, and periodic effects."""

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 5))
    X[:, 1] = np.abs(X[:, 1]) + 1e-6
    error = rng.normal(size=n_samples)
    nonlinear_term = (
        0.8 * X[:, 0] ** 2
        + 0.6 * np.log(X[:, 1])
        + 0.7 * np.sin(X[:, 2])
        + error
    )
    probability = 1.0 / (1.0 + np.exp(-nonlinear_term))
    y = (probability > 0.5).astype(int)
    return SimulationData(_frame(X), y)


def correlated_linear_regression(
    n_samples: int = 2000,
    include_correlated_noise: bool = False,
    noise_correlation: float = 0.6,
    random_state: int = 42,
) -> SimulationData:
    """Generate a linear regression setting with correlated predictors.

    X1-X3 have pairwise correlations 0.8, 0.6, and 0.7. When
    ``include_correlated_noise`` is true, X4 is also correlated with X1-X3 but
    is not included in the target-generating equation.
    """

    rng = np.random.default_rng(random_state)

    if include_correlated_noise:
        covariance = np.array(
            [
                [1.0, 0.8, 0.6, noise_correlation],
                [0.8, 1.0, 0.7, noise_correlation],
                [0.6, 0.7, 1.0, noise_correlation],
                [noise_correlation, noise_correlation, noise_correlation, 1.0],
            ]
        )
        covariance = _nearest_correlation(covariance)
        correlated = rng.multivariate_normal(np.zeros(4), covariance, size=n_samples)
        X = np.column_stack([correlated, rng.normal(size=n_samples)])
    else:
        covariance = np.array(
            [
                [1.0, 0.8, 0.6],
                [0.8, 1.0, 0.7],
                [0.6, 0.7, 1.0],
            ]
        )
        correlated = rng.multivariate_normal(np.zeros(3), covariance, size=n_samples)
        X = np.column_stack([correlated, rng.normal(size=(n_samples, 2))])

    error = rng.normal(size=n_samples)
    y = 0.8 * X[:, 0] + 0.6 * X[:, 1] + 0.7 * X[:, 2] + error
    return SimulationData(_frame(X), y)


def correlated_nonlinear_regression(
    n_samples: int = 2000,
    noise_correlation: float = 0.6,
    random_state: int = 42,
) -> SimulationData:
    """Generate a nonlinear regression setting with correlated predictors."""

    rng = np.random.default_rng(random_state)
    covariance = np.array(
        [
            [1.0, 0.8, 0.6, noise_correlation],
            [0.8, 1.0, 0.7, noise_correlation],
            [0.6, 0.7, 1.0, noise_correlation],
            [noise_correlation, noise_correlation, noise_correlation, 1.0],
        ]
    )
    covariance = _nearest_correlation(covariance)
    correlated = rng.multivariate_normal(np.zeros(4), covariance, size=n_samples)
    X = np.column_stack([correlated, rng.normal(size=n_samples)])
    X[:, 1] = np.abs(X[:, 1]) + 1e-6

    error = rng.normal(size=n_samples)
    y = 0.8 * X[:, 0] ** 2 + 0.6 * np.log(X[:, 1]) + 0.7 * np.sin(X[:, 2]) + error
    return SimulationData(_frame(X), y)


def correlated_noise_sweep(
    rho: float,
    n_samples: int = 2000,
    random_state: int = 42,
) -> SimulationData:
    """Vary the correlation of irrelevant X4 with X1-X3."""

    if rho < 0.0 or rho > 0.9:
        raise ValueError("rho should be between 0.0 and 0.9 for this experiment.")
    return correlated_linear_regression(
        n_samples=n_samples,
        include_correlated_noise=True,
        noise_correlation=rho,
        random_state=random_state,
    )


def dominant_signal_sweep(
    beta: float,
    n_samples: int = 2000,
    random_state: int = 42,
) -> SimulationData:
    """Generate correlated predictors with a dominant X1 effect."""

    rng = np.random.default_rng(random_state)
    covariance = np.array(
        [
            [1.0, 0.8, 0.6, 0.6],
            [0.8, 1.0, 0.7, 0.6],
            [0.6, 0.7, 1.0, 0.6],
            [0.6, 0.6, 0.6, 1.0],
        ]
    )
    correlated = rng.multivariate_normal(np.zeros(4), covariance, size=n_samples)
    X = np.column_stack([correlated, rng.normal(size=n_samples)])
    error = rng.normal(size=n_samples)
    y = X[:, 0] + beta * X[:, 1] + beta * X[:, 2] + error
    return SimulationData(_frame(X), y)


def get_permutation_stability_case(
    task: Literal["regression", "binary"],
    relationship: Literal["linear", "nonlinear"],
    n_samples: int = 2000,
    random_state: int = 42,
) -> SimulationData:
    """Return a simulation case for the permutation-count analysis."""

    if task == "regression" and relationship == "linear":
        return linear_regression(n_samples, random_state)
    if task == "regression" and relationship == "nonlinear":
        return nonlinear_regression(n_samples, random_state)
    if task == "binary" and relationship == "linear":
        return linear_classification(n_samples, random_state)
    if task == "binary" and relationship == "nonlinear":
        return nonlinear_classification(n_samples, random_state)
    raise ValueError("Unsupported task/relationship combination.")
