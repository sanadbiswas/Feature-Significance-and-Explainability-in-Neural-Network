"""Run the neural-network simulation experiments reported in the paper.

This script intentionally contains only the experiment families described in the
published neural-network paper: linear/nonlinear relationships, multicollinearity,
effect-strength analysis, and permutation-count stability.

Examples
--------
python experiments/run_simulations.py --experiment core --permutations 100
python experiments/run_simulations.py --experiment correlation --permutations 100
python experiments/run_simulations.py --experiment effect-strength --permutations 100
python experiments/run_simulations.py --experiment permutation-count
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from targetperm_nn import NetworkConfig, NeuralTargetPermutationTest
from targetperm_nn.simulation import (
    correlated_linear_regression,
    correlated_noise_sweep,
    correlated_nonlinear_regression,
    dominant_signal_sweep,
    get_permutation_stability_case,
    linear_regression,
    nonlinear_regression,
)


def run_test(data, task: str, permutations: int, hidden_units: tuple[int, ...], seed: int):
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(data.X),
        columns=data.X.columns,
    )

    config = NetworkConfig(
        hidden_units=hidden_units,
        epochs=100,
        batch_size=128,
        validation_split=0.2,
        patience=10,
        verbose=0,
    )
    test = NeuralTargetPermutationTest(
        task=task,
        n_permutations=permutations,
        network_config=config,
        random_state=seed,
    )
    return test.fit(X, data.y).summary()


def core_experiments(permutations: int, n_samples: int, seed: int) -> pd.DataFrame:
    scenarios = [
        ("linear_independent", linear_regression(n_samples, seed), (10,)),
        ("nonlinear_independent", nonlinear_regression(n_samples, seed), (10, 10)),
        (
            "linear_correlated_signals",
            correlated_linear_regression(n_samples, False, random_state=seed),
            (10,),
        ),
        (
            "linear_correlated_signal_and_noise",
            correlated_linear_regression(n_samples, True, 0.6, seed),
            (10,),
        ),
        (
            "nonlinear_correlated_signal_and_noise",
            correlated_nonlinear_regression(n_samples, 0.6, seed),
            (10, 10),
        ),
    ]

    frames = []
    for name, data, architecture in scenarios:
        result = run_test(data, "regression", permutations, architecture, seed)
        result.insert(0, "scenario", name)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def correlation_experiment(permutations: int, n_samples: int, seed: int) -> pd.DataFrame:
    frames = []
    for rho in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        data = correlated_noise_sweep(rho, n_samples, seed)
        result = run_test(data, "regression", permutations, (10,), seed)
        result.insert(0, "rho", rho)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def effect_strength_experiment(
    permutations: int,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    frames = []
    for beta in (0.10, 0.12, 0.14, 0.16, 0.18, 0.20):
        data = dominant_signal_sweep(beta, n_samples, seed)
        result = run_test(data, "regression", permutations, (10,), seed)
        result.insert(0, "beta", beta)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def permutation_count_experiment(n_samples: int, seed: int) -> pd.DataFrame:
    frames = []
    for task in ("regression", "binary"):
        for relationship in ("linear", "nonlinear"):
            for permutations in (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000):
                data = get_permutation_stability_case(task, relationship, n_samples, seed)
                architecture = (10,) if relationship == "linear" else (10, 10)
                result = run_test(data, task, permutations, architecture, seed)
                result.insert(0, "n_permutations", permutations)
                result.insert(0, "relationship", relationship)
                result.insert(0, "task", task)
                frames.append(result)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=("core", "correlation", "effect-strength", "permutation-count"),
        default="core",
    )
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/simulations"))
    args = parser.parse_args()

    if args.experiment == "core":
        results = core_experiments(args.permutations, args.n_samples, args.seed)
    elif args.experiment == "correlation":
        results = correlation_experiment(args.permutations, args.n_samples, args.seed)
    elif args.experiment == "effect-strength":
        results = effect_strength_experiment(args.permutations, args.n_samples, args.seed)
    else:
        results = permutation_count_experiment(args.n_samples, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.experiment}.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
