"""Run the published neural target-permutation significance test on tabular data.

The script is intentionally dataset-agnostic. It accepts any CSV containing a target
column, standardizes numeric predictors, trains the shallow neural significance model,
and saves feature-level test statistics and empirical p-values.

Examples
--------
Binary classification:
python experiments/run_real_data_significance.py \
    --csv data.csv --target outcome --task binary --permutations 200

Regression:
python experiments/run_real_data_significance.py \
    --csv data.csv --target response --task regression --permutations 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from targetperm_nn import NetworkConfig, NeuralTargetPermutationTest


def prepare_features(frame: pd.DataFrame, target: str):
    X = frame.drop(columns=[target])
    y = frame[target].to_numpy()

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_transformed = preprocessor.fit_transform(X).astype(np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return X_transformed, y, feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--task",
        choices=("regression", "binary", "multiclass"),
        required=True,
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/feature_significance.csv"))
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    if args.target not in frame.columns:
        raise ValueError(f"Target column '{args.target}' was not found in {args.csv}.")

    X, y, feature_names = prepare_features(frame, args.target)

    n_classes = None
    if args.task == "multiclass":
        classes, y = np.unique(y, return_inverse=True)
        n_classes = len(classes)
    elif args.task == "binary":
        classes, y = np.unique(y, return_inverse=True)
        if len(classes) != 2:
            raise ValueError("Binary task requires exactly two target classes.")

    config = NetworkConfig(
        hidden_units=(32, 16),
        learning_rate=1e-3,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        patience=10,
        verbose=0,
    )

    test = NeuralTargetPermutationTest(
        task=args.task,
        n_permutations=args.permutations,
        network_config=config,
        n_classes=n_classes,
        random_state=args.seed,
    )
    result = test.fit(X, y, feature_names=feature_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.summary().to_csv(args.output, index=False)
    print(result.summary().to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
