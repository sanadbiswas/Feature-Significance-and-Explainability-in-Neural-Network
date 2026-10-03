# Target-Permutation Feature Significance for Neural Networks

Clean reference implementation of the method published in:

> Biswas, S., Grundlingh, N., Boardman, J., White, J., & Le, L. (2025). **A Target Permutation Test for Statistical Significance of Feature Importance in Differentiable Models.** *Electronics, 14*(3), 571. https://doi.org/10.3390/electronics14030571

This repository contains **only the neural-network target-permutation methodology and experiment families described in the published paper**. It does not add XGBoost, LightGBM, or new statistical procedures.

## Method

For a fitted neural network, the paper measures the global importance of feature \(X_j\) using the mean absolute input gradient

\[
\tau_j = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{\partial \hat{Y}_i}{\partial X_{ij}}\right|.
\]

To assess statistical significance:

1. Fit the neural network on the observed data \((X, Y)\).
2. Compute the observed feature statistics \(\tau_j\).
3. Randomly permute the target \(Y\) while keeping \(X\) unchanged.
4. Fit the same neural-network architecture with the same training settings to each permuted target.
5. Compute the feature statistics for every permuted model.
6. Compare the observed statistic with its empirical null distribution.

The empirical p-value implemented here follows the formula reported in the paper:

\[
p_j = \frac{\#\{b:\tau_j^{(b)} > \tau_j\}}{B},
\]

where \(B\) is the number of target permutations.

## Why target permutation?

Permuting the target breaks the predictor-target relationship while leaving the predictor matrix and its correlation structure intact. The procedure therefore evaluates all input features simultaneously without individually shuffling predictors.

## Installation

```bash
# Clone the repository
git clone https://github.com/sanadbiswas/Feature-Significance-and-Explainability-in-Neural-Network.git
cd Feature-Significance-and-Explainability-in-Neural-Network

# Install the package
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest -q
```

## Quick start

```python
from sklearn.preprocessing import StandardScaler

from targetperm_nn import NetworkConfig, NeuralTargetPermutationTest
from targetperm_nn.simulation import linear_regression

# Example data
data = linear_regression(n_samples=1000, random_state=42)
X = StandardScaler().fit_transform(data.X)

# The paper used shallow networks for the permutation process.
config = NetworkConfig(
    hidden_units=(10,),
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    patience=10,
)

test = NeuralTargetPermutationTest(
    task="regression",
    n_permutations=100,
    network_config=config,
    random_state=42,
)

result = test.fit(X, data.y, feature_names=data.X.columns)
print(result.summary())
print(result.significant_features(alpha=0.05))
```

The summary contains one row per feature:

```text
feature    test_statistic    p_value
X1         ...               ...
X2         ...               ...
...
```

## Run on your own tabular dataset

The real-data runner is dataset-agnostic and accepts a CSV file.

Binary classification:

```bash
python experiments/run_real_data_significance.py \
  --csv data.csv \
  --target outcome \
  --task binary \
  --permutations 200
```

Regression:

```bash
python experiments/run_real_data_significance.py \
  --csv data.csv \
  --target response \
  --task regression \
  --permutations 200
```

Multiclass classification is also supported with `--task multiclass`. Numeric predictors are median-imputed and standardized; categorical predictors are imputed and one-hot encoded.

## Published simulation experiments

The repository provides reusable generators and runners for the experiment families reported in the paper.

### 1. Linear and nonlinear relationships

The synthetic data contain five predictors. `X1`, `X2`, and `X3` generate the target, while `X4` and `X5` are noise features.

Linear regression:

\[
Y = 0.8X_1 + 0.6X_2 + 0.7X_3 + \epsilon.
\]

Nonlinear regression:

\[
Y = 0.8X_1^2 + 0.6\log(X_2) + 0.7\sin(X_3) + \epsilon.
\]

Corresponding binary-classification generators are also included, following the logistic construction described in the paper.

Run the core simulation set:

```bash
python experiments/run_simulations.py --experiment core --permutations 100
```

### 2. Multicollinearity

The implementation includes the scenarios in which the true signal features are correlated and the scenario in which an irrelevant `X4` is correlated with the true signal features while remaining absent from the target-generating equation.

```bash
python experiments/run_simulations.py --experiment correlation --permutations 100
```

### 3. Dominant-signal/effect-strength experiment

The paper also studied

\[
Y = X_1 + \beta X_2 + \beta X_3 + \epsilon,
\]

with \(\beta\) ranging from 0.10 to 0.20.

```bash
python experiments/run_simulations.py --experiment effect-strength --permutations 100
```

### 4. Number of permutations

The permutation-count experiment covers the four cases reported in the paper: regression/classification crossed with linear/nonlinear relationships, with permutation counts from 100 through 1000.

```bash
python experiments/run_simulations.py --experiment permutation-count
```

This is computationally expensive because a new neural network is fitted for every target permutation.

## Repository structure

```text
.
├── src/
│   └── targetperm_nn/
│       ├── __init__.py
│       ├── models.py          # general feed-forward network builder
│       ├── significance.py    # target permutation and empirical p-values
│       └── simulation.py      # simulation designs from the paper
├── experiments/
│   ├── run_simulations.py
│   └── run_real_data_significance.py
├── examples/
│   └── basic_usage.py
├── tests/
│   └── test_significance.py
├── .github/workflows/
│   └── tests.yml
├── pyproject.toml
├── CITATION.cff
└── original notebooks retained from the published project
```

## Legacy notebooks

The original HELOC notebooks remain in the repository unchanged. They are retained as historical research artifacts from the published project.

The refactored implementation in `src/targetperm_nn/` follows the **published mathematical specification**, including the mean absolute input-gradient statistic in Equation (2). The old HELOC notebook contains exploratory/legacy code and should not be treated as the canonical package implementation.

## Scope

This repository intentionally focuses on the already-published neural-network work. The cleaned code:

- implements target permutation for neural-network feature significance;
- supports regression, binary classification, and multiclass classification;
- reproduces the simulation families described in the paper;
- provides a general CSV runner for real tabular data;
- preserves the original notebooks rather than rewriting the historical research artifacts;
- does **not** require rerunning the paper's full 12-dataset benchmark simply to use the package.

## Citation

If you use this implementation, please cite:

```bibtex
@article{biswas2025target,
  title   = {A Target Permutation Test for Statistical Significance of Feature Importance in Differentiable Models},
  author  = {Biswas, Sanad and Grundlingh, Nina and Boardman, Jonathan and White, Joseph and Le, Linh},
  journal = {Electronics},
  volume  = {14},
  number  = {3},
  pages   = {571},
  year    = {2025},
  doi     = {10.3390/electronics14030571}
}
```

## Notes on reproducibility

Neural-network training is stochastic and results can vary slightly across hardware, TensorFlow/CUDA versions, and random seeds. The code exposes `random_state` and keeps model/training settings fixed between the observed and target-permuted fits, as required by the methodology.
