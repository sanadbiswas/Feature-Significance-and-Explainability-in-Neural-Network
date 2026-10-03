# Target-Permutation Feature Significance for Neural Networks

Implementation of the target-permutation method presented in:

> Biswas, S., Grundlingh, N., Boardman, J., White, J., & Le, L. (2025). **A Target Permutation Test for Statistical Significance of Feature Importance in Differentiable Models.** *Electronics, 14*(3), 571. https://doi.org/10.3390/electronics14030571

An earlier conference version of this work appeared in:

> Biswas, S., Grundlingh, N., Boardman, J., White, J., & Le, L. (2024). **Target Permutation for Feature Significance and Applications in Neural Networks.** *2024 International Conference on Machine Learning and Applications (ICMLA)*, 1115–1120. https://doi.org/10.1109/ICMLA61862.2024.00170

This repository provides reusable code for neural-network feature significance testing, simulation experiments, and tabular-data applications based on the methodology described in these works.

## Method

For a fitted neural network, the global importance of feature $X_j$ is measured using the mean absolute input gradient:

$$
\tau_j = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{\partial \hat{Y}_i}{\partial X_{ij}}\right|.
$$

Statistical significance is assessed as follows:

1. Fit the neural network on the observed data $(X, Y)$.
2. Compute the observed feature statistics $\tau_j$.
3. Randomly permute the target $Y$ while keeping $X$ unchanged.
4. Fit the same neural-network architecture and training configuration to each permuted target.
5. Compute the feature statistics for each permuted model.
6. Compare the observed statistic with its empirical null distribution.

The empirical p-value is

$$
p_j = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\left(\tau_j^{(b)} > \tau_j\right),
$$

where $B$ is the number of target permutations and $\mathbf{1}(\cdot)$ is the indicator function.

## Why target permutation?

Permuting the target breaks the predictor-target relationship while preserving the predictor matrix and its correlation structure. This allows the significance of all input features to be evaluated simultaneously without permuting each predictor separately.

## Installation

```bash
git clone https://github.com/sanadbiswas/Feature-Significance-and-Explainability-in-Neural-Network.git
cd Feature-Significance-and-Explainability-in-Neural-Network
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
pytest -q
```

## Quick start

```python
from sklearn.preprocessing import StandardScaler

from targetperm_nn import NetworkConfig, NeuralTargetPermutationTest
from targetperm_nn.simulation import linear_regression

data = linear_regression(n_samples=1000, random_state=42)
X = StandardScaler().fit_transform(data.X)

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

## Run on a tabular dataset

The real-data runner accepts a CSV file and supports regression, binary classification, and multiclass classification.

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

For multiclass classification, use `--task multiclass`. Numeric predictors are median-imputed and standardized; categorical predictors are imputed and one-hot encoded.

## Simulation experiments

The repository includes the simulation settings used to study linear and nonlinear relationships, multicollinearity, signal strength, and the number of target permutations.

### Linear and nonlinear relationships

The synthetic datasets contain five predictors. `X1`, `X2`, and `X3` contribute to the target, while `X4` and `X5` are noise features.

Linear regression:

$$
Y = 0.8X_1 + 0.6X_2 + 0.7X_3 + \epsilon.
$$

Nonlinear regression:

$$
Y = 0.8X_1^2 + 0.6\log(X_2) + 0.7\sin(X_3) + \epsilon.
$$

Binary-classification versions of the linear and nonlinear settings are also included.

```bash
python experiments/run_simulations.py --experiment core --permutations 100
```

### Multicollinearity

The multicollinearity experiments include correlated signal features and settings in which an irrelevant feature is correlated with the signal features.

```bash
python experiments/run_simulations.py --experiment correlation --permutations 100
```

### Effect strength

The effect-strength experiment uses

$$
Y = X_1 + \beta X_2 + \beta X_3 + \epsilon,
$$

with $\beta$ ranging from 0.10 to 0.20.

```bash
python experiments/run_simulations.py --experiment effect-strength --permutations 100
```

### Number of permutations

The permutation-count experiment evaluates regression and classification under both linear and nonlinear relationships for permutation counts from 100 to 1000.

```bash
python experiments/run_simulations.py --experiment permutation-count
```

Because a neural network is fitted for each target permutation, large permutation counts can require substantial computation.

## Repository structure

```text
.
├── src/
│   └── targetperm_nn/
│       ├── __init__.py
│       ├── models.py
│       ├── significance.py
│       └── simulation.py
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
├── requirements.txt
└── CITATION.cff
```

## Original notebooks

The original HELOC notebooks used during the research project are retained in the repository for reference. The reusable implementation is provided in `src/targetperm_nn/`.

## Citation

If you use this code, please cite the journal article:

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

Conference paper:

```bibtex
@inproceedings{biswas2024target,
  title     = {Target Permutation for Feature Significance and Applications in Neural Networks},
  author    = {Biswas, Sanad and Grundlingh, Nina and Boardman, Jonathan and White, Joseph and Le, Linh},
  booktitle = {2024 International Conference on Machine Learning and Applications (ICMLA)},
  pages     = {1115--1120},
  year      = {2024},
  doi       = {10.1109/ICMLA61862.2024.00170}
}
```

## Reproducibility

Neural-network training is stochastic, so results may vary slightly across hardware, TensorFlow/CUDA versions, and random seeds. The implementation exposes `random_state` and uses the same architecture and training configuration for the observed and target-permuted models.
