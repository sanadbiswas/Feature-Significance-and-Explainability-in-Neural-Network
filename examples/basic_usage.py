"""Minimal example of the published neural target-permutation test."""

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
print("Significant at alpha=0.05:", result.significant_features(alpha=0.05))
