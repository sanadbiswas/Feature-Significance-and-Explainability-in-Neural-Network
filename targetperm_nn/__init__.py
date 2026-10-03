"""Neural-network feature significance using target permutation."""

from .models import NetworkConfig, build_tabular_network
from .significance import NeuralSignificanceResult, NeuralTargetPermutationTest

__all__ = [
    "NetworkConfig",
    "NeuralSignificanceResult",
    "NeuralTargetPermutationTest",
    "build_tabular_network",
]
