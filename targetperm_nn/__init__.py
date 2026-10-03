"""Target-permutation feature significance testing for neural networks.

This package is a clean reference implementation of the method described in:
Biswas et al. (2025), "A Target Permutation Test for Statistical Significance
of Feature Importance in Differentiable Models."
"""

from .models import NetworkConfig, build_tabular_network
from .significance import NeuralSignificanceResult, NeuralTargetPermutationTest

__all__ = [
    "NetworkConfig",
    "NeuralSignificanceResult",
    "NeuralTargetPermutationTest",
    "build_tabular_network",
]

__version__ = "0.1.0"
