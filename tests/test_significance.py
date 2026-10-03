import numpy as np

from targetperm_nn.significance import empirical_p_values


def test_empirical_p_values_match_paper_formula():
    observed = np.array([2.0, 1.0])
    null = np.array(
        [
            [3.0, 0.5],
            [1.0, 2.0],
            [4.0, 0.2],
            [0.0, 3.0],
        ]
    )

    p_values = empirical_p_values(observed, null)
    np.testing.assert_allclose(p_values, np.array([0.5, 0.5]))


def test_empirical_p_values_reject_bad_shape():
    observed = np.array([1.0, 2.0])
    null = np.array([1.0, 2.0])

    try:
        empirical_p_values(observed, null)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for one-dimensional null statistics.")
