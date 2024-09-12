import numpy as np
import pytest  # noqa: F401
from quantum_inspired_algorithms.quantum_inspired import compute_ls_probs


def test_compute_ls_probs():
    """Test LS probabilities."""
    A = np.arange(100, dtype=np.float64).reshape((10, 10))

    A_ls_prob_rows, A_ls_prob_columns, _, _, _ = compute_ls_probs(A)
    assert np.allclose(np.sum(A_ls_prob_rows), 1)
    assert np.all(np.allclose(np.sum(A_ls_prob_columns, axis=1), 1))
    assert np.all(A_ls_prob_rows > -1e-8)
    assert np.all(A_ls_prob_columns > -1e-8)


if __name__ == "__main__":
    test_compute_ls_probs()
