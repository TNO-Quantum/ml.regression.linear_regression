import numpy as np
import pytest  # noqa: F401

from tno.quantum.ml.regression.linear_regression._quantum_inspired import (
    compute_ls_probs,
)


def test_compute_ls_probs() -> None:
    """Test LS probabilities."""
    A = np.arange(100, dtype=np.float64).reshape((10, 10))

    A_ls_prob_rows, A_ls_prob_columns_2d, A_ls_prob_columns, A_ls_prob_rows_2d, _ = (
        compute_ls_probs(A)
    )

    tolerance = -1e-8

    assert np.allclose(np.sum(A_ls_prob_rows), 1)
    assert np.all(np.allclose(np.sum(A_ls_prob_columns_2d, axis=1), 1))
    assert np.all(A_ls_prob_rows > tolerance)
    assert np.all(A_ls_prob_columns_2d > tolerance)

    assert np.allclose(np.sum(A_ls_prob_columns), 1)
    assert np.all(np.allclose(np.sum(A_ls_prob_rows_2d, axis=0), 1))
    assert np.all(A_ls_prob_columns > tolerance)
    assert np.all(A_ls_prob_rows_2d > tolerance)


if __name__ == "__main__":
    test_compute_ls_probs()
