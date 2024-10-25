import numpy as np
import pytest  # noqa: F401
from numpy.typing import NDArray
from quantum_inspired_algorithms.quantum_inspired import compute_ls_probs
from quantum_inspired_algorithms.sketching import FKV
from quantum_inspired_algorithms.sketching import Halko


def _get_FKV_sketcher(A: NDArray[np.float64], r: int, c: int) -> FKV:
    """Load dummy sketcher."""
    A_ls_prob_rows, A_ls_prob_columns_2d, _, _, A_frobenius = compute_ls_probs(A)
    r = 30
    c = 40
    random_state = np.random.RandomState(7)
    return FKV(
        A,
        r,
        c,
        A_ls_prob_rows,
        A_ls_prob_columns_2d,
        A_frobenius,
        random_state,
    )


def _get_Halko_sketcher(A: NDArray[np.float64], r: int, c: int) -> Halko:
    """Load dummy sketcher."""
    _, _, A_ls_prob_columns, _, _ = compute_ls_probs(A)
    random_state = np.random.RandomState(7)
    return Halko(
        A,
        r,
        c,
        A_ls_prob_columns,
        random_state,
    )


def test_FKV_dimensions():
    """Test dimensions of sketches."""
    A = np.arange(100 * 100, dtype=np.float64).reshape((100, 100))
    r = 30
    c = 40
    sketcher = _get_FKV_sketcher(A, r, c)

    left_sketch_matrix = sketcher.left_project(A)
    right_sketch_matrix = sketcher.right_project(A)
    left_right_sketch_matrix = sketcher.right_project(sketcher.left_project(A))

    assert left_sketch_matrix.shape == (30, 100)
    assert right_sketch_matrix.shape == (100, 40)
    assert left_right_sketch_matrix.shape == (30, 40)


def test_Halko_dimensions():
    """Test dimensions of sketches."""
    A = np.arange(100 * 100, dtype=np.float64).reshape((100, 100))
    r = 30
    c = 40
    sketcher = _get_Halko_sketcher(A, r, c)

    left_sketch_matrix = sketcher.left_project(A)
    right_sketch_matrix = sketcher.right_project(A)
    left_right_sketch_matrix = sketcher.right_project(sketcher.left_project(A))

    assert left_sketch_matrix.shape == (r + 10, 100)
    assert right_sketch_matrix.shape == (100, r + 10)
    assert left_right_sketch_matrix.shape == (r + 10, r + 10)


def test_FKV_samplers():
    """Test FKV-based samplers."""
    A = np.arange(100 * 100, dtype=np.float64).reshape((100, 100))
    r = 30
    c = 40
    A_predict = np.arange(150 * 100, dtype=np.float64).reshape((150, 100))
    sketcher = _get_FKV_sketcher(A, r, c)

    sketcher.set_up_column_sampler(A)
    assert np.all(np.allclose(np.sum(sketcher._R_ls_prob_columns, axis=1), 1))
    assert sketcher._R_ls_prob_columns.shape == (r, A.shape[1])
    assert sketcher._n_cols == A.shape[1]

    sketcher.set_up_row_sampler(A_predict)
    assert np.all(np.allclose(np.sum(sketcher._C_ls_prob_rows, axis=0), 1))
    assert sketcher._C_ls_prob_rows.shape == (A_predict.shape[0], c)
    assert sketcher._n_rows == A_predict.shape[0]
