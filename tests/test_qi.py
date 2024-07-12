import logging
import numpy as np
import pandas as pd
import pytest
from numpy.linalg import pinv
from quantum_inspired_algorithms import quantum_inspired as qi
from quantum_inspired_algorithms.visualization import plot_solution

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def _load_data():
    """Load sample dataset."""
    rng = np.random.RandomState(9)
    rank = 3
    m = 500
    n = 250
    A = rng.normal(0, 1, (m, n))
    U, S, V = np.linalg.svd(A, full_matrices=False)
    S[rank:] = 0
    A = U @ np.diag(S) @ V
    x = rng.normal(0, 1, n)
    b = A @ x

    top_percent = 0.1
    size = A.shape[1]
    top_size = int(top_percent * size)

    return A, b, top_size


def test_solve_qi():
    """Test quantum-inspired linear solver."""
    # Load data
    A, b, _ = _load_data()

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    n_entries_x = 10
    rng = np.random.RandomState(7)
    sampled_indices, sampled_x = qi.solve_qi(A, b, r, c, rank, n_samples, n_entries_x, rng)
    assert np.all(sampled_indices == np.asarray([234, 106, 136, 54, 130, 36, 161, 150, 173, 32]))
    assert np.allclose(
        sampled_x,
        [
            -0.13825295,
            0.06593373,
            -0.08765071,
            -0.02747893,
            0.18578372,
            -0.23101082,
            0.17040479,
            0.12723163,
            0.19693845,
            0.21950633,
        ],
    )


def test_solve_qi_ridge():
    """Test quantum-inspired ridge regression."""
    # Load data
    A, b, _ = _load_data()

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    n_entries_x = 10
    rng = np.random.RandomState(7)
    func = lambda arg: (arg**2 + 0.3) / arg
    sampled_indices, sampled_x = qi.solve_qi(A, b, r, c, rank, n_samples, n_entries_x, rng, func=func)
    print(sampled_indices)
    print(sampled_x)

    assert np.all(sampled_indices == np.asarray([234, 106, 136, 54, 130, 36, 161, 150, 173, 32]))
    assert np.allclose(
        sampled_x,
        [
            -0.13821783,
            0.06591698,
            -0.08762728,
            -0.02747263,
            0.18573576,
            -0.23094832,
            0.17035964,
            0.12719866,
            0.1968863,
            0.21944595,
        ],
    )


def test_finding_largest_entries():
    """Test quantum-inspired least squares."""
    # Load data
    A, b, top_size = _load_data()

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    n_entries_x = 1000
    rng = np.random.RandomState(7)
    sampled_indices, sampled_x = qi.solve_qi(A, b, r, c, rank, n_samples, n_entries_x, rng)

    # Find most frequent outcomes
    unique_x_idx, counts = np.unique(sampled_indices, return_counts=True)
    sort_idx = np.flip(np.argsort(counts))
    x_idx = unique_x_idx[sort_idx][:top_size]

    # Solve using pseudoinverse
    x_sol = pinv(A).dot(b)

    # Compare results
    df = pd.DataFrame({"x_idx_samples": sampled_indices, "x_samples": sampled_x})
    x_entries = df.groupby("x_idx_samples")["x_samples"].mean().values
    n_matches = plot_solution(
        x_sol,
        x_idx,
        "test_finding_largest_entries",
        expected_solution=np.abs(x_entries),
        solution=np.abs(x_sol)[unique_x_idx],
        expected_counts=counts,
        counts=n_entries_x * np.abs(x_sol)[unique_x_idx] ** 2,
    )

    assert n_matches == 21


if __name__ == "__main__":
    test_finding_largest_entries()
