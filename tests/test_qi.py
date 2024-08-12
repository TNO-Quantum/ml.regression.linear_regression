import logging
import numpy as np
import pandas as pd
import pytest
from numpy.linalg import norm
from numpy.linalg import pinv
from quantum_inspired_algorithms import quantum_inspired as qi
from quantum_inspired_algorithms.visualization import plot_solution
from quantum_inspired_algorithms.estimator import QILinearEstimator

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def _load_data(underdetermined: bool = False):
    """Load sample dataset."""
    rng = np.random.RandomState(9)
    rank = 3
    m = 500
    n = 250
    A = rng.normal(0, 1, (m, n))
    U, S, V = np.linalg.svd(A, full_matrices=False)
    S[rank:] = 0
    A = U @ np.diag(S) @ V
    if underdetermined:
        A = A.T
    x = rng.normal(0, 1, A.shape[1])
    b = A @ x

    top_percent = 0.1
    top_size = int(top_percent * A.shape[1])

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
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_x = qi.predict_x(A, n_entries_x)
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


def test_solve_qi_b():
    """Test quantum-inspired linear solver to predict `b`."""
    # Load data
    A, b, _ = _load_data()

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    n_entries_b = 10
    rng = np.random.RandomState(7)
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_b = qi.predict_b(A, n_entries_b)
    assert np.all(sampled_indices == np.asarray([156, 366, 487, 293, 170, 145, 330, 302, 431, 277]))
    assert np.allclose(
        sampled_b,
        [
            -6.53513244,
            -4.44876807,
            -1.75511705,
            -1.89623062,
            2.05828132,
            -1.69277884,
            -3.94735972,
            -4.23570172,
            4.34278576,
            3.62448208,
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
    n_entries_b = 0
    rng = np.random.RandomState(7)
    func = lambda arg: (arg**2 + 0.3) / arg
    # qi = QILinearEstimator(r, c, rank, n_samples, rng, func=func)
    # qi = qi.fit(A, b)
    # sampled_indices, sampled_x = qi.predict_b(A, n_entries_x)
    sampled_indices, sampled_x, _, _ = qi.solve_qi(
        A, b, r, c, rank, n_samples, n_entries_x, n_entries_b, rng, func=func
    )
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


def test_finding_largest_entries_b_underdetermined():
    """Test quantum-inspired least squares."""
    # Load data
    A, b, top_size = _load_data(underdetermined=True)

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 80
    n_samples = 100
    n_entries_x = 0
    n_entries_b = 1000
    rng = np.random.RandomState(111)
    _, _, sampled_indices, sampled_b = qi.solve_qi(A, b, r, c, rank, n_samples, n_entries_x, n_entries_b, rng)

    # Find most frequent outcomes
    unique_b_idx, counts = np.unique(sampled_indices, return_counts=True)
    sort_idx = np.flip(np.argsort(counts))
    b_idx = unique_b_idx[sort_idx][:top_size]

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    df_counts = df.groupby("b_idx_samples").count()
    unique_sampled_indices = df_mean.keys()
    unique_sampled_b = df_mean.values
    n_matches = plot_solution(
        b,
        b_idx,
        "test_finding_largest_entries_b_underdetermined",
        expected_solution=np.abs(b)[unique_sampled_indices],
        solution=np.abs(unique_sampled_b),
        expected_counts=n_entries_b * np.abs(b / norm(b))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 40


def test_finding_largest_entries_b():
    """Test quantum-inspired least squares."""
    # Load data
    A, b, top_size = _load_data()

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 80
    n_samples = 100
    n_entries_x = 0
    n_entries_b = 1000
    rng = np.random.RandomState(111)
    _, _, sampled_indices, sampled_b = qi.solve_qi(A, b, r, c, rank, n_samples, n_entries_x, n_entries_b, rng)

    # Find most frequent outcomes
    unique_b_idx, counts = np.unique(sampled_indices, return_counts=True)
    sort_idx = np.flip(np.argsort(counts))
    b_idx = unique_b_idx[sort_idx][:top_size]

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    df_counts = df.groupby("b_idx_samples").count()
    unique_sampled_indices = df_mean.keys()
    unique_sampled_b = df_mean.values
    n_matches = plot_solution(
        b,
        b_idx,
        "test_finding_largest_entries_b",
        expected_solution=np.abs(b)[unique_sampled_indices],
        solution=np.abs(unique_sampled_b),
        expected_counts=n_entries_b * np.abs(b / norm(b))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 16


def test_finding_largest_entries_x():
    """Test quantum-inspired least squares."""
    # Load data
    A, b, top_size = _load_data()

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    n_entries_x = 1000
    n_entries_b = 0
    rng = np.random.RandomState(7)
    sampled_indices, sampled_x, _, _ = qi.solve_qi(A, b, r, c, rank, n_samples, n_entries_x, n_entries_b, rng)

    # Find most frequent outcomes
    unique_x_idx, counts = np.unique(sampled_indices, return_counts=True)
    sort_idx = np.flip(np.argsort(counts))
    x_idx = unique_x_idx[sort_idx][:top_size]

    # Solve using pseudoinverse
    x_sol = pinv(A).dot(b)

    # Compare results
    df = pd.DataFrame({"x_idx_samples": sampled_indices, "x_samples": sampled_x})
    df_mean = df.groupby("x_idx_samples")["x_samples"].mean()
    df_counts = df.groupby("x_idx_samples").count()
    unique_sampled_indices = df_mean.keys()
    unique_sampled_x = df_mean.values
    n_matches = plot_solution(
        x_sol,
        x_idx,
        "test_finding_largest_entries_x",
        expected_solution=np.abs(x_sol)[unique_sampled_indices],
        solution=np.abs(unique_sampled_x),
        expected_counts=n_entries_x * np.abs(x_sol / norm(x_sol))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 21


if __name__ == "__main__":
    test_finding_largest_entries_b()
    test_finding_largest_entries_x()
