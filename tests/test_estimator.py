import copy
import logging
from typing import Any
import numpy as np
import pandas as pd
import pytest  # noqa: F401
from numpy.linalg import norm
from numpy.linalg import pinv
from numpy.typing import NDArray
from quantum_inspired_algorithms.estimator import QILinearEstimator
from quantum_inspired_algorithms.visualization import plot_solution

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def _load_data(underdetermined: bool = False) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load sample dataset."""
    rng = np.random.RandomState(9)
    rank = 3
    m = 500
    n = 250
    A = rng.randint(low=-1, high=2, size=(m, n))
    U, S, V = np.linalg.svd(A, full_matrices=False)
    S[rank:] = 0
    A = U @ np.diag(S) @ V
    if underdetermined:
        A = A.T
    x = rng.normal(0, 1, A.shape[1])
    b = A @ x

    return A, b, x


def _normalize(array: NDArray[Any]) -> NDArray[np.float64]:
    return array / norm(array)


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

    print(sampled_indices)
    print(sampled_x)
    assert np.all(sampled_indices == np.asarray([7, 231, 130, 140, 151, 40, 232, 228, 203, 63]))
    assert np.allclose(
        sampled_x,
        [
            0.03417937,
            -0.08006763,
            -0.11710657,
            -0.07868093,
            -0.10553537,
            -0.04694688,
            -0.10450383,
            0.03896918,
            0.03279467,
            -0.08570982,
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

    print(sampled_indices)
    print(sampled_b)
    assert np.all(sampled_indices == np.asarray([416, 359, 326, 200, 287, 295, 241, 374, 444, 305]))
    assert np.allclose(
        sampled_b,
        [
            1.32662421,
            -3.13238187,
            1.20473695,
            1.92923047,
            0.76955249,
            1.39890586,
            0.82789967,
            -3.11694485,
            1.21146775,
            -1.49789864,
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

    def func(arg: float) -> float:
        return (arg**2 + 0.3) / arg

    qi = QILinearEstimator(r, c, rank, n_samples, rng, func=func)
    qi = qi.fit(A, b)
    sampled_indices, sampled_x = qi.predict_x(A, n_entries_x)

    print(sampled_indices)
    print(sampled_x)
    assert np.all(sampled_indices == np.asarray([7, 231, 130, 140, 151, 40, 232, 228, 203, 63]))
    assert np.allclose(
        sampled_x,
        [
            0.03416916,
            -0.08004339,
            -0.11707113,
            -0.07865541,
            -0.10550235,
            -0.04693136,
            -0.10447066,
            0.03895664,
            0.03278557,
            -0.0856838,
        ],
    )


def test_finding_largest_entries_b_underdetermined():
    """Test quantum-inspired least squares."""
    # Load data
    A, b, _ = _load_data(underdetermined=True)
    top_size = 50

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 80
    n_samples = 100
    n_entries_b = 1000
    rng = np.random.RandomState(111)
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_b = qi.predict_b(A, n_entries_b)

    # Find most frequent outcomes
    unique_b_idx, counts = np.unique(sampled_indices, return_counts=True)
    sort_idx = np.flip(np.argsort(counts))
    b_idx = unique_b_idx[sort_idx][:top_size]

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    df_counts = df.groupby("b_idx_samples").count()
    unique_sampled_indices = df_mean.keys()
    unique_sampled_b = np.asarray(df_mean.values)
    n_matches = plot_solution(
        b,
        b_idx,
        "test_finding_largest_entries_b_underdetermined",
        expected_solution=_normalize(b)[unique_sampled_indices],
        solution=_normalize(unique_sampled_b),
        expected_counts=n_entries_b * np.abs(b / norm(b))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 41


def test_finding_largest_entries_b_underdetermined_fixed():
    """Test quantum-inspired least squares with fixed columns."""
    # Generate data
    A, b, x = _load_data(underdetermined=True)
    top_size = 25

    fixed_columns_idx = list(range(2))
    rng = np.random.RandomState(9)
    A[:, fixed_columns_idx] = rng.randint(low=1, high=3, size=(A.shape[0], len(fixed_columns_idx)))
    A = A / norm(A, axis=0)[None, :]
    x[fixed_columns_idx] = 80
    b = A @ x
    x[fixed_columns_idx] = 0
    b_without_fixed = A @ x

    # Solve using pseudoinverse
    A_zeroed_fixed_columns = copy.deepcopy(A)
    A_zeroed_fixed_columns[:, fixed_columns_idx] = 0
    b_without_fixed_pinv = A @ pinv(A_zeroed_fixed_columns).dot(b)
    b_without_fixed_pinv_idx = np.flip(np.argsort(np.abs(b_without_fixed_pinv)))[:top_size]

    # Compare results
    n_matches = plot_solution(
        b_without_fixed,
        b_without_fixed_pinv_idx,
        "test_finding_largest_entries_b_underdetermined_fixed_pinv",
        expected_solution=_normalize(b_without_fixed),
        solution=_normalize(b_without_fixed_pinv),
    )

    assert n_matches == 8

    # Solve using quantum-inspired algorithm (ignore fixed columns)
    rank = 3
    r = 70
    c = 80
    n_samples = 100
    n_entries_b = 1000
    rng = np.random.RandomState(111)
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_b = qi.predict_b(A, n_entries_b)

    # Find most frequent outcomes
    unique_b_idx, counts = np.unique(sampled_indices, return_counts=True)
    sort_idx = np.flip(np.argsort(counts))
    b_idx = unique_b_idx[sort_idx][:top_size]

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    unique_sampled_indices = df_mean.keys()
    unique_sampled_b = np.asarray(df_mean.values)
    n_matches = plot_solution(
        b_without_fixed,
        b_idx,
        "test_finding_largest_entries_b_underdetermined_fixed_qi_ignore_fixed",
        expected_solution=_normalize(b_without_fixed[unique_sampled_indices]),
        solution=_normalize(unique_sampled_b),
    )

    assert n_matches == 7

    # Solve using quantum-inspired algorithm (with fixed columns)
    rank = 3
    r = 70
    c = 80
    n_samples = 100
    n_entries_b = 1000
    rng = np.random.RandomState(111)
    qi = QILinearEstimator(r, c, rank, n_samples, rng, fixed_columns_idx=fixed_columns_idx)
    qi = qi.fit(A, b)
    sampled_indices, sampled_b = qi.predict_b(A, n_entries_b)

    # Find most frequent outcomes
    unique_b_idx, counts = np.unique(sampled_indices, return_counts=True)
    sort_idx = np.flip(np.argsort(counts))
    b_idx = unique_b_idx[sort_idx][:top_size]

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    unique_sampled_indices = df_mean.keys()
    unique_sampled_b = np.asarray(df_mean.values)
    n_matches = plot_solution(
        b_without_fixed,
        b_idx,
        "test_finding_largest_entries_b_underdetermined_fixed_qi",
        expected_solution=_normalize(b_without_fixed[unique_sampled_indices]),
        solution=_normalize(unique_sampled_b),
    )

    assert n_matches == 16


def test_finding_largest_entries_b():
    """Test quantum-inspired least squares."""
    # Load data
    A, b, _ = _load_data()
    top_size = 25

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 80
    n_samples = 100
    n_entries_b = 1000
    rng = np.random.RandomState(111)
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_b = qi.predict_b(A, n_entries_b)

    # Find most frequent outcomes
    unique_b_idx, counts = np.unique(sampled_indices, return_counts=True)
    sort_idx = np.flip(np.argsort(counts))
    b_idx = unique_b_idx[sort_idx][:top_size]

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    df_counts = df.groupby("b_idx_samples").count()
    unique_sampled_indices = df_mean.keys()
    unique_sampled_b = np.asarray(df_mean.values)
    n_matches = plot_solution(
        b,
        b_idx,
        "test_finding_largest_entries_b",
        expected_solution=_normalize(b)[unique_sampled_indices],
        solution=_normalize(unique_sampled_b),
        expected_counts=n_entries_b * np.abs(b / norm(b))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 17


def test_finding_largest_entries_x():
    """Test quantum-inspired least squares."""
    # Load data
    A, b, _ = _load_data()
    top_size = 25

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    n_entries_x = 1000
    rng = np.random.RandomState(7)
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_x = qi.predict_x(A, n_entries_x)

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
    unique_sampled_x = np.asarray(df_mean.values)
    n_matches = plot_solution(
        x_sol,
        x_idx,
        "test_finding_largest_entries_x",
        expected_solution=_normalize(x_sol)[unique_sampled_indices],
        solution=_normalize(unique_sampled_x),
        expected_counts=n_entries_x * np.abs(x_sol / norm(x_sol))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 20


if __name__ == "__main__":
    test_finding_largest_entries_b_underdetermined_fixed()
