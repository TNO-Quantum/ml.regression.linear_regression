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

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    df_counts = df.groupby("b_idx_samples").count()
    unique_sampled_indices = np.asarray(df_mean.keys())
    unique_sampled_b = np.asarray(df_mean.values)
    sort_idx = np.flip(np.argsort(np.abs(unique_sampled_b)))
    b_idx = unique_sampled_indices[sort_idx][:top_size]
    n_matches = plot_solution(
        b,
        b_idx,
        "test_finding_largest_entries_b_underdetermined",
        expected_solution=_normalize(b)[unique_sampled_indices],
        solution=_normalize(unique_sampled_b),
        expected_counts=n_entries_b * np.abs(b / norm(b))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 43


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

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    df_counts = df.groupby("b_idx_samples").count()
    unique_sampled_indices = np.asarray(df_mean.keys())
    unique_sampled_b = np.asarray(df_mean.values)
    sort_idx = np.flip(np.argsort(np.abs(unique_sampled_b)))
    b_idx = unique_sampled_indices[sort_idx][:top_size]
    n_matches = plot_solution(
        b,
        b_idx,
        "test_finding_largest_entries_b",
        expected_solution=_normalize(b)[unique_sampled_indices],
        solution=_normalize(unique_sampled_b),
        expected_counts=n_entries_b * np.abs(b / norm(b))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 24


def test_pseudoinverse():
    """Test pseudoinverse."""
    # Load data
    A, b, _ = _load_data()
    rank = 3

    # Solve using pseudoinverse
    x_sol = pinv(A).dot(b)

    # Solve using pseudoinverse II
    U, S, V = np.linalg.svd(A, full_matrices=False)
    A_pinv = V[:rank, :].T @ (np.diag(1 / S[:rank])) @ U[:, :rank].T
    x_sol2 = A_pinv @ b

    # Solve using pseudoinverse III
    sigmas = S[:rank]
    lambdas = []
    for ell in range(rank):
        lambdas.append(1 / (sigmas[ell]) ** 2 * np.sum(A * (np.outer(b, V[ell, :]))))
    x_sol3 = np.squeeze(A.T @ (U[:, :rank] @ (np.asarray(lambdas)[:, None] / sigmas[:, None])))

    assert np.allclose(x_sol, x_sol2)
    assert np.allclose(x_sol, x_sol3)


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

    # Solve using pseudoinverse
    x_sol = pinv(A).dot(b)

    # Compare results
    df = pd.DataFrame({"x_idx_samples": sampled_indices, "x_samples": sampled_x})
    df_mean = df.groupby("x_idx_samples")["x_samples"].mean()
    df_counts = df.groupby("x_idx_samples").count()
    unique_sampled_indices = np.asarray(df_mean.keys())
    unique_sampled_x = np.asarray(df_mean.values)
    sort_idx = np.flip(np.argsort(np.abs(unique_sampled_x)))
    x_idx = unique_sampled_indices[sort_idx][:top_size]
    n_matches = plot_solution(
        x_sol,
        x_idx,
        "test_finding_largest_entries_x",
        expected_solution=_normalize(x_sol)[unique_sampled_indices],
        solution=_normalize(unique_sampled_x),
        expected_counts=n_entries_x * np.abs(x_sol / norm(x_sol))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 21


if __name__ == "__main__":
    test_finding_largest_entries_b()
