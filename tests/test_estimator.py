import logging
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
    A = rng.normal(0, 1, (m, n))
    U, S, V = np.linalg.svd(A, full_matrices=False)
    S[rank:] = 0
    A = U @ np.diag(S) @ V
    if underdetermined:
        A = A.T
    x = rng.normal(0, 1, A.shape[1])
    b = A @ x

    return A, b, x


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

    print(sampled_indices)
    print(sampled_b)
    assert np.all(sampled_indices == np.asarray([91, 295, 326, 83, 282, 55, 447, 268, 156, 393]))
    assert np.allclose(
        sampled_b,
        [
            -3.18036958,
            -1.12204103,
            -4.5962418,
            -1.18912406,
            -4.92860809,
            3.69603232,
            -2.04046866,
            7.99696582,
            -6.82906746,
            -3.55111518,
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


@pytest.mark.parametrize("sketcher_name,n_matches_expected", [("fkv", 48), ("halko", 49)])
def test_finding_largest_entries_b_underdetermined(sketcher_name: str, n_matches_expected: int):
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
    qi = QILinearEstimator(r, c, rank, n_samples, rng, sketcher_name=sketcher_name)
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
        f"test_finding_largest_entries_b_underdetermined_{sketcher_name}",
        expected_solution=b[unique_sampled_indices],
        solution=unique_sampled_b,
        expected_counts=n_entries_b * np.abs(b / norm(b))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == n_matches_expected


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
        expected_solution=b[unique_sampled_indices],
        solution=unique_sampled_b,
        expected_counts=n_entries_b * np.abs(b / norm(b))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 23


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
        expected_solution=x_sol[unique_sampled_indices],
        solution=unique_sampled_x,
        expected_counts=n_entries_x * np.abs(x_sol / norm(x_sol))[unique_sampled_indices] ** 2,
        counts=np.squeeze(np.round(df_counts.values)),
    )

    assert n_matches == 23


if __name__ == "__main__":
    test_solve_qi_b()
