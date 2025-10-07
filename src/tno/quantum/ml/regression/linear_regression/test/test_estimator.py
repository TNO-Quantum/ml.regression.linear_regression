import logging
from typing import SupportsFloat

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import pinv
from numpy.typing import ArrayLike, NDArray

from tno.quantum.ml.components import check_estimator_serializable
from tno.quantum.ml.regression.linear_regression import QILinearEstimator

logging.basicConfig(
    format="%(levelname)s:%(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def _compute_n_matches(
    x_reference: ArrayLike,
    expected_top_idx: ArrayLike,
) -> int:
    """Computes the number of matches between `x_reference` and `expected_top_idx`.

    This function determines how many of the indices provided in `expected_top_idx` are
    also present among the indices corresponding to the `k` largest values in
    `x_reference`, where `k` is the number of indices in `expected_top_idx`.

    Args:
        x_reference: a 1-D array-like object of numerical values. The function will
                     consider the absolute values of this array.
        expected_top_idx: a 1-D array-like object of expected top indices.

    Returns:
        The number of indices that are present in both `expected_top_idx` and the top
        `k` indices of `x_reference`.
    """
    # Ensure inputs are numpy arrays for consistency
    x_reference = np.asarray(x_reference)
    expected_top_idx = np.asarray(expected_top_idx)

    # Determine k, the number of top elements to consider
    k = expected_top_idx.size
    if k == 0:
        return 0

    # Get the indices that would sort the absolute values of x_reference
    # in descending order and take the top k.
    top_k_indices = np.flip(np.argsort(np.abs(x_reference)))[:k]

    # Use set intersection to find the number of common indices
    return len(set(top_k_indices) & set(expected_top_idx))


def test_compute_n_matches() -> None:
    """Test computation of the number of matches."""
    x_reference = [0.1, 0.5, 0.2, 0.8]
    expected_top_idx = [1, 3]
    assert _compute_n_matches(x_reference, expected_top_idx) == 2

    x_reference = [10, 5, 2, 8]
    expected_top_idx = [0, 2]  # Expecting indices 0 and 2 to be the top.
    # The actual top 2 indices are 0 and 3 (for values 10 and 8).
    # Only index 0 is a match.
    assert _compute_n_matches(x_reference, expected_top_idx) == 1


def _load_data(
    underdetermined: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    x = np.asarray(rng.normal(0, 1, A.shape[1]), dtype=np.float64)
    b = A @ x

    return A, b, x


def test_solve_qi() -> None:
    """Test quantum-inspired linear solver."""
    # Load data
    A, b, _ = _load_data()

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    n_entries_x = 10
    rng = 7
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_x = qi.sample_prediction_x(A, n_entries_x)

    print(sampled_indices)  # noqa: T201
    print(sampled_x)  # noqa: T201
    assert np.all(
        sampled_indices == np.asarray([44, 127, 239, 228, 81, 19, 249, 3, 208, 72])
    )
    assert np.allclose(
        sampled_x,
        [
            -0.12064711,
            -0.07698437,
            -0.26102075,
            -0.12162569,
            0.08193343,
            0.14820182,
            -0.16637135,
            0.16646659,
            -0.21766188,
            -0.09149265,
        ],
    )


def test_solve_qi_b() -> None:
    """Test quantum-inspired linear solver to predict `b`."""
    # Load data
    A, b, _ = _load_data()

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    n_entries_b = 10
    rng = 7
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_b = qi.sample_prediction_b(A, n_entries_b)

    print(sampled_indices)  # noqa: T201
    print(sampled_b)  # noqa: T201
    assert np.all(
        sampled_indices == np.asarray([231, 461, 461, 381, 50, 334, 323, 381, 59, 174])
    )
    assert np.allclose(
        sampled_b,
        [
            5.44005314,
            4.49647611,
            4.49647611,
            -4.93110314,
            3.88793886,
            -5.33528434,
            2.32686741,
            -4.93110314,
            3.58986512,
            3.30552512,
        ],
    )


def test_solve_qi_ridge() -> None:
    """Test quantum-inspired ridge regression."""
    # Load data
    A, b, _ = _load_data()

    # Solve using quantum-inspired algorithm
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    n_entries_x = 10
    rng = 7

    def func(arg: SupportsFloat) -> SupportsFloat:
        arg = float(arg)
        return (arg**2 + 0.3) / arg

    qi = QILinearEstimator(r, c, rank, n_samples, rng, func=func)
    qi = qi.fit(A, b)
    sampled_indices, sampled_x = qi.sample_prediction_x(A, n_entries_x)

    print(sampled_indices)  # noqa: T201
    print(sampled_x)  # noqa: T201
    assert np.all(
        sampled_indices == np.asarray([44, 127, 239, 228, 81, 19, 249, 3, 208, 72])
    )
    assert np.allclose(
        sampled_x,
        [
            -0.12061417,
            -0.07696354,
            -0.26095131,
            -0.12159334,
            0.08191188,
            0.14816215,
            -0.16632772,
            0.1664221,
            -0.2176028,
            -0.09146953,
        ],
    )


@pytest.mark.parametrize(
    ("sketcher_name", "random_state", "n_matches_expected"),
    [("fkv", 111, 48), ("halko", 11, 49)],
)
def test_finding_largest_entries_b_underdetermined(
    sketcher_name: str, random_state: int, n_matches_expected: int
) -> None:
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
    rng = random_state
    qi = QILinearEstimator(r, c, rank, n_samples, rng, sketcher_name=sketcher_name)
    qi = qi.fit(A, b)
    sampled_indices, sampled_b = qi.sample_prediction_b(A, n_entries_b)

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})  # noqa: PD901
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    unique_sampled_indices = np.asarray(df_mean.keys())
    unique_sampled_b = np.asarray(df_mean.values)
    sort_idx = np.flip(np.argsort(np.abs(unique_sampled_b)))
    b_idx = unique_sampled_indices[sort_idx][:top_size]
    n_matches = _compute_n_matches(b, b_idx)

    assert n_matches == n_matches_expected


def test_finding_largest_entries_b() -> None:
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
    rng = 111
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_b = qi.sample_prediction_b(A, n_entries_b)

    # Compare results
    df = pd.DataFrame({"b_idx_samples": sampled_indices, "b_samples": sampled_b})  # noqa: PD901
    df_mean = df.groupby("b_idx_samples")["b_samples"].mean()
    unique_sampled_indices = np.asarray(df_mean.keys())
    unique_sampled_b = np.asarray(df_mean.values)
    sort_idx = np.flip(np.argsort(np.abs(unique_sampled_b)))
    b_idx = unique_sampled_indices[sort_idx][:top_size]
    n_matches = _compute_n_matches(b, b_idx)

    assert n_matches == 23


def test_pseudoinverse() -> None:
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
        lambdas.append(1 / (sigmas[ell]) ** 2 * np.sum(A * (np.outer(b, V[ell, :]))))  # noqa: PERF401
    x_sol3 = np.squeeze(
        A.T @ (U[:, :rank] @ (np.asarray(lambdas)[:, None] / sigmas[:, None]))
    )

    assert np.allclose(x_sol, x_sol2)
    assert np.allclose(x_sol, x_sol3)


def test_finding_largest_entries_x() -> None:
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
    rng = 7
    qi = QILinearEstimator(r, c, rank, n_samples, rng)
    qi = qi.fit(A, b)
    sampled_indices, sampled_x = qi.sample_prediction_x(A, n_entries_x)

    # Solve using pseudoinverse
    x_sol = pinv(A).dot(b)

    # Compare results
    df = pd.DataFrame({"x_idx_samples": sampled_indices, "x_samples": sampled_x})  # noqa: PD901
    df_mean = df.groupby("x_idx_samples")["x_samples"].mean()
    unique_sampled_indices = np.asarray(df_mean.keys())
    unique_sampled_x = np.asarray(df_mean.values)
    sort_idx = np.flip(np.argsort(np.abs(unique_sampled_x)))
    x_idx = unique_sampled_indices[sort_idx][:top_size]
    n_matches = _compute_n_matches(x_sol, x_idx)

    assert n_matches == 23


@pytest.mark.parametrize(
    "sketcher_name",
    ["fkv", "halko"],
)
def test_serialization(sketcher_name: str) -> None:
    """Test correct serialization of estimator."""
    # Load data
    A, b, _ = _load_data()

    # Define parameters
    rank = 3
    r = 70
    c = 70
    n_samples = 100
    rng = 7

    # Check serialization
    qi1 = QILinearEstimator(r, c, rank, n_samples, rng, sketcher_name=sketcher_name)
    check_estimator_serializable(qi1)

    qi2 = QILinearEstimator(r, c, rank, n_samples, rng, sketcher_name=sketcher_name)
    qi2 = qi2.fit(A, b)
    check_estimator_serializable(qi2)


if __name__ == "__main__":
    test_serialization(sketcher_name="fkv")
