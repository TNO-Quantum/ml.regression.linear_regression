import logging
from typing import Callable
import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray


def compute_ls_probs(
    A: NDArray[np.float64],
    ignore_columns_idx: list[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute length-square (LS) probability distributions for sampling `A`.

    Args:
        A: coefficient matrix.
        ignore_columns_idx: indices of columns whose LS probability should be forced to be zero.

    Returns:
        LS probability distribution for rows,
        LS probability distribution for columns,
        row norms,
        column norms,
        Frobenius norm
    """
    if ignore_columns_idx:
        # Compute row norms
        mask = np.ones(A.shape[1], dtype=bool)
        mask[ignore_columns_idx] = False
        A_row_norms = la.norm(A[:, mask], axis=1)
        A_row_norms_squared = A_row_norms**2

        # Compute column norms
        A_column_norms = la.norm(A, axis=0)
        A_column_norms[ignore_columns_idx] = 0

        # Compute Frobenius norm
        A_frobenius = np.sqrt(np.sum(A_row_norms_squared))

        # Compute LS probabilities for rows
        A_ls_prob_rows = A_row_norms_squared / A_frobenius**2

        # Compute LS probabilities for columns
        A_ls_prob_columns = A**2 / A_row_norms_squared[:, None]
        A_ls_prob_columns[:, ignore_columns_idx] = 0
    else:
        # Compute row norms
        A_row_norms = la.norm(A, axis=1)
        A_row_norms_squared = A_row_norms**2

        # Compute column norms
        A_column_norms = la.norm(A, axis=0)

        # Compute Frobenius norm
        A_frobenius = np.sqrt(np.sum(A_row_norms_squared))

        # Compute LS probabilities for rows
        A_ls_prob_rows = A_row_norms_squared / A_frobenius**2

        # Compute LS probabilities for columns
        A_ls_prob_columns = A**2 / A_row_norms_squared[:, None]

    return A_ls_prob_rows, A_ls_prob_columns, A_row_norms, A_column_norms, A_frobenius


def compute_C_and_R(
    A: NDArray[np.float64],
    r: int,
    c: int,
    A_row_norms: NDArray[np.float64],
    A_ls_prob_rows: NDArray[np.float64],
    A_ls_prob_columns: NDArray[np.float64],
    A_frobenius: NDArray[np.float64],
    rng: np.random.RandomState,
    A_fixed_columns_idx: list[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.uint32], NDArray[np.uint32]]:
    """Compute matrices `C` and `R` by sampling rows and columns of matrix `A`.

    Note: LS stands for length-square.

    Args:
        A: coefficient matrix.
        r: number of rows to sample.
        c: number of columns to sample.
        A_row_norms: norm of the rows of `A`.
        A_ls_prob_rows: row LS probability distribution of `A`.
        A_ls_prob_columns: column LS probability distribution of `A`.
        A_frobenius: Frobenius norm of `A`.
        rng: random state.
        A_fixed_columns_idx: indices of columns that should always be considered.
                             Columns excluded from this list will be randomly selected.

    Returns:
        matrix `C`,
        matrix `R`,
        column LS probability distribution for matrix `R`,
        sampled row indices,
        sampled column indices
    """
    m_rows, n_cols = A.shape

    # Sample row indices
    A_sampled_rows_idx = rng.choice(m_rows, r, replace=True, p=A_ls_prob_rows).astype(np.uint32)

    # Sample column indices
    A_sampled_columns_idx = np.zeros(c + len(A_fixed_columns_idx), dtype=np.uint32)
    for j in range(c):
        # Sample row index uniformly at random
        i = rng.choice(A_sampled_rows_idx, replace=True)

        # Sample column from LS distribution of row `i`
        A_sampled_columns_idx[j] = rng.choice(n_cols, 1, p=A_ls_prob_columns[i, :])[0]

    if A_fixed_columns_idx:
        A_sampled_columns_idx[c:] = A_fixed_columns_idx

    # Discard duplicates and sort in asc order
    A_sampled_rows_idx = np.unique(A_sampled_rows_idx)
    A_sampled_columns_idx = np.unique(A_sampled_columns_idx)

    # Recompute `r` and `c`
    r = len(A_sampled_rows_idx)
    c = len(A_sampled_columns_idx)

    # Build `R`
    R = A[A_sampled_rows_idx, :] * A_frobenius / (np.sqrt(r) * A_row_norms[A_sampled_rows_idx, None])

    # Build `C`
    C = R[:, A_sampled_columns_idx] * A_frobenius / (np.sqrt(c) * la.norm(R[:, A_sampled_columns_idx], axis=0))

    # Build LS distribution to sample columns from matrix `R`
    R_ls_prob_columns = R**2 / la.norm(R, axis=1)[:, None] ** 2

    return C, R, R_ls_prob_columns, A_sampled_rows_idx, A_sampled_columns_idx


def estimate_lambdas(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    n_samples: int,
    rank: int,
    w: NDArray[np.float64],
    sigma: NDArray[np.float64],
    A_sampled_rows_idx: NDArray[np.uint32],
    A_row_norms: NDArray[np.float64],
    A_ls_prob_rows: NDArray[np.float64],
    A_ls_prob_columns: NDArray[np.float64],
    A_frobenius: NDArray[np.float64],
    rng: np.random.RandomState,
    func: Callable[[float], float],
    A_fixed_columns_idx: list[int],
) -> NDArray[np.float64]:
    """Estimate lambda coefficients.

    Args:
        A: coefficient matrix.
        b: vector b.
        n_samples: number of samples to estimate inner products.
                   Note: the sampling is  performed from entries of `A`,
                   so there are `A.shape[0] * A.shape[1]` possible entries.
        rank: rank used to approximate matrix `A`.
        w: left-singular vector of `C`.
        sigma: singular values of `C`.
        A_sampled_rows_idx: indices of the `r` sampled rows of matrix A
        A_row_norms: norm of the rows of `A`.
        A_ls_prob_rows: row LS probability distribution of `A`.
        A_ls_prob_columns: column LS probability distribution of `A`.
        A_frobenius: Frobenius norm of `A`.
        rng: random state.
        func: function to transform singular values when estimating lambda coefficients.
        A_fixed_columns_idx: indices of columns that should always be considered.
                             Columns excluded from this list will be randomly selected.

    Returns:
        lambda coefficients
    """
    m_rows, n_cols = A.shape
    r = len(A_sampled_rows_idx)
    n_realizations = 10
    lambdas_realizations = np.zeros((n_realizations, rank))
    for realization_i in range(n_realizations):
        logging.info(f"---Realization {realization_i}")
        for ell in range(rank):
            # 1. Generate sample indices
            samples_i = []
            samples_j = []
            if not A_fixed_columns_idx:
                for _ in range(n_samples):
                    i = rng.choice(m_rows, 1, replace=True, p=A_ls_prob_rows)[0]
                    j = rng.choice(n_cols, 1, p=A_ls_prob_columns[i])[0]
                    samples_i.append(i)
                    samples_j.append(j)
            else:
                percentage_of_fixed = len(A_fixed_columns_idx) / A.shape[1]
                n_samples_nonfixed = int((1 - percentage_of_fixed) * n_samples)
                for _ in range(n_samples_nonfixed):
                    i = rng.choice(m_rows, 1, replace=True, p=A_ls_prob_rows)[0]
                    j = rng.choice(n_cols, 1, p=A_ls_prob_columns[i])[0]
                    samples_i.append(i)
                    samples_j.append(j)

                n_samples_per_fixed_column = int(percentage_of_fixed * n_samples / len(A_fixed_columns_idx))
                for j in A_fixed_columns_idx:
                    for _ in range(n_samples_per_fixed_column):
                        i = rng.choice(m_rows, 1, replace=True, p=A_ls_prob_rows)[0]
                    samples_i.append(i)
                    samples_j.append(j)

            # 2. Approximate lambda using Monte Carlo estimation

            # Estimate right-singular vector
            R = (
                A[A_sampled_rows_idx[:, None], np.asarray(samples_j)[None, :]]
                * A_frobenius
                / (np.sqrt(r) * A_row_norms[A_sampled_rows_idx, None])
            )
            v_approx = R.T @ (w[:, ell] / sigma[ell])

            # Compute entries of outer product between `b` and `v_approx`
            outer_prod_b_v = np.squeeze(b[samples_i]) * v_approx

            # Estimate inner product between `A` and `outer_prod_b_v`
            inner_prod = np.mean(A_frobenius**2 / A[samples_i, samples_j] * outer_prod_b_v)

            # Compute lambda
            lambdas_realizations[realization_i, ell] = inner_prod / sigma[ell] / func(sigma[ell])

    lambdas = np.median(lambdas_realizations, axis=0)

    return lambdas


def sample_from_b(
    A: NDArray[np.float64],
    A_sampled_columns_idx: NDArray[np.uint32],
    A_column_norms: NDArray[np.float64],
    A_ls_prob_rows: NDArray[np.float64],
    A_frobenius: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_norm: float,
    rng: np.random.RandomState,
) -> tuple[int, float]:
    """Perform length-square (LS) sampling from the predicted `b`.

    Args:
        A: coefficient matrix.
        A_sampled_columns_idx: indices of the `c` sampled rows of matrix A.
        A_column_norms: norm of the columns of `A`.
        A_ls_prob_rows: row LS probability distribution of `A`.
        A_frobenius: Frobenius norm of `A`.
        phi: vector phi.
        phi_norm: norm of `phi`.
        rng: random state.

    Returns:
        index of the sampled entry,
        entry value
    """
    m_rows = A.shape[0]
    c = len(A_sampled_columns_idx)
    C = A[:, A_sampled_columns_idx] * A_frobenius / (np.sqrt(c) * A_column_norms[None, A_sampled_columns_idx])

    while True:
        # Sample column index uniformly at random
        sample_i = rng.choice(m_rows, 1, replace=True, p=A_ls_prob_rows)[0]

        # Sample row of `C`
        C_i = C[sample_i, :]
        C_i_norm = la.norm(C_i)

        # Determine if we output `sample_i`
        dot_prod_C_i_omega = np.dot(C_i, phi)
        prob = (dot_prod_C_i_omega / (phi_norm * C_i_norm)) ** 2
        if rng.binomial(1, prob) == 1:
            return sample_i, dot_prod_C_i_omega


def sample_from_x(
    A: NDArray[np.float64],
    A_sampled_rows_idx: NDArray[np.uint32],
    A_row_norms: NDArray[np.float64],
    R_ls_prob_columns: NDArray[np.float64],
    A_frobenius: NDArray[np.float64],
    omega: NDArray[np.float64],
    omega_norm: float,
    rng: np.random.RandomState,
) -> tuple[int, float]:
    """Perform length-square (LS) sampling from the solution vector.

    Args:
        A: coefficient matrix.
        A_sampled_rows_idx: indices of the `r` sampled rows of matrix A.
        A_row_norms: norm of the rows of `A`.
        R_ls_prob_columns: column LS probability distribution of `R`.
        A_frobenius: Frobenius norm of `A`.
        omega: vector omega.
        omega_norm: norm of `omega`.
        rng: random state.

    Returns:
        index of the sampled entry,
        entry value
    """
    n_cols = A.shape[1]
    r = len(A_sampled_rows_idx)
    R = A[A_sampled_rows_idx, :] * A_frobenius / (np.sqrt(r) * A_row_norms[A_sampled_rows_idx, None])

    while True:
        # Sample row index uniformly at random
        sample_i = rng.choice(r)

        # Sample column index from LS distribution of corresponding row
        sample_j = rng.choice(n_cols, 1, p=R_ls_prob_columns[sample_i])[0]

        # Sample column of `R`
        R_j = R[:, sample_j]
        R_j_norm = la.norm(R_j)

        # Determine if we output `sample_j`
        dot_prod_R_j_omega = np.dot(R_j, omega)
        prob = (dot_prod_R_j_omega / (omega_norm * R_j_norm)) ** 2
        if rng.binomial(1, prob) == 1:
            return sample_j, dot_prod_R_j_omega
