import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray


def compute_ls_probs(A) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute length-square (LS) probability distributions for sampling `A`.

    Args:
        A: coefficient matrix.

    Returns:
        LS probability distribution for rows,
        LS probability distribution for columns,
        row norms,
        Frobenius norm
    """
    # Compute row norms squared
    A_row_norms = la.norm(A, axis=1)
    A_row_norms_squared = A_row_norms**2

    # Compute Frobenius norm
    A_frobenius = np.sqrt(np.sum(A_row_norms_squared))

    # Compute LS probabilities for rows
    A_ls_prob_rows = A_row_norms_squared / A_frobenius**2

    # Compute LS probabilities for columns
    A_ls_prob_columns = A**2 / A_row_norms_squared[:, None]

    return A_ls_prob_rows, A_ls_prob_columns, A_row_norms, A_frobenius


def compute_C_and_R(
    A: NDArray,
    r: int,
    c: int,
    A_row_norms: NDArray,
    A_ls_prob_rows: NDArray,
    A_ls_prob_columns: NDArray,
    A_frobenius: NDArray,
    rng: np.random.RandomState,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
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

    Returns:
        matrix `C`,
        matrix `R`,
        column LS probability distribution for matrix `R`,
        sampled row indices
    """
    m_rows, n_cols = A.shape

    # Sample row indices
    A_sampled_rows_idx = rng.choice(m_rows, r, replace=True, p=A_ls_prob_rows)

    # Sample column indices
    sampled_columns_idx = np.zeros(c, dtype=np.uint32)
    for j in range(c):
        # Sample row index uniformly at random
        i = rng.choice(A_sampled_rows_idx, replace=True)

        # Sample column from LS distribution of row `A[i]`
        sampled_columns_idx[j] = rng.choice(n_cols, 1, p=A_ls_prob_columns[i])

    # Build `R`
    R = (
        A[A_sampled_rows_idx, :]
        * A_frobenius
        / (np.sqrt(r) * A_row_norms[A_sampled_rows_idx, None])
    )

    # Build `C`
    C = (
        R[:, sampled_columns_idx]
        * A_frobenius
        / (np.sqrt(c) * la.norm(R[:, sampled_columns_idx], axis=0))
    )

    # Build LS distribution to sample columns from matrix `R`
    R_ls_prob_columns = R**2 / la.norm(R, axis=1)[:, None] ** 2

    return C, R, R_ls_prob_columns, A_sampled_rows_idx


def estimate_lambdas(
    A: NDArray,
    b: NDArray,
    n_samples: int,
    rank: int,
    w: NDArray,
    sigma: NDArray,
    A_sampled_rows_idx: NDArray,
    A_row_norms: NDArray,
    A_ls_prob_rows: NDArray,
    A_ls_prob_columns: NDArray,
    A_frobenius: NDArray,
    rng: np.random.RandomState,
) -> NDArray:
    """Estimate lambda coefficients.

    Args:
        A: coefficient matrix.
        n_samples: number of samples to estimate inner products.
                   Note: the sampling is  performed from entries of `A`,
                   so there are `A.shape[0] * A.shape[1]` possible entries.
        rank: rank used to approximate matrix `A`.
        w: left-singular vector of `C`.
        sigma: singular values of `C`.
        A_sampled_rows_idx: indices of the r sampled rows of matrix A
        A_row_norms: norm of the rows of `A`.
        A_ls_prob_rows: row LS probability distribution of `A`.
        A_ls_prob_columns: column LS probability distribution of `A`.
        A_frobenius: Frobenius norm of `A`.
        rng: random state.

    Returns:
        lambda coefficients
    """
    m_rows, n_cols = A.shape
    r = len(A_sampled_rows_idx)
    n_realizations = 10
    lambdas_realizations = np.zeros((n_realizations, rank))
    for realization_i in range(n_realizations):
        for l in range(rank):
            # 1. Generate sample indices
            samples_i = []
            samples_j = []
            for _ in range(n_samples):
                i = rng.choice(m_rows, 1, replace=True, p=A_ls_prob_rows)[0]
                j = rng.choice(n_cols, 1, p=A_ls_prob_columns[i])[0]
                samples_i.append(i)
                samples_j.append(j)

            # 2. Approximate lambda using Monte Carlo estimation

            # Estimate right-singular vector
            R = (
                A[A_sampled_rows_idx[:, None], np.asarray(samples_j)[None, :]]
                * A_frobenius
                / (np.sqrt(r) * A_row_norms[A_sampled_rows_idx, None])
            )
            v_approx = R.T @ (w[:, l] / sigma[l])

            # Compute entries of outer product between `b` and `v_approx`
            outer_prod_b_v = b[samples_i] * v_approx

            # Estimate inner product between `A` and `outer_prod_b_v`
            lambdas_realizations[realization_i, l] = np.mean(
                A_frobenius**2
                / A[samples_i, samples_j]
                * outer_prod_b_v
                / sigma[l] ** 2
            )

    lambdas = np.median(lambdas_realizations, axis=0)

    return lambdas


def sample_from_x(
    A: NDArray,
    A_sampled_rows_idx: NDArray,
    A_row_norms: NDArray,
    R_ls_prob_columns: NDArray,
    A_frobenius: NDArray,
    omega: NDArray,
    omega_norm: float,
    rng: np.random.RandomState,
) -> tuple[int, float]:
    """Perform length-square (LS) sampling of the solution vector.

    Args:
        A: coefficient matrix.
        A_sampled_rows_idx: indices of the r sampled rows of matrix A
        A_row_norms: norm of the rows of `A`.
        A_ls_prob_columns: column LS probability distribution of `A`.
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

    while True:
        # Sample row index uniformly at random
        sample_i = rng.choice(r)

        # Sample column index from LS distribution of corresponding row
        sample_j = rng.choice(n_cols, 1, p=R_ls_prob_columns[sample_i])[0]

        # Sample column of `R`
        R_j = (
            A[A_sampled_rows_idx, sample_j]
            * A_frobenius
            / (np.sqrt(r) * A_row_norms[A_sampled_rows_idx])
        )
        R_j_norm = la.norm(R_j)

        # Determine if we output `sample_j`
        dot_prod_R_j_omega = np.dot(R_j, omega)
        prob = (dot_prod_R_j_omega / (omega_norm * R_j_norm)) ** 2
        if rng.binomial(1, prob) == 1:
            return sample_j, dot_prod_R_j_omega


def solve_qi(
    A: NDArray,
    b: NDArray,
    r: int,
    c: int,
    rank: int,
    n_samples: int,
    n_entries_x: int,
    rng: np.random.RandomState,
) -> tuple[NDArray, NDArray]:
    """Solves linear system of equations using a quantum-inspired algorithm.

    Args:
        A: coefficient matrix.
        b: vector b.
        r: number of rows to sample from A.
        c: number of columns to sample from A.
        rank: rank used to approximate matrix A.
        n_samples: number of samples to estimate inner products.
                   Note: the sampling is  performed from entries of `A`,
                   so there are `A.shape[0] * A.shape[1]` possible entries.
        n_entries_x: number of entries to be sampled from the solution vector.
        rng: random state.

    Returns:
        sampled indices,
        sampled entries
    """
    # 1. Generate length-square probability distributions to sample from matrix `A`
    A_ls_prob_rows, A_ls_prob_columns, A_row_norms, A_frobenius = compute_ls_probs(A)

    # 2. Build matrix `C` by sampling `r` rows and `c` columns
    C, _, R_ls_prob_columns, A_sampled_rows_idx = compute_C_and_R(
        A, r, c, A_row_norms, A_ls_prob_rows, A_ls_prob_columns, A_frobenius, rng
    )

    # 3. Compute the SVD of `C`
    w, sigma, _ = la.svd(C, full_matrices=False)

    # 4. Estimate lambda coefficients
    lambdas = estimate_lambdas(
        A,
        b,
        n_samples,
        rank,
        w,
        sigma,
        A_sampled_rows_idx,
        A_row_norms,
        A_ls_prob_rows,
        A_ls_prob_columns,
        A_frobenius,
        rng,
    )

    # 5. Sample solution vector

    # Compute `omega`
    omega = w[:, :rank] @ (lambdas / sigma[:rank])
    omega_norm = float(la.norm(omega))

    # Sample entries of solution vector `x`
    sampled_indices = np.zeros(n_entries_x, dtype=np.uint32)
    sampled_x = np.zeros(n_entries_x)
    for t in range(n_entries_x):
        sampled_indices[t], sampled_x[t] = sample_from_x(
            A,
            A_sampled_rows_idx,
            A_row_norms,
            R_ls_prob_columns,
            A_frobenius,
            omega,
            omega_norm,
            rng,
        )

    return sampled_indices, sampled_x
