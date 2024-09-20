import logging
import warnings
from typing import Callable
import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray
from quantum_inspired_algorithms.sketching import Sketcher


def compute_ls_probs(
    A: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute length-square (LS) probability distributions for sampling `A`.

    Args:
        A: coefficient matrix.

    Returns:
        LS probability distribution for rows,
        LS probability distribution for columns (2D),
        LS probability distribution for columns,
        LS probability distribution for rows (2D),
        Frobenius norm
    """
    # Compute row norms
    A_row_norms = la.norm(A, axis=1)
    A_row_norms_squared = A_row_norms**2

    # Compute column norms
    A_column_norms = la.norm(A, axis=0)
    A_column_norms_squared = A_column_norms**2

    # Compute Frobenius norm
    A_frobenius = np.sqrt(np.sum(A_row_norms_squared))

    # Compute LS probabilities for rows
    A_ls_prob_rows = A_row_norms_squared / A_frobenius**2

    # Compute LS probabilities for columns
    A_ls_prob_columns_2d = A**2 / A_row_norms_squared[:, None]

    # Compute LS probabilities for columns
    A_ls_prob_columns = A_column_norms_squared / A_frobenius**2

    # Compute LS probabilities for rows
    A_ls_prob_rows_2d = A**2 / A_column_norms_squared[None, :]

    return (
        A_ls_prob_rows,
        A_ls_prob_columns_2d,
        A_ls_prob_columns,
        A_ls_prob_rows_2d,
        A_frobenius,
    )


def estimate_lambdas(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    n_samples: int,
    rank: int,
    w: NDArray[np.float64],
    sigma: NDArray[np.float64],
    sketcher: Sketcher,
    A_ls_prob_rows: NDArray[np.float64],
    A_ls_prob_columns: NDArray[np.float64],
    A_frobenius: NDArray[np.float64],
    rng: np.random.RandomState,
    func: Callable[[float], float],
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
        sketcher: sketcher to left project `A`.
        A_ls_prob_rows: row LS probability distribution of `A`.
        A_ls_prob_columns: column LS probability distribution of `A` (2D).
        A_frobenius: Frobenius norm of `A`.
        rng: random state.
        func: function to transform singular values when estimating lambda coefficients.

    Returns:
        lambda coefficients
    """
    m_rows, n_cols = A.shape
    n_realizations = 10
    lambdas_realizations = np.zeros((n_realizations, rank))
    for realization_i in range(n_realizations):
        logging.info(f"---Realization {realization_i}")
        for ell in range(rank):
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
            R = sketcher.left_project(A[:, np.asarray(samples_j)])
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
    sketcher: Sketcher,
    phi: NDArray[np.float64],
    phi_norm: float,
    rng: np.random.RandomState,
    max_n_sampling_attempts: int = int(1e5),
) -> tuple[int, float]:
    """Perform length-square (LS) sampling from the predicted `b`.

    Args:
        A: coefficient matrix.
        sketcher: sketcher to left project `A` and sample its rows.
        phi: vector phi.
        phi_norm: norm of `phi`.
        rng: random state.
        max_n_sampling_attempts: maximum number of sampling attempts.

    Returns:
        index of the sampled entry,
        entry value
    """
    C = sketcher.right_project(A)
    sketcher.set_up_row_sampler(A)

    for _ in range(max_n_sampling_attempts):
        # Sample row index
        sample_i = sketcher.sample_row_idx(rng)

        # Sample row of `C`
        C_i = C[sample_i, :]
        C_i_norm = la.norm(C_i)

        # Determine if we output `sample_i`
        dot_prod_C_i_omega = np.dot(C_i, phi)
        prob = (dot_prod_C_i_omega / (phi_norm * C_i_norm)) ** 2
        if rng.binomial(1, prob) == 1:
            return sample_i, dot_prod_C_i_omega

    message = (
        f"Maximum number of sampling attempts ({max_n_sampling_attempts}) exceeded. Returning sample from last attempt."
    )
    warnings.warn(message, RuntimeWarning)
    logging.warning(message)

    return sample_i, dot_prod_C_i_omega


def sample_from_x(
    A: NDArray[np.float64],
    sketcher: Sketcher,
    omega: NDArray[np.float64],
    omega_norm: float,
    rng: np.random.RandomState,
    max_n_sampling_attempts: int = int(1e5),
) -> tuple[int, float]:
    """Perform length-square (LS) sampling from the solution vector.

    Args:
        A: coefficient matrix.
        sketcher: sketcher to left project `A` and sample its columns.
        omega: vector omega.
        omega_norm: norm of `omega`.
        rng: random state.
        max_n_sampling_attempts: maximum number of sampling attempts.

    Returns:
        index of the sampled entry,
        entry value
    """
    R = sketcher.left_project(A)
    sketcher.set_up_column_sampler(A)

    for _ in range(max_n_sampling_attempts):
        # Sample column index
        sample_j = sketcher.sample_column_idx(rng)

        # Sample column of `R`
        R_j = R[:, sample_j]
        R_j_norm = la.norm(R_j)

        # Determine if we output `sample_j`
        dot_prod_R_j_omega = np.dot(R_j, omega)
        prob = (dot_prod_R_j_omega / (omega_norm * R_j_norm)) ** 2
        if rng.binomial(1, prob) == 1:
            return sample_j, dot_prod_R_j_omega

    message = (
        f"Maximum number of sampling attempts ({max_n_sampling_attempts}) exceeded. Returning sample from last attempt."
    )
    warnings.warn(message, RuntimeWarning)
    logging.warning(message)

    return sample_j, dot_prod_R_j_omega
