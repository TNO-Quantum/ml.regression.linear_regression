from __future__ import annotations
import logging
import warnings
from typing import Callable
from typing import Optional
import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray
from quantum_inspired_algorithms.quantum_inspired import compute_ls_probs
from quantum_inspired_algorithms.quantum_inspired import estimate_lambdas
from quantum_inspired_algorithms.quantum_inspired import sample_from_b
from quantum_inspired_algorithms.quantum_inspired import sample_from_x
from quantum_inspired_algorithms.sketching import FKV


class EstimatorError(Exception):
    """Module exception."""

    def __init__(self, message: str):
        """Init EstimatorError."""
        super().__init__(message)


class QILinearEstimator:
    """Quantum-inspired linear estimator."""

    def __init__(
        self,
        r: int,
        c: int,
        rank: int,
        n_samples: int,
        random_state: np.random.RandomState,
        sigma_threshold: float = 1e-15,
        func: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Init QILinearEstimator.

        Args:
            r: number of rows to sample from A.
            c: number of columns to sample from A.
            rank: rank used to approximate matrix A.
            n_samples: number of samples to estimate inner products.
                       Note: the sampling is  performed from entries of `A`,
                       so there are `A.shape[0] * A.shape[1]` possible entries.
            random_state: random state.
            sigma_threshold: the argument `rank` is recomputed in case it is higher
                             the number of singular values below this threhold.
            func: function to transform singular values when estimating lambda coefficients.
                  This can be used for Tikhonov regularization purposes.
        """
        self.r = r
        self.c = c
        self.rank = rank
        self.n_samples = n_samples
        self.random_state = random_state
        self.sigma_threshold = sigma_threshold
        self.func = func

    def fit(
        self,
        A: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> QILinearEstimator:
        """Fit data using quantum-inspired algorithm.

        Args:
            A: coefficient matrix.
            b: vector `b`.
        """
        # Validate input
        if self.rank <= 0:
            raise ValueError("`rank` should be greater than 0")

        if self.r <= self.rank or self.c <= self.rank:
            raise ValueError("both `r` and `c` should be greater than `rank`")

        if self.n_samples <= 1:
            raise ValueError("`n_samples` should be greater than 1")

        if self.sigma_threshold <= 0:
            raise ValueError("`sigma_threshold` should be greater than 0")

        # 1. Generate length-square probability distributions to sample from matrix `A`
        logging.info("1. Generate length-square probability distributions to sample from matrix `A`")
        (
            self.A_ls_prob_rows_,
            self.A_ls_prob_columns_2d_,
            _,
            _,
            self.A_frobenius_,
        ) = compute_ls_probs(A)

        # 2. Build matrix `C`
        logging.info("2. Build matrix `C`")
        self.sketcher_ = FKV(
            A,
            self.r,
            self.c,
            self.A_ls_prob_rows_,
            self.A_ls_prob_columns_2d_,
            self.A_frobenius_,
            self.random_state,
        )
        C = self.sketcher_.right_project(self.sketcher_.left_project(A))

        # 3. Compute the SVD of `C`
        logging.info("3. Compute the SVD of `C`")
        self.w_left_, self.sigma_, self.w_right_T_ = la.svd(C, full_matrices=False)

        # Recompute rank
        self.rank_ = self.rank
        rank_recomputed = np.count_nonzero(self.sigma_ > self.sigma_threshold)
        if rank_recomputed < self.rank:
            message = f"Desired rank: {self.rank}; recomputed: {rank_recomputed}"
            warnings.warn(message, RuntimeWarning)
            logging.warning(message)
            self.rank_ = rank_recomputed

        # 4. Estimate lambda coefficients
        logging.info("4. Estimate lambda coefficients")
        if self.func is None:

            def func_(arg: float) -> float:
                return arg

            func = func_
        else:
            func = self.func
        self.lambdas_ = estimate_lambdas(
            A,
            b,
            self.n_samples,
            self.rank_,
            self.w_left_,
            self.sigma_,
            self.sketcher_,
            self.A_ls_prob_rows_,
            self.A_ls_prob_columns_2d_,
            self.A_frobenius_,
            self.random_state,
            func,
        )

        return self

    def _check_is_fitted(self):
        """Check if the `fit` method has been called."""
        for attribute_name in [
            "A_ls_prob_rows_",
            "A_ls_prob_columns_2d_",
            "A_frobenius_",
            "sketcher_",
            "w_left_",
            "sigma_",
            "w_right_T_",
            "rank_",
            "lambdas_",
        ]:
            if not hasattr(self, attribute_name):
                raise EstimatorError("Please call `fit` before making predictions")

    def predict_x(
        self,
        A: NDArray[np.float64],
        n_entries_x: int,
    ) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
        """Predict `x` using quantum-inspired model.

        Args:
            A: coefficient matrix.
            n_entries_x: number of entries to be sampled from the solution vector `x`.
                         Set this to 0 to skip this sampling step.

        Returns:
            Samples of predicted values and corresponding indices.
        """
        self._check_is_fitted()

        if n_entries_x == 0:
            raise ValueError("`n_entries_x` should be greater than 0")

        logging.info("Sample predicted `x`")

        # Compute `omega`
        omega = self.w_left_[:, : self.rank_] @ (self.lambdas_ / self.sigma_[: self.rank_])
        omega_norm = float(la.norm(omega))

        # Sample entries of solution vector `x`
        sampled_indices_x = np.zeros(n_entries_x, dtype=np.uint32)
        sampled_x = np.zeros(n_entries_x)
        for t in range(n_entries_x):
            sampled_indices_x[t], sampled_x[t] = sample_from_x(
                A,
                self.sketcher_,
                omega,
                omega_norm,
                self.random_state,
            )
            if (t + 1) % 100 == 0:
                logging.info(f"---{t + 1} entries sampled out of {n_entries_x}")

        return sampled_indices_x, sampled_x

    def predict_b(
        self,
        A: NDArray[np.float64],
        n_entries_b: int,
    ) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
        """Predict `b` using quantum-inspired model.

        Args:
            A: coefficient matrix.
            n_entries_b: number of entries to be sampled from the predicted `b`.

        Returns:
            Samples of predicted values and corresponding indices.
        """
        self._check_is_fitted()

        if n_entries_b == 0:
            raise ValueError("`n_entries_b` should be greater than 0")

        logging.info("Sample predicted `b`")

        # Compute `phi`
        phi = self.w_right_T_.T[:, : self.rank_] @ self.lambdas_
        phi_norm = float(la.norm(phi))

        # Sample entries of `b`
        sampled_indices_b = np.zeros(n_entries_b, dtype=np.uint32)
        sampled_b = np.zeros(n_entries_b)
        for t in range(n_entries_b):
            sampled_indices_b[t], sampled_b[t] = sample_from_b(
                A,
                self.sketcher_,
                phi,
                phi_norm,
                self.random_state,
            )
            if (t + 1) % 100 == 0:
                logging.info(f"---{t + 1} entries sampled out of {n_entries_b}")

        return sampled_indices_b, sampled_b
