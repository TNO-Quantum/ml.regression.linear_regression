"""Module with definitions of quantum-inspired estimators."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, SupportsFloat, SupportsInt

import numpy as np
from numpy import linalg as la
from numpy.typing import ArrayLike, NDArray

from tno.quantum.ml.components import SerializableEstimator
from tno.quantum.ml.regression.linear_regression._quantum_inspired import (
    compute_ls_probs,
    estimate_lambdas,
    sample_from_b,
    sample_from_x,
)
from tno.quantum.ml.regression.linear_regression._sketching import FKV, Halko, Sketcher
from tno.quantum.utils.serialization import Serializable
from tno.quantum.utils.validation import (
    check_arraylike,
    check_int,
    check_random_state,
    check_real,
)

logger = logging.getLogger(__name__)


def _serialize_sketcher(sketcher: Sketcher) -> dict[str, Any]:
    dict_ = {}
    for key, value in sketcher.__dict__.items():
        dict_[key] = Serializable.serialize(value)
    return dict_


def _deserialize_sketcher(
    data: dict[str, Any], sketcher_name: str = "fkv"
) -> FKV | Halko:
    data = {key: Serializable.deserialize(value) for key, value in data.items()}
    sketcher_class: type[FKV | Halko]
    if sketcher_name == "fkv":
        sketcher_class = FKV
    elif sketcher_name == "halko":
        sketcher_class = Halko
    else:
        message = '`sketcher_name` should be either "fkv" or "halko"'
        raise ValueError(message)
    sketcher = sketcher_class.__new__(sketcher_class)
    sketcher.__dict__.update(data)
    return sketcher


Serializable.register(
    FKV, _serialize_sketcher, lambda x: _deserialize_sketcher(x, sketcher_name="fkv")
)
Serializable.register(
    Halko,
    _serialize_sketcher,
    lambda x: _deserialize_sketcher(x, sketcher_name="halko"),
)


class EstimatorError(Exception):
    """Module exception."""

    def __init__(self, message: str) -> None:
        """Init :py:class:`EstimatorError`."""
        super().__init__(message)


class QILinearEstimator(SerializableEstimator):  # noqa: PLW1641
    """Quantum-inspired linear estimator."""

    def __init__(  # noqa: PLR0913
        self,
        r: SupportsInt,
        c: SupportsInt,
        rank: SupportsInt,
        n_samples: SupportsInt,
        random_state: SupportsInt | None = None,
        sigma_threshold: SupportsFloat = 1e-15,
        sketcher_name: str = "fkv",
        func: Callable[[SupportsFloat], SupportsFloat] | None = None,
    ) -> None:
        """Init :py:class:`QILinearEstimator`.

        Args:
            r: number of rows to sample from `A`.
            c: number of columns to sample from `A`.
            rank: rank used to approximate matrix `A`.
            n_samples: number of samples to estimate inner products.
                       Note: the sampling is  performed from entries of `A`,
                       so there are ``A.shape[0] * A.shape[1]`` possible entries.
            random_state: random seed.
            sigma_threshold: the argument `rank` is recomputed in case it is higher
                             the number of singular values below this threhold.
            sketcher_name: name of sketching method: ``"fkv"`` or ``"halko"``.
            func: function to transform singular values when estimating lambda
                  coefficients. This can be used for Tikhonov regularization purposes.
        """
        self.r = r
        self.c = c
        self.rank = rank
        self.n_samples = n_samples
        self.random_state = random_state
        self.sigma_threshold = sigma_threshold
        self.sketcher_name = sketcher_name
        self.func = func

    def fit(
        self,
        A: ArrayLike,
        b: ArrayLike,
    ) -> QILinearEstimator:
        """Fit data using quantum-inspired algorithm.

        Args:
            A: coefficient matrix `A`.
            b: vector `b`.
        """
        # Validate input
        A = check_arraylike(A, "A", ndim=2)
        A = np.asarray(A, dtype=np.float64)
        b = check_arraylike(b, "b", ndim=1)
        b = np.asarray(b, dtype=np.float64)
        self.rank = check_int(self.rank, "rank", l_bound=1, l_inclusive=True)
        self.r = check_int(self.r, "r", l_bound=self.rank, l_inclusive=False)
        self.c = check_int(self.c, "c", l_bound=self.rank, l_inclusive=False)
        self.n_samples = check_int(
            self.n_samples, "n_samples ", l_bound=2, l_inclusive=True
        )
        self.sigma_threshold = check_real(
            self.sigma_threshold, "sigma_threshold", l_bound=0, l_inclusive=False
        )

        # Get random state
        rng = check_random_state(self.random_state, "random_state")

        # 1. Generate length-square probability distributions to sample from matrix `A`
        logger.info(
            "1. Generate length-square probability distributions to sample "
            "from matrix `A`"
        )
        (
            A_ls_prob_rows,
            A_ls_prob_columns_2d,
            A_ls_prob_columns,
            _,
            A_frobenius,
        ) = compute_ls_probs(A)

        # 2. Build matrix `C`
        logger.info("2. Build matrix `C`")
        self.sketcher_: Sketcher
        if self.sketcher_name == "fkv":
            self.sketcher_ = FKV(
                A,
                self.r,
                self.c,
                A_ls_prob_rows,
                A_ls_prob_columns_2d,
                A_frobenius,
                rng,
            )
        elif self.sketcher_name == "halko":
            self.sketcher_ = Halko(
                A,
                self.r,
                self.c,
                A_ls_prob_columns,
                rng,
            )
        else:
            message = '`sketcher_name` should be either "fkv" or "halko"'
            raise ValueError(message)

        C = self.sketcher_.right_project(self.sketcher_.left_project(A))

        # 3. Compute the SVD of `C`
        logger.info("3. Compute the SVD of `C`")
        self.w_left_, self.sigma_, self.w_right_T_ = la.svd(C, full_matrices=False)

        # Recompute rank
        self.rank_ = self.rank
        rank_recomputed = int(np.count_nonzero(self.sigma_ > self.sigma_threshold))
        if rank_recomputed < self.rank:
            message = f"Desired rank: {self.rank}; recomputed: {rank_recomputed}"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            logger.warning(message)
            self.rank_ = rank_recomputed

        # 4. Estimate lambda coefficients
        logger.info("4. Estimate lambda coefficients")
        func: Callable[[SupportsFloat], SupportsFloat]
        if self.func is None:

            def func_(arg: SupportsFloat) -> SupportsFloat:
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
            A_ls_prob_rows,
            A_ls_prob_columns_2d,
            A_frobenius,
            rng,
            func,
        )

        return self

    def _check_is_fitted(self) -> None:
        """Check if the `fit` method has been called."""
        for attribute_name in [
            "sketcher_",
            "w_left_",
            "sigma_",
            "w_right_T_",
            "rank_",
            "lambdas_",
        ]:
            if not hasattr(self, attribute_name):
                message = "Please call `fit` before making predictions"
                raise EstimatorError(message)

    def sample_prediction_x(
        self,
        A: ArrayLike,
        n_entries_x: SupportsInt,
    ) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
        """Samples predictions of `x` using quantum-inspired model.

        Args:
            A: coefficient matrix `A`.
            n_entries_x: number of entries to be sampled from the solution vector `x`.
                         Set this to 0 to skip this sampling step.

        Returns:
            Samples of predicted values and corresponding indices.
        """
        self._check_is_fitted()
        rng = check_random_state(self.random_state, "random_state")

        A = check_arraylike(A, "A", ndim=2)
        A = np.asarray(A, dtype=np.float64)
        n_entries_x = check_int(n_entries_x, "n_entries_x", l_bound=1, l_inclusive=True)

        logger.info("Sample predicted `x`")

        # Compute `omega`
        omega = self.w_left_[:, : self.rank_] @ (
            self.lambdas_ / self.sigma_[: self.rank_]
        )
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
                rng,
            )
            if (t + 1) % 100 == 0:
                logger.info("---%s entries sampled out of %s", t + 1, n_entries_x)

        return sampled_indices_x, sampled_x

    def sample_prediction_b(
        self,
        A: ArrayLike,
        n_entries_b: SupportsInt,
    ) -> tuple[NDArray[np.uint32], NDArray[np.float64]]:
        """Sample predictions of `b` using quantum-inspired model.

        Args:
            A: coefficient matrix `A`.
            n_entries_b: number of entries to be sampled from the predicted `b`.

        Returns:
            Samples of predicted values and corresponding indices.
        """
        self._check_is_fitted()
        rng = check_random_state(self.random_state, "random_state")

        A = check_arraylike(A, "A", ndim=2)
        A = np.asarray(A, dtype=np.float64)
        n_entries_b = check_int(n_entries_b, "n_entries_b", l_bound=1, l_inclusive=True)

        logger.info("Sample predicted `b`")

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
                rng,
            )
            if (t + 1) % 100 == 0:
                logger.info("---%s entries sampled out of %s", t + 1, n_entries_b)

        return sampled_indices_b, sampled_b

    def __eq__(self, other: object) -> bool:
        """Check for equality for serialization purposes."""
        if not isinstance(other, QILinearEstimator):
            return NotImplemented

        # Get the dictionaries of attributes
        self_attrs = self.__dict__
        other_attrs = other.__dict__

        # Check the keys
        if set(self_attrs.keys()) != set(other_attrs.keys()):
            return False

        # Iterate and compare values
        for key, self_value in self_attrs.items():
            other_value = other_attrs[key]

            # Check if they are numpy arrays and compare accordingly
            is_self_np = isinstance(self_value, np.ndarray)
            is_other_np = isinstance(other_value, np.ndarray)

            if is_self_np and is_other_np:
                if not np.array_equal(self_value, other_value):
                    return False
            elif is_self_np != is_other_np or self_value != other_value:
                return False

        return True
