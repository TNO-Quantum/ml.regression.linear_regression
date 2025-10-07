"""This module contains sketching algorithms for reducing dimensionality.

A "sketch" refers to a technique for compressing large matrices into
much smaller representations while approximately preserving their
essential properties.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_range_finder


class Sketcher(ABC):  # noqa: PLW1641
    """Base class for sketchers."""

    @abstractmethod
    def left_project(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        """Define left projector."""

    @abstractmethod
    def right_project(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        """Define right projector."""

    @abstractmethod
    def set_up_column_sampler(self, A: NDArray[np.float64]) -> None:
        """Set up column sampler for left sketch."""

    @abstractmethod
    def set_up_row_sampler(self, A: NDArray[np.float64]) -> None:
        """Set up row sampler for right sketch."""

    @abstractmethod
    def sample_column_idx(self, rng: np.random.RandomState) -> int:
        """Sample column index of left sketch."""

    @abstractmethod
    def sample_row_idx(self, rng: np.random.RandomState) -> int:
        """Sample row index of right sketch."""

    def __eq__(self, other: object) -> bool:
        """Check for equality for serialization purposes."""
        if not isinstance(other, Sketcher):
            return NotImplemented

        # Get the dictionaries of attributes
        self_attrs = self.__dict__
        other_attrs = other.__dict__

        # Check the keys
        if self_attrs.keys() != other_attrs.keys():
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


class FKV(Sketcher):
    """FKV (Frieze-Kannan-Vempala) sketching."""

    def __init__(  # noqa: PLR0913
        self,
        A: NDArray[np.float64],
        r: int,
        c: int,
        ls_prob_rows: NDArray[np.float64],
        ls_prob_columns_2d: NDArray[np.float64],
        frobenius: float,
        rng: np.random.RandomState,
    ) -> None:
        """Init :py:class:`FKV`.

        Note: LS stands for length-square.

        Args:
            A: coefficient matrix `A`.
            r: number of rows for left projection matrix.
            c: number of columns for right projection matrix.
            ls_prob_rows: row LS probability distribution of `A`.
            ls_prob_columns_2d: column LS probability distribution of `A` (2D).
            frobenius: Frobenius norm of `A`.
            rng: random state.
        """
        m_rows, n_cols = A.shape

        # Sample row indices
        sampled_rows_idx = rng.choice(m_rows, r, replace=True, p=ls_prob_rows).astype(
            np.uint32
        )

        # Sample column indices
        sampled_columns_idx = np.zeros(c, dtype=np.uint32)
        for j in range(c):
            # Sample row index uniformly at random
            i = rng.choice(sampled_rows_idx, replace=True)

            # Sample column from LS distribution of row `i`
            sampled_columns_idx[j] = rng.choice(n_cols, 1, p=ls_prob_columns_2d[i, :])[
                0
            ]

        # Compute norms
        sampled_row_norms = la.norm(A[sampled_rows_idx, :], axis=1)
        R = (
            A[sampled_rows_idx, :]
            * frobenius
            / (np.sqrt(r) * sampled_row_norms[:, None])
        )
        R_sampled_column_norms = la.norm(R[:, sampled_columns_idx], axis=0)

        # Set sketching parameters
        self._r = r
        self._c = c
        self._sampled_rows_idx = sampled_rows_idx
        self._sampled_columns_idx = sampled_columns_idx
        self._frobenius = frobenius
        self._sampled_row_norms = sampled_row_norms
        self._R_sampled_column_norms = R_sampled_column_norms

    def left_project(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        """Define left projector."""
        return np.asarray(
            M[self._sampled_rows_idx, :]
            * self._frobenius
            / (np.sqrt(self._r) * self._sampled_row_norms[:, None])
        )

    def right_project(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        """Define right projector."""
        return np.asarray(
            M[:, self._sampled_columns_idx]
            * self._frobenius
            / (np.sqrt(self._c) * self._R_sampled_column_norms[None, :])
        )

    def set_up_column_sampler(self, A: NDArray[np.float64]) -> None:
        """Build LS distribution to sample columns from matrix `R`."""
        R = self.left_project(A)
        self._R_ls_prob_columns = R**2 / la.norm(R, axis=1)[:, None] ** 2
        self._n_cols = A.shape[1]

    def set_up_row_sampler(self, A: NDArray[np.float64]) -> None:
        """Build LS distribution to sample rows from matrix `C`."""
        C = self.right_project(A)
        self._C_ls_prob_rows = C**2 / la.norm(C, axis=0)[None, :] ** 2
        self._n_rows = A.shape[0]

    def sample_column_idx(self, rng: np.random.RandomState) -> int:
        """Sample a column index from `R`."""
        # Sample row index uniformly at random
        sample_i = rng.choice(self._r)

        # Sample column index from LS distribution of corresponding row
        sample_j = rng.choice(self._n_cols, 1, p=self._R_ls_prob_columns[sample_i, :])[
            0
        ]

        return int(sample_j)

    def sample_row_idx(self, rng: np.random.RandomState) -> int:
        """Sample a row index from `C`."""
        # Sample col index uniformly at random
        sample_j = rng.choice(self._c)

        # Sample row index from LS distribution of corresponding column
        sample_i = rng.choice(self._n_rows, 1, p=self._C_ls_prob_rows[:, sample_j])[0]

        return int(sample_i)


class Halko(Sketcher):
    """Halko sketching."""

    def __init__(
        self,
        A: NDArray[np.float64],
        r: int,
        c: int,
        ls_prob_columns: NDArray[np.float64],
        rng: np.random.RandomState,
    ) -> None:
        """Init :py:class:`Halko`.

        Note: LS stands for length-square.

        Args:
            A: coefficient matrix `A`.
            r: number of rows for left projection matrix.
            c: number of columns for right projection matrix.
            ls_prob_columns: column LS probability distribution of `A`.
            rng: random state.
        """
        self._Q_left = Halko._get_low_dimensional_projector(
            A, axis=0, n_components=r, random_state=rng
        )
        self._Q_right = Halko._get_low_dimensional_projector(
            self._Q_left @ A, axis=1, n_components=c, random_state=rng
        )
        self._ls_prob_columns = ls_prob_columns

    @classmethod
    def _get_low_dimensional_projector(
        cls,
        M: NDArray[np.float64],
        axis: int,
        n_components: int,
        random_state: np.random.RandomState,
    ) -> NDArray[np.float64]:
        """Find random matrix to reduce dimensionality of axis of `M`."""
        n_oversamples = 10
        n_random = n_components + n_oversamples
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4
        if axis == 1:
            M = M.T
        Q = np.asarray(
            randomized_range_finder(
                M,
                size=n_random,
                n_iter=n_iter,
                power_iteration_normalizer="auto",
                random_state=random_state,
            )
        )
        if axis == 0:
            Q = Q.T

        return np.asarray(Q)

    def left_project(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        """Define left projector."""
        return self._Q_left @ M

    def right_project(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        """Define right projector."""
        return M @ self._Q_right

    def set_up_column_sampler(self, A: NDArray[np.float64]) -> None:
        """No setup required."""

    def set_up_row_sampler(self, A: NDArray[np.float64]) -> None:
        """Build LS distribution to sample rows from matrix `C`."""
        A_row_norms = la.norm(A, axis=1)
        A_row_norms_squared = A_row_norms**2
        A_frobenius = np.sqrt(np.sum(A_row_norms_squared))
        self._ls_prob_rows = A_row_norms_squared / A_frobenius**2
        self._n_rows = A.shape[0]

    def sample_column_idx(self, rng: np.random.RandomState) -> int:
        """Sample a column index."""
        n_cols = self._ls_prob_columns.size
        sample_j = rng.choice(n_cols, 1, p=self._ls_prob_columns)[0]

        return int(sample_j)

    def sample_row_idx(self, rng: np.random.RandomState) -> int:
        """Sample a row index."""
        sample_i = rng.choice(self._n_rows, 1, p=self._ls_prob_rows)[0]

        return int(sample_i)
