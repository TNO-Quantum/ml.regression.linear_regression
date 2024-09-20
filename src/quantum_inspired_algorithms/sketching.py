from abc import ABC
from abc import abstractmethod
import numpy as np
from numpy import linalg as la
from numpy.typing import NDArray


class Sketcher(ABC):
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


class FKV(Sketcher):
    """FKV sketching."""

    def __init__(
        self,
        A: NDArray[np.float64],
        r: int,
        c: int,
        ls_prob_rows: NDArray[np.float64],
        ls_prob_columns_2d: NDArray[np.float64],
        frobenius: NDArray[np.float64],
        rng: np.random.RandomState,
    ) -> None:
        """Init QILinearEstimator.

        Note: LS stands for length-square.

        Args:
            A: coefficient matrix.
            r: number of rows for left projection matrix.
            c: number of columns for right projection matrix.
            ls_prob_rows: row LS probability distribution of `A`.
            ls_prob_columns_2d: column LS probability distribution of `A` (2D).
            frobenius: Frobenius norm of `A`.
            rng: random state.
        """
        m_rows, n_cols = A.shape

        # Sample row indices
        sampled_rows_idx = rng.choice(m_rows, r, replace=True, p=ls_prob_rows).astype(np.uint32)

        # Sample column indices
        sampled_columns_idx = np.zeros(c, dtype=np.uint32)
        for j in range(c):
            # Sample row index uniformly at random
            i = rng.choice(sampled_rows_idx, replace=True)

            # Sample column from LS distribution of row `i`
            sampled_columns_idx[j] = rng.choice(n_cols, 1, p=ls_prob_columns_2d[i, :])[0]

        # Compute norms
        sampled_row_norms = la.norm(A[sampled_rows_idx, :], axis=1)
        R = A[sampled_rows_idx, :] * frobenius / (np.sqrt(r) * sampled_row_norms[:, None])
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
        return M[self._sampled_rows_idx, :] * self._frobenius / (np.sqrt(self._r) * self._sampled_row_norms[:, None])

    def right_project(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        """Define right projector."""
        return (
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
        sample_j = rng.choice(self._n_cols, 1, p=self._R_ls_prob_columns[sample_i, :])[0]

        return sample_j

    def sample_row_idx(self, rng: np.random.RandomState) -> int:
        """Sample a row index from `C`."""
        # Sample col index uniformly at random
        sample_j = rng.choice(self._c)

        # Sample row index from LS distribution of corresponding column
        sample_i = rng.choice(self._n_rows, 1, p=self._C_ls_prob_rows[:, sample_j])[0]

        return sample_i
