"""This module implements a quantum-inspired algorithm for linear regression.

Assume a linear system of the form `$Ax=b$` where:

- `$A$` is the training data.
- `$x$` is a vector of unknown coefficients.
- `$b$` is a vector of target values.

The class :py:class:`QILinearEstimator` provides three methods:

- :py:meth:`~QILinearEstimator.fit`: for model fitting using `$A$` and `$b$`.
- :py:meth:`~QILinearEstimator.sample_prediction_x`: for sampling entries of
  the estimated coefficient vector.
- :py:meth:`~QILinearEstimator.sample_prediction_b`: for sampling entries of predictions
  corresponding to (un)observed target values.


Basic usage
-----------

>>> import numpy as np
>>> from sklearn.datasets import make_low_rank_matrix
>>> from sklearn.model_selection import train_test_split
>>> from tno.quantum.ml.regression.linear_regression import QILinearEstimator
>>>
>>> rng = np.random.RandomState(7)
>>>
>>> # Generate example data
>>> m = 700
>>> n = 100
>>> A = make_low_rank_matrix(n_samples=m, n_features=n, effective_rank=3,
...     random_state=rng, tail_strength=0.1)
>>> x = rng.normal(0, 1, A.shape[1])
>>> b = A @ x
>>>
>>> # Create training and test datasets
>>> A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.3,
...     random_state=rng)
>>>
>>> # Fit quantum-inspired model
>>> rank = 3
>>> r = 100
>>> c = 30
>>> n_samples = 100  # for Monte Carlo methods
>>> qi = QILinearEstimator(r, c, rank, n_samples, rng, sketcher_name="fkv")
>>> qi = qi.fit(A_train, b_train)
>>>
>>> # Sample from b (vector of predictions)
>>> n_entries_b = 1000
>>> sampled_indices_b, sampled_b = qi.sample_prediction_b(A_test, n_entries_b)
"""

from tno.quantum.ml.regression.linear_regression._estimator import QILinearEstimator

__all__: list[str] = ["QILinearEstimator"]

__version__ = "1.0.0"
