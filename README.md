## How to use quantum_inspired_algorithms

This package provides quantum-inspired algorithms for linear regression. It assumes a 
linear system of the form `Ax=b`, where `A` is the training data, `x` is a vector of
unknown coefficients, and `b` is a vector of target values.

The class `estimator.QILinearEstimator` provides three methods:
`fit`, `predict_x`, and `predict_b`. Once the `fit` method has been called using `A` and `b`,
`predict_x` can be used to sample entries of the estimated coefficient vector. Alternatively,
`predict_b` can be used to sample entries of predictions corresponding to (un)observed
target values.

The project setup is documented in [project_setup.md](project_setup.md).

## Installation

To install quantum_inspired_algorithms from GitHub repository, do:

```console
git clone git@github.com:QuantumApplicationLab/quantum-inspired-algorithms.git
cd quantum-inspired-algorithms
python -m pip install .
```

## Example

```python
import numpy as np
from sklearn.datasets import make_low_rank_matrix
from sklearn.model_selection import train_test_split
from quantum_inspired_algorithms.estimator import QILinearEstimator

rng = np.random.RandomState(7)

# Generate example data
m = 700
n = 100
A = make_low_rank_matrix(n_samples=m, n_features=n, effective_rank=3, random_state=rng, tail_strength=0.1)
x = rng.normal(0, 1, A.shape[1])
b = A @ x

# Create training and test datasets
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.3, random_state=rng)

# Fit quantum-inspired model
rank = 3
r = 100
c = 30
n_samples = 100  # for Monte Carlo methods
qi = QILinearEstimator(r, c, rank, n_samples, rng, sketcher_name="fkv")
qi = qi.fit(A_train, b_train)

# Sample from b (vector of predictions)
n_entries_b = 1000
sampled_indices_b, sampled_b = qi.predict_b(A_test, n_entries_b)
```

More examples can be found in the `tests` directory.

## Contributing

If you want to contribute to the development of quantum_inspired_algorithms,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

The algorithms found in this repository have been developed by the Quantum Application Lab
and have been based on:

- https://github.com/XanaduAI/quantum-inspired-algorithms
- "Quantum-inspired algorithms in practice", by Juan Miguel Arrazola, Alain Delgado, Bhaskar Roy Bardhan, and Seth Lloyd. 2020-08-13, volume 4, page 307. Quantum 4, 307 (2020).
- "Quantum-inspired low-rank stochastic regression with logarithmic dependence on the dimension", by András Gilyén, Seth Lloyd, Ewin Tang. (2018). ArXiv, abs/1811.04909.

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
