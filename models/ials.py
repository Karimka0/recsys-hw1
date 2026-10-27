"""ALS с confidence для наблюдавшихся отзывов."""

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from .utils import Recommender


class IALS(Recommender):
    def __init__(self, factors=32, regularization=1.0, alpha=20.0, iterations=12, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def target_values(self, frame, cutoff):
        return (frame.rating.to_numpy() >= 4).astype(float)

    def _update(self, target, fixed):
        base = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        result = np.zeros((target.shape[0], self.factors))
        for row in range(target.shape[0]):
            lo, hi = target.indptr[row:row + 2]
            j = target.indices[lo:hi]
            if not len(j):
                continue
            y = fixed[j]
            # Нулевые таргеты остаются в CSR: confidence этих пар тоже учитывается.
            a = base + self.alpha * (y.T @ y)
            b = y.T @ ((1 + self.alpha) * target.data[lo:hi])
            result[row] = cho_solve(cho_factor(a, lower=True, check_finite=False), b, check_finite=False)
        return result

    def fit(self, frame, shape, cutoff=None, **kwargs):
        cutoff = int(frame.timestamp.max()) if cutoff is None else cutoff
        targets = sparse.csr_matrix((self.target_values(frame, cutoff), (frame.u, frame.i)), shape=shape)
        targets.sort_indices()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.01, (shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._update(targets, self.item_factors)
            self.item_factors = self._update(targets.T.tocsr(), self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
