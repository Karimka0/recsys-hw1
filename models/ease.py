"""EASE с нулевой диагональю матрицы весов."""

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from .utils import Recommender


class EASE(Recommender):
    def __init__(self, regularization=100.0):
        self.regularization = regularization

    def fit(self, frame, shape, **kwargs):
        good = frame[frame.rating >= 4]
        self.history = sparse.csr_matrix((np.ones(len(good)), (good.u, good.i)), shape=shape)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::shape[1] + 1] += self.regularization
        inverse = cho_solve(cho_factor(gram, lower=True, overwrite_a=True, check_finite=False),
                            np.eye(shape[1]), check_finite=False)
        self.weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
