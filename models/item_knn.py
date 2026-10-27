"""Item-based CF с косинусным сходством."""

import numpy as np
from scipy import sparse
from .utils import Recommender


class ItemKNN(Recommender):
    def __init__(self, neighbors=100, shrinkage=10.0):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, frame, shape, **kwargs):
        good = frame[frame.rating >= 4]
        self.history = sparse.csr_matrix(
            (np.ones(len(good)), (good.u, good.i)), shape=shape
        )
        overlap = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(overlap))
        sim = overlap / np.maximum(norms[:, None] * norms[None, :], 1e-12)
        sim *= overlap / (overlap + self.shrinkage + 1e-12)
        np.fill_diagonal(sim, 0)
        rows, cols, values = [], [], []
        for j in range(shape[1]):
            ix = np.argsort(-sim[j], kind="stable")[:self.neighbors]
            ix = ix[sim[j, ix] > 0]
            rows.extend([j] * len(ix)); cols.extend(ix); values.extend(sim[j, ix])
        self.similarity = sparse.csr_matrix((values, (rows, cols)), shape=(shape[1], shape[1]))
        return self

    def score(self, users):
        return (self.history[users] @ self.similarity).toarray()
