"""Факторизационная машина второго порядка."""

import numpy as np
import torch
from torch import nn
from scipy import sparse
from .utils import Recommender, history_features


class FactorizationMachine(Recommender):
    def __init__(self, factors=32, learning_rate=0.01, regularization=1e-5,
                 epochs=12, negatives=3, batch_size=4096, seed=42, metadata=True):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.negatives = negatives
        self.batch_size = batch_size
        self.seed = seed
        self.metadata = metadata

    @staticmethod
    def forward(ids, bias, linear, factors):
        v = factors[ids]
        return bias + linear[ids].sum(1) + 0.5 * ((v.sum(1) ** 2 - (v ** 2).sum(1)).sum(1))

    def fit(self, frame, shape, cutoff=None, **kwargs):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        cutoff = int(frame.timestamp.max()) if cutoff is None else cutoff
        uf, itf, n_features = history_features(frame, shape, cutoff, self.metadata)
        self.user_features = torch.as_tensor(uf, dtype=torch.long)
        self.item_features = torch.as_tensor(itf, dtype=torch.long)
        bias = nn.Parameter(torch.zeros(()))
        linear = nn.Parameter(torch.zeros(n_features))
        factors = nn.Parameter(torch.randn(n_features, self.factors) * 0.01)
        optimizer = torch.optim.Adam([bias, linear, factors], lr=self.learning_rate)
        seen = sparse.csr_matrix((np.ones(len(frame)), (frame.u, frame.i)), shape=shape)
        # Исключаем все известные товары, включая низкие оценки.
        seen_sets = [set(seen.indices[seen.indptr[u]:seen.indptr[u+1]]) for u in range(shape[0])]
        positive_users = frame.loc[frame.rating >= 4, "u"].to_numpy()
        self.loss_history = []
        for _ in range(self.epochs):
            negative_users = np.repeat(positive_users, self.negatives)
            negative_items = rng.integers(shape[1], size=len(negative_users))
            bad = np.fromiter((i in seen_sets[u] for u, i in zip(negative_users, negative_items)), bool)
            while bad.any():
                negative_items[bad] = rng.integers(shape[1], size=bad.sum())
                bad = np.fromiter((i in seen_sets[u] for u, i in zip(negative_users, negative_items)), bool)
            users = np.concatenate([frame.u.to_numpy(), negative_users])
            items = np.concatenate([frame.i.to_numpy(), negative_items])
            labels = np.concatenate([(frame.rating.to_numpy() >= 4).astype(np.float32),
                                     np.zeros(len(negative_users), dtype=np.float32)])
            order = rng.permutation(len(users))
            epoch_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                ix = order[start:start + self.batch_size]
                ids = torch.cat([self.user_features[users[ix]], self.item_features[items[ix]]], dim=1)
                prediction = self.forward(ids, bias, linear, factors)
                loss = nn.functional.binary_cross_entropy_with_logits(prediction, torch.from_numpy(labels[ix]))
                # L2 для линейных весов и факторов.
                loss = loss + self.regularization * (linear.square().sum() + factors.square().sum())
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += float(loss.detach()) * len(ix)
            self.loss_history.append(epoch_loss / len(order))
        self.bias = float(bias.detach())
        self.linear = linear.detach().numpy().copy()
        self.factors_matrix = factors.detach().numpy().copy()
        # Отдельно суммируем признаки пользователя и товара для матричного скоринга.
        self.user_vectors, self.user_bias = self._aggregate(uf)
        self.item_vectors, self.item_bias = self._aggregate(itf)
        return self

    def _aggregate(self, ids):
        vectors = self.factors_matrix[ids]
        sums = vectors.sum(1)
        bias = self.linear[ids].sum(1) + 0.5 * (sums ** 2 - (vectors ** 2).sum(1)).sum(1)
        return sums, bias

    def score(self, users):
        return (self.bias + self.user_bias[users, None] + self.item_bias[None, :]
                + self.user_vectors[users] @ self.item_vectors.T)
