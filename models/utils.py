"""Загрузка данных, признаки и оценка рекомендаций."""

import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/"
FILES = ["reviews_Baby_5.json.gz", "meta_Baby.json.gz"]



HASHES = {'meta_Baby.json.gz': '0915eee1158307e64258229126faec4e42fe23d09bc092601e35b2dc66caeba3',
 'reviews_Baby_5.json.gz': 'f9d7571fdbf4490fda6e302879215ff58e7d74baf025dcd9c99db9d502241b49'}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_data():
    cache = Path(os.environ.get("RECSYS_CACHE", ROOT / ".cache"))
    cache.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = cache / name
        if not path.exists():
            temporary = path.with_suffix(".part")
            print(f"Загрузка {name}", flush=True)
            urllib.request.urlretrieve(BASE_URL + name, temporary)
            temporary.replace(path)
        if file_sha256(path) != HASHES[name]:
            raise ValueError(f"Не совпал SHA-256: {path}. Удалите файл и повторите загрузку.")
    return cache


def load_data():
    cache = download_data()
    records = []
    with gzip.open(cache / FILES[0], "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            records.append({"user_id": record.get("reviewerID"), "item_id": record.get("asin"),
                            "rating": record.get("overall"), "timestamp": record.get("unixReviewTime")})
    raw = pd.DataFrame(records)
    clean = raw.dropna(subset=["user_id", "item_id", "rating", "timestamp"]).copy()
    clean = clean[clean.rating.between(1, 5) & (clean.timestamp > 0)]
    clean = clean.sort_values(["timestamp", "user_id", "item_id"], kind="stable")
    clean = clean.drop_duplicates(["user_id", "item_id"], keep="first").reset_index(drop=True)
    clean.timestamp = clean.timestamp.astype("int64")
    if not (100_000 <= len(clean) <= 2_000_000 and clean.user_id.nunique() >= 1000 and clean.item_id.nunique() >= 1000):
        raise ValueError("Clean data does not satisfy assignment size requirements")
    wanted = set(clean.item_id)
    metadata = {}
    metadata_rows = 0
    with gzip.open(cache / FILES[1], "rt", encoding="utf-8") as stream:
        for line in stream:
            metadata_rows += 1
            # Old UCSD product files are Python literals, not strict JSON. No eval().
            record = ast.literal_eval(line)
            asin = record.get("asin")
            if asin not in wanted or asin in metadata:
                continue
            paths = record.get("categories") or []
            category = next((str(p[1]) for p in paths if len(p) > 1),
                            next((str(p[0]) for p in paths if p), "UNKNOWN"))
            metadata[asin] = {"item_id": asin, "title": record.get("title") or "",
                              "category": category, "brand": record.get("brand") or ""}
    meta = pd.DataFrame(metadata.values())
    report = {"raw_interactions": len(raw), "clean_interactions": len(clean),
              "users": clean.user_id.nunique(), "items": clean.item_id.nunique(),
              "missing_by_field": raw.isna().sum().to_dict(),
              "duplicate_pairs": int(raw.duplicated(["user_id", "item_id"]).sum()),
              "invalid_or_missing_rows": len(raw) - len(raw.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
                  .loc[lambda d: d.rating.between(1, 5) & (d.timestamp > 0)]),
              "metadata_source_rows": metadata_rows, "metadata_items_matched": len(meta),
              "sparsity": 1 - len(clean) / (clean.user_id.nunique() * clean.item_id.nunique())}
    return raw, clean, meta, report


def chronological_split(clean):
    times = np.sort(clean.timestamp.to_numpy())
    t1 = int(times[int(0.8 * len(times))])
    t2 = int(times[int(0.9 * len(times))])
    raw_train = clean[clean.timestamp < t1].copy()
    raw_validation = clean[(clean.timestamp >= t1) & (clean.timestamp < t2)].copy()
    raw_test = clean[clean.timestamp >= t2].copy()
    users = np.sort(raw_train.user_id.unique())
    items = np.sort(raw_train.item_id.unique())
    user_index = {v: k for k, v in enumerate(users)}
    item_index = {v: k for k, v in enumerate(items)}

    def encode(frame):
        keep = frame.user_id.isin(user_index) & frame.item_id.isin(item_index)
        out = frame[keep].copy()
        out["u"] = out.user_id.map(user_index).astype("int64")
        out["i"] = out.item_id.map(item_index).astype("int64")
        return out.reset_index(drop=True)

    train, validation, test = map(encode, [raw_train, raw_validation, raw_test])
    shape = (len(users), len(items))
    positive_train_users = set(train.loc[train.rating >= 4, "u"])
    validation_users = np.sort([u for u in validation.loc[validation.rating >= 4, "u"].unique() if u in positive_train_users])
    test_users = np.sort([u for u in test.loc[test.rating >= 4, "u"].unique() if u in positive_train_users])
    report = {"train_before": t1, "validation_before": t2, "shape": list(shape),
              "validation_evaluation_users": len(validation_users), "test_evaluation_users": len(test_users)}
    for name, raw, warm in [("train", raw_train, train), ("validation", raw_validation, validation), ("test", raw_test, test)]:
        report[name] = {"raw_rows": len(raw), "warm_rows": len(warm),
                        "cold_user_rows": int((~raw.user_id.isin(user_index)).sum()),
                        "cold_item_rows": int((~raw.item_id.isin(item_index)).sum()),
                        "removed_rows_union": len(raw) - len(warm),
                        "positive_rows": int((warm.rating >= 4).sum())}
    assert raw_train.timestamp.max() < raw_validation.timestamp.min() <= raw_validation.timestamp.max() < raw_test.timestamp.min()
    return {"train": train, "validation": validation, "test": test, "shape": shape,
            "validation_users": validation_users, "test_users": test_users,
            "user_ids": users, "item_ids": items, "t1": t1, "t2": t2, "report": report}


def seen_matrix(frame, shape):
    return sparse.csr_matrix((np.ones(len(frame)), (frame.u, frame.i)), shape=shape)


def history_features(frame, shape, cutoff, metadata=True):
    nu, ni = shape
    user = [np.arange(nu, dtype=np.int64)]
    item = [nu + np.arange(ni, dtype=np.int64)]
    offset = nu + ni
    if metadata:
        for axis, size, destination in [("u", nu, user), ("i", ni, item)]:
            count = np.bincount(frame[axis], minlength=size)
            total = np.bincount(frame[axis], weights=frame.rating, minlength=size)
            mean = total / np.maximum(count, 1)
            # Fixed bin boundaries, no validation/test statistics or fitted quantiles.
            for values, edges in [(count, [1, 3, 5, 10, 20, 50, 100]),
                                  (mean, [2, 3, 3.5, 4, 4.5, 4.9])]:
                destination.append(offset + np.digitize(values, edges))
                offset += len(edges) + 1
        first = np.full(ni, cutoff, dtype=np.int64)
        np.minimum.at(first, frame.i.to_numpy(), frame.timestamp.to_numpy())
        item.append(offset + np.digitize((cutoff - first) / 86400, [30, 90, 180, 365, 730, 1460]))
        offset += 7
    return np.column_stack(user), np.column_stack(item), offset


class Recommender:
    def recommend(self, users, seen, k=10):
        users = np.asarray(users, dtype=np.int64)
        scores = np.asarray(self.score(users), dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Model produced non-finite scores")
        known = seen[users].tocoo()
        scores[known.row, known.col] = -np.inf
        if (np.isfinite(scores).sum(axis=1) < k).any():
            raise ValueError("Fewer than k unseen candidates")
        # При равных скорах выбираем меньший ID.
        return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def evaluate(model, heldout, users, seen, k=10, batch_size=128):
    relevance = heldout[heldout.rating >= 4].groupby("u").i.agg(set).to_dict()
    values = []
    discounts = 1 / np.log2(np.arange(2, k + 2))
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        top = model.recommend(batch, seen, k)
        for u, ranking in zip(batch, top):
            known = set(seen.indices[seen.indptr[u]:seen.indptr[u + 1]])
            relevant = relevance[u] - known
            if not relevant:
                raise ValueError("Нет релевантных товаров для оценки")
            hits = np.isin(ranking, list(relevant)).astype(float)
            ndcg = hits @ discounts / discounts[:min(k, len(relevant))].sum()
            recall = hits.sum() / len(relevant)
            values.append((ndcg, recall))
    ndcg, recall = np.asarray(values).mean(axis=0)
    return {"NDCG@10": float(ndcg), "Recall@10": float(recall)}
