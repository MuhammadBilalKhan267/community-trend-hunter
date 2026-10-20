"""Embedding-input deduplication and statistical candidate grouping."""
from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


def unique_units(units: list[dict]) -> tuple[list[str], np.ndarray]:
    texts, lookup, indices = [], {}, []
    for unit in units:
        text = unit["text"]
        if text not in lookup:
            lookup[text] = len(texts)
            texts.append(text)
        indices.append(lookup[text])
    return texts, np.asarray(indices, dtype=np.int32)


def cluster_vectors(vectors: np.ndarray, target_size: int = 80) -> tuple[np.ndarray, np.ndarray]:
    n = len(vectors)
    if n == 0:
        return np.empty(0, dtype=int), np.empty((0, 0), dtype=np.float32)
    if n < 8:
        return np.arange(n), vectors.copy()
    # A computational microcluster count, not a fixed topic taxonomy. Sparse
    # candidates can survive; every proposed topic still requires review.
    k = min(384, max(8, math.ceil(n / target_size)))
    k = min(k, n)
    with threadpool_limits(limits=4):
        estimator = MiniBatchKMeans(n_clusters=k, batch_size=1024, random_state=42,
                                   n_init=3, max_iter=100, reassignment_ratio=.005)
        labels = estimator.fit_predict(vectors)
    centers = estimator.cluster_centers_.astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    return labels, centers
