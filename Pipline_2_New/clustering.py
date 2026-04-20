"""Первичная кластеризация по эмбеддингам (HDBSCAN / Agglomerative) и пост-очистка по центроидам."""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    method: str,
    num_speakers: Optional[int],
    agglomerative_threshold: float,
    hdbscan_min_cluster_size: int,
) -> np.ndarray:
    """Возвращает метки 0..K-1 для каждой строки embeddings (n, dim)."""
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == 1:
        return np.zeros((1,), dtype=np.int64)

    X = _l2_normalize_rows(embeddings.astype(np.float32))

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("Для clusterer=hdbscan установите пакет hdbscan") from e
        cl = hdbscan.HDBSCAN(
            min_cluster_size=max(2, hdbscan_min_cluster_size),
            metric="euclidean",
            cluster_selection_epsilon=0.0,
        )
        labels = cl.fit_predict(X)
        noise = labels == -1
        if np.all(noise):
            labels = np.zeros(len(labels), dtype=np.int64)
        else:
            next_id = int(labels.max()) + 1
            for i in range(len(labels)):
                if labels[i] == -1:
                    labels[i] = next_id
                    next_id += 1
        return np.asarray(labels, dtype=np.int64)

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity

    S = cosine_similarity(X)
    D = 1.0 - np.clip(S, -1.0, 1.0)
    np.fill_diagonal(D, 0.0)
    if num_speakers is not None and int(num_speakers) > 0:
        k_req = int(num_speakers)
        k = min(k_req, n)
        if k < 2:
            return np.zeros((n,), dtype=np.int64)
        if k < k_req:
            warnings.warn(
                f"num_speakers={k_req}, но векторов всего {n} — "
                f"нельзя больше кластеров, чем сегментов; используется n_clusters={k}",
                UserWarning,
                stacklevel=2,
            )
        agg = AgglomerativeClustering(
            n_clusters=k,
            linkage="average",
            metric="precomputed",
        )
        return agg.fit_predict(D).astype(np.int64)
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=agglomerative_threshold,
        linkage="average",
        metric="precomputed",
    )
    return agg.fit_predict(D).astype(np.int64)


def merge_clusters_by_centroid(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    merge_cosine_min: float = 0.92,
) -> np.ndarray:
    """Объединяет кластеры, если косинус между центроидами выше порога."""
    labels = labels.astype(np.int64).copy()
    uniq = sorted(set(labels.tolist()))
    if len(uniq) <= 1:
        return labels
    X = _l2_normalize_rows(embeddings.astype(np.float32))
    changed = True
    while changed:
        changed = False
        uniq = sorted(set(labels.tolist()))
        cents: List[np.ndarray] = []
        for u in uniq:
            m = labels == u
            if not np.any(m):
                cents.append(np.zeros(X.shape[1], dtype=np.float32))
                continue
            c = X[m].mean(axis=0)
            n = float(np.linalg.norm(c) + 1e-9)
            cents.append((c / n).astype(np.float32))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                sim = float(np.dot(cents[i], cents[j]))
                if sim >= merge_cosine_min:
                    a, b = uniq[i], uniq[j]
                    labels[labels == b] = a
                    changed = True
                    break
            if changed:
                break
    # перенумеровать 0..K-1
    mapping = {}
    nxt = 0
    for u in sorted(set(labels.tolist())):
        mapping[u] = nxt
        nxt += 1
    return np.array([mapping[int(x)] for x in labels.tolist()], dtype=np.int64)


def nearest_cluster_labels(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """centroids: (K, dim) L2-normalized."""
    X = _l2_normalize_rows(embeddings.astype(np.float32))
    sims = X @ centroids.T
    return np.argmax(sims, axis=1).astype(np.int64)


def merge_speaker_labels_by_embedding(
    rows: list[dict],
    embeddings: np.ndarray,
    *,
    cos_thresh: float = 0.92,
) -> list[dict]:
    """
    Optional cleanup после LLM: если метки «разъехались», но центроиды близки — union по порогу косинуса.
    Требует len(rows) == embeddings.shape[0] (по одному эмбеддингу на сегмент).
    """
    if len(rows) != int(embeddings.shape[0]):
        return rows
    n = len(rows)
    labels = [str(r.get("speaker", "")) for r in rows]
    uniq: list[str] = []
    for u in labels:
        if u not in uniq:
            uniq.append(u)
    if len(uniq) <= 1:
        return rows
    X = _l2_normalize_rows(embeddings.astype(np.float32))
    cents: dict[str, np.ndarray] = {}
    for u in uniq:
        idx = [i for i in range(n) if labels[i] == u]
        v = X[idx].mean(axis=0)
        nv = float(np.linalg.norm(v) + 1e-9)
        cents[u] = (v / nv).astype(np.float32)

    parent = {u: u for u in uniq}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a, b = uniq[i], uniq[j]
            if float(np.dot(cents[a], cents[b])) >= cos_thresh:
                union(a, b)

    root_map = {u: find(u) for u in uniq}
    canon: dict[str, str] = {}
    for u in uniq:
        r = root_map[u]
        members = [x for x in uniq if root_map[x] == r]
        canon[u] = min(members)

    out: list[dict] = []
    for r, row in zip(labels, rows):
        rr = dict(row)
        rr["speaker"] = canon.get(r, r)
        out.append(rr)
    return out
