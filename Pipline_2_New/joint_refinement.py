"""Joint refinement: α·cos(emb) + β·text_similarity + γ·temporal_proximity → граф → спектральная кластеризация."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Tuple

import numpy as np

def _midpoint(t0: float, t1: float) -> float:
    return 0.5 * (float(t0) + float(t1))


def text_similarity(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def joint_similarity_matrix(
    embeddings: np.ndarray,
    texts: List[str],
    intervals: List[Tuple[float, float]],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    temporal_sigma_sec: float = 3.0,
) -> np.ndarray:
    """Симметричная матрица сходства (n, n), диагональ 0."""
    n = embeddings.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    X = embeddings.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    mids = np.array([_midpoint(t0, t1) for t0, t1 in intervals], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            cos_ij = float(np.dot(Xn[i], Xn[j]))
            cos_part = (cos_ij + 1.0) * 0.5
            txt = text_similarity(texts[i], texts[j])
            dt = abs(float(mids[i] - mids[j]))
            temp = float(np.exp(-dt / max(1e-6, temporal_sigma_sec)))
            s = alpha * cos_part + beta * txt + gamma * temp
            s = max(0.0, min(1.0, s))
            out[i, j] = s
            out[j, i] = s
    return out


def graph_partition_labels(
    similarity: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
) -> np.ndarray:
    """Спектральная кластеризация по предвычисленной матрице сходства."""
    n = similarity.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == 1:
        return np.zeros((1,), dtype=np.int64)
    k = int(max(1, min(n_clusters, n)))
    W = np.clip(similarity.astype(np.float64), 0.0, None)
    np.fill_diagonal(W, 0.0)
    W = W + 1e-6
    try:
        from sklearn.cluster import SpectralClustering

        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=random_state,
            assign_labels="kmeans",
        )
        return sc.fit_predict(W).astype(np.int64)
    except Exception:
        # fallback: использовать метки по порядку как один кластер
        return np.zeros((n,), dtype=np.int64)
