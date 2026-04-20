"""Временное сглаживание последовательности спикеров по сегментам (не Viterbi как основа).

Опционально: лёгкий HMM (hmmlearn) по дискретным меткам.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

LabelSeq = List[int]
Intervals = List[Tuple[float, float]]


def median_smooth_labels(
    labels: Sequence[int],
    *,
    window: int = 3,
) -> List[int]:
    labels = list(labels)
    n = len(labels)
    if n == 0 or window <= 1:
        return labels
    w = window | 1
    r = w // 2
    out: List[int] = []
    for i in range(n):
        lo = max(0, i - r)
        hi = min(n, i + r + 1)
        chunk = sorted(labels[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def remove_micro_switches(
    intervals: Intervals,
    labels: Sequence[int],
    *,
    min_run_sec: float = 0.35,
) -> List[int]:
    """Короткие изолированные «островки» меток сливаются с соседями."""
    labels = list(labels)
    n = len(labels)
    if n <= 2:
        return labels
    durs = [max(0.0, float(b - a)) for a, b in intervals]
    out = labels[:]
    changed = True
    while changed:
        changed = False
        for i in range(1, n - 1):
            if out[i - 1] == out[i + 1] and out[i] != out[i - 1] and durs[i] < min_run_sec:
                out[i] = out[i - 1]
                changed = True
    return out


def hmm_smooth_labels(
    labels: Sequence[int],
    intervals: Intervals,
    *,
    n_iter: int = 30,
) -> List[int]:
    """Опционально: CategoricalHMM (hmmlearn) по последовательности дискретных меток."""
    del intervals
    try:
        from hmmlearn.hmm import CategoricalHMM  # type: ignore[import-untyped]
    except ImportError:
        return list(labels)

    y = np.asarray(list(labels), dtype=np.int64)
    if y.size == 0:
        return []
    uniq = sorted(set(y.tolist()))
    K = len(uniq)
    if K <= 1:
        return list(labels)
    remap = {u: i for i, u in enumerate(uniq)}
    inv = {i: u for u, i in remap.items()}
    obs = np.array([[remap[int(v)]] for v in y.tolist()], dtype=np.int64)
    lengths = np.array([len(obs)], dtype=np.int64)
    n_symbols = K
    model = CategoricalHMM(
        n_components=K,
        n_iter=n_iter,
        random_state=0,
        n_features=n_symbols,
    )
    try:
        model.fit(obs, lengths)
        _, st = model.decode(obs, lengths)
    except Exception:
        return list(labels)
    return [inv[int(x)] for x in st.reshape(-1).tolist()]
