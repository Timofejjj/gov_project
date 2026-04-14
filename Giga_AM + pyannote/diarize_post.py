"""Постобработка диаризации: overlap, склейка одного спикера, сглаживание границ."""
from __future__ import annotations


def resolve_overlaps(
    turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """
    Пересекающиеся интервалы разных спикеров режутся на атомарные отрезки;
    на каждом отрезке выбирается спикер с наибольшим перекрытием (tie: более ранний start).
    """
    if not turns:
        return []
    pts: set[float] = set()
    for t0, t1, _ in turns:
        pts.add(float(t0))
        pts.add(float(t1))
    bounds = sorted(pts)
    atomic: list[tuple[float, float, str]] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 1e-9:
            continue
        best_ov = -1.0
        best_start = -1.0
        best_spk: str | None = None
        for t0, t1, spk in turns:
            if t0 >= b or t1 <= a:
                continue
            ov = min(t1, b) - max(t0, a)
            st = float(t0)
            # при равном перекрытии — более поздний старт сегмента (типичный «кто перебил»)
            if ov > best_ov + 1e-12 or (
                abs(ov - best_ov) < 1e-12 and (best_spk is None or st > best_start + 1e-12)
            ):
                best_ov = ov
                best_start = st
                best_spk = str(spk)
        if best_spk is not None:
            atomic.append((a, b, best_spk))
    return _merge_touching_same_speaker(atomic)


def _merge_touching_same_speaker(
    turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    out: list[list[float | str]] = []
    for t0, t1, spk in turns:
        if not out:
            out.append([t0, t1, spk])
            continue
        p0, p1, ps = out[-1]
        if abs(float(p1) - t0) < 1e-6 and ps == spk:
            out[-1][1] = max(float(p1), t1)
        else:
            out.append([t0, t1, spk])
    return [(float(a), float(b), str(c)) for a, b, c in out]


def merge_adjacent_same_speaker(
    turns: list[tuple[float, float, str]],
    max_gap_sec: float,
) -> list[tuple[float, float, str]]:
    """Склейка подряд идущих кусков одного спикера при паузе не больше max_gap_sec."""
    if not turns:
        return []
    turns = sorted(turns, key=lambda x: (x[0], x[1]))
    merged: list[tuple[float, float, str]] = [turns[0]]
    for t0, t1, spk in turns[1:]:
        p0, p1, ps = merged[-1]
        if spk == ps and t0 - p1 <= max_gap_sec + 1e-9:
            merged[-1] = (p0, max(p1, t1), ps)
        else:
            merged.append((t0, t1, spk))
    return merged


def refine_segment_boundaries(
    turns: list[tuple[float, float, str]],
    *,
    min_segment_sec: float,
    max_intrusion_sec: float,
    pad_sec: float,
) -> list[tuple[float, float, str]]:
    """
    Сглаживание: слияние очень коротких «врезок» в соседа, лёгкий pad границ,
    повторная склейка одного спикера после pad.
    """
    if not turns:
        return []
    turns = sorted(turns, key=lambda x: (x[0], x[1]))
    absorbed = absorb_short_segments(turns, max_dur_sec=max_intrusion_sec)
    padded: list[tuple[float, float, str]] = []
    for t0, t1, spk in absorbed:
        padded.append((max(0.0, t0 - pad_sec), t1 + pad_sec, spk))
    padded = merge_adjacent_same_speaker(padded, max_gap_sec=0.0)
    padded = absorb_short_segments(padded, max_dur_sec=min_segment_sec)
    final: list[tuple[float, float, str]] = []
    for t0, t1, spk in padded:
        if t1 - t0 >= min_segment_sec * 0.5:
            final.append((t0, t1, spk))
    return merge_adjacent_same_speaker(final, max_gap_sec=0.25)


def absorb_short_segments(
    turns: list[tuple[float, float, str]],
    max_dur_sec: float,
) -> list[tuple[float, float, str]]:
    """
    Сегменты короче max_dur_sec сливаются с соседом (слева предпочтительно, иначе справа),
    чтобы не гонять ASR на микро-куски и убрать шумные врезки.
    """
    if not turns:
        return []
    items = sorted([(float(a), float(b), str(c)) for a, b, c in turns], key=lambda x: (x[0], x[1]))

    def merge_left(short: tuple[float, float, str], seq: list[tuple[float, float, str]]):
        t0, t1, spk = short
        if not seq:
            return
        p0, p1, ps = seq[-1]
        seq[-1] = (p0, max(p1, t1), ps)

    # L → R: вклеить короткий в предыдущий интервал (один спикер слева «захватывает» время)
    left_pass: list[tuple[float, float, str]] = []
    for seg in items:
        t0, t1, spk = seg
        if t1 - t0 >= max_dur_sec:
            left_pass.append((t0, t1, spk))
        else:
            if left_pass:
                merge_left(seg, left_pass)
            else:
                left_pass.append((t0, t1, spk))

    # R → L: оставшиеся короткие у начала цепочки вклеиваем вправо
    right_pass: list[tuple[float, float, str]] = []
    for seg in reversed(left_pass):
        t0, t1, spk = seg
        if t1 - t0 >= max_dur_sec:
            right_pass.append((t0, t1, spk))
        else:
            if right_pass:
                n0, n1, ns = right_pass[-1]
                right_pass[-1] = (min(n0, t0), n1, ns)
            else:
                right_pass.append((t0, t1, spk))
    out = list(reversed(right_pass))
    return _merge_touching_same_speaker(out)


def postprocess_diarization_turns(
    turns: list[tuple[float, float, str]],
    *,
    min_segment_sec: float,
    merge_same_speaker_gap_sec: float,
    max_intrusion_sec: float,
    boundary_pad_sec: float,
) -> list[tuple[float, float, str]]:
    t = resolve_overlaps(turns)
    t = merge_adjacent_same_speaker(t, max_gap_sec=merge_same_speaker_gap_sec)
    t = refine_segment_boundaries(
        t,
        min_segment_sec=min_segment_sec,
        max_intrusion_sec=max_intrusion_sec,
        pad_sec=boundary_pad_sec,
    )
    t = merge_adjacent_same_speaker(t, max_gap_sec=merge_same_speaker_gap_sec)
    return t
