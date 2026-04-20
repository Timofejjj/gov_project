"""Выбор устройства для torch (CUDA / MPS / CPU)."""

from __future__ import annotations

from typing import Callable

import torch


def resolve_compute_device(request: str) -> str:
    """
    - auto: cuda при наличии, иначе mps (Apple), иначе cpu
    - cuda / cuda:0 — как у torch; при недоступности CUDA → cpu с понижением
    - mps — при недоступности → cpu
    - cpu
    """
    r = (request or "auto").strip().lower()
    if r == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if r == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if r.startswith("cuda"):
        if torch.cuda.is_available():
            return request.strip()
        return "cpu"
    return "cpu"


def log_why_cpu_if_needed(request: str, resolved: str, log_fn: Callable[[str], None]) -> None:
    """Если остались на CPU при ожидании GPU — поясняем по логам PyTorch."""
    req = (request or "").strip().lower()
    if resolved != "cpu":
        return
    cuda_built = torch.version.cuda is not None
    avail = torch.cuda.is_available()
    if req == "auto" or req.startswith("cuda"):
        if not cuda_built:
            log_fn(
                "CUDA: в этой сборке PyTorch нет CUDA (torch.version.cuda is None) — "
                "нужен пакет torch+cu… с https://pytorch.org/get-started/locally/"
            )
        elif not avail:
            log_fn(
                "CUDA: в сборке есть CUDA, но torch.cuda.is_available()=False — "
                "проверьте драйвер NVIDIA и совместимость версии CUDA с PyTorch"
            )


def describe_device(device_str: str) -> str:
    dev = torch.device(device_str)
    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            idx = dev.index if dev.index is not None else 0
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "cuda"
        return f"{device_str} ({name})"
    if dev.type == "mps":
        return "mps (Apple GPU)"
    if dev.type == "cpu" and torch.version.cuda is None:
        return "cpu (PyTorch без CUDA — установите torch+cu… для GPU)"
    return "cpu"
