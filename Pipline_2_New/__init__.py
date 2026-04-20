"""Pipline_2_New: диаризация через согласование ECAPA + ASR + время + граф + LLM (без pyannote)."""

from Pipline_2_New.run_pipeline import build_turns, run_multimodal_pipeline

__all__ = ["run_multimodal_pipeline", "build_turns"]
