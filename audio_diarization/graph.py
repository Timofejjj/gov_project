from __future__ import annotations

import os
from typing import Literal

from langgraph.graph import END, StateGraph

from audio_diarization.nodes import (
    fail_fast,
    poll_once,
    start_transcription,
    upload_wav,
    wait_between_polls,
)
from audio_diarization.state import DiarizationState


def _after_poll(state: DiarizationState) -> Literal["done", "fail", "wait"]:
    s = state.get("job_status", "")
    if s == "completed":
        return "done"
    if s == "error":
        return "fail"
    return "wait"


def build_graph():
    g = StateGraph(DiarizationState)
    g.add_node("upload_wav", upload_wav)
    g.add_node("start_transcription", start_transcription)
    g.add_node("poll_once", poll_once)
    g.add_node("wait_between_polls", wait_between_polls)
    g.add_node("fail_fast", fail_fast)

    g.set_entry_point("upload_wav")
    g.add_edge("upload_wav", "start_transcription")
    g.add_edge("start_transcription", "poll_once")
    g.add_conditional_edges(
        "poll_once",
        _after_poll,
        {
            "done": END,
            "fail": "fail_fast",
            "wait": "wait_between_polls",
        },
    )
    g.add_edge("wait_between_polls", "poll_once")
    g.add_edge("fail_fast", END)

    return g.compile()


def run_diarization(local_wav_path: str) -> DiarizationState:
    """Удобная обёртка: путь к WAV → финальное состояние графа."""
    if not os.path.isfile(local_wav_path):
        raise FileNotFoundError(local_wav_path)
    app = build_graph()
    return app.invoke({"local_wav_path": local_wav_path})
