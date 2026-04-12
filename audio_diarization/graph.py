from __future__ import annotations

import os
from typing import Literal

from langgraph.graph import END, StateGraph

from audio_diarization.nodes import fail_fast, transcribe_streaming
from audio_diarization.state import DiarizationState


def _after_stream(state: DiarizationState) -> Literal["done", "fail"]:
    if state.get("error"):
        return "fail"
    return "done"


def build_graph():
    g = StateGraph(DiarizationState)
    g.add_node("transcribe_streaming", transcribe_streaming)
    g.add_node("fail_fast", fail_fast)

    g.set_entry_point("transcribe_streaming")
    g.add_conditional_edges(
        "transcribe_streaming",
        _after_stream,
        {
            "done": END,
            "fail": "fail_fast",
        },
    )
    g.add_edge("fail_fast", END)

    return g.compile()


def run_diarization(local_wav_path: str) -> DiarizationState:
    """Путь к WAV → PCM 16 kHz mono → AssemblyAI Streaming (Whisper RT) с параметрами UI."""
    if not os.path.isfile(local_wav_path):
        raise FileNotFoundError(local_wav_path)
    app = build_graph()
    return app.invoke({"local_wav_path": local_wav_path})
