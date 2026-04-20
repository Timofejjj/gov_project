"""Общие константы пайплайна v2 (согласованы с Pipline_1_New/run_pipeline.py где уместно)."""

SAMPLE_RATE = 16_000
MAX_ASR_CHUNK_SEC = 20.0
MIN_UTTERANCE_SEC = 0.12
EMBED_MIN_SEGMENT_SEC = 0.6
EMBED_FRAME_MS = 20

# VAD → сегменты (node_speech_windows в Pipline_1_New)
MERGE_MAX_GAP_SEC = 0.25
MERGE_MIN_LEN_SEC = 0.25
PAD_SEC = 0.05
MAX_SEGMENT_LEN_SEC = 45.0
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 90
VAD_SPEECH_PAD_MS = 30

# Склейка соседних фраз одного спикера в финальных turns (по умолчанию выключено для быстрых смен спикера)
REFINE_MERGE_GAP_SEC = 0.0
