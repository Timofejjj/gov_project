"""
LLM-постобработка стенограммы (один проход), без импорта Pipline_1_New.run_pipeline
(тот тянет SpeechBrain/pyannote и ломает lazy-import k2_fsa).

Промпт первого прохода: _LLM_SPEAKER_SYSTEM. Второй проход (refine) отключён.

API: только OpenAI-совместимый клиент через OPENAI_API_KEY (+ опционально OPENAI_BASE_URL / LLM_BASE_URL).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

_LLM_SPEAKER_SYSTEM = """Ты — эксперт-редактор стенограмм. Твоя задача — превратить сырой ASR-вывод в структурированный диалог с согласованными спикерами.

РАБОТАЙ ТОЛЬКО С ТЕМ ТЕКСТОМ, КОТОРЫЙ ЕСТЬ ВО ВХОДНОМ JSON.
Нельзя добавлять новые слова, перефразировать, исправлять грамматику или выдумывать смысл.

ОБЩАЯ ЦЕЛЬ
1) Разделить слитые реплики (см. правила split ниже).
2) Назначить каждой реплике одного из фиксированных спикеров.
3) Сохранить текст максимально близко к ASR, удаляя только ведущие тире и лишние пробелы.

ВХОД
Во входе может быть:
- массив объектов с полями id, text, speaker и/или другими метаданными;
- num_speakers_hint — ожидаемое число спикеров;
- speakers_observed — уже встречающиеся метки.

ПРАВИЛА ВЫБОРА СПИКЕРОВ
1) Сначала прочитай весь диалог целиком.
2) Определи, сколько людей реально говорит.
3) Выбери ОДИН фиксированный набор спикеров и используй его во всём ответе.
4) Если в тексте явно есть имена или роли — используй их.
5) Если имён нет — используй нейтральные роли:
   ROLE_1, ROLE_2, ... ROLE_N
6) Если num_speakers_hint задан как целое N > 0, то в финальном ответе должно быть РОВНО N различных значений speaker.
7) В финальном JSON запрещены любые значения speaker вида SPEAKER_1, SPEAKER_2 и т.п.
8) Нельзя добавлять новых спикеров по ходу ответа.

ПРАВИЛО РАЗБИЕНИЯ ПО ТИРЕ
Тирe в начале реплики — это обязательный маркер отдельной реплики.

Считать маркером нужно только тире, если оно стоит:
- в начале текста,
- или сразу после явного конца предыдущей реплики/фразы.

Если в одном text встречается несколько реплик с тире, например:
"— Да. — А вы? — Хорошо."
то нужно разбить это на несколько объектов, по одному на каждую реплику.

Если тире стоит внутри фразы, например:
"Ну да — а вы?"
то это не маркер смены спикера, split не делать.

ПРАВИЛО РАЗБИЕНИЯ ПО ПРЕДЛОЖЕНИЯМ (БЕЗ ТИРЕ)
Иногда ASR склеивает несколько реплик в один text без тире. В этом случае разрешено разбивать по границам предложений:
- потенциальный фрагмент-реплика часто начинается с заглавной буквы (А-Я/Ё или A-Z) и заканчивается точкой ".";
- если подряд идут несколько таких завершённых предложений, их МОЖНО разделить на отдельные объекты.

Ограничения и здравый смысл:
- Не делай split внутри сокращений/аббревиатур (например: "г.", "ул.", "т.д.", "и т.п.", "др.", "т.е.", инициалы).
- Не режь предложения, если это похоже на одну реплику одного человека (единая мысль без смены роли).
- Делай split по предложениям только если по смыслу и по глобальному контексту диалога видно, что это разные реплики (часто — чередование вопрос/ответ).

РАСПРЕДЕЛЕНИЕ РЕПЛИК МЕЖДУ СПИКЕРАМИ
1) Используй смысл диалога: вопрос/ответ, обращения, согласие, отрицание, контекст соседних реплик.
2) Если после split один исходный сегмент дал несколько реплик, обычно это чередование говорящих.
3) Назначай спикеров последовательно и согласованно по всему диалогу.
4) Меняй распределение только при явной логической ошибке.
5) Если число реально различимых говорящих отличается от num_speakers_hint:
   - если говорящих больше, аккуратно объединяй очевидные дубликаты;
   - если говорящих меньше, не выдумывай лишних без явных оснований.
6) Приоритет всегда у логики диалога и строгого ограничения по числу спикеров.

ЧТО МОЖНО И ЧТО НЕЛЬЗЯ МЕНЯТЬ
Можно:
- убрать ведущие тире;
- убрать лишние пробелы вокруг реплики;
- разбить один исходный объект на несколько, если там несколько реплик.

Нельзя:
- добавлять новые слова;
- исправлять стиль, пунктуацию, орфографию;
- менять смысл;
- объединять реплики без необходимости.

ФОРМАТ ВЫХОДА
Верни СТРОГО JSON-массив.

Каждый объект должен иметь вид:
{
  "id": <int>,
  "speaker": "<speaker>",
  "text": "<text>",
  "source_ids": [<int>]
}

ТРЕБОВАНИЯ К JSON
- Только JSON-массив, без markdown, без пояснений, без обёрток.
- source_ids всегда содержит только исходный id этого сегмента: [id].
- text должен содержать реплику без ведущего тире.
- speaker должен быть одним из заранее выбранных спикеров.
- В ответе не должно быть лишних полей.
- В ответе должно быть ровно N уникальных speaker, если num_speakers_hint = N.

ФИНАЛЬНАЯ ПРОВЕРКА ПЕРЕД ОТВЕТОМ
Проверь:
- все ли тире-реплики разделены;
- нет ли speaker вида SPEAKER_i;
- соблюдено ли число уникальных спикеров;
- соответствует ли распределение логике диалога;
- сохранён ли текст без лишних изменений.

Возвращай только итоговый JSON-массив.
"""


_LLM_REFINE_SYSTEM = """Ты — строгий редактор JSON-стенограммы (второй проход, refine).

У тебя есть:
1) исходный ASR (список реплик с id/start/end/duration_sec/text/speaker)
2) черновик правок после первого прохода (JSON-массив объектов с полями id/speaker/text/source_ids)

ОГРАНИЧЕНИЕ ПО ЧИСЛУ СПИКЕРОВ:
- Во входных метаданных может быть num_speakers_hint. Если это целое число N > 0, то в финальном ответе ДОЛЖНО быть ровно N различных значений speaker.

ГЛАВНОЕ ТРЕБОВАНИЕ ЭТОГО ПРОХОДА — УБРАТЬ МЕТКИ SPEAKER_i:
- В черновике (pass1) speaker может быть вида SPEAKER_1 … SPEAKER_10.
- В финальном ответе НЕ ДОЛЖНО быть ни одного значения speaker вида SPEAKER_i.
- Вместо этого используй РОЛИ (например: "Оператор", "Клиент", "Сотрудник", "Заявитель", "Интервьюер", "Интервьюируемый")
  или нейтральные роли "ROLE_1", "ROLE_2", ... "ROLE_N" (если по тексту нельзя понять конкретную роль).

ВАЖНО: SPEAKER_i МОГУТ БЫТЬ УЖЕ РАСПРЕДЕЛЕНЫ ПРАВИЛЬНО — ОПИРАЙСЯ НА НИХ:
- Относись к pass1 как к исходной разметке: если SPEAKER_1 и SPEAKER_2 уже логично разделены, не “переигрывай” распределение.
- Твоя задача — сделать стабильное ПЕРЕИМЕНОВАНИЕ: выбери отображение SPEAKER_i → ROLE_j и применяй его последовательно по всему диалогу.
- Перераспределяй реплики между спикерами ТОЛЬКО если видишь явные логические ошибки (вопрос/ответ, обращения, противоречия контексту).

КАК РАСПРЕДЕЛЯТЬ РОЛИ, ЕСЛИ num_speakers_hint ЗАДАН:
- Используй ровно N ролей. Не создавай ролей больше N и не оставляй меньше N.
- Если в черновике уже есть K разных SPEAKER_i:
  - Если K == N: просто переименуй каждый SPEAKER_i в одну роль (1:1).
  - Если K > N: аккуратно объедини очевидные дубликаты (например, SPEAKER_3 по смыслу совпадает с SPEAKER_1), затем переименуй в N ролей.
  - Если K < N: только если это действительно видно по диалогу, раздели одного из SPEAKER_i на дополнительную роль; иначе используй нейтральные ROLE_* и распределяй минимально-инвазивно.

Задачи refine:
- Проверь логику диалога и согласованность спикеров/ролей.
- Исправь оставшиеся ошибки merge/split (source_ids), если они противоречат смыслу.
- Сохрани текст максимально близко к ASR/черновику. Не добавляй новые слова; можно только убрать ведущие маркеры тире/лишние пробелы.

ФОРМАТ ОТВЕТА:
Верни СТРОГО один JSON-массив объектов в том же формате, что и в первом проходе:
{
  "id": <int>,
  "speaker": "<ROLE>",
  "text": "<text>",
  "source_ids": [<int>]
}

Запрещено:
- Любые значения speaker вида SPEAKER_i в финальном JSON.
- Любые пояснения/markdown — только JSON-массив.
"""


def _llm_log(message: str, *, indent: int = 0) -> None:
    pad = "  " * max(0, indent)
    print(f"[P2][llm]{pad} {message}", flush=True)


def _extract_json_array(text: str) -> str:
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            pp = p.strip()
            if pp.startswith("{") or pp.startswith("["):
                s = pp
                break
    m = re.search(r"\[[\s\S]*\]\s*$", s)
    if m:
        return m.group(0).strip()
    i = s.find("[")
    j = s.rfind("]")
    if i >= 0 and j > i:
        return s[i : j + 1].strip()
    return s


def _llm_chat_json_array(
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    timeout_sec: float,
    base_url: str | None = None,
) -> list[dict]:
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    # Как в Pipline_1_New/run_pipeline.py: явный base_url → OPENAI_BASE_URL → LLM_BASE_URL.
    if not base_url:
        base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "").strip() or None

    if not openai_key:
        raise RuntimeError("Нужен OPENAI_API_KEY в .env")
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError("Для OPENAI_API_KEY установите пакет openai") from e
    kw: dict[str, Any] = {"api_key": openai_key, "timeout": timeout_sec}
    if base_url:
        kw["base_url"] = base_url
    client = OpenAI(**kw)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    content = (resp.choices[0].message.content or "").strip()

    raw = _extract_json_array(content)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise RuntimeError("LLM вернул не JSON-массив")
    out: list[dict] = []
    for it in data:
        if isinstance(it, dict):
            out.append(it)
    return out


def llm_speaker_correction(
    rows: list[dict],
    *,
    model: str,
    num_speakers: int | None,
    temperature: float,
    timeout_sec: float,
    second_pass: bool,
    base_url: str | None = None,
) -> list[dict]:
    if not rows:
        return rows

    # Второй проход (refine, _LLM_REFINE_SYSTEM) отключён — используется только pass1.
    second_pass = False

    _llm_log(
        f"старт: входных реплик={len(rows)}, model={model!r}, second_pass={bool(second_pass)}",
        indent=0,
    )

    indexed: list[dict] = []
    for i, r in enumerate(rows, start=1):
        try:
            s = float(r.get("start"))
            e = float(r.get("end"))
        except Exception:
            continue
        txt = str(r.get("text", "")).strip()
        if not txt:
            continue
        indexed.append(
            {
                "id": i,
                "start": round(s, 3),
                "end": round(e, 3),
                "duration_sec": round(max(0.0, e - s), 3),
                "speaker": str(r.get("speaker", "")).strip(),
                "text": txt,
            }
        )
    if not indexed:
        _llm_log("нет валидных строк для отправки (пустые/битые) → пропуск", indent=0)
        return rows

    speakers = sorted({str(x.get("speaker", "")).strip() for x in indexed if str(x.get("speaker", "")).strip()})
    meta_lines = [
        f"model={model}",
        f"num_speakers_hint={num_speakers if num_speakers is not None else 'unknown'}",
        f"speakers_observed={len(speakers)} ({', '.join(speakers)})",
    ]
    user_pass1 = "\n".join(
        [
            "Ниже JSON массива реплик (ASR). Исправь speaker/text.",
            "Метаданные:",
            *meta_lines,
            "",
            json.dumps(indexed, ensure_ascii=False),
        ]
    )

    _llm_log(f"pass1: отправка {len(indexed)} реплик…", indent=0)
    patch1 = _llm_chat_json_array(
        system=_LLM_SPEAKER_SYSTEM,
        user=user_pass1,
        model=model,
        temperature=temperature,
        timeout_sec=timeout_sec,
        base_url=base_url,
    )
    _llm_log(f"pass1: получено объектов={len(patch1)}", indent=0)

    patch_final = patch1
    if second_pass:
        draft = json.dumps(patch1, ensure_ascii=False)
        user_pass2 = "\n".join(
            [
                "Сделай второй проход (refine) по черновику.",
                "Метаданные:",
                *meta_lines,
                "",
                "ASR:",
                json.dumps(indexed, ensure_ascii=False),
                "",
                "Черновик (pass1):",
                draft,
            ]
        )
        _llm_log("pass2: refine…", indent=0)
        patch2 = _llm_chat_json_array(
            system=_LLM_REFINE_SYSTEM,
            user=user_pass2,
            model=model,
            temperature=temperature,
            timeout_sec=timeout_sec,
            base_url=base_url,
        )
        _llm_log(f"pass2: получено объектов={len(patch2)}", indent=0)
        patch_final = patch2 if patch2 else patch1

    by_id: dict[int, dict] = {}
    for x in indexed:
        try:
            by_id[int(x["id"])] = x
        except Exception:
            continue

    def _parse_src_ids(p: dict) -> list[int] | None:
        src = p.get("source_ids", None)
        if isinstance(src, list) and src:
            ids: list[int] = []
            for it in src:
                if str(it).strip().lstrip("-").isdigit():
                    ids.append(int(it))
            ids = sorted(set(ids))
            return ids if ids and all(i in by_id for i in ids) else None
        pid = p.get("id", None)
        if str(pid).strip().lstrip("-").isdigit():
            i = int(pid)
            return [i] if i in by_id else None
        return None

    class _DSU:
        def __init__(self) -> None:
            self.p: dict[int, int] = {}

        def find(self, x: int) -> int:
            self.p.setdefault(x, x)
            if self.p[x] != x:
                self.p[x] = self.find(self.p[x])
            return self.p[x]

        def union(self, a: int, b: int) -> None:
            ra, rb = self.find(a), self.find(b)
            if ra != rb:
                self.p[rb] = ra

    dsu = _DSU()
    for p in patch_final:
        src = _parse_src_ids(p)
        if not src or len(src) < 2:
            continue
        a0 = src[0]
        for b in src[1:]:
            dsu.union(a0, b)

    merge_root: dict[int, int] = {i: dsu.find(i) for i in by_id.keys()}

    singles: dict[int, list[dict]] = {}
    multis: list[tuple[tuple[int, ...], dict]] = []
    for p in patch_final:
        src = _parse_src_ids(p)
        if not src:
            continue
        if len(src) == 1:
            singles.setdefault(src[0], []).append(p)
        else:
            multis.append((tuple(src), p))

    consumed: set[int] = set()
    rebuilt: list[dict] = []

    def _emit_split(i: int, parts: list[dict]) -> None:
        base = by_id[i]
        t0 = float(base["start"])
        t1 = float(base["end"])
        duration = t1 - t0

        valid_parts = [p for p in parts if str(p.get("text", "")).strip()]
        if not valid_parts:
            rebuilt.append(
                {
                    "speaker": str(base.get("speaker", "")),
                    "start": round(t0, 3),
                    "end": round(t1, 3),
                    "text": str(base.get("text", "")),
                }
            )
            return

        total_chars = sum(len(str(p.get("text", ""))) for p in valid_parts)

        cur_t = t0
        for j, p in enumerate(valid_parts):
            p_text = str(p.get("text", "")).strip()
            p_spk = str(p.get("speaker", "")).strip() or str(base.get("speaker", ""))

            clean_text = re.sub(r"^[—\-\s]+", "", p_text).capitalize()

            share = len(p_text) / total_chars
            seg_dur = share * duration

            seg_end = cur_t + seg_dur
            if j < len(valid_parts) - 1:
                actual_end = max(cur_t + 0.1, seg_end - 0.05)
            else:
                actual_end = t1

            rebuilt.append(
                {
                    "speaker": p_spk,
                    "start": round(cur_t, 3),
                    "end": round(actual_end, 3),
                    "text": clean_text,
                }
            )
            cur_t = actual_end + 0.05

    def _emit_merge(ids: list[int], parts: list[dict]) -> None:
        ids = sorted(set(ids))
        t0 = min(float(by_id[i]["start"]) for i in ids)
        t1 = max(float(by_id[i]["end"]) for i in ids)
        texts: list[str] = []
        spk = ""
        for p in parts:
            tx = str(p.get("text", "")).strip()
            if tx:
                texts.append(tx)
            sp = str(p.get("speaker", "")).strip()
            if sp and not spk:
                spk = sp
        merged_text = " ".join(texts).strip()
        if not merged_text:
            merged_text = " ".join(str(by_id[i].get("text", "")) for i in ids).strip()
        if not spk:
            spk = str(by_id[ids[0]].get("speaker", ""))
        rebuilt.append({"speaker": spk, "start": round(t0, 3), "end": round(t1, 3), "text": merged_text})

    roots_done: set[int] = set()
    for i in sorted(by_id.keys()):
        r = merge_root[i]
        if r in roots_done:
            continue
        members = sorted({k for k, rr in merge_root.items() if rr == r})
        if len(members) <= 1:
            continue
        roots_done.add(r)

        mparts: list[dict] = []
        for tup, p in multis:
            st = set(tup)
            if st.issubset(set(members)) and len(st) > 1:
                mparts.append(p)
        if not mparts:
            continue

        _emit_merge(members, mparts)
        consumed.update(members)

    for i in sorted(by_id.keys()):
        if i in consumed:
            continue
        parts = singles.get(i, [])
        if len(parts) >= 2:
            _emit_split(i, parts)
            consumed.add(i)
            continue
        if len(parts) == 1:
            p = parts[0]
            base = by_id[i]
            sp = str(p.get("speaker", "")).strip() or str(base.get("speaker", ""))
            tx = str(p.get("text", "")).strip() or str(base.get("text", ""))
            rebuilt.append(
                {
                    "speaker": sp,
                    "start": round(float(base["start"]), 3),
                    "end": round(float(base["end"]), 3),
                    "text": tx,
                }
            )
            consumed.add(i)
            continue

        base = by_id[i]
        rebuilt.append(
            {
                "speaker": str(base.get("speaker", "")),
                "start": round(float(base["start"]), 3),
                "end": round(float(base["end"]), 3),
                "text": str(base.get("text", "")),
            }
        )
        consumed.add(i)

    if not rebuilt:
        _llm_log("выход пустой после сборки → оставляем исходные строки", indent=0)
        return rows

    rebuilt.sort(key=lambda r: (float(r["start"]), float(r["end"])))
    _llm_log(f"готово: выходных реплик={len(rebuilt)}", indent=0)
    return rebuilt
