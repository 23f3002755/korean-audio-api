import os
import json
import re
import base64
import hashlib
import asyncio
from statistics import mean, median, pstdev, pvariance, mode

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

EMAIL = "23f3002755@ds.study.iitm.ac.in"
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_BASE = "https://aipipe.org/openai/v1"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

HEAD = {
    "Authorization": f"Bearer {AIPIPE_TOKEN}",
    "Content-Type": "application/json",
}

_CACHE = {}

last_debug_info = {}
last_audio_bytes = b""
last_audio_mime = "audio/wav"
audio_history = []


def cache_key(*parts):
    text = "||".join(map(str, parts))
    return hashlib.sha256(text.encode()).hexdigest()


def parse_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0)) if match else {}


async def chat(messages, model="gpt-4o", max_tokens=1500, retries=4):
    key = cache_key(
        "chat",
        model,
        json.dumps(messages, sort_keys=True, default=str)
    )

    if key in _CACHE:
        return _CACHE[key]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    last_error = ""

    async with httpx.AsyncClient(timeout=90) as client:
        for attempt in range(retries):
            response = await client.post(
                f"{AIPIPE_BASE}/chat/completions",
                headers=HEAD,
                json=payload,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:160]}"
                await asyncio.sleep(1.5 * (attempt + 1))
                continue

            response.raise_for_status()

            result = response.json()["choices"]["message"]["content"]
            _CACHE[key] = result
            return result

    raise RuntimeError(f"Chat failed: {last_error}")


GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]


async def gemini_transcribe(payload, debug, attempts_per_model=3):
    last_error = ""

    async with httpx.AsyncClient(timeout=120) as client:
        for model_name in GEMINI_MODELS:
            for attempt in range(attempts_per_model):
                try:
                    response = await client.post(
                        f"https://aipipe.org/geminiv1beta/models/{model_name}:generateContent",
                        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
                        json=payload,
                    )

                    if response.status_code in (429, 500, 502, 503, 504):
                        last_error = (
                            f"HTTP {response.status_code} on {model_name}: "
                            f"{response.text[:160]}"
                        )
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue

                    response.raise_for_status()

                    text = (
                        response.json()["candidates"]["content"]["parts"]["text"]
                        .strip()
                    )

                    debug["transcribe_model"] = model_name
                    return text

                except (KeyError, IndexError, TypeError):
                    last_error = f"Empty or invalid response on {model_name}"
                    break

                except Exception as error:
                    last_error = (
                        f"{type(error).__name__} on {model_name}: {str(error)[:160]}"
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))

    debug["transcribe_error"] = last_error
    return ""


def find_audio_base64(body):
    audio_id = None
    audio_base64 = ""

    if isinstance(body, dict):
        for key, value in body.items():
            lowered = str(key).lower()

            if isinstance(value, str):
                if (
                    ("audio" in lowered or "data" in lowered
                     or "b64" in lowered or "base64" in lowered)
                    and len(value) > 200
                ):
                    if len(value) > len(audio_base64):
                        audio_base64 = value

                elif "id" in lowered and not audio_id:
                    audio_id = value

    return audio_id, audio_base64


def detect_mime(audio):
    if audio.startswith(b"ID3") or audio[:2] in (
        b"\xff\xfb",
        b"\xff\xf3",
        b"\xff\xf2",
    ):
        return "audio/mp3"

    if audio.startswith(b"OggS"):
        return "audio/ogg"

    if audio.startswith(b"fLaC"):
        return "audio/flac"

    if audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        return "audio/wav"

    if audio.startswith(b"\x1aE\xdf\xa3"):
        return "audio/webm"

    if len(audio) > 8 and audio[4:8] == b"ftyp":
        return "audio/mp4"

    return "audio/wav"


def normalize_column_name(name):
    if not isinstance(name, str):
        return name

    name = name.strip()
    name = name.replace("점수 1", "점수1")
    name = name.replace("점수 2", "점수2")
    name = name.replace("점수일", "점수1")
    name = name.replace("점수이", "점수2")

    return name


@app.get("/")
async def root():
    return {
        "ok": True,
        "email": EMAIL,
        "endpoint": "/answer-audio",
    }


@app.get("/debug")
def get_debug():
    return last_debug_info


@app.get("/transcripts")
def get_transcripts():
    return {
        "count": len(audio_history),
        "calls": list(reversed(audio_history)),
    }


@app.get("/last-audio")
def get_last_audio():
    extensions = {
        "audio/mp3": "mp3",
        "audio/mpeg": "mp3",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
        "audio/wav": "wav",
        "audio/webm": "webm",
        "audio/mp4": "m4a",
    }

    extension = extensions.get(last_audio_mime, "bin")

    return Response(
        content=last_audio_bytes,
        media_type=last_audio_mime,
        headers={
            "Content-Disposition": f'attachment; filename="q6_audio.{extension}"'
        },
    )


@app.post("/answer-audio")
async def answer_audio(request: Request):
    global last_debug_info
    global last_audio_bytes
    global last_audio_mime

    raw_request = await request.body()
    content_type = request.headers.get("content-type", "")

    last_debug_info = {
        "content_type": content_type,
        "raw_len": len(raw_request),
    }

    body = {}
    audio_id = None
    audio_base64 = ""

    try:
        if "application/json" in content_type or raw_request[:1] in (b"{", b"["):
            body = json.loads(raw_request)
            last_debug_info["body_keys"] = (
                list(body.keys()) if isinstance(body, dict) else "non-dict"
            )
            audio_id, audio_base64 = find_audio_base64(body)

        else:
            try:
                form = await request.form()
                last_debug_info["form_keys"] = list(form.keys())

                for _, value in form.items():
                    data = await value.read() if hasattr(value, "read") else None
                    if data:
                        last_audio_bytes = data

            except Exception:
                pass

            if not last_audio_bytes and raw_request:
                last_audio_bytes = raw_request

            audio_base64 = (
                base64.b64encode(last_audio_bytes).decode()
                if last_audio_bytes else ""
            )

    except Exception as error:
        last_debug_info["parse_error"] = str(error)

    last_debug_info["body_id"] = audio_id
    last_debug_info["audio_b64_len"] = len(audio_base64)

    transcript = ""

    try:
        audio = base64.b64decode(audio_base64) if audio_base64 else last_audio_bytes
        last_audio_bytes = audio
        last_debug_info["magic_bytes"] = audio[:16].hex()

        mime = detect_mime(audio)
        last_audio_mime = mime
        last_debug_info["detected_mime"] = mime

        transcription_payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Transcribe this audio precisely in Korean. "
                                "Output ONLY the Korean transcription."
                            )
                        },
                        {
                            "inlineData": {
                                "mimeType": mime,
                                "data": audio_base64,
                            }
                        },
                    ]
                }
            ]
        }

        transcript = await gemini_transcribe(
            transcription_payload,
            last_debug_info,
        )

    except Exception as error:
        transcript = ""
        last_debug_info["exception"] = str(error)

    last_debug_info["transcript"] = transcript

    prompt = f"""
The Korean transcript below describes a tabular dataset and/or requested
statistics. Extract the schema and exact statistics.

Return ONLY valid JSON in this format:

{{
  "columns": [],
  "data_rows": [],
  "num_rows": null,
  "explicit_stats": {{}},
  "requested_stats": []
}}

Rules:
- Preserve Korean column names exactly.
- For score columns use exactly "점수1" and "점수2", never "점수 1" or "점수 2".
- 평균 = mean
- 표준편차 = std
- 분산 = variance
- 최솟값 or 최소 = min
- 최댓값 or 최대 = max
- 중앙값 or 중간값 = median
- 최빈값 = mode
- 범위 = range
- 허용값 = allowed_values
- A에서 B 사이 = value_range
- 양의 상관관계 = positive
- 음의 상관관계 = negative
- Do not invent data.
- If actual rows are spoken, place them in data_rows.
- If only a row count is spoken, place it in num_rows.
- Put explicitly spoken values inside explicit_stats.
- requested_stats may contain only:
  mean, std, variance, min, max, median, mode, range,
  allowed_values, value_range, correlation.

TRANSCRIPT:
{transcript}
"""

    columns = []
    data_rows = []
    requested_stats = []
    num_rows = None
    explicit_stats = {}

    try:
        raw_llm = await chat(
            [{"role": "user", "content": prompt}],
            model="gpt-4o",
            max_tokens=1500,
        )

        last_debug_info["raw_llm"] = raw_llm

        extracted = parse_json(raw_llm)

        if isinstance(extracted, dict):
            columns = extracted.get("columns", []) or []
            data_rows = extracted.get("data_rows", []) or []
            requested_stats = extracted.get("requested_stats", []) or []
            num_rows = extracted.get("num_rows")
            explicit_stats = extracted.get("explicit_stats", {}) or {}

    except Exception as error:
        last_debug_info["extraction_error"] = str(error)

    columns = [normalize_column_name(column) for column in columns if isinstance(column, str)]

    referenced_columns = []

    for value in explicit_stats.values():
        if isinstance(value, dict):
            for column in value:
                column = normalize_column_name(column)
                if column not in referenced_columns:
                    referenced_columns.append(column)

    for column in referenced_columns:
        if column not in columns:
            columns.append(column)

    full_stat_list = [
        "mean",
        "std",
        "variance",
        "min",
        "max",
        "median",
        "mode",
        "range",
        "allowed_values",
        "value_range",
        "correlation",
    ]

    if not requested_stats:
        requested_stats = full_stat_list.copy()

    requested_stats = [
        stat for stat in dict.fromkeys(requested_stats)
        if stat in full_stat_list
    ]

    output = {
        "rows": num_rows if num_rows is not None else len(data_rows),
        "columns": columns,
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": [],
    }

    def column_values(index):
        values = []

        for row in data_rows:
            try:
                values.append(float(row[index]))
            except Exception:
                pass

        return values

    data_exists = len(data_rows) > 0

    for index, column in enumerate(columns):
        values = column_values(index)

        if not values:
            continue

        if "mean" in requested_stats:
            output["mean"][column] = mean(values)

        if "std" in requested_stats:
            output["std"][column] = pstdev(values) if len(values) > 1 else 0.0

        if "variance" in requested_stats:
            output["variance"][column] = pvariance(values) if len(values) > 1 else 0.0

        if "min" in requested_stats:
            output["min"][column] = min(values)

        if "max" in requested_stats:
            output["max"][column] = max(values)

        if "median" in requested_stats:
            output["median"][column] = median(values)

        if "mode" in requested_stats:
            try:
                output["mode"][column] = mode(values)
            except Exception:
                output["mode"][column] = values

        if "range" in requested_stats:
            output["range"][column] = max(values) - min(values)

        if "value_range" in requested_stats:
            output["value_range"][column] = [min(values), max(values)]

    normalized_explicit = {}

    for stat_name, stat_value in explicit_stats.items():
        if isinstance(stat_value, dict):
            normalized_explicit[stat_name] = {
                normalize_column_name(column): value
                for column, value in stat_value.items()
            }
        else:
            normalized_explicit[stat_name] = stat_value

    explicit_stats = normalized_explicit

    for stat_name, stat_value in explicit_stats.items():
        if (
            stat_name in output
            and isinstance(output[stat_name], dict)
            and isinstance(stat_value, dict)
        ):
            output[stat_name].update(stat_value)

    raw_correlation = explicit_stats.get("correlation")

    if isinstance(raw_correlation, list):
        fixed_correlation = []

        for item in raw_correlation:
            if isinstance(item, dict) and item.get("x") and item.get("y"):
                fixed_correlation.append({
                    "x": normalize_column_name(item["x"]),
                    "y": normalize_column_name(item["y"]),
                    "type": item.get("type", "positive"),
                })

        output["correlation"] = fixed_correlation

    def has_explicit_stat(stat_name):
        value = explicit_stats.get(stat_name)
        return (
            isinstance(value, dict) and bool(value)
        ) or (
            isinstance(value, list) and bool(value)
        )

    if set(requested_stats) != set(full_stat_list):
        target_stats = [
            stat_name
            for stat_name in full_stat_list
            if stat_name in requested_stats
        ]
    elif data_exists:
        target_stats = full_stat_list.copy()
    else:
        target_stats = [
            stat_name
            for stat_name in full_stat_list
            if has_explicit_stat(stat_name)
        ]

    for stat_name in full_stat_list:
        if stat_name == "correlation":
            continue
        if stat_name not in target_stats:
            output[stat_name] = {}

    if "correlation" not in target_stats:
        output["correlation"] = []

    audio_history.append({
        "audio_id": audio_id,
        "detected_mime": last_debug_info.get("detected_mime"),
        "transcript": transcript,
        "raw_llm": last_debug_info.get("raw_llm"),
        "requested_stats": requested_stats,
        "target_keys": target_stats,
        "answer": output,
    })

    if len(audio_history) > 50:
        del audio_history

    return output
