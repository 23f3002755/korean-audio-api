import os
import json
import base64

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Korean Audio Dataset API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")


def detect_mime(audio: bytes) -> str:
    if audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        return "audio/wav"

    if audio.startswith(b"ID3") or audio[:2] in (
        b"\xff\xfb",
        b"\xff\xf3",
        b"\xff\xf2"
    ):
        return "audio/mpeg"

    if audio.startswith(b"OggS"):
        return "audio/ogg"

    if audio.startswith(b"fLaC"):
        return "audio/flac"

    if audio.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio/webm"

    if len(audio) >= 8 and audio[4:8] == b"ftyp":
        return "audio/mp4"

    return "audio/wav"


def normalize_column_name(name):
    if not isinstance(name, str):
        return name

    name = name.strip()

    # Exact names required by this activity.
    name = name.replace("점수 1", "점수1")
    name = name.replace("점수 2", "점수2")
    name = name.replace("점수일", "점수1")
    name = name.replace("점수이", "점수2")

    return name


def normalize_answer(answer):
    if not isinstance(answer, dict):
        return {}

    # Fix column names in the columns array.
    if isinstance(answer.get("columns"), list):
        answer["columns"] = [
            normalize_column_name(column)
            for column in answer["columns"]
        ]

    # Fix dictionary keys for all per-column statistics.
    dictionary_stats = [
        "mean",
        "std",
        "variance",
        "min",
        "max",
        "median",
        "mode",
        "range",
        "allowed_values",
        "value_range"
    ]

    for stat in dictionary_stats:
        old_value = answer.get(stat)

        if isinstance(old_value, dict):
            answer[stat] = {
                normalize_column_name(column): value
                for column, value in old_value.items()
            }

    # Fix correlation column names.
    if isinstance(answer.get("correlation"), list):
        fixed_correlation = []

        for item in answer["correlation"]:
            if isinstance(item, dict):
                item = item.copy()

                if "x" in item:
                    item["x"] = normalize_column_name(item["x"])

                if "y" in item:
                    item["y"] = normalize_column_name(item["y"])

            fixed_correlation.append(item)

        answer["correlation"] = fixed_correlation

    return answer


@app.get("/")
async def home():
    return {
        "status": "Korean audio API is running"
    }


@app.post("/answer-audio")
async def answer_audio(request: Request):
    if not AIPIPE_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="AIPIPE_TOKEN is not configured in Render."
        )

    try:
        body = await request.json()
        audio_base64 = body["audio_base64"]
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid request. Expected JSON with audio_base64."
        )

    mime_type = detect_mime(audio_bytes)

    prompt = """
Listen carefully to this Korean audio. It describes a tabular dataset and/or
requested dataset statistics.

Return ONLY valid JSON. Do not return Markdown, explanations, comments, or code
fences.

The JSON MUST always have exactly these top-level keys:

{
  "rows": 0,
  "columns": [],
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
  "correlation": []
}

Rules:
- Preserve Korean column names exactly as spoken.
- For Korean score columns, always use "점수1" and "점수2".
- Never use "점수 1", "점수 2", "점수일", or "점수이".
- Populate ONLY statistics explicitly stated or requested in the audio.
- For any statistic not requested, return {}.
- For correlation not requested, return [].
- Do not calculate, infer, or invent data values that are not given in the audio.
- Do not add additional top-level JSON keys.

Korean statistics:
- 평균 = mean
- 표준편차 = std
- 분산 = variance
- 최소 or 최솟값 = min
- 최대 or 최댓값 = max
- 중앙값 or 중간값 = median
- 최빈값 = mode
- 범위 = range
- 허용값 or 허용된 값 = allowed_values
- A에서 B 사이 = value_range using [A, B]
- 양의 상관관계 = [{"x":"column1","y":"column2","type":"positive"}]
- 음의 상관관계 = [{"x":"column1","y":"column2","type":"negative"}]

Make sure the response is valid JSON.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": audio_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json"
        }
    }

    gemini_url = (
        "https://aipipe.org/geminiv1beta/models/"
        "gemini-2.5-flash-lite:generateContent"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                gemini_url,
                headers={
                    "Authorization": f"Bearer {AIPIPE_TOKEN}"
                },
                json=payload
            )

            response.raise_for_status()

        output_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        answer = json.loads(output_text)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(error)}"
        )

    answer = normalize_answer(answer)

    required_keys = [
        "rows",
        "columns",
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
        "correlation"
    ]

    default_values = {
        "rows": 0,
        "columns": [],
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
        "correlation": []
    }

    return {
        key: answer.get(key, default_values[key])
        for key in required_keys
    }
