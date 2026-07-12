import os
import json
import base64

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
    if audio.startswith(b"ID3") or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
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


@app.get("/")
async def home():
    return {"status": "running"}


@app.post("/answer-audio")
async def answer_audio(request: Request):
    if not AIPIPE_TOKEN:
        raise HTTPException(status_code=500, detail="AIPIPE_TOKEN is not configured")

    try:
        body = await request.json()
        audio_b64 = body["audio_base64"]
        audio = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON or audio_base64")

    mime_type = detect_mime(audio)

    prompt = """
Listen carefully to this Korean audio. It describes a dataset and/or requested
dataset statistics.

Return ONLY valid JSON. Do not include markdown, explanations, or code fences.

Your JSON MUST always have exactly these top-level keys:
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
- Populate ONLY statistics explicitly stated or requested in the audio.
- Leave every non-requested statistic as {}.
- Leave correlation as [] unless correlation is explicitly mentioned.
- "평균" = mean.
- "표준편차" = std.
- "분산" = variance.
- "최솟값" = min.
- "최댓값" = max.
- "중앙값" or "중간값" = median.
- "최빈값" = mode.
- "범위" = range.
- "허용값" = allowed_values.
- "A에서 B 사이" = value_range with [A, B].
- "양의 상관관계" = [{"x":"...","y":"...","type":"positive"}].
- "음의 상관관계" = [{"x":"...","y":"...","type":"negative"}].
- Do not calculate or invent values not present in the audio.
- Ensure returned JSON parses correctly.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": audio_b64
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

    url = (
        "https://aipipe.org/geminiv1beta/models/"
        "gemini-2.5-flash-lite:generateContent"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
                json=payload
            )
            response.raise_for_status()

        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        answer = json.loads(text)

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {error}")

    required = [
        "rows", "columns", "mean", "std", "variance", "min", "max",
        "median", "mode", "range", "allowed_values", "value_range",
        "correlation"
    ]

    default = {
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

    return {key: answer.get(key, default[key]) for key in required}
