import os
import re
import json
import base64
import tempfile
import statistics
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI(title="Korean Audio Dataset API")

# Downloads the multilingual Whisper model the first time it runs.
# "base" is more accurate than "tiny" for Korean.
model = WhisperModel("base", device="cpu", compute_type="int8")


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def normal_number(value: Any):
    """Convert NumPy-like / numeric values to normal JSON numbers."""
    if isinstance(value, float):
        return round(value, 10)
    return value


def detect_extension(audio: bytes) -> str:
    if audio[:4] == b"RIFF":
        return ".wav"
    if audio[:3] == b"ID3" or audio[:2] == b"\xff\xfb":
        return ".mp3"
    if audio[:4] == b"OggS":
        return ".ogg"
    if audio[:4] == b"fLaC":
        return ".flac"
    if audio[:4] == b"\x1aE\xdf\xa3":
        return ".webm"
    return ".wav"


def transcribe_audio(audio_bytes: bytes) -> str:
    extension = detect_extension(audio_bytes)

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
        f.write(audio_bytes)
        file_path = f.name

    try:
        segments, _ = model.transcribe(
            file_path,
            language="ko",
            beam_size=5,
            vad_filter=True
        )
        return " ".join(segment.text for segment in segments).strip()
    finally:
        os.unlink(file_path)


def korean_to_number(token: str):
    """
    Supports ordinary digits and basic Korean spoken number words.
    Extend this if your debug logs show different formats.
    """
    token = token.strip().replace(",", "")

    try:
        return float(token) if "." in token else int(token)
    except ValueError:
        pass

    korean_digits = {
        "영": 0, "공": 0,
        "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
        "육": 6, "칠": 7, "팔": 8, "구": 9
    }

    if token in korean_digits:
        return korean_digits[token]

    # Basic Korean number parsing: 십이=12, 이십삼=23, 백=100
    total = 0
    current = 0
    units = {"십": 10, "백": 100, "천": 1000}

    for char in token:
        if char in korean_digits:
            current = korean_digits[char]
        elif char in units:
            if current == 0:
                current = 1
            total += current * units[char]
            current = 0
        else:
            return None

    total += current
    return total if total != 0 else None


def extract_scores(transcript: str):
    """
    Extract numerical pairs after Korean score labels.
    The expected output columns are exactly 점수1 and 점수2.
    """

    cleaned = transcript.replace("점수 1", "점수1").replace("점수 2", "점수2")
    cleaned = cleaned.replace("점수일", "점수1").replace("점수이", "점수2")

    # First attempt: use standard numbers in the transcript.
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    values = [float(x) if "." in x else int(x) for x in numbers]

    # Common case: transcript lists score1, score2 repeatedly.
    if len(values) >= 2 and len(values) % 2 == 0:
        score1 = values[0::2]
        score2 = values[1::2]
        return score1, score2

    # Second attempt: Korean number tokens.
    korean_tokens = re.findall(r"[영공일이삼사오육칠팔구십백천]+", cleaned)
    korean_values = []

    for token in korean_tokens:
        value = korean_to_number(token)
        if value is not None:
            korean_values.append(value)

    if len(korean_values) >= 2 and len(korean_values) % 2 == 0:
        return korean_values[0::2], korean_values[1::2]

    raise ValueError(
        f"Could not find paired score values. Transcript was: {transcript}"
    )


def mode_values(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    highest = max(counts.values())
    return sorted(
        [normal_number(x) for x, count in counts.items() if count == highest]
    )


def column_stats(values):
    minimum = min(values)
    maximum = max(values)

    return {
        "mean": normal_number(statistics.mean(values)),
        "std": normal_number(statistics.pstdev(values)),
        "variance": normal_number(statistics.pvariance(values)),
        "min": normal_number(minimum),
        "max": normal_number(maximum),
        "median": normal_number(statistics.median(values)),
        "mode": mode_values(values),
        "range": normal_number(maximum - minimum),
        "allowed_values": sorted(set(normal_number(x) for x in values)),
        "value_range": [normal_number(minimum), normal_number(maximum)]
    }


def pearson_correlation(x, y):
    if len(x) < 2:
        return 0.0

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator_x = sum((a - mean_x) ** 2 for a in x) ** 0.5
    denominator_y = sum((b - mean_y) ** 2 for b in y) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    return normal_number(numerator / (denominator_x * denominator_y))


@app.get("/")
def home():
    return {"status": "Korean audio API is running"}


@app.post("/answer-audio")
async def answer_audio(request: AudioRequest):
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        transcript = transcribe_audio(audio_bytes)
        score1, score2 = extract_scores(transcript)

        if len(score1) != len(score2):
            raise ValueError("점수1 and 점수2 have unequal row counts.")

        s1 = column_stats(score1)
        s2 = column_stats(score2)

        correlation_value = pearson_correlation(score1, score2)

        if correlation_value > 0:
            correlation_type = "positive"
        elif correlation_value < 0:
            correlation_type = "negative"
        else:
            correlation_type = "none"

        # Do NOT change these column names.
        return {
            "rows": len(score1),
            "columns": ["점수1", "점수2"],
            "mean": {
                "점수1": s1["mean"],
                "점수2": s2["mean"]
            },
            "std": {
                "점수1": s1["std"],
                "점수2": s2["std"]
            },
            "variance": {
                "점수1": s1["variance"],
                "점수2": s2["variance"]
            },
            "min": {
                "점수1": s1["min"],
                "점수2": s2["min"]
            },
            "max": {
                "점수1": s1["max"],
                "점수2": s2["max"]
            },
            "median": {
                "점수1": s1["median"],
                "점수2": s2["median"]
            },
            "mode": {
                "점수1": s1["mode"],
                "점수2": s2["mode"]
            },
            "range": {
                "점수1": s1["range"],
                "점수2": s2["range"]
            },
            "allowed_values": {
                "점수1": s1["allowed_values"],
                "점수2": s2["allowed_values"]
            },
            "value_range": {
                "점수1": s1["value_range"],
                "점수2": s2["value_range"]
            },
            "correlation": [
                {
                    "x": "점수1",
                    "y": "점수2",
                    "type": correlation_type
                }
            ]
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail={"error": str(error)}
        )
