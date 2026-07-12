from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Korean Audio Dataset API")


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


@app.get("/")
def home():
    return {
        "status": "Korean audio API is running"
    }


@app.post("/answer-audio")
async def answer_audio(request: AudioRequest):
    return {
        "rows": 0,
        "columns": ["점수1", "점수2"],

        "mean": {
            "점수1": 0,
            "점수2": 0
        },

        "std": {
            "점수1": 0,
            "점수2": 0
        },

        "variance": {
            "점수1": 0,
            "점수2": 0
        },

        "min": {
            "점수1": 0,
            "점수2": 0
        },

        "max": {
            "점수1": 0,
            "점수2": 0
        },

        "median": {
            "점수1": 0,
            "점수2": 0
        },

        "mode": {
            "점수1": [0],
            "점수2": [0]
        },

        "range": {
            "점수1": 0,
            "점수2": 0
        },

        "allowed_values": {},

        "value_range": {
            "점수1": [0, 0],
            "점수2": [0, 0]
        },

        "correlation": []
    }
