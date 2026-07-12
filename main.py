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
