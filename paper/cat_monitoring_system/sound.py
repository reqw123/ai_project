from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import requests
import uvicorn

app = FastAPI()

API_KEY = "cat_ai_123"

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen2.5:3b"

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(
    req: ChatRequest,
    x_api_key: str = Header(None)
):

    if x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": req.message,
            "stream": False
        }
    )

    result = response.json()

    return {
        "user": req.message,
        "ai": result["response"]
    }

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )