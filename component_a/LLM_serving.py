import time
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)

# ── config ────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME      = "phi3:mini"
REQUEST_TIMEOUT = 120.0
HEALTH_TIMEOUT  = 5.0
# ──────────────────────────────────────────────────────────

app = FastAPI(title="Miinla LLM Server", version="1.0.0")


class PromptRequest(BaseModel):
    prompt:      str
    model:       str   = MODEL_NAME
    temperature: float = 0.2
    max_tokens:  int   = 512


class LLMResponse(BaseModel):
    answer:     str
    model:      str
    latency_ms: float


@app.get("/health")
async def health():
    """Quick liveness check — confirms Ollama is up before the pipeline starts."""
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
        logger.info("health check passed — Ollama reachable")
        return {"status": "ok", "ollama": "reachable"}
    except Exception as e:
        logger.error("health check failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Ollama not reachable: {e}")


@app.post("/generate", response_model=LLMResponse)
async def generate(request: PromptRequest):
    """Forward prompt to Ollama and return the response with wall-clock latency."""
    payload = {
        "model":  request.model,
        "prompt": request.prompt,
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        },
    }

    logger.debug("sending prompt to %s (model=%s)", OLLAMA_BASE_URL, request.model)
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate", json=payload
            )
            resp.raise_for_status()
    except httpx.RequestError as e:
        logger.error("connection error reaching Ollama: %s", e)
        raise HTTPException(status_code=503, detail=f"Ollama connection error: {e}")
    except httpx.HTTPStatusError as e:
        logger.error("Ollama returned HTTP %s", e.response.status_code)
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Ollama error: {e.response.text}"
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = resp.json()

    logger.info(
        "generation complete — model=%s latency=%.0fms",
        data.get("model", request.model),
        elapsed_ms
    )

    return LLMResponse(
        answer=data.get("response", "").strip(),
        model=data.get("model", request.model),
        latency_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    uvicorn.run("LLM_serving:app", host="0.0.0.0", port=8000, reload=False)