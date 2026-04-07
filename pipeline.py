"""
pipeline.py — Miinla AI Pipeline end-to-end runner

Connects all four components in sequence:
  Audio/Text → Whisper ASR → PII Redaction → RAG Retrieval → LLM Answer

Usage:
    python pipeline.py --audio path/to/audio.wav --doc path/to/document.txt
    python pipeline.py --text "Was ist Datenschutz?"  --doc path/to/document.txt
"""

import argparse
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(__file__))

from component_b.Rag_Pipeline  import RAGPipeline
from component_c.ASR           import transcribe
from component_d.PII_redaction import redact_and_log
from utils.logger              import get_logger

logger = get_logger(__name__)

# ── config ────────────────────────────────────────────────
LLM_SERVER_HEALTH = "http://localhost:8000/health"
LLM_SERVER_CMD    = "python component_a/LLM_serving.py"
HEALTH_TIMEOUT    = 2
# ──────────────────────────────────────────────────────────


def _check_llm_server() -> None:
    """Fail fast if the LLM server isn't up — saves 20s of silent waiting."""
    try:
        requests.get(LLM_SERVER_HEALTH, timeout=HEALTH_TIMEOUT).raise_for_status()
        logger.info("LLM server reachable at %s", LLM_SERVER_HEALTH)
    except Exception:
        print(f"\n[ERROR] LLM server not running. Start it first:\n  {LLM_SERVER_CMD}\n")
        logger.error("LLM server unreachable — aborting pipeline")
        raise SystemExit(1)


def run_pipeline(
    audio_path:    str | None,
    text_query:    str | None,
    document_path: str,
    whisper_model: str  = "base",
    reset_db:      bool = False,
) -> dict:
    """Run the full pipeline and return the result dict."""

    _check_llm_server()

    print("\n" + "=" * 60)
    print("  Pipeline: Audio/Text → ASR → PII Redaction → RAG → LLM")
    print("=" * 60)

    # ── Step 1: ASR ───────────────────────────────────────────
    if audio_path:
        print("\n[STEP 1] Voice Transcription (Whisper ASR)")
        logger.info("step 1 — ASR | audio=%s model=%s", audio_path, whisper_model)
        asr_result = transcribe(audio_path, model_size=whisper_model)
        raw_query  = asr_result["transcript"]
    elif text_query:
        print("\n[STEP 1] Skipping ASR — using provided text query")
        logger.info("step 1 — ASR skipped | text query provided")
        raw_query = text_query
    else:
        raise ValueError("Provide --audio or --text")

    print(f"\n  Raw Query: {raw_query}")

    # ── Step 2: PII Redaction ─────────────────────────────────
    print("\n[STEP 2] PII Redaction\n")
    logger.info("step 2 — PII redaction")
    safe_query = redact_and_log(raw_query)

    # print redaction result to terminal for transparency
    print(f"  Original : {raw_query}")
    print(f"  Redacted : {safe_query}")

    # ── Step 3: RAG + LLM ────────────────────────────────────
    print("\n[STEP 3] RAG Retrieval + LLM Generation")
    logger.info("step 3 — RAG query | doc=%s", document_path)
    rag    = RAGPipeline(document_path, reset_db=reset_db)
    result = rag.query(safe_query)

    logger.info(
        "pipeline complete | latency=%.0fms answer_len=%d",
        result["latency_ms"],
        len(result["answer"])
    )

    # ── Final Output ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL ANSWER")
    print("=" * 60)
    print(f"Original Query : {raw_query}")
    print(f"Redacted Query : {safe_query}")
    print(f"Answer         : {result['answer']}")
    print(f"LLM Latency    : {result['latency_ms']} ms")
    print("=" * 60 + "\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Miinla AI Pipeline — end-to-end runner"
    )
    parser.add_argument("--audio",         type=str,
                        help="Path to audio file (.wav / .mp3)")
    parser.add_argument("--text",          type=str,
                        help="Direct text query — skips Whisper ASR")
    parser.add_argument("--doc",           type=str, required=True,
                        help="Path to German document (.pdf / .txt)")
    parser.add_argument("--whisper-model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--reset-db",      action="store_true",
                        help="Wipe and rebuild the ChromaDB vector store")
    args = parser.parse_args()

    run_pipeline(
        audio_path    = args.audio,
        text_query    = args.text,
        document_path = args.doc,
        whisper_model = args.whisper_model,
        reset_db      = args.reset_db,
    )