import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import whisper

from utils.logger import get_logger

logger = get_logger(__name__)

# ── config ────────────────────────────────────────────────
DEFAULT_MODEL = "base"       # balanced speed/accuracy on CPU
WHISPER_LANG  = "de"         # force German — avoids misdetection on short clips
SAMPLE_TEXT   = (
    "Was sind die wichtigsten Grundsätze der Datenschutz-Grundverordnung? "
    "Die DSGVO schützt personenbezogene Daten in der Europäischen Union."
)
SAMPLE_OUTPUT = os.path.join("data", "sample_german.wav")
# ──────────────────────────────────────────────────────────


def transcribe(audio_path: str, model_size: str = DEFAULT_MODEL) -> dict:
    """
    Transcribe a German audio file using Whisper (local, CPU-safe).

    model_size choices:
      tiny  — fastest, lower accuracy, good for quick tests
      base  — balanced; default for this project
      small — better accuracy, noticeably slower on CPU
    fp16 disabled because most dev machines run this on CPU only.
    """
    logger.info("loading Whisper model: %s", model_size)
    model = whisper.load_model(model_size)

    logger.info("transcribing: %s", audio_path)
    start = time.perf_counter()

    result = model.transcribe(
        audio_path,
        language=WHISPER_LANG,
        task="transcribe",
        fp16=False,       # CPU-only machines don't support fp16
        verbose=False,
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    transcript = result["text"].strip()

    logger.info("transcript  : %s", transcript)
    logger.info("language    : %s", result.get("language", "unknown"))
    logger.info("latency     : %.0f ms", elapsed_ms)

    return {
        "transcript": transcript,
        "language":   result.get("language", "unknown"),
        "model":      model_size,
        "latency_ms": elapsed_ms,
    }


def generate_sample_audio(output_path: str = SAMPLE_OUTPUT) -> str:
    """
    Create a short German .wav using gTTS for testing when no real audio is available.
    Not used in production — convenience helper for local dev only.
    """
    from gtts import gTTS

    # make sure data/ exists before writing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("generating sample German audio → %s", output_path)
    gTTS(text=SAMPLE_TEXT, lang="de").save(output_path)
    logger.info("sample audio saved: %s", output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe German audio with Whisper")
    parser.add_argument("--audio",           type=str,
                        help="Path to audio file (.wav / .mp3)")
    parser.add_argument("--model",           type=str, default=DEFAULT_MODEL,
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a sample German audio clip and transcribe it")
    args = parser.parse_args()

    if args.generate_sample:
        audio  = generate_sample_audio()
        result = transcribe(audio, args.model)
    elif args.audio:
        result = transcribe(args.audio, args.model)
    else:
        print("Provide --audio <file> or use --generate-sample to create a test clip.")
        raise SystemExit(1)

    # only final result goes to terminal — everything else is in logs/pipeline.log
    print(f"\nTranscript : {result['transcript']}")
    print(f"Language   : {result['language']}")
    print(f"Latency    : {result['latency_ms']} ms")