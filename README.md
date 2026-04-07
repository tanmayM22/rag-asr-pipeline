# AI Pipeline — Thesis Applicant Task

End-to-end prototype: Voice ASR → PII Redaction → RAG → Local LLM

## Hardware

| Spec | Details |
|------|---------|
| **Machine** | MacBook Air M1 (2020) |
| **Storage** | 256 GB SSD |
| **RAM** | 8 GB Unified Memory |
| **GPU** | Apple M1 8-core GPU (Metal / MPS) |
| **OS** | macOS (Apple Silicon — arm64) |

> **Note on M1 Unified Memory:** Apple M1 uses a unified memory architecture — the CPU and GPU share the same 8 GB pool. Ollama automatically leverages Apple Metal (MPS) for acceleration, delivering ~25–45 tokens/sec for phi3:mini without any extra configuration. This is significantly faster than CPU-only x86 machines with the same RAM.

---

## Model Choice

- **LLM**: `phi3:mini` via Ollama — 2.7B parameters, ~3.1 GB RAM during inference, ~25–45 tok/s on M1 with Metal acceleration
  - Alternative: `mistral:7b-instruct` is also viable on M1 8GB but may cause memory pressure alongside other apps
- **Embeddings**: `intfloat/multilingual-e5-small` — fast, good German coverage, runs efficiently on CPU
- **ASR**: Whisper `base` — automatically uses Apple MPS (Metal Performance Shaders) on M1 for ~10–20x real-time speed
- **NER**: `de_core_news_sm` — lightweight German spaCy model, runs on CPU

---

## M1-Specific Performance Tips

```bash
# Add to ~/.zshrc for better Ollama performance on M1 Air
export OLLAMA_NUM_PARALLEL=1
# Single-threaded execution prevents Metal kernel contention on M1
# and avoids memory bandwidth bottlenecks — consistently faster for single-user use

# Verify Ollama is using Metal (should show GPU layers in output)
ollama run phi3:mini "test" --verbose
```

> **Thermal note:** The M1 Air has passive cooling. After ~3–5 minutes of sustained LLM inference, the chip may throttle. For evaluation runs, allow 30–60 seconds between queries for best results.

---

## Quick Start

```bash
# 1. Install Ollama for Apple Silicon (arm64)
#    Download from: https://ollama.com/download/mac
#    Then pull the model:
ollama pull phi3:mini
ollama serve          # Keep this running in Terminal 1

# 2. Setup Python environment (use Python 3.11 — best M1 compatibility)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm

# 3. Download German DSGVO document
python -c "
import urllib.request
urllib.request.urlretrieve(
  'https://de.wikipedia.org/wiki/Datenschutz-Grundverordnung?action=raw',
  'dsgvo.txt')
print('Downloaded dsgvo.txt')
"

# 4. Run the full pipeline (text mode — no audio needed)
python pipeline.py --text "Was sind die Grundsätze der DSGVO?" --doc dsgvo.txt --reset-db

# 5. Run with voice input (generates a sample German audio clip first)
cd component_c
python asr.py --generate-sample --model base
cd ..
python pipeline.py --audio component_c/sample_german.wav --doc dsgvo.txt
```

---

## What Works

- FastAPI LLM endpoint with async httpx and latency measurement
- RAG pipeline: ingest → chunk → embed → ChromaDB → retrieve → generate
- Whisper ASR with forced German language decoding (MPS-accelerated on M1)
- spaCy PII redaction for PER/ORG/LOC with placeholder tokens
- Full end-to-end `pipeline.py` integration

---

## Known Limitations

- Fixed-size character chunking splits German compound words at boundaries
- `phi3:mini` produces shorter, less precise answers than larger models (trade-off for 8 GB RAM)
- Whisper `base` occasionally mishears German umlauts (ü, ö, ä) in noisy audio
- `de_core_news_sm` misses uncommon person names, abbreviations, and foreign-origin names
- M1 Air thermal throttling may slow LLM inference after sustained use (>3–5 min)

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LLM model** | `phi3:mini` | Fits in 8 GB unified memory; Metal-accelerated on M1; fast enough for evaluation |
| **Chunking** | Fixed 500-char, 50-char overlap | Simple, reproducible; production would use sentence-aware splitting |
| **Embedding model** | `multilingual-e5-small` | Fast, no GPU needed, strong German language recall |
| **Vector DB** | ChromaDB | Persistent on disk, easy to inspect; no need for FAISS setup complexity |
| **Temperature** | 0.1 | Low temperature for factual, retrieval-grounded generation |
| **Whisper model** | `base` | Best balance of speed and accuracy for 8 GB M1; `tiny` is too error-prone for German |

---

## Repository Structure

```
rag-asr-pipeline/
├── README.md
├── requirements.txt
├── pipeline.py                  ← End-to-end integration script
├── component_a/
│   └── LLM_serving.py           ← FastAPI + Ollama server
├── component_b/
│   └── Rag_Pipeline.py          ← RAG pipeline
├── component_c/
│   └── ASR.py                   ← Whisper ASR
├── component_d/
│   └── PII_redaction.py         ← spaCy PII redaction
├── data/
│   ├── dsgvo_sample.txt         ← Sample German document
│   ├── query.mp3                ← Test audio query
│   └── sample_german.wav        ← Sample German voice recording
├── Evalution/
│   └── QA_Pairs.json            ← 5 test Q&A pairs with results
├── logs/
│   └── pipeline.log             ← Application execution logs
├── Slides/
│   └── Miinla_Task_Presentation.pdf
└── utils/
    └── logger.py                ← Logging configuration
```
