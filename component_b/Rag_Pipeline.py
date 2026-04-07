import argparse
import time
from typing import List
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_TOKEN", "dummy") 

import chromadb
import httpx
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

logger = get_logger(__name__)

# ── config ────────────────────────────────────────────────
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "german_docs"
LLM_ENDPOINT    = "http://localhost:11434/api/generate"
LLM_MODEL       = "phi3:mini"
CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 50    # overlap to avoid cutting sentences mid-context
TOP_K           = 3     # how many chunks to retrieve per query
LLM_TIMEOUT     = 120.0
# ──────────────────────────────────────────────────────────


def load_document(path: str) -> str:
    """Read a .txt or .pdf file and return its full text content."""
    logger.info("loading document: %s", path)

    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    logger.info("loaded %d characters from %s", len(text), path)
    return text


def chunk_text(text: str) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.

    Overlap prevents losing context when a sentence spans a chunk boundary.
    Known limitation: doesn't respect sentence boundaries — German compound
    nouns can get split mid-word. A RecursiveCharacterTextSplitter would
    improve this in production.
    """
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    logger.info("created %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def embed_chunks(chunks: List[str], embedder: SentenceTransformer) -> List[List[float]]:
    """Embed document chunks. intfloat/e5 requires a 'passage:' prefix for indexing."""
    prefixed   = [f"passage: {c}" for c in chunks]
    embeddings = embedder.encode(
        prefixed, normalize_embeddings=True, show_progress_bar=True
    )
    logger.debug("embedded %d chunks", len(chunks))
    return embeddings.tolist()


def get_collection(reset: bool = False) -> chromadb.Collection:
    """Return the ChromaDB collection, optionally wiping it first."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("existing ChromaDB collection deleted")
        except Exception:
            pass

    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def store_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    collection: chromadb.Collection
) -> None:
    """Persist chunks and their embeddings into ChromaDB."""
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    logger.info("stored %d chunks in ChromaDB", len(chunks))


def retrieve(
    query: str,
    collection: chromadb.Collection,
    embedder: SentenceTransformer
) -> List[str]:
    """Embed the query and return the top-k most relevant document chunks."""
    # e5 models use 'query:' prefix for retrieval (vs 'passage:' for indexing)
    q_emb   = embedder.encode([f"query: {query}"], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=TOP_K)

    chunks = results["documents"][0]
    dists  = results["distances"][0]

    logger.info("retrieved top-%d chunks for query: %s", TOP_K, query)
    for i, (chunk, dist) in enumerate(zip(chunks, dists)):
        logger.debug("  [%d] dist=%.4f | %s...", i + 1, dist, chunk[:80])

    return chunks


def generate_answer(query: str, chunks: List[str]) -> dict:
    """Pass retrieved chunks as context to the LLM and return its answer."""
    context = "\n\n".join(f"[Kontext {i+1}]\n{c}" for i, c in enumerate(chunks))
    prompt  = (
        "Du bist ein hilfreicher Assistent. Beantworte die Frage ausschließlich\n"
        "auf Basis der Kontextinformationen. Wenn die Antwort nicht enthalten ist, sage das.\n\n"
        f"{context}\n\n"
        f"Frage: {query}\n"
        "Antwort:"
    )

    logger.debug("sending prompt to LLM (%s)", LLM_MODEL)
    start = time.perf_counter()

    try:
        with httpx.Client(timeout=LLM_TIMEOUT) as client:
            resp = client.post(LLM_ENDPOINT, json={
                "model":   LLM_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.1, "num_predict": 512},
            })
            resp.raise_for_status()
    except httpx.RequestError as e:
        logger.error("LLM connection failed: %s", e)
        return {"answer": f"[ERROR] {e}", "latency_ms": 0}
    except httpx.HTTPStatusError as e:
        logger.error("LLM returned HTTP %s: %s", e.response.status_code, e.response.text)
        return {"answer": f"[ERROR] HTTP {e.response.status_code}", "latency_ms": 0}

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("LLM response received — latency=%.0fms", latency_ms)

    return {
        "answer":     resp.json().get("response", "").strip(),
        "latency_ms": latency_ms,
    }


class RAGPipeline:
    """Orchestrates the full load → chunk → embed → store → retrieve → generate flow."""

    def __init__(self, document_path: str, reset_db: bool = False):
        self.embedder   = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = get_collection(reset=reset_db)

        if reset_db or self.collection.count() == 0:
            text   = load_document(document_path)
            chunks = chunk_text(text)
            embs   = embed_chunks(chunks, self.embedder)
            store_chunks(chunks, embs, self.collection)
        else:
            # skip re-indexing if the DB already has chunks from a previous run
            logger.info("reusing existing DB (%d chunks)", self.collection.count())

    def query(self, question: str) -> dict:
        """Retrieve relevant chunks and generate an answer for the given question."""
        chunks = retrieve(question, self.collection, self.embedder)
        result = generate_answer(question, chunks)
        return {"question": question, **result, "context_chunks": chunks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline standalone")
    parser.add_argument("--doc",      required=True,                    help="Path to document")
    parser.add_argument("--question", default="Was ist die DSGVO?",     help="Query to answer")
    parser.add_argument("--reset",    action="store_true",              help="Wipe and rebuild the DB")
    args = parser.parse_args()

    rag    = RAGPipeline(args.doc, reset_db=args.reset)
    result = rag.query(args.question)

    print(f"\nFrage  : {result['question']}")
    print(f"Antwort: {result['answer']}")
    print(f"Latenz : {result['latency_ms']} ms")