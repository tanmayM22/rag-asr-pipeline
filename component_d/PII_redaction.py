import subprocess
import sys
from typing import Tuple
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spacy

from utils.logger import get_logger

logger = get_logger(__name__)

# ── config ────────────────────────────────────────────────
SPACY_MODEL   = "de_core_news_sm"
ENTITY_LABELS = {"PER", "ORG", "LOC"}

# maps spaCy's internal label names to human-readable placeholder tokens
LABEL_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}
# ──────────────────────────────────────────────────────────


def _load_spacy_model() -> spacy.language.Language:
    """Load the German NER model, downloading it automatically if missing."""
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        logger.warning("%s not found — downloading now", SPACY_MODEL)
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
            check=True
        )
        return spacy.load(SPACY_MODEL)


nlp = _load_spacy_model()


def redact_pii(text: str) -> Tuple[str, dict]:
    """
    Detect PER, ORG, and LOC entities and replace them with typed placeholders.

    Replacements are applied back-to-front to keep character offsets valid.
    Returns the redacted string and a map of {placeholder: original_text}
    so the caller can audit what was masked.

    Known gap: spaCy's de_core_news_sm misses some entities (e.g. IBAN,
    email addresses, Steuernummer). A custom EntityRuler would cover those.
    """
    doc      = nlp(text)
    counters = {label: 0 for label in ENTITY_LABELS}
    entity_map, replacements = {}, []

    for ent in doc.ents:
        if ent.label_ not in ENTITY_LABELS:
            continue

        counters[ent.label_] += 1
        label       = LABEL_MAP.get(ent.label_, ent.label_)
        placeholder = f"[{label}_{counters[ent.label_]}]"
        entity_map[placeholder] = ent.text
        replacements.append((ent.start_char, ent.end_char, placeholder))

    # apply from end → start so earlier offsets stay valid
    redacted = text
    for start, end, ph in sorted(replacements, key=lambda x: x[0], reverse=True):
        redacted = redacted[:start] + ph + redacted[end:]

    logger.debug("redacted %d entities from input text", len(entity_map))
    return redacted, entity_map


def redact_and_log(text: str) -> str:
    """Redact PII from text and return the sanitised version."""
    redacted, entity_map = redact_pii(text)

    logger.info("original : %s", text)
    logger.info("redacted : %s", redacted)

    if entity_map:
        for placeholder, original in entity_map.items():
            logger.info("  %s → %s", placeholder, original)
    else:
        logger.info("no PII detected")

    return redacted


if __name__ == "__main__":
    examples = [
        "Hallo, ich bin Maria Schneider von der Mustermann GmbH in Frankfurt.",
        "Dr. Klaus Müller arbeitet bei Siemens AG in München.",
        "Die Datenschutzbeauftragte Frau Petra Weber ist in Berlin tätig.",
        "Ein anonymer Nutzer stellte eine Anfrage ohne persönliche Daten.",
    ]
    for example in examples:
        redacted, entity_map = redact_pii(example)
        print(f"\nOriginal : {example}")
        print(f"Redacted : {redacted}")
        if entity_map:
            for ph, orig in entity_map.items():
                print(f"  {ph} → {orig}")
        else:
            print("  No PII detected.")