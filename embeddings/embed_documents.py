"""
embed_documents.py
Loads raw JSON documents from data/raw/, cleans them, generates
sentence embeddings using sentence-transformers, and stores them in ChromaDB.
"""

import json
import os
import re
import sys

from sentence_transformers import SentenceTransformer

# Add project root to path so we can import sibling packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from vector_db.chroma_manager import add_documents, get_client

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CLEANED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")
os.makedirs(CLEANED_DIR, exist_ok=True)

# Embedding model — free, open-source, ~90 MB
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Alternatives (better quality, larger):
# EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic text normalisation + deduplication of scraped title repetition."""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove HTML entities
    text = re.sub(r"&[a-z]+;", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove very short tokens of garbage
    text = re.sub(r"\b\w{1}\b", "", text)
    # Fix scraper duplication: "Title - Source. Title Source" pattern
    # Split on first period, check if second half starts with similar text
    if ". " in text:
        first, rest = text.split(". ", 1)
        # If rest begins with the same ~30 chars as first, it's a duplicate
        if len(first) > 20 and rest.lower().startswith(first[:30].lower()):
            text = first.strip()
    return text


def clean_document(doc: dict) -> dict | None:
    """Return a cleaned copy of a document, or None if it should be dropped."""
    content = clean_text(doc.get("content", ""))
    title = clean_text(doc.get("title", ""))

    # Drop if too short after cleaning
    if len(content) < 40:
        return None

    # Merge title into content for richer embedding signal
    combined = f"{title}. {content}" if title and title not in content else content

    return {**doc, "content": combined[:3000], "title": title}


# ── Load raw files ────────────────────────────────────────────────────────────

def load_raw_documents() -> list[dict]:
    """Load all JSON files from data/raw/ and return a flat list of docs."""
    all_docs = []
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]

    if not raw_files:
        print("[embed_documents] No raw JSON files found. Run the scrapers first.")
        return []

    for fname in raw_files:
        path = os.path.join(RAW_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_docs.extend(data)
            elif isinstance(data, dict):
                all_docs.append(data)
            print(f"  Loaded {len(data) if isinstance(data, list) else 1} docs from {fname}")
        except Exception as e:
            print(f"  Error loading {fname}: {e}")

    return all_docs


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(docs: list[dict]) -> list[dict]:
    seen_ids = set()
    seen_content = set()
    unique = []
    for d in docs:
        doc_id = d.get("id", "")
        content_key = d.get("content", "")[:100]  # fingerprint by first 100 chars
        if doc_id in seen_ids or content_key in seen_content:
            continue
        seen_ids.add(doc_id)
        seen_content.add(content_key)
        unique.append(d)
    return unique


# ── Embedding ─────────────────────────────────────────────────────────────────

def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> list[list[float]]:
    """Generate embeddings in batches; returns list of float vectors."""
    print(f"[embed_documents] Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize → cosine = dot product
    )
    return embeddings.tolist()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(reset: bool = False) -> int:
    """
    Full pipeline: load → clean → deduplicate → embed → store.
    Returns number of documents stored.
    """
    print("[embed_documents] Starting embedding pipeline...")

    # 1. Load
    raw_docs = load_raw_documents()
    if not raw_docs:
        return 0
    print(f"  Loaded {len(raw_docs)} raw documents")

    # 2. Clean
    cleaned = [clean_document(d) for d in raw_docs]
    cleaned = [d for d in cleaned if d is not None]
    print(f"  After cleaning: {len(cleaned)} documents")

    # 3. Deduplicate
    cleaned = deduplicate(cleaned)
    print(f"  After deduplication: {len(cleaned)} unique documents")

    # Save cleaned corpus
    cleaned_path = os.path.join(CLEANED_DIR, "cleaned_documents.json")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"  Saved cleaned docs → {cleaned_path}")

    # 4. Generate embeddings
    print(f"[embed_documents] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [d["content"] for d in cleaned]
    embeddings = generate_embeddings(texts, model)

    # 5. Store in ChromaDB
    if reset:
        from vector_db.chroma_manager import reset_collection
        reset_collection()

    client = get_client()
    added = add_documents(cleaned, embeddings, client=client)
    print(f"[embed_documents] Stored {added} documents in ChromaDB.")

    return added


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB before inserting")
    args = parser.parse_args()
    total = run(reset=args.reset)
    print(f"Done. Total documents in DB: {total}")
