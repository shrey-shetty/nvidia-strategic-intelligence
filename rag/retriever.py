"""
retriever.py
Semantic and hybrid retrieval over ChromaDB.
All intelligence modules call retrieve() to get relevant documents.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from sentence_transformers import SentenceTransformer
from vector_db.chroma_manager import query, keyword_filter_query, get_client

# Use the same model as embed_documents.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model: SentenceTransformer = None   # lazy-loaded singleton


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_query(text: str) -> list[float]:
    """Convert a query string to an embedding vector."""
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


NOISE_PATTERNS = [
    "my garage", "running in my", "making me about", "&#x27;", "&#x2f;",
    "per hour", "$/hr", "i am a", "in all seriousness", "buy nvda puts",
    "show hn:", "ask hn:", "tell hn:",
]

def _is_noise(doc: dict) -> bool:
    """Return True if document is low-quality HN comment noise."""
    text = doc.get("content", "").lower()
    return any(p in text for p in NOISE_PATTERNS)

def retrieve(
    query_text: str,
    n_results: int = 8,
    use_hybrid: bool = True,
    source_filter: str = None,
) -> list[dict]:
    """
    Retrieve the top-n most relevant documents for a query.

    Args:
        query_text:    Natural language query string.
        n_results:     Number of results to return.
        use_hybrid:    If True, blend semantic similarity with keyword overlap.
        source_filter: Optional ChromaDB 'source' metadata filter.

    Returns:
        List of dicts with keys: id, content, metadata, similarity.
    """
    embedding = embed_query(query_text)
    client = get_client()

    where = {"source": source_filter} if source_filter else None

    if use_hybrid:
        keywords = [w for w in query_text.lower().split() if len(w) > 3]
        results = keyword_filter_query(
            embedding, keywords, n_results=n_results + 5, client=client
        )
    else:
        results = query(embedding, n_results=n_results + 5, where=where, client=client)

    # Filter noise and return top n
    results = [r for r in results if not _is_noise(r)]
    return results[:n_results]


def retrieve_multi(
    queries: list[str],
    n_per_query: int = 5,
    deduplicate: bool = True,
) -> list[dict]:
    """
    Run multiple queries and merge results, highest similarity first.
    Useful for building a broad evidence pool across several topics.
    """
    seen_ids: set[str] = set()
    all_docs: list[dict] = []

    for q in queries:
        docs = retrieve(q, n_results=n_per_query, use_hybrid=True)
        for doc in docs:
            if deduplicate and doc["id"] in seen_ids:
                continue
            seen_ids.add(doc["id"])
            all_docs.append(doc)

    all_docs.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return all_docs


def retrieve_by_source_type(
    query_text: str,
    source_type: str,
    n_results: int = 5,
) -> list[dict]:
    """Retrieve documents filtered to a specific source_type (e.g. 'community')."""
    embedding = embed_query(query_text)
    client = get_client()
    where = {"source_type": source_type}
    return query(embedding, n_results=n_results, where=where, client=client)


if __name__ == "__main__":
    results = retrieve("NVIDIA AI data center growth opportunity", n_results=5)
    for r in results:
        print(f"[{r['similarity']:.3f}] {r['metadata'].get('title','')[:80]}")
        print(f"  {r['content'][:150]}\n")
