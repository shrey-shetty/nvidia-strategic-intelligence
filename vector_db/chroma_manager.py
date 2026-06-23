"""
chroma_manager.py
Manages the ChromaDB vector store: initialization, insertion, deletion,
and semantic search. All other modules interact with ChromaDB through this.
"""

import os
import chromadb

# Persist the DB next to the vector_db folder
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

COLLECTION_NAME = "nvidia_intelligence"


def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=DB_DIR)


def get_collection(client: chromadb.PersistentClient = None):
    """Return (creating if needed) the main collection."""
    if client is None:
        client = get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_documents(
    documents: list[dict],
    embeddings: list[list[float]],
    client: chromadb.PersistentClient = None,
) -> int:
    """
    Add documents with pre-computed embeddings to ChromaDB.
    Each document dict must have at least: id, content.
    Returns the number of documents successfully added.
    """
    collection = get_collection(client)

    ids = []
    texts = []
    metas = []
    embeds = []

    for doc, emb in zip(documents, embeddings):
        ids.append(doc["id"])
        texts.append(doc["content"][:3000])  # ChromaDB has per-doc size limits
        metas.append({
            "title": doc.get("title", "")[:200],
            "url": doc.get("url", "")[:500],
            "source": doc.get("source", ""),
            "source_type": doc.get("source_type", ""),
            "published": str(doc.get("published", "")),
            "company": doc.get("company", "NVIDIA"),
        })
        embeds.append(emb)

    if not ids:
        return 0

    # Upsert in batches of 100 to avoid memory issues
    batch_size = 100
    added = 0
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        batch_metas = metas[i : i + batch_size]
        batch_embeds = embeds[i : i + batch_size]
        try:
            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metas,
                embeddings=batch_embeds,
            )
            added += len(batch_ids)
        except Exception as e:
            print(f"  [chroma_manager] Batch {i}-{i+batch_size} error: {e}")

    return added


def query(
    query_embedding: list[float],
    n_results: int = 10,
    where: dict = None,
    client: chromadb.PersistentClient = None,
) -> list[dict]:
    """
    Semantic search: returns top-n documents closest to query_embedding.
    Optionally filter by metadata using ChromaDB 'where' syntax.
    Returns list of dicts with keys: id, content, metadata, distance.
    """
    collection = get_collection(client)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception as e:
        print(f"  [chroma_manager] Query error: {e}")
        return []

    output = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
        output.append({
            "id": doc_id,
            "content": doc_text,
            "metadata": meta,
            "distance": dist,
            "similarity": round(1 - dist, 4),  # cosine → similarity
        })

    return output


def keyword_filter_query(
    query_embedding: list[float],
    keywords: list[str],
    n_results: int = 10,
    client: chromadb.PersistentClient = None,
) -> list[dict]:
    """
    Retrieve top-n semantically relevant docs, then re-rank by keyword presence.
    This gives a lightweight hybrid search without a dedicated BM25 index.
    """
    candidates = query(query_embedding, n_results=n_results * 3, client=client)
    scored = []
    for doc in candidates:
        text = doc["content"].lower()
        kw_score = sum(1 for kw in keywords if kw.lower() in text)
        doc["keyword_hits"] = kw_score
        doc["hybrid_score"] = doc["similarity"] + 0.1 * kw_score
        scored.append(doc)

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return scored[:n_results]


def get_stats(client: chromadb.PersistentClient = None) -> dict:
    """Return basic statistics about the collection."""
    collection = get_collection(client)
    count = collection.count()

    # Sample metadata to find unique sources
    sources = set()
    if count > 0:
        try:
            sample = collection.get(limit=min(count, 500), include=["metadatas"])
            for meta in sample.get("metadatas", []):
                sources.add(meta.get("source", "unknown"))
        except Exception:
            pass

    return {
        "total_documents": count,
        "collection_name": COLLECTION_NAME,
        "db_path": DB_DIR,
        "unique_sources": list(sources),
        "num_sources": len(sources),
    }


def reset_collection(client: chromadb.PersistentClient = None):
    """Delete and recreate the collection (use with caution)."""
    if client is None:
        client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[chroma_manager] Collection '{COLLECTION_NAME}' deleted.")
    except Exception:
        pass
    get_collection(client)
    print(f"[chroma_manager] Collection '{COLLECTION_NAME}' recreated.")


if __name__ == "__main__":
    stats = get_stats()
    print("ChromaDB stats:", stats)
