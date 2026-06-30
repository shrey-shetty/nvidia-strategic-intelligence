"""
agents/indexer_agent.py
IndexerAgent — embeds and stores documents in ChromaDB.
Calls embed_documents.run() and reports how many docs were stored.
Wired into the LangGraph pipeline via agents/nodes.py::indexer_node.
"""

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_indexer(reset: bool = False) -> dict:
    """
    Load raw JSON files, generate embeddings, store in ChromaDB.
    Called by agents/nodes.py::indexer_node after scraping.
    """
    from embeddings.embed_documents import run as embed_run

    print("[IndexerAgent] Starting embedding pipeline...")
    count = embed_run(reset=reset)

    result = {
        "status":            "complete",
        "documents_indexed": count,
    }

    print(f"[IndexerAgent] Indexed {count} documents into ChromaDB")
    return result


if __name__ == "__main__":
    print(json.dumps(run_indexer(), indent=2))
