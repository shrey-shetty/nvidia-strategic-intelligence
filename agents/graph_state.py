"""
agents/graph_state.py
LangGraph pipeline state definition.
PipelineState is the single shared object that flows through every node.
Each node reads from it and writes its results back into it.
"""

from typing import TypedDict, Optional


class PipelineState(TypedDict):
    # ── Stage flags ───────────────────────────────────────────────
    collect:   bool          # whether to run scraping
    reset_db:  bool          # whether to wipe ChromaDB before indexing
    top_n:     int           # number of items per category

    # ── ScraperNode output ────────────────────────────────────────
    scrape_result: Optional[dict]   # {total_documents, nvidia_official, ...}

    # ── IndexerNode output ────────────────────────────────────────
    index_result: Optional[dict]    # {documents_indexed}

    # ── AnalystNode output ────────────────────────────────────────
    opportunities: Optional[list]
    risks:         Optional[list]
    trends:        Optional[list]

    # ── CEONode output ────────────────────────────────────────────
    recommendations: Optional[list]
    ceo_briefing:    Optional[str]

    # ── Pipeline metadata ─────────────────────────────────────────
    errors:  Optional[list[str]]
    status:  Optional[str]          # "running" | "complete" | "error"
