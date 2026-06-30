"""
agents/nodes.py
LangGraph node functions — one per pipeline stage.

Each node is a thin adapter: it pulls what it needs from PipelineState,
delegates the actual work to the matching agent module, and returns a
PARTIAL state dict with only the fields it updates. LangGraph merges
these partial updates automatically.

Nodes:
  scraper_node  — delegates to agents.scraper_agent.run_scrapers()
  indexer_node  — delegates to agents.indexer_agent.run_indexer()
  analyst_node  — delegates to agents.analyst_agent.run_analysis()
  ceo_node      — delegates to agents.ceo_agent.run_ceo_reasoning()
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.graph_state import PipelineState


# ── Node 1: Scraper ───────────────────────────────────────────────────────────

def scraper_node(state: PipelineState) -> dict:
    """
    Collect live NVIDIA intelligence from three independent sources:
      - NVIDIA newsroom + blog (RSS)
      - News RSS feeds (Google News, Yahoo Finance, TechCrunch, etc.)
      - Hacker News (Algolia API — no credentials needed)

    Skipped if state['collect'] is False (use cached raw data).
    """
    if not state.get("collect", True):
        print("[ScraperNode] Skipping — using cached raw data")
        return {"scrape_result": {"status": "skipped", "total_documents": 0}}

    from agents.scraper_agent import run_scrapers

    result = run_scrapers()
    return {"scrape_result": result}


# ── Node 2: Indexer ───────────────────────────────────────────────────────────

def indexer_node(state: PipelineState) -> dict:
    """
    Clean, embed, and store documents in ChromaDB.
    Uses all-MiniLM-L6-v2 (local, no API key) for 384-dim embeddings.
    Reads raw JSON files written by scraper_node.
    """
    from agents.indexer_agent import run_indexer

    result = run_indexer(reset=state.get("reset_db", False))
    return {"index_result": result}


# ── Node 3: Analyst ───────────────────────────────────────────────────────────

def analyst_node(state: PipelineState) -> dict:
    """
    Extract strategic intelligence from ChromaDB using:
      - Semantic retrieval (cosine similarity via ChromaDB)
      - Keyword signal scoring (deterministic — no LLM needed)
      - Topic clustering for trend detection

    Returns structured opportunities, risks, and trends.
    All fields computed deterministically for reliability and auditability.
    """
    from agents.analyst_agent import run_analysis

    result = run_analysis(top_n=state.get("top_n", 5))
    return {
        "opportunities": result["opportunities"],
        "risks":         result["risks"],
        "trends":        result["trends"],
    }


# ── Node 4: CEO ───────────────────────────────────────────────────────────────

def ceo_node(state: PipelineState) -> dict:
    """
    Generate strategic recommendations and the CEO executive briefing.

    Recommendations: computed deterministically (keyword/signal scoring).
    CEO Briefing:    one LLM call to Hugging Face Inference API (Mistral-7B).

    Using the LLM only for free-text prose avoids unreliable JSON output
    from smaller open-source models — a deliberate design choice for reliability.

    Persists results to data/analysis_results.json for the dashboard.
    """
    from agents.ceo_agent import run_ceo_reasoning

    analysis = {
        "opportunities": state.get("opportunities", []),
        "risks":         state.get("risks", []),
        "trends":        state.get("trends", []),
    }
    results = run_ceo_reasoning(analysis, top_n=state.get("top_n", 5) + 1)

    return {
        "recommendations": results["recommendations"],
        "ceo_briefing":    results["ceo_briefing"],
        "status":          "complete",
    }


# ── Conditional edge: should we abort after indexing? ─────────────────────────

def should_continue(state: PipelineState) -> str:
    """
    Conditional edge after IndexerNode.
    If no documents were indexed (e.g. no raw data), route to END.
    Otherwise continue to AnalystNode.
    """
    index_result = state.get("index_result", {})
    if not index_result or index_result.get("documents_indexed", 0) == 0:
        print("[Graph] ⚠️  No documents indexed — aborting pipeline")
        return "abort"
    return "continue"
