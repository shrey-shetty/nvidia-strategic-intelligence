"""
agents/orchestrator.py
LangGraph stateful pipeline graph for the NVIDIA Strategic Intelligence Agent.

Architecture — Directed Acyclic Graph (DAG):

    START
      │
      ▼
  ScraperNode ──── collects live data from 3 sources
      │
      ▼
  IndexerNode ──── embeds + stores in ChromaDB
      │
      ▼ (conditional edge)
  should_continue? ── "abort" ──► END
      │ "continue"
      ▼
  AnalystNode ──── detects opportunities, risks, trends
      │
      ▼
   CEONode ──── generates recommendations + executive briefing
      │
      ▼
     END

Why LangGraph over AutoGen / CrewAI:
  - Models the pipeline as an explicit stateful graph (nodes + edges)
  - PipelineState flows through every node — no side-channel data passing
  - Conditional edges handle failure gracefully (no documents → abort)
  - Industry-standard: used in production at LinkedIn, Replit, Uber
  - Checkpointing support for long-running pipelines (extensible)

Usage:
    from agents.orchestrator import run_pipeline
    results = run_pipeline(collect=True, reset_db=False, top_n=5)
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END

from agents.graph_state import PipelineState
from agents.nodes import (
    scraper_node,
    indexer_node,
    analyst_node,
    ceo_node,
    should_continue,
)


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph pipeline.

    Graph topology:
        scraper → indexer → [conditional] → analyst → ceo → END
                                         ↘ END (if no docs indexed)

    Returns a compiled graph ready for .invoke().
    """
    graph = StateGraph(PipelineState)

    # ── Register nodes ────────────────────────────────────────────
    graph.add_node("scraper",  scraper_node)
    graph.add_node("indexer",  indexer_node)
    graph.add_node("analyst",  analyst_node)
    graph.add_node("ceo",      ceo_node)

    # ── Entry point ───────────────────────────────────────────────
    graph.set_entry_point("scraper")

    # ── Edges ─────────────────────────────────────────────────────
    graph.add_edge("scraper", "indexer")

    # Conditional edge: abort if indexing produced 0 documents
    graph.add_conditional_edges(
        "indexer",
        should_continue,
        {
            "continue": "analyst",
            "abort":    END,
        },
    )

    graph.add_edge("analyst", "ceo")
    graph.add_edge("ceo",     END)

    return graph.compile()


def run_pipeline(
    collect:  bool = True,
    reset_db: bool = False,
    top_n:    int  = 5,
) -> dict:
    """
    Execute the full LangGraph intelligence pipeline.

    Args:
        collect:  If True, run all scrapers for fresh data.
                  If False, skip scraping and use cached raw JSON files.
        reset_db: If True, wipe ChromaDB before re-indexing.
        top_n:    Number of opportunities / risks / trends / recommendations.

    Returns:
        Final PipelineState dict containing all results.
    """
    print("\n" + "="*60)
    print("NVIDIA STRATEGIC INTELLIGENCE — LANGGRAPH PIPELINE")
    print("="*60)
    print(f"  collect={collect}  reset_db={reset_db}  top_n={top_n}")
    print("="*60 + "\n")

    pipeline = build_graph()

    # Initial state — every node reads from and writes back to this
    initial_state: PipelineState = {
        "collect":         collect,
        "reset_db":        reset_db,
        "top_n":           top_n,
        "scrape_result":   None,
        "index_result":    None,
        "opportunities":   None,
        "risks":           None,
        "trends":          None,
        "recommendations": None,
        "ceo_briefing":    None,
        "errors":          [],
        "status":          "running",
    }

    final_state = pipeline.invoke(initial_state)

    # ── Summary ───────────────────────────────────────────────────
    scrape  = final_state.get("scrape_result", {}) or {}
    index   = final_state.get("index_result",  {}) or {}
    opps    = final_state.get("opportunities", []) or []
    risks   = final_state.get("risks",         []) or []
    trends  = final_state.get("trends",        []) or []
    recs    = final_state.get("recommendations",[]) or []

    print("\n" + "="*60)
    print("LANGGRAPH PIPELINE COMPLETE")
    print("="*60)
    print(f"  ScraperNode:   {scrape.get('total_documents', 'skipped')} documents collected")
    print(f"  IndexerNode:   {index.get('documents_indexed', 0)} documents indexed")
    print(f"  AnalystNode:   {len(opps)} opportunities · {len(risks)} risks · {len(trends)} trends")
    print(f"  CEONode:       {len(recs)} strategic recommendations")
    print(f"\n  Graph edges:   scraper → indexer → [conditional] → analyst → ceo → END")
    print(f"  Dashboard:     streamlit run dashboard/app.py")
    print("="*60 + "\n")

    return final_state


def print_graph_structure() -> None:
    """
    Print the graph's node and edge structure.
    Useful for the oral exam to show the pipeline topology.
    """
    pipeline = build_graph()
    print("\n── LangGraph Pipeline Structure ──")
    print("Nodes:", list(pipeline.get_graph().nodes))
    print("Edges:")
    for edge in pipeline.get_graph().edges:
        print(f"  {edge}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--reset",      action="store_true")
    parser.add_argument("--graph",      action="store_true", help="Print graph structure")
    args = parser.parse_args()

    if args.graph:
        print_graph_structure()
    else:
        run_pipeline(
            collect=not args.no_collect,
            reset_db=args.reset,
        )
