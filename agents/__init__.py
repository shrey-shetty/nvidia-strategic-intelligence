"""
agents/
LangGraph pipeline package for the NVIDIA Strategic Intelligence Agent.

Graph topology:
    START → ScraperNode → IndexerNode → [conditional] →
    AnalystNode → CEONode → END

Each node in nodes.py is a thin LangGraph adapter that delegates
the actual work to its matching agent module:
    scraper_node  → scraper_agent.run_scrapers()
    indexer_node  → indexer_agent.run_indexer()
    analyst_node  → analyst_agent.run_analysis()
    ceo_node      → ceo_agent.run_ceo_reasoning()

Modules:
    graph_state.py      — PipelineState TypedDict (shared node state)
    nodes.py             — LangGraph node adapters
    scraper_agent.py      — Live data collection (NVIDIA, news, HN)
    indexer_agent.py      — Embedding + ChromaDB storage
    analyst_agent.py      — Opportunity/risk/trend detection
    ceo_agent.py           — Recommendations + executive briefing (HF LLM)
    orchestrator.py        — build_graph(), run_pipeline()
    strategic_agent.py     — Goal→Plan→Retrieve→Analyze→Decide→
                              Recommend→Validate reasoning loop
"""

from agents.graph_state import PipelineState
from agents.nodes import (
    scraper_node, indexer_node, analyst_node, ceo_node, should_continue,
)
from agents.orchestrator import (
    build_graph, run_pipeline, print_graph_structure,
)
from agents.strategic_agent import StrategicAgent

__all__ = [
    "PipelineState",
    "scraper_node", "indexer_node", "analyst_node", "ceo_node", "should_continue",
    "build_graph", "run_pipeline", "print_graph_structure",
    "StrategicAgent",
]
