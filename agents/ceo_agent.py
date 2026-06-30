"""
agents/ceo_agent.py
CEOAgent — strategic reasoning engine.
Receives analysis results from AnalystAgent and generates:
  - Strategic recommendations (deterministic)
  - CEO executive briefing (LLM via HF Inference API)
Saves final results to data/analysis_results.json for the dashboard.
Wired into the LangGraph pipeline via agents/nodes.py::ceo_node.
"""

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "analysis_results.json"
)


def run_ceo_reasoning(analysis: dict, top_n: int = 6) -> dict:
    """
    Generate recommendations and CEO briefing from AnalystAgent output.
    Saves results to data/analysis_results.json.
    Called by agents/nodes.py::ceo_node after AnalystAgent completes.

    Args:
        analysis: dict returned by agents/analyst_agent.run_analysis()
        top_n:    number of recommendations to generate

    Returns:
        Full results dict including recommendations and ceo_briefing.
    """
    from rag.llm_engine import generate_recommendations, generate_ceo_briefing

    opportunities = analysis.get("opportunities", [])
    risks         = analysis.get("risks", [])
    trends        = analysis.get("trends", [])

    print("[CEOAgent] Generating strategic recommendations...")
    recommendations = generate_recommendations(top_n=top_n)

    print("[CEOAgent] Generating executive briefing via LLM...")
    briefing = generate_ceo_briefing(
        opportunities=opportunities,
        risks=risks,
        trends=trends,
        recommendations=recommendations,
    )

    results = {
        "opportunities":   opportunities,
        "risks":           risks,
        "trends":          trends,
        "recommendations": recommendations,
        "ceo_briefing":    briefing,
    }

    # Persist for dashboard
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[CEOAgent] Results saved → {RESULTS_PATH}")
    print(f"[CEOAgent] Generated {len(recommendations)} recommendations")
    return results


if __name__ == "__main__":
    # Standalone test: load cached analysis if available
    from agents.analyst_agent import run_analysis
    analysis = run_analysis(top_n=5)
    result   = run_ceo_reasoning(analysis)
    print("\n── CEO BRIEFING ──\n")
    print(result["ceo_briefing"])
