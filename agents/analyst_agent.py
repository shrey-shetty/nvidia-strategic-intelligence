"""
agents/analyst_agent.py
AnalystAgent — strategic intelligence analysis.
Runs opportunity, risk, and trend detection — all deterministic (no LLM).
Returns structured JSON results consumed by the CEO Agent.
Wired into the LangGraph pipeline via agents/nodes.py::analyst_node.
"""

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_analysis(top_n: int = 5) -> dict:
    """
    Run all three detectors and return structured intelligence.
    Called by agents/nodes.py::analyst_node after indexing.
    """
    from intelligence.opportunity_detector import detect_opportunities
    from intelligence.risk_detector        import detect_risks
    from intelligence.trend_detector       import detect_trends

    print("[AnalystAgent] Detecting opportunities...")
    opportunities = detect_opportunities(top_n=top_n)

    print("[AnalystAgent] Detecting risks...")
    risks = detect_risks(top_n=top_n)

    print("[AnalystAgent] Detecting trends...")
    trends = detect_trends(top_n=top_n + 1)  # +1 for the sovereign AI theme

    result = {
        "status":        "complete",
        "opportunities": opportunities,
        "risks":         risks,
        "trends":        trends,
        "counts": {
            "opportunities": len(opportunities),
            "risks":         len(risks),
            "trends":        len(trends),
        },
    }

    print(
        f"[AnalystAgent] Analysis complete — "
        f"{len(opportunities)} opps, {len(risks)} risks, {len(trends)} trends"
    )
    return result


if __name__ == "__main__":
    result = run_analysis()
    # Print summary without full evidence lists
    print(json.dumps(result["counts"], indent=2))
