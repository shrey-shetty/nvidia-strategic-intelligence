"""
agents/strategic_agent.py
Explicit strategic agent loop for dashboard-driven reasoning.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intelligence.opportunity_detector import (
    detect_opportunities,
    OPPORTUNITY_KEYWORDS,
    CATEGORY_SIGNALS as OPPORTUNITY_CATEGORY_SIGNALS,
)
from intelligence.risk_detector import (
    detect_risks,
    RISK_KEYWORDS,
    CATEGORY_SIGNALS as RISK_CATEGORY_SIGNALS,
)
from intelligence.trend_detector import detect_trends
from rag.llm_engine import query_llm, grounding_rejection_message, OUT_OF_SCOPE_MESSAGE
from rag.retriever import retrieve_multi


def _flatten_category_signals(category_signals: dict) -> set:
    return {term for terms in category_signals.values() for term in terms}


class StrategicAgent:
    """Run an explicit Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate loop."""

    MAX_RETRIES = 2

    # Out-of-scope query guard: rejects questions unrelated to NVIDIA's
    # business strategy instead of letting the LLM answer/hallucinate.
    # Merged with the keyword lists already defined in
    # intelligence/opportunity_detector.py and intelligence/risk_detector.py
    # (OPPORTUNITY_KEYWORDS, RISK_KEYWORDS, and both files' CATEGORY_SIGNALS)
    # rather than duplicating those terms here.
    IN_SCOPE_TERMS = (
        {
            "nvidia", "gpu", "ai chip", "data center", "blackwell", "cuda",
            "h100", "a100", "stock", "nvda", "revenue", "competitor", "market",
            "risk", "opportunity", "trend", "strategy", "investment",
            "partnership", "regulation", "export", "china", "demand",
            "supply chain", "earnings", "growth", "amd", "intel", "tsmc",
            "semiconductor", "chip", "ai", "llm", "cloud",
        }
        | set(OPPORTUNITY_KEYWORDS)
        | set(RISK_KEYWORDS)
        | _flatten_category_signals(OPPORTUNITY_CATEGORY_SIGNALS)
        | _flatten_category_signals(RISK_CATEGORY_SIGNALS)
    )

    def _is_in_scope(self, query: str) -> bool:
        """Case-insensitive substring match against IN_SCOPE_TERMS."""
        q = query.lower()
        return any(term in q for term in self.IN_SCOPE_TERMS)

    def __init__(self, goal: str):
        self.goal = goal.strip()
        self.trace: list[dict] = []
        self.memory: dict = {
            "plan": [],
            "retrieved_docs": [],
            "opportunities": [],
            "risks": [],
            "trends": [],
            "decision": "",
            "recommendation": "",
        }

    def _append_trace(self, step: str, content: str) -> None:
        self.trace.append({"step": step, "content": content})

    def _deterministic_recommendation(self) -> str:
        opportunities = self.memory.get("opportunities", [])
        risks = self.memory.get("risks", [])
        trends = self.memory.get("trends", [])

        top_opp = opportunities[0].get("title", "AI infrastructure expansion") if opportunities else "AI infrastructure expansion"
        top_risk = risks[0].get("title", "regulatory and competitive pressure") if risks else "regulatory and competitive pressure"
        top_trend = trends[0].get("trend_name", "Data Center & Cloud") if trends else "Data Center & Cloud"

        return (
            f"Prioritise {top_opp.lower()} while aligning execution to the {top_trend} trend. "
            f"Pair that with a concrete mitigation plan for {top_risk.lower()}, focusing on "
            f"partner expansion, accelerated product delivery, and tighter regulatory scenario planning."
        )

    def _build_recommendation_prompt(self) -> str:
        opportunities = self.memory.get("opportunities", [])[:3]
        risks = self.memory.get("risks", [])[:3]
        trends = self.memory.get("trends", [])[:3]
        decision = self.memory.get("decision", "")

        opp_text = "; ".join(item.get("title", "") for item in opportunities) or "No major opportunities detected"
        risk_text = "; ".join(item.get("title", "") for item in risks) or "No major risks detected"
        trend_text = "; ".join(item.get("trend_name", "") for item in trends) or "No major trends detected"

        return (
            "You are NVIDIA's strategic advisor. Produce a concise executive recommendation in 2 short paragraphs. "
            "Be specific, practical, and focused on immediate management action.\n\n"
            f"GOAL: {self.goal}\n"
            f"DECISION FOCUS: {decision}\n"
            f"TOP OPPORTUNITIES: {opp_text}\n"
            f"TOP RISKS: {risk_text}\n"
            f"TOP TRENDS: {trend_text}\n"
        )

    def run(self) -> dict:
        goal_text = self.goal or "Provide NVIDIA with the most important strategic action to take next."
        self.goal = goal_text
        self._append_trace("GOAL", goal_text)

        if not self._is_in_scope(goal_text):
            self._append_trace(
                "VALIDATE",
                f"Query rejected as out of scope: \"{goal_text}\" does not relate to NVIDIA's business strategy.",
            )
            raise ValueError(OUT_OF_SCOPE_MESSAGE)

        plan = [
            "Retrieve evidence tied to the user's strategic question.",
            "Cross-check broad opportunity, risk, and trend signals.",
            "Choose the highest-leverage action and validate the recommendation.",
        ]
        self.memory["plan"] = plan
        self._append_trace("PLAN", " ".join(plan))

        # Grounding check against the user's actual question ONLY — not the
        # broader query set below, which always includes generic NVIDIA
        # boilerplate ("NVIDIA opportunities risks trends strategic outlook")
        # that returns high-similarity NVIDIA docs regardless of whether the
        # real goal_text is on-topic, masking ungrounded queries if checked
        # against the combined pool (confirmed empirically: a Germany-climate
        # goal scored 0.72 combined vs. 0.18 alone).
        goal_only_docs = retrieve_multi([goal_text], n_per_query=4)
        rejection = grounding_rejection_message(goal_only_docs)
        if rejection:
            self._append_trace("VALIDATE", f"Query rejected as ungrounded: {rejection}")
            raise ValueError(rejection)

        queries = [
            goal_text,
            f"NVIDIA strategy {goal_text}",
            "NVIDIA opportunities risks trends strategic outlook",
        ]
        docs = retrieve_multi(queries, n_per_query=4)
        self.memory["retrieved_docs"] = docs
        retrieve_summary = (
            f"Retrieved {len(docs)} documents. Top signals: "
            + "; ".join(doc.get("metadata", {}).get("title", "Untitled")[:70] for doc in docs[:3])
        ) if docs else "No documents retrieved from the knowledge base."
        self._append_trace("RETRIEVE", retrieve_summary)

        opportunities = detect_opportunities(top_n=3)
        risks = detect_risks(top_n=3)
        trends = detect_trends(top_n=3)
        self.memory["opportunities"] = opportunities
        self.memory["risks"] = risks
        self.memory["trends"] = trends
        self._append_trace(
            "ANALYZE",
            (
                f"Opportunities: {', '.join(item.get('title', '') for item in opportunities[:2]) or 'None'}. "
                f"Risks: {', '.join(item.get('title', '') for item in risks[:2]) or 'None'}. "
                f"Trends: {', '.join(item.get('trend_name', '') for item in trends[:2]) or 'None'}."
            ),
        )

        top_opp = opportunities[0].get("title", "AI infrastructure growth") if opportunities else "AI infrastructure growth"
        top_risk = risks[0].get("title", "geopolitical and regulatory exposure") if risks else "geopolitical and regulatory exposure"
        top_trend = trends[0].get("trend_name", "Data Center & Cloud") if trends else "Data Center & Cloud"
        decision = (
            f"Lean into {top_opp} because it aligns with {top_trend}, "
            f"while reducing downside from {top_risk}."
        )
        self.memory["decision"] = decision
        self._append_trace("DECIDE", decision)

        recommendation = query_llm(self._build_recommendation_prompt(), max_tokens=220, temperature=0.2)
        self.memory["recommendation"] = recommendation
        self._append_trace("RECOMMEND", recommendation)

        validated = recommendation
        for attempt in range(self.MAX_RETRIES + 1):
            if validated and not validated.startswith("[LLM"):
                validation_note = "Recommendation passed validation checks."
                if attempt > 0:
                    validation_note = f"Recommendation recovered after {attempt} retry attempt(s)."
                self._append_trace("VALIDATE", validation_note)
                break

            if attempt < self.MAX_RETRIES:
                failed_value = validated
                validated = query_llm(self._build_recommendation_prompt(), max_tokens=220, temperature=0.2)
                self.memory["recommendation"] = validated
                self._append_trace(
                    "VALIDATE",
                    f"LLM validation failed with '{failed_value[:80]}'. Retrying ({attempt + 1}/{self.MAX_RETRIES}).",
                )
            else:
                validated = self._deterministic_recommendation()
                self.memory["recommendation"] = validated
                self._append_trace(
                    "VALIDATE",
                    "LLM remained unavailable after retries. Fell back to a deterministic recommendation.",
                )

        return {
            "goal": self.goal,
            "recommendation": self.memory["recommendation"],
            "trace": self.trace,
            "memory": self.memory,
        }
