"""
test_hallucination_grounding.py
Guards against the agent fabricating facts that aren't present in the
retrieved source documents.

Scope:
  - opportunity_detector / risk_detector / trend_detector / generate_recommendations
    are deterministic (no LLM call builds their structured fields), so we assert
    their "evidence" text is a verbatim substring of a real source document —
    any mismatch means the code is inventing evidence.
  - generate_ceo_briefing / answer_strategic_question call an LLM for free text,
    which can't be hallucination-checked deterministically. Instead we assert
    (a) the deterministic fallback (used when the LLM is unavailable) only
    echoes real input titles, and (b) the prompt/context built for the LLM is
    itself grounded in real retrieved documents and excludes noise. Verifying
    the live LLM response text would require an LLM-as-judge test, which is
    out of scope here.
"""
import copy
import re
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intelligence import opportunity_detector, risk_detector, trend_detector
from rag import llm_engine


def fake_retrieve(docs):
    """Replacement for rag.retriever.retrieve that ignores the query and
    returns a fresh copy of a fixed document pool, regardless of n_results."""
    def _retrieve(query_text, n_results=8, use_hybrid=True, source_filter=None):
        return copy.deepcopy(docs)
    return _retrieve


def fake_retrieve_multi(docs):
    def _retrieve_multi(queries, n_per_query=5, deduplicate=True):
        return copy.deepcopy(docs)
    return _retrieve_multi


def assert_grounded(evidence_list, source_docs):
    """Every evidence string must be a verbatim substring of at least one
    source document's content — i.e. traceable, not invented."""
    assert evidence_list, "expected non-empty evidence"
    for ev in evidence_list:
        assert any(ev in d["content"] for d in source_docs), (
            f"Evidence not found verbatim in any source document: {ev!r}"
        )


# ── Fixtures ────────────────────────────────────────────────────────────────

OPP_DOCS = [
    {
        "id": "opp-cloud-1",
        "content": (
            "NVIDIA announced a major partnership with Amazon Web Services to expand its data center AI "
            "infrastructure footprint. This new partnership and growth in cloud demand creates a meaningful "
            "billion dollar revenue opportunity for the company in 2026."
        ),
        "metadata": {"title": "NVIDIA Partners with AWS on Data Center AI Expansion"},
        "similarity": 0.81,
    },
    {
        "id": "opp-health-1",
        "content": (
            "NVIDIA launched BioNeMo, a new healthcare AI platform aimed at drug discovery. The launch signals "
            "strong momentum in NVIDIA's push into the healthcare market and represents a record investment "
            "in life sciences."
        ),
        "metadata": {"title": "NVIDIA Launches BioNeMo Healthcare AI Platform"},
        "similarity": 0.74,
    },
]

OPP_NOISE_DOC = {
    "id": "opp-noise-1",
    "content": (
        "My garage project saw incredible growth and partnership interest, making me about $5/hr though. "
        "NVIDIA's enterprise partnership pipeline for data center growth remains strong heading into next year."
    ),
    "metadata": {"title": "Hacker News: My NVIDIA Side Project"},
    "similarity": 0.7,
}

RISK_DOCS = [
    {
        "id": "risk-export-1",
        "content": (
            "The US government imposed new export restrictions and a ban on selling advanced GPUs to China. "
            "This regulatory risk and ban could cause a significant decline in NVIDIA's addressable market."
        ),
        "metadata": {"title": "US Export Ban Threatens NVIDIA China Sales"},
        "similarity": 0.79,
    },
    {
        "id": "risk-competitor-1",
        "content": (
            "AMD unveiled a new GPU directly challenging NVIDIA's data center lineup. Analysts warn this "
            "competition and market share loss could pressure NVIDIA's pricing power and margins."
        ),
        "metadata": {"title": "AMD Launches Competing Data Center GPU"},
        "similarity": 0.69,
    },
]

TREND_DOCS = [
    {
        "id": "trend-genai-1",
        "content": (
            "Generative AI and large language model adoption continues to accelerate across enterprise cloud "
            "deployments, with foundation model providers racing to scale inference capacity." * 1
        ),
        "metadata": {"title": "Generative AI Adoption Accelerates in the Enterprise"},
        "similarity": 0.77,
    },
    {
        "id": "trend-genai-2",
        "content": (
            "ChatGPT-style language model usage in enterprise settings has grown sharply, pushing cloud "
            "providers to invest heavily in generative AI infrastructure."
        ),
        "metadata": {"title": "Enterprises Race to Deploy LLM-Based Products"},
        "similarity": 0.72,
    },
]

REC_DOCS = [
    {
        "id": "rec-datacenter-1",
        "content": (
            "NVIDIA's data center and cloud infrastructure investment continues to grow, with Blackwell GPU "
            "demand from hyperscalers reaching record levels this quarter."
        ),
        "metadata": {"title": "NVIDIA Data Center Demand Hits Record Levels"},
        "similarity": 0.83,
    },
    {
        "id": "rec-partner-1",
        "content": (
            "NVIDIA expanded its partner ecosystem with new agreements alongside AWS, Azure, and Google Cloud "
            "to deploy next-generation AI infrastructure."
        ),
        "metadata": {"title": "NVIDIA Expands Cloud Partner Ecosystem"},
        "similarity": 0.76,
    },
    {
        "id": "rec-cuda-1",
        "content": (
            "Developers continue to favour NVIDIA's CUDA software platform and ecosystem over alternatives, "
            "reinforcing the company's competitive moat among AI developers."
        ),
        "metadata": {"title": "CUDA Ecosystem Remains Developer Favourite"},
        "similarity": 0.71,
    },
    {
        "id": "rec-auto-1",
        "content": (
            "NVIDIA's automotive division announced new robotics and edge partnerships, diversifying its "
            "revenue base beyond the core data center business."
        ),
        "metadata": {"title": "NVIDIA Automotive and Robotics Push Continues"},
        "similarity": 0.68,
    },
    {
        "id": "rec-china-1",
        "content": (
            "China export regulation and government policy changes continue to create uncertainty for "
            "NVIDIA's addressable market in the region."
        ),
        "metadata": {"title": "China Policy Changes Create Regulatory Uncertainty"},
        "similarity": 0.66,
    },
    {
        "id": "rec-sovereign-1",
        "content": (
            "Several national governments in Europe and India are investing in sovereign AI infrastructure, "
            "creating new government procurement opportunities for NVIDIA."
        ),
        "metadata": {"title": "Sovereign AI Investment Grows Across Europe and India"},
        "similarity": 0.64,
    },
]


# ── Opportunity detector ─────────────────────────────────────────────────────

def test_opportunity_evidence_is_grounded_in_source_docs(monkeypatch):
    monkeypatch.setattr(opportunity_detector, "retrieve", fake_retrieve(OPP_DOCS))
    results = opportunity_detector.detect_opportunities(top_n=2)
    assert results and results[0]["title"] != "No data available"
    for opp in results:
        assert_grounded(opp["evidence"], OPP_DOCS)


def test_opportunity_evidence_excludes_noise_sentences(monkeypatch):
    monkeypatch.setattr(opportunity_detector, "retrieve", fake_retrieve([OPP_NOISE_DOC]))
    results = opportunity_detector.detect_opportunities(top_n=1)
    for opp in results:
        for ev in opp["evidence"]:
            assert "garage" not in ev.lower()
            assert "$5/hr" not in ev.lower()
        assert_grounded(opp["evidence"], [OPP_NOISE_DOC])


# ── Risk detector ────────────────────────────────────────────────────────────

def test_risk_evidence_is_grounded_in_source_docs(monkeypatch):
    monkeypatch.setattr(risk_detector, "retrieve", fake_retrieve(RISK_DOCS))
    results = risk_detector.detect_risks(top_n=2)
    assert results and results[0]["title"] != "No data available"
    for risk in results:
        assert_grounded(risk["evidence"], RISK_DOCS)


# ── Trend detector ───────────────────────────────────────────────────────────

def test_trend_evidence_is_grounded_in_source_docs(monkeypatch):
    monkeypatch.setattr(trend_detector, "retrieve_multi", fake_retrieve_multi(TREND_DOCS))
    results = trend_detector.detect_trends(top_n=3)
    assert results and results[0]["trend_name"] != "No data available"
    for trend in results:
        for ev in trend["evidence"]:
            assert any(ev in d["content"] for d in TREND_DOCS), (
                f"Trend evidence not found verbatim in any source document: {ev!r}"
            )


# ── CEO agent: generate_recommendations (deterministic) ─────────────────────

def test_recommendations_evidence_is_grounded_in_source_docs(monkeypatch):
    monkeypatch.setattr(llm_engine, "retrieve_multi", fake_retrieve_multi(REC_DOCS))
    results = llm_engine.generate_recommendations(top_n=6)
    assert results
    for rec in results:
        assert_grounded(rec["supporting_evidence"], REC_DOCS)

        m = re.search(r'Intelligence signal: "(.*?)\."', rec["rationale"])
        if m:
            quoted_title = m.group(1)
            cleaned_source_titles = {
                llm_engine._clean_title(d["metadata"]["title"]) for d in REC_DOCS
            }
            assert quoted_title in cleaned_source_titles, (
                f"Rationale quotes a title not present in any source document: {quoted_title!r}"
            )


def test_recommendations_excludes_low_quality_docs(monkeypatch):
    noisy_doc = {
        "id": "rec-noise-1",
        "content": "buy nvda calls, yolo, this is going to the moon. data center growth is real.",
        "metadata": {"title": "Reddit Yolo Post"},
        "similarity": 0.9,
    }
    too_short_doc = {
        "id": "rec-short-1",
        "content": "NVIDIA data center growth.",
        "metadata": {"title": "Too Short"},
        "similarity": 0.95,
    }
    monkeypatch.setattr(
        llm_engine, "retrieve_multi", fake_retrieve_multi([noisy_doc, too_short_doc] + REC_DOCS)
    )
    results = llm_engine.generate_recommendations(top_n=6)
    used_titles = {rec["supporting_evidence"][0] for rec in results if rec["supporting_evidence"]}
    for ev in used_titles:
        assert "yolo" not in ev.lower()
        assert "buy nvda" not in ev.lower()


# ── CEO agent: generate_ceo_briefing ─────────────────────────────────────────

def test_briefing_fallback_only_echoes_real_input_titles(monkeypatch):
    """When the LLM is unavailable, the deterministic fallback must only ever
    surface the titles it was actually given — never fabricate new ones."""
    monkeypatch.setattr(llm_engine, "query_llm", lambda *a, **k: "")

    opportunities = [{"title": "Real Opportunity Title A"}, {"title": "Real Opportunity Title B"}]
    risks = [{"title": "Real Risk Title A"}]
    recommendations = [
        {"recommendation": "Real Recommendation A"},
        {"recommendation": "Real Recommendation B"},
        {"recommendation": "Real Recommendation C"},
    ]

    briefing = llm_engine.generate_ceo_briefing(
        opportunities=opportunities, risks=risks, trends=[], recommendations=recommendations
    )

    for item, key in [(opportunities, "title"), (risks, "title")]:
        for entry in item:
            assert entry[key] in briefing
    for rec in recommendations:
        assert rec["recommendation"] in briefing


def test_briefing_prompt_contains_only_real_input_titles(monkeypatch):
    """The prompt sent to the LLM must be built only from the real titles
    passed in — guards against fabricated placeholder text leaking into
    what the model is told actually happened."""
    captured = {}

    def fake_query_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "x" * 100  # long enough to skip the fallback path

    monkeypatch.setattr(llm_engine, "query_llm", fake_query_llm)

    opportunities = [{"title": "AI Datacenter Demand Surge"}]
    risks = [{"title": "China Export Restrictions"}]
    recommendations = [{"recommendation": "Accelerate AI Infrastructure Investment"}]

    llm_engine.generate_ceo_briefing(
        opportunities=opportunities, risks=risks, trends=[], recommendations=recommendations
    )

    prompt = captured["prompt"]
    assert "AI Datacenter Demand Surge" in prompt
    assert "China Export Restrictions" in prompt
    assert "Accelerate AI Infrastructure Investment" in prompt
    # nothing besides these three real titles should appear in the data lines
    for line in prompt.splitlines():
        if line.startswith("Key opportunities:"):
            assert line.strip() == "Key opportunities: AI Datacenter Demand Surge"
        if line.startswith("Key risks:"):
            assert line.strip() == "Key risks: China Export Restrictions"
        if line.startswith("Recommended actions:"):
            assert line.strip() == "Recommended actions: Accelerate AI Infrastructure Investment"


# ── CEO agent: answer_strategic_question ─────────────────────────────────────

def test_answer_question_context_is_grounded_and_excludes_noise(monkeypatch):
    clean_doc = {
        "id": "qa-1",
        "content": "NVIDIA's data center revenue grew sharply on strong AI chip demand this quarter.",
        "metadata": {"title": "NVIDIA Data Center Revenue Surges"},
        "similarity": 0.85,
    }
    noisy_doc = {
        "id": "qa-2",
        "content": "buy nvda puts, yolo, this stock is way overvalued lol",
        "metadata": {"title": "Reddit Yolo Thread"},
        "similarity": 0.9,
    }

    monkeypatch.setattr(llm_engine, "retrieve_multi", fake_retrieve_multi([clean_doc, noisy_doc]))

    captured = {}

    def fake_query_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "Answer text."

    monkeypatch.setattr(llm_engine, "query_llm", fake_query_llm)

    llm_engine.answer_strategic_question("What is happening with NVIDIA's revenue?")

    prompt = captured["prompt"]
    evidence_block = prompt.split("EVIDENCE:")[1]
    assert "NVIDIA Data Center Revenue Surges" in evidence_block
    assert "data center revenue grew sharply" in evidence_block
    # noisy/low-quality document must never reach the LLM context
    assert "yolo" not in evidence_block.lower()
    assert "buy nvda" not in evidence_block.lower()
