"""
rag/llm_engine.py
The LLM/RAG engine — shared Hugging Face access layer used across the system.
Uses Hugging Face Inference Providers (via LangChain) to:
  - Answer strategic questions with evidence
  - Generate prioritised recommendations
  - Produce the CEO Executive Briefing
  - Provide query_llm() / check_hf_status() to intelligence detectors,
    agents/ceo_agent.py, and agents/strategic_agent.py

SETUP:
  1. Get a free Hugging Face API token: https://huggingface.co/settings/tokens
     Make sure "Make calls to Inference Providers" is enabled for the token.
  2. Put it in .env (see .env.example):
       HF_API_TOKEN=hf_xxxxxxxxxxxx
  3. The free tier has a limited monthly request quota — enough for all analysis.
"""

import re
import sys
import os
import json
from typing import Any
from dotenv import load_dotenv
from huggingface_hub import whoami
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.messages import HumanMessage

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    _LANGCHAIN_HF_IMPORT_ERROR = None
except ImportError as e:
    ChatHuggingFace = Any
    HuggingFaceEndpoint = None
    _LANGCHAIN_HF_IMPORT_ERROR = e

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

from rag.retriever import retrieve_multi

# ── Hugging Face configuration ────────────────────────────────────────────────
# Primary model: Mistral 7B Instruct — strong, free, open-source
HF_MODEL      = os.environ.get("HF_MODEL",      "mistralai/Mistral-7B-Instruct-v0.2")
# Fast model for interactive Q&A (smaller, lower latency)
HF_FAST_MODEL = os.environ.get("HF_FAST_MODEL", "microsoft/Phi-3-mini-4k-instruct")
# Inference Providers backend. "auto" only considers providers explicitly
# enabled on the account at https://huggingface.co/settings/inference-providers
# — both default models above are exclusively served by featherless-ai, so
# "auto" 400s with model_not_supported unless that provider is enabled there.
# Pinning it directly here avoids depending on that account setting.
HF_PROVIDER   = os.environ.get("HF_PROVIDER", "featherless-ai")
# ─────────────────────────────────────────────────────────────────────────────

# ── Retrieval-grounding guardrail ───────────────────────────────────────────
# Minimum cosine similarity (chroma_manager.py: similarity = 1 - cosine
# distance) a retrieved document must hit before we let the LLM answer from
# it. Chosen empirically: on-topic NVIDIA queries scored 0.67-0.79 top-doc
# similarity; off-topic queries (climate, recipes, capitals) scored 0.18-0.39
# even though ChromaDB always returns its nearest neighbours regardless of
# relevance. 0.5 sits in the gap between those two clusters with margin on
# both sides — see threshold_check.py audit run during development.
GROUNDING_THRESHOLD = 0.5

# Static fallback for the out-of-scope keyword guard (agents/strategic_agent.py).
OUT_OF_SCOPE_MESSAGE = (
    "This question doesn't appear related to NVIDIA's strategy, market "
    "position, or AI industry topics covered by this system."
)


def grounding_rejection_message(docs: list[dict], threshold: float = GROUNDING_THRESHOLD) -> str | None:
    """
    Returns a specific rejection message if `docs` aren't grounded enough to
    answer from, or None if at least one doc clears the threshold. Distinguishes
    two cases so the message reflects the actual reason, not a generic claim:
      - zero documents retrieved at all
      - documents retrieved, but none relevant enough (reports the actual
        top similarity score and threshold so it's auditable, not vague)
    """
    if not docs:
        return "No documents in the knowledge base matched this query."
    best = max(d.get("similarity", 0) for d in docs)
    if best < threshold:
        return (
            f"The most relevant documents found (similarity: {best:.2f}) "
            f"weren't closely related enough (threshold: {threshold}) to "
            f"confidently answer this question."
        )
    return None


def _get_token() -> str:
    """
    Fetch the Hugging Face API token from the environment at call time.
    Raises EnvironmentError with clear instructions if not set.
    """
    token = os.environ.get("HF_API_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "HF_API_TOKEN is not set.\n"
            "Get a free token at: https://huggingface.co/settings/tokens\n"
            "Then add it to .env:\n"
            "  HF_API_TOKEN=hf_xxxxxxxxxxxx"
        )
    return token


def _get_chat_model(model_id: str, max_tokens: int, temperature: float) -> ChatHuggingFace:
    if _LANGCHAIN_HF_IMPORT_ERROR is not None or HuggingFaceEndpoint is None:
        raise EnvironmentError(
            "langchain-huggingface is not installed.\n"
            "Install project dependencies with:\n"
            "  pip install -r requirements.txt"
        ) from _LANGCHAIN_HF_IMPORT_ERROR

    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        provider=HF_PROVIDER,
        huggingfacehub_api_token=_get_token(),
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),  # HF rejects 0.0
        do_sample=temperature > 0.01,
        timeout=120,
    )
    return ChatHuggingFace(llm=endpoint)


def query_llm(prompt: str, max_tokens: int = 512, temperature: float = 0.3, fast: bool = False) -> str:
    """
    Send a prompt to the Hugging Face Inference Providers chat API (via LangChain)
    and return the response text. fast=True uses the smaller/faster model for
    interactive Q&A. Falls back to a deterministic placeholder if no token or API error.
    """
    model = HF_FAST_MODEL if fast else HF_MODEL

    try:
        chat = _get_chat_model(model, max_tokens, temperature)
    except EnvironmentError as e:
        return f"[LLM Unavailable] {e}"

    try:
        response = chat.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()
    except HfHubHTTPError as e:
        status_code = e.response.status_code if e.response is not None else None
        if status_code in (401, 403):
            return (
                "[LLM Unavailable] Hugging Face token missing, invalid, or lacks "
                "'Inference Providers' permission. Check at: "
                "https://huggingface.co/settings/tokens"
            )
        if status_code == 503:
            return "[LLM Loading] Model is loading on HF servers. Retry in ~30 seconds."
        return f"[LLM Error] HTTP {status_code}: {e}"
    except Exception as e:
        return f"[LLM Error] {e}"


def check_hf_status() -> dict:
    """Return dict with 'available' bool, configured models, and token status."""
    try:
        token = _get_token()
        token_set = True
    except EnvironmentError:
        token = None
        token_set = False

    status = {
        "available":  token_set,
        "token_set":  token_set,
        "model":      HF_MODEL,
        "fast_model": HF_FAST_MODEL,
        "provider":   HF_PROVIDER,
    }
    if token_set:
        try:
            info = whoami(token=token)
            status["token_valid"] = True
            status["hf_username"] = info.get("name", "")
        except HfHubHTTPError:
            status["token_valid"] = False
    return status


# ── Core agent methods ────────────────────────────────────────────────────────


def _clean_title(title: str) -> str:
    """Fix common scraper artifacts in titles."""
    title = re.sub(r"(\w)'\s", r"\1's ", title)
    title = re.sub(r"(\w)'$",  r"\1's",  title)
    title = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&\.]{2,35}\.\.\.$', '', title).strip()
    title = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&]{2,30}$',         '', title).strip()
    title = title.strip('…').strip()
    return title


def generate_recommendations(context_docs: list[dict] = None, top_n: int = 6) -> list[dict]:
    """
    Build strategic recommendations from retrieved documents.
    Structured fields (priority, risk_level, timeline) are computed deterministically
    from keyword/signal counting — no LLM call needed here.
    LLM is reserved for the free-text CEO briefing only.
    """
    if context_docs is None:
        context_docs = retrieve_multi(
            [
                "NVIDIA strategic growth opportunities AI market",
                "NVIDIA risks challenges competitors threats",
                "NVIDIA technology trends future roadmap",
                "NVIDIA financial performance revenue growth",
            ],
            n_per_query=6,
        )

    if not context_docs:
        return _fallback_recommendations()

    POSITIVE_SIGNALS = [
        "growth", "opportunity", "partner", "launch", "invest", "expand",
        "record", "demand", "surge", "lead", "billion", "milestone",
    ]
    NEGATIVE_SIGNALS = [
        "risk", "threat", "decline", "ban", "shortage", "competition",
        "loss", "concern", "restrict", "drop", "miss", "warn",
    ]
    ACTION_THEMES = [
        ("Accelerate AI infrastructure investment",          "Investment",      ["data center","cloud","h100","blackwell","infrastructure"]),
        ("Expand partner ecosystem and cloud alliances",     "Partnership",     ["partner","aws","azure","google","microsoft","cloud"]),
        ("Strengthen competitive moat through CUDA",        "Competitive",     ["cuda","software","ecosystem","developer","platform"]),
        ("Diversify revenue beyond data center",            "Diversification", ["automotive","healthcare","gaming","robotics","edge"]),
        ("Manage regulatory and geopolitical risks",        "Risk",            ["china","export","regulation","ban","govern","policy"]),
        ("Invest in sovereign AI and national infrastructure","Sovereign",      ["sovereign","national","government","ministry","country","europe","india"]),
    ]

    NOISE_PATTERNS = [
        "my garage", "i am ", "i'm ", "my house", "i have", "i've",
        "&#x27;", "&#x2f;", "$/hr", "making me", "buy nvda", "buy puts",
        "short ", "yolo", "in all seriousness",
    ]

    scored: list[dict] = []
    for doc in context_docs:
        text = doc["content"].lower()
        doc["_pos"] = sum(1 for s in POSITIVE_SIGNALS if s in text)
        doc["_neg"] = sum(1 for s in NEGATIVE_SIGNALS if s in text)
        scored.append(doc)

    timeline_map = {
        "Investment":      "short-term (0-6mo)",
        "Partnership":     "medium-term (6-18mo)",
        "Competitive":     "short-term (0-6mo)",
        "Diversification": "long-term (18mo+)",
        "Risk":            "short-term (0-6mo)",
        "Sovereign":       "medium-term (6-18mo)",
    }

    results: list[dict] = []
    used_ids: set = set()

    for rec_title, rec_type, theme_kws in ACTION_THEMES[:top_n]:

        def is_quality(d: dict) -> bool:
            if len(d["content"]) < 80:
                return False
            text_lower = d["content"].lower()
            return not any(noise in text_lower for noise in NOISE_PATTERNS)

        theme_docs = sorted(
            [
                d for d in scored
                if d["id"] not in used_ids
                and is_quality(d)
                and any(k in d["content"].lower() for k in theme_kws)
            ],
            key=lambda d: d["similarity"] + d["_pos"] * 0.1,
            reverse=True,
        )

        if not theme_docs:
            theme_docs = [d for d in scored if d["id"] not in used_ids and is_quality(d)]

        if not theme_docs:
            continue

        best = theme_docs[0]
        used_ids.add(best["id"])

        priority = (
            "High"   if best["similarity"] > 0.70 or best["_pos"] >= 3 else
            "Medium" if best["similarity"] > 0.55 or best["_pos"] >= 1 else
            "Low"
        )
        risk_level = (
            "High"   if best["_neg"] >= 3 else
            "Medium" if best["_neg"] >= 1 else
            "Low"
        )
        timeline = timeline_map.get(rec_type, "medium-term (6-18mo)")

        sentences  = re.split(r'(?<=[.!?])\s+', best["content"])
        evidence   = [s.strip() for s in sentences if len(s) > 30][:2]
        if not evidence:
            evidence = [best["content"][:180]]

        doc_title = best["metadata"].get("title", "")
        doc_title_clean = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&]{2,30}$', '', doc_title).strip()

        content_snip = re.sub(r'\s+', ' ', best["content"]).strip()
        if content_snip.lower().startswith(doc_title_clean[:35].lower()):
            content_snip = content_snip[len(doc_title_clean):].strip().lstrip(".-– ")
        content_snip = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&]{2,30}$', '', content_snip).strip()
        if '. ' in content_snip[20:]:
            mid    = content_snip.find('. ', 20)
            second = content_snip[mid + 2:]
            if second.lower().startswith(content_snip[:20].lower()):
                content_snip = content_snip[:mid + 1]

        if len(doc_title_clean) > 20:
            rationale = (
                f'Intelligence signal: "{doc_title_clean}." '
                f"This directly supports the case to {rec_title.lower()} "
                f"as a high-priority strategic action."
            )
        else:
            rationale = (
                f"Market signals and competitive dynamics indicate "
                f"{rec_title.lower()} is a high-priority action for NVIDIA's strategic roadmap."
            )

        results.append({
            "recommendation": rec_title,
            "rationale":      rationale,
            "priority":       priority,
            "expected_impact": {
                "revenue":         "Directly contributes to revenue diversification and growth.",
                "market_position": "Strengthens NVIDIA's competitive positioning in key segments.",
                "timeline":        timeline,
            },
            "risk_level":          risk_level,
            "risk_description":    f"Inaction risks ceding ground to competitors in {rec_type.lower()} strategy.",
            "supporting_evidence": evidence,
        })

    if not results:
        return _fallback_recommendations()

    order = {"High": 0, "Medium": 1, "Low": 2}
    results.sort(key=lambda r: order.get(r["priority"], 1))
    return results


def generate_ceo_briefing(
    opportunities:   list[dict] = None,
    risks:           list[dict] = None,
    trends:          list[dict] = None,
    recommendations: list[dict] = None,
) -> str:
    """
    Generate a concise executive summary for the CEO dashboard.
    Answers: What happened? Why does it matter? What should we do next?
    Uses Hugging Face Inference API; deterministic fallback if unavailable.
    """
    opp_titles  = [o.get("title", "")[:80] for o in (opportunities  or [])[:3]]
    risk_titles = [r.get("title", "")[:80] for r in (risks          or [])[:3]]
    rec_titles  = [r.get("recommendation", "")[:80] for r in (recommendations or [])[:3]]

    prompt = (
        "Write a 3-paragraph executive briefing for NVIDIA's CEO.\n\n"
        f"Key opportunities: {'; '.join(opp_titles)}\n"
        f"Key risks: {'; '.join(risk_titles)}\n"
        f"Recommended actions: {'; '.join(rec_titles)}\n\n"
        "The first paragraph covers SITUATION: what is happening in NVIDIA's market right now.\n"
        "The second paragraph covers SIGNIFICANCE: why these developments matter strategically.\n"
        "The third paragraph covers ACTION: the top 3 things management should do immediately.\n\n"
        "Write in plain business English. Be specific. No filler phrases. "
        "Do not label the paragraphs. Do not write 'Paragraph 1', 'Paragraph 2', 'SITUATION', "
        "'SIGNIFICANCE', 'ACTION', or any other heading. Write flowing prose directly, "
        "with no titles or numbering of any kind."
    )

    result = query_llm(prompt, max_tokens=450, temperature=0.2)

    # Strip paragraph labels the LLM adds despite instructions
    result = re.sub(r'Paragraph\s*\d+\s*[:\-]?\s*', '', result).strip()

    # Deterministic fallback if LLM is unavailable or returns garbage
    if not result or len(result) < 50 or result.startswith("[LLM"):
        opp_line  = "; ".join(opp_titles)  or "AI infrastructure demand growth"
        risk_line = "; ".join(risk_titles) or "competitive and regulatory pressures"
        r1 = rec_titles[0] if len(rec_titles) > 0 else "Accelerate AI infrastructure investment"
        r2 = rec_titles[1] if len(rec_titles) > 1 else "Strengthen partner ecosystem alliances"
        r3 = rec_titles[2] if len(rec_titles) > 2 else "Manage regulatory and geopolitical risk exposure proactively"
        result = (
            f"**SITUATION**\nNVIDIA is operating in a high-velocity AI market characterised by "
            f"surging demand and intensifying competition. Key developments: {opp_line}. "
            f"Material risks include: {risk_line}.\n\n"
            f"**SIGNIFICANCE**\nThe convergence of record AI chip demand and competitive and "
            f"regulatory pressures creates a critical strategic inflection point. NVIDIA's "
            f"technology leadership and CUDA ecosystem moat remain its primary competitive "
            f"advantages, but proactive management of geopolitical exposure is essential to "
            f"sustaining growth.\n\n"
            f"**ACTIONS REQUIRED**\nManagement should immediately prioritise: "
            f"(1) {r1}. (2) {r2}. (3) {r3}."
        )

    return result


def answer_strategic_question(question: str, n_context: int = 8) -> str:
    """
    Answer a free-form strategic question about NVIDIA using RAG.
    Used for the interactive Q&A panel in the dashboard.
    Uses the fast (smaller) HF model for lower latency.
    """
    docs = retrieve_multi([question], n_per_query=n_context)

    NOISE = [
        "my garage", "i'm ", "i am ", "&#x27;", "&#x2f;",
        "$/hr", "making me", "in all seriousness",
        "buy nvda", "buy puts", "short ", "yolo",
    ]
    docs = [d for d in docs if not any(n in d["content"].lower() for n in NOISE)]

    rejection = grounding_rejection_message(docs)
    if rejection:
        raise ValueError(rejection)

    context = "\n\n".join(
        f"[{i+1}] {d['metadata'].get('title','')[:60]}: {d['content'][:150]}"
        for i, d in enumerate(docs[:5])
    )

    prompt = (
        "You are NVIDIA's strategic advisor. Answer this question using the evidence below. "
        "Be specific and focus on strategic implications. Write 2-3 short paragraphs.\n\n"
        f"QUESTION: {question}\n\n"
        f"EVIDENCE:\n{context}\n\nAnswer:"
    )

    return query_llm(prompt, max_tokens=350, temperature=0.2, fast=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_list(response: str, fallback: list) -> list:
    """Safely parse a JSON list from an LLM response string."""
    try:
        start = response.find("[")
        end   = response.rfind("]") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
            if isinstance(parsed, list) and parsed:
                return parsed
    except Exception:
        pass
    return fallback


def _fallback_recommendations() -> list[dict]:
    return [
        {
            "recommendation": "Run data collection pipeline first",
            "rationale": (
                "No documents in the knowledge base yet. "
                "Execute main.py to collect and embed data."
            ),
            "priority": "High",
            "expected_impact": {
                "revenue":         "N/A",
                "market_position": "N/A",
                "timeline":        "immediate",
            },
            "risk_level":          "Low",
            "risk_description":    "System cannot generate insights without data.",
            "supporting_evidence": [],
        }
    ]


if __name__ == "__main__":
    status = check_hf_status()
    print("HF status:", status)
    answer = answer_strategic_question("What are NVIDIA's biggest opportunities in 2025?")
    print("\n" + answer)
