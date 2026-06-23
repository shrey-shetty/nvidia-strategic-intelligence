"""
ceo_agent.py
The AI CEO Agent — the core reasoning engine of the system.
Uses a local LLM via Ollama to:
  - Answer strategic questions with evidence
  - Generate prioritised recommendations
  - Produce the CEO Executive Briefing

SETUP:
  1. Install Ollama from https://ollama.com
  2. Pull a model:
       ollama pull llama3.1:8b        (recommended, ~5 GB)
       ollama pull mistral:7b-instruct (alternative)
       ollama pull qwen2.5:7b         (alternative)
  3. Make sure Ollama is running: `ollama serve`
"""

import re
import sys
import os
import json
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from rag.retriever import retrieve_multi

# ── Ollama configuration ──────────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL",      "qwen2.5:3b")    # for analysis
OLLAMA_FAST_MODEL = os.environ.get("OLLAMA_FAST_MODEL", "qwen2.5:0.5b") # for Q&A
# ─────────────────────────────────────────────────────────────────────────────


def query_llm(prompt: str, max_tokens: int = 1024, temperature: float = 0.3, fast: bool = False) -> str:
    """
    Send a prompt to the local Ollama LLM and return the response text.
    Falls back to a placeholder message if Ollama is not running.
    fast=True uses the smaller model for interactive Q&A.
    """
    model = OLLAMA_FAST_MODEL if fast else OLLAMA_MODEL
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "[LLM Unavailable] Ollama is not running. "
            "Start it with: ollama serve\n"
            f"Then pull a model: ollama pull {OLLAMA_MODEL}"
        )
    except Exception as e:
        return f"[LLM Error] {e}"


def check_ollama_status() -> dict:
    """Return dict with 'running' bool and 'model' name."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"running": True, "available_models": models, "configured_model": OLLAMA_MODEL}
    except Exception:
        return {"running": False, "available_models": [], "configured_model": OLLAMA_MODEL}


# ── Core agent methods ────────────────────────────────────────────────────────


def _clean_title(title: str) -> str:
    """Fix common scraper artifacts in titles."""
    # Fix broken apostrophes: "Isn'" -> "Isn't", "NVIDIA'" -> "NVIDIA's"
    title = re.sub(r"(\w)'\s", r"\1's ", title)
    title = re.sub(r"(\w)'$",  r"\1's",  title)
    # Remove trailing source attribution
    title = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&\.]{2,35}\.\.\.$', '', title).strip()
    title = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&]{2,30}$',         '', title).strip()
    # Remove leading/trailing ellipsis
    title = title.strip('…').strip()
    return title


def generate_recommendations(context_docs: list[dict] = None, top_n: int = 5) -> list[dict]:
    """
    Build strategic recommendations from retrieved documents.
    Structured fields (priority, risk_level, timeline) are computed from signals.
    LLM is called only for the CEO briefing paragraph — not here.
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
        ("Accelerate AI infrastructure investment",       "Investment",      ["data center","cloud","h100","blackwell","infrastructure"]),
        ("Expand partner ecosystem and cloud alliances",  "Partnership",     ["partner","aws","azure","google","microsoft","cloud"]),
        ("Strengthen competitive moat through CUDA",      "Competitive",     ["cuda","software","ecosystem","developer","platform"]),
        ("Diversify revenue beyond data center",          "Diversification", ["automotive","healthcare","gaming","robotics","edge"]),
        ("Manage regulatory and geopolitical risks",      "Risk",            ["china","export","regulation","ban","govern","policy"]),
    ]

    # Noise patterns that indicate low-quality community content (HN/Reddit comments)
    NOISE_PATTERNS = [
        "my garage", "i am ", "i'm ", "my house", "i have", "i've",
        "&#x27;", "&#x2f;", "$/hr", "making me", "buy nvda", "buy puts",
        "short ", "yolo", "in all seriousness",
    ]

    # Pre-score every document for positive/negative signal density
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
    }

    results: list[dict] = []
    used_ids: set = set()

    for rec_title, rec_type, theme_kws in ACTION_THEMES[:top_n]:

        # ── 1. Find candidate docs for this theme ─────────────────────────────
        def is_quality(d: dict) -> bool:
            """Reject very short docs and personal/noisy community content."""
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

        # Fall back to any quality doc not yet used if no theme match
        if not theme_docs:
            theme_docs = [d for d in scored if d["id"] not in used_ids and is_quality(d)]

        if not theme_docs:
            continue  # skip theme if no usable document exists

        # ── 2. Select the single best document for this recommendation ────────
        best = theme_docs[0]
        used_ids.add(best["id"])  # mark as used — only done once, for the chosen doc

        # ── 3. Derive structured fields from retrieval signals ─────────────────
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

        # ── 4. Extract supporting evidence sentences from the chosen doc ───────
        sentences  = re.split(r'(?<=[.!?])\s+', best["content"])
        evidence   = [s.strip() for s in sentences if len(s) > 30][:2]
        if not evidence:
            evidence = [best["content"][:180]]

        # ── 5. Build rationale from document metadata (no LLM required) ───────
        doc_title = best["metadata"].get("title", "")
        # Strip trailing source attribution e.g. "Title – Reuters"
        doc_title_clean = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&]{2,30}$', '', doc_title).strip()

        # Build content snippet, removing any leading duplication of the title
        content_snip = re.sub(r'\s+', ' ', best["content"]).strip()
        if content_snip.lower().startswith(doc_title_clean[:35].lower()):
            content_snip = content_snip[len(doc_title_clean):].strip().lstrip(".-– ")
        # Strip trailing source attribution from snippet too
        content_snip = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&]{2,30}$', '', content_snip).strip()
        # Remove duplicated opening phrase (artifact of some scrapers)
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

    # Sort High → Medium → Low
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
    LLM is used here; a deterministic fallback is provided if Ollama is unavailable.
    """
    opp_titles  = [o.get("title", "")[:80] for o in (opportunities  or [])[:3]]
    risk_titles = [r.get("title", "")[:80] for r in (risks          or [])[:3]]
    rec_titles  = [r.get("recommendation", "")[:80] for r in (recommendations or [])[:3]]

    prompt = f"""Write a 3-paragraph executive briefing for NVIDIA's CEO.

Key opportunities: {"; ".join(opp_titles)}
Key risks: {"; ".join(risk_titles)}
Recommended actions: {"; ".join(rec_titles)}

Paragraph 1 - SITUATION: What is happening in NVIDIA's market right now.
Paragraph 2 - SIGNIFICANCE: Why these developments matter strategically.
Paragraph 3 - ACTION: The top 3 things management should do immediately.

Write in plain business English. Be specific. No filler phrases."""

    result = query_llm(prompt, max_tokens=400, temperature=0.2)

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
    """
    docs = retrieve_multi([question], n_per_query=n_context)

    if not docs:
        return "No relevant documents found. Please run the data collection pipeline first."

    # Filter out low-quality community noise before building context
    NOISE = [
        "my garage", "i'm ", "i am ", "&#x27;", "&#x2f;",
        "$/hr", "making me", "in all seriousness",
        "buy nvda", "buy puts", "short ", "yolo",
    ]
    docs = [d for d in docs if not any(n in d["content"].lower() for n in NOISE)]

    if not docs:
        return "Insufficient quality documents found. Please refresh the data pipeline."

    context = "\n\n".join(
        f"[{i+1}] {d['metadata'].get('title','')[:60]}: {d['content'][:150]}"
        for i, d in enumerate(docs[:5])
    )

    prompt = f"""You are NVIDIA's strategic advisor. Answer this question using the evidence below.
Be specific and focus on strategic implications. Write 2-3 short paragraphs.

QUESTION: {question}

EVIDENCE:
{context}

Answer:"""

    return query_llm(prompt, max_tokens=300, temperature=0.2, fast=True)


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
    status = check_ollama_status()
    print("Ollama status:", status)
    if status["running"]:
        answer = answer_strategic_question("What are NVIDIA's biggest opportunities in 2025?")
        print("\n" + answer)