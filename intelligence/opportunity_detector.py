"""
opportunity_detector.py
Builds structured opportunities directly from ChromaDB retrieval results.
LLM is used only to enrich descriptions — structured fields are computed, not generated.
This guarantees output even when the LLM can't produce valid JSON.
"""

import sys, os, json, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from rag.retriever import retrieve
from rag.ceo_agent import query_llm

detector_name = "opportunity"

OPPORTUNITY_QUERIES = [
    "NVIDIA new market expansion opportunity revenue growth",
    "NVIDIA partnership collaboration announcement launch",
    "NVIDIA AI data center investment demand surge",
    "NVIDIA semiconductor growth emerging technology",
    "NVIDIA autonomous vehicle robotics healthcare expansion",
]

OPPORTUNITY_KEYWORDS = [
    "opportunity","growth","expand","partnership","launch","invest","acquisition",
    "revenue","record","beat","innovation","collaboration","demand","surge","win",
    "breakthrough","new market","billion","milestone","strong","lead",
]

CATEGORY_SIGNALS = {
    "Market Expansion":  ["market","expand","growth","billion","segment","region"],
    "Technology":        ["gpu","ai","cuda","blackwell","hopper","architecture","chip","compute"],
    "Partnership":       ["partner","collaborat","agreement","deal","alliance","join"],
    "Product Launch":    ["launch","release","announce","new product","unveil","introduce"],
    "Financial":         ["revenue","earnings","profit","stock","investor","valuation"],
}


def _infer_category(text):
    text_l = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_l) for cat, kws in CATEGORY_SIGNALS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Market Expansion"


def _infer_impact(score, keyword_hits):
    if score > 0.75 or keyword_hits >= 4: return "High"
    if score > 0.55 or keyword_hits >= 2: return "Medium"
    return "Low"


def _get_description(title, content):
    """Extract a clean, non-duplicated description from scraped content."""
    # Collapse whitespace
    clean = re.sub(r'\s+', ' ', content).strip()
    
    # Pattern 1: "Source. Title Source" — content starts with source name
    # Check if content starts with a short word (source name) before the real text
    if ". " in clean[:50]:
        parts = clean.split(". ", 1)
        if len(parts[0]) < 30:  # short source prefix like "Pluang" or "Yahoo Finance"
            clean = parts[1].strip()
    
    # Pattern 2: title is duplicated — strip it
    if clean.lower().startswith(title[:35].lower()):
        clean = clean[len(title):].strip().lstrip('.-– ')
    
    # Pattern 3: ends with " Source Name" repetition — strip trailing source
    clean = re.sub(r'\s+[A-Z][A-Za-z\s&]{2,25}$', '', clean).strip()
    
    # If still too short, build meaningful description from title keywords
    if len(clean) < 40:
        title_l = title.lower()
        if any(w in title_l for w in ["partnership","collaborat","deal","alliance"]):
            return f"NVIDIA's strategic partnership activity signals growing ecosystem expansion and technology adoption momentum."
        elif any(w in title_l for w in ["revenue","growth","surge","soar","record","billion"]):
            return f"Strong financial performance signals sustained AI infrastructure demand and NVIDIA's market leadership position."
        elif any(w in title_l for w in ["invest","launch","unveil","announce","deploy"]):
            return f"NVIDIA's continued investment and product launches reinforce its competitive moat in AI compute markets."
        elif any(w in title_l for w in ["market","semiconductor","industry","demand"]):
            return f"Expanding semiconductor market conditions create favourable tailwinds for NVIDIA's data center and AI product lines."
        else:
            return f"Intelligence signals indicate a meaningful strategic opportunity for NVIDIA to strengthen its market position."
    
    return clean[:280] + ("…" if len(clean) > 280 else "")




EVIDENCE_NOISE = [
    "my garage", "making me", "$/hr", "&#x27;", "&#x2f;",
    "in all seriousness", "running in my", "show hn", "ask hn",
    "pytorch on java", "smile-deep", "jvm",
]

def _is_clean(text):
    t = text.lower()
    return not any(n in t for n in EVIDENCE_NOISE) and len(text) > 30

def _clean_title(t):
    import re
    t = re.sub(r"(\w)'\s", r"\1's ", t)
    t = re.sub(r"(\w)'$", r"\1's", t)
    t = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&\.]{2,35}\.\.\.$', '', t).strip()
    t = re.sub(r'\s[-–]\s[A-Z][A-Za-z\s&]{2,30}$', '', t).strip()
    return t.strip('…').strip()

def detect_opportunities(n_per_query=6, top_n=5):
    print("[opportunity_detector] Detecting opportunities...")

    candidates = {}
    for q in OPPORTUNITY_QUERIES:
        for doc in retrieve(q, n_results=n_per_query, use_hybrid=True):
            did = doc["id"]
            if did not in candidates or doc["similarity"] > candidates[did]["similarity"]:
                candidates[did] = doc

    if not candidates:
        return _fallback_opportunities()

    # Score each document
    scored = []
    for doc in candidates.values():
        text = doc["content"].lower()
        kw_hits = sum(1 for kw in OPPORTUNITY_KEYWORDS if kw in text)
        doc["_kw_hits"] = kw_hits
        doc["_total_score"] = round(doc["similarity"] * 2 + kw_hits * 0.3, 3)
        scored.append(doc)

    scored.sort(key=lambda d: d["_total_score"], reverse=True)
    top_docs = scored[:top_n]

    # Build structured objects — no LLM for structure
    results = []
    for doc in top_docs:
        title   = doc["metadata"].get("title", doc["content"][:70])
        # Strip trailing source attribution like " - Yahoo Finance"
        title   = _clean_title(re.sub(r'\s[-–]\s[A-Z][^-–]{2,40}$', '', title).strip())
        content = doc["content"]
        sim     = doc["similarity"]
        kw_hits = doc["_kw_hits"]

        impact   = _infer_impact(sim, kw_hits)
        category = _infer_category(content)
        conf     = round(min(sim + kw_hits * 0.04, 0.98), 2)

        # Extract 1-2 evidence phrases (sentences containing keywords)
        sentences = re.split(r'(?<=[.!?])\s+', content)
        ev_raw = [s.strip() for s in sentences if any(kw in s.lower() for kw in OPPORTUNITY_KEYWORDS) and _is_clean(s)]
        seen = set()
        evidence = []
        for e in ev_raw:
            key = e[:40].lower()
            if key not in seen and len(e) > 20:
                seen.add(key)
                evidence.append(e)
                if len(evidence) >= 2: break
        if not evidence:
            evidence = [content[:180]]

        desc = _get_description(title, content)

        results.append({
            "title":            title[:120],
            "description":      desc,
            "impact_level":     impact,
            "confidence_score": conf,
            "evidence":         evidence,
            "category":         category,
        })

    print(f"[opportunity_detector] Found {len(results)} opportunities")
    return results


def _fallback_opportunities():
    return [{"title":"No data available","description":"Run scrapers and embeddings first.","impact_level":"Low","confidence_score":0.0,"evidence":[],"category":"N/A"}]


if __name__ == "__main__":
    print(json.dumps(detect_opportunities(), indent=2))
