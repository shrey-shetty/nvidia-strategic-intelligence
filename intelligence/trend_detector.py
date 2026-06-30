"""
trend_detector.py
Detects trends by clustering documents into topic buckets.
No LLM needed for structure — cluster analysis drives the output.
LLM only used for one-sentence opportunity descriptions.
"""

import sys, os, json, re
from collections import Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from rag.retriever import retrieve, retrieve_multi
from rag.llm_engine import query_llm

TREND_QUERIES = [
    "emerging AI technology trend 2025 2026",
    "NVIDIA next generation GPU Blackwell architecture",
    "generative AI adoption enterprise cloud",
    "autonomous vehicle self-driving AI chip",
    "NVIDIA robotics humanoid physical AI",
    "data center energy efficiency AI accelerator",
    "edge AI inference on-device deployment",
    "AI regulation policy government semiconductor",
]

TOPIC_SEEDS = {
    "Generative AI & LLMs":      {"kws":["generative","llm","chatgpt","language model","gpt","foundation model"],"horizon":"Now (0-6mo)","category":"Technology"},
    "Data Center & Cloud":       {"kws":["data center","cloud","hyperscaler","aws","azure","h100","h200","blackwell","rack"],"horizon":"Now (0-6mo)","category":"Market"},
    "Autonomous Vehicles":       {"kws":["autonomous","self-driving","robotaxi","drive orin","lidar"],"horizon":"Near-term (6-18mo)","category":"Technology"},
    "Robotics & Embodied AI":    {"kws":["robot","humanoid","embodied","physical ai","isaac","manipulation"],"horizon":"Near-term (6-18mo)","category":"Technology"},
    "Healthcare & Life Science": {"kws":["drug discovery","genomics","medical imaging","healthcare ai","bionemo","clinical"],"horizon":"Long-term (18mo+)","category":"Market"},
    "Edge AI & Inference":       {"kws":["edge","inference","deployment","jetson","on-device","mobile ai"],"horizon":"Near-term (6-18mo)","category":"Technology"},
    "Energy & Sustainability":   {"kws":["energy","power consumption","sustainability","green","efficiency","cooling"],"horizon":"Long-term (18mo+)","category":"Regulatory"},
    "Gaming & Creative AI":      {"kws":["gaming","rtx","dlss","metaverse","omniverse","creative","rendering"],"horizon":"Now (0-6mo)","category":"Market"},
    "AI Regulation & Policy":    {"kws":["regulation","policy","govern","compliance","act","law","export","ban"],"horizon":"Near-term (6-18mo)","category":"Regulatory"},
    "Competitive Landscape":     {"kws":["amd","intel","qualcomm","competitor","market share","challenge","threat"],"horizon":"Now (0-6mo)","category":"Competitive"},
}

NVIDIA_OPPORTUNITIES = {
    "Generative AI & LLMs":      "Expand NIM microservices and partner with enterprise software vendors for LLM deployment.",
    "Data Center & Cloud":       "Accelerate Blackwell adoption through cloud partnerships and as-a-service offerings.",
    "Autonomous Vehicles":       "Scale Drive Orin platform and deepen OEM integrations for next-gen vehicle programs.",
    "Robotics & Embodied AI":    "Commercialise Isaac platform and capture humanoid robot compute market.",
    "Healthcare & Life Science": "Grow BioNeMo partnerships and position Clara as the standard for medical AI.",
    "Edge AI & Inference":       "Expand Jetson ecosystem into industrial and retail edge deployments.",
    "Energy & Sustainability":   "Lead on compute-per-watt efficiency and market energy savings as a competitive advantage.",
    "Gaming & Creative AI":      "Leverage DLSS and RTX as entry points for creator tool AI monetisation.",
    "AI Regulation & Policy":    "Proactively shape policy and develop compliant chip variants for restricted markets.",
    "Competitive Landscape":     "Maintain CUDA ecosystem moat and accelerate software differentiation.",
}


def _score_topic(text):
    text_l = text.lower()
    scores = {t: sum(1 for kw in info["kws"] if kw in text_l) for t, info in TOPIC_SEEDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Generative AI & LLMs"


def _get_trend_description(topic, snippet):
    """Build clean description from document snippet."""
    clean = re.sub(r'\s+', ' ', snippet).strip()
    
    # Strip leading source name pattern "Source. "
    if ". " in clean[:60]:
        parts = clean.split(". ", 1)
        if len(parts[0]) < 35 and len(parts) > 1:
            clean = parts[1].strip()
    
    # Remove trailing source/title duplication after period
    if ". " in clean[50:]:
        # Check if text after period is a repetition of text before it
        mid = clean.find(". ", 50)
        first_half = clean[:mid].lower()
        second_half = clean[mid+2:].lower()
        if second_half.startswith(first_half[:25]):
            clean = clean[:mid+1]
    
    clean = clean.strip()
    if len(clean) < 40:
        return f"{topic} is an emerging area with significant strategic implications for NVIDIA's product roadmap and revenue growth."
    return clean[:260] + ("…" if len(clean) > 260 else "")


def detect_trends(n_per_query=5, top_n=6):
    print("[trend_detector] Detecting trends...")

    all_docs = retrieve_multi(TREND_QUERIES, n_per_query=n_per_query)
    if not all_docs:
        return _fallback_trends()

    # Cluster docs into topic buckets
    topic_buckets = {t: [] for t in TOPIC_SEEDS}
    for doc in all_docs:
        topic_buckets[_score_topic(doc["content"])].append(doc)

    # Sort by doc count (signal strength)
    sorted_topics = sorted(
        [(t, docs) for t, docs in topic_buckets.items() if docs],
        key=lambda x: (len(x[1]), sum(d["similarity"] for d in x[1])),
        reverse=True
    )[:top_n]

    results = []
    for topic, docs in sorted_topics:
        info    = TOPIC_SEEDS[topic]
        best    = max(docs, key=lambda d: d["similarity"])
        avg_sim = sum(d["similarity"] for d in docs) / len(docs)
        relevance = round(min(avg_sim + len(docs) * 0.03, 0.97), 2)

        evidence = []
        for d in docs[:2]:
            snip = d["content"][:150].strip()
            if snip:
                evidence.append(snip)

        desc = _get_trend_description(topic, best["content"][:200])

        results.append({
            "trend_name":        topic,
            "description":       desc,
            "category":          info["category"],
            "relevance_score":   relevance,
            "time_horizon":      info["horizon"],
            "nvidia_opportunity": NVIDIA_OPPORTUNITIES.get(topic, "Monitor and invest proactively."),
            "evidence":          evidence,
            "doc_count":         len(docs),
        })

    print(f"[trend_detector] Found {len(results)} trends")
    return results


def get_keyword_frequency(top_n_keywords=20):
    docs = retrieve_multi(TREND_QUERIES, n_per_query=3)
    stop = {"the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","is","are",
            "was","were","be","been","has","have","had","will","would","can","could","may","might","that",
            "this","these","those","its","it","as","also","said","new","more","than","into","about",
            "after","over","nvidia","their","they","its","which","who","has","have"}
    counter = Counter()
    for doc in docs:
        words = doc["content"].lower().split()
        counter.update([w.strip(".,;:!?\"'()[]") for w in words if len(w) > 3 and w not in stop])
    return counter.most_common(top_n_keywords)


def _fallback_trends():
    return [{"trend_name":"No data available","description":"Run scrapers and embeddings first.","category":"N/A","relevance_score":0.0,"time_horizon":"N/A","nvidia_opportunity":"Collect data first.","evidence":[],"doc_count":0}]


if __name__ == "__main__":
    print(json.dumps(detect_trends(), indent=2))
