"""
app.py — NVIDIA Strategic Intelligence Dashboard
Executive-grade redesign: white-ground financial report aesthetic,
DM Sans + DM Mono typography, NVIDIA green used sparingly as live signal.
"""

import sys, os, json
from datetime import datetime

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from vector_db.chroma_manager import get_stats
from rag.retriever import retrieve_multi
from rag.ceo_agent import (
    generate_recommendations, generate_ceo_briefing,
    answer_strategic_question, check_ollama_status,
)
from intelligence.opportunity_detector import detect_opportunities
from intelligence.risk_detector import detect_risks
from intelligence.trend_detector import detect_trends, get_keyword_frequency

st.set_page_config(
    page_title="NVIDIA Intelligence",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset ── */
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #FFFFFF !important; color: #0F1923; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F1923 !important;
    border-right: none !important;
    width: 240px !important;
}
[data-testid="stSidebar"] * { color: #8A9BB0 !important; }
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    color: #8A9BB0 !important;
    padding: 9px 16px !important;
    border-radius: 4px !important;
    display: block;
    cursor: pointer;
    transition: all 0.15s;
    border: none !important;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #FFFFFF !important; background: #1C2B3A !important; }
[data-testid="stSidebar"] .stRadio label[data-baseweb] { background: transparent; }

/* Active nav item */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio input:checked ~ label {
    color: #FFFFFF !important;
    background: #1C2B3A !important;
    border-left: 2px solid #76B900 !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #F4F6F8 !important;
    border: 1px solid #E8ECF0 !important;
    border-radius: 4px !important;
    padding: 20px 24px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #6B7C93 !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #0F1923 !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; color: #76B900 !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #F4F6F8 !important;
    border: 1px solid #E8ECF0 !important;
    border-radius: 4px !important;
    margin-bottom: 6px !important;
    box-shadow: none !important;
}
[data-testid="stExpander"]:hover { border-color: #76B900 !important; }
.streamlit-expanderHeader {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #0F1923 !important;
    padding: 14px 18px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #E8ECF0 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6B7C93 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    color: #0F1923 !important;
    border-bottom: 2px solid #76B900 !important;
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    font-weight: 500 !important;
    background: #0F1923 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 10px 20px !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #1C2B3A !important; }

/* ── Text inputs ── */
.stTextArea textarea, .stSelectbox > div > div {
    background: #F4F6F8 !important;
    border: 1px solid #E8ECF0 !important;
    border-radius: 4px !important;
    color: #0F1923 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Progress ── */
.stProgress > div > div > div { background: #76B900 !important; }
.stProgress > div > div { background: #E8ECF0 !important; border-radius: 2px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #E8ECF0 !important;
    border-radius: 4px !important;
}
[data-testid="stDataFrame"] * {
    color: #1A1F2E !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #E8ECF0 !important; margin: 28px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #F4F6F8; }
::-webkit-scrollbar-thumb { background: #CBD5E0; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #76B900; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #76B900 !important; }

/* ── Alert/info boxes ── */
.stAlert { border-radius: 4px !important; border-left-width: 3px !important; }
</style>
""", unsafe_allow_html=True)


POSITIVE_WORDS = {
    "growth","record","beat","strong","partner","launch","expand","profit",
    "gain","rise","innovation","leader","success","surge","opportunity",
    "invest","breakthrough","demand","revenue","win","supercomputer","ai",
    "accelerat","dominat","ahead","alliance","factory","agent","boost",
    "record-breaking","adoption","milestone","exceed","outperform","bull",
    "rally","upgrade","buy","overweight","positive","soar","jump","climb",
    "advance","pioneer","cutting-edge","next-gen","deploy","scale","power",
}

NEGATIVE_WORDS = {
    "decline","risk","threat","ban","fall","loss","concern","warn",
    "shortage","lawsuit","fine","restrict","competition","drop","fear",
    "negative","down","miss","weak","cut","reduce","penalty","dive",
    "crash","bear","sell","underweight","downgrade","short","bubble",
    "overvalued","sanction","export","control","tariff","probe","antitrust",
    "delay","cancel","slow","disappoint","miss","below","underperform",
}

def simple_sentiment(text):
    """
    Score sentiment by checking if any positive/negative keyword
    appears as a substring in any word — catches plurals, conjugations,
    and compound words (e.g. 'accelerating' matches 'accelerat').
    Returns a float in [-1, +1].
    """
    words = text.lower().split()
    pos = sum(1 for w in words if any(p in w for p in POSITIVE_WORDS))
    neg = sum(1 for w in words if any(n in w for n in NEGATIVE_WORDS))
    total = pos + neg
    return round((pos - neg) / total, 3) if total > 0 else 0.0

def sentiment_label(score):
    if score > 0.2:  return "Bullish"
    if score < -0.2: return "Bearish"
    return "Neutral"

def flatten_results(data):
    if not data: return []
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data[0]
    return data

def eyebrow(text):
    st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;letter-spacing:0.14em;text-transform:uppercase;color:#76B900;margin:0 0 6px 0;font-weight:500;">{text}</p>', unsafe_allow_html=True)

def page_title(text):
    st.markdown(f'<h1 style="font-family:\'DM Sans\',sans-serif;font-size:1.9rem;font-weight:700;color:#0F1923;margin:0 0 4px 0;letter-spacing:-0.02em;">{text}</h1>', unsafe_allow_html=True)

def subtitle(text):
    st.markdown(f'<p style="font-family:\'DM Sans\',sans-serif;font-size:0.9rem;color:#0F1923;margin:0 0 28px 0;">{text}</p>', unsafe_allow_html=True)

def severity_color(s):
    return {"Critical":"#C41E3A","High":"#DC3545","Medium":"#E67E22","Low":"#27AE60"}.get(s,"#6B7C93")

def impact_color(s):
    return {"High":"#C41E3A","Medium":"#E67E22","Low":"#27AE60"}.get(s,"#6B7C93")

def pill(label, color, bg):
    return f'<span style="font-family:\'DM Mono\',monospace;font-size:0.65rem;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:{color};background:{bg};padding:3px 9px;border-radius:2px;">{label}</span>'

CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#0F1923", size=11),
    xaxis=dict(gridcolor="#E8ECF0", linecolor="#E8ECF0", tickfont=dict(color="#6B7C93")),
    yaxis=dict(gridcolor="#E8ECF0", linecolor="#E8ECF0", tickfont=dict(color="#6B7C93")),
    margin=dict(l=8, r=8, t=36, b=8),
    title_font=dict(color="#0F1923", size=13, family="DM Sans"),
)

# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=1800)
def cached_opportunities():   return detect_opportunities(top_n=5)
@st.cache_data(show_spinner=False, ttl=1800)
def cached_risks():            return detect_risks(top_n=5)
@st.cache_data(show_spinner=False, ttl=1800)
def cached_trends():           return detect_trends(top_n=6)
@st.cache_data(show_spinner=False, ttl=1800)
def cached_recommendations():  return generate_recommendations(top_n=5)
@st.cache_data(show_spinner=False, ttl=1800)
def cached_market_docs():
    return retrieve_multi(["NVIDIA recent news announcement","NVIDIA competitor AMD Intel activity","NVIDIA AI technology product"], n_per_query=8)
@st.cache_data(show_spinner=False, ttl=1800)
def cached_sentiment_docs():
    base = cached_market_docs()
    extra = retrieve_multi(["NVIDIA stock investor sentiment community opinion"], n_per_query=15)
    seen = {d["id"] for d in base}
    return base + [d for d in extra if d["id"] not in seen]
@st.cache_data(show_spinner=False, ttl=1800)
def cached_keyword_freq():     return get_keyword_frequency(top_n_keywords=20)
@st.cache_data(show_spinner=False, ttl=3600)
def cached_briefing(oj, rj, tj, cj):
    return generate_ceo_briefing(opportunities=json.loads(oj), risks=json.loads(rj), trends=json.loads(tj), recommendations=json.loads(cj))

# Warm the cache on startup so first page load is instant
if "warmed" not in st.session_state:
    cached_market_docs()
    cached_sentiment_docs()
    cached_opportunities()
    cached_risks()
    cached_trends()
    st.session_state["warmed"] = True

# ── Cache warmup on first load ────────────────────────────────────────────────
if "warmed" not in st.session_state:
    with st.spinner("Initialising intelligence systems…"):
        cached_market_docs()
        cached_opportunities()
        cached_risks()
        cached_trends()
        cached_recommendations()
        cached_keyword_freq()
        cached_sentiment_docs()
    st.session_state["warmed"] = True
# ── Sidebar ───────────────────────────────────────────────────────────────────
ollama = check_ollama_status()
stats  = get_stats()

with st.sidebar:
    # Logo + live pulse signature element
    st.markdown("""
    <div style="padding:24px 20px 20px 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg"
             width="90" style="filter:brightness(0) saturate(100%) invert(52%) sepia(99%) saturate(500%) hue-rotate(50deg);display:block;margin-bottom:16px;" />
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.16em;
                    color:#3D5166;text-transform:uppercase;margin-bottom:20px;">
            Strategic Intelligence
        </div>
        <!-- Live signal pulse — the signature element -->
        <div style="position:relative;height:24px;margin-bottom:24px;overflow:hidden;">
            <svg viewBox="0 0 200 24" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;">
                <polyline points="0,12 30,12 40,4 50,20 60,12 80,12 90,2 100,22 110,12 140,12 150,6 160,18 170,12 200,12"
                    fill="none" stroke="#76B900" stroke-width="1.5" opacity="0.7"/>
                <circle cx="200" cy="12" r="3" fill="#76B900">
                    <animate attributeName="opacity" values="1;0.2;1" dur="1.8s" repeatCount="indefinite"/>
                </circle>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status block
    dot = "🟢" if ollama["running"] else "🔴"
    st.markdown(f"""
    <div style="margin:0 12px 20px 12px;padding:12px 14px;background:#1C2B3A;border-radius:4px;">
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;
                    color:#3D5166;text-transform:uppercase;margin-bottom:8px;">System</div>
        <div style="font-size:0.78rem;color:#CBD5E0;margin-bottom:3px;">{dot} {ollama['configured_model'] if ollama['running'] else 'LLM Offline'}</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#3D5166;">
            {stats['total_documents']:,} docs &nbsp;·&nbsp; {stats['num_sources']} sources
        </div>
    </div>
    """, unsafe_allow_html=True)

    section = st.radio("", [
        "Overview",
        "Market Intelligence",
        "Opportunities",
        "Risk Monitor",
        "Sentiment",
        "Recommendations",
        "CEO Briefing",
        "Ask the Agent",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("↺  Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"""
    <div style="position:fixed;bottom:20px;left:0;width:240px;text-align:center;
                font-family:'DM Mono',monospace;font-size:0.58rem;color:#3D5166;">
        {datetime.now().strftime('%d %b %Y  %H:%M')}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Overview
# ══════════════════════════════════════════════════════════════════
if section == "Overview":
    eyebrow("Intelligence Platform · Live")
    page_title("NVIDIA Corporation")
    subtitle(f"Strategic intelligence feed — updated every 30 minutes from {stats['num_sources']} live sources")

    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents indexed", f"{stats['total_documents']:,}")
    c2.metric("Active sources",    stats['num_sources'])
    c3.metric("Industry",          "Semiconductors")
    c4.metric("LLM engine",        ollama['configured_model'] if ollama['running'] else "Offline")

    st.markdown("<hr>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        eyebrow("Data Sources")
        if stats["unique_sources"]:
            src_map = {
                "nvidia_blog":"NVIDIA Blog","nvidia_press_releases":"NVIDIA Newsroom",
                "nvidia_newsroom":"NVIDIA Newsroom","google_news_nvidia":"Google News · NVIDIA",
                "google_news_nvidia_tech":"Google News · Tech","yahoo_finance_nvda":"Yahoo Finance · NVDA",
                "seeking_alpha_nvda":"Seeking Alpha · NVDA","google_news_nvidia_competitors":"Google News · Competitors",
                "google_news_ai_market":"Google News · AI Market","techcrunch_ai":"TechCrunch · AI",
                "venturebeat_ai":"VentureBeat · AI",
            }
            type_map = {
                "nvidia_blog":"Official","nvidia_press_releases":"Official","nvidia_newsroom":"Official",
                "google_news_nvidia":"News","google_news_nvidia_tech":"News","yahoo_finance_nvda":"Financial",
                "seeking_alpha_nvda":"Financial","google_news_nvidia_competitors":"Competitive",
                "google_news_ai_market":"Market","techcrunch_ai":"Tech Media","venturebeat_ai":"Tech Media",
            }
            rows = ""
            for i, src in enumerate(stats["unique_sources"]):
                display = src_map.get(src, src.replace("_"," ").title())
                src_type = type_map.get(src, "Other")
                type_colors = {"Official":("#76B900","#0A1F06"),"News":("#3B82F6","#0A1929"),"Financial":("#8B5CF6","#1A0A29"),"Competitive":("#EF4444","#290A0A"),"Market":("#F59E0B","#291A0A"),"Tech Media":("#06B6D4","#001A1F"),"Other":("#6B7C93","#1A1F24")}
                tc, bc = type_colors.get(src_type, ("#6B7C93","#1A1F24"))
                bg = "#FFFFFF" if i % 2 == 0 else "#F9FAFB"
                rows += f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                            padding:11px 18px;background:{bg};border-bottom:1px solid #E8ECF0;">
                    <span style="font-size:0.84rem;color:#0F1923;font-weight:500;">{display}</span>
                    <span style="font-family:'DM Mono',monospace;font-size:0.62rem;font-weight:500;
                                 letter-spacing:0.08em;text-transform:uppercase;color:{tc};
                                 background:{bc};padding:3px 8px;border-radius:2px;">{src_type}</span>
                </div>"""
            st.markdown(f'<div style="border:1px solid #E8ECF0;border-radius:4px;overflow:hidden;">{rows}</div>', unsafe_allow_html=True)
        else:
            st.info("No sources indexed. Run `python main.py --collect` then `python main.py --embed`.")

    with col_r:
        eyebrow("Pipeline Status")
        items = [
            ("LLM Engine", ollama["configured_model"] if ollama["running"] else "Offline", ollama["running"]),
            ("Vector Store", "ChromaDB · Persistent", True),
            ("Embedding Model", "all-MiniLM-L6-v2", True),
            ("Retrieval", "Semantic + Keyword Hybrid", True),
            ("Sentiment", "Lexicon · Real-time", True),
        ]
        for label, value, ok in items:
            dot_c = "#76B900" if ok else "#C41E3A"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:13px 16px;border:1px solid #E8ECF0;border-radius:4px;
                        margin-bottom:5px;background:#FAFAFA;">
                <span style="font-size:0.82rem;color:#6B7C93;">{label}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:{dot_c};font-weight:500;">● {value}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Market Intelligence
# ══════════════════════════════════════════════════════════════════
elif section == "Market Intelligence":
    eyebrow("Live Feed")
    page_title("Market Intelligence")
    subtitle(f"Real-time news, competitor signals, and technology keywords from {stats['num_sources']} monitored sources")
    with st.spinner("Loading…"):
        market_docs = cached_market_docs()
        kw_freq     = cached_keyword_freq()

    by_type = {}
    for doc in market_docs:
        k = doc["metadata"].get("source_type","other")
        by_type.setdefault(k,[]).append(doc)

    tabs = st.tabs(["Recent News", "Company Announcements", "Competitor Activity", "Keyword Signals"])

    with tabs[0]:
        news = by_type.get("financial_news",[]) + by_type.get("tech_news",[]) + by_type.get("market_news",[])
        if not news: news = market_docs
        for doc in news[:10]:
            title = doc["metadata"].get("title","Article")[:100]
            src   = doc["metadata"].get("source","").replace("_"," ").title()
            date  = doc["metadata"].get("published","")[:10]
            url   = doc["metadata"].get("url","")
            sim   = doc.get("similarity",0)
            sent  = simple_sentiment(doc["content"])
            sc, sl = ("#27AE60","Bullish") if sent>0.2 else ("#C41E3A","Bearish") if sent<-0.2 else ("#E67E22","Neutral")

            with st.expander(f"**{title}**"):
                col1, col2 = st.columns([3,1])
                with col1:
                    st.markdown(f'<p style="font-size:0.87rem;color:#4A5568;line-height:1.75;">{doc["content"][:500]}</p>', unsafe_allow_html=True)
                    if url:
                        st.markdown(f'<a href="{url}" target="_blank" style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#76B900;text-decoration:none;letter-spacing:0.04em;">→ READ FULL ARTICLE</a>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div style="background:#F4F6F8;border:1px solid #E8ECF0;border-radius:4px;padding:14px;">
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:3px;">Source</div>
                        <div style="font-size:0.78rem;color:#0F1923;font-weight:500;margin-bottom:10px;">{src}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:3px;">Published</div>
                        <div style="font-size:0.78rem;color:#0F1923;margin-bottom:10px;">{date}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:3px;">Relevance</div>
                        <div style="font-size:0.78rem;color:#76B900;font-weight:600;margin-bottom:10px;">{sim:.0%}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:3px;">Signal</div>
                        <div style="font-size:0.78rem;color:{sc};font-weight:600;">{sl}</div>
                    </div>""", unsafe_allow_html=True)

    with tabs[1]:
        press = by_type.get("press_release",[]) + by_type.get("blog",[])
        if not press: press = [d for d in market_docs if "nvidia" in d["metadata"].get("source","").lower()]
        for doc in (press or market_docs)[:8]:
            with st.expander(f"**{doc['metadata'].get('title','')[:100]}**"):
                st.markdown(f'<p style="font-size:0.87rem;color:#4A5568;line-height:1.75;">{doc["content"][:500]}</p>', unsafe_allow_html=True)
                st.caption(f"{doc['metadata'].get('source','?').replace('_',' ').title()}  ·  {doc['metadata'].get('published','')[:10]}")

    with tabs[2]:
        comp = by_type.get("competitor_news",[])
        if not comp: comp = [d for d in market_docs if any(k in d["content"].lower() for k in ["amd","intel","qualcomm","google","amazon","microsoft"])]
        for doc in (comp or [])[:8]:
            with st.expander(f"**{doc['metadata'].get('title','')[:100]}**"):
                st.markdown(f'<p style="font-size:0.87rem;color:#4A5568;line-height:1.75;">{doc["content"][:500]}</p>', unsafe_allow_html=True)
                st.caption(f"{doc['metadata'].get('source','?').replace('_',' ').title()}  ·  {doc['metadata'].get('published','')[:10]}")
        if not comp:
            st.info("No competitor signals in current dataset. Collect fresh data to update.")

    with tabs[3]:
        if kw_freq:
            words, counts = zip(*kw_freq)
            fig = go.Figure(go.Bar(
                x=list(counts), y=list(words), orientation="h",
                marker=dict(color=["#76B900" if c == max(counts) else "#CBD5E0" for c in counts], line=dict(width=0)),
                hovertemplate="%{y}: %{x} mentions<extra></extra>",
            ))
            fig.update_layout(**CHART, title="Most frequent terms across all sources", height=460)
            fig.update_yaxes(tickfont=dict(family="DM Mono", size=10, color="#6B7C93"))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Opportunities
# ══════════════════════════════════════════════════════════════════
elif section == "Opportunities":
    eyebrow("AI Analysis")
    page_title("Opportunity Monitor")
    subtitle("Strategic growth signals extracted from live intelligence")

    with st.spinner("Analysing…"):
        opportunities = flatten_results(cached_opportunities())

    if not opportunities or not isinstance(opportunities[0], dict) or opportunities[0].get("title") == "No data available":
        st.warning("No opportunities detected. Run `python main.py --collect` then `--embed`.")
    else:
        high = sum(1 for o in opportunities if o.get("impact_level")=="High")
        med  = sum(1 for o in opportunities if o.get("impact_level")=="Medium")
        low  = sum(1 for o in opportunities if o.get("impact_level")=="Low")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total identified", len(opportunities))
        c2.metric("High impact",  high)
        c3.metric("Medium impact",med)
        c4.metric("Low impact",   low)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Confidence chart
        opp_df = pd.DataFrame([{
            "Title":      o.get("title","")[:52] + ("…" if len(o.get("title",""))>52 else ""),
            "Confidence": float(o.get("confidence_score",0.5)),
            "Impact":     o.get("impact_level","Medium"),
        } for o in opportunities])

        color_seq = {"High":"#C41E3A","Medium":"#E67E22","Low":"#27AE60"}
        fig = go.Figure()
        for level, color in color_seq.items():
            sub = opp_df[opp_df["Impact"]==level]
            if not sub.empty:
                fig.add_trace(go.Bar(x=sub["Confidence"], y=sub["Title"], orientation="h",
                    name=level, marker_color=color, marker_line_width=0,
                    hovertemplate="%{y}<br>Confidence: %{x:.0%}<extra></extra>"))
        fig.update_layout(**CHART, title="Opportunity confidence scores", height=240,
                          barmode="stack", legend=dict(orientation="h", y=1.18, font=dict(family="DM Sans",size=11,color="#6B7C93")))
        fig.update_xaxes(tickformat=".0%", range=[0,1])
        st.plotly_chart(fig, use_container_width=True)

        eyebrow("Opportunity Details")
        for i, opp in enumerate(opportunities, 1):
            impact = opp.get("impact_level","Medium")
            conf   = float(opp.get("confidence_score",0.5))
            cat    = opp.get("category","")
            ic     = impact_color(impact)
            ibg    = {"High":"#FFF0F0","Medium":"#FFF8F0","Low":"#F0FFF4"}.get(impact,"#F4F6F8")

            with st.expander(f"**{opp.get('title','Opportunity')[:85]}**"):
                col_l, col_r = st.columns([3,1])
                with col_l:
                    st.markdown(f'<p style="font-size:0.87rem;color:#4A5568;line-height:1.75;margin-bottom:14px;">{opp.get("description","")}</p>', unsafe_allow_html=True)
                    evidence = opp.get("evidence",[])
                    if evidence:
                        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:6px;">Supporting Evidence</p>', unsafe_allow_html=True)
                        for ev in evidence:
                            st.markdown(f'<div style="border-left:2px solid #76B900;padding:5px 12px;margin-bottom:4px;font-size:0.82rem;color:#4A5568;">{ev}</div>', unsafe_allow_html=True)
                with col_r:
                    st.markdown(f"""
                    <div style="background:#F4F6F8;border:1px solid #E8ECF0;border-radius:4px;padding:16px;">
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:5px;">Impact</div>
                        <div style="margin-bottom:12px;"><span style="font-family:'DM Mono',monospace;font-size:0.65rem;font-weight:500;text-transform:uppercase;color:{ic};background:{ibg};padding:3px 9px;border-radius:2px;">{impact}</span></div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:5px;">Category</div>
                        <div style="font-size:0.8rem;color:#0F1923;margin-bottom:12px;">{cat}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:5px;">Confidence</div>
                        <div style="font-size:1.3rem;font-weight:700;color:#0F1923;">{conf:.0%}</div>
                    </div>""", unsafe_allow_html=True)
                    st.progress(conf)


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Risk Monitor
# ══════════════════════════════════════════════════════════════════
elif section == "Risk Monitor":
    eyebrow("Threat Assessment")
    page_title("Risk Monitor")
    subtitle("Identified threats, vulnerabilities, and mitigation guidance")

    with st.spinner("Analysing…"):
        risks = flatten_results(cached_risks())

    if not risks or not isinstance(risks[0], dict) or risks[0].get("title") == "No data available":
        st.warning("No risks detected. Run the pipeline first.")
    else:
        sev_counts = {"Critical":0,"High":0,"Medium":0,"Low":0}
        for r in risks:
            sev_counts[r.get("severity_level","Medium")] = sev_counts.get(r.get("severity_level","Medium"),0)+1

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total risks",  len(risks))
        c2.metric("Critical",     sev_counts["Critical"])
        c3.metric("High",         sev_counts["High"])
        c4.metric("Medium",       sev_counts["Medium"])

        st.markdown("<hr>", unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])
        with col_l:
            risk_df = pd.DataFrame([{
                "Risk":       r.get("title","")[:52] + ("…" if len(r.get("title",""))>52 else ""),
                "Severity":   r.get("severity_level","Medium"),
                "Confidence": float(r.get("confidence_score",0.5)),
            } for r in risks])
            sev_colors = {"Critical":"#7F0819","High":"#C41E3A","Medium":"#E67E22","Low":"#27AE60"}
            fig = go.Figure()
            for sev, color in sev_colors.items():
                sub = risk_df[risk_df["Severity"]==sev]
                if not sub.empty:
                    fig.add_trace(go.Bar(x=sub["Confidence"], y=sub["Risk"], orientation="h",
                        name=sev, marker_color=color, marker_line_width=0,
                        hovertemplate="%{y}<br>%{x:.0%}<extra></extra>"))
            fig.update_layout(**CHART, title="Risk severity map", height=260,
                              barmode="stack", legend=dict(orientation="h", y=1.18, font=dict(family="DM Sans",size=11,color="#6B7C93")))
            fig.update_xaxes(tickformat=".0%", range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            cat_counts = {}
            for r in risks:
                c = r.get("category","Other")
                cat_counts[c] = cat_counts.get(c,0)+1
            fig2 = go.Figure(go.Pie(
                labels=list(cat_counts.keys()), values=list(cat_counts.values()), hole=0.6,
                marker=dict(colors=["#C41E3A","#E67E22","#3B82F6","#8B5CF6","#06B6D4","#76B900","#6B7C93"],
                            line=dict(color="#FFFFFF",width=2)),
                textfont=dict(size=11, family="DM Sans"),
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            fig2.update_layout(**CHART, title="Distribution by category", height=260,
                               legend=dict(font=dict(family="DM Sans",color="#6B7C93",size=10)))
            st.plotly_chart(fig2, use_container_width=True)

        eyebrow("Risk Details")
        for i, risk in enumerate(risks, 1):
            sev  = risk.get("severity_level","Medium")
            conf = float(risk.get("confidence_score",0.5))
            cat  = risk.get("category","")
            sc   = severity_color(sev)
            sbg  = {"Critical":"#FFF0F0","High":"#FFF0F0","Medium":"#FFF8F0","Low":"#F0FFF4"}.get(sev,"#F4F6F8")

            with st.expander(f"**{risk.get('title','Risk')[:85]}**"):
                col_l2, col_r2 = st.columns([3,1])
                with col_l2:
                    st.markdown(f'<p style="font-size:0.87rem;color:#4A5568;line-height:1.75;">{risk.get("description","")}</p>', unsafe_allow_html=True)
                    mit = risk.get("mitigation","")
                    if mit:
                        st.markdown(f'<div style="border-left:2px solid #76B900;padding:6px 14px;margin-top:10px;font-size:0.83rem;color:#2D6A4F;background:#F0FFF4;border-radius:0 3px 3px 0;"><strong>Mitigation:</strong> {mit}</div>', unsafe_allow_html=True)
                    evidence = risk.get("evidence",[])
                    if evidence:
                        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin:12px 0 6px 0;">Evidence</p>', unsafe_allow_html=True)
                        for ev in evidence:
                            st.markdown(f'<div style="border-left:2px solid #C41E3A;padding:5px 12px;margin-bottom:4px;font-size:0.82rem;color:#4A5568;">{ev}</div>', unsafe_allow_html=True)
                with col_r2:
                    st.markdown(f"""
                    <div style="background:#F4F6F8;border:1px solid #E8ECF0;border-radius:4px;padding:16px;">
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:5px;">Severity</div>
                        <div style="margin-bottom:12px;"><span style="font-family:'DM Mono',monospace;font-size:0.65rem;font-weight:500;text-transform:uppercase;color:{sc};background:{sbg};padding:3px 9px;border-radius:2px;">{sev}</span></div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:5px;">Category</div>
                        <div style="font-size:0.8rem;color:#0F1923;margin-bottom:12px;">{cat}</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:5px;">Confidence</div>
                        <div style="font-size:1.3rem;font-weight:700;color:#0F1923;">{conf:.0%}</div>
                    </div>""", unsafe_allow_html=True)
                    st.progress(conf)


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Sentiment
# ══════════════════════════════════════════════════════════════════
elif section == "Sentiment":
    eyebrow("Signal Analysis")
    page_title("Sentiment Monitor")
    subtitle("Aggregated market and public perception across all monitored sources")

    with st.spinner(""):
        all_docs = cached_sentiment_docs()

    if not all_docs:
        st.warning("No documents. Run the pipeline first.")
    else:
        records = []
        for doc in all_docs:
            score = simple_sentiment(doc["content"])
            src_type = doc["metadata"].get("source_type","other")
            records.append({
                "Title":       doc["metadata"].get("title","")[:60],
                "Source":      doc["metadata"].get("source","").replace("_"," ").title(),
                "Source Type": src_type,
                "Score":       score,
                "Signal":      sentiment_label(score),
                "Date":        doc["metadata"].get("published","")[:10],
            })
        df = pd.DataFrame(records)

        news_s = df[df["Source Type"].isin(["financial_news","tech_news","market_news","analyst_report","competitor_news"])]["Score"]
        comm_s = df[df["Source Type"].isin(["community","blog","press_release"])]["Score"]
        news_avg = float(news_s.mean()) if len(news_s)>0 else float(df["Score"].mean())
        comm_avg = float(comm_s.mean()) if len(comm_s)>0 else float(df["Score"].mean())
        overall  = float(df["Score"].mean())

        def sc(s): return "#27AE60" if s>0.2 else "#C41E3A" if s<-0.2 else "#E67E22"

        c1,c2,c3 = st.columns(3)
        for col, lbl, score in [(c1,"News sentiment",news_avg),(c2,"Public sentiment",comm_avg),(c3,"Overall",overall)]:
            sl = sentiment_label(score)
            col.markdown(f"""
            <div style="background:#F4F6F8;border:1px solid #E8ECF0;border-radius:4px;padding:22px 20px;text-align:center;">
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:#6B7C93;margin-bottom:10px;">{lbl}</div>
                <div style="font-size:1.7rem;font-weight:700;color:{sc(score)};letter-spacing:-0.01em;">{sl}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#6B7C93;margin-top:4px;">{score:+.3f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            fig1 = go.Figure(go.Histogram(
                x=df["Score"], nbinsx=14,
                marker=dict(color="#CBD5E0", line=dict(width=0)),
                hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra></extra>",
            ))
            fig1.add_vline(x=0.2, line_dash="dot", line_color="#27AE60", line_width=1)
            fig1.add_vline(x=-0.2, line_dash="dot", line_color="#C41E3A", line_width=1)
            fig1.add_vline(x=overall, line_dash="solid", line_color="#76B900", line_width=2, annotation_text="avg", annotation_position="top right")
            fig1.update_layout(**CHART, title="Score distribution", height=250)
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            lc = df["Signal"].value_counts()
            colors = {"Bullish":"#27AE60","Neutral":"#E67E22","Bearish":"#C41E3A"}
            fig2 = go.Figure(go.Pie(
                labels=lc.index.tolist(), values=lc.values.tolist(), hole=0.6,
                marker=dict(colors=[colors.get(l,"#CBD5E0") for l in lc.index], line=dict(color="#FFF",width=2)),
                textfont=dict(family="DM Sans"), hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            fig2.update_layout(**CHART, title="Signal breakdown", height=250,
                               legend=dict(font=dict(family="DM Sans",size=11,color="#6B7C93")))
            st.plotly_chart(fig2, use_container_width=True)

        eyebrow("Document-level signals")
        disp = df[["Title","Source","Score","Signal","Date"]].sort_values("Score")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — Recommendations
# ══════════════════════════════════════════════════════════════════
elif section == "Recommendations":
    eyebrow("CEO Advisory")
    page_title("Strategic Recommendations")
    subtitle("Evidence-based actions prioritised by the AI Strategic Advisor")

    with st.spinner("Generating recommendations…"):
        recs = flatten_results(cached_recommendations())

    if not recs:
        st.warning("No recommendations yet. Ensure Ollama is running and data is loaded.")
    else:
        high = sum(1 for r in recs if r.get("priority")=="High")
        med  = sum(1 for r in recs if r.get("priority")=="Medium")
        low  = sum(1 for r in recs if r.get("priority")=="Low")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total", len(recs))
        c2.metric("High priority",   high)
        c3.metric("Medium priority", med)
        c4.metric("Low priority",    low)

        st.markdown("<hr>", unsafe_allow_html=True)

        priority_order = {"High":0,"Medium":1,"Low":2}
        for i, rec in enumerate(sorted(recs, key=lambda r: priority_order.get(r.get("priority","Low"),2)), 1):
            priority   = rec.get("priority","Medium")
            risk_level = rec.get("risk_level","Medium")
            pc = impact_color(priority)
            pbg = {"High":"#FFF0F0","Medium":"#FFF8F0","Low":"#F0FFF4"}.get(priority,"#F4F6F8")

            with st.expander(f"**{rec.get('recommendation','Recommendation')[:88]}**"):
                col_l, col_r = st.columns([3,1])
                with col_l:
                    st.markdown(f'<p style="font-size:0.87rem;color:#4A5568;line-height:1.75;margin-bottom:14px;">{rec.get("rationale","")}</p>', unsafe_allow_html=True)
                    impact = rec.get("expected_impact",{})
                    if impact and isinstance(impact,dict):
                        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:6px;">Expected Impact</p>', unsafe_allow_html=True)
                        for k, v in impact.items():
                            st.markdown(f'<div style="border-left:2px solid #76B900;padding:4px 12px;margin-bottom:4px;font-size:0.82rem;color:#4A5568;"><strong style="color:#0F1923;">{k.replace("_"," ").title()}:</strong> {v}</div>', unsafe_allow_html=True)
                    evidence = rec.get("supporting_evidence",[])
                    if evidence:
                        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin:12px 0 6px 0;">Evidence</p>', unsafe_allow_html=True)
                        for ev in evidence:
                            st.markdown(f'<div style="border-left:2px solid #CBD5E0;padding:4px 12px;margin-bottom:4px;font-size:0.82rem;color:#4A5568;">{ev}</div>', unsafe_allow_html=True)
                with col_r:
                    rc = impact_color(risk_level)
                    rbg = {"High":"#FFF0F0","Medium":"#FFF8F0","Low":"#F0FFF4"}.get(risk_level,"#F4F6F8")
                    st.markdown(f"""
                    <div style="background:#F4F6F8;border:1px solid #E8ECF0;border-radius:4px;padding:16px;">
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:5px;">Priority</div>
                        <div style="margin-bottom:12px;"><span style="font-family:'DM Mono',monospace;font-size:0.65rem;font-weight:500;text-transform:uppercase;color:{pc};background:{pbg};padding:3px 9px;border-radius:2px;">{priority}</span></div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:5px;">Risk Level</div>
                        <div style="margin-bottom:12px;"><span style="font-family:'DM Mono',monospace;font-size:0.65rem;font-weight:500;text-transform:uppercase;color:{rc};background:{rbg};padding:3px 9px;border-radius:2px;">{risk_level}</span></div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:5px;">Risk Note</div>
                        <div style="font-size:0.78rem;color:#6B7C93;line-height:1.5;">{rec.get("risk_description","")[:110]}</div>
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — CEO Briefing
# ══════════════════════════════════════════════════════════════════
elif section == "CEO Briefing":
    eyebrow("Executive Summary · Confidential")
    page_title("CEO Intelligence Brief")
    subtitle(f"AI-synthesised executive summary  ·  {datetime.now().strftime('%d %B %Y')}")

    with st.spinner(""):
        opps   = flatten_results(cached_opportunities())
        risks  = flatten_results(cached_risks())
        trends = flatten_results(cached_trends())
        recs   = flatten_results(cached_recommendations())

    with st.spinner("Generating brief…"):
        briefing = cached_briefing(json.dumps(opps), json.dumps(risks), json.dumps(trends), json.dumps(recs))

    # Header bar
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:16px 24px;background:#0F1923;border-radius:4px;margin-bottom:20px;">
        <div>
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;
                        text-transform:uppercase;color:#3D5166;margin-bottom:3px;">Classified · CEO Eyes Only</div>
            <div style="font-size:1rem;font-weight:600;color:#FFFFFF;">NVIDIA Corporation — Strategic Intelligence Brief</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#3D5166;">
            {datetime.now().strftime('%d %b %Y  ·  %H:%M')}
        </div>
    </div>
    <div style="background:#FAFAFA;border:1px solid #E8ECF0;border-left:3px solid #76B900;
                border-radius:0 4px 4px 0;padding:32px 40px;font-size:0.93rem;
                line-height:1.95;color:#2D3748;font-family:'DM Sans',sans-serif;">
        {briefing.replace(chr(10), "<br>")}
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    for col, lbl, items, key, color in [
        (c1, "Top Opportunities", opps[:3], "title",      "#27AE60"),
        (c2, "Top Risks",         risks[:3], "title",     "#C41E3A"),
        (c3, "Emerging Trends",   trends[:3],"trend_name","#76B900"),
    ]:
        with col:
            eyebrow(lbl)
            for item in items:
                txt = item.get(key,"")[:65]
                col.markdown(f'<div style="padding:9px 0;border-bottom:1px solid #E8ECF0;font-size:0.83rem;color:#4A5568;"><span style="color:{color};margin-right:8px;font-weight:700;">›</span>{txt}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — Ask the Agent
# ══════════════════════════════════════════════════════════════════
elif section == "Ask the Agent":
    eyebrow("RAG Query Interface")
    page_title("Ask the AI CEO Agent")
    subtitle("Pose any strategic question and receive an evidence-backed answer from the RAG pipeline")

    st.markdown("<hr>", unsafe_allow_html=True)

    presets = [
        "What are NVIDIA's biggest opportunities in AI right now?",
        "What are the main risks facing NVIDIA from competitors?",
        "Which technologies should NVIDIA prioritise investing in?",
        "What is current investor sentiment toward NVIDIA?",
        "If you were NVIDIA's CEO today, what would you do next?",
    ]

    eyebrow("Preset questions")
    cols = st.columns(len(presets))
    for i, (col, q) in enumerate(zip(cols, presets)):
        if col.button(q[:28]+"…", key=f"p{i}"):
            st.session_state["agent_q"] = q

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    user_q = st.text_area(
        "Your question",
        value=st.session_state.get("agent_q",""),
        height=80,
        placeholder="e.g. What should NVIDIA do about the China export restrictions?",
    )

    if st.button("Submit to CEO Agent", use_container_width=False) and user_q.strip():
        with st.spinner("Retrieving evidence and reasoning…"):
            answer = answer_strategic_question(user_q.strip())

        eyebrow("Agent Response")
        st.markdown(f"""
        <div style="background:#FAFAFA;border:1px solid #E8ECF0;border-left:3px solid #76B900;
                    border-radius:0 4px 4px 0;padding:28px 36px;font-size:0.91rem;
                    line-height:1.9;color:#2D3748;font-family:'DM Sans',sans-serif;margin-top:12px;">
            {answer.replace(chr(10),"<br>")}
        </div>""", unsafe_allow_html=True)

        with st.expander("View retrieved evidence documents"):
            from rag.retriever import retrieve_multi as rm
            docs = rm([user_q], n_per_query=5)
            for doc in docs[:5]:
                st.markdown(f"""
                <div style="padding:16px 20px;border:1px solid #E8ECF0;border-radius:4px;margin-bottom:8px;background:#FAFAFA;">
                    <div style="font-size:0.85rem;font-weight:600;color:#0F1923;margin-bottom:6px;">{doc["metadata"].get("title","")[:85]}</div>
                    <div style="font-size:0.82rem;color:#6B7C93;line-height:1.6;">{doc["content"][:280]}</div>
                    <div style="margin-top:10px;font-family:'DM Mono',monospace;font-size:0.65rem;color:#6B7C93;">
                        {doc["metadata"].get("source","?").replace("_"," ").title()} &nbsp;·&nbsp; relevance: <span style="color:#76B900;font-weight:600;">{doc.get("similarity",0):.0%}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
