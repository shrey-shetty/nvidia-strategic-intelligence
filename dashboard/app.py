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
from rag.llm_engine import (
    generate_recommendations, generate_ceo_briefing,
    answer_strategic_question, check_hf_status,
)
from agents.strategic_agent import StrategicAgent
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
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #FFFFFF !important; background: #1C2B3A !important; }
[data-testid="stSidebar"] .stRadio label[data-baseweb] { background: transparent; }
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
[data-testid="stSidebar"] .stRadio div[role="radio"] {
    display: none !important;
}

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

/* ── Buttons — default dark style ── */
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

/* ── Preset question buttons — light chip style ──
   Streamlit 1.36 doesn't expose per-widget-key testids on buttons, so we
   can't target st.button(key="p0") directly. Instead we rely on structure:
   the presets are flat buttons in the main content area (not inside
   st.columns), while every other stMain button (Submit / Run Full Agent
   Loop) lives inside a stHorizontalBlock — so that's used to tell them
   apart, and to restore the dark style for the column-wrapped ones below. */
section.main .stButton > button {
    background: #F4F6F8 !important;
    color: #0F1923 !important;
    border: 1px solid #E8ECF0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    letter-spacing: 0 !important;
    text-align: left !important;
    padding: 5px 12px !important;
    border-radius: 4px !important;
    min-height: 0 !important;
    line-height: 1.3 !important;
}
section.main .stButton > button:hover {
    background: #FFFFFF !important;
    border-color: #76B900 !important;
    color: #76B900 !important;
}

/* Restore the dark default for buttons inside column layouts
   (Submit to CEO Agent / Run Full Agent Loop) */
section.main [data-testid="stHorizontalBlock"] .stButton > button {
    background: #0F1923 !important;
    color: #FFFFFF !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-align: center !important;
    padding: 10px 20px !important;
    border-radius: 3px !important;
}
section.main [data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #1C2B3A !important;
    border-color: transparent !important;
    color: #FFFFFF !important;
}

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
/* Streamlit's alert text color follows the OS prefers-color-scheme setting
   independently of this dashboard's forced-light page background. Under a
   dark-OS-preference browser it renders near-white text (confirmed via
   computed style: rgb(255,255,194)) on the still-pale-yellow background,
   making it unreadable. Pin both background and text explicitly so they
   never vary by OS theme — values match this dashboard's normal (light-OS)
   rendering, which is already correct and matches the rest of the light
   main-content theme. */
[data-testid="stAlertContentWarning"] {
    background: #FFF8DC !important;
}
[data-testid="stAlertContentWarning"],
[data-testid="stAlertContentWarning"] p {
    color: #926C05 !important;
}
[data-testid="stAlertContentInfo"] {
    background: #E7F3FE !important;
}
[data-testid="stAlertContentInfo"],
[data-testid="stAlertContentInfo"] p {
    color: #0F4C75 !important;
}
</style>
""", unsafe_allow_html=True)


POSITIVE_WORDS = {
    "growth","record","beat","strong","partner","launch","expand","profit",
    "gain","rise","innovation","leader","success","surge","opportunity",
    "invest","breakthrough","demand","revenue","win","supercomputer",
    "accelerat","dominat","ahead","alliance","factory","agent","boost",
    "record-breaking","adoption","milestone","exceed","outperform","bull",
    "rally","upgrade","buy","overweight","positive","soar","jump","climb",
    "advance","pioneer","cutting-edge","next-gen","deploy","scale","power",
}
# NOTE: bare "ai" was removed — as a 2-character substring it matched almost
# every NVIDIA document (since "AI" appears in nearly all of them) and
# systematically inflated Bullish scores regardless of actual sentiment.
# Confirmed via audit: 5/6 sampled News docs scored +1.000 Bullish purely
# from this one match, on headlines that were actually neutral or negative.

NEGATIVE_WORDS = {
    "decline","risk","threat","ban","fall","loss","concern","warn",
    "shortage","lawsuit","fine","restrict","competition","drop","fear",
    "negative","down","miss","weak","cut","reduce","penalty","dive",
    "crash","bear","sell","underweight","downgrade","short","bubble",
    "overvalued","sanction","export","control","tariff","probe","antitrust",
    "delay","cancel","slow","disappoint","miss","below","underperform",
    "investigation","headwind",
}
# Added "investigation" (regulatory-risk term not covered by probe/antitrust
# alone) and "headwind" (standard finance-negative term). Considered "trail"
# and "loser" (both observed scoring false-positive Bullish in the audit)
# but rejected them: "trail" collides with "trailing" (neutral finance
# language — trailing revenue, trailing P/E) and "loser" collides with
# "closer" — both would reintroduce the same substring false-positive bug
# just fixed for "ai".

def simple_sentiment(text):
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

NVIDIA_GREEN = "#76b900"
CHART_RED    = "#e74c3c"
CHART_GRAY   = "#888888"
CHART_TEXT   = "#0F1923"

CHART = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color=CHART_TEXT, size=11),
    xaxis=dict(gridcolor="#2a2a2a", linecolor="#2a2a2a", tickfont=dict(color=CHART_TEXT)),
    yaxis=dict(gridcolor="#2a2a2a", linecolor="#2a2a2a", tickfont=dict(color=CHART_TEXT)),
    margin=dict(l=8, r=8, t=40, b=8),
    title_font=dict(color=CHART_TEXT, size=17, family="DM Sans"),
)
PLOTLY_CONFIG = {"displaylogo": False}

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

# Warm the cache on startup
if "warmed" not in st.session_state:
    cached_market_docs()
    cached_sentiment_docs()
    cached_opportunities()
    cached_risks()
    cached_trends()
    st.session_state["warmed"] = True

# ── Sidebar ───────────────────────────────────────────────────────────────────
hf    = check_hf_status()
stats = get_stats()

with st.sidebar:
    st.markdown("""
    <div style="padding:24px 20px 20px 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg"
             width="90" style="filter:brightness(0) saturate(100%) invert(52%) sepia(99%) saturate(500%) hue-rotate(50deg);display:block;margin-bottom:16px;" />
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.16em;
                    color:#3D5166;text-transform:uppercase;margin-bottom:20px;">
            Strategic Intelligence
        </div>
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

    dot = "🟢" if hf["token_set"] else "🔴"
    st.markdown(f"""
    <div style="margin:0 12px 20px 12px;padding:12px 14px;background:#1C2B3A;border-radius:4px;">
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;
                    color:#3D5166;text-transform:uppercase;margin-bottom:8px;">System</div>
        <div style="font-size:0.78rem;color:#CBD5E0;margin-bottom:3px;">{dot} {hf['model'].split('/')[-1] if hf['token_set'] else 'No HF Token'}</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#3D5166;">
            {stats['total_documents']:,} docs &nbsp;·&nbsp; {stats['num_sources']} sources
        </div>
    </div>
    """, unsafe_allow_html=True)

    section = st.radio("Navigation", [
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
    if st.button("↺  Refresh data", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    if st.button("⚙  Run pipeline", width='stretch'):
        st.cache_data.clear()
        with st.spinner("Running LangGraph pipeline…"):
            from agents.orchestrator import run_pipeline
            run_pipeline(collect=True, reset_db=False, top_n=5)
        st.success("Pipeline complete!")
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
    subtitle(f"Strategic intelligence feed — run the pipeline to refresh · {stats['num_sources']} live sources")

    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, font_size in [
        (c1, "Documents Indexed", f"{stats['total_documents']:,}", "1.4rem"),
        (c2, "Active Sources",    str(stats['num_sources']), "1.4rem"),
        (c3, "Industry",          "Semiconductors", "1.05rem"),
        (c4, "LLM Engine",        hf['model'].split('/')[-1] if hf['token_set'] else "No Token", "1.05rem"),
    ]:
        col.markdown(f"""
        <div style="background:#F4F6F8;border:1px solid #E8ECF0;border-radius:4px;padding:20px 24px;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.12em;
                        text-transform:uppercase;color:#6B7C93;margin-bottom:8px;">{label}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:{font_size};font-weight:700;
                        color:#0F1923;word-break:break-word;line-height:1.2;">{value}</div>
        </div>""", unsafe_allow_html=True)

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
                "hacker_news_top":"Community","hacker_news":"Community",
            }
            rows = ""
            for i, src in enumerate(stats["unique_sources"]):
                display  = src_map.get(src, src.replace("_"," ").title())
                src_type = type_map.get(src, "Other")
                type_colors = {
                    "Official":   ("#76B900","#0A1F06"),
                    "News":       ("#3B82F6","#0A1929"),
                    "Financial":  ("#8B5CF6","#1A0A29"),
                    "Competitive":("#EF4444","#290A0A"),
                    "Market":     ("#F59E0B","#291A0A"),
                    "Tech Media": ("#06B6D4","#001A1F"),
                    "Community":  ("#10B981","#06271D"),
                    "Other":      ("#6B7C93","#1A1F24"),
                }
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
            st.info("No sources indexed. Run `python main.py --all`.")

    with col_r:
        if stats["unique_sources"]:
            src_counts = {}
            for src in stats["unique_sources"]:
                src_type = type_map.get(src, "Other")
                src_counts[src_type] = src_counts.get(src_type, 0) + 1

            fig_src = go.Figure(go.Pie(
                labels=list(src_counts.keys()),
                values=list(src_counts.values()),
                hole=0.55,
                marker=dict(
                    colors=["#76B900","#3B82F6","#8B5CF6","#EF4444","#F59E0B","#06B6D4","#10B981"],
                    line=dict(color="#1a1a1a", width=2),
                ),
                textfont=dict(family="DM Sans", color="#e0e0e0"),
                hovertemplate="%{label}: %{value} sources<extra></extra>",
            ))
            fig_src.update_layout(**CHART, title="Source distribution", height=260,
                annotations=[dict(text="Sources", showarrow=False, font=dict(size=13, color=CHART_TEXT, family="DM Sans"))])
            st.plotly_chart(fig_src, width='stretch', config=PLOTLY_CONFIG, theme=None)

        eyebrow("Pipeline Status")
        items = [
            ("Orchestration",   "LangGraph · DAG Pipeline",    True),
            ("LLM Engine",      hf["model"].split("/")[-1] if hf["token_set"] else "No Token", hf["token_set"]),
            ("Vector Store",    "ChromaDB · Persistent",       True),
            ("Embedding Model", "all-MiniLM-L6-v2",            True),
            ("Retrieval",       "Semantic + Keyword Hybrid",    True),
            ("Sentiment",       "Lexicon · Real-time",          True),
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

    by_type = {}
    for doc in market_docs:
        k = doc["metadata"].get("source_type","other")
        by_type.setdefault(k,[]).append(doc)

    tabs = st.tabs(["Recent News", "Company Announcements", "Competitor Activity"])

    with tabs[0]:
        news = by_type.get("financial_news",[]) + by_type.get("tech_news",[]) + by_type.get("market_news",[])
        if not news: news = market_docs
        for doc in news[:10]:
            title = doc["metadata"].get("title","Article")[:100].strip()
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
            with st.expander(f"**{doc['metadata'].get('title','')[:100].strip()}**"):
                st.markdown(f'<p style="font-size:0.87rem;color:#4A5568;line-height:1.75;">{doc["content"][:500]}</p>', unsafe_allow_html=True)
                st.caption(f"{doc['metadata'].get('source','?').replace('_',' ').title()}  ·  {doc['metadata'].get('published','')[:10]}")
                url = doc["metadata"].get("url","")
                if url:
                    st.markdown(f'<a href="{url}" target="_blank" style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#76B900;text-decoration:none;letter-spacing:0.04em;">→ READ FULL ARTICLE</a>', unsafe_allow_html=True)

    with tabs[2]:
        comp = by_type.get("competitor_news",[])
        if not comp: comp = [d for d in market_docs if any(k in d["content"].lower() for k in ["amd","intel","qualcomm","google","amazon","microsoft"])]
        for doc in (comp or [])[:8]:
            with st.expander(f"**{doc['metadata'].get('title','')[:100].strip()}**"):
                st.markdown(f'<p style="font-size:0.87rem;color:#4A5568;line-height:1.75;">{doc["content"][:500]}</p>', unsafe_allow_html=True)
                st.caption(f"{doc['metadata'].get('source','?').replace('_',' ').title()}  ·  {doc['metadata'].get('published','')[:10]}")
                url = doc["metadata"].get("url","")
                if url:
                    st.markdown(f'<a href="{url}" target="_blank" style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#76B900;text-decoration:none;letter-spacing:0.04em;">→ READ FULL ARTICLE</a>', unsafe_allow_html=True)
        if not comp:
            st.info("No competitor signals in current dataset. Collect fresh data to update.")


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
        st.warning("No opportunities detected. Run `python main.py --all`.")
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

        opp_df = pd.DataFrame([{
            "Title":      o.get("title","")[:52] + ("…" if len(o.get("title",""))>52 else ""),
            "Confidence": float(o.get("confidence_score",0.5)),
            "Impact":     o.get("impact_level","Medium"),
        } for o in opportunities]).sort_values("Confidence")

        fig = go.Figure(go.Bar(
            x=opp_df["Confidence"], y=opp_df["Title"], orientation="h",
            marker_color=NVIDIA_GREEN, marker_line_width=0,
            customdata=opp_df["Impact"],
            hovertemplate="%{y}<br>Confidence: %{x:.0%}<br>Impact: %{customdata}<extra></extra>",
        ))
        fig.update_layout(**CHART, title="Opportunity confidence scores", height=240)
        fig.update_xaxes(tickformat=".0%", range=[0,1])
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG, theme=None)

        eyebrow("Opportunity Details")
        for i, opp in enumerate(opportunities, 1):
            impact  = opp.get("impact_level","Medium")
            conf    = float(opp.get("confidence_score",0.5))
            cat     = opp.get("category","")
            ic      = impact_color(impact)
            ibg     = {"High":"#FFF0F0","Medium":"#FFF8F0","Low":"#F0FFF4"}.get(impact,"#F4F6F8")
            opp_url = opp.get("url","")

            with st.expander(f"**{opp.get('title','Opportunity')[:85].strip()}**"):
                col_l, col_r = st.columns([3,1])
                with col_l:
                    st.markdown(f'<p style="font-size:0.87rem;color:#4A5568;line-height:1.75;margin-bottom:14px;">{opp.get("description","")}</p>', unsafe_allow_html=True)
                    evidence = opp.get("evidence",[])
                    if evidence:
                        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;color:#6B7C93;margin-bottom:6px;">Supporting Evidence</p>', unsafe_allow_html=True)
                        for ev in evidence:
                            ev_html = f'<a href="{opp_url}" target="_blank" style="color:#76B900;">{ev}</a>' if opp_url else ev
                            st.markdown(f'<div style="border-left:2px solid #76B900;padding:5px 12px;margin-bottom:4px;font-size:0.82rem;color:#4A5568;">{ev_html}</div>', unsafe_allow_html=True)
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

        risk_df = pd.DataFrame([{
            "Risk":       r.get("title","")[:52] + ("…" if len(r.get("title",""))>52 else ""),
            "Severity":   r.get("severity_level","Medium"),
            "Confidence": float(r.get("confidence_score",0.5)),
        } for r in risks]).sort_values("Confidence")

        fig = go.Figure(go.Bar(
            x=risk_df["Confidence"], y=risk_df["Risk"], orientation="h",
            marker_color=CHART_RED, marker_line_width=0,
            customdata=risk_df["Severity"],
            hovertemplate="%{y}<br>Confidence: %{x:.0%}<br>Severity: %{customdata}<extra></extra>",
        ))
        fig.update_layout(**CHART, title="Risk severity map", height=260)
        fig.update_xaxes(tickformat=".0%", range=[0,1])
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG, theme=None)

        eyebrow("Risk Details")
        for i, risk in enumerate(risks, 1):
            sev      = risk.get("severity_level","Medium")
            conf     = float(risk.get("confidence_score",0.5))
            cat      = risk.get("category","")
            sc       = severity_color(sev)
            sbg      = {"Critical":"#FFF0F0","High":"#FFF0F0","Medium":"#FFF8F0","Low":"#F0FFF4"}.get(sev,"#F4F6F8")
            risk_url = risk.get("url","")

            with st.expander(f"**{risk.get('title','Risk')[:85].strip()}**"):
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
                            ev_html = f'<a href="{risk_url}" target="_blank" style="color:#E74C3C;text-decoration:none;">{ev}</a>' if risk_url else ev
                            st.markdown(f'<div style="border-left:2px solid #E74C3C;padding:8px 12px;margin-bottom:6px;font-size:0.82rem;color:#6B7C93;">{ev_html}</div>', unsafe_allow_html=True)
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
            score    = simple_sentiment(doc["content"])
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

        news_s   = df[df["Source Type"].isin(["financial_news","tech_news","market_news","analyst_report","competitor_news"])]["Score"]
        comm_s   = df[df["Source Type"].isin(["community","blog","press_release"])]["Score"]
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

        lc     = df["Signal"].value_counts()
        colors = {"Bullish": NVIDIA_GREEN, "Neutral": CHART_GRAY, "Bearish": CHART_RED}
        fig2   = go.Figure(go.Pie(
            labels=lc.index.tolist(), values=lc.values.tolist(), hole=0.6,
            marker=dict(colors=[colors.get(l, CHART_GRAY) for l in lc.index], line=dict(color="#1a1a1a", width=2)),
            textfont=dict(family="DM Sans", color="#e0e0e0"),
            hovertemplate="%{label}: %{value}<extra></extra>",
        ))
        fig2.update_layout(**CHART, title="Signal breakdown", height=320,
                           legend=dict(font=dict(family="DM Sans", size=11, color=CHART_TEXT)),
                           annotations=[dict(text=sentiment_label(overall), showarrow=False, font=dict(size=18, color=CHART_TEXT, family="DM Sans"))])
        st.plotly_chart(fig2, width='stretch', config=PLOTLY_CONFIG, theme=None)

        eyebrow("Document-level signals")
        disp = df[["Title","Source","Score","Signal","Date"]].sort_values("Score")
        st.dataframe(disp, width='stretch', hide_index=True)


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
        st.warning("No recommendations yet. Ensure HF_API_TOKEN is set and data is loaded.")
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
            pc  = impact_color(priority)
            pbg = {"High":"#FFF0F0","Medium":"#FFF8F0","Low":"#F0FFF4"}.get(priority,"#F4F6F8")

            with st.expander(f"**{rec.get('recommendation','Recommendation')[:88].strip()}**"):
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
                    rc  = impact_color(risk_level)
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

    if trends and isinstance(trends[0], dict) and trends[0].get("trend_name") != "No data available":
        eyebrow("Trend Momentum")
        trend_df = pd.DataFrame([{
            "Trend":     t.get("trend_name","")[:28],
            "Relevance": float(t.get("relevance_score",0)),
        } for t in trends]).sort_values("Relevance", ascending=False)
        fig_trend = go.Figure(go.Scatter(
            x=trend_df["Trend"], y=trend_df["Relevance"],
            mode="lines+markers", line=dict(color=NVIDIA_GREEN, width=2),
            marker=dict(color=NVIDIA_GREEN, size=6),
            fill="tozeroy", fillcolor="rgba(118,185,0,0.25)",
            hovertemplate="%{x}<br>Relevance: %{y:.0%}<extra></extra>",
        ))
        fig_trend.update_layout(**CHART, title="Emerging trend momentum", height=260)
        fig_trend.update_yaxes(tickformat=".0%", range=[0,1])
        st.plotly_chart(fig_trend, width='stretch', config=PLOTLY_CONFIG, theme=None)

    st.markdown("<hr>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    for col, lbl, items, key, color in [
        (c1, "Top Opportunities", opps[:3],   "title",      "#27AE60"),
        (c2, "Top Risks",         risks[:3],  "title",      "#C41E3A"),
        (c3, "Emerging Trends",   trends[:3], "trend_name", "#76B900"),
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

    trace_colors = {
        "GOAL":      "#76B900",
        "PLAN":      "#3498DB",
        "RETRIEVE":  "#9B59B6",
        "ANALYZE":   "#E67E22",
        "DECIDE":    "#E74C3C",
        "RECOMMEND": "#27AE60",
        "VALIDATE":  "#1ABC9C",
    }

    st.markdown("<hr>", unsafe_allow_html=True)

    presets = [
        "What are NVIDIA's biggest opportunities in AI right now?",
        "What are the main risks facing NVIDIA from competitors?",
        "Which technologies should NVIDIA prioritise investing in?",
        "What is current investor sentiment toward NVIDIA?",
        "If you were NVIDIA's CEO today, what would you do next?",
    ]

    eyebrow("Preset questions")
    for i, q in enumerate(presets):
        if st.button(q, key=f"p{i}", width='stretch'):
            st.session_state["agent_q"] = q

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    user_q = st.text_area(
        "Your question",
        value=st.session_state.get("agent_q",""),
        height=80,
        placeholder="e.g. What should NVIDIA do about the China export restrictions?",
    )

    button_col1, button_col2 = st.columns([1, 1])
    submit_clicked = button_col1.button(
        "Submit to CEO Agent",
        width='content',
        help="Quick RAG answer: retrieves evidence for the question only and "
             "answers it directly with the fast model.",
    )
    loop_clicked = button_col2.button(
        "Run Full Agent Loop",
        width='content',
        help="Full Goal→Plan→Retrieve→Analyze→Decide→Recommend→Validate loop: also "
             "factors in NVIDIA's broader opportunities/risks/trends, "
             "so the answer can differ from 'Submit to CEO Agent'.",
    )

    if submit_clicked and user_q.strip():
        try:
            with st.spinner("Retrieving evidence and reasoning…"):
                answer = answer_strategic_question(user_q.strip())
        except ValueError as e:
            st.warning(str(e))
        else:
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

    if loop_clicked and user_q.strip():
        try:
            with st.spinner("Running full strategic agent loop…"):
                result = StrategicAgent(goal=user_q.strip()).run()
        except ValueError as e:
            st.warning(str(e))
        else:
            eyebrow("Full Agent Recommendation")
            st.markdown(f"""
            <div style="background:#FAFAFA;border:1px solid #E8ECF0;border-left:3px solid #76B900;
                        border-radius:0 4px 4px 0;padding:28px 36px;font-size:0.91rem;
                        line-height:1.9;color:#2D3748;font-family:'DM Sans',sans-serif;margin-top:12px;">
                {result["recommendation"].replace(chr(10),"<br>")}
            </div>""", unsafe_allow_html=True)

            eyebrow("Agent Trace")
            for step in result["trace"]:
                step_name    = step.get("step","STEP")
                border_color = trace_colors.get(step_name,"#CBD5E0")
                st.markdown(f"""
                <div style="background:#FAFAFA;border:1px solid #E8ECF0;border-left:4px solid {border_color};
                            border-radius:0 4px 4px 0;padding:18px 20px;margin-top:10px;">
                    <div style="font-family:'DM Mono',monospace;font-size:0.66rem;letter-spacing:0.12em;
                                text-transform:uppercase;color:{border_color};margin-bottom:8px;">{step_name}</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;line-height:1.75;color:#2D3748;">
                        {step.get("content","").replace(chr(10),"<br>")}
                    </div>
                </div>""", unsafe_allow_html=True)