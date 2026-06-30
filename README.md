# NVIDIA Strategic Intelligence Agent

An AI-powered executive intelligence system that continuously collects live information about NVIDIA, indexes it in a vector database, surfaces strategic insights through an executive dashboard, and pushes briefings to Slack and Google Sheets.

**Stack:** Python · LangGraph · ChromaDB · sentence-transformers · Hugging Face Inference API · Streamlit · Slack · Google Sheets

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                         │
│  nvidia_scraper.py   news_scraper.py   community_scraper.py     │
│  (Newsroom/IR/Blog)  (RSS/GNews feeds) (Hacker News Algolia)    │
└──────────────────────────────┬──────────────────────────────────┘
                               │  raw JSON (data/raw/)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Embedding & Indexing Layer                      │
│  embed_documents.py                                             │
│  • Clean & deduplicate → all-MiniLM-L6-v2 embeddings           │
│  • Store in ChromaDB (persistent, metadata filtering)           │
└──────────────────────────────┬──────────────────────────────────┘
                               │  ChromaDB (data/chroma_db/)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               Intelligence & RAG Layer                           │
│  retriever.py          → Hybrid semantic + keyword search       │
│  opportunity_detector  → Keyword-signal opportunity scoring     │
│  risk_detector         → Negative-signal risk scoring           │
│  trend_detector        → Emerging theme frequency analysis      │
│  llm_engine.py         → HF Inference API (Mistral-7B-Instruct) │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Agentic Orchestration Layer (LangGraph)             │
│  orchestrator.py  — StateGraph DAG pipeline                     │
│  ScraperNode → IndexerNode → [conditional] → AnalystNode →     │
│  CEONode → END                                                  │
│                                                                 │
│  strategic_agent.py — Explicit reasoning loop                   │
│  Goal → Plan → Retrieve → Analyze → Decide → Recommend →       │
│  Validate                                                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Presentation Layer                             │
│  dashboard/app.py  — Streamlit Executive Dashboard              │
│  Sections: Overview · Market Intelligence · Opportunities ·     │
│  Risk Monitor · Sentiment · Recommendations · CEO Briefing ·    │
│  Ask the Agent                                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Integration Layer                              │
│  slack_notifier.py  — CEO Briefing + top signals → Slack        │
│  sheets_logger.py   — One row per run → Google Sheets tracker   │
└─────────────────────────────────────────────────────────────────┘
```

---

## LangGraph Pipeline

The pipeline is orchestrated as a **stateful directed acyclic graph (DAG)** using LangGraph. Every node reads from and writes to a shared `PipelineState` TypedDict — no side-channel data passing.

```
START
  │
  ▼
ScraperNode ──── collects live data from 3 sources
  │
  ▼
IndexerNode ──── embeds + stores in ChromaDB
  │
  ▼ (conditional edge)
should_continue? ── "abort" ──► END  (if 0 documents indexed)
  │ "continue"
  ▼
AnalystNode ──── detects opportunities, risks, trends (deterministic)
  │
  ▼
CEONode ──── generates recommendations + executive briefing (HF LLM)
  │
  ▼
END
```

**Why LangGraph over AutoGen / CrewAI:**
- Models the pipeline as an explicit stateful graph (nodes + edges)
- `PipelineState` flows through every node — shared, inspectable state
- Conditional edges handle failure gracefully (no documents → abort)
- Industry-standard: used in production at LinkedIn, Replit, Uber

---

## StrategicAgent — Explicit Reasoning Loop

The `StrategicAgent` class implements a true agentic workflow — not just RAG:

```
Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate
```

Each step is logged to a trace, rendered live in the dashboard's "Ask the Agent" tab. The VALIDATE step retries up to 2 times if the LLM is unavailable, then falls back to a deterministic recommendation.

**Why this is an agent, not a pipeline:**
A simple RAG pipeline does: prompt → retrieve → LLM → response.  
The `StrategicAgent` adds: goal decomposition, sub-query planning, cross-signal analysis, autonomous decision-making, output validation, and retry logic.

### Out-of-scope / ungrounded query guardrails

Before any LLM call, both "Ask the Agent" paths (`StrategicAgent.run()` and `answer_strategic_question()` in `rag/llm_engine.py`) reject queries the system isn't equipped to answer. Two layers, cheapest first:

1. **Keyword scope filter** (`StrategicAgent._is_in_scope()` only) — fast, no retrieval needed. Checks the query against `IN_SCOPE_TERMS`, merged from this project's own detector keyword lists.
2. **Retrieval-grounding check** (`grounding_rejection_message()` in `rag/llm_engine.py`, both paths) — checks the top retrieved document's cosine similarity against `GROUNDING_THRESHOLD = 0.5`, chosen empirically: on-topic NVIDIA queries scored 0.67–0.79, off-topic queries scored 0.18–0.39.

---

## Data Flow

```
Live sources → Scrapers → data/raw/*.json
                              │
                         embed_documents.py
                         (clean → embed → deduplicate)
                              │
                         ChromaDB vector store
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
        opportunity_     risk_detector  trend_detector
        detector.py      (keyword       (frequency
        (signal          scoring)       analysis)
        scoring)
               │              │              │
               └──────────────┴──────────────┘
                              │
                        ceo_agent.py
                   (deterministic structured fields)
                   (HF Inference API → CEO briefing)
                              │
                       dashboard/app.py
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        slack_notifier.py           sheets_logger.py
        (Slack channel post)        (Google Sheets row)
```

---

## Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Orchestration | LangGraph `StateGraph` | Stateful DAG, conditional routing, industry-standard |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Free, local, 90 MB, strong semantic search |
| Vector DB | ChromaDB | Built-in persistence, metadata filtering, no server required |
| LLM | Hugging Face Inference API (Mistral-7B-Instruct-v0.2) | Free tier, no local GPU needed, open-source |
| Fast LLM | Phi-3-mini-4k-instruct (HF) | Low latency for interactive Q&A and StrategicAgent |
| Search | Hybrid (cosine similarity + keyword boost) | Handles proper nouns (NVIDIA, Blackwell, CUDA) better than pure semantic |
| Scrapers | requests + BeautifulSoup + feedparser + HN Algolia API | No paid credentials required |
| Dashboard | Streamlit + Plotly | Rapid prototyping, reactive UI |
| Notifications | Slack Incoming Webhooks | Free, zero-credential push to team channel |
| History | Google Sheets API + gspread | Free, persistent run history, trend tracking |

---

## AI Pipeline Design Decisions

### LLM Scope (Key Design Principle)

The LLM is called in **three places only** — all for free-text prose generation:

| Location | Model | Purpose |
|----------|-------|---------|
| `ceo_agent.py` → `generate_ceo_briefing()` | Mistral-7B | Executive briefing |
| `llm_engine.py` → `answer_strategic_question()` | Phi-3-mini | Dashboard Q&A |
| `strategic_agent.py` → `StrategicAgent.run()` | Phi-3-mini | Agent recommendations |

All structured outputs (opportunities, risks, trends, priorities, confidence scores) are computed **deterministically** via keyword/signal counting. Small instruction-tuned models produce unreliable JSON — deterministic computation is auditable, reproducible, and never hallucinates structured fields.

### Hugging Face Inference API vs. Ollama
| | Hugging Face API | Ollama (previous) |
|--|--|--|
| Setup | Set 1 env variable | Install app, pull model (~5 GB) |
| GPU required | No (cloud inference) | Yes (local) |
| Cost | Free tier (~1000 req/day) | Free but needs hardware |
| Models | Mistral-7B, Phi-3, Llama3 | Same models locally |
| Internet | Required | Not required after pull |

### Hybrid Search
Retriever combines:
1. **Semantic search** — cosine similarity via ChromaDB + `all-MiniLM-L6-v2`
2. **Keyword boosting** — exact-match on query tokens, especially proper nouns

This prevents pure semantic search from missing queries like "Blackwell H100 supply" where token overlap matters.

### ChromaDB over FAISS
- Persistent by default (no manual save/load)
- Metadata filtering with `$eq` syntax
- Human-readable SQLite backend for debugging

### Sentiment Analysis
Keyword counting over a domain-tailored positive/negative word list (`POSITIVE_WORDS`/`NEGATIVE_WORDS` in `dashboard/app.py`). Chosen for **transparency and auditability** — captures finance/tech jargon (e.g. "overweight", "tariff", "antitrust") that general-purpose sentiment tools miss.

---

## Setup & Installation

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv

.venv\Scripts\Activate.ps1


pip install -r requirements.txt
```

**Windows users:** use `run.ps1` / `dashboard.ps1` instead — they call `.venv\Scripts\python.exe` directly, so they always use the right environment regardless of activation state.

### 2. Set Hugging Face API token (free)
Get your token at: https://huggingface.co/settings/tokens

```bash
cp .env.example .env
# edit .env and set HF_API_TOKEN=hf_xxxxxxxxxxxx
```

### 3. Run the full pipeline
```powershell

python main.py --all
```

### 4. Launch the dashboard
```powershell
streamlit run dashboard/app.py
```

---

## Integration Setup (Optional)

### Slack
After every pipeline run, posts the CEO Briefing + top opportunities/risks to a Slack channel.

1. Go to https://api.slack.com/apps → Create App → Incoming Webhooks → Add to Workspace
2. Copy the webhook URL
3. Add to `.env`:
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
```

### Google Sheets
Appends one row per pipeline run — builds a historical trend tracker over time.

1. Enable Google Sheets API + Google Drive API at https://console.cloud.google.com
2. Create a Service Account → download JSON key → save as `credentials/google_credentials.json`
3. Share your Google Sheet with the service account's `client_email` as Editor
4. Add to `.env`:
```
GOOGLE_SHEET_ID=your_sheet_id_here
GOOGLE_CREDENTIALS_PATH=credentials/google_credentials.json
```

---

## CLI Reference

```bash
python main.py --all                # full pipeline: scrape → embed → analyze → brief → push
python main.py --all --reset        # wipe ChromaDB first, then full pipeline
python main.py --no-collect         # skip scraping, use cached raw data
python main.py --analyze            # analysis + briefing only (data already indexed)
python main.py --graph              # print LangGraph node/edge structure
python main.py --status             # show system status (DB + HF token + integrations)
python main.py --no-integrations    # skip Slack + Sheets after pipeline
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_API_TOKEN` | *(required)* | Hugging Face API token |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Primary model for CEO briefing |
| `HF_FAST_MODEL` | `microsoft/Phi-3-mini-4k-instruct` | Fast model for interactive Q&A and StrategicAgent |
| `HF_PROVIDER` | `featherless-ai` | HF Inference Providers backend — pinned because both models are served exclusively by featherless-ai |
| `SLACK_WEBHOOK_URL` | *(optional)* | Slack incoming webhook URL |
| `GOOGLE_SHEET_ID` | *(optional)* | Google Sheets document ID |
| `GOOGLE_CREDENTIALS_PATH` | `credentials/google_credentials.json` | GCP service account key |

---

## Data Sources

| Source | Type | File |
|--------|------|------|
| NVIDIA Newsroom | Official | `nvidia_official.json` |
| NVIDIA Investor Relations | Official | `nvidia_official.json` |
| NVIDIA Blog | Official | `nvidia_official.json` |
| NVIDIA Developer Blog | Official | `nvidia_official.json` |
| Google News RSS (NVIDIA) | News | `news_articles.json` |
| Financial news RSS | News | `news_articles.json` |
| Hacker News Algolia search | Community | `community_posts.json` |
| Hacker News top stories | Community | `community_posts.json` |

Minimum collected: **100 documents** across **3 independent sources**.

---

## Project Structure

```
nvidia-strategic-intelligence/
├── main.py                         # CLI entry point + integration trigger
├── run.ps1                         # Windows: runs main.py through .venv
├── dashboard.ps1                   # Windows: runs the dashboard through .venv
├── requirements.txt
├── README.md
├── .env.example                    # Template — copy to .env and fill in
├── scrapers/
│   ├── nvidia_scraper.py           # NVIDIA official sources
│   ├── news_scraper.py             # RSS news feeds
│   └── community_scraper.py        # Hacker News (Algolia API)
├── embeddings/
│   └── embed_documents.py          # Clean → embed → store in ChromaDB
├── vector_db/
│   └── chroma_manager.py           # ChromaDB CRUD + hybrid search
├── rag/
│   ├── retriever.py                # Hybrid semantic + keyword retrieval
│   └── llm_engine.py               # HF Inference API + CEO briefing + Q&A
├── intelligence/
│   ├── opportunity_detector.py     # Signal-based opportunity scoring (deterministic)
│   ├── risk_detector.py            # Negative-signal risk scoring (deterministic)
│   └── trend_detector.py           # Keyword frequency trend detection (deterministic)
├── agents/
│   ├── graph_state.py              # PipelineState TypedDict
│   ├── nodes.py                    # LangGraph node adapters
│   ├── scraper_agent.py            # Live data collection
│   ├── indexer_agent.py            # Embedding + ChromaDB storage
│   ├── analyst_agent.py            # Opportunity/risk/trend detection
│   ├── ceo_agent.py                # Recommendations + executive briefing
│   ├── orchestrator.py             # StateGraph DAG: build_graph(), run_pipeline()
│   └── strategic_agent.py          # Goal→Plan→Retrieve→Analyze→Decide→Recommend→Validate
├── integrations/
│   ├── slack_notifier.py           # Post briefing to Slack channel
│   └── sheets_logger.py            # Append run history to Google Sheets
├── dashboard/
│   └── app.py                      # Streamlit executive dashboard
├── tests/
│   └── test_hallucination_grounding.py  # Asserts structured fields are grounded in source docs
├── test_pipeline.py                # Smoke test: embed → ChromaDB → retrieval
├── credentials/                    # GCP service account key (not committed)
└── data/
    ├── raw/                        # Scraped JSON files
    ├── cleaned/                    # Deduplicated + cleaned docs
    ├── analysis_results.json       # Latest analysis output
    └── chroma_db/                  # Persistent vector store (not committed)
```