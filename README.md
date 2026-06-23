# NVIDIA Strategic Intelligence Agent
### AI-Powered Executive Decision Support System
**Course:** Natural Language Processing — Final Examination Project  
**Supervisor:** Prof. Swati Chandna  
**Student:** Shreya Shetty  
**Submission Date:** 22 June 2026

---

## Project Overview

This project implements an end-to-end AI Strategic Intelligence Agent for NVIDIA Corporation. The system autonomously collects live information from multiple public sources, processes it through a Retrieval-Augmented Generation (RAG) pipeline, and delivers executive-level strategic insights via an interactive dashboard.

The system answers the central question posed by the project brief:

> *"If you were the CEO today, what would you do next — and why?"*

Every recommendation is grounded in retrieved evidence. No output is fabricated.

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the local LLM server (in a separate terminal)
ollama serve

# 3. Run the complete pipeline
python main.py --all

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

**To override the default LLM without editing code:**
```powershell
$env:OLLAMA_MODEL="qwen2.5:3b"   # PowerShell
set OLLAMA_MODEL=qwen2.5:3b       # Windows CMD
```

---

## Step-by-Step Pipeline Commands

| Command | Purpose |
|---------|---------|
| `python main.py --status` | Check system health: document count, Ollama connectivity |
| `python main.py --collect` | Scrape all 3 source categories into `data/raw/` |
| `python main.py --embed` | Clean, deduplicate, embed, and index into ChromaDB |
| `python main.py --analyze` | Run strategic analysis (requires Ollama running) |
| `python main.py --all` | Execute complete pipeline in sequence |
| `python main.py --all --reset` | Wipe ChromaDB and rebuild from scratch |
| `streamlit run dashboard/app.py` | Launch the Executive Intelligence Dashboard |

---

## System Architecture

The system is organised into four layers, each with a single responsibility:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                        │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ nvidia_scraper  │  │  news_scraper    │  │  community_   │  │
│  │                 │  │                  │  │  scraper      │  │
│  │ • Press releases│  │ • Google News    │  │               │  │
│  │ • NVIDIA blog   │  │ • Yahoo Finance  │  │ • Hacker News │  │
│  │ • Newsroom RSS  │  │ • TechCrunch     │  │ • HN Algolia  │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬───────┘  │
│           └───────────────────┼────────────────────┘           │
│                               ▼                                 │
│                      data/raw/*.json                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PROCESSING & INDEXING LAYER                    │
│                                                                 │
│  embed_documents.py                                             │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │  Load JSON  │──▶│  Clean & Dedup   │──▶│  Embed (SBERT)  │  │
│  └─────────────┘   └──────────────────┘   └────────┬────────┘  │
│                         MD5 hashing                 │           │
│                         removes duplicates          ▼           │
│                                           ┌─────────────────┐  │
│                                           │   ChromaDB      │  │
│                                           │  (cosine index) │  │
│                                           └────────┬────────┘  │
└────────────────────────────────────────────────────┼───────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INTELLIGENCE LAYER (RAG)                      │
│                                                                 │
│  retriever.py                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Query → Embed query → Semantic Search → Hybrid Rerank  │   │
│  └───────────────────────────┬─────────────────────────────┘   │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐             │
│          ▼                   ▼                   ▼             │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐     │
│  │ opportunity_ │  │  risk_detector   │  │trend_detector│     │
│  │  detector    │  │                  │  │              │     │
│  └──────┬───────┘  └────────┬─────────┘  └──────┬───────┘     │
│         └──────────────────┼────────────────────┘             │
│                             ▼                                   │
│                     ceo_agent.py (Ollama LLM)                  │
│                     • Strategic Recommendations                 │
│                     • CEO Briefing (free-text)                  │
│                     • Interactive Q&A                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│                                                                 │
│  streamlit dashboard/app.py  (8 interactive sections)          │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ Company  │ │  Market    │ │Opportunity│ │ Risk Monitor   │  │
│  │ Overview │ │Intelligence│ │ Monitor  │ │                │  │
│  ├──────────┤ ├────────────┤ ├──────────┤ ├────────────────┤  │
│  │Sentiment │ │ Strategic  │ │   CEO    │ │  Ask CEO Agent │  │
│  │ Analysis │ │   Recs     │ │ Briefing │ │   (Q&A RAG)    │  │
│  └──────────┘ └────────────┘ └──────────┘ └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
Internet Sources
      │
      ├── NVIDIA RSS / Newsroom ──────────────┐
      ├── Google News / Yahoo Finance RSS ─────┤──▶ Raw JSON (data/raw/)
      └── Hacker News Public API ─────────────┘
                                                        │
                                              Clean + Deduplicate
                                              (MD5 content hashing)
                                                        │
                                              Sentence Transformer
                                              (all-MiniLM-L6-v2)
                                              384-dim dense vectors
                                                        │
                                              ChromaDB (cosine similarity)
                                                        │
                            ┌───────────────────────────┤
                            │                           │
                    Strategic Query              User Query (Q&A)
                            │                           │
                    Semantic + Hybrid             Semantic Search
                       Retrieval                        │
                            │                           │
                    Evidence Pool               Context Documents
                            │                           │
                            └───────────┬───────────────┘
                                        │
                                 Ollama LLM
                               (qwen2.5:3b, Q4_K_M)
                                        │
                              ┌─────────┴────────┐
                              │                  │
                     Structured Output       CEO Briefing
                     (Opps/Risks/Recs)       (Free text)
                              │
                       Streamlit Dashboard
```

---

## Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Data Collection | `feedparser`, `requests`, `BeautifulSoup4` | Lightweight, credential-free, RSS-native |
| Community Data | Hacker News Firebase + Algolia API | Fully open, no OAuth, high signal-to-noise ratio for tech discussions |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` | 90 MB, CPU-friendly, strong MTEB benchmark scores for semantic similarity |
| Vector Database | ChromaDB (persistent mode) | Automatic disk persistence, cosine similarity, metadata filtering — no manual serialisation |
| LLM Inference | Ollama + `qwen2.5:3b` (Q4_K_M GGUF) | 100% local, no API key, runs on consumer hardware with 4-bit quantisation |
| Dashboard | Streamlit + Plotly | Python-native, zero frontend code, interactive charts out of the box |
| Language | Python 3.10+ | Full NLP/ML ecosystem compatibility |

---

## Project Structure

```
nvidia-strategic-intelligence/
│
├── data/
│   ├── raw/                    # Raw scraped JSON — one file per source category
│   ├── cleaned/                # Deduplicated, normalised documents
│   └── chroma_db/              # Persistent ChromaDB vector store (on-disk)
│
├── scrapers/
│   ├── nvidia_scraper.py       # NVIDIA press releases, blog, newsroom via RSS
│   ├── news_scraper.py         # Google News, Yahoo Finance, TechCrunch via RSS
│   └── community_scraper.py    # Hacker News discussions via public API (no credentials)
│
├── embeddings/
│   └── embed_documents.py      # Load → clean → MD5 dedup → embed → store in ChromaDB
│
├── vector_db/
│   └── chroma_manager.py       # ChromaDB interface: insert, semantic search, hybrid search, stats
│
├── intelligence/
│   ├── opportunity_detector.py  # ChromaDB retrieval + keyword scoring → structured opportunities
│   ├── risk_detector.py         # ChromaDB retrieval + keyword scoring → structured risks
│   └── trend_detector.py        # Topic clustering + frequency analysis → emerging trends
│
├── rag/
│   ├── retriever.py             # Semantic search + keyword reranking (hybrid retrieval)
│   └── ceo_agent.py             # Ollama LLM: recommendations, CEO briefing, strategic Q&A
│
├── dashboard/
│   └── app.py                   # Streamlit executive dashboard (8 sections)
│
├── main.py                      # CLI orchestrator: --collect / --embed / --analyze / --all
├── test_pipeline.py             # End-to-end smoke test (no internet or Ollama required)
├── requirements.txt
└── README.md
```

---

## AI Pipeline — Technical Detail

### 1. Retrieval-Augmented Generation (RAG)

RAG grounds every LLM output in retrieved evidence, preventing hallucination. The pipeline:

1. Each scraped document is embedded into a 384-dimensional dense vector using `all-MiniLM-L6-v2`
2. Vectors are stored in ChromaDB with cosine similarity indexing
3. At inference time, a strategic query is embedded using the same model
4. The top-k most similar documents are retrieved from ChromaDB
5. Retrieved documents are injected into the LLM prompt as context
6. The LLM reasons over the evidence and produces a grounded output

### 2. Hybrid Search

Pure semantic search can retrieve topically similar but imprecise documents. Hybrid search combines two signals:

```
final_score = cosine_similarity + 0.1 × keyword_hit_count
```

- **Semantic component:** captures conceptual relevance (e.g. "GPU shortage" matches "chip supply constraint")
- **Keyword component:** boosts documents containing exact query terms, improving precision
- This approach is known in the literature as *dense + sparse hybrid retrieval*

### 3. Structured Intelligence (Without LLM JSON)

Opportunities, risks, and trends are computed deterministically from retrieval signals — not generated by the LLM. This design choice was made because small local models (3B parameters) are unreliable for structured JSON output. Instead:

- **Severity / impact level** is inferred from cosine similarity score and keyword hit count
- **Category** is inferred by matching content against category-specific keyword sets
- **Confidence score** is derived from similarity + keyword density
- **Evidence** is extracted as the highest-signal sentences from retrieved documents

The LLM is called only for free-text outputs (CEO Briefing, strategic Q&A) where structured formatting is not required.

### 4. Sentiment Analysis

Lexicon-based scoring — no external API or model required:

```python
score = (positive_hits - negative_hits) / (positive_hits + negative_hits)
# score ∈ [-1.0, +1.0]
# > +0.2  → Bullish
# < -0.2  → Bearish
# otherwise → Neutral
```

Matching uses substring search (not exact word match) to capture morphological variants — e.g. `"accelerat"` matches `"accelerating"`, `"accelerated"`, `"acceleration"`.

---

## Key Design Decisions

**Why ChromaDB over FAISS?**  
ChromaDB persists to disk automatically, supports metadata filtering (by source, date, source type), and requires no manual index serialisation. FAISS is faster at scale but requires manual save/load and has no native metadata support — a significant drawback for a multi-source system where filtering by source type is essential.

**Why `all-MiniLM-L6-v2` over `bge-base-en-v1.5`?**  
`all-MiniLM-L6-v2` is approximately 90 MB and runs fast on CPU, making it viable on any development machine without a GPU. `bge-base-en-v1.5` scores higher on MTEB but is 3× larger and significantly slower on CPU. The embedding model is configurable in `retriever.py` and `embed_documents.py` for future upgrades.

**Why Ollama over Hugging Face `transformers` directly?**  
Ollama manages model loading, 4-bit GGUF quantisation, and inference batching — reducing peak VRAM from ~6 GB (full precision) to ~2 GB (Q4_K_M). It exposes a simple HTTP API that decouples the dashboard from the inference backend, allowing model swaps without code changes.

**Why MD5 deduplication?**  
The same article frequently appears across multiple RSS feeds (e.g. a NVIDIA press release indexed by both Google News and Yahoo Finance). MD5 hashing of document content provides a fast, deterministic duplicate fingerprint. Documents with identical hashes are dropped before embedding, preventing retrieval bias toward over-represented content.

**Why Hacker News instead of Reddit?**  
Reddit's API returned HTTP 403 errors without OAuth credentials during development. Hacker News provides two fully open endpoints — the Firebase real-time API and the Algolia search API — with no authentication, no rate limits, and high signal quality for technology discussions. This makes the system reproducible out-of-the-box for any evaluator.

**Why deterministic scoring for structured outputs instead of LLM JSON?**  
During development, `qwen2.5:3b` produced malformed or inconsistent JSON in approximately 40% of structured output calls under time pressure. Falling back to deterministic keyword and similarity scoring produces 100% reliable structured outputs while reserving the LLM for tasks where free-text generation adds genuine value (briefings, Q&A).

---