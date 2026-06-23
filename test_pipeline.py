"""
test_pipeline.py  —  Smoke test for embed + ChromaDB + retrieval pipeline.
Run from inside nvidia-strategic-intelligence/:
    python test_pipeline.py
Does NOT require internet or Ollama — just sentence-transformers + chromadb.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("NVIDIA Strategic Intelligence — Pipeline Smoke Test")
print("=" * 55)

# ── 1. Mock documents (simulating scraped data) ───────────
mock_docs = [
    {
        "id": "doc001",
        "title": "NVIDIA Reports Record Revenue on AI Chip Demand",
        "content": "NVIDIA reported record quarterly revenue driven by surging demand for AI chips. Data center revenue grew 427% year-over-year. CEO Jensen Huang called it the beginning of a new industrial revolution.",
        "source": "nvidia_press_releases", "source_type": "press_release",
        "published": "2025-05-01", "url": "", "company": "NVIDIA", "collected_at": "2025-05-01",
    },
    {
        "id": "doc002",
        "title": "AMD Launches MI300X to Challenge NVIDIA H100",
        "content": "AMD unveiled its MI300X GPU targeting data center AI workloads, competing directly with NVIDIA's H100. Analysts warn NVIDIA may face margin pressure from intensifying GPU competition.",
        "source": "google_news_nvidia_competitors", "source_type": "competitor_news",
        "published": "2025-04-15", "url": "", "company": "NVIDIA", "collected_at": "2025-04-15",
    },
    {
        "id": "doc003",
        "title": "US Export Controls Tighten on Advanced Chips to China",
        "content": "The US government expanded export restrictions on advanced AI chips including NVIDIA A100 and H100 to China, impacting a significant portion of NVIDIA's addressable market.",
        "source": "google_news_nvidia", "source_type": "financial_news",
        "published": "2025-04-20", "url": "", "company": "NVIDIA", "collected_at": "2025-04-20",
    },
    {
        "id": "doc004",
        "title": "NVIDIA Partners with Major Cloud Providers on Blackwell GPU",
        "content": "NVIDIA announced expanded partnerships with AWS, Google Cloud, and Microsoft Azure to deploy its next-generation Blackwell architecture GPUs for enterprise AI inference.",
        "source": "nvidia_blog", "source_type": "press_release",
        "published": "2025-05-05", "url": "", "company": "NVIDIA", "collected_at": "2025-05-05",
    },
    {
        "id": "doc005",
        "title": "Reddit Discussion: NVDA Stock — Is the AI Bubble Real?",
        "content": "r/investing thread: NVIDIA's valuation is stretched at 35x revenue but the AI infrastructure buildout is real. Supply chain risks from TSMC concentration are a concern. Long-term thesis remains intact.",
        "source": "reddit_r_investing", "source_type": "community",
        "published": "2025-04-28", "url": "", "company": "NVIDIA", "collected_at": "2025-04-28",
    },
    {
        "id": "doc006",
        "title": "NVIDIA Expands into Healthcare AI with Drug Discovery Platform",
        "content": "NVIDIA launched BioNeMo, a cloud platform for drug discovery using generative AI. Partnerships with AstraZeneca and Amgen signal a major push into the healthcare vertical.",
        "source": "nvidia_blog", "source_type": "blog",
        "published": "2025-05-03", "url": "", "company": "NVIDIA", "collected_at": "2025-05-03",
    },
]

# Save mock data to data/raw/
os.makedirs("data/raw", exist_ok=True)
with open("data/raw/mock_test.json", "w") as f:
    json.dump(mock_docs, f, indent=2)
print(f"\n✅ Step 1: Created {len(mock_docs)} mock documents in data/raw/mock_test.json")

# ── 2. Embed + store in ChromaDB ──────────────────────────
print("\n⏳ Step 2: Embedding documents (downloads model ~90 MB on first run)...")
from sentence_transformers import SentenceTransformer
from vector_db.chroma_manager import get_client, add_documents, get_stats, reset_collection

reset_collection()   # start fresh for this test

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d["content"] for d in mock_docs]
embeddings = model.encode(texts, normalize_embeddings=True).tolist()

client = get_client()
added = add_documents(mock_docs, embeddings, client=client)
print(f"✅ Step 2: {added} documents stored in ChromaDB")

# ── 3. Retrieval ──────────────────────────────────────────
print("\n⏳ Step 3: Testing semantic retrieval...")
from rag.retriever import retrieve, retrieve_multi

results = retrieve("NVIDIA AI chip growth opportunity", n_results=3, use_hybrid=True)
print(f"✅ Step 3: Retrieved {len(results)} docs for 'NVIDIA AI chip growth opportunity'")
for r in results:
    print(f"   [{r['similarity']:.3f}] {r['metadata']['title'][:70]}")

# ── 4. Stats ──────────────────────────────────────────────
stats = get_stats()
print(f"\n✅ Step 4: ChromaDB stats → {stats['total_documents']} docs, {stats['num_sources']} sources")

print("\n" + "=" * 55)
print("ALL TESTS PASSED ✅")
print("=" * 55)
print("\nNext steps:")
print("  1. Run real scrapers:       python main.py --collect")
print("  2. Embed real data:         python main.py --embed --reset")
print("  3. Launch dashboard:        streamlit run dashboard/app.py")
print("  (Ensure Ollama is running for LLM features: ollama serve)")
