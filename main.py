"""
main.py  —  NVIDIA Strategic Intelligence Agent: Orchestration Entry Point

Usage:
  python main.py --collect          # scrape all sources
  python main.py --embed            # clean, embed, store in ChromaDB
  python main.py --analyze          # run intelligence analysis + print summary
  python main.py --all              # collect + embed + analyze in one step
  python main.py --status           # show system status
  python main.py --reset            # wipe ChromaDB and re-embed

After running --all, launch the dashboard:
  streamlit run dashboard/app.py
"""

import argparse
import json
import os
import sys

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ── Step 1: Collect ───────────────────────────────────────────────────────────
def collect():
    print("\n" + "="*60)
    print("STEP 1: LIVE DATA COLLECTION")
    print("="*60)

    from scrapers.nvidia_scraper import run as nvidia_run
    from scrapers.news_scraper   import run as news_run
    from scrapers.community_scraper import run as community_run

    nvidia_docs = nvidia_run()
    news_docs   = news_run()
    community_docs = community_run()

    total = len(nvidia_docs) + len(news_docs) + len(community_docs)
    print(f"\n✅ Collection complete: {total} total documents")
    print(f"   • NVIDIA official: {len(nvidia_docs)}")
    print(f"   • News articles:   {len(news_docs)}")
    print(f"   • Community posts:    {len(community_docs)}")

    if total < 100:
        print(f"\n⚠️  Warning: only {total} documents collected (minimum is 100).")
        print("   Check your internet connection and try again.")
    return total


# ── Step 2: Embed ─────────────────────────────────────────────────────────────
def embed(reset: bool = False):
    print("\n" + "="*60)
    print("STEP 2: EMBEDDING & INDEXING")
    print("="*60)

    from embeddings.embed_documents import run as embed_run
    count = embed_run(reset=reset)
    print(f"\n✅ Embedding complete: {count} documents indexed in ChromaDB")
    return count


# ── Step 3: Analyze ───────────────────────────────────────────────────────────
def analyze():
    print("\n" + "="*60)
    print("STEP 3: STRATEGIC ANALYSIS")
    print("="*60)

    from intelligence.opportunity_detector import detect_opportunities
    from intelligence.risk_detector        import detect_risks
    from intelligence.trend_detector       import detect_trends
    from rag.ceo_agent                     import generate_recommendations, generate_ceo_briefing

    print("\n[1/4] Detecting opportunities...")
    opps = detect_opportunities(top_n=5)

    print("\n[2/4] Detecting risks...")
    risks = detect_risks(top_n=5)

    print("\n[3/4] Detecting trends...")
    trends = detect_trends(top_n=6)

    print("\n[4/4] Generating strategic recommendations...")
    recs = generate_recommendations(top_n=5)

    # Print summary to console
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    print(f"\n📈 OPPORTUNITIES ({len(opps)} found):")
    for o in opps:
        print(f"  [{o.get('impact_level','?')}] {o.get('title','')}")

    print(f"\n⚠️  RISKS ({len(risks)} found):")
    for r in risks:
        print(f"  [{r.get('severity_level','?')}] {r.get('title','')}")

    print(f"\n📡 TRENDS ({len(trends)} found):")
    for t in trends:
        print(f"  [{t.get('time_horizon','?')}] {t.get('trend_name','')}")

    print(f"\n🎯 RECOMMENDATIONS ({len(recs)} generated):")
    for rec in recs:
        print(f"  [{rec.get('priority','?')}] {rec.get('recommendation','')}")

    # CEO Briefing
    print("\n" + "="*60)
    print("CEO EXECUTIVE BRIEFING")
    print("="*60)
    briefing = generate_ceo_briefing(
        opportunities=opps,
        risks=risks,
        trends=trends,
        recommendations=recs,
    )
    print(briefing)

    # Save analysis results to disk
    results = {
        "opportunities":    opps,
        "risks":            risks,
        "trends":           trends,
        "recommendations":  recs,
        "ceo_briefing":     briefing,
    }
    out_path = os.path.join(ROOT, "data", "analysis_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Analysis saved → {out_path}")

    return results


# ── Status ────────────────────────────────────────────────────────────────────
def status():
    print("\n" + "="*60)
    print("SYSTEM STATUS")
    print("="*60)

    from vector_db.chroma_manager import get_stats
    from rag.ceo_agent            import check_ollama_status

    db_stats = get_stats()
    llm_status = check_ollama_status()

    print(f"\n📊 Knowledge Base:")
    print(f"   Documents indexed: {db_stats['total_documents']}")
    print(f"   Unique sources:    {db_stats['num_sources']}")
    print(f"   Sources:           {', '.join(db_stats['unique_sources'][:5])}")

    print(f"\n🤖 LLM (Ollama):")
    print(f"   Running:           {llm_status['running']}")
    print(f"   Configured model:  {llm_status['configured_model']}")
    if llm_status["running"]:
        print(f"   Available models:  {', '.join(llm_status['available_models'])}")
    else:
        print("   → Start Ollama:  ollama serve")
        print("   → Pull model:    ollama pull llama3.1:8b")

    # Check raw data files
    raw_dir = os.path.join(ROOT, "data", "raw")
    if os.path.exists(raw_dir):
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
        print(f"\n📁 Raw data files: {raw_files}")
    else:
        print("\n📁 No raw data yet — run: python main.py --collect")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Strategic Intelligence Agent — Orchestration"
    )
    parser.add_argument("--collect", action="store_true", help="Scrape live data from all sources")
    parser.add_argument("--embed",   action="store_true", help="Embed and index documents into ChromaDB")
    parser.add_argument("--analyze", action="store_true", help="Run strategic analysis and generate CEO briefing")
    parser.add_argument("--all",     action="store_true", help="Run full pipeline: collect → embed → analyze")
    parser.add_argument("--status",  action="store_true", help="Show system status")
    parser.add_argument("--reset",   action="store_true", help="Reset ChromaDB before embedding")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py --all                  # full pipeline")
        print("  streamlit run dashboard/app.py        # launch dashboard")
        return

    if args.status:
        status()
        return

    if args.all:
        collect()
        embed(reset=args.reset)
        analyze()
        print("\n🚀 Ready! Launch the dashboard:")
        print("   streamlit run dashboard/app.py")
        return

    if args.collect:
        collect()

    if args.embed:
        embed(reset=args.reset)

    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()

