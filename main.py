"""
main.py — NVIDIA Strategic Intelligence Agent
LangGraph stateful pipeline entry point.

Usage:
  python main.py --all              # full pipeline: scrape → embed → analyze → brief
  python main.py --all --reset      # wipe ChromaDB first, then full pipeline
  python main.py --no-collect       # skip scraping, use cached raw data
  python main.py --analyze          # analysis + briefing only (data already indexed)
  python main.py --graph            # print LangGraph node/edge structure
  python main.py --status           # show system status

After running --all, launch the dashboard:
  streamlit run dashboard/app.py

LLM Setup (Hugging Face — free):
  1. Get token: https://huggingface.co/settings/tokens
  2. Windows:   $env:HF_API_TOKEN = "hf_xxxxxxxxxxxx"
     Linux/Mac: export HF_API_TOKEN="hf_xxxxxxxxxxxx"

Integration Setup (optional):
  Slack:         Add SLACK_WEBHOOK_URL to .env
  Google Sheets: Add GOOGLE_SHEET_ID + GOOGLE_CREDENTIALS_PATH to .env
"""

import argparse
import json
import os
import sys

# Windows consoles often default to a legacy codepage (e.g. cp1252) that
# can't encode the emoji/arrows used throughout this project's output,
# crashing with UnicodeEncodeError mid-run. Force UTF-8 on stdout/stderr
# so a failed print never masks the real error it was trying to report.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def status():
    print("\n" + "="*60)
    print("SYSTEM STATUS")
    print("="*60)

    from vector_db.chroma_manager import get_stats
    from rag.llm_engine           import check_hf_status

    db_stats   = get_stats()
    llm_status = check_hf_status()

    print(f"\n📊 Knowledge Base:")
    print(f"   Documents indexed: {db_stats['total_documents']}")
    print(f"   Unique sources:    {db_stats['num_sources']}")
    print(f"   Sources:           {', '.join(db_stats['unique_sources'][:6])}")

    print(f"\n🤖 LLM (Hugging Face Inference API):")
    print(f"   Token set:     {llm_status['token_set']}")
    print(f"   Token valid:   {llm_status.get('token_valid', 'N/A')}")
    print(f"   Primary model: {llm_status['model']}")
    print(f"   Fast model:    {llm_status['fast_model']}")

    print(f"\n📊 LangGraph Pipeline:")
    print(f"   Nodes:  ScraperNode → IndexerNode → AnalystNode → CEONode")
    print(f"   Edges:  Sequential DAG with conditional abort after IndexerNode")
    print(f"   State:  PipelineState (TypedDict) shared across all nodes")

    print(f"\n🔗 Integrations:")
    slack_set  = bool(os.environ.get("SLACK_WEBHOOK_URL", "").strip())
    sheet_set  = bool(os.environ.get("GOOGLE_SHEET_ID", "").strip())
    creds_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "credentials/google_credentials.json")
    creds_ok   = os.path.exists(creds_path)
    print(f"   Slack:         {'✅ configured' if slack_set else '⬜ not configured (add SLACK_WEBHOOK_URL to .env)'}")
    print(f"   Google Sheets: {'✅ configured' if sheet_set and creds_ok else '⬜ not configured (add GOOGLE_SHEET_ID to .env)'}")

    raw_dir = os.path.join(ROOT, "data", "raw")
    if os.path.exists(raw_dir):
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
        print(f"\n📁 Raw data files: {raw_files}")
    else:
        print("\n📁 No raw data yet — run: python main.py --all")


def _push_integrations(final_state: dict) -> None:
    """
    Push pipeline results to Slack and Google Sheets.
    Both are optional — they skip gracefully if not configured.
    Called after every successful pipeline run.
    """
    print("\n── Pushing to integrations ──")

    try:
        from integrations.slack_notifier import post_intelligence_brief
        post_intelligence_brief(final_state)
    except Exception as e:
        print(f"[SlackNotifier] Error: {e}")

    try:
        from integrations.sheets_logger import log_pipeline_run
        log_pipeline_run(final_state)
    except Exception as e:
        print(f"[SheetsLogger] Error: {e}")


def analyze_only():
    from agents.nodes import analyst_node, ceo_node
    from agents.graph_state import PipelineState

    print("\n" + "="*60)
    print("ANALYSIS ONLY — running AnalystNode → CEONode")
    print("="*60)

    state: PipelineState = {
        "collect": False, "reset_db": False, "top_n": 5,
        "scrape_result": None, "index_result": {"documents_indexed": 1},
        "opportunities": None, "risks": None, "trends": None,
        "recommendations": None, "ceo_briefing": None,
        "errors": [], "status": "running",
    }

    state.update(analyst_node(state))
    state.update(ceo_node(state))

    print("\n── CEO BRIEFING ──\n")
    print(state.get("ceo_briefing", ""))

    _push_integrations(state)
    return state


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Strategic Intelligence Agent — LangGraph Pipeline"
    )
    parser.add_argument("--all",         action="store_true", help="Full pipeline")
    parser.add_argument("--no-collect",  action="store_true", help="Skip scraping")
    parser.add_argument("--analyze",     action="store_true", help="Analysis + briefing only")
    parser.add_argument("--status",      action="store_true", help="Show system status")
    parser.add_argument("--reset",       action="store_true", help="Reset ChromaDB")
    parser.add_argument("--graph",       action="store_true", help="Print graph structure")
    parser.add_argument("--no-integrations", action="store_true",
                        help="Skip Slack + Sheets after pipeline")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py --all                  # full LangGraph pipeline")
        print("  python main.py --no-collect           # skip scraping")
        print("  python main.py --graph                # show graph topology")
        print("  streamlit run dashboard/app.py        # launch dashboard")
        return

    if args.status:
        status()
        return

    if args.graph:
        from agents.orchestrator import print_graph_structure
        print_graph_structure()
        return

    if args.analyze:
        analyze_only()
        return

    from agents.orchestrator import run_pipeline
    final_state = run_pipeline(
        collect=not args.no_collect,
        reset_db=args.reset,
        top_n=5,
    )

    if not args.no_integrations:
        _push_integrations(final_state)

    print("\n🚀 Ready! Launch the dashboard:")
    print("   streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()