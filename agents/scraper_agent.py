"""
agents/scraper_agent.py
ScraperAgent — live data collection from NVIDIA's public sources.
Calls all three scrapers and returns a summary dict.
Wired into the LangGraph pipeline via agents/nodes.py::scraper_node.
"""

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_scrapers() -> dict:
    """
    Execute all three scrapers and return a summary dict.
    Called by agents/nodes.py::scraper_node.
    """
    from scrapers.nvidia_scraper    import run as nvidia_run
    from scrapers.news_scraper      import run as news_run
    from scrapers.community_scraper import run as community_run

    print("[ScraperAgent] Starting data collection...")

    nvidia_docs    = nvidia_run()
    news_docs      = news_run()
    community_docs = community_run()

    total = len(nvidia_docs) + len(news_docs) + len(community_docs)

    result = {
        "status":          "complete",
        "total_documents": total,
        "nvidia_official": len(nvidia_docs),
        "news_articles":   len(news_docs),
        "community_posts": len(community_docs),
    }

    print(f"[ScraperAgent] Collection complete: {total} documents")
    return result


if __name__ == "__main__":
    print(json.dumps(run_scrapers(), indent=2))
