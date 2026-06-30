"""
news_scraper.py
Collects NVIDIA-related financial and technology news from multiple
public RSS feeds (Google News, Yahoo Finance, Reuters, etc.).
Saves raw documents to data/raw/.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import json
import os
import hashlib
from datetime import datetime
from urllib.parse import quote

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Multiple independent news RSS sources for NVIDIA
NEWS_FEEDS = [
    {
        "name": "google_news_nvidia",
        "url": "https://news.google.com/rss/search?q=NVIDIA+stock+AI&hl=en-US&gl=US&ceid=US:en",
        "source_type": "financial_news",
    },
    {
        "name": "google_news_nvidia_tech",
        "url": "https://news.google.com/rss/search?q=NVIDIA+GPU+technology&hl=en-US&gl=US&ceid=US:en",
        "source_type": "tech_news",
    },
    {
        "name": "yahoo_finance_nvda",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
        "source_type": "financial_news",
    },
    {
        "name": "seeking_alpha_nvda",
        "url": "https://seekingalpha.com/api/sa/combined/NVDA.xml",
        "source_type": "analyst_report",
    },
    {
        "name": "google_news_nvidia_competitors",
        "url": "https://news.google.com/rss/search?q=AMD+Intel+GPU+AI+chip&hl=en-US&gl=US&ceid=US:en",
        "source_type": "competitor_news",
    },
    {
        "name": "google_news_ai_market",
        "url": "https://news.google.com/rss/search?q=artificial+intelligence+semiconductor+market&hl=en-US&gl=US&ceid=US:en",
        "source_type": "market_news",
    },
    {
        "name": "techcrunch_ai",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "source_type": "tech_news",
    },
    {
        "name": "venturebeat_ai",
        "url": "https://venturebeat.com/category/ai/feed/",
        "source_type": "tech_news",
    },
]


def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _clean_html(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator=" ").strip()


def scrape_feed(feed_info: dict) -> list[dict]:
    documents = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        # feedparser can accept headers via request_headers
        parsed = feedparser.parse(feed_info["url"], request_headers=headers)

        for entry in parsed.entries[:40]:  # cap per feed
            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            link = entry.get("link", "")
            published = entry.get("published", str(datetime.now()))

            clean_summary = _clean_html(summary)
            content = f"{title}. {clean_summary}"

            if len(content) < 30:
                continue

            # Filter: must mention NVIDIA or competitor context
            keywords = ["nvidia", "nvda", "gpu", "ai chip", "amd", "intel", "artificial intelligence",
                        "semiconductor", "data center", "machine learning", "deep learning", "cuda"]
            if not any(k in content.lower() for k in keywords):
                continue

            doc = {
                "id": _doc_id(content),
                "title": title,
                "content": content,
                "url": link,
                "published": published,
                "source": feed_info["name"],
                "source_type": feed_info["source_type"],
                "company": "NVIDIA",
                "collected_at": datetime.now().isoformat(),
            }
            documents.append(doc)

        print(f"  [news_scraper] {feed_info['name']}: {len(documents)} articles")
    except Exception as e:
        print(f"  [news_scraper] Error on {feed_info['name']}: {e}")
    return documents


def deduplicate(docs: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for d in docs:
        if d["id"] not in seen:
            seen.add(d["id"])
            unique.append(d)
    return unique


def run() -> list[dict]:
    print("[news_scraper] Collecting news from multiple RSS feeds...")
    all_docs = []

    for feed in NEWS_FEEDS:
        all_docs.extend(scrape_feed(feed))

    all_docs = deduplicate(all_docs)

    out_path = os.path.join(RAW_DIR, "news_articles.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    print(f"[news_scraper] Saved {len(all_docs)} unique articles → {out_path}")
    return all_docs


if __name__ == "__main__":
    docs = run()
    print(f"Total: {len(docs)}")
