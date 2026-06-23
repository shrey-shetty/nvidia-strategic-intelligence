"""
nvidia_scraper.py
Collects NVIDIA press releases and investor relations news via RSS feeds
and targeted HTML scraping. Saves raw documents to data/raw/.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import json
import os
import hashlib
from datetime import datetime

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# NVIDIA public RSS / sitemap sources
NVIDIA_FEEDS = [
    {
        "name": "nvidia_press_releases",
        "url": "https://nvidianews.nvidia.com/rss/",
        "source_type": "press_release",
    },
    {
        "name": "nvidia_blog",
        "url": "https://blogs.nvidia.com/feed/",
        "source_type": "blog",
    },
]

# Fallback: scrape NVIDIA newsroom page directly
NVIDIA_NEWSROOM_URL = "https://nvidianews.nvidia.com/news/latest"


def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def scrape_rss_feed(feed_info: dict) -> list[dict]:
    """Parse an RSS feed and return a list of document dicts."""
    documents = []
    try:
        parsed = feedparser.parse(feed_info["url"])
        for entry in parsed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            link = entry.get("link", "")
            published = entry.get("published", str(datetime.now()))

            # Clean HTML from summary
            soup = BeautifulSoup(summary, "html.parser")
            clean_text = soup.get_text(separator=" ").strip()

            content = f"{title}. {clean_text}"
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
            if len(content) > 50:
                documents.append(doc)

        print(f"  [nvidia_scraper] {feed_info['name']}: {len(documents)} articles")
    except Exception as e:
        print(f"  [nvidia_scraper] Error scraping {feed_info['name']}: {e}")
    return documents


def scrape_newsroom_html() -> list[dict]:
    """Fallback: scrape the NVIDIA newsroom page for headlines."""
    documents = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        resp = requests.get(NVIDIA_NEWSROOM_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # NVIDIA newsroom uses article cards
        for card in soup.select("article, .news-item, .press-release-item")[:30]:
            title_el = card.select_one("h2, h3, h1, .title")
            body_el = card.select_one("p, .summary, .description")
            link_el = card.select_one("a[href]")

            title = title_el.get_text(strip=True) if title_el else ""
            body = body_el.get_text(strip=True) if body_el else ""
            link = link_el["href"] if link_el else ""
            if link and not link.startswith("http"):
                link = "https://nvidianews.nvidia.com" + link

            content = f"{title}. {body}"
            if len(content) > 50:
                doc = {
                    "id": _doc_id(content),
                    "title": title,
                    "content": content,
                    "url": link,
                    "published": datetime.now().isoformat(),
                    "source": "nvidia_newsroom",
                    "source_type": "press_release",
                    "company": "NVIDIA",
                    "collected_at": datetime.now().isoformat(),
                }
                documents.append(doc)

        print(f"  [nvidia_scraper] newsroom HTML: {len(documents)} articles")
    except Exception as e:
        print(f"  [nvidia_scraper] Newsroom HTML fallback error: {e}")
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
    print("[nvidia_scraper] Collecting NVIDIA official sources...")
    all_docs = []

    for feed in NVIDIA_FEEDS:
        all_docs.extend(scrape_rss_feed(feed))

    # Supplement with HTML scrape if we got too few
    if len(all_docs) < 20:
        all_docs.extend(scrape_newsroom_html())

    all_docs = deduplicate(all_docs)

    # Save to disk
    out_path = os.path.join(RAW_DIR, "nvidia_official.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    print(f"[nvidia_scraper] Saved {len(all_docs)} documents → {out_path}")
    return all_docs


if __name__ == "__main__":
    docs = run()
    print(f"Total: {len(docs)}")
