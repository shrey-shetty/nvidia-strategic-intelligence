"""
community_scraper.py (replaces reddit_scraper.py)
Collects NVIDIA-related discussions from Hacker News public API.
No credentials required — completely free and open.

Hacker News API docs: https://hacker-news.firebaseio.com/v0/
"""

import json
import os
import hashlib
import time
import requests
from datetime import datetime

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

HN_API   = "https://hacker-news.firebaseio.com/v0"
HN_ALGOLIA = "https://hn.algolia.com/api/v1/search"

SEARCH_QUERIES = [
    "NVIDIA", "NVDA", "GPU AI", "NVIDIA Blackwell",
    "CUDA deep learning", "NVIDIA data center",
]

POST_LIMIT = 30  # per query


def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def scrape_hn_algolia() -> list[dict]:
    """
    Use Hacker News Algolia search API — fastest and most reliable.
    Returns posts mentioning NVIDIA sorted by recency.
    """
    documents = []
    headers = {"User-Agent": "nvidia-intel-agent/0.1"}

    for query in SEARCH_QUERIES:
        try:
            resp = requests.get(
                HN_ALGOLIA,
                params={
                    "query":    query,
                    "tags":     "story",
                    "hitsPerPage": POST_LIMIT,
                },
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])

            for hit in hits:
                title   = hit.get("title", "")
                url     = hit.get("url", "")
                story_text = hit.get("story_text") or ""
                points  = hit.get("points", 0)
                num_comments = hit.get("num_comments", 0)
                created = hit.get("created_at", datetime.now().isoformat())
                hn_url  = f"https://news.ycombinator.com/item?id={hit.get('objectID','')}"

                content = f"{title}. {story_text}".strip()
                if len(content) < 20:
                    content = title

                # Filter for NVIDIA relevance
                if not any(k in content.lower() for k in [
                    "nvidia", "nvda", "gpu", "cuda", "ai chip",
                    "blackwell", "hopper", "semiconductor", "jensen"
                ]):
                    continue

                doc = {
                    "id":           _doc_id(title + str(hit.get("objectID",""))),
                    "title":        title,
                    "content":      content[:2000],
                    "url":          url or hn_url,
                    "published":    created,
                    "source":       "hacker_news",
                    "source_type":  "community",
                    "company":      "NVIDIA",
                    "score":        points,
                    "num_comments": num_comments,
                    "collected_at": datetime.now().isoformat(),
                }
                documents.append(doc)

            print(f"  [community_scraper] HN '{query}': {len(hits)} fetched")
            time.sleep(0.3)

        except Exception as e:
            print(f"  [community_scraper] Error on '{query}': {e}")

    return documents


def scrape_hn_top_stories() -> list[dict]:
    """
    Scrape top HN stories and filter for NVIDIA relevance.
    Supplements the search results with trending stories.
    """
    documents = []
    headers   = {"User-Agent": "nvidia-intel-agent/0.1"}

    try:
        # Get top 200 story IDs
        resp = requests.get(f"{HN_API}/topstories.json", headers=headers, timeout=10)
        story_ids = resp.json()[:100]  # check top 100

        nvidia_count = 0
        for sid in story_ids:
            if nvidia_count >= 15:
                break
            try:
                r = requests.get(f"{HN_API}/item/{sid}.json", headers=headers, timeout=8)
                item = r.json()
                if not item or item.get("type") != "story":
                    continue

                title = item.get("title", "")
                if not any(k in title.lower() for k in ["nvidia","nvda","gpu","cuda","ai chip","blackwell","jensen"]):
                    continue

                content = f"{title}. {item.get('text','')}"
                doc = {
                    "id":           _doc_id(str(sid)),
                    "title":        title,
                    "content":      content[:2000],
                    "url":          item.get("url", f"https://news.ycombinator.com/item?id={sid}"),
                    "published":    datetime.utcfromtimestamp(item.get("time", 0)).isoformat(),
                    "source":       "hacker_news_top",
                    "source_type":  "community",
                    "company":      "NVIDIA",
                    "score":        item.get("score", 0),
                    "num_comments": item.get("descendants", 0),
                    "collected_at": datetime.now().isoformat(),
                }
                documents.append(doc)
                nvidia_count += 1
                time.sleep(0.1)

            except Exception:
                continue

        print(f"  [community_scraper] HN top stories: {nvidia_count} NVIDIA stories found")

    except Exception as e:
        print(f"  [community_scraper] HN top stories error: {e}")

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
    print("[community_scraper] Collecting from Hacker News (no credentials needed)...")

    docs = scrape_hn_algolia()
    docs.extend(scrape_hn_top_stories())
    docs = deduplicate(docs)

    out_path = os.path.join(RAW_DIR, "community_posts.json")  # keep filename for compatibility
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"[community_scraper] Saved {len(docs)} posts → {out_path}")
    return docs


if __name__ == "__main__":
    docs = run()
    print(f"Total: {len(docs)}")

