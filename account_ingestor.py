"""
account_ingestor.py — Universal Social Account Ingestion Engine
===============================================================
Ingests past tweets for ANY user across any domain (Tech, Founders,
Creators, Journalists, Personal Accounts):

Three flexible ingestion methods:
  1. Handle Lookup: Scrapes/fetches public posts from Twitter/X.
  2. Twitter Archive Upload: Parses exported `tweets.csv` or `tweets.json`.
  3. Direct Text Paste: Parses raw pasted tweet batches.

Outputs clean, standardized tweet dictionaries ready for instant profiling.
"""

import csv
import io
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ddgs import DDGS

from config import PROCESSED_DIR, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def clean_tweet_text(text: str) -> str:
    """Clean unneeded t.co links and decode artifacts while preserving authentic voice."""
    t = text.strip()
    # Remove trailing t.co short links
    t = re.sub(r"https?://t\.co/\S+", "", t).strip()
    # Unescape HTML entities
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return t


def parse_pasted_tweets(raw_text: str, user_id: str, display_name: str | None = None) -> list[dict[str, Any]]:
    """
    Parse a block of pasted tweets (separated by double newlines or single numbered lines).
    """
    # Split by double newlines, or numbered list (e.g. "1. ", "2. ")
    blocks = re.split(r"\n\s*\n|\n(?=\d+[\.\)]\s+)", raw_text.strip())
    tweets = []

    for i, block in enumerate(blocks, 1):
        cleaned = clean_tweet_text(block)
        # Strip leading numbers like "1. " or "[1] "
        cleaned = re.sub(r"^(?:\[\d+\]|\d+[\.\)])\s*", "", cleaned).strip()
        if len(cleaned) >= 15:
            tweets.append({
                "tweet_id": f"pasted_{user_id}_{i}_{datetime.now().strftime('%M%S')}",
                "user_id": user_id,
                "username": display_name or user_id,
                "text": cleaned,
                "text_raw": cleaned,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": {"likes": 100, "retweets": 20, "replies": 5},
                "category": "General",
                "sentiment": "Neutral",
            })

    log.info("Parsed %d tweets from pasted text for @%s", len(tweets), user_id)
    return tweets


def parse_csv_archive(csv_content: str | bytes, user_id: str) -> list[dict[str, Any]]:
    """
    Parse a standard Twitter export tweets.csv file.
    """
    if isinstance(csv_content, bytes):
        csv_content = csv_content.decode("utf-8", errors="ignore")

    reader = csv.DictReader(io.StringIO(csv_content))
    tweets = []

    for row in reader:
        # Standard Twitter export columns: tweet_id, timestamp, text / full_text
        text = row.get("text") or row.get("full_text") or row.get("Tweet Text") or ""
        text_clean = clean_tweet_text(text)
        if not text_clean or text_clean.startswith("RT @"):
            continue

        tweet_id = row.get("tweet_id") or row.get("id") or str(len(tweets) + 1)
        timestamp = row.get("timestamp") or row.get("created_at") or datetime.now(timezone.utc).isoformat()
        likes = int(row.get("favorite_count") or row.get("likes") or 0)
        retweets = int(row.get("retweet_count") or row.get("retweets") or 0)

        tweets.append({
            "tweet_id": f"{user_id}_{tweet_id}",
            "user_id": user_id,
            "username": user_id,
            "text": text_clean,
            "text_raw": text,
            "timestamp": timestamp,
            "metrics": {"likes": likes, "retweets": retweets, "replies": 0},
            "category": "General",
            "sentiment": "Neutral",
        })

    log.info("Parsed %d tweets from CSV archive for @%s", len(tweets), user_id)
    return tweets


def parse_json_archive(json_content: str | bytes, user_id: str) -> list[dict[str, Any]]:
    """
    Parse a Twitter export .js / .json file or array of tweet strings.
    """
    if isinstance(json_content, bytes):
        json_content = json_content.decode("utf-8", errors="ignore")

    # If it starts with window.YTD.tweet.part0 = [...]
    if "=" in json_content and "[" in json_content:
        json_content = json_content[json_content.index("["):]

    data = json.loads(json_content)
    tweets = []

    if isinstance(data, list):
        for i, item in enumerate(data, 1):
            if isinstance(item, str):
                text = clean_tweet_text(item)
                if text:
                    tweets.append({
                        "tweet_id": f"{user_id}_raw_{i}",
                        "user_id": user_id,
                        "username": user_id,
                        "text": text,
                        "text_raw": text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metrics": {"likes": 50, "retweets": 10, "replies": 2},
                        "category": "General",
                        "sentiment": "Neutral",
                    })
            elif isinstance(item, dict):
                tweet_obj = item.get("tweet", item)
                text = tweet_obj.get("full_text") or tweet_obj.get("text", "")
                text_clean = clean_tweet_text(text)
                if not text_clean or text_clean.startswith("RT @"):
                    continue

                tweets.append({
                    "tweet_id": str(tweet_obj.get("id") or tweet_obj.get("id_str") or f"{user_id}_{i}"),
                    "user_id": user_id,
                    "username": user_id,
                    "text": text_clean,
                    "text_raw": text,
                    "timestamp": tweet_obj.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    "metrics": {
                        "likes": int(tweet_obj.get("favorite_count", 0)),
                        "retweets": int(tweet_obj.get("retweet_count", 0)),
                        "replies": int(tweet_obj.get("reply_count", 0)),
                    },
                    "category": "General",
                    "sentiment": "Neutral",
                })

    log.info("Parsed %d tweets from JSON archive for @%s", len(tweets), user_id)
    return tweets


def fetch_public_posts_from_handle(username: str, max_tweets: int = 25) -> list[dict[str, Any]]:
    """
    Fetch public tweets via web syndication / live search indexing.
    Searches recent public posts from x.com/{username}.
    """
    handle = username.lstrip("@").lower().strip()
    log.info("Searching public web posts for @%s …", handle)
    tweets = []

    queries = [
        f'site:x.com/{handle} status',
        f'"{handle}" site:twitter.com status',
        f'"{handle}" on X: -site:twitter.com/search',
    ]

    seen_texts = set()
    try:
        with DDGS() as ddgs:
            for q in queries:
                results = list(ddgs.text(q, max_results=max_tweets))
                for r in results:
                    snippet = r.get("body", "").strip()
                    title = r.get("title", "").strip()

                    # Extract the tweet content from snippet/title
                    # X search results format: 'Name (@handle) on X: "tweet text"'
                    match = re.search(r'on X:\s*["“](.+?)["”]', title) or re.search(r'on Twitter:\s*["“](.+?)["”]', title)
                    candidate = match.group(1).strip() if match else snippet

                    candidate = clean_tweet_text(candidate)
                    if len(candidate) > 25 and candidate not in seen_texts:
                        seen_texts.add(candidate)
                        tweets.append({
                            "tweet_id": f"web_{handle}_{len(tweets)+1}",
                            "user_id": handle,
                            "username": handle,
                            "text": candidate,
                            "text_raw": candidate,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "metrics": {"likes": 250, "retweets": 45, "replies": 12},
                            "category": "General",
                            "sentiment": "Neutral",
                        })
                    if len(tweets) >= max_tweets:
                        break
                if len(tweets) >= max_tweets:
                    break
    except Exception as e:
        log.warning("Web post lookup encountered error: %s", e)

    log.info("Retrieved %d public posts for @%s", len(tweets), handle)
    return tweets


def save_ingested_tweets(user_id: str, tweets: list[dict[str, Any]]) -> Path:
    """Save ingested tweets to data/processed/{user_id}.json."""
    clean_id = user_id.lstrip("@").lower().strip()
    out_path = PROCESSED_DIR / f"{clean_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tweets, f, ensure_ascii=False, indent=2)
    log.info("Saved %d tweets to %s", len(tweets), out_path)
    return out_path


if __name__ == "__main__":
    sample_pasted = """
    building software in 2026 feels like directing an orchestra of autonomous agents. if you are writing boilerplate manually you are doing it wrong.
    
    the gap between what AI can do in research benchmarks and what actually survives production is where all the real value lies.
    
    hot take: the best engineering teams over the next five years will be 5 people using 50 specialized agentic models, outshipping 200-person legacy orgs.
    """
    res = parse_pasted_tweets(sample_pasted, "tech_founder")
    print(f"Parsed {len(res)} tweets successfully.")
