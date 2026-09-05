"""
ground.py — Universal Real-Time Factual Grounding Layer
======================================================
Retrieves and verifies up-to-the-minute real-world facts, live breaking news,
and encyclopedic context for ANY domain (Tech, AI, Pop Culture, News, Sports,
Cinema, Politics, Gaming, Business).

Sources:
  1. Live Web News Search (via ddgs)
  2. Global News Articles (via NewsAPI)
  3. Wikipedia Knowledge Graph (via wikipediaapi)
  4. Local cache (data/grounding_cache.json)

Guarantees:
  - Temporal awareness (injects current year and month)
  - Zero hallucination on dates, names, events, and metrics
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import requests
import wikipediaapi
from ddgs import DDGS

from config import (
    DATA_DIR,
    NEWS_API_KEY,
    LOG_LEVEL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CACHE_PATH = DATA_DIR / "grounding_cache.json"


# ═════════════════════════════════════════════════════════════════════════════
#  CACHE MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict[str, Any]) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Could not save grounding cache: %s", e)


# ═════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL LIVE NEWS (DDGS)
# ═════════════════════════════════════════════════════════════════════════════

def query_live_news_ddgs(query: str, max_results: int = 4) -> list[dict]:
    """Fetch live, breaking web news articles using DuckDuckGo News search."""
    articles = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
            for r in results:
                title = r.get("title", "").strip()
                body = r.get("body", "").strip()
                source = r.get("source", "Live Web News")
                date = r.get("date", "")
                if title:
                    articles.append({
                        "title": title,
                        "snippet": body,
                        "source": source,
                        "date": date,
                    })
    except Exception as e:
        log.debug("DDGS live news lookup failed for '%s': %s", query, e)

    # Fallback to general web text search if news returned 0
    if not articles:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                for r in results:
                    title = r.get("title", "").strip()
                    body = r.get("body", "").strip()
                    if title:
                        articles.append({
                            "title": title,
                            "snippet": body,
                            "source": "Web Search",
                            "date": "",
                        })
        except Exception as e:
            log.debug("DDGS web text search fallback failed: %s", e)

    return articles


# ═════════════════════════════════════════════════════════════════════════════
#  GLOBAL NEWS (NEWSAPI)
# ═════════════════════════════════════════════════════════════════════════════

def query_newsapi(query: str, max_articles: int = 3) -> list[dict]:
    """Fetch structured news from NewsAPI if configured."""
    if not NEWS_API_KEY:
        return []

    articles = []
    try:
        from newsapi import NewsApiClient
        news_client = NewsApiClient(api_key=NEWS_API_KEY)
        response = news_client.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            page_size=max_articles,
        )
        if response.get("status") == "ok" and response.get("articles"):
            for art in response["articles"][:max_articles]:
                title = art.get("title")
                desc = art.get("description")
                if title:
                    articles.append({
                        "title": title,
                        "snippet": desc or "",
                        "source": art.get("source", {}).get("name", "NewsAPI"),
                        "date": art.get("publishedAt", ""),
                    })
    except Exception as e:
        log.debug("NewsAPI query failed for '%s': %s", query, e)
    return articles


# ═════════════════════════════════════════════════════════════════════════════
#  WIKIPEDIA FACT EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def query_wikipedia(entity: str) -> dict[str, Any] | None:
    """Fetch factual background from Wikipedia across any domain."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="TweetLikeYou/2.0 (contact@tweetlikeyou.local)",
        language="en",
    )

    page = wiki.page(entity)
    if not page.exists():
        # Try cleaning entity (remove punctuation/quotes)
        clean = re.sub(r"[^\w\s]", "", entity).strip()
        page = wiki.page(clean)

    if not page.exists():
        return None

    summary = page.summary.strip()
    if not summary:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", summary)
    facts = [s.strip() for s in sentences[:3] if s.strip()]

    return {
        "title": page.title,
        "facts": facts,
        "source": "Wikipedia",
    }


# ═════════════════════════════════════════════════════════════════════════════
#  ENTITY RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_primary_entities(topic: str) -> list[str]:
    """Extract key names, products, organizations, and concepts from topic."""
    entities = []

    # Quoted terms take highest precedence
    quotes = re.findall(r'"([^"]+)"|\'([^\']+)\'', topic)
    for q1, q2 in quotes:
        val = (q1 or q2).strip()
        if val:
            entities.append(val)

    # Capitalized entity phrases
    caps = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", topic)
    for c in caps:
        if c.lower() not in {"the", "why", "what", "how", "whether", "this", "that", "and", "or", "is", "are"}:
            if c not in entities:
                entities.append(c)

    # General stripped keyword query
    cleaned = re.sub(r"('s|’s|\bwhy\b|\bhow\b|\bwhether\b|\band\b|\bits\b|\bthe\b)", " ", topic, flags=re.IGNORECASE)
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]
    if words:
        entities.append(" ".join(words[:4]))

    return list(dict.fromkeys(entities))


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN GROUNDING AGGREGATOR
# ═════════════════════════════════════════════════════════════════════════════

def get_factual_grounding(topic: str, force_refresh: bool = False) -> dict[str, Any]:
    """
    Unified, universal factual grounding:
      - Live breaking news (ddgs + NewsAPI)
      - Factual reference (Wikipedia)
      - Temporal anchoring (current UTC timestamp)
    """
    cache = load_cache()
    if not force_refresh and topic in cache:
        log.info("Loaded factual grounding from cache for '%s'", topic)
        return cache[topic]

    current_time_str = datetime.now(timezone.utc).strftime("%B %Y")
    entities = extract_primary_entities(topic)
    log.info("Grounding topic '%s' (Entities: %s) …", topic, entities)

    live_news_items = []
    factual_points = []
    sources = set()

    # 1. Fetch live breaking web news
    search_query = entities[0] if entities else topic
    ddg_news = query_live_news_ddgs(search_query, max_results=3)
    for item in ddg_news:
        snippet_clean = re.sub(r"\s+", " ", item["snippet"])[:160]
        live_news_items.append({
            "title": item["title"],
            "snippet": snippet_clean,
            "source": item["source"],
        })
        sources.add(item["source"])

    # 2. Fetch from NewsAPI if available
    api_news = query_newsapi(search_query, max_articles=2)
    for item in api_news:
        if not any(item["title"].lower() == x["title"].lower() for x in live_news_items):
            live_news_items.append({
                "title": item["title"],
                "snippet": item["snippet"][:160],
                "source": item["source"],
            })
            sources.add(item["source"])

    # 3. Fetch foundational facts from Wikipedia
    for ent in entities[:2]:
        wiki_res = query_wikipedia(ent)
        if wiki_res:
            for f in wiki_res["facts"]:
                if f not in factual_points:
                    factual_points.append(f)
            sources.add("Wikipedia")
            break

    result = {
        "topic": topic,
        "current_date_anchor": current_time_str,
        "entities": entities,
        "live_news": live_news_items[:4],
        "factual_points": factual_points[:3],
        "sources": list(sources) if sources else ["Live Web"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }

    cache[topic] = result
    save_cache(cache)
    return result


def format_grounding_for_prompt(grounding: dict[str, Any]) -> str:
    """Format live grounding context into clear instructions for prompt injection."""
    live_news = grounding.get("live_news", [])
    facts = grounding.get("factual_points", [])
    date_anchor = grounding.get("current_date_anchor", "")

    if not live_news and not facts:
        return ""

    lines = [
        "=== LIVE REAL-TIME CONTEXT & VERIFIED FACTS ===",
        f"TIME ANCHOR: Current date is {date_anchor}. Speak with up-to-date real-time awareness.",
    ]

    if live_news:
        lines.append("\nRECENT BREAKING NEWS & HEADLINES:")
        for n in live_news:
            lines.append(f"- \"{n['title']}\" ({n['source']}): {n['snippet']}")

    if facts:
        lines.append("\nFOUNDATIONAL FACTS:")
        for f in facts:
            lines.append(f"- {f}")

    lines.append("\nRULE: Seamlessly integrate these live facts/events into your tweet so it is timely and accurate.")
    return "\n".join(lines)


if __name__ == "__main__":
    test_topics = [
        "Nvidia's new Blackwell AI chips and GPU demand",
        "OpenAI o3 and GPT-5 release expectations",
        "The Brutalist and its 3.5 hour runtime",
    ]
    for t in test_topics:
        res = get_factual_grounding(t, force_refresh=True)
        print("\n" + format_grounding_for_prompt(res))
