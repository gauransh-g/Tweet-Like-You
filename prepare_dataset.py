"""
prepare_dataset.py — Phase 1: Dataset preparation
==================================================
Reads raw JSON files from data/raw/, normalises every tweet to a unified
schema, cleans text (without destroying style), and writes clean JSON files
to data/processed/.

Run:
    python prepare_dataset.py

Output:
    data/processed/davidehrlich.json
    data/processed/kevin_thecritic.json
    data/processed/theinsneider.json
    data/processed/all_users.json      ← combined dataset
    data/processed/dataset_stats.json  ← summary statistics
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from config import (
    RAW_FILES,
    PROCESSED_DIR,
    CATEGORY_MAP,
    INSNEIDER_META,
    LOG_LEVEL,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  TEXT CLEANING
#  Rule: preserve style (!, ?, ..., CAPS, emoji, slang, hashtags, mentions)
#        remove only: broken encoding, URLs, tracking params, whitespace noise
# ═════════════════════════════════════════════════════════════════════════════

# Broken Windows-1252 / mojibake sequences seen in TheInSneider data
_ENCODING_FIXES = [
    (r"\ufffd|\u0091|\u0092|\u0093|\u0094|\u0095|\u0096|\u0097", ""),   # replacement chars
    (r"\u00e2\u0080\u0099",  "'"),   # â€™ → '
    (r"\u00e2\u0080\u009c",  '"'),   # â€œ → "
    (r"\u00e2\u0080\u009d",  '"'),   # â€  → "
    (r"\u00e2\u0080\u0094",  "—"),   # â€" → —
    (r"\u00e2\u0080\u0093",  "–"),   # â€" → –
    (r"\u00e2\u0080\u00a6",  "…"),   # â€¦ → …
    (r"&amp;",               "&"),
    (r"&lt;",                "<"),
    (r"&gt;",                ">"),
    (r"&quot;",              '"'),
]

# URL pattern (https://t.co/... and similar)
_URL_RE = re.compile(r"https?://\S+")

# Multiple spaces / tabs → single space
_WHITESPACE_RE = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """
    Clean tweet text while preserving stylistic information.

    Removed:  URLs, broken encoding artifacts, excess whitespace
    Preserved: !, ?, ..., CAPS, emoji, slang, hashtags (#), mentions (@),
               punctuation, sentence structure
    """
    if not text:
        return ""

    # Fix broken encoding artifacts
    for pattern, replacement in _ENCODING_FIXES:
        text = re.sub(pattern, replacement, text)

    # Remove URLs (t.co links carry no stylistic value)
    text = _URL_RE.sub("", text)

    # Collapse multiple spaces/tabs (but NOT newlines — keep structure)
    text = _WHITESPACE_RE.sub(" ", text)

    # Strip leading/trailing whitespace only
    text = text.strip()

    return text


# ═════════════════════════════════════════════════════════════════════════════
#  TIMESTAMP NORMALISATION
# ═════════════════════════════════════════════════════════════════════════════

def normalise_timestamp(raw: str) -> str:
    """
    Convert any timestamp format to ISO 8601 UTC string.
    Handles:
      - "2026-04-06T17:50:19+00:00"  (david / kevin)
      - "Tue, 07 Apr 2026 05:43:44 GMT"  (TheInSneider — RFC 2822)
    Returns "" if unparseable.
    """
    if not raw:
        return ""
    try:
        # Try ISO first
        dt = datetime.fromisoformat(raw)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    try:
        # Try RFC 2822 (email/HTTP date format)
        dt = parsedate_to_datetime(raw)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    log.warning("Could not parse timestamp: %s", raw)
    return raw  # return as-is rather than losing data


# ═════════════════════════════════════════════════════════════════════════════
#  CATEGORY NORMALISATION
# ═════════════════════════════════════════════════════════════════════════════

def normalise_category(raw: str) -> str:
    """Map messy category labels to a clean canonical set."""
    if not raw:
        return "General"
    key = raw.strip().lower()
    return CATEGORY_MAP.get(key, raw.strip().title())


# ═════════════════════════════════════════════════════════════════════════════
#  SCHEMA NORMALISERS  (one per raw format)
# ═════════════════════════════════════════════════════════════════════════════

def _build_tweet(
    user_id: str,
    username: str,
    tweet_id: str,
    text: str,
    timestamp: str,
    category: str,
    sentiment: str,
    metrics: dict,
) -> dict:
    """Return a tweet in the canonical unified schema."""
    cleaned = clean_text(text)
    return {
        "user_id":   user_id,
        "username":  username,
        "tweet_id":  str(tweet_id),
        "text":      cleaned,
        "text_raw":  text,            # keep original for debugging
        "timestamp": normalise_timestamp(timestamp),
        "category":  normalise_category(category),
        "sentiment": sentiment if sentiment else "Neutral",
        "metrics": {
            "likes":     int(metrics.get("likes", 0)),
            "reposts":   int(metrics.get("reposts", 0)),
            "replies":   int(metrics.get("replies", 0)),
            "bookmarks": int(metrics.get("bookmarks", 0)),
            "quotes":    int(metrics.get("quotes", 0)),
            "views":     int(metrics.get("views", 0)),
        },
    }


def normalise_david_kevin(raw_tweet: dict) -> dict:
    """Normalise a tweet from david.json or kevin.json (already mostly clean)."""
    m = raw_tweet.get("metrics", {})
    return _build_tweet(
        user_id   = raw_tweet.get("user_id", ""),
        username  = raw_tweet.get("username", ""),
        tweet_id  = raw_tweet.get("tweet_id", ""),
        text      = raw_tweet.get("text", ""),
        timestamp = raw_tweet.get("timestamp", ""),
        category  = raw_tweet.get("category", "General"),
        sentiment = raw_tweet.get("sentiment", "Neutral"),
        metrics   = m if isinstance(m, dict) else {},
    )


def normalise_insneider(raw_tweet: dict) -> dict:
    """
    Normalise a tweet from TheInSneider.json.
    Raw schema differences:
      - No user_id / username (add from config)
      - created_at instead of timestamp (RFC 2822 format)
      - No category / sentiment (infer defaults)
      - Metrics are flat (not nested)
    """
    return _build_tweet(
        user_id   = INSNEIDER_META["user_id"],
        username  = INSNEIDER_META["username"],
        tweet_id  = raw_tweet.get("tweet_id", ""),
        text      = raw_tweet.get("text", ""),
        timestamp = raw_tweet.get("created_at", ""),
        category  = "General",    # no category in raw — will be enriched in Phase 2
        sentiment = "Neutral",    # no sentiment in raw — will be enriched in Phase 2
        metrics   = {
            "likes":     raw_tweet.get("likes", 0),
            "reposts":   raw_tweet.get("reposts", 0),
            "replies":   raw_tweet.get("replies", 0),
            "bookmarks": raw_tweet.get("bookmarks", 0),
            "quotes":    raw_tweet.get("quotes", 0),
            "views":     raw_tweet.get("views", 0),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
#  QUALITY FILTERS
# ═════════════════════════════════════════════════════════════════════════════

def is_valid(tweet: dict) -> bool:
    """
    Return False for tweets that should be dropped:
      - Empty text after cleaning
      - Text shorter than 5 characters (almost certainly noise)
      - Pure retweet markers with no original content
      - Duplicate tweet IDs (handled separately)
    """
    text = tweet.get("text", "").strip()
    if not text or len(text) < 5:
        return False
    # Drop bare RT markers (no content)
    if re.match(r"^RT @\w+:?\s*$", text):
        return False
    return True


def deduplicate(tweets: list[dict]) -> list[dict]:
    """Remove duplicate tweet_ids, keeping first occurrence."""
    seen = set()
    unique = []
    for t in tweets:
        tid = t["tweet_id"]
        if tid not in seen:
            seen.add(tid)
            unique.append(t)
    return unique


# ═════════════════════════════════════════════════════════════════════════════
#  STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_stats(user_id: str, tweets: list[dict]) -> dict:
    """Return a summary stats dict for one user's dataset."""
    if not tweets:
        return {}

    lengths = [len(t["text"]) for t in tweets]
    categories = {}
    sentiments = {}
    total_likes = 0
    total_views = 0

    for t in tweets:
        cat = t["category"]
        categories[cat] = categories.get(cat, 0) + 1
        sent = t["sentiment"]
        sentiments[sent] = sentiments.get(sent, 0) + 1
        total_likes += t["metrics"]["likes"]
        total_views += t["metrics"]["views"]

    timestamps = sorted(t["timestamp"] for t in tweets if t["timestamp"])

    return {
        "user_id":        user_id,
        "total_tweets":   len(tweets),
        "avg_length":     round(sum(lengths) / len(lengths), 1),
        "min_length":     min(lengths),
        "max_length":     max(lengths),
        "date_start":     timestamps[0] if timestamps else "",
        "date_end":       timestamps[-1] if timestamps else "",
        "categories":     dict(sorted(categories.items(), key=lambda x: -x[1])),
        "sentiments":     dict(sorted(sentiments.items(), key=lambda x: -x[1])),
        "total_likes":    total_likes,
        "total_views":    total_views,
        "avg_likes":      round(total_likes / len(tweets), 1),
        "avg_views":      round(total_views / len(tweets), 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def process_user(user_id: str, raw_path: Path) -> tuple[list[dict], dict]:
    """
    Load, normalise, clean, and validate tweets for one user.
    Returns (clean_tweets, stats).
    """
    log.info("Processing %s from %s …", user_id, raw_path.name)

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Pick the right normaliser
    if user_id == "theinsneider":
        normalise = normalise_insneider
    else:
        normalise = normalise_david_kevin

    # Normalise all tweets
    tweets = [normalise(t) for t in raw]

    # Quality filter
    before = len(tweets)
    tweets = [t for t in tweets if is_valid(t)]
    dropped_invalid = before - len(tweets)

    # Deduplicate
    before = len(tweets)
    tweets = deduplicate(tweets)
    dropped_dupes = before - len(tweets)

    # Sort chronologically
    tweets.sort(key=lambda t: t["timestamp"])

    stats = compute_stats(user_id, tweets)
    stats["dropped_invalid"] = dropped_invalid
    stats["dropped_duplicates"] = dropped_dupes

    log.info(
        "  %-20s → %4d tweets  (dropped %d invalid, %d dupes)",
        user_id, len(tweets), dropped_invalid, dropped_dupes,
    )
    return tweets, stats


def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 1: Dataset Preparation")
    log.info("=" * 60)

    all_tweets = []
    all_stats  = []

    for user_id, raw_path in RAW_FILES.items():
        if not raw_path.exists():
            log.error("Raw file not found: %s — skipping.", raw_path)
            continue

        tweets, stats = process_user(user_id, raw_path)

        # Write per-user processed file
        out_path = PROCESSED_DIR / f"{user_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tweets, f, ensure_ascii=False, indent=2)
        log.info("  Saved → %s", out_path.relative_to(Path.cwd()))

        all_tweets.extend(tweets)
        all_stats.append(stats)

    # Write combined dataset
    combined_path = PROCESSED_DIR / "all_users.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_tweets, f, ensure_ascii=False, indent=2)
    log.info("Combined → %s  (%d total tweets)", combined_path.relative_to(Path.cwd()), len(all_tweets))

    # Write stats
    stats_path = PROCESSED_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_tweets": len(all_tweets),
                "total_users":  len(all_stats),
                "users":        all_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info("Stats   → %s", stats_path.relative_to(Path.cwd()))

    # ── Summary table ─────────────────────────────────────────────────────
    log.info("")
    log.info("─" * 60)
    log.info("SUMMARY")
    log.info("─" * 60)
    log.info("%-22s %8s %10s %10s", "User", "Tweets", "Avg Likes", "Avg Views")
    log.info("%-22s %8s %10s %10s", "─" * 22, "─" * 8, "─" * 10, "─" * 10)
    for s in all_stats:
        log.info(
            "%-22s %8d %10.1f %10.0f",
            s["user_id"], s["total_tweets"], s["avg_likes"], s["avg_views"]
        )
    log.info("─" * 60)
    log.info("%-22s %8d", "TOTAL", len(all_tweets))
    log.info("─" * 60)
    log.info("Phase 1 complete. Run Phase 2 next: python preprocess.py")


if __name__ == "__main__":
    main()
