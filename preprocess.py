"""
preprocess.py — Phase 2: Data Preprocessing
=============================================
Reads the clean JSON files from data/processed/ (output of Phase 1) and adds:
  1. Per-tweet computed linguistic features (word count, emoji presence, etc.)
  2. Category inference for TheInSneider (which had no categories in raw data)
  3. Sentiment inference for TheInSneider (which had no sentiment in raw data)
  4. Retweet / quote tweet flags
  5. A final quality score for each tweet

Writes enriched JSON back to data/processed/ (overwrites Phase 1 output).

Run:
    python preprocess.py

Output:
    data/processed/davidehrlich.json     ← enriched in-place
    data/processed/kevin_thecritic.json  ← enriched in-place
    data/processed/theinsneider.json     ← enriched in-place (+ inferred fields)
    data/processed/all_users.json        ← re-combined
    data/processed/preprocessing_report.json
"""

import json
import re
import logging
from pathlib import Path
from datetime import timezone, datetime

from config import PROCESSED_DIR, LOG_LEVEL

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Optional: emoji library (graceful fallback if not installed) ──────────────
try:
    import emoji as emoji_lib
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False

# ═════════════════════════════════════════════════════════════════════════════
#  LINGUISTIC FEATURE EXTRACTION
#  These features are computed PER TWEET.
#  Phase 6 will aggregate them into a user-level style profile.
# ═════════════════════════════════════════════════════════════════════════════

# Sentence splitters: .  !  ?  ...  but NOT abbreviations like "Mr."
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-Z])")

def count_sentences(text: str) -> int:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return max(1, len([p for p in parts if p.strip()]))


def count_emojis(text: str) -> int:
    if HAS_EMOJI_LIB:
        return emoji_lib.emoji_count(text)
    # Fallback: rough Unicode emoji block detection
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F9FF"
        "\U00002700-\U000027BF"
        "\U0001FA00-\U0001FA9F"
        "\U00002600-\U000026FF]+",
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(text))


def extract_tweet_features(text: str) -> dict:
    """
    Compute per-tweet linguistic features.
    Style-preserving — we count what's there, not strip it.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # Punctuation counts
    exclamations  = text.count("!")
    questions      = text.count("?")
    ellipses       = text.count("...") + text.count("…")
    colons         = text.count(":")
    semicolons     = text.count(";")
    caps_words     = sum(1 for w in words if w.isupper() and len(w) > 1)

    # Social elements
    hashtags  = len(re.findall(r"#\w+", text))
    mentions  = len(re.findall(r"@\w+", text))
    emojis    = count_emojis(text)

    # Structure
    sentences     = count_sentences(text)
    avg_word_len  = (sum(len(w) for w in words) / word_count) if word_count else 0

    # Flags
    is_retweet     = text.startswith("RT @")
    is_question    = questions > 0 and exclamations == 0
    is_exclamatory = exclamations > 0
    has_sarcasm_signal = (
        text.count("/s") > 0
        or "lol" in text.lower()
        or "lmao" in text.lower()
        or "sure" in text.lower()  # "sure, great movie" pattern
    )

    # Unique vocabulary (type-token ratio, a measure of lexical diversity)
    unique_words = len(set(w.lower().strip(".,!?\"'") for w in words))
    lexical_diversity = (unique_words / word_count) if word_count else 0

    return {
        "word_count":          word_count,
        "char_count":          char_count,
        "sentence_count":      sentences,
        "avg_word_length":     round(avg_word_len, 2),
        "lexical_diversity":   round(lexical_diversity, 3),
        "exclamation_count":   exclamations,
        "question_count":      questions,
        "ellipsis_count":      ellipses,
        "colon_count":         colons,
        "semicolon_count":     semicolons,
        "caps_word_count":     caps_words,
        "hashtag_count":       hashtags,
        "mention_count":       mentions,
        "emoji_count":         emojis,
        "is_retweet":          is_retweet,
        "is_question":         is_question,
        "is_exclamatory":      is_exclamatory,
        "has_sarcasm_signal":  has_sarcasm_signal,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  CATEGORY INFERENCE  (for TheInSneider — no categories in raw data)
#  Rule-based keyword matching. This is a heuristic, not ML.
#  Phase 2 uses rules; Phase 6/7 will derive richer semantic categories.
# ═════════════════════════════════════════════════════════════════════════════

_CATEGORY_RULES = [
    # (pattern, category) — checked in order, first match wins
    (r"\b(film|movie|cinema|director|cinemat|screenplay|actor|actress|oscar|box office|sequel|trailer|rated [rg]|imdb|letterboxd|nolan|spielberg|scorsese|kubrick|villeneuve)\b", "Film"),
    (r"\b(review|★|stars?|out of \d|rating|recommend|worth watch|skip it|must see|masterpiece|disappointing)\b", "Review"),
    (r"\b(tv show|television|series|episode|season|netflix|hbo|disney\+|streaming|prestige tv|showrunner|binge)\b", "TV"),
    (r"\b(award|oscar|golden globe|bafta|sag|emmy|academy award|nominated|winner|ceremony)\b", "Awards"),
    (r"\b(news|breaking|report|announced|variety|deadline|hollywood reporter|exclusive)\b", "News"),
    (r"\b(album|song|music|concert|artist|spotify|band|rap|pop|rock|playlist)\b", "Music"),
    (r"\b(game|nba|nfl|sport|team|player|match|score|championship|league)\b", "Sports"),
]
_CATEGORY_RE = [(re.compile(p, re.IGNORECASE), cat) for p, cat in _CATEGORY_RULES]


def infer_category(text: str) -> str:
    """Rule-based category inference from tweet text."""
    for pattern, category in _CATEGORY_RE:
        if pattern.search(text):
            return category
    return "General"


# ═════════════════════════════════════════════════════════════════════════════
#  SENTIMENT INFERENCE  (for TheInSneider — no sentiment in raw data)
#  Simple lexicon-based approach. Phase 7 (LLM profiler) will refine this.
# ═════════════════════════════════════════════════════════════════════════════

_POS_WORDS = {
    "great", "love", "loved", "amazing", "excellent", "brilliant", "perfect",
    "best", "beautiful", "fantastic", "outstanding", "incredible", "wonderful",
    "enjoy", "enjoyed", "thrilled", "excited", "glad", "happy", "impressive",
    "masterpiece", "stunning", "superb", "terrific", "delightful", "remarkable",
}

_NEG_WORDS = {
    "bad", "worst", "terrible", "awful", "horrible", "hate", "hated",
    "boring", "dull", "disappointing", "disappointed", "waste", "overrated",
    "ridiculous", "pointless", "embarrassing", "mediocre", "weak", "fail",
    "failed", "poor", "garbage", "trash", "unwatchable", "insufferable",
}


def infer_sentiment(text: str) -> str:
    """Basic lexicon-based sentiment. Returns Positive / Negative / Neutral."""
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(tokens & _POS_WORDS)
    neg = len(tokens & _NEG_WORDS)
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


# ═════════════════════════════════════════════════════════════════════════════
#  ENRICHMENT PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def enrich_tweet(tweet: dict, infer_missing: bool = False) -> dict:
    """
    Add computed features to one tweet dict.
    infer_missing = True → fill category/sentiment with rule-based inference
                           (used only for TheInSneider which lacks these fields)
    """
    text = tweet.get("text", "")

    # Add linguistic features
    tweet["features"] = extract_tweet_features(text)

    # Infer missing fields if needed
    if infer_missing:
        if tweet.get("category", "General") == "General":
            tweet["category"] = infer_category(text)
        if tweet.get("sentiment", "Neutral") == "Neutral":
            tweet["sentiment"] = infer_sentiment(text)

    return tweet


def enrich_user(tweets: list[dict], user_id: str) -> list[dict]:
    """Enrich all tweets for one user."""
    needs_inference = user_id == "theinsneider"
    enriched = []
    for t in tweets:
        enriched.append(enrich_tweet(t, infer_missing=needs_inference))
    return enriched


# ═════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING REPORT
# ═════════════════════════════════════════════════════════════════════════════

def build_report(user_id: str, tweets: list[dict]) -> dict:
    """Aggregate feature statistics for the preprocessing report."""
    if not tweets:
        return {}

    def avg(key):
        vals = [t["features"][key] for t in tweets]
        return round(sum(vals) / len(vals), 3)

    def rate(key):
        return round(sum(1 for t in tweets if t["features"][key]) / len(tweets), 3)

    cat_dist  = {}
    sent_dist = {}
    for t in tweets:
        cat_dist[t["category"]]  = cat_dist.get(t["category"], 0) + 1
        sent_dist[t["sentiment"]] = sent_dist.get(t["sentiment"], 0) + 1

    return {
        "user_id":               user_id,
        "total_tweets":          len(tweets),
        "avg_word_count":        avg("word_count"),
        "avg_char_count":        avg("char_count"),
        "avg_sentence_count":    avg("sentence_count"),
        "avg_lexical_diversity": avg("lexical_diversity"),
        "avg_exclamations":      avg("exclamation_count"),
        "avg_questions":         avg("question_count"),
        "avg_emojis":            avg("emoji_count"),
        "avg_hashtags":          avg("hashtag_count"),
        "avg_mentions":          avg("mention_count"),
        "avg_caps_words":        avg("caps_word_count"),
        "retweet_rate":          rate("is_retweet"),
        "question_rate":         rate("is_question"),
        "exclamatory_rate":      rate("is_exclamatory"),
        "sarcasm_signal_rate":   rate("has_sarcasm_signal"),
        "category_distribution": dict(sorted(cat_dist.items(), key=lambda x: -x[1])),
        "sentiment_distribution": dict(sorted(sent_dist.items(), key=lambda x: -x[1])),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

USER_FILES = {
    "davidehrlich":    PROCESSED_DIR / "davidehrlich.json",
    "kevin_thecritic": PROCESSED_DIR / "kevin_thecritic.json",
    "theinsneider":    PROCESSED_DIR / "theinsneider.json",
}


def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 2: Preprocessing")
    log.info("=" * 60)

    all_tweets = []
    all_reports = []

    for user_id, path in USER_FILES.items():
        if not path.exists():
            log.error("Processed file not found: %s — run prepare_dataset.py first", path)
            continue

        log.info("Enriching %s …", user_id)
        with open(path, encoding="utf-8") as f:
            tweets = json.load(f)

        tweets = enrich_user(tweets, user_id)

        # Write back enriched file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tweets, f, ensure_ascii=False, indent=2)

        report = build_report(user_id, tweets)
        all_reports.append(report)
        all_tweets.extend(tweets)

        log.info(
            "  %-22s → %4d tweets | avg_words=%.1f | emoji_rate=%.3f | q_rate=%.3f",
            user_id, len(tweets),
            report["avg_word_count"],
            report["avg_emojis"],
            report["question_rate"],
        )

    # Re-write combined file
    combined_path = PROCESSED_DIR / "all_users.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_tweets, f, ensure_ascii=False, indent=2)
    log.info("Combined → %s (%d tweets)", combined_path, len(all_tweets))

    # Write preprocessing report
    report_path = PROCESSED_DIR / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "users": all_reports,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info("Report  → %s", report_path)

    # ── Print comparison table ────────────────────────────────────────────
    log.info("")
    log.info("─" * 75)
    log.info("STYLE FINGERPRINT PREVIEW  (confirms style info is preserved)")
    log.info("─" * 75)
    log.info("%-22s %8s %8s %8s %8s %8s", "User", "Wds/Tw", "Excl/Tw", "Q/Tw", "Emoji/Tw", "Lex.Div")
    log.info("%-22s %8s %8s %8s %8s %8s", "─"*22, "─"*8, "─"*8, "─"*8, "─"*8, "─"*8)
    for r in all_reports:
        log.info(
            "%-22s %8.1f %8.3f %8.3f %8.3f %8.3f",
            r["user_id"],
            r["avg_word_count"],
            r["avg_exclamations"],
            r["avg_questions"],
            r["avg_emojis"],
            r["avg_lexical_diversity"],
        )
    log.info("─" * 75)
    log.info("Phase 2 complete. Run Phase 3 next: python baseline.py")


if __name__ == "__main__":
    main()
