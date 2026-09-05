"""
classifier.py — Phase 10: Topic Classification & Intent Routing
================================================================
Routes user inputs/topics into specific categories, intents, and persona-specific
formatting adjustments before generation.

Categories:
  - Review (film critiques, ratings, impressions)
  - News/Industry (scoops, box office, studio deals, casting)
  - Film/Culture (theatrical debates, auteur legacy, trends)
  - Humorous (satire, timeline jokes, cynical takes)

Outputs:
  - Classification dict with category, sentiment, entities, and routing flags
"""

import json
import logging
import re
from typing import Any

from config import LOG_LEVEL
from llm_client import generate

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Heuristic pattern matchers for instant classification
REVIEW_PATTERNS = [
    r"\b(review|rating|thoughts on|runtime|earned|masterpiece|movie|film|watch|watched|seen|pacing)\b",
]
NEWS_PATTERNS = [
    r"\b(scoop|breaking|exclusive|box office|gross|budget|studio|casting|cast as|firing|deal|newsletter|report)\b",
]
INDUSTRY_PATTERNS = [
    r"\b(oscar|oscars|academy award|nominations|predictable|festival|cannes|venice|sundance|streaming)\b",
]
HUMOROUS_PATTERNS = [
    r"\b(lol|lmao|slop|fatigue|meme|dying|suicide note|ruined|cringe|joke)\b",
]


def heuristic_classify(topic: str) -> dict[str, str]:
    """Fast, deterministic categorization using keyword heuristics."""
    lower = topic.lower()

    if any(re.search(p, lower) for p in NEWS_PATTERNS):
        cat = "News/Industry"
    elif any(re.search(p, lower) for p in INDUSTRY_PATTERNS):
        cat = "Industry/Awards"
    elif any(re.search(p, lower) for p in REVIEW_PATTERNS):
        cat = "Review"
    elif any(re.search(p, lower) for p in HUMOROUS_PATTERNS):
        cat = "Humorous"
    else:
        cat = "Film/Culture"

    # Sentiment inference
    if any(w in lower for w in ["fatigue", "killed", "predictable", "lazy", "slop", "ruined"]):
        sent = "Cynical/Critical"
    elif any(w in lower for w in ["practical effects", "staggering", "imax", "triumph", "brilliant", "cinematography"]):
        sent = "Positive/Awe"
    elif any(w in lower for w in ["runtime", "hours", "long"]):
        sent = "Mixed/Intrigue"
    else:
        sent = "Analytical"

    return {"category": cat, "sentiment_intent": sent}


def classify_topic_intent(topic: str, user_id: str | None = None) -> dict[str, Any]:
    """
    Classify a topic and generate persona-specific routing directives.
    Combines rule-based heuristics with targeted persona adaptations.
    """
    heuristics = heuristic_classify(topic)
    cat = heuristics["category"]
    sent = heuristics["sentiment_intent"]

    # Extract potential entities (capitalized words or known phrases)
    entities = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", topic)
    entities = [e for e in entities if e.lower() not in {"the", "why", "whether", "and", "its", "this", "year"}]

    # Persona-specific routing instructions
    guidelines = []
    recommended_prefix = ""

    if user_id == "theinsneider":
        if cat in ("News/Industry", "Industry/Awards"):
            guidelines.append("Use insider scoop framing; reference sources or studio presentations.")
            guidelines.append("Use trailing ellipses (...) and ALL-CAPS lead-in tags.")
            recommended_prefix = "HOT RUMOR: " if "fatigue" in topic.lower() or "runtime" in topic.lower() else "EXCLUSIVE: "
        else:
            guidelines.append("Deliver sharp, unvarnished insider reaction.")

    elif user_id == "kevin_thecritic":
        if cat in ("Review", "Film/Culture"):
            guidelines.append("Provide a balanced evaluation emphasizing emotional resonance and pacing.")
            guidelines.append("Conclude with a clear numerical rating (e.g. 3.5/5 or 4/5).")
            guidelines.append("Use first-person perspective ('my frame of reference', 'I felt').")
        else:
            guidelines.append("Maintain an earnest, measured perspective.")

    elif user_id == "davidehrlich":
        if cat in ("Review", "Film/Culture", "Humorous"):
            guidelines.append("Begin in lowercase without capitalizing the first word.")
            guidelines.append("Blend poetic, high-brow film critique with cynical deadpan wit.")
            guidelines.append("Deploy a sharp parenthetical aside or expressive casual adjective.")

    return {
        "topic": topic,
        "category": cat,
        "sentiment_intent": sent,
        "extracted_entities": entities,
        "user_id": user_id,
        "recommended_prefix": recommended_prefix,
        "routing_guidelines": guidelines,
    }


def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 10: Topic Classifier & Routing")
    log.info("=" * 60)

    test_cases = [
        ("The Brutalist and its 3.5 hour runtime", "kevin_thecritic"),
        ("Oppenheimer's use of practical effects and IMAX filmmaking", "davidehrlich"),
        ("Why superhero fatigue is real", "theinsneider"),
        ("The Oscar nominations being predictable this year", "theinsneider"),
        ("Whether streaming has killed the theatrical experience", "davidehrlich"),
    ]

    for topic, user in test_cases:
        res = classify_topic_intent(topic, user)
        log.info("\nTopic: '%s'", topic)
        log.info("  User: @%s | Category: %s | Sentiment: %s", user, res["category"], res["sentiment_intent"])
        log.info("  Entities: %s", res["extracted_entities"])
        if res["recommended_prefix"]:
            log.info("  Recommended Prefix: '%s'", res["recommended_prefix"])
        for g in res["routing_guidelines"]:
            log.info("  Directive: -> %s", g)

    log.info("\n%s", "=" * 60)
    log.info("Phase 10 complete! Run Phase 11 next: python rerank.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
