"""
rerank.py — Phase 11: Multi-Candidate Generation & Stylometric Reranker
=======================================================================
Implements the Best-of-N candidate generation strategy:
  1. Generates N diverse candidate tweets (varying temperature / phrasing)
  2. Evaluates each candidate against the critic's Phase 6 stylometric fingerprint
  3. Computes a multi-factor Stylometric Alignment Score (0 - 100)
  4. Reranks and selects the optimal "Champion" tweet

Outputs:
  - Ranked candidates with score breakdowns
  - Champion tweet selected for production output
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    PROFILES_DIR,
    LOG_LEVEL,
    NUM_CANDIDATES,
)
from llm_client import generate
from dynamic_prompt import (
    load_embedding_model,
    get_chroma_collection,
    retrieve_few_shots,
    build_dynamic_prompt,
)
from classifier import classify_topic_intent
from ground import get_factual_grounding, format_grounding_for_prompt

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def score_candidate(
    candidate_text: str,
    user_id: str,
    persona: dict,
    style: dict,
    topic: str,
) -> dict[str, Any]:
    """
    Score a generated candidate against the ground-truth stylometric profile.
    Returns composite score (0-100) and breakdown of sub-scores.
    """
    text = candidate_text.strip().strip('"')
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # 1. Target Length Score (max 25 pts)
    target_words = style.get("length_and_flow", {}).get("avg_words_per_tweet", 28)
    word_diff = abs(word_count - target_words)
    length_score = max(0.0, 25.0 - (word_diff * 1.5))

    # 2. Capitalization Habits Score (max 20 pts)
    starts_lower_rate = style.get("capitalization_habits", {}).get("starts_with_lowercase_rate", 0.0)
    first_alpha = next((c for c in text if c.isalpha()), None)
    is_lower_start = first_alpha.islower() if first_alpha else False

    if starts_lower_rate > 0.4:  # e.g. David Ehrlich
        cap_score = 20.0 if is_lower_start else 8.0
    else:  # e.g. Kevin or TheInSneider
        cap_score = 20.0 if not is_lower_start else 0.0

    # Extra bonus for ALL-CAPS words for Sneider
    if user_id == "theinsneider" and any(w.isupper() and len(w) > 1 for w in words):
        cap_score = min(20.0, cap_score + 5.0)

    # 3. Signature Punctuation Score (max 20 pts)
    punct_score = 10.0
    punct_profile = style.get("punctuation_fingerprint", {})

    if punct_profile.get("parentheses_tweet_rate", 0.0) > 0.12 and ("(" in text and ")" in text):
        punct_score += 10.0  # Ehrlich bonus
    if punct_profile.get("ellipses_per_tweet", 0.0) > 0.25 and ("..." in text or "…" in text):
        punct_score += 10.0  # Sneider bonus
    if user_id == "kevin_thecritic" and re.search(r"\b\d(?:\.\d)?/5\b", text):
        punct_score += 10.0  # Kevin rating bonus

    punct_score = min(20.0, punct_score)

    # 4. Vocabulary & Signature Phrases Score (max 20 pts)
    char_words = persona.get("vocabulary_and_phrasing", {}).get("characteristic_words", [])
    matched_words = [w for w in char_words if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)]
    vocab_score = min(20.0, 10.0 + (len(matched_words) * 5.0))

    # 5. Anti-Cliché Penalty (penalty up to -30 pts)
    banned = persona.get("vocabulary_and_phrasing", {}).get("banned_cliches", [
        "delve", "testament", "tapestry", "in a world where", "masterpiece", "needless to say"
    ])
    cliche_hits = [b for b in banned if re.search(rf"\b{re.escape(b)}\b", text, re.IGNORECASE)]
    cliche_penalty = len(cliche_hits) * 15.0

    # 6. Character Limit Guardrail
    length_penalty = 0.0
    if char_count > 280:
        length_penalty = 50.0  # disqualify tweets over Twitter limit

    composite_score = round(
        max(0.0, min(100.0, length_score + cap_score + punct_score + vocab_score - cliche_penalty - length_penalty)),
        1
    )

    return {
        "text": text,
        "word_count": word_count,
        "char_count": char_count,
        "composite_score": composite_score,
        "breakdown": {
            "length_score": round(length_score, 1),
            "cap_score": round(cap_score, 1),
            "punct_score": round(punct_score, 1),
            "vocab_score": round(vocab_score, 1),
            "cliche_penalty": cliche_penalty,
            "matched_characteristic_words": matched_words,
            "cliche_violations": cliche_hits,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MULTI-CANDIDATE GENERATION & SELECTION
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
#  EDITORIAL ANGLE MATRIX
# ═════════════════════════════════════════════════════════════════════════════

EDITORIAL_ANGLES = [
    {
        "id": "hot_take",
        "name": "The Hot Take",
        "tagline": "Unfiltered, provocative, and passionately bold",
        "temp": 0.92,
        "directive": (
            "EDITORIAL ANGLE: The Unfiltered Hot Take.\n"
            "- Take a bold, sharp, or passionate stance on this topic.\n"
            "- Do not hedge or write diplomatic corporate PR.\n"
            "- Let the emotional core of the persona shine through with authentic intensity."
        ),
    },
    {
        "id": "signature_classic",
        "name": "The Signature Classic",
        "tagline": "Quintessential voice and natural comfort zone",
        "temp": 0.82,
        "directive": (
            "EDITORIAL ANGLE: The Signature Classic.\n"
            "- Write in the author's quintessential comfort-zone style.\n"
            "- Perfectly balance their habitual sentence structure, perspective, and signature phrasing.\n"
            "- This should feel like the exact tweet this person would post today."
        ),
    },
    {
        "id": "deep_analysis",
        "name": "The Deep Analysis",
        "tagline": "Fact-dense, authoritative commentary",
        "temp": 0.70,
        "directive": (
            "EDITORIAL ANGLE: The Deep Analysis.\n"
            "- Integrate specific verified facts and temporal context seamlessly.\n"
            "- Thoughtful, substantive, and authoritative.\n"
            "- Avoid generic fluff—ground the tweet in actual real-world specifics."
        ),
    },
    {
        "id": "punchy_hook",
        "name": "The Punchy One-Liner",
        "tagline": "Viral, witty, maximum retweet potential",
        "temp": 0.88,
        "directive": (
            "EDITORIAL ANGLE: The Punchy One-Liner.\n"
            "- Strictly concise: MAXIMUM 18 words.\n"
            "- A razor-sharp hook, devastating punchline, or memorable catchphrase.\n"
            "- Designed for high virality and instant retweets."
        ),
    },
]


def transform_tweet(tweet_text: str, user_id: str, action: str) -> str:
    """Apply targeted transformation to an existing tweet while preserving author DNA."""
    p_path = PROFILES_DIR / f"{user_id}_persona.json"
    with open(p_path, "r", encoding="utf-8") as f:
        persona = json.load(f)

    system_prompt = persona.get("system_prompt", "")

    if action == "spicier":
        instruction = "Rewrite this tweet to make it significantly spicier, edgier, and more provocative with stronger emotional bite. Keep the author's voice intact."
    elif action == "hinglish":
        instruction = "Rewrite this tweet to infuse natural, authentic Hinglish (Roman Hindi + English) and colloquial Desi slang (bhai, kya scene hai, ekdum, yaar, mast) while preserving the core message."
    elif action == "shorten":
        instruction = "Trim this tweet down into a punchy, high-impact one-liner under 15 words. Maximum bite, zero fluff."
    else:
        instruction = "Refine this tweet for maximum authenticity and engagement."

    prompt = (
        f"{system_prompt}\n\n"
        f"Original Tweet: \"{tweet_text}\"\n\n"
        f"Task: {instruction}\n"
        f"Respond with ONLY the rewritten tweet text in quotes or plain text, no commentary."
    )
    res = generate(prompt, temperature=0.85)
    return res.strip().strip('"')


def generate_and_rerank(
    topic: str,
    user_id: str,
    embed_model,
    collection,
    n_candidates: int = 4,
    language_blend: str = "auto",
) -> dict[str, Any]:
    """
    End-to-end Editorial Angle Matrix Pipeline:
      1. Classifies intent (Phase 10)
      2. Injects real-time factual grounding (Phase 9)
      3. Injects dynamic few-shots & language directives (Phase 8)
      4. Generates across the 4 purposeful Editorial Angles
      5. Evaluates and scores each angle with stylometric DNA match
    """
    p_path = PROFILES_DIR / f"{user_id}_persona.json"
    s_path = PROFILES_DIR / f"{user_id}_style_profile.json"
    with open(p_path, "r", encoding="utf-8") as f:
        persona = json.load(f)
    with open(s_path, "r", encoding="utf-8") as f:
        style = json.load(f)

    # 1. Classification & Routing (Phase 10)
    intent = classify_topic_intent(topic, user_id)

    # 2. Factual Grounding (Phase 9)
    grounding = get_factual_grounding(topic)
    grounding_text = format_grounding_for_prompt(grounding)

    # 3. Dynamic Prompt (Phase 8)
    few_shots = retrieve_few_shots(topic, user_id, embed_model, collection, top_k=5)
    base_prompt = build_dynamic_prompt(user_id, topic, few_shots, persona, style)

    # 4. Language & Cultural Directive
    lang_profile = style.get("language_and_cultural_profile", {})
    detected_lang = lang_profile.get("primary_language", "english")
    active_lang = language_blend if language_blend and language_blend != "auto" else detected_lang

    language_directive = ""
    if active_lang == "hinglish":
        vernacular_tokens = ", ".join(lang_profile.get("top_vernacular_tokens", ["bhai", "yaar", "kya", "nahi", "ekdum", "mast", "scene"]))
        language_directive = (
            f"\n=== CRITICAL LINGUISTIC DIRECTIVE (AUTHENTIC HINGLISH ESSENCE) ===\n"
            f"- Compose naturally in authentic Hinglish (Roman Hindi code-mixed with English).\n"
            f"- Seamlessly use Indian internet conversational markers and slang (e.g. {vernacular_tokens}).\n"
            f"- DO NOT translate into formal or textbook Western English. Sound 100% natural to Indian social media culture.\n"
        )
    elif active_lang == "hindi":
        language_directive = (
            "\n=== CRITICAL LINGUISTIC DIRECTIVE (HINDI ESSENCE) ===\n"
            "- Write authentic Hindi in Devanagari script matching the persona's tone and stance.\n"
        )

    # Combine Base Prompt + Factual Grounding + Language Directive + Intent
    enhanced_prompt = base_prompt
    if grounding_text:
        enhanced_prompt = f"{grounding_text}\n\n{enhanced_prompt}"
    if language_directive:
        enhanced_prompt += language_directive
    if intent["routing_guidelines"]:
        routing_block = "\n".join(f"- {g}" for g in intent["routing_guidelines"])
        enhanced_prompt += f"\n=== INTENT DIRECTIVES ===\n{routing_block}\n"

    # 5. Generate across Editorial Angles
    selected_angles = EDITORIAL_ANGLES[:n_candidates]
    angle_results = []

    for idx, angle in enumerate(selected_angles, 1):
        angle_prompt = (
            f"{enhanced_prompt}\n"
            f"=== {angle['name'].upper()} ===\n"
            f"{angle['directive']}\n"
            f"Remember: Output ONLY the tweet text. No hashtags unless customary for @{user_id}."
        )

        try:
            cand_text = generate(angle_prompt, temperature=angle["temp"])
            cand_score = score_candidate(cand_text, user_id, persona, style, topic)
            cand_score["candidate_id"] = idx
            cand_score["angle_id"] = angle["id"]
            cand_score["angle_name"] = angle["name"]
            cand_score["angle_tagline"] = angle["tagline"]
            cand_score["temperature"] = angle["temp"]
            cand_score["viral_potential"] = min(98.0, round(cand_score["composite_score"] * 0.85 + (20.0 if angle["id"] in {"hot_take", "punchy_hook"} else 10.0), 1))
            cand_score["authenticity_score"] = cand_score["composite_score"]
            angle_results.append(cand_score)
        except Exception as e:
            log.warning("Angle '%s' generation failed: %s", angle['id'], e)

    if not angle_results:
        raise RuntimeError(f"All editorial angle generations failed for {user_id}")

    # Rank descending by composite alignment score
    ranked_angles = sorted(angle_results, key=lambda x: x["composite_score"], reverse=True)
    champion = ranked_angles[0]

    return {
        "user_id": user_id,
        "topic": topic,
        "category": intent["category"],
        "sentiment_intent": intent["sentiment_intent"],
        "language_blend": active_lang,
        "champion_tweet": champion["text"],
        "champion_score": champion["composite_score"],
        "champion_angle": champion["angle_name"],
        "champion_metrics": {
            "words": champion["word_count"],
            "chars": champion["char_count"],
        },
        "editorial_angles": angle_results,
        "all_ranked_candidates": ranked_angles,
        "language_profile": lang_profile,
    }


def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 11: Multi-Candidate Reranker")
    log.info("=" * 60)

    embed_model = load_embedding_model()
    collection = get_chroma_collection()

    test_topic = "The Brutalist and its 3.5 hour runtime"
    for user_id in ["davidehrlich", "kevin_thecritic", "theinsneider"]:
        log.info("\nGenerating & Reranking 3 candidates for @%s …", user_id)
        result = generate_and_rerank(test_topic, user_id, embed_model, collection, n_candidates=3)
        log.info("  Champion Score: %.1f / 100", result["champion_score"])
        log.info("  Champion Tweet: \"%s\"", result["champion_tweet"])
        for c in result["all_ranked_candidates"]:
            log.info("    [Cand #%d | Score: %.1f | Temp: %.2f] \"%s\"",
                     c["candidate_id"], c["composite_score"], c["temperature"], c["text"][:70])

    log.info("\n%s", "=" * 60)
    log.info("Phase 11 complete! Run Phase 12/13 next: python evaluate.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
