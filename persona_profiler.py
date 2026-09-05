"""
persona_profiler.py — Phase 7: LLM Persona & Voice Profiler
============================================================
Synthesizes quantitative stylometric data (Phase 6) and high-engagement
sample tweets into a comprehensive natural-language Persona Profile and
production System Prompt for each critic.

Inputs:
    data/processed/{user_id}.json
    data/profiles/{user_id}_style_profile.json

Outputs:
    data/profiles/{user_id}_persona.json
    data/profiles/{user_id}_system_prompt.txt
    data/profiles/personas_summary.md
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    PROCESSED_DIR,
    PROFILES_DIR,
    LOG_LEVEL,
    GEMINI_MODEL,
)
from llm_client import generate, session_request_count

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

USERS = ["davidehrlich", "kevin_thecritic", "theinsneider"]


# ═════════════════════════════════════════════════════════════════════════════
#  SAMPLE REPRESENTATIVE TWEETS
# ═════════════════════════════════════════════════════════════════════════════

def select_representative_tweets(tweets: list[dict], n_top: int = 8, n_diverse: int = 6) -> list[dict]:
    """
    Select high-engagement and sentiment-diverse tweets to showcase
    the critic's quintessential voice.
    """
    # Exclude retweets and extremely short tweets (< 30 chars)
    candidates = [
        t for t in tweets
        if not t.get("is_retweet", False) and len(t.get("text", "")) >= 30
    ]

    # Sort by likes descending
    sorted_by_likes = sorted(
        candidates,
        key=lambda x: x.get("metrics", {}).get("likes", 0) if isinstance(x.get("metrics"), dict) else x.get("likes", 0),
        reverse=True,
    )
    top_tweets = sorted_by_likes[:n_top]
    selected_ids = {t["tweet_id"] for t in top_tweets}

    # Add diverse sentiment / categories
    diverse_tweets = []
    seen_categories = set()
    for t in candidates:
        if t["tweet_id"] in selected_ids:
            continue
        cat = t.get("category", "General")
        sent = t.get("sentiment", "Neutral")
        key = (cat, sent)
        if key not in seen_categories:
            diverse_tweets.append(t)
            seen_categories.add(key)
            if len(diverse_tweets) >= n_diverse:
                break

    all_sampled = top_tweets + diverse_tweets
    return all_sampled


# ═════════════════════════════════════════════════════════════════════════════
#  LLM PERSONA EXTRACTION PROMPT
# ═════════════════════════════════════════════════════════════════════════════

PERSONA_PROMPT_TEMPLATE = """You are an expert computational linguist and literary analyst specializing in Twitter voice cloning.
Analyze the provided stylometric statistics and sample tweets for film critic "@{user_id}".

--- QUANTITATIVE STYLOMETRIC SIGNATURE (PHASE 6 DATA) ---
{stylometric_summary}

--- SAMPLE TWEETS FROM @{user_id} ---
{sample_tweets_formatted}

--- OBJECTIVE ---
Synthesize a deep, precise persona profile in valid JSON format.
Your output must be strictly valid JSON matching this schema:

{{
  "user_id": "{user_id}",
  "display_name": "<full name or handle>",
  "role_and_background": "<1-2 sentences on who they are: public figure, tech founder, sports icon, journalist, film critic, or creator>",
  "personality_and_tone": [
    "<trait 1, e.g. swaggering, humorous, acerbic, visionary, street-smart>",
    "<trait 2>",
    "<trait 3>",
    "<trait 4>"
  ],
  "cultural_and_domain_worldview": "<What they care about, their core biases, cultural context, worldview, and industry stance>",
  "language_and_cultural_flavor": "<Linguistic profile: Natural Hinglish with colloquial Desi slang | Tech founder English | Punchy British criticism | Devanagari Hindi>",
  "stylistic_rules": [
    "<concrete writing rule 1, e.g. Start ~58% of tweets in lowercase>",
    "<concrete writing rule 2, e.g. Use parenthetical remarks or signature emojis>",
    "<concrete writing rule 3, e.g. Keep avg length around 27 words>",
    "<concrete writing rule 4, e.g. Code-mix Hindi and English naturally if Hinglish speaker>"
  ],
  "vocabulary_and_phrasing": {{
    "signature_catchphrases": ["<phrase 1>", "<phrase 2>", "<phrase 3>"],
    "characteristic_words": ["<word 1>", "<word 2>", "<word 3>", "<word 4>"],
    "banned_cliches": ["delve", "testament", "tapestry", "in a world where", "masterpiece (unless earned)", "needless to say"]
  }},
  "system_prompt": "<A complete, highly prescriptive, first-person system prompt that can be fed directly to an LLM to generate authentic tweets matching this exact persona. IMPORTANT: If this persona tweets in Hinglish, you MUST command the LLM to compose in authentic, natural Hinglish using their signature Roman Hindi vocabulary and colloquial Indian social media rhythm, NOT stiff formal English!>"
}}

Respond ONLY with valid JSON. Do not include markdown code fence formatting or other surrounding text.
"""


def format_stylometric_summary(style_profile: dict) -> str:
    """Format key metrics from Phase 6 style profile for prompt injection."""
    lex = style_profile.get("lexical_richness", {})
    flow = style_profile.get("length_and_flow", {})
    cap = style_profile.get("capitalization_habits", {})
    punct = style_profile.get("punctuation_fingerprint", {})
    gram = style_profile.get("grammatical_profile", {})
    soc = style_profile.get("social_microblogging", {})
    ngram = style_profile.get("distinctive_ngrams", [])

    top_ngrams = [f"{item['phrase']} ({item['score']:.2f})" for item in ngram[:8]]

    summary = (
        f"- Avg words per tweet: {flow.get('avg_words_per_tweet', 25)}\n"
        f"- Avg chars per tweet: {flow.get('avg_chars_per_tweet', 140)}\n"
        f"- Type-Token Ratio (TTR): {lex.get('type_token_ratio_ttr', 0.2):.4f}\n"
        f"- Yule's K (lexical diversity): {lex.get('yules_characteristic_k', 65):.1f}\n"
        f"- Starts with lowercase: {cap.get('starts_with_lowercase_rate', 0.0)*100:.1f}%\n"
        f"- Starts with capital: {cap.get('starts_with_capital_rate', 1.0)*100:.1f}%\n"
        f"- Top ALL-CAPS words: {', '.join(cap.get('top_all_caps_words', [])[:5])}\n"
        f"- Parentheses tweet rate: {punct.get('parentheses_tweet_rate', 0.0)*100:.1f}%\n"
        f"- Ellipses per tweet: {punct.get('ellipses_per_tweet', 0.0):.3f}\n"
        f"- Exclamations per 100 words: {punct.get('exclamations_per_100_words', 0.0):.2f}\n"
        f"- Questions per 100 words: {punct.get('questions_per_100_words', 0.0):.2f}\n"
        f"- Emoji rate per tweet: {soc.get('emoji_rate_per_tweet', 0.0):.3f}\n"
        f"- Top distinctive n-grams: {', '.join(top_ngrams)}\n"
        f"- Pronoun bias (1st person): {gram.get('pronoun_perspective', {}).get('1st_person', 0.5)*100:.1f}%\n"
    )

    lang = style_profile.get("language_and_cultural_profile", {})
    if lang:
        summary += (
            f"- Primary language: {lang.get('primary_language', 'english')}\n"
            f"- Hinglish / Vernacular density: {lang.get('hinglish_density', 0.0)*100:.1f}%\n"
            f"- Code-mixing frequency: {lang.get('code_mixing_rate', 0.0)*100:.1f}%\n"
            f"- Top vernacular / Hindi / Hinglish tokens: {', '.join(lang.get('top_vernacular_tokens', []))}\n"
            f"- Vernacular adaptation active: {lang.get('is_vernacular_adapted', False)}\n"
        )
    return summary


def clean_json_response(raw_text: str) -> dict:
    """Safely parse JSON response from LLM."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback regex extraction of outermost JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def build_persona_for_user(user_id: str) -> dict:
    """Generate persona profile for a single user using Gemini."""
    log.info("\n%s", "─" * 60)
    log.info("BUILDING PERSONA: %s", user_id)
    log.info("─" * 60)

    tweets_path = PROCESSED_DIR / f"{user_id}.json"
    style_path = PROFILES_DIR / f"{user_id}_style_profile.json"

    if not tweets_path.exists():
        raise FileNotFoundError(f"Missing tweets file: {tweets_path}")
    if not style_path.exists():
        raise FileNotFoundError(f"Missing style profile: {style_path}")

    with open(tweets_path, "r", encoding="utf-8") as f:
        tweets = json.load(f)
    with open(style_path, "r", encoding="utf-8") as f:
        style_profile = json.load(f)

    # 1. Sample representative tweets
    sampled_tweets = select_representative_tweets(tweets, n_top=8, n_diverse=6)
    sample_lines = []
    for i, t in enumerate(sampled_tweets, 1):
        likes = t.get("metrics", {}).get("likes", 0) if isinstance(t.get("metrics"), dict) else t.get("likes", 0)
        sample_lines.append(f"[{i}] (Likes: {likes}, Cat: {t.get('category')}) {t.get('text')}")
    sample_text = "\n".join(sample_lines)

    # 2. Build prompt
    stylometric_summary = format_stylometric_summary(style_profile)
    prompt = PERSONA_PROMPT_TEMPLATE.format(
        user_id=user_id,
        stylometric_summary=stylometric_summary,
        sample_tweets_formatted=sample_text,
    )

    # 3. Call LLM
    log.info("Querying LLM for persona synthesis …")
    raw_response = generate(prompt, temperature=0.7)
    persona_data = clean_json_response(raw_response)

    # Add metadata
    persona_data["generated_at"] = datetime.now(timezone.utc).isoformat()
    persona_data["model_used"] = GEMINI_MODEL
    persona_data["sample_tweets_used"] = len(sampled_tweets)

    return persona_data


def generate_personas_markdown(personas: dict[str, dict]) -> str:
    """Create a clean markdown summary of all personas."""
    lines = [
        "# Persona Profiles & Voice Guidelines",
        "",
        f"*Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        "",
    ]

    for user_id, p in personas.items():
        lines.append(f"## `@{user_id}` — {p.get('display_name', user_id)}")
        lines.append(f"**Role:** {p.get('role_and_background')}")
        lines.append("")
        lines.append(f"**Tone Traits:** {', '.join(p.get('personality_and_tone', []))}")
        lines.append("")
        lines.append(f"**Cinematic Worldview:** {p.get('cinematic_worldview')}")
        lines.append("")
        lines.append("**Key Stylistic Rules:**")
        for rule in p.get("stylistic_rules", []):
            lines.append(f"- {rule}")
        lines.append("")
        vocab = p.get("vocabulary_and_phrasing", {})
        lines.append(f"**Characteristic Words:** {', '.join(vocab.get('characteristic_words', []))}")
        lines.append(f"**Banned Clichés:** {', '.join(vocab.get('banned_cliches', []))}")
        lines.append("")
        lines.append("### Compiled System Prompt")
        lines.append("```text")
        lines.append(p.get("system_prompt", "").strip())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 7: Persona Profiler")
    log.info("=" * 60)

    personas = {}

    for user_id in USERS:
        try:
            persona = build_persona_for_user(user_id)
            personas[user_id] = persona

            # Save individual persona JSON
            out_json = PROFILES_DIR / f"{user_id}_persona.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(persona, f, ensure_ascii=False, indent=2)
            log.info("  Saved persona JSON → %s", out_json)

            # Save raw system prompt
            out_prompt = PROFILES_DIR / f"{user_id}_system_prompt.txt"
            with open(out_prompt, "w", encoding="utf-8") as f:
                f.write(persona.get("system_prompt", ""))
            log.info("  Saved system prompt → %s", out_prompt)

            time.sleep(2.0)  # Pacing between users

        except Exception as e:
            log.error("Failed to build persona for %s: %s", user_id, e)

    # Save combined personas markdown
    summary_md = generate_personas_markdown(personas)
    summary_path = PROFILES_DIR / "personas_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    log.info("\nPersonas summary saved → %s", summary_path)

    log.info("\n%s", "=" * 60)
    log.info("Phase 7 complete! Total requests this session: %d", session_request_count())
    log.info("Run Phase 8 next: python dynamic_prompt.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
