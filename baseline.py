"""
baseline.py — Phase 3: Stupidly Simple Baseline (Baseline 0)
=============================================================
Before RAG. Before embeddings. Before persona.

Just:
    Topic → LLM → Tweet

This is Baseline 0. Every future improvement must beat this.
Save outputs so we can compare later in Phase 12/13.

Run:
    python baseline.py

Output:
    data/baselines/baseline0_results.json
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GENERATION_TEMP,
    MAX_OUTPUT_TOKENS,
    DATA_DIR,
    LOG_LEVEL,
)

load_dotenv(override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
BASELINE_DIR = DATA_DIR / "baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

# ── Users in our dataset ─────────────────────────────────────────────────────
USERS = ["davidehrlich", "kevin_thecritic", "theinsneider"]

# ── Test topics (diverse — film + general, to stress-test generalization) ─────
TEST_TOPICS = [
    "Oppenheimer's use of practical effects and IMAX filmmaking",
    "Whether streaming has killed the theatrical experience",
    "The Brutalist and its 3.5 hour runtime",
    "Why superhero fatigue is real",
    "Dune Part Two's desert cinematography",
    "The Oscar nominations being predictable this year",
]


# ═════════════════════════════════════════════════════════════════════════════
#  GEMINI CLIENT
# ═════════════════════════════════════════════════════════════════════════════

def make_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


# ═════════════════════════════════════════════════════════════════════════════
#  BASELINE 0 PROMPT
#  As simple as possible — just topic, no user context.
#  This is intentionally bad. That's the point.
# ═════════════════════════════════════════════════════════════════════════════

BASELINE_PROMPT = """\
Write a single complete tweet (under 280 characters) about the following topic.

Topic: {topic}

Rules:
- MUST be a complete sentence — do NOT cut off mid-thought
- Under 280 characters total
- No hashtags unless they feel completely natural
- Sound like a film critic or movie enthusiast with a strong opinion
- Output ONLY the tweet text, nothing else. No quotes around it.
"""


def generate_baseline_tweet(client: genai.Client, topic: str) -> dict:
    """
    Generate one tweet with zero user context (pure LLM, topic only).
    Returns a result dict with the generated text and metadata.
    """
    prompt = BASELINE_PROMPT.format(topic=topic)

    start = time.time()
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=GENERATION_TEMP,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        elapsed = round(time.time() - start, 2)
        text = response.text.strip()

        # Check finish reason — warn if truncated
        try:
            finish_reason = response.candidates[0].finish_reason
            if str(finish_reason) not in ("FinishReason.STOP", "STOP", "1"):
                log.warning("Finish reason: %s (may be truncated)", finish_reason)
        except Exception:
            pass

        # Strip surrounding quotes if LLM added them
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()

        return {
            "status":      "ok",
            "text":        text,
            "char_count":  len(text),
            "elapsed_s":   elapsed,
            "model":       GEMINI_MODEL,
        }

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        log.error("Generation failed for topic '%s': %s", topic, e)
        return {
            "status":    "error",
            "error":     str(e),
            "elapsed_s": elapsed,
            "model":     GEMINI_MODEL,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 3: Baseline 0 (LLM only)")
    log.info("=" * 60)

    client = make_client()
    results = []

    for topic in TEST_TOPICS:
        log.info("\nTopic: %s", topic)
        # Generate once per topic (no user — this is user-agnostic baseline)
        result = generate_baseline_tweet(client, topic)

        entry = {
            "topic":    topic,
            "baseline": "baseline_0",
            "result":   result,
        }
        results.append(entry)

        if result["status"] == "ok":
            log.info("  Generated (%d chars, %.2fs):", result["char_count"], result["elapsed_s"])
            log.info("  → %s", result["text"])
        else:
            log.warning("  FAILED: %s", result.get("error", "unknown"))

        # Polite rate limit pause
        time.sleep(0.5)

    # ── Save results ──────────────────────────────────────────────────────
    out_path = BASELINE_DIR / "baseline0_results.json"
    output = {
        "baseline":      "baseline_0",
        "description":   "LLM only — no user context, no RAG, no persona",
        "model":         GEMINI_MODEL,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "total_topics":  len(TEST_TOPICS),
        "results":       results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("\n%s", "=" * 60)
    log.info("Baseline 0 saved → %s", out_path)
    log.info("=" * 60)
    log.info("")
    log.info("WHAT THIS PROVES:")
    log.info("  The LLM can write coherent film tweets but has NO idea")
    log.info("  who 'you' are. Every improvement from Phase 4 onward")
    log.info("  must beat this generic output.")
    log.info("")
    log.info("Next: python embed.py  (Phase 4 — embeddings + ChromaDB)")


if __name__ == "__main__":
    main()
