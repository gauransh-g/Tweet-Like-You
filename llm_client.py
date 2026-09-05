"""
llm_client.py — Centralized Gemini LLM client
===============================================
Handles:
  - Rate limiting (auto-retry on 429 with exponential backoff)
  - Request tracking (warns when approaching daily quota)
  - Consistent config across all phases

Import this everywhere instead of calling genai directly.

Usage:
    from llm_client import generate

    text = generate("Write a tweet about Oppenheimer")
"""

import time
import logging
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GENERATION_TEMP,
    MAX_OUTPUT_TOKENS,
)

log = logging.getLogger(__name__)

# ── Singleton client ──────────────────────────────────────────────────────────
_client: genai.Client | None = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ── Request counter (in-memory, resets per run) ───────────────────────────────
_request_count = 0
DAILY_QUOTA_WARN = 15   # warn when this many requests made in one session


# ═════════════════════════════════════════════════════════════════════════════
#  CORE GENERATION FUNCTION WITH RETRY
# ═════════════════════════════════════════════════════════════════════════════

def generate(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    max_retries: int = 5,
    initial_wait: float = 30.0,
) -> str:
    """
    Generate text from Gemini with automatic retry on 429 rate limits.

    Args:
        prompt:           The full prompt string
        model:            Override model (default: config.GEMINI_MODEL)
        temperature:      Override temperature (default: config.GENERATION_TEMP)
        max_output_tokens: Override token limit (default: config.MAX_OUTPUT_TOKENS)
        max_retries:      How many times to retry on 429
        initial_wait:     Seconds to wait on first 429 (doubles each retry)

    Returns:
        Generated text string, stripped of surrounding whitespace/quotes.

    Raises:
        Exception if all retries exhausted or non-429 error.
    """
    global _request_count

    client      = get_client()
    use_model   = model or GEMINI_MODEL
    use_temp    = temperature if temperature is not None else GENERATION_TEMP
    use_tokens  = max_output_tokens or MAX_OUTPUT_TOKENS

    wait = initial_wait

    for attempt in range(max_retries + 1):
        try:
            _request_count += 1
            if _request_count >= DAILY_QUOTA_WARN:
                log.warning(
                    "⚠️  Request #%d this session — approaching daily quota. "
                    "Consider spacing calls.", _request_count
                )

            response = client.models.generate_content(
                model=use_model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=use_temp,
                    max_output_tokens=use_tokens,
                ),
            )

            text = response.text.strip()
            # Strip surrounding quotes if LLM wrapped output
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1].strip()

            return text

        except Exception as e:
            err_str = str(e)

            # 429 Rate limit → wait and retry
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Extract retry delay from error if available
                import re
                delay_match = re.search(r"retry[^\d]*(\d+\.?\d*)\s*s", err_str, re.IGNORECASE)
                suggested_wait = float(delay_match.group(1)) + 5 if delay_match else wait

                if attempt < max_retries:
                    log.warning(
                        "Rate limit hit (attempt %d/%d). Waiting %.0fs before retry …",
                        attempt + 1, max_retries, suggested_wait
                    )
                    time.sleep(suggested_wait)
                    wait = min(wait * 2, 120)   # exponential backoff, cap at 2 min
                    continue
                else:
                    log.error("Rate limit: all %d retries exhausted.", max_retries)
                    raise

            # Any other error → raise immediately
            log.error("LLM generation error: %s", e)
            raise

    raise RuntimeError("generate() exited retry loop without returning")


# ═════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: session stats
# ═════════════════════════════════════════════════════════════════════════════

def session_request_count() -> int:
    """Return number of LLM requests made this Python session."""
    return _request_count
