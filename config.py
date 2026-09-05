"""
config.py — Central configuration for Tweet Like You.
All paths, model names, and constants live here.
Import this wherever you need settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Disable TensorFlow BEFORE any ML library imports ─────────────────────────
# sentence-transformers only needs PyTorch. TF conflicts with google-genai's
# protobuf requirements. Setting these here ensures they're in place early.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── Project root (this file's parent) ────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Directory structure ───────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
PROFILES_DIR    = DATA_DIR / "profiles"
CHROMA_DIR      = ROOT / "chroma_db"
LOGS_DIR        = ROOT / "logs"

# Ensure directories exist at import time
for _d in [RAW_DIR, PROCESSED_DIR, PROFILES_DIR, CHROMA_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw dataset files ─────────────────────────────────────────────────────────
RAW_FILES = {
    "davidehrlich":   RAW_DIR / "david.json",
    "kevin_thecritic": RAW_DIR / "kevin.json",
    "theinsneider":   RAW_DIR / "TheInSneider.json",
}

# ── API Keys (loaded from .env) ───────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
NEWS_API_KEY    = os.getenv("NEWS_API_KEY", "")
TMDB_API_KEY    = os.getenv("TMDB_API_KEY", "")   # optional
OMDB_API_KEY    = os.getenv("OMDB_API_KEY", "")   # optional

# ── LLM settings ─────────────────────────────────────────────────────────────
# Uses new google-genai SDK (google-generativeai is deprecated)
# gemini-3.5-flash-lite: higher free req/day  ← use this for batch jobs
# gemini-3.6-flash:      20  free req/day     ← use for best quality only
GEMINI_MODEL         = "gemini-3.5-flash-lite"  # high free quota
GEMINI_MODEL_PRO     = "gemini-3.6-flash"        # best quality, use sparingly
GENERATION_TEMP      = 0.85                 # default creativity
MAX_OUTPUT_TOKENS    = 2048                 # gemini-3.6-flash uses thinking tokens internally;
                                            # need headroom beyond the tweet itself
NUM_CANDIDATES       = 5                    # Phase 11: multi-candidate generation

# ── Embedding settings ────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_DIM        = 1024
CHROMA_COLLECTION    = "tweet_memories"

# ── RAG / retrieval settings ──────────────────────────────────────────────────
TOP_K_RETRIEVAL      = 15    # how many historical tweets to retrieve

# ── Category normalisation map (for kevin.json) ───────────────────────────────
CATEGORY_MAP = {
    # Film-related
    "film":              "Film",
    "movies":            "Film",
    "movie":             "Film",
    "movie review":      "Review",
    "film criticism":    "Review",
    "movie criticism":   "Review",
    "movie review":      "Review",
    "film discussion":   "Film",
    "movie discussion":  "Film",
    "film industry":     "Film",
    "tv/film industry":  "Film",
    "animation":         "Film",
    "movie/tv":          "TV",
    "tv/movie":          "TV",
    "tv/film":           "TV",
    "comic/tv":          "TV",
    # TV
    "tv":                "TV",
    "tv review":         "Review",
    # Other
    "general":           "General",
    "personal":          "Personal",
    "news":              "News",
    "music":             "Music",
    "sports":            "Sports",
    "awards":            "Awards",
    "podcast":           "General",
    "celebrity":         "General",
}

# ── TheInSneider user metadata (missing from raw file) ───────────────────────
INSNEIDER_META = {
    "user_id":  "theinsneider",
    "username": "@TheInSneider",
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
