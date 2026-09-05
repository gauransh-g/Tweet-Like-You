"""
profile_builder.py — Automated On-The-Fly Persona Profiler & Vector Indexer
============================================================================
Given ANY newly ingested tweets for @any_user:
  1. Computes deep stylometrics (lexical richness, casing, punctuation, length)
  2. Synthesizes a custom Persona System Prompt via LLM
  3. Indexes the tweets into ChromaDB collection `tweet_memories`
  4. Saves all profile artifacts so the user is instantly selectable across the platform
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import spacy

from config import (
    PROCESSED_DIR,
    PROFILES_DIR,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    LOG_LEVEL,
)
from stylometry import analyze_user_corpus
from persona_profiler import build_persona_for_user
from account_ingestor import save_ingested_tweets

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_and_index_profile(
    user_id: str,
    tweets: list[dict[str, Any]],
    nlp: spacy.Language | None = None,
    embed_model=None,
    collection=None,
    status_callback=None,
) -> dict[str, Any]:
    """
    Complete on-the-fly profiling pipeline for any new account.
    Takes ~5-15 seconds.
    """
    clean_id = user_id.lstrip("@").lower().strip()
    log.info("=" * 60)
    log.info("PROFILING USER: @%s (%d tweets)", clean_id, len(tweets))
    log.info("=" * 60)

    # 1. Save ingested tweets
    if status_callback:
        status_callback("💾 Saving and normalizing tweets …")
    save_ingested_tweets(clean_id, tweets)

    # 2. Extract Stylometrics (Phase 6)
    if status_callback:
        status_callback("🧬 Analyzing stylometric DNA (lexical richness, pacing, punctuation) …")
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])

    style_profile = analyze_user_corpus(clean_id, tweets, nlp)
    style_out = PROFILES_DIR / f"{clean_id}_style_profile.json"
    with open(style_out, "w", encoding="utf-8") as f:
        json.dump(style_profile, f, ensure_ascii=False, indent=2)
    log.info("Saved style profile -> %s", style_out)

    # 3. Synthesize Persona & Voice Guidelines (Phase 7)
    if status_callback:
        status_callback("🎭 Synthesizing Persona System Prompt with AI …")
    persona_profile = build_persona_for_user(clean_id)
    persona_out = PROFILES_DIR / f"{clean_id}_persona.json"
    with open(persona_out, "w", encoding="utf-8") as f:
        json.dump(persona_profile, f, ensure_ascii=False, indent=2)

    prompt_out = PROFILES_DIR / f"{clean_id}_system_prompt.txt"
    with open(prompt_out, "w", encoding="utf-8") as f:
        f.write(persona_profile.get("system_prompt", ""))
    log.info("Saved persona -> %s", persona_out)

    # 4. Index into ChromaDB Semantic Vector Store (Phase 4)
    if status_callback:
        status_callback("🧠 Indexing memory vectors into ChromaDB …")
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(EMBEDDING_MODEL)

    if collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION)

    # Embed and upsert in batches
    texts = [t["text"] for t in tweets]
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    ids = [t["tweet_id"] for t in tweets]
    metadatas = [{
        "user_id": clean_id,
        "username": t.get("username", clean_id),
        "timestamp": t.get("timestamp", ""),
        "category": t.get("category", "General"),
        "sentiment": t.get("sentiment", "Neutral"),
        "likes": t.get("metrics", {}).get("likes", 0) if isinstance(t.get("metrics"), dict) else 0,
    } for t in tweets]

    # Delete existing entries for this user if re-indexing
    try:
        collection.delete(where={"user_id": clean_id})
    except Exception:
        pass

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    log.info("Indexed %d vectors for @%s in ChromaDB", len(ids), clean_id)

    if status_callback:
        status_callback("✅ Account successfully cloned and ready to tweet!")

    return {
        "user_id": clean_id,
        "display_name": persona_profile.get("display_name", clean_id),
        "tweets_count": len(tweets),
        "style_profile": style_profile,
        "persona": persona_profile,
    }


if __name__ == "__main__":
    sample_tech_tweets = [
        {"tweet_id": "t1", "user_id": "sama_clone", "text": "compute is the currency of the future and energy is the backing reserve."},
        {"tweet_id": "t2", "user_id": "sama_clone", "text": "most startups fail not because building the product is hard, but because finding 100 people who love it is."},
        {"tweet_id": "t3", "user_id": "sama_clone", "text": "superintelligence will happen much faster than institutions can update their mental models."},
        {"tweet_id": "t4", "user_id": "sama_clone", "text": "optimism is an underrated competitive advantage. cynical people rarely build the future."},
        {"tweet_id": "t5", "user_id": "sama_clone", "text": "the best career advice is to work with the smartest, most ambitious people on things that matter."},
    ]
    res = build_and_index_profile("sama_clone", sample_tech_tweets)
    print("Profile built successfully:", res["user_id"])
