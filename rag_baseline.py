"""
rag_baseline.py — Phase 5: RAG Baseline (Baseline 1)
======================================================
Adds semantic retrieval to the generation pipeline.

Pipeline:
    Topic → Embedding → ChromaDB → Top-K tweets → LLM → Generated tweet

This gives us Baseline 1: Semantic RAG.

Compare with Baseline 0 (LLM only) to see if retrieved context helps.
This is the first system that actually knows who the user is (by example).

Run:
    python rag_baseline.py

Output:
    data/baselines/baseline1_rag_results.json
"""

import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from llm_client import generate, session_request_count

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GENERATION_TEMP,
    MAX_OUTPUT_TOKENS,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    PROCESSED_DIR,
    DATA_DIR,
    TOP_K_RETRIEVAL,
    LOG_LEVEL,
)

load_dotenv(override=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASELINE_DIR = DATA_DIR / "baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

# ── Same test topics as Baseline 0 (needed for comparison) ───────────────────
TEST_TOPICS = [
    "Oppenheimer's use of practical effects and IMAX filmmaking",
    "Whether streaming has killed the theatrical experience",
    "The Brutalist and its 3.5 hour runtime",
    "Why superhero fatigue is real",
    "Dune Part Two's desert cinematography",
    "The Oscar nominations being predictable this year",
]

USERS = ["davidehrlich", "kevin_thecritic", "theinsneider"]


# ═════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS (cached — only loads once per run)
# ═════════════════════════════════════════════════════════════════════════════

def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    log.info("Loading embedding model …")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("  Loaded in %.1fs", time.time() - t0)
    return model


def get_chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=CHROMA_COLLECTION)
    log.info("ChromaDB: %d vectors loaded", collection.count())
    return collection


# ═════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL
# ═════════════════════════════════════════════════════════════════════════════

def retrieve_relevant_tweets(
    query: str,
    user_id: str,
    embed_model,
    collection,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Embed the query and retrieve the top-K most semantically similar tweets
    from the given user's history.
    """
    query_vec = embed_model.encode(
        [f"Represent this sentence for searching relevant passages: {query}"],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        where={"user_id": user_id},
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if results and results["documents"]:
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            retrieved.append({
                "text":       doc,
                "similarity": round(1.0 - dist, 4),
                "category":   meta.get("category", ""),
                "sentiment":  meta.get("sentiment", ""),
                "likes":      meta.get("likes", 0),
            })

    return retrieved


# ═════════════════════════════════════════════════════════════════════════════
#  PROMPT TEMPLATE
# ═════════════════════════════════════════════════════════════════════════════

RAG_PROMPT = """You are a tweet ghostwriter. Below are examples of how this specific person writes tweets, ordered by relevance to the topic:

--- PAST TWEETS FOR REFERENCE ---
{retrieved_tweets}
--- END PAST TWEETS ---

Now write a new tweet in THEIR voice, style, and tone about the following topic:

{topic}

INSTRUCTIONS:
- Write ONE complete tweet (under 280 characters)
- Sound exactly like the person above — match their voice, tone, and style
- Do NOT copy any past tweet directly — write something new
- Do NOT add hashtags unless the person uses them naturally
- Do NOT explain yourself — output ONLY the tweet text
"""


def format_retrieved_tweets(retrieved: list[dict]) -> str:
    """Format retrieved tweets for injection into the prompt."""
    lines = []
    for i, t in enumerate(retrieved, 1):
        lines.append(f"[{i}] (sim={t['similarity']:.2f}, likes={t['likes']}) {t['text']}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_rag_tweet(
    topic: str,
    user_id: str,
    embed_model,
    collection,
) -> dict:
    """Generate one tweet using RAG — topic + retrieved historical tweets."""

    # Step 1: Retrieve relevant context
    retrieved = retrieve_relevant_tweets(topic, user_id, embed_model, collection)

    # Step 2: Build prompt
    prompt = RAG_PROMPT.format(
        retrieved_tweets=format_retrieved_tweets(retrieved),
        topic=topic,
    )

    # Step 3: Generate via centralized llm_client
    start = time.time()
    try:
        text = generate(prompt)
        elapsed = round(time.time() - start, 2)

        return {
            "status":          "ok",
            "text":            text,
            "char_count":      len(text),
            "elapsed_s":       elapsed,
            "model":           GEMINI_MODEL,
            "top_k":           len(retrieved),
            "avg_similarity":  round(sum(r["similarity"] for r in retrieved) / len(retrieved), 4) if retrieved else 0.0,
            "retrieved_tweets": retrieved,
        }

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        log.error("Generation failed for user %s, topic '%s': %s", user_id, topic, e)
        return {
            "status":    "error",
            "error":     str(e),
            "elapsed_s": elapsed,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 5: Baseline 1 (RAG)")
    log.info("=" * 60)

    embed_model = load_embedding_model()
    collection  = get_chroma_collection()

    out_path = BASELINE_DIR / "baseline1_rag_results.json"
    existing_map = {}
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for entry in prev_data.get("results", []):
                    res = entry.get("result", {})
                    if res.get("status") == "ok" and res.get("text"):
                        existing_map[(entry.get("user_id"), entry.get("topic"))] = res
            if existing_map:
                log.info("Loaded %d previously successful generations (will skip)", len(existing_map))
        except Exception as e:
            log.warning("Could not load previous results: %s", e)

    all_results = []

    for user_id in USERS:
        log.info("\n%s", "─" * 60)
        log.info("USER: %s", user_id)
        log.info("─" * 60)

        for topic in TEST_TOPICS:
            log.info("\nTopic: %s", topic)
            if (user_id, topic) in existing_map:
                result = existing_map[(user_id, topic)]
                log.info("  [Cached] (%d chars, avg_sim=%.3f):",
                         result.get("char_count", len(result.get("text", ""))),
                         result.get("avg_similarity", 0.0))
                log.info("  → %s", result.get("text", ""))
            else:
                result = generate_rag_tweet(topic, user_id, embed_model, collection)
                if result["status"] == "ok":
                    log.info("  Generated (%d chars, %.2fs, avg_sim=%.3f):",
                             result["char_count"], result["elapsed_s"], result["avg_similarity"])
                    log.info("  → %s", result["text"])
                else:
                    log.warning("  FAILED: %s", result.get("error"))

                time.sleep(2.0)  # Rate pacing

            entry = {
                "user_id":  user_id,
                "topic":    topic,
                "baseline": "baseline_1_rag",
                "result":   result,
            }
            all_results.append(entry)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = BASELINE_DIR / "baseline1_rag_results.json"
    output = {
        "baseline":     "baseline_1_rag",
        "description":  "LLM + semantic RAG — top-15 relevant tweets injected per user",
        "model":        GEMINI_MODEL,
        "top_k":        TOP_K_RETRIEVAL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "users":        USERS,
        "topics":       TEST_TOPICS,
        "results":      all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("\n%s", "=" * 60)
    log.info("Baseline 1 (RAG) saved → %s", out_path)
    log.info("=" * 60)
    log.info("")
    log.info("NEXT STEPS:")
    log.info("  - Compare outputs with baseline0_results.json")
    log.info("  - Do RAG outputs sound more like each specific user?")
    log.info("  - Run Phase 6 next: python stylometry.py")


if __name__ == "__main__":
    main()
