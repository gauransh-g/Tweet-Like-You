"""
embed.py — Phase 4: Embeddings + Vector Database
==================================================
Reads processed tweets from data/processed/ and:
  1. Loads mxbai-embed-large-v1 (1024-dim embedding model)
  2. Embeds every tweet's text
  3. Stores vectors + metadata in ChromaDB (local)
  4. Runs a quick sanity-check retrieval test

This is the semantic memory layer for RAG (Phase 5).

Run:
    python embed.py

Output:
    chroma_db/          ← persistent ChromaDB vector store
    data/processed/embedding_report.json

One-time operation — once embedded, ChromaDB is reused by all later phases.
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

from tqdm import tqdm

from config import (
    PROCESSED_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHROMA_COLLECTION,
    TOP_K_RETRIEVAL,
    LOG_LEVEL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── User files ────────────────────────────────────────────────────────────────
USER_FILES = {
    "davidehrlich":    PROCESSED_DIR / "davidehrlich.json",
    "kevin_thecritic": PROCESSED_DIR / "kevin_thecritic.json",
    "theinsneider":    PROCESSED_DIR / "theinsneider.json",
}

# ── Batch size for embedding (keeps RAM manageable on CPU) ────────────────────
EMBED_BATCH_SIZE = 64


# ═════════════════════════════════════════════════════════════════════════════
#  LOAD EMBEDDING MODEL
# ═════════════════════════════════════════════════════════════════════════════

def load_embedding_model():
    """Load mxbai-embed-large from sentence-transformers (downloads on first run)."""
    from sentence_transformers import SentenceTransformer
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    log.info("  (First run downloads ~670MB — cached after that)")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("  Model loaded in %.1fs", time.time() - t0)
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  CHROMADB SETUP
# ═════════════════════════════════════════════════════════════════════════════

def get_chroma_collection(reset: bool = False):
    """
    Get or create the ChromaDB collection.
    reset=True → wipes existing collection (useful when re-embedding).
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            log.info("Deleted existing collection '%s'", CHROMA_COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    log.info("ChromaDB collection '%s' — %d existing docs", CHROMA_COLLECTION, collection.count())
    return collection


# ═════════════════════════════════════════════════════════════════════════════
#  EMBEDDING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def embed_user_tweets(
    user_id: str,
    tweets: list[dict],
    model,
    collection,
) -> dict:
    """
    Embed all tweets for one user and upsert into ChromaDB.
    Returns stats dict.
    """
    log.info("Embedding %d tweets for %s …", len(tweets), user_id)
    t0 = time.time()

    # Check which tweet_ids are already in the DB (skip re-embedding)
    existing_ids = set()
    try:
        existing = collection.get(where={"user_id": user_id}, include=[])
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    new_tweets = [t for t in tweets if t["tweet_id"] not in existing_ids]
    if not new_tweets:
        log.info("  All tweets already embedded — skipping.")
        return {"user_id": user_id, "embedded": 0, "skipped": len(tweets)}

    log.info("  %d new / %d already stored", len(new_tweets), len(existing_ids))

    # Process in batches
    embedded_count = 0
    for i in tqdm(range(0, len(new_tweets), EMBED_BATCH_SIZE),
                  desc=f"  {user_id}", unit="batch"):

        batch = new_tweets[i : i + EMBED_BATCH_SIZE]
        texts = [t["text"] for t in batch]

        # Generate embeddings
        vectors = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit vectors → cosine = dot product
        )

        # Build metadata for each tweet
        ids       = [t["tweet_id"] for t in batch]
        metadatas = []
        for t in batch:
            m = t.get("metrics", {})
            feat = t.get("features", {})
            metadatas.append({
                "user_id":       t["user_id"],
                "username":      t["username"],
                "timestamp":     t["timestamp"],
                "category":      t["category"],
                "sentiment":     t["sentiment"],
                "likes":         int(m.get("likes", 0)),
                "reposts":       int(m.get("reposts", 0)),
                "views":         int(m.get("views", 0)),
                "word_count":    int(feat.get("word_count", 0)),
                "emoji_count":   int(feat.get("emoji_count", 0)),
                "hashtag_count": int(feat.get("hashtag_count", 0)),
            })

        collection.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        embedded_count += len(batch)

    elapsed = round(time.time() - t0, 1)
    log.info("  Done — %d tweets embedded in %.1fs", embedded_count, elapsed)

    return {
        "user_id":  user_id,
        "embedded": embedded_count,
        "skipped":  len(existing_ids),
        "elapsed_s": elapsed,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL TEST
#  Run this BEFORE Phase 5 to confirm retrieval actually works.
# ═════════════════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    ("Oppenheimer and IMAX cinematography",   "davidehrlich"),
    ("superhero movies are getting worse",    "kevin_thecritic"),
    ("Academy Awards nominations controversy","theinsneider"),
]


def test_retrieval(model, collection):
    """
    Run a few test queries and print retrieved tweets.
    Manually inspect — do the results make sense?
    """
    log.info("")
    log.info("─" * 60)
    log.info("RETRIEVAL SANITY CHECK")
    log.info("─" * 60)

    for query, expected_user in TEST_QUERIES:
        log.info("\nQuery: '%s'  (expected user: %s)", query, expected_user)

        query_vec = model.encode([query], normalize_embeddings=True).tolist()

        results = collection.query(
            query_embeddings=query_vec,
            n_results=5,
            where={"user_id": expected_user},
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]   # cosine distance (lower = more similar)

        for rank, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
            similarity = round(1 - dist, 3)   # cosine similarity
            log.info(
                "  [%d] sim=%.3f | %s | %s",
                rank, similarity, meta["category"], doc[:100]
            )

    log.info("")
    log.info("─" * 60)
    log.info("If results look relevant → Phase 4 is working correctly.")
    log.info("If results look random  → check embedding model / collection.")
    log.info("─" * 60)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 4: Embeddings + ChromaDB")
    log.info("=" * 60)

    # Load model
    model = load_embedding_model()

    # Get ChromaDB collection (reset=True to re-embed from scratch)
    collection = get_chroma_collection(reset=True)

    all_stats = []

    for user_id, path in USER_FILES.items():
        if not path.exists():
            log.error("Processed file not found: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            tweets = json.load(f)

        stats = embed_user_tweets(user_id, tweets, model, collection)
        all_stats.append(stats)

    total = collection.count()
    log.info("")
    log.info("Total vectors in ChromaDB: %d", total)

    # Retrieval sanity check
    test_retrieval(model, collection)

    # Save report
    report_path = PROCESSED_DIR / "embedding_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at":   datetime.now(timezone.utc).isoformat(),
                "model":          EMBEDDING_MODEL,
                "embedding_dim":  EMBEDDING_DIM,
                "collection":     CHROMA_COLLECTION,
                "total_vectors":  total,
                "users":          all_stats,
            },
            f, ensure_ascii=False, indent=2,
        )
    log.info("Report → %s", report_path)
    log.info("")
    log.info("Phase 4 complete. Run Phase 5 next: python rag_baseline.py")


if __name__ == "__main__":
    main()
