"""
dynamic_prompt.py — Phase 8: Dynamic Prompt Synthesis & Baseline 2
===================================================================
Fuses:
  1. Persona System Prompt & Voice Guidelines (Phase 7)
  2. Stylometric Guardrails & Target Metrics (Phase 6)
  3. Dynamic Few-Shot Examples from ChromaDB Vector Store (Phase 4)
  4. Banned AI Clichés & Formatting Rules

Generates Baseline 2: Persona-Grounded Semantic RAG tweets.

Outputs:
    data/baselines/baseline2_persona_rag_results.json
    data/baselines/baseline_comparison.md
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DATA_DIR,
    PROFILES_DIR,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    LOG_LEVEL,
)
from llm_client import generate, session_request_count

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASELINE_DIR = DATA_DIR / "baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

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
#  MODEL & RETRIEVAL HELPERS
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


def retrieve_few_shots(
    query: str,
    user_id: str,
    embed_model,
    collection,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve top-K most relevant past tweets to serve as dynamic few-shot examples."""
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
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            retrieved.append({
                "text": doc,
                "similarity": round(1.0 - dist, 4),
                "category": meta.get("category", ""),
                "sentiment": meta.get("sentiment", ""),
                "likes": meta.get("likes", 0),
            })

    return retrieved


# ═════════════════════════════════════════════════════════════════════════════
#  DYNAMIC PROMPT COMPOSER
# ═════════════════════════════════════════════════════════════════════════════

def build_dynamic_prompt(
    user_id: str,
    topic: str,
    few_shots: list[dict],
    persona_data: dict,
    style_data: dict,
) -> str:
    """
    Synthesize an end-to-end prompt incorporating:
    - Persona system prompt
    - Specific stylometric guidelines
    - Retrieved few-shot examples
    - Negative constraints (anti-clichés)
    """
    sys_prompt = persona_data.get("system_prompt", "").strip()

    # Stylometric rules
    flow = style_data.get("length_and_flow", {})
    cap = style_data.get("capitalization_habits", {})
    punct = style_data.get("punctuation_fingerprint", {})
    vocab = persona_data.get("vocabulary_and_phrasing", {})

    target_words = int(flow.get("avg_words_per_tweet", 28))
    lowercase_rate = cap.get("starts_with_lowercase_rate", 0.0)

    # Specific casing instruction
    if lowercase_rate > 0.4:
        casing_rule = "- CASING: You frequently start tweets in lowercase (over 50% of the time). Feel free to start in lowercase for an off-the-cuff, authentic feel."
    else:
        casing_rule = "- CASING: Always start with standard capitalization."

    # Specific punctuation cues
    punct_rules = []
    if punct.get("parentheses_tweet_rate", 0.0) > 0.12:
        punct_rules.append("- Use parenthetical asides or snarky observations in parentheses where fitting.")
    if punct.get("ellipses_per_tweet", 0.0) > 0.25:
        punct_rules.append("- Use trailing ellipses (...) or punchy dashes to create anticipation or conversational rhythm.")

    punct_section = "\n".join(punct_rules) if punct_rules else "- Use natural punctuation matching your voice."

    # Few-shot formatting
    examples = []
    for i, t in enumerate(few_shots, 1):
        examples.append(f'Example {i} (Likes: {t["likes"]} | Sim: {t["similarity"]:.2f}):\n"{t["text"]}"')
    few_shots_text = "\n\n".join(examples)

    banned_cliches = vocab.get("banned_cliches", [
        "delve", "testament", "tapestry", "in a world where", "masterpiece", "needless to say", "buckle up"
    ])
    char_words = vocab.get("characteristic_words", [])

    prompt = f"""{sys_prompt}

=== STYLOMETRIC & FORMATTING GUARDRAILS ===
- LENGTH: Target approximately {target_words} words (under 280 characters total). Do NOT pad with filler words.
{casing_rule}
{punct_section}
- VOCABULARY: Weave in your characteristic expressions naturally: {', '.join(char_words[:4])}
- BANNED CLICHÉS: Under NO circumstances use standard AI phrases such as: {', '.join(banned_cliches)}.
- NO HASHTAGS: Do not append generic promotional hashtags.

=== YOUR AUTHENTIC PAST TWEETS FOR REFERENCE ===
{few_shots_text}

=== TOPIC TO TWEET ABOUT ===
{topic}

INSTRUCTIONS:
Write ONE single complete tweet about the topic above.
Match your exact persona, rhythm, and tone.
Output ONLY the tweet text. Do not wrap in quotes or add explanations.
"""
    return prompt


# ═════════════════════════════════════════════════════════════════════════════
#  GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_persona_rag_tweet(
    topic: str,
    user_id: str,
    embed_model,
    collection,
    persona_data: dict,
    style_data: dict,
) -> dict:
    """Generate one tweet using Dynamic Persona RAG."""
    few_shots = retrieve_few_shots(topic, user_id, embed_model, collection, top_k=5)
    prompt = build_dynamic_prompt(user_id, topic, few_shots, persona_data, style_data)

    start = time.time()
    try:
        text = generate(prompt, temperature=0.8)
        elapsed = round(time.time() - start, 2)
        words = text.split()

        return {
            "status": "ok",
            "text": text,
            "char_count": len(text),
            "word_count": len(words),
            "elapsed_s": elapsed,
            "model": GEMINI_MODEL,
            "few_shots_count": len(few_shots),
            "avg_similarity": round(sum(f["similarity"] for f in few_shots) / len(few_shots), 4) if few_shots else 0.0,
            "retrieved_tweets": few_shots,
        }
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        log.error("Generation failed for user %s, topic '%s': %s", user_id, topic, e)
        return {
            "status": "error",
            "error": str(e),
            "elapsed_s": elapsed,
        }


def generate_comparative_table_markdown(
    b0_data: dict,
    b1_data: dict,
    b2_data: dict,
) -> str:
    """Generate side-by-side markdown comparison of Baseline 0 vs Baseline 1 vs Baseline 2."""
    b0_map = {r["topic"]: r.get("result", {}).get("text", "") for r in b0_data.get("results", [])}
    b1_map = {(r["user_id"], r["topic"]): r.get("result", {}).get("text", "") for r in b1_data.get("results", [])}
    b2_map = {(r["user_id"], r["topic"]): r.get("result", {}).get("text", "") for r in b2_data.get("results", [])}

    lines = [
        "# Evolution Across Baselines: Baseline 0 vs Baseline 1 vs Baseline 2",
        "",
        f"*Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        "",
        "- **Baseline 0:** Generic LLM (Topic only, zero persona, zero retrieval)",
        "- **Baseline 1:** Semantic RAG (Topic + 15 retrieved tweets, generic prompt)",
        "- **Baseline 2:** Dynamic Persona RAG (Topic + Persona System Prompt + Stylometric Guardrails + Dynamic Few-Shots)",
        "",
    ]

    for topic in TEST_TOPICS:
        lines.append(f"## Topic: *\"{topic}\"*")
        lines.append("")
        b0_text = b0_map.get(topic, "N/A")
        lines.append(f"**Baseline 0 (Generic LLM):**\n> \"{b0_text}\"\n")

        for u in USERS:
            b1_text = b1_map.get((u, topic), "N/A")
            b2_text = b2_map.get((u, topic), "N/A")
            lines.append(f"### `@{u}`")
            lines.append(f"- **Baseline 1 (Raw RAG):**\n  > \"{b1_text}\"")
            lines.append(f"- **Baseline 2 (Persona + Stylometric RAG):**\n  > \"{b2_text}\"")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 8: Dynamic Prompt & Baseline 2")
    log.info("=" * 60)

    embed_model = load_embedding_model()
    collection = get_chroma_collection()

    # Load personas and style profiles
    personas = {}
    styles = {}
    for u in USERS:
        p_path = PROFILES_DIR / f"{u}_persona.json"
        s_path = PROFILES_DIR / f"{u}_style_profile.json"
        with open(p_path, "r", encoding="utf-8") as f:
            personas[u] = json.load(f)
        with open(s_path, "r", encoding="utf-8") as f:
            styles[u] = json.load(f)

    all_results = []

    for user_id in USERS:
        log.info("\n%s", "─" * 60)
        log.info("GENERATING FOR: %s", user_id)
        log.info("─" * 60)

        for topic in TEST_TOPICS:
            log.info("\nTopic: %s", topic)
            result = generate_persona_rag_tweet(
                topic, user_id, embed_model, collection, personas[user_id], styles[user_id]
            )

            entry = {
                "user_id": user_id,
                "topic": topic,
                "baseline": "baseline_2_persona_rag",
                "result": result,
            }
            all_results.append(entry)

            if result["status"] == "ok":
                log.info("  Generated (%d words, %d chars, %.2fs):",
                         result["word_count"], result["char_count"], result["elapsed_s"])
                log.info("  → %s", result["text"])
            else:
                log.warning("  FAILED: %s", result.get("error"))

            time.sleep(2.0)  # Gentle rate pacing

    # Save Baseline 2 output
    out_json = BASELINE_DIR / "baseline2_persona_rag_results.json"
    output = {
        "baseline": "baseline_2_persona_rag",
        "description": "LLM + Persona System Prompt + Stylometric Guardrails + Dynamic Few-Shots",
        "model": GEMINI_MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "users": USERS,
        "topics": TEST_TOPICS,
        "results": all_results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info("\nBaseline 2 saved → %s (%d/%d successful)",
             out_json, sum(1 for r in all_results if r["result"].get("status") == "ok"), len(all_results))

    # Generate 3-way comparative markdown
    b0_path = BASELINE_DIR / "baseline0_results.json"
    b1_path = BASELINE_DIR / "baseline1_rag_results.json"
    if b0_path.exists() and b1_path.exists():
        with open(b0_path, "r", encoding="utf-8") as f:
            b0_data = json.load(f)
        with open(b1_path, "r", encoding="utf-8") as f:
            b1_data = json.load(f)

        comp_md = generate_comparative_table_markdown(b0_data, b1_data, output)
        comp_path = BASELINE_DIR / "baseline_comparison.md"
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write(comp_md)
        log.info("Comparative summary saved → %s", comp_path)

    log.info("\n%s", "=" * 60)
    log.info("Phase 8 complete! Total requests this session: %d", session_request_count())
    log.info("Run Phase 9 next: python ground.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
