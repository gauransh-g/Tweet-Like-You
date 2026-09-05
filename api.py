"""
api.py — High-Performance FastAPI Backend for Tweet Like You
============================================================
Exposes modern REST endpoints for:
  - Persona discovery & linguistic profiling
  - Real-time 4-Angle Editorial Matrix generation
  - In-flight tweet transformations (Spicier, Hinglish, Shorten)
  - Live factual web grounding (DuckDuckGo News + Wikipedia)
  - On-the-fly voice cloning and ChromaDB indexing
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root is in path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import PROFILES_DIR, PROCESSED_DIR, DATA_DIR
from dynamic_prompt import load_embedding_model, get_chroma_collection
from ground import get_factual_grounding
from rerank import generate_and_rerank, transform_tweet
from account_ingestor import fetch_public_posts_from_handle, parse_pasted_tweets
from profile_builder import build_and_index_profile
import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

app = FastAPI(
    title="Tweet Like You — Universal AI Ghostwriter API",
    version="2.5.0",
    description="Universal Social Voice Cloner, Multilingual Hinglish Engine & Editorial Angle Matrix API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory resources
EMBED_MODEL = None
CHROMA_COLLECTION = None
SPACY_NLP = None


def get_resources():
    global EMBED_MODEL, CHROMA_COLLECTION, SPACY_NLP
    if EMBED_MODEL is None:
        log.info("Lazy-loading embedding model …")
        EMBED_MODEL = load_embedding_model()
    if CHROMA_COLLECTION is None:
        log.info("Connecting to ChromaDB …")
        CHROMA_COLLECTION = get_chroma_collection()
    if SPACY_NLP is None:
        log.info("Loading spaCy pipeline …")
        SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner"])
    return EMBED_MODEL, CHROMA_COLLECTION, SPACY_NLP


# ── Pydantic Request/Response Models ─────────────────────────────────────────

class GenerateRequest(BaseModel):
    topic: str
    user_id: str
    language_blend: Optional[str] = "auto"   # "auto", "hinglish", "english", "hindi"
    use_grounding: Optional[bool] = True
    n_candidates: Optional[int] = 4


class TransformRequest(BaseModel):
    tweet_text: str
    user_id: str
    action: str  # "spicier", "hinglish", "shorten"


class CloneRequest(BaseModel):
    handle: str
    method: str  # "paste", "handle", "archive"
    pasted_text: Optional[str] = None
    user_email: Optional[str] = None


# ── Demo Users & Auth Storage ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


DEMO_USERS = {
    "demo@tweetlikeyou.com": {
        "id": "user_demo",
        "name": "Arjun Sharma",
        "email": "demo@tweetlikeyou.com",
        "role": "Creator & Culture Workspace",
        "avatar": "A",
        "password": "demo123",
        "personas": ["beingsalmankhan", "ravishastriofc"],
    },
    "founder@silicon.ai": {
        "id": "user_founder",
        "name": "Sarah Chen",
        "email": "founder@silicon.ai",
        "role": "Tech Founder & AI Workspace",
        "avatar": "S",
        "password": "founder123",
        "personas": ["sama_clone", "davidehrlich", "kevin_thecritic"],
    }
}


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "Tweet Like You API", "version": "2.5.0"}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    email = req.email.strip().lower()
    if email not in DEMO_USERS:
        user = {
            "id": f"user_{email.split('@')[0]}",
            "name": email.split('@')[0].capitalize(),
            "email": email,
            "role": "Personal Workspace",
            "avatar": email[0].upper(),
            "password": req.password,
            "personas": [],
        }
        DEMO_USERS[email] = user
        return {"success": True, "user": user}

    user = DEMO_USERS[email]
    if user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid password")

    return {"success": True, "user": user}


@app.get("/api/personas")
def list_personas(user_email: Optional[str] = None):
    """Discover all cloned accounts with full linguistic profile and bio."""
    personas = []
    for p_file in sorted(PROFILES_DIR.glob("*_persona.json")):
        user_id = p_file.stem.replace("_persona", "")
        style_file = PROFILES_DIR / f"{user_id}_style_profile.json"
        
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            
            s_data = {}
            if style_file.exists():
                with open(style_file, "r", encoding="utf-8") as sf:
                    s_data = json.load(sf)
                    
            flow = s_data.get("length_and_flow", {})
            cap = s_data.get("capitalization_habits", {})
            punct = s_data.get("punctuation_fingerprint", {})
            lang = s_data.get("language_and_cultural_profile", {})

            display_name = p_data.get("display_name") or user_id
            avatar = display_name[0].upper() if display_name else user_id[0].upper()

            personas.append({
                "user_id": user_id,
                "display_name": display_name,
                "role": p_data.get("role_and_background", ""),
                "avatar": avatar,
                "primary_language": lang.get("primary_language", "english"),
                "is_vernacular_adapted": lang.get("is_vernacular_adapted", False),
                "top_vernacular_tokens": lang.get("top_vernacular_tokens", []),
                "voice_dna": {
                    "target_words": round(flow.get("avg_words_per_tweet", 25)),
                    "lowercase_starts": round(cap.get("starts_with_lowercase_rate", 0.0) * 100),
                    "parentheses_rate": round(punct.get("parentheses_tweet_rate", 0.0) * 100),
                    "ttr": round(s_data.get("lexical_richness", {}).get("type_token_ratio_ttr", 0.2), 3),
                },
                "catchphrases": p_data.get("vocabulary_and_phrasing", {}).get("signature_catchphrases", []),
            })
        except Exception as e:
            log.warning("Could not load persona %s: %s", user_id, e)

    # If user_email is provided, optionally filter or tag personas belonging to this user
    user_email_clean = user_email.strip().lower() if user_email else None
    user_allowed_personas = None
    if user_email_clean and user_email_clean in DEMO_USERS:
        user_allowed_personas = set(DEMO_USERS[user_email_clean].get("personas", []))

    for p in personas:
        if user_allowed_personas is not None:
            p["is_in_workspace"] = p["user_id"] in user_allowed_personas
        else:
            p["is_in_workspace"] = True

    # If filtered view requested, filter down to user's workspace
    if user_allowed_personas is not None and len(user_allowed_personas) > 0:
        filtered = [p for p in personas if p["user_id"] in user_allowed_personas]
        return {"personas": filtered, "all_personas_count": len(personas)}

    return {"personas": personas, "all_personas_count": len(personas)}


@app.get("/api/persona/{user_id}")
def get_persona(user_id: str):
    p_file = PROFILES_DIR / f"{user_id}_persona.json"
    s_file = PROFILES_DIR / f"{user_id}_style_profile.json"
    if not p_file.exists():
        raise HTTPException(status_code=404, detail=f"Persona @{user_id} not found")

    with open(p_file, "r", encoding="utf-8") as f:
        persona = json.load(f)
    style = {}
    if s_file.exists():
        with open(s_file, "r", encoding="utf-8") as sf:
            style = json.load(sf)

    return {"persona": persona, "style": style}


@app.delete("/api/persona/{user_id}")
def delete_persona(user_id: str):
    """Purge persona files, processed dataset, and ChromaDB vector embeddings."""
    clean_id = user_id.strip().lstrip("@")
    log.info("Purging persona @%s ...", clean_id)

    deleted_files = []
    for ext in ["_persona.json", "_style_profile.json", "_system_prompt.txt"]:
        target = PROFILES_DIR / f"{clean_id}{ext}"
        if target.exists():
            try:
                target.unlink()
                deleted_files.append(str(target.name))
            except Exception as e:
                log.warning("Could not delete file %s: %s", target, e)

    proc_target = PROCESSED_DIR / f"{clean_id}.json"
    if proc_target.exists():
        try:
            proc_target.unlink()
            deleted_files.append(str(proc_target.name))
        except Exception as e:
            log.warning("Could not delete processed file %s: %s", proc_target, e)

    # Purge vectors from ChromaDB
    try:
        _, col, _ = get_resources()
        col.delete(where={"user_id": clean_id})
        log.info("Purged ChromaDB vectors for @%s", clean_id)
    except Exception as e:
        log.warning("ChromaDB vector delete error for @%s: %s", clean_id, e)

    # Remove from DEMO_USERS lists
    for u in DEMO_USERS.values():
        if clean_id in u.get("personas", []):
            u["personas"].remove(clean_id)

    return {
        "success": True,
        "deleted_user_id": clean_id,
        "deleted_files": deleted_files,
        "message": f"Successfully deleted @{clean_id} and purged associated vectors.",
    }


@app.post("/api/generate")
def generate_angles(req: GenerateRequest):
    """Generate tweets across all 4 strategic editorial angles."""
    log.info("Generating angles for @%s on topic '%s' (language: %s)", req.user_id, req.topic, req.language_blend)
    try:
        model, col, _ = get_resources()
        result = generate_and_rerank(
            topic=req.topic,
            user_id=req.user_id,
            embed_model=model,
            collection=col,
            n_candidates=req.n_candidates or 4,
            language_blend=req.language_blend or "auto",
        )

        grounding = get_factual_grounding(req.topic) if req.use_grounding else {"live_news": [], "factual_points": []}

        return {
            "success": True,
            "user_id": req.user_id,
            "topic": req.topic,
            "category": result["category"],
            "sentiment_intent": result["sentiment_intent"],
            "language_blend": result["language_blend"],
            "champion_tweet": result["champion_tweet"],
            "champion_score": result["champion_score"],
            "champion_angle": result["champion_angle"],
            "champion_metrics": result["champion_metrics"],
            "editorial_angles": result["editorial_angles"],
            "grounding": grounding,
            "language_profile": result.get("language_profile", {}),
        }
    except Exception as e:
        log.exception("Generation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transform")
def transform(req: TransformRequest):
    """Apply targeted transformation (spicier, hinglish, shorten)."""
    try:
        new_text = transform_tweet(req.tweet_text, req.user_id, req.action)
        return {
            "success": True,
            "original_text": req.tweet_text,
            "transformed_text": new_text,
            "action": req.action,
        }
    except Exception as e:
        log.exception("Transform error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clone")
def clone_account(req: CloneRequest):
    """Ingest past tweets and build a fresh Persona & ChromaDB index on-the-fly."""
    clean_handle = req.handle.strip().lstrip("@")
    if not clean_handle:
        raise HTTPException(status_code=400, detail="Handle is required")

    log.info("Cloning account @%s via method: %s", clean_handle, req.method)
    tweets = []

    if req.method == "paste":
        if not req.pasted_text or not req.pasted_text.strip():
            raise HTTPException(status_code=400, detail="Pasted text cannot be empty")
        tweets = parse_pasted_tweets(req.pasted_text, user_id=clean_handle)
    elif req.method == "handle":
        tweets = fetch_public_posts_from_handle(clean_handle)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported method: {req.method}")

    if len(tweets) < 3:
        raise HTTPException(status_code=400, detail=f"Need at least 3 distinct tweets to clone voice. Found {len(tweets)}.")

    try:
        model, col, nlp = get_resources()
        result = build_and_index_profile(
            user_id=clean_handle,
            tweets=tweets,
            nlp=nlp,
            embed_model=model,
            collection=col,
        )

        if req.user_email:
            u_clean = req.user_email.strip().lower()
            if u_clean in DEMO_USERS and result["user_id"] not in DEMO_USERS[u_clean]["personas"]:
                DEMO_USERS[u_clean]["personas"].append(result["user_id"])

        return {
            "success": True,
            "user_id": result["user_id"],
            "display_name": result["display_name"],
            "tweets_count": result["tweets_count"],
            "language_profile": result["style_profile"].get("language_and_cultural_profile", {}),
            "message": f"Successfully cloned @{result['user_id']} with {result['tweets_count']} tweets!",
        }
    except Exception as e:
        log.exception("Cloning error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/grounding")
def grounding_check(topic: str):
    """Fetch live breaking news and encyclopedic facts for any query."""
    res = get_factual_grounding(topic)
    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
