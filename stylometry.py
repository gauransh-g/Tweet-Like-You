"""
stylometry.py — Phase 6: Stylometric Feature Profiling
======================================================
Extracts deep stylistic and linguistic fingerprints for each user:
  - Lexical diversity (TTR, Yule's K, hapax legomena)
  - Sentence length and readability (Flesch score)
  - Capitalization habits (lowercase starters, ALL-CAPS words)
  - Punctuation fingerprints (!, ?, ellipses, dashes, parentheses)
  - Part-of-Speech distributions (nouns, verbs, adjectives, adverbs)
  - Pronoun perspective (1st vs 2nd vs 3rd person)
  - Emoji, hashtag, and mention patterns
  - Distinctive TF-IDF n-grams and signature openers/closers

Inputs:
    data/processed/{user_id}.json

Outputs:
    data/profiles/{user_id}_style_profile.json
    data/profiles/all_style_profiles.json
    data/profiles/stylometric_comparison.md
"""

import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

try:
    import emoji as emoji_lib
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False

from config import (
    PROCESSED_DIR,
    PROFILES_DIR,
    LOG_LEVEL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

USER_FILES = {
    "davidehrlich":    PROCESSED_DIR / "davidehrlich.json",
    "kevin_thecritic": PROCESSED_DIR / "kevin_thecritic.json",
    "theinsneider":    PROCESSED_DIR / "theinsneider.json",
}

# Pronoun sets
PRONOUNS_1ST = {"i", "me", "my", "myself", "mine", "we", "us", "our", "ours", "ourselves"}
PRONOUNS_2ND = {"you", "your", "yours", "yourself", "yourselves"}
PRONOUNS_3RD = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves", "it", "its", "itself"}


def calculate_yules_k(words: list[str]) -> float:
    """
    Yule's Characteristic K: a length-independent metric of vocabulary richness.
    Lower K = richer vocabulary.
    K = 10^4 * (sum(i^2 * V_i) - N) / N^2
    """
    if not words:
        return 0.0
    n = len(words)
    freq_counts = Counter(Counter(words).values())
    sum_i2_vi = sum((i ** 2) * vi for i, vi in freq_counts.items())
    if n <= 1:
        return 0.0
    k = 10000.0 * (sum_i2_vi - n) / (n ** 2)
    return round(k, 2)


def count_emojis(text: str) -> list[str]:
    """Extract list of emojis found in text."""
    if HAS_EMOJI_LIB:
        return [c["emoji"] for c in emoji_lib.emoji_list(text)]
    pattern = re.compile(
        "[\U0001F300-\U0001F9FF\U00002700-\U000027BF\U0001FA00-\U0001FA9F\U00002600-\U000026FF]+",
        flags=re.UNICODE,
    )
    return pattern.findall(text)


# ── Unambiguous Hinglish & Vernacular Cultural Lexicon ────────────────────────
# Note: Exclude English homographs ("the", "to", "are", "hi", "log", "mere", "aao", "so")
UNAMBIGUOUS_HINGLISH = {
    # Pronouns & Address
    "bhai", "yaar", "boss", "guru", "baba", "bhau", "bidu", "dost", "dosti", "apna", "apni", "apne",
    "apun", "hum", "mera", "meri", "tera", "teri", "tere", "tum", "aap", "unka", "inka",
    "yeh", "woh", "janta", "bhaijaan", "ustad", "janab", "mitra", "dost",
    # Verbs & Auxiliary
    "hai", "hain", "tha", "thi", "hoga", "hogi", "hoge", "karo", "karna", "karega", "karegi",
    "raha", "rahi", "rahe", "dekho", "deko", "suno", "samjho", "batao", "chalo",
    "bolo", "bolta", "kehta", "lagta", "milta", "chahiye", "samajh", "aata", "aati", "gaya", "gayi",
    "diya", "liya", "rakho", "chod", "chhod", "bana", "banao", "dikhao",
    # Adverbs, Conjunctions, Particles
    "nahi", "nahin", "aur", "toh", "lekin", "magar",
    "kya", "kyun", "kyu", "kaise", "kaisa", "kaisi", "kahan", "kitna", "kitni",
    "bohot", "bahut", "kuch", "sabse", "sirf", "zara", "thoda", "ekdum", "poora", "puri",
    # Slang, Emotional Markers, Expressions
    "mast", "zabardast", "sahi", "pakka", "scene", "jhakaas", "badiya", "gazab", "lallantop",
    "chinta", "khushi", "pyaar", "dhamaka", "shandar", "zindabad", "bawal", "fadu", "chillar",
    "arre", "waah", "matlab", "baat", "dil", "shuru", "khatam", "solid", "kadak",
    "faad", "tashan", "jugaad", "fundoo", "dhamaal", "dhansu", "funda", "panga", "lafda",
    "bhasad", "bindaas", "nautanki", "machao", "machayenge",
    "sukh", "dukh", "bhara", "dilse", "jiyo"
}


def analyze_language_dna(texts: list[str]) -> dict:
    """Analyze script composition, Hinglish density, and code-mixing index."""
    total_words = 0
    hinglish_word_count = 0
    devanagari_char_count = 0
    total_chars = 0
    code_mixed_tweets = 0
    found_vernacular_words = Counter()

    for t in texts:
        clean_text = t.strip()
        total_chars += len(clean_text)
        
        # Devanagari Unicode block \u0900-\u097F
        dev_chars = len(re.findall(r"[\u0900-\u097F]", clean_text))
        devanagari_char_count += dev_chars
        
        words = [re.sub(r"[^\w]", "", w).lower() for w in clean_text.split() if w]
        words = [w for w in words if w and not w.isnumeric()]
        
        tweet_hinglish = 0
        tweet_english = 0
        
        for w in words:
            total_words += 1
            if w in UNAMBIGUOUS_HINGLISH:
                hinglish_word_count += 1
                tweet_hinglish += 1
                found_vernacular_words[w] += 1
            elif len(w) > 2:
                tweet_english += 1
                
        if tweet_hinglish >= 1 and tweet_english >= 1:
            code_mixed_tweets += 1

    total_tweets = len(texts) if texts else 1
    hinglish_density = round((hinglish_word_count / total_words), 4) if total_words else 0.0
    devanagari_density = round((devanagari_char_count / total_chars), 4) if total_chars else 0.0
    code_mixing_rate = round((code_mixed_tweets / total_tweets), 4)

    if devanagari_density > 0.15:
        primary = "hindi"
    elif hinglish_density > 0.015 or code_mixing_rate > 0.06:
        primary = "hinglish"
    else:
        primary = "english"

    return {
        "primary_language": primary,
        "hinglish_density": hinglish_density,
        "devanagari_char_density": devanagari_density,
        "code_mixing_rate": code_mixing_rate,
        "top_vernacular_tokens": [w for w, _ in found_vernacular_words.most_common(12)],
        "is_vernacular_adapted": primary in {"hinglish", "hindi"},
    }


def analyze_user_corpus(user_id: str, tweets: list[dict], nlp: spacy.Language) -> dict:
    """Compute comprehensive stylometric fingerprint for a single user."""
    log.info("Analyzing %d tweets for %s …", len(tweets), user_id)

    raw_texts = [t["text"] for t in tweets]
    total_tweets = len(tweets)
    lang_dna = analyze_language_dna(raw_texts)

    # 1. Punctuation & Formatting stats
    exclamations = sum(t.count("!") for t in raw_texts)
    questions = sum(t.count("?") for t in raw_texts)
    ellipses = sum(t.count("...") + t.count("…") for t in raw_texts)
    colons = sum(t.count(":") for t in raw_texts)
    semicolons = sum(t.count(";") for t in raw_texts)
    dashes = sum(t.count("—") + t.count("--") for t in raw_texts)
    parentheses_tweets = sum(1 for t in raw_texts if "(" in t and ")" in t)

    # Capitalization stats
    starts_lower = 0
    starts_upper = 0
    all_caps_words = []
    total_chars = sum(len(t) for t in raw_texts)
    total_upper_chars = sum(sum(1 for c in t if c.isupper()) for t in raw_texts)

    for text in raw_texts:
        stripped = text.strip()
        if stripped:
            first_alpha = next((c for c in stripped if c.isalpha()), None)
            if first_alpha:
                if first_alpha.islower():
                    starts_lower += 1
                elif first_alpha.isupper():
                    starts_upper += 1

        words = stripped.split()
        for w in words:
            clean_w = re.sub(r"[^\w]", "", w)
            if clean_w.isupper() and len(clean_w) > 1 and not clean_w.isnumeric():
                all_caps_words.append(clean_w)

    top_caps_words = [w for w, _ in Counter(all_caps_words).most_common(10) if w not in {"RT", "A", "I"}]

    # Emojis, Hashtags, Mentions
    all_emojis = []
    all_hashtags = []
    all_mentions = []
    for text in raw_texts:
        all_emojis.extend(count_emojis(text))
        all_hashtags.extend(re.findall(r"#\w+", text))
        all_mentions.extend(re.findall(r"@\w+", text))

    # Readability via textstat
    readability_scores = []
    if HAS_TEXTSTAT:
        for text in raw_texts:
            try:
                readability_scores.append(textstat.flesch_reading_ease(text))
            except Exception:
                pass
    avg_readability = round(float(np.mean(readability_scores)), 1) if readability_scores else 65.0

    # 2. NLP Batch Parsing (spaCy)
    all_tokens = []
    all_lemmas = []
    all_pos = Counter()
    pronoun_counts = {"1st_person": 0, "2nd_person": 0, "3rd_person": 0}
    sentence_counts = []
    words_per_sentence = []

    # Batch parse using nlp.pipe for high performance
    for doc in nlp.pipe(raw_texts, batch_size=200):
        sents = list(doc.sents)
        sentence_counts.append(len(sents))
        for sent in sents:
            words = [tok for tok in sent if not tok.is_punct and not tok.is_space]
            if words:
                words_per_sentence.append(len(words))

        for token in doc:
            if not token.is_punct and not token.is_space:
                token_lower = token.text.lower()
                all_tokens.append(token_lower)
                all_lemmas.append(token.lemma_.lower())
                all_pos[token.pos_] += 1

                if token_lower in PRONOUNS_1ST:
                    pronoun_counts["1st_person"] += 1
                elif token_lower in PRONOUNS_2ND:
                    pronoun_counts["2nd_person"] += 1
                elif token_lower in PRONOUNS_3RD:
                    pronoun_counts["3rd_person"] += 1

    total_words = len(all_tokens)
    vocab = set(all_tokens)
    vocab_size = len(vocab)
    ttr = round(vocab_size / total_words, 4) if total_words else 0.0

    word_freqs = Counter(all_tokens)
    hapax_count = sum(1 for _, count in word_freqs.items() if count == 1)
    hapax_ratio = round(hapax_count / vocab_size, 4) if vocab_size else 0.0
    yules_k = calculate_yules_k(all_tokens)

    # POS Ratios
    pos_ratios = {
        pos: round(count / total_words, 4) for pos, count in all_pos.items()
    } if total_words else {}

    total_pronouns = sum(pronoun_counts.values())
    pronoun_ratios = {
        p: round(count / total_pronouns, 4) for p, count in pronoun_counts.items()
    } if total_pronouns else {}

    # 3. Openers & Closers
    openers = Counter()
    closers = Counter()
    for text in raw_texts:
        words = text.strip().split()
        if len(words) >= 2:
            op = " ".join(words[:2]).lower()
            openers[op] += 1
            cl = " ".join(words[-2:]).lower()
            closers[cl] += 1

    top_openers = [k for k, _ in openers.most_common(8)]
    top_closers = [k for k, _ in closers.most_common(8)]

    # Category and Sentiment breakdown
    categories = Counter(t.get("category", "General") for t in tweets)
    sentiments = Counter(t.get("sentiment", "Neutral") for t in tweets)

    profile = {
        "user_id": user_id,
        "sample_size": total_tweets,
        "lexical_richness": {
            "total_tokens": total_words,
            "vocab_size": vocab_size,
            "type_token_ratio_ttr": ttr,
            "yules_characteristic_k": yules_k,
            "hapax_legomena_ratio": hapax_ratio,
            "avg_word_length": round(float(np.mean([len(w) for w in all_tokens])), 2) if all_tokens else 0.0,
        },
        "length_and_flow": {
            "avg_words_per_tweet": round(total_words / total_tweets, 1),
            "avg_chars_per_tweet": round(total_chars / total_tweets, 1),
            "avg_sentences_per_tweet": round(float(np.mean(sentence_counts)), 2) if sentence_counts else 1.0,
            "avg_words_per_sentence": round(float(np.mean(words_per_sentence)), 2) if words_per_sentence else 0.0,
            "flesch_reading_ease": avg_readability,
        },
        "capitalization_habits": {
            "starts_with_lowercase_rate": round(starts_lower / total_tweets, 4),
            "starts_with_capital_rate": round(starts_upper / total_tweets, 4),
            "uppercase_char_ratio": round(total_upper_chars / total_chars, 4) if total_chars else 0.0,
            "top_all_caps_words": top_caps_words,
        },
        "punctuation_fingerprint": {
            "exclamations_per_100_words": round((exclamations / total_words) * 100, 3) if total_words else 0.0,
            "questions_per_100_words": round((questions / total_words) * 100, 3) if total_words else 0.0,
            "ellipses_per_tweet": round(ellipses / total_tweets, 3),
            "colons_per_tweet": round(colons / total_tweets, 3),
            "semicolons_per_tweet": round(semicolons / total_tweets, 3),
            "dashes_per_tweet": round(dashes / total_tweets, 3),
            "parentheses_tweet_rate": round(parentheses_tweets / total_tweets, 3),
        },
        "grammatical_profile": {
            "noun_ratio": pos_ratios.get("NOUN", 0.0) + pos_ratios.get("PROPN", 0.0),
            "verb_ratio": pos_ratios.get("VERB", 0.0) + pos_ratios.get("AUX", 0.0),
            "adj_ratio": pos_ratios.get("ADJ", 0.0),
            "adv_ratio": pos_ratios.get("ADV", 0.0),
            "pronoun_ratio": pos_ratios.get("PRON", 0.0),
            "pronoun_perspective": pronoun_ratios,
        },
        "social_microblogging": {
            "emoji_rate_per_tweet": round(len(all_emojis) / total_tweets, 3),
            "top_emojis": [e for e, _ in Counter(all_emojis).most_common(5)],
            "hashtag_rate_per_tweet": round(len(all_hashtags) / total_tweets, 3),
            "top_hashtags": [h for h, _ in Counter(all_hashtags).most_common(5)],
            "mention_rate_per_tweet": round(len(all_mentions) / total_tweets, 3),
        },
        "signature_phrases": {
            "top_openers": top_openers,
            "top_closers": top_closers,
        },
        "topic_and_tone": {
            "categories": dict(categories.most_common()),
            "sentiments": dict(sentiments.most_common()),
        },
        "language_and_cultural_profile": lang_dna,
    }

    return profile


def extract_comparative_tfidf(users_tweets: dict[str, list[str]], top_n: int = 15) -> dict[str, list[dict]]:
    """Compute distinctive n-grams for each author relative to the others using TF-IDF."""
    corpus = []
    user_order = list(users_tweets.keys())

    for u in user_order:
        corpus.append(" ".join(users_tweets[u]))

    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.9,
        min_df=1,
        stop_words="english",
        max_features=5000,
    )
    matrix = tfidf.fit_transform(corpus)
    feature_names = np.array(tfidf.get_feature_names_out())

    results = {}
    for idx, u in enumerate(user_order):
        row = matrix.getrow(idx).toarray().flatten()
        top_indices = np.argsort(row)[::-1][:top_n]
        results[u] = [
            {"phrase": str(feature_names[i]), "score": round(float(row[i]), 4)}
            for i in top_indices if row[i] > 0
        ]

    return results


def generate_comparison_markdown(profiles: dict[str, dict], tfidf_distinctive: dict[str, list[dict]]) -> str:
    """Create a clean markdown document comparing stylometric fingerprints."""
    users = list(profiles.keys())
    lines = [
        "# Stylometric Comparison & Style Fingerprints",
        "",
        f"*Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        "",
        "## 1. Overview & Lexical Diversity",
        "",
        "| Metric | " + " | ".join(f"`@{u}`" for u in users) + " |",
        "|---| " + " | ".join("---" for _ in users) + " |",
        "| **Sample Size (Tweets)** | " + " | ".join(str(profiles[u]["sample_size"]) for u in users) + " |",
        "| **Avg Words / Tweet** | " + " | ".join(str(profiles[u]["length_and_flow"]["avg_words_per_tweet"]) for u in users) + " |",
        "| **Avg Chars / Tweet** | " + " | ".join(str(profiles[u]["length_and_flow"]["avg_chars_per_tweet"]) for u in users) + " |",
        "| **Type-Token Ratio (TTR)** | " + " | ".join(f"{profiles[u]['lexical_richness']['type_token_ratio_ttr']:.4f}" for u in users) + " |",
        "| **Yule's Characteristic K** (lower = richer) | " + " | ".join(f"{profiles[u]['lexical_richness']['yules_characteristic_k']:.1f}" for u in users) + " |",
        "| **Flesch Reading Ease** | " + " | ".join(f"{profiles[u]['length_and_flow']['flesch_reading_ease']:.1f}" for u in users) + " |",
        "",
        "## 2. Capitalization & Punctuation Signature",
        "",
        "| Trait | " + " | ".join(f"`@{u}`" for u in users) + " |",
        "|---| " + " | ".join("---" for _ in users) + " |",
        "| **Starts with Lowercase** | " + " | ".join(f"{profiles[u]['capitalization_habits']['starts_with_lowercase_rate']*100:.1f}%" for u in users) + " |",
        "| **Starts with Capital** | " + " | ".join(f"{profiles[u]['capitalization_habits']['starts_with_capital_rate']*100:.1f}%" for u in users) + " |",
        "| **Parentheses Tweet Rate** | " + " | ".join(f"{profiles[u]['punctuation_fingerprint']['parentheses_tweet_rate']*100:.1f}%" for u in users) + " |",
        "| **Exclamations / 100 words** | " + " | ".join(f"{profiles[u]['punctuation_fingerprint']['exclamations_per_100_words']:.2f}" for u in users) + " |",
        "| **Questions / 100 words** | " + " | ".join(f"{profiles[u]['punctuation_fingerprint']['questions_per_100_words']:.2f}" for u in users) + " |",
        "| **Ellipses / Tweet** | " + " | ".join(f"{profiles[u]['punctuation_fingerprint']['ellipses_per_tweet']:.3f}" for u in users) + " |",
        "| **Emoji Rate / Tweet** | " + " | ".join(f"{profiles[u]['social_microblogging']['emoji_rate_per_tweet']:.3f}" for u in users) + " |",
        "| **Top ALL-CAPS Words** | " + " | ".join(", ".join(profiles[u]['capitalization_habits']['top_all_caps_words'][:3]) for u in users) + " |",
        "",
        "## 3. Grammatical & Pronoun Perspective",
        "",
        "| Part of Speech | " + " | ".join(f"`@{u}`" for u in users) + " |",
        "|---| " + " | ".join("---" for _ in users) + " |",
        "| **Noun / Propn Ratio** | " + " | ".join(f"{profiles[u]['grammatical_profile']['noun_ratio']*100:.1f}%" for u in users) + " |",
        "| **Verb / Aux Ratio** | " + " | ".join(f"{profiles[u]['grammatical_profile']['verb_ratio']*100:.1f}%" for u in users) + " |",
        "| **Adjective Ratio** | " + " | ".join(f"{profiles[u]['grammatical_profile']['adj_ratio']*100:.1f}%" for u in users) + " |",
        "| **Adverb Ratio** | " + " | ".join(f"{profiles[u]['grammatical_profile']['adv_ratio']*100:.1f}%" for u in users) + " |",
        "| **1st Person Pronouns ('I', 'we')** | " + " | ".join(f"{profiles[u]['grammatical_profile']['pronoun_perspective'].get('1st_person', 0)*100:.1f}%" for u in users) + " |",
        "| **2nd Person Pronouns ('you')** | " + " | ".join(f"{profiles[u]['grammatical_profile']['pronoun_perspective'].get('2nd_person', 0)*100:.1f}%" for u in users) + " |",
        "| **3rd Person Pronouns ('they', 'it')** | " + " | ".join(f"{profiles[u]['grammatical_profile']['pronoun_perspective'].get('3rd_person', 0)*100:.1f}%" for u in users) + " |",
        "",
        "## 4. Most Distinctive N-Grams (TF-IDF vs Other Critics)",
        "",
    ]

    for u in users:
        top_phrases = [f"`{p['phrase']}` ({p['score']:.3f})" for p in tfidf_distinctive.get(u, [])[:8]]
        lines.append(f"### `@{u}`")
        lines.append("- Distinctive vocabulary: " + ", ".join(top_phrases))
        top_ops = [f'"{op}"' for op in profiles[u]["signature_phrases"]["top_openers"][:4]]
        lines.append("- Common openers: " + ", ".join(top_ops))
        lines.append("")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Tweet Like You — Phase 6: Stylometric Profiling")
    log.info("=" * 60)

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading spaCy NLP pipeline (en_core_web_sm) …")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    profiles = {}
    users_tweets_text = {}

    for user_id, path in USER_FILES.items():
        if not path.exists():
            log.error("File not found: %s", path)
            continue

        with open(path, "r", encoding="utf-8") as f:
            tweets = json.load(f)

        users_tweets_text[user_id] = [t["text"] for t in tweets]
        profile = analyze_user_corpus(user_id, tweets, nlp)
        profiles[user_id] = profile

        # Save individual profile
        user_out = PROFILES_DIR / f"{user_id}_style_profile.json"
        with open(user_out, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        log.info("  Saved style profile → %s", user_out)

    # Compute comparative TF-IDF n-grams across all 3
    log.info("Computing comparative TF-IDF distinctive n-grams …")
    tfidf_distinctive = extract_comparative_tfidf(users_tweets_text)

    for user_id in profiles:
        profiles[user_id]["distinctive_ngrams"] = tfidf_distinctive.get(user_id, [])

    # Save combined profiles
    combined_out = PROFILES_DIR / "all_style_profiles.json"
    with open(combined_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "profiles": profiles,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info("Combined profiles saved → %s", combined_out)

    # Generate and save markdown comparison
    md_content = generate_comparison_markdown(profiles, tfidf_distinctive)
    md_out = PROFILES_DIR / "stylometric_comparison.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_content)
    log.info("Comparison markdown saved → %s", md_out)

    # Print summary console table
    log.info("\n" + "=" * 70)
    log.info("PHASE 6 STYLOMETRIC SUMMARY")
    log.info("=" * 70)
    log.info("%-18s %10s %10s %10s %10s %12s", "Critic", "TTR", "Yule's K", "Words/Tw", "LowerStart", "Parentheses")
    log.info("%-18s %10s %10s %10s %10s %12s", "─"*18, "─"*10, "─"*10, "─"*10, "─"*10, "─"*12)
    for u in profiles:
        p = profiles[u]
        log.info(
            "%-18s %10.4f %10.1f %10.1f %9.1f%% %11.1f%%",
            f"@{u}",
            p["lexical_richness"]["type_token_ratio_ttr"],
            p["lexical_richness"]["yules_characteristic_k"],
            p["length_and_flow"]["avg_words_per_tweet"],
            p["capitalization_habits"]["starts_with_lowercase_rate"] * 100,
            p["punctuation_fingerprint"]["parentheses_tweet_rate"] * 100,
        )
    log.info("=" * 70)
    log.info("Phase 6 complete. Run Phase 7 next: python persona_profiler.py\n")


if __name__ == "__main__":
    main()
