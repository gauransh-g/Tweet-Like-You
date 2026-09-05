"""
app.py — Universal Social Voice Cloner & Dynamic Ghostwriter
============================================================
Clone ANY voice (Founders, Journalists, Creators, Film Critics, Everyday Users)
and generate tweets on ANY topic with real-time live news and facts.

Features:
  1. Universal Account Connector: Paste tweets, upload archive (CSV/JSON), or lookup handle
  2. Instant On-The-Fly Stylometrics & ChromaDB Vector Indexing
  3. Live Real-Time Web News Grounding (DDGS + NewsAPI + Wikipedia)
  4. Best-of-N Candidate Generation with Stylometric Reranking
  5. One-Click Share to X / Twitter Intent Integration

Run:
    streamlit run app.py
"""

import json
import os
import urllib.parse
from pathlib import Path

import spacy
import streamlit as st

from config import (
    DATA_DIR,
    PROFILES_DIR,
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
)
from dynamic_prompt import (
    load_embedding_model,
    get_chroma_collection,
)
from classifier import classify_topic_intent
from ground import get_factual_grounding, format_grounding_for_prompt
from rerank import generate_and_rerank
from account_ingestor import (
    parse_pasted_tweets,
    parse_csv_archive,
    parse_json_archive,
    fetch_public_posts_from_handle,
)
from profile_builder import build_and_index_profile

# ── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tweet Like You — Universal AI Voice Cloner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .tweet-card {
        background: linear-gradient(145deg, #16181f, #10121a);
        border: 1px solid #282c3f;
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
    }
    .tweet-header {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }
    .tweet-avatar {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3182ce, #805ad5);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        margin-right: 12px;
    }
    .tweet-author {
        font-weight: 700;
        font-size: 1.1rem;
        color: #f7fafc;
        margin-right: 6px;
    }
    .tweet-handle {
        color: #718096;
        font-size: 0.92rem;
    }
    .tweet-body {
        font-size: 1.25rem;
        line-height: 1.55;
        color: #edf2f7;
        margin: 14px 0;
        word-break: break-word;
        font-weight: 400;
    }
    .metric-badge {
        display: inline-block;
        background: #2b6cb0;
        color: #ffffff;
        font-size: 0.8rem;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-score {
        background: #2f855a;
        color: #fff;
    }
    .badge-cat {
        background: #6b46c1;
        color: #fff;
    }
    .news-box {
        background: #1a202c;
        border-left: 4px solid #3182ce;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Model Loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing NLP pipelines and Vector DB …")
def get_shared_resources():
    embed_model = load_embedding_model()
    collection = get_chroma_collection()
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return embed_model, collection, nlp


def discover_available_personas() -> dict[str, str]:
    """Scan data/profiles to discover all existing cloned accounts."""
    personas = {}
    for p_file in sorted(PROFILES_DIR.glob("*_persona.json")):
        user_id = p_file.stem.replace("_persona", "")
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                display_name = data.get("display_name") or user_id
                role = data.get("role_and_background", "")
                personas[user_id] = f"@{user_id} — {display_name} ({role[:35]}…)"
        except Exception:
            personas[user_id] = f"@{user_id}"
    return personas


def load_critic_data(user_id: str):
    p_path = PROFILES_DIR / f"{user_id}_persona.json"
    s_path = PROFILES_DIR / f"{user_id}_style_profile.json"
    with open(p_path, "r", encoding="utf-8") as f:
        persona = json.load(f)
    with open(s_path, "r", encoding="utf-8") as f:
        style = json.load(f)
    return persona, style


# ── App Header ────────────────────────────────────────────────────────────────
st.title("⚡ Tweet Like You")
st.markdown(
    "**Universal Social Voice Cloner & Dynamic AI Ghostwriter.** "
    "Connect any social account, profile your writing DNA on-the-fly, "
    "and generate authentic tweets grounded in **live, real-time web facts**."
)

# ── Session State for Persona Synchronization ──────────────────────────────────
personas_map = discover_available_personas()
persona_keys = list(personas_map.keys())

# If the tab selectbox was changed, sync to active_persona first
if "main_tab_persona_select" in st.session_state and st.session_state["main_tab_persona_select"] in personas_map:
    st.session_state["active_persona"] = st.session_state["main_tab_persona_select"]

if "active_persona" not in st.session_state or st.session_state["active_persona"] not in persona_keys:
    st.session_state["active_persona"] = persona_keys[0] if persona_keys else "davidehrlich"

if "current_topic" not in st.session_state:
    st.session_state["current_topic"] = "Nvidia Blackwell GPU demand and AI infrastructure"

if "latest_generation" not in st.session_state:
    st.session_state["latest_generation"] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("👤 Active Persona")
    active_persona = st.session_state["active_persona"]
    persona_data, style_data = load_critic_data(active_persona)

    st.markdown(f"### **{persona_data.get('display_name', active_persona)}**")
    st.markdown(f"**`@{active_persona}`**")
    st.caption(persona_data.get("role_and_background", "Cloned Persona Profile"))

    st.markdown("---")
    st.subheader("🧬 Voice DNA Blueprint")

    flow = style_data.get("length_and_flow", {})
    cap = style_data.get("capitalization_habits", {})
    punct = style_data.get("punctuation_fingerprint", {})

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Target Words", f"{flow.get('avg_words_per_tweet', 28):.0f}")
        st.metric("Lowercase Starts", f"{cap.get('starts_with_lowercase_rate', 0.0)*100:.1f}%")
    with col_s2:
        st.metric("TTR Richness", f"{style_data.get('lexical_richness', {}).get('type_token_ratio_ttr', 0.2):.3f}")
        st.metric("Parentheses", f"{punct.get('parentheses_tweet_rate', 0.0)*100:.1f}%")

    st.markdown("---")
    st.subheader("⚙️ Generation Settings")
    num_candidates = st.slider("Candidates Generated (N)", min_value=1, max_value=5, value=3)
    use_grounding = st.toggle("Enable Live Web News Grounding", value=True)


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_gen, tab_connect, tab_bench, tab_style, tab_prompts = st.tabs([
    "🚀 Live Ghostwriter",
    "🔗 Connect Social Account / Clone Any Voice",
    "📊 Baseline Benchmarks",
    "🧬 Linguistic Signatures",
    "📜 Persona System Prompts",
])


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1: LIVE GHOSTWRITER
# ═════════════════════════════════════════════════════════════════════════════
with tab_gen:
    # Prominent In-Tab Persona Selector (Single Source of Truth)
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        active_persona = st.session_state["active_persona"]
        active_idx = persona_keys.index(active_persona) if active_persona in persona_keys else 0

        def on_tab_persona_change():
            st.session_state["active_persona"] = st.session_state["main_tab_persona_select"]

        selected_user = st.selectbox(
            "🎭 Choose Persona to Tweet As:",
            options=persona_keys,
            index=active_idx,
            format_func=lambda x: personas_map.get(x, x),
            key="main_tab_persona_select",
            on_change=on_tab_persona_change,
        )
        st.session_state["active_persona"] = selected_user
    with col_sel2:
        st.write("")
        st.write("")
        st.caption(f"Active: **@{selected_user}**")

    # Refresh active data for the selected persona
    persona_data, style_data = load_critic_data(selected_user)

    st.info(f"✨ **Ghostwriting as:** {persona_data.get('display_name', selected_user)} (`@{selected_user}`) — *{persona_data.get('role_and_background', '')}*")

    # Universal quick-pick topics across Tech, News, Cinema, Sports, Culture
    st.markdown("**Quick Topics Across Domains:**")
    quick_topics = [
        "Nvidia Blackwell GPU demand and AI infrastructure",
        "OpenAI new autonomous agent release and safety concerns",
        "India vs Australia test match drama and team performance",
        "The Brutalist and its 3.5 hour runtime",
        "Whether streaming has killed the theatrical experience",
    ]

    chip_cols = st.columns(len(quick_topics))
    for i, q in enumerate(quick_topics):
        if chip_cols[i].button(q[:18] + "…", key=f"chip_{i}"):
            st.session_state["current_topic"] = q
            st.rerun()

    topic_input = st.text_input(
        "Enter any topic, breaking headline, or opinion to tweet about:",
        value=st.session_state["current_topic"],
        key="topic_input_box",
    )
    st.session_state["current_topic"] = topic_input

    # Generation Button with Session-State Persistence
    if st.button("🔥 Generate Champion Tweet", type="primary", use_container_width=True):
        if not topic_input.strip():
            st.warning("⚠️ Please enter a topic to tweet about.")
        else:
            with st.spinner(f"Routing intent, retrieving live facts, and generating Best-of-{num_candidates} candidates for @{selected_user} …"):
                try:
                    embed_model, collection, _ = get_shared_resources()
                    result = generate_and_rerank(
                        topic_input.strip(),
                        selected_user,
                        embed_model,
                        collection,
                        n_candidates=num_candidates,
                    )
                    intent = classify_topic_intent(topic_input.strip(), selected_user)
                    grounding = get_factual_grounding(topic_input.strip()) if use_grounding else {"live_news": [], "factual_points": []}

                    st.session_state["latest_generation"] = {
                        "result": result,
                        "intent": intent,
                        "grounding": grounding,
                        "user_id": selected_user,
                        "topic": topic_input.strip(),
                        "persona_data": persona_data,
                    }
                except Exception as ex:
                    st.error(f"❌ Generation error: {str(ex)}")
                    log.exception("Error during tweet generation: %s", ex)

    # Render Champion Tweet Result Card (Persists across reruns)
    if st.session_state.get("latest_generation"):
        gen = st.session_state["latest_generation"]
        result = gen["result"]
        intent = gen["intent"]
        grounding = gen["grounding"]
        gen_user = gen["user_id"]
        p_data = gen.get("persona_data", persona_data)

        # Intent & Score Badges
        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
        col_b1.markdown(f'<span class="metric-badge badge-cat">{intent["category"]}</span>', unsafe_allow_html=True)
        col_b2.markdown(f'<span class="metric-badge">{intent["sentiment_intent"]}</span>', unsafe_allow_html=True)
        col_b3.markdown(f'<span class="metric-badge badge-score">Stylometric Alignment: {result["champion_score"]}/100</span>', unsafe_allow_html=True)

        # Live News Radar
        if use_grounding and (grounding.get("live_news") or grounding.get("factual_points")):
            with st.expander("🌐 Live Real-Time News & Facts Integrated into Tweet"):
                st.caption(f"Time Anchor: {grounding.get('current_date_anchor', 'Present')} | Sources: {', '.join(grounding.get('sources', []))}")
                if grounding.get("live_news"):
                    st.markdown("**Recent Breaking Headlines:**")
                    for n in grounding["live_news"]:
                        st.markdown(f"- **{n['title']}** *({n['source']})*: {n['snippet']}")
                if grounding.get("factual_points"):
                    st.markdown("**Encyclopedic Facts:**")
                    for f in grounding["factual_points"]:
                        st.markdown(f"- {f}")

        # Champion Tweet Display Card
        champion_text = result["champion_tweet"]
        display_author = p_data.get("display_name", gen_user)
        first_char = display_author[0].upper() if display_author else "U"
        encoded_tweet = urllib.parse.quote(champion_text)
        twitter_share_url = f"https://twitter.com/intent/tweet?text={encoded_tweet}"

        st.markdown(f"""
        <div class="tweet-card">
            <div class="tweet-header">
                <div class="tweet-avatar">{first_char}</div>
                <div>
                    <span class="tweet-author">{display_author}</span>
                    <span class="tweet-handle">@{gen_user} · Just now</span>
                </div>
            </div>
            <div class="tweet-body">
                {champion_text}
            </div>
            <div style="color: #718096; font-size: 0.85rem; border-top: 1px solid #282c3f; padding-top: 12px; margin-top: 12px;">
                💬 248 &nbsp;&nbsp;&nbsp; 🔁 1,420 &nbsp;&nbsp;&nbsp; ❤️ 9,842 &nbsp;&nbsp;&nbsp; 📊 {result["champion_metrics"]["words"]} words ({result["champion_metrics"]["chars"]} chars)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Actions
        col_act1, col_act2 = st.columns([1, 4])
        col_act1.link_button("🚀 Post to X / Twitter", twitter_share_url, use_container_width=True)

        # Ranked Candidates Breakdown
        if len(result["all_ranked_candidates"]) > 1:
            with st.expander(f"🔍 Inspect All {len(result['all_ranked_candidates'])} Candidates & Scoring Breakdown"):
                for c in result["all_ranked_candidates"]:
                    b = c["breakdown"]
                    st.markdown(f"**Candidate #{c['candidate_id']}** (Score: `{c['composite_score']}` | Temp: `{c['temperature']}` | Words: `{c['word_count']}`)")
                    st.markdown(f"> *\"{c['text']}\"*")
                    st.caption(
                        f"Length pts: {b['length_score']} | Casing pts: {b['cap_score']} | "
                        f"Punctuation pts: {b['punct_score']} | Vocab pts: {b['vocab_score']} | "
                        f"Cliché penalty: -{b['cliche_penalty']}"
                    )
                    st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2: CONNECT SOCIAL ACCOUNT / CLONE ANY VOICE
# ═════════════════════════════════════════════════════════════════════════════
with tab_connect:
    st.subheader("🔗 Connect Any Account & Clone Writing Voice")
    st.markdown(
        "Import past tweets from yourself, a tech founder, journalist, creator, or friend. "
        "The system will extract their stylometric DNA, synthesize a custom Persona Prompt, "
        "and index their memories into ChromaDB."
    )

    ingest_method = st.radio(
        "Choose how to import past tweets:",
        ["✍️ Paste Recent Tweets", "📁 Upload Twitter/X Archive (CSV/JSON)", "🔍 Lookup Public Handle"],
        horizontal=True,
    )

    col_in1, col_in2 = st.columns([2, 1])
    with col_in1:
        new_handle = st.text_input("Account Handle (e.g. @myname or @sama):", value="@tech_founder")
        new_name = st.text_input("Display Name:", value="Tech Innovator")

    # Initialize staged tweets state
    if "staged_tweets" not in st.session_state:
        st.session_state["staged_tweets"] = []

    if ingest_method == "✍️ Paste Recent Tweets":
        st.markdown("**Paste 5–25 past tweets** (separated by newlines or blank lines):")
        pasted_text = st.text_area(
            "Paste Tweets Here:",
            height=200,
            value=(
                "the gap between AI research demos and battle-tested production code is where all the actual startup value is created.\n\n"
                "hot take: the best engineering teams in 2026 are 5 people directing 50 specialized autonomous agents.\n\n"
                "optimism is an underrated competitive advantage. cynical commentators never build anything enduring.\n\n"
                "shipping fast and learning from real user friction beats debating abstractions in a meeting room every time.\n\n"
                "hardware constraints are real. compute and energy will dictate the next decade of technology."
            ),
        )
        if st.button("📥 Stage Pasted Tweets"):
            parsed = parse_pasted_tweets(pasted_text, new_handle, new_name)
            if parsed:
                st.session_state["staged_tweets"] = parsed
                st.success(f"Staged {len(parsed)} tweets for @{new_handle.lstrip('@')}!")
            else:
                st.error("No valid tweets found. Please ensure tweets have text.")

    elif ingest_method == "📁 Upload Twitter/X Archive (CSV/JSON)":
        st.markdown("Upload your exported `tweets.csv` or `tweets.json` from X Settings:")
        uploaded_file = st.file_uploader("Upload Archive File", type=["csv", "json"])
        if uploaded_file and st.button("📥 Stage Uploaded File"):
            content = uploaded_file.read()
            if uploaded_file.name.endswith(".csv"):
                st.session_state["staged_tweets"] = parse_csv_archive(content, new_handle)
            else:
                st.session_state["staged_tweets"] = parse_json_archive(content, new_handle)
            st.success(f"Staged {len(st.session_state['staged_tweets'])} tweets from archive!")

    elif ingest_method == "🔍 Lookup Public Handle":
        st.markdown(f"Fetch public tweets for **{new_handle}** from web syndication:")
        if st.button("🔍 Search & Stage Public Posts"):
            with st.spinner("Searching public posts on X …"):
                found = fetch_public_posts_from_handle(new_handle)
                if found:
                    st.session_state["staged_tweets"] = found
                    st.success(f"Found and staged {len(found)} public posts for @{new_handle.lstrip('@')}!")
                else:
                    st.warning(f"Could not automatically retrieve public posts for {new_handle} (Twitter/X blocks scrapers). Please use the '✍️ Paste Recent Tweets' tab to paste 5-15 tweets directly!")

    # If tweets are staged, display preview and the single Clone button
    staged = st.session_state.get("staged_tweets", [])
    if staged:
        st.markdown("---")
        st.info(f"📋 **Staged Tweets Ready:** {len(staged)} tweets prepared for **{new_handle}**")
        with st.expander("👁️ Preview Staged Tweets (First 5)"):
            for t in staged[:5]:
                st.markdown(f"- \"{t['text']}\"")

        if st.button("🧬 Analyze & Clone Voice Now", type="primary", use_container_width=True):
            embed_model, collection, nlp = get_shared_resources()

            status_placeholder = st.empty()
            def update_status(msg):
                status_placeholder.info(msg)

            with st.spinner("Cloning voice across stylometric & semantic layers …"):
                profile_result = build_and_index_profile(
                    new_handle,
                    staged,
                    nlp=nlp,
                    embed_model=embed_model,
                    collection=collection,
                    status_callback=update_status,
                )

            new_user_id = profile_result["user_id"]
            st.session_state["active_persona"] = new_user_id
            st.session_state["main_tab_persona_select"] = new_user_id
            st.session_state["staged_tweets"] = []
            st.balloons()
            st.success(f"🎉 Cloned **@{new_user_id}**! It is now automatically selected as your active persona.")
            st.rerun()

    # Cloned Personas Library Section
    st.markdown("---")
    st.subheader("📚 Cloned Personas Library")
    st.caption("All personas currently cloned and ready to tweet:")

    current_personas = discover_available_personas()
    for uid, label in current_personas.items():
        lib_col1, lib_col2 = st.columns([4, 1])
        with lib_col1:
            is_active = (uid == st.session_state.get("active_persona"))
            status_tag = " ✅ *(Currently Active)*" if is_active else ""
            st.markdown(f"**{label}**{status_tag}")
        with lib_col2:
            if not is_active:
                if st.button("Select & Tweet", key=f"activate_{uid}", use_container_width=True):
                    st.session_state["active_persona"] = uid
                    st.session_state["main_tab_persona_select"] = uid
                    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3: BASELINE BENCHMARKS
# ═════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.subheader("Quantitative Evolution Across Baselines")
    eval_path = DATA_DIR / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            ev = json.load(f)

        b0_s = ev["baseline_0_generic"]["summary"]
        b1_s = ev["baseline_1_rag"]["summary"]
        b2_s = ev["baseline_2_persona_rag"]["summary"]

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Persona Fidelity", f"{b2_s['mean_fidelity_score']}/100", f"+{b2_s['mean_fidelity_score']-b0_s['mean_fidelity_score']:.1f} vs B0")
        m_col2.metric("Word Error", f"{b2_s['mean_word_error']} wds", f"-{b0_s['mean_word_error']-b2_s['mean_word_error']:.1f} wds")
        m_col3.metric("Casing Accuracy", f"{b2_s['casing_accuracy_pct']}%", f"+{b2_s['casing_accuracy_pct']-b0_s['casing_accuracy_pct']:.1f}%")
        m_col4.metric("ROUGE-L Recall", f"{b2_s['mean_rougeL']:.3f}", "Author Vocabulary")

    report_path = DATA_DIR / "evaluation_report.md"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4: LINGUISTIC SIGNATURES
# ═════════════════════════════════════════════════════════════════════════════
with tab_style:
    st.subheader("Comparative Stylometric Signatures")
    style_md_path = PROFILES_DIR / "stylometric_comparison.md"
    if style_md_path.exists():
        with open(style_md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 5: PERSONA SYSTEM PROMPTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_prompts:
    st.subheader(f"Active Persona System Prompt: @{selected_user}")
    st.text_area(
        "Compiled System Prompt",
        value=persona_data.get("system_prompt", ""),
        height=220,
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Key Stylistic Rules:**")
        for r in persona_data.get("stylistic_rules", []):
            st.markdown(f"- {r}")
    with col_p2:
        vocab = persona_data.get("vocabulary_and_phrasing", {})
        st.markdown("**Characteristic Expressions:**")
        st.write(vocab.get("characteristic_words", []))
        st.markdown("**Banned AI Clichés:**")
        st.write(vocab.get("banned_cliches", []))
