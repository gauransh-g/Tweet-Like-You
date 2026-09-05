import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Flame, 
  Target, 
  Brain, 
  Zap, 
  Share2, 
  Copy, 
  Check, 
  Globe, 
  UserPlus, 
  RefreshCw, 
  MessageSquare,
  Repeat,
  Heart,
  Sliders,
  Scissors,
  ExternalLink,
  ShieldCheck,
  User,
  Trash2,
  Sun,
  Moon,
  LogOut,
  Layers,
  Lock,
  Mail,
  X
} from 'lucide-react';

const QUICK_TOPICS = [
  "India vs Australia test match drama and team performance",
  "Nvidia Blackwell GPU demand and AI infrastructure",
  "OpenAI autonomous agent release and safety concerns",
  "Whether Bollywood mass cinema is making a comeback",
  "The Brutalist and its 3.5 hour runtime"
];

export default function App() {
  // Theme state: dark / light
  const [theme, setTheme] = useState(() => localStorage.getItem("app_theme") || "dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("app_theme", theme);
  }, [theme]);

  // Auth / Workspace state
  const [currentUser, setCurrentUser] = useState(() => {
    const saved = localStorage.getItem("auth_user");
    return saved ? JSON.parse(saved) : {
      email: "demo@tweetlikeyou.com",
      name: "Arjun (Creator)",
      role: "Content Creator"
    };
  });
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginError, setLoginError] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);

  // App core state
  const [personas, setPersonas] = useState([]);
  const [activeUser, setActiveUser] = useState(null);
  const [topic, setTopic] = useState("India vs Australia test match drama and team performance");
  const [languageBlend, setLanguageBlend] = useState("auto");
  const [useGrounding, setUseGrounding] = useState(true);
  const [loading, setLoading] = useState(false);
  const [generationData, setGenerationData] = useState(null);
  const [copiedId, setCopiedId] = useState(null);
  const [transformingId, setTransformingId] = useState(null);
  
  // Clone Tab State
  const [activeTab, setActiveTab] = useState("studio"); // "studio" | "clone"
  const [cloneHandle, setCloneHandle] = useState("");
  const [cloneMethod, setCloneMethod] = useState("paste"); // "paste" | "handle"
  const [pasteTweets, setPasteTweets] = useState("");
  const [cloneLoading, setCloneLoading] = useState(false);
  const [cloneStatus, setCloneStatus] = useState("");

  // Load available personas for active user's workspace
  const fetchPersonas = async (userEmail) => {
    try {
      const email = userEmail !== undefined ? userEmail : (currentUser?.email || "");
      const res = await fetch(`/api/personas?user_email=${encodeURIComponent(email)}`);
      const data = await res.json();
      if (data.personas) {
        setPersonas(data.personas);
        if (data.personas.length > 0) {
          setActiveUser(prev => {
            if (prev && data.personas.some(p => p.user_id === prev.user_id)) {
              return prev;
            }
            return data.personas[0];
          });
        } else {
          setActiveUser(null);
        }
      }
    } catch (e) {
      console.error("Failed to load personas:", e);
    }
  };

  useEffect(() => {
    fetchPersonas(currentUser?.email);
  }, [currentUser?.email]);

  // Handle Login submission
  const handleLoginSubmit = async (e, manualEmail, manualPass) => {
    if (e) e.preventDefault();
    const emailToUse = manualEmail || loginEmail;
    const passToUse = manualPass || loginPassword;

    if (!emailToUse || !passToUse) return;
    setLoginLoading(true);
    setLoginError("");

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: emailToUse.trim(), password: passToUse.trim() })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        const userObj = {
          email: data.user.email,
          name: data.user.name,
          role: data.user.role
        };
        setCurrentUser(userObj);
        localStorage.setItem("auth_user", JSON.stringify(userObj));
        setShowLoginModal(false);
        setLoginEmail("");
        setLoginPassword("");
        await fetchPersonas(userObj.email);
      } else {
        setLoginError(data.detail || "Invalid credentials");
      }
    } catch (err) {
      setLoginError("Login connection error: " + err.message);
    } finally {
      setLoginLoading(false);
    }
  };

  // Delete Persona
  const handleDeletePersona = async (userId, displayName) => {
    const ok = window.confirm(`Permanently remove @${userId} (${displayName}) from your workspace?`);
    if (!ok) return;

    try {
      const email = currentUser?.email || "";
      const res = await fetch(`/api/persona/${userId}?user_email=${encodeURIComponent(email)}`, {
        method: "DELETE"
      });
      const data = await res.json();
      if (res.ok && data.success) {
        await fetchPersonas(currentUser?.email);
      } else {
        alert("Failed to delete persona: " + (data.detail || "Server error"));
      }
    } catch (err) {
      alert("Error deleting persona: " + err.message);
    }
  };

  // Generate 4-Angle Editorial Matrix
  const handleGenerate = async () => {
    if (!activeUser || !topic.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: topic.trim(),
          user_id: activeUser.user_id,
          language_blend: languageBlend,
          use_grounding: useGrounding,
          n_candidates: 4
        })
      });
      const data = await res.json();
      if (res.ok) {
        setGenerationData(data);
      } else {
        alert("Generation failed: " + (data.detail || "Server error"));
      }
    } catch (e) {
      alert("Error contacting API: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  // One-click tweet transformation
  const handleTransform = async (angleId, currentText, action) => {
    setTransformingId(angleId);
    try {
      const res = await fetch("/api/transform", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tweet_text: currentText,
          user_id: activeUser.user_id,
          action
        })
      });
      const data = await res.json();
      if (data.success && generationData) {
        const updatedAngles = generationData.editorial_angles.map(a => {
          if (a.angle_id === angleId) {
            return { ...a, text: data.transformed_text };
          }
          return a;
        });
        setGenerationData({ ...generationData, editorial_angles: updatedAngles });
      }
    } catch (e) {
      alert("Transform failed: " + e.message);
    } finally {
      setTransformingId(null);
    }
  };

  const copyTweet = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleCloneSubmit = async (e) => {
    e.preventDefault();
    if (!cloneHandle.trim()) return;
    setCloneLoading(true);
    setCloneStatus("Analyzing linguistic DNA and compiling ChromaDB vectors...");

    try {
      const res = await fetch("/api/clone", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          handle: cloneHandle.trim(),
          method: cloneMethod,
          pasted_text: pasteTweets,
          user_email: currentUser?.email || ""
        })
      });
      const data = await res.json();
      if (res.ok) {
        setCloneStatus(`Successfully cloned @${data.user_id}`);
        await fetchPersonas(currentUser?.email);
        setTimeout(() => {
          setActiveTab("studio");
          setCloneStatus("");
          setPasteTweets("");
          setCloneHandle("");
        }, 1200);
      } else {
        setCloneStatus("Error: " + (data.detail || "Cloning failed"));
      }
    } catch (err) {
      setCloneStatus("Connection error: " + err.message);
    } finally {
      setCloneLoading(false);
    }
  };

  const getAngleIcon = (id) => {
    switch (id) {
      case "hot_take": return <Flame size={14} color="#fb7185" />;
      case "signature_classic": return <Target size={14} color="#10b981" />;
      case "deep_analysis": return <Brain size={14} color="#0284c7" />;
      case "punchy_hook": return <Zap size={14} color="#d97706" />;
      default: return <Sparkles size={14} color="#10b981" />;
    }
  };

  const getAngleBadgeClass = (id) => {
    switch (id) {
      case "hot_take": return "badge-hot-take";
      case "signature_classic": return "badge-classic";
      case "deep_analysis": return "badge-analysis";
      case "punchy_hook": return "badge-punchy";
      default: return "";
    }
  };

  return (
    <div style={{ maxWidth: "1280px", margin: "0 auto", padding: "28px 24px 80px" }}>
      {/* ── Top Header ─────────────────────────────────────────── */}
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "32px", flexWrap: "wrap", gap: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <img 
            src="/logo.png" 
            alt="Tweet Like You Emblem" 
            style={{ 
              width: "36px", 
              height: "36px", 
              borderRadius: "8px", 
              objectFit: "cover", 
              border: "1px solid var(--border-subtle)",
              boxShadow: "0 2px 8px rgba(0,0,0,0.15)"
            }} 
          />
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <h1 style={{ fontSize: "1.55rem", fontWeight: "700", color: "var(--text-main)" }}>
                Tweet Like You
              </h1>
              <span className="badge" style={{ background: "rgba(125, 125, 125, 0.1)", color: "var(--text-dim)", border: "1px solid var(--border-subtle)" }}>
                v2.5
              </span>
            </div>
            <p style={{ color: "var(--text-dim)", fontSize: "0.82rem", marginTop: "2px" }}>
              Universal Social Voice Cloner, Multilingual Engine & Editorial Matrix
            </p>
          </div>
        </div>

        {/* Action Controls & Navigation */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
          {/* Light / Dark Mode Toggle */}
          <button 
            onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
            className="btn-secondary"
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            style={{ padding: "8px 11px" }}
          >
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </button>

          {/* Workspace / Login Account Selector */}
          <button
            onClick={() => setShowLoginModal(true)}
            className="btn-secondary"
            style={{ display: "flex", alignItems: "center", gap: "7px" }}
            title="Switch Workspace / Account"
          >
            <User size={14} color="var(--primary-accent)" />
            <span style={{ fontWeight: "600" }}>{currentUser?.name || "Workspace"}</span>
            <span style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
              ({personas.length} personas)
            </span>
          </button>

          {/* Navigation Switch */}
          <div style={{ display: "flex", gap: "4px", background: "var(--pill-bg)", padding: "3px", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
            <button 
              onClick={() => setActiveTab("studio")} 
              className={`btn-pill ${activeTab === "studio" ? "active" : ""}`}
              style={{ display: "flex", alignItems: "center", gap: "6px" }}
            >
              <Sliders size={13} />
              Studio
            </button>
            <button 
              onClick={() => setActiveTab("clone")} 
              className={`btn-pill ${activeTab === "clone" ? "active" : ""}`}
              style={{ display: "flex", alignItems: "center", gap: "6px" }}
            >
              <UserPlus size={13} />
              Clone Voice
            </button>
          </div>
        </div>
      </header>

      {/* ── Tab 1: Studio Mode ─────────────────────────────────── */}
      {activeTab === "studio" && (
        <>
          {/* Persona Selection Carousel */}
          <section style={{ marginBottom: "26px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--text-muted)", fontSize: "0.82rem", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                <User size={14} />
                <span>Active Persona Workspace</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span style={{ fontSize: "0.76rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
                  {personas.length} profiles available
                </span>
              </div>
            </div>

            {personas.length === 0 ? (
              <div className="glass-panel" style={{ padding: "32px", textAlign: "center" }}>
                <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginBottom: "14px" }}>
                  No personas in this workspace yet.
                </p>
                <button onClick={() => setActiveTab("clone")} className="btn-primary">
                  <UserPlus size={15} /> Clone a Profile Now
                </button>
              </div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: "12px" }}>
                {personas.map((p) => {
                  const isSelected = activeUser?.user_id === p.user_id;
                  return (
                    <div
                      key={p.user_id}
                      onClick={() => setActiveUser(p)}
                      className="glass-panel"
                      style={{
                        padding: "14px 16px",
                        cursor: "pointer",
                        border: isSelected ? "1px solid var(--border-active)" : "1px solid var(--border-subtle)",
                        background: isSelected ? "rgba(16, 185, 129, 0.05)" : "var(--bg-card)",
                        position: "relative"
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "8px", marginBottom: "8px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px", overflow: "hidden" }}>
                          <div className="avatar-circle">
                            {p.avatar}
                          </div>
                          <div style={{ overflow: "hidden" }}>
                            <div style={{ fontWeight: "600", fontSize: "0.9rem", color: "var(--text-main)", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
                              {p.display_name}
                            </div>
                            <div style={{ fontSize: "0.78rem", color: "var(--text-dim)" }}>
                              @{p.user_id}
                            </div>
                          </div>
                        </div>

                        {/* Delete Persona Icon Button */}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeletePersona(p.user_id, p.display_name);
                          }}
                          className="btn-delete-persona"
                          title={`Delete @${p.user_id}`}
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>

                      <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "8px", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden", lineHeight: "1.4" }}>
                        {p.role || "Custom Cloned Voice"}
                      </div>

                      <div style={{ display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap" }}>
                        {p.is_vernacular_adapted && (
                          <span className="badge badge-hinglish" style={{ fontSize: "0.68rem", padding: "2px 6px" }}>
                            Hinglish Active
                          </span>
                        )}
                        <span style={{ fontSize: "0.72rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
                          ~{p.voice_dna?.target_words || 25} wds · {p.voice_dna?.lowercase_starts || 0}% lower
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          {/* Composer Box */}
          <section className="glass-panel" style={{ padding: "20px 24px", marginBottom: "30px" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px", flexWrap: "wrap", gap: "10px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <Sliders size={16} color="var(--primary-accent)" />
                <h2 style={{ fontSize: "1.05rem", fontWeight: "600", color: "var(--text-main)" }}>
                  Prompt & Topic Composer
                </h2>
              </div>

              {/* Language Blend Selector */}
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ fontSize: "0.78rem", color: "var(--text-dim)" }}>Language:</span>
                {["auto", "hinglish", "english"].map((lang) => (
                  <button
                    key={lang}
                    onClick={() => setLanguageBlend(lang)}
                    className={`btn-pill ${languageBlend === lang ? "active" : ""}`}
                    style={{ textTransform: "capitalize", fontSize: "0.72rem", padding: "3px 8px" }}
                  >
                    {lang === "hinglish" ? "Hinglish" : lang}
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Topic Chips */}
            <div style={{ display: "flex", gap: "6px", overflowX: "auto", paddingBottom: "8px", marginBottom: "12px" }}>
              {QUICK_TOPICS.map((q, i) => (
                <button
                  key={i}
                  onClick={() => setTopic(q)}
                  className="btn-pill"
                  style={{ whiteSpace: "nowrap", flexShrink: 0, fontSize: "0.75rem" }}
                >
                  {q.slice(0, 34)}…
                </button>
              ))}
            </div>

            {/* Topic Input */}
            <div style={{ position: "relative", marginBottom: "14px" }}>
              <textarea
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                rows={2}
                placeholder="Enter topic, breaking news, or statement..."
                style={{
                  width: "100%",
                  padding: "12px 14px",
                  background: "var(--input-bg)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "8px",
                  color: "var(--text-main)",
                  fontSize: "0.92rem",
                  fontFamily: "var(--font-body)",
                  resize: "none",
                  outline: "none"
                }}
              />
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px" }}>
              <label style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", fontSize: "0.82rem", color: "var(--text-muted)" }}>
                <input
                  type="checkbox"
                  checked={useGrounding}
                  onChange={(e) => setUseGrounding(e.target.checked)}
                  style={{ accentColor: "var(--primary-accent)" }}
                />
                <Globe size={15} color={useGrounding ? "var(--blue-accent)" : "var(--text-dim)"} />
                Real-Time Web News Grounding
              </label>

              <button
                onClick={handleGenerate}
                disabled={loading || !activeUser}
                className="btn-primary"
                style={{ minWidth: "190px" }}
              >
                {loading ? (
                  <>
                    <RefreshCw size={15} className="animate-spin" />
                    Generating Matrix...
                  </>
                ) : (
                  <>
                    <Sparkles size={15} />
                    Generate Matrix
                  </>
                )}
              </button>
            </div>
          </section>

          {/* ── Generated 4-Angle Editorial Matrix ────────────────────────── */}
          {generationData && (
            <section style={{ marginBottom: "40px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px", flexWrap: "wrap", gap: "8px" }}>
                <div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: "700", display: "flex", alignItems: "center", gap: "8px" }}>
                    Editorial Matrix
                  </h2>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-dim)" }}>
                    Author: <strong>@{generationData.user_id}</strong> · Language: <strong>{generationData.language_blend}</strong> · Category: <strong>{generationData.category}</strong>
                  </p>
                </div>

                <span className="badge badge-classic" style={{ fontSize: "0.72rem" }}>
                  Champion: {generationData.champion_angle} ({generationData.champion_score}/100)
                </span>
              </div>

              {/* Grid of 4 Distinct Editorial Angles */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(310px, 1fr))", gap: "16px" }}>
                {generationData.editorial_angles.map((angle) => {
                  const isChampion = generationData.champion_angle === angle.angle_name;
                  const encodedText = encodeURIComponent(angle.text);
                  const twitterUrl = `https://twitter.com/intent/tweet?text=${encodedText}`;

                  return (
                    <div 
                      key={angle.angle_id} 
                      className={`tweet-card ${isChampion ? "is-champion" : ""}`}
                    >
                      {/* Header */}
                      <div className="tweet-card-header">
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                          {getAngleIcon(angle.angle_id)}
                          <span className={`badge ${getAngleBadgeClass(angle.angle_id)}`}>
                            {angle.angle_name}
                          </span>
                        </div>
                        <span style={{ fontSize: "0.72rem", fontFamily: "var(--font-mono)", color: "var(--text-dim)" }}>
                          Score {angle.composite_score}/100 · {angle.viral_potential}% Virality
                        </span>
                      </div>

                      {/* Author row */}
                      <div className="tweet-user-info" style={{ marginBottom: "10px" }}>
                        <div className="avatar-circle" style={{ width: "32px", height: "32px", fontSize: "0.85rem" }}>
                          {activeUser?.avatar}
                        </div>
                        <div>
                          <div className="author-name" style={{ fontSize: "0.85rem" }}>
                            {activeUser?.display_name}
                            <ShieldCheck size={13} color="var(--blue-accent)" />
                          </div>
                          <div className="author-handle">
                            @{generationData.user_id}
                          </div>
                        </div>
                      </div>

                      {/* Tweet Content */}
                      <div className="tweet-text-body">
                        {angle.text}
                      </div>

                      {/* Stats & Actions Row */}
                      <div className="tweet-stats-bar">
                        <div style={{ display: "flex", gap: "14px" }}>
                          <span className="stat-item"><MessageSquare size={13} /> 312</span>
                          <span className="stat-item"><Repeat size={13} /> 1.4K</span>
                          <span className="stat-item"><Heart size={13} /> 8.9K</span>
                        </div>
                        <span style={{ fontSize: "0.72rem", fontFamily: "var(--font-mono)" }}>
                          {angle.word_count} wds · {angle.char_count} chars
                        </span>
                      </div>

                      {/* Action Bar */}
                      <div style={{ display: "flex", gap: "6px", marginTop: "12px" }}>
                        <button
                          onClick={() => copyTweet(angle.angle_id, angle.text)}
                          className="btn-secondary"
                          style={{ flex: 1 }}
                        >
                          {copiedId === angle.angle_id ? <Check size={14} color="var(--primary-accent)" /> : <Copy size={14} />}
                          {copiedId === angle.angle_id ? "Copied" : "Copy"}
                        </button>
                        <a
                          href={twitterUrl}
                          target="_blank"
                          rel="noreferrer"
                          className="btn-primary"
                          style={{ textDecoration: "none", padding: "6px 12px", fontSize: "0.8rem" }}
                        >
                          <ExternalLink size={13} /> Post to X
                        </a>
                      </div>

                      {/* Quick Transformation Strip */}
                      <div className="transform-strip">
                        <button
                          onClick={() => handleTransform(angle.angle_id, angle.text, "spicier")}
                          disabled={transformingId === angle.angle_id}
                          className="btn-transform"
                        >
                          <Flame size={11} /> Spicier
                        </button>
                        <button
                          onClick={() => handleTransform(angle.angle_id, angle.text, "hinglish")}
                          disabled={transformingId === angle.angle_id}
                          className="btn-transform"
                        >
                          <Globe size={11} /> Hinglish Blend
                        </button>
                        <button
                          onClick={() => handleTransform(angle.angle_id, angle.text, "shorten")}
                          disabled={transformingId === angle.angle_id}
                          className="btn-transform"
                        >
                          <Scissors size={11} /> Shorten
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Real-Time Live Grounding Radar */}
              {generationData.grounding && (generationData.grounding.live_news?.length > 0 || generationData.grounding.factual_points?.length > 0) && (
                <div className="glass-panel" style={{ marginTop: "20px", padding: "16px 20px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "10px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <Globe size={15} color="var(--blue-accent)" />
                      <h4 style={{ fontSize: "0.92rem", fontWeight: "600", color: "var(--text-main)" }}>
                        Verified Web & Real-Time Context
                      </h4>
                    </div>
                    <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
                      Anchor: {generationData.grounding.current_date_anchor || "Present"}
                    </span>
                  </div>

                  {generationData.grounding.live_news?.length > 0 && (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "8px" }}>
                      {generationData.grounding.live_news.slice(0, 3).map((n, i) => (
                        <div key={i} style={{ background: "rgba(125, 125, 125, 0.08)", padding: "8px 12px", borderRadius: "6px", border: "1px solid var(--border-subtle)" }}>
                          <div style={{ fontSize: "0.8rem", fontWeight: "600", color: "var(--text-main)", marginBottom: "3px" }}>
                            {n.title}
                          </div>
                          <div style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
                            {n.source} · {n.snippet}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </section>
          )}
        </>
      )}

      {/* ── Tab 2: Clone Any Voice ─────────────────────────────── */}
      {activeTab === "clone" && (
        <section className="glass-panel" style={{ padding: "28px", maxWidth: "760px", margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "10px" }}>
            <UserPlus size={20} color="var(--primary-accent)" />
            <h2 style={{ fontSize: "1.3rem", fontWeight: "700" }}>
              Clone Writing Voice into Workspace
            </h2>
          </div>
          <p style={{ color: "var(--text-muted)", fontSize: "0.88rem", marginBottom: "20px" }}>
            Cloned personas will be added specifically to <strong>{currentUser?.name}</strong>'s workspace.
          </p>

          <form onSubmit={handleCloneSubmit}>
            <div style={{ marginBottom: "16px" }}>
              <label style={{ display: "block", fontSize: "0.82rem", fontWeight: "600", color: "var(--text-muted)", marginBottom: "6px" }}>
                Twitter / X Handle:
              </label>
              <div style={{ display: "flex", alignItems: "center", background: "var(--input-bg)", border: "1px solid var(--border-subtle)", borderRadius: "8px", padding: "0 12px" }}>
                <span style={{ color: "var(--text-dim)", fontWeight: "600" }}>@</span>
                <input
                  type="text"
                  required
                  placeholder="handle"
                  value={cloneHandle}
                  onChange={(e) => setCloneHandle(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "10px 8px",
                    background: "transparent",
                    border: "none",
                    color: "var(--text-main)",
                    fontSize: "0.9rem",
                    outline: "none"
                  }}
                />
              </div>
            </div>

            <div style={{ marginBottom: "16px" }}>
              <label style={{ display: "block", fontSize: "0.82rem", fontWeight: "600", color: "var(--text-muted)", marginBottom: "6px" }}>
                Ingestion Method:
              </label>
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  type="button"
                  onClick={() => setCloneMethod("paste")}
                  className={`btn-pill ${cloneMethod === "paste" ? "active" : ""}`}
                >
                  Paste Tweets
                </button>
                <button
                  type="button"
                  onClick={() => setCloneMethod("handle")}
                  className={`btn-pill ${cloneMethod === "handle" ? "active" : ""}`}
                >
                  Public Web Search
                </button>
              </div>
            </div>

            {cloneMethod === "paste" && (
              <div style={{ marginBottom: "20px" }}>
                <label style={{ display: "block", fontSize: "0.82rem", fontWeight: "600", color: "var(--text-muted)", marginBottom: "6px" }}>
                  Paste Past Tweets (one per line):
                </label>
                <textarea
                  required
                  rows={5}
                  placeholder={`bhai match kya tight gaya! pura paisa vasool match tha...\nAlways back yourself when the chips are down.\nekdum badiya performance by the team!`}
                  value={pasteTweets}
                  onChange={(e) => setPasteTweets(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "12px",
                    background: "var(--input-bg)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "8px",
                    color: "var(--text-main)",
                    fontSize: "0.85rem",
                    fontFamily: "var(--font-mono)",
                    resize: "vertical",
                    outline: "none"
                  }}
                />
              </div>
            )}

            {cloneStatus && (
              <div style={{ padding: "10px 14px", borderRadius: "6px", background: "rgba(125, 125, 125, 0.08)", border: "1px solid var(--border-subtle)", marginBottom: "16px", fontSize: "0.85rem", color: "var(--text-main)" }}>
                {cloneStatus}
              </div>
            )}

            <button
              type="submit"
              disabled={cloneLoading}
              className="btn-primary"
              style={{ width: "100%", padding: "11px" }}
            >
              {cloneLoading ? (
                <>
                  <RefreshCw size={15} className="animate-spin" />
                  Indexing Vectors...
                </>
              ) : (
                <>
                  <UserPlus size={15} />
                  Analyze & Clone Voice
                </>
              )}
            </button>
          </form>
        </section>
      )}

      {/* ── Login & Workspace Modal ─────────────────────────────── */}
      {showLoginModal && (
        <div className="modal-overlay" onClick={() => setShowLoginModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <Layers size={18} color="var(--primary-accent)" />
                <h3 style={{ fontSize: "1.15rem", fontWeight: "700" }}>Account & Workspaces</h3>
              </div>
              <button 
                onClick={() => setShowLoginModal(false)}
                style={{ background: "transparent", border: "none", color: "var(--text-dim)", cursor: "pointer" }}
              >
                <X size={18} />
              </button>
            </div>

            <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginBottom: "18px" }}>
              Select a pre-configured demo account or enter your credentials to access your isolated workspace:
            </p>

            {/* Quick Demo Login Cards */}
            <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginBottom: "20px" }}>
              <button
                type="button"
                onClick={() => handleLoginSubmit(null, "demo@tweetlikeyou.com", "demo123")}
                className="glass-panel"
                style={{ 
                  padding: "12px 14px", 
                  textAlign: "left", 
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  border: currentUser?.email === "demo@tweetlikeyou.com" ? "1px solid var(--border-active)" : "1px solid var(--border-subtle)"
                }}
              >
                <div>
                  <div style={{ fontWeight: "600", fontSize: "0.88rem", color: "var(--text-main)" }}>
                    Arjun (Content Creator)
                  </div>
                  <div style={{ fontSize: "0.76rem", color: "var(--text-dim)" }}>
                    demo@tweetlikeyou.com · Salman Khan, Ravi Shastri
                  </div>
                </div>
                <span className="badge badge-classic" style={{ fontSize: "0.68rem" }}>Demo 1</span>
              </button>

              <button
                type="button"
                onClick={() => handleLoginSubmit(null, "founder@silicon.ai", "founder123")}
                className="glass-panel"
                style={{ 
                  padding: "12px 14px", 
                  textAlign: "left", 
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  border: currentUser?.email === "founder@silicon.ai" ? "1px solid var(--border-active)" : "1px solid var(--border-subtle)"
                }}
              >
                <div>
                  <div style={{ fontWeight: "600", fontSize: "0.88rem", color: "var(--text-main)" }}>
                    Sarah (Tech Founder)
                  </div>
                  <div style={{ fontSize: "0.76rem", color: "var(--text-dim)" }}>
                    founder@silicon.ai · Sam Altman, Tech Critics
                  </div>
                </div>
                <span className="badge badge-analysis" style={{ fontSize: "0.68rem" }}>Demo 2</span>
              </button>
            </div>

            <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "16px", marginBottom: "16px" }}>
              <div style={{ fontSize: "0.76rem", color: "var(--text-dim)", textTransform: "uppercase", fontWeight: "600", marginBottom: "12px" }}>
                Or Sign In Manually
              </div>

              <form onSubmit={(e) => handleLoginSubmit(e)}>
                <div style={{ marginBottom: "10px" }}>
                  <div style={{ display: "flex", alignItems: "center", background: "var(--input-bg)", border: "1px solid var(--border-subtle)", borderRadius: "8px", padding: "0 10px" }}>
                    <Mail size={14} color="var(--text-dim)" />
                    <input
                      type="email"
                      required
                      placeholder="Email address"
                      value={loginEmail}
                      onChange={(e) => setLoginEmail(e.target.value)}
                      style={{ width: "100%", padding: "9px 8px", background: "transparent", border: "none", color: "var(--text-main)", fontSize: "0.85rem", outline: "none" }}
                    />
                  </div>
                </div>

                <div style={{ marginBottom: "12px" }}>
                  <div style={{ display: "flex", alignItems: "center", background: "var(--input-bg)", border: "1px solid var(--border-subtle)", borderRadius: "8px", padding: "0 10px" }}>
                    <Lock size={14} color="var(--text-dim)" />
                    <input
                      type="password"
                      required
                      placeholder="Password"
                      value={loginPassword}
                      onChange={(e) => setLoginPassword(e.target.value)}
                      style={{ width: "100%", padding: "9px 8px", background: "transparent", border: "none", color: "var(--text-main)", fontSize: "0.85rem", outline: "none" }}
                    />
                  </div>
                </div>

                {loginError && (
                  <div style={{ color: "var(--rose-accent)", fontSize: "0.78rem", marginBottom: "10px" }}>
                    {loginError}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loginLoading}
                  className="btn-primary"
                  style={{ width: "100%", padding: "9px" }}
                >
                  {loginLoading ? "Authenticating..." : "Sign In to Workspace"}
                </button>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
