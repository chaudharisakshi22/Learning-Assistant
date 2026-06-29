"""
app.py  –  Personalized Learning Assistant
───────────────────────────────────────────
Run with:  streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from modules.document_loader   import load_any
from modules.vector_store      import (
    build_vectorstore, load_vectorstore, add_documents, get_retriever
)
from modules.quiz_generator    import generate_quiz, parse_quiz
from modules.performance_tracker import (
    record_quiz_result, get_summary_stats, get_weak_topics
)
from modules.recommender       import get_recommendations

load_dotenv()

st.set_page_config(
    page_title="Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --ink:    #1a1a2e;
    --slate:  #16213e;
    --accent: #0f3460;
    --blue:   #4361ee;
    --teal:   #4cc9f0;
    --green:  #06d6a0;
    --red:    #ef233c;
    --bg:     #f0f4ff;
    --card:   #ffffff;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #16213e;
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stButton button {
    background: var(--blue);
    color: white !important;
    border-radius: 8px;
    border: none;
    width: 100%;
}
.card {
    background: var(--card);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 2px 12px rgba(67,97,238,.08);
    margin-bottom: 1.2rem;
}
.card-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: .8rem;
}
.quiz-q {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: .6rem;
}
.correct   { color: #06d6a0; font-weight: 700; }
.incorrect { color: #ef233c; font-weight: 700; }
.stProgress > div > div { background: var(--blue); }
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: .8rem;
    box-shadow: 0 2px 8px rgba(67,97,238,.07);
}
.stButton > button {
    background: var(--blue);
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .88; }

/* Welcome banner */
.welcome-banner {
    background: linear-gradient(135deg, #0f3460 0%, #4361ee 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.feature-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(67,97,238,.08);
    border-top: 3px solid #4361ee;
}
            /* Fix file uploader in dark sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #1a365d;
    border-radius: 10px;
    border: 2px dashed #4361ee;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: #1a365d !important;
    color: white !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section > div {
    background: #1a365d !important;
    color: white !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #4361ee !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
}

[data-testid="stSidebar"] small {
    color: #a0aec0 !important;
}           
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
for key, default in {
    "db": None,
    "quiz_questions": [],
    "quiz_topic": "",
    "quiz_answers": {},
    "quiz_submitted": False,
    "all_topics": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.db is None:
    st.session_state.db = load_vectorstore()

# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.2rem 0 .5rem 0;">
        <div style="font-size:2.8rem;">🎓</div>
        <div style="font-size:1.25rem; font-weight:800; color:white; letter-spacing:.5px;">
            Learning Assistant
        </div>
        <div style="font-size:.75rem; color:#a0aec0; margin-top:2px;">
            Powered by HuggingFace · LangChain · RAG
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="margin-bottom:1rem;">
        <div style="font-size:.72rem; font-weight:700; color:#a0aec0;
                    letter-spacing:1.5px; text-transform:uppercase; margin-bottom:.7rem;">
            How It Works
        </div>
        <div style="display:flex; align-items:flex-start; gap:.6rem; margin-bottom:.6rem;">
            <div style="background:#4361ee; color:white; border-radius:50%;
                        width:22px; height:22px; display:flex; align-items:center;
                        justify-content:center; font-size:.7rem; font-weight:700; flex-shrink:0;">1</div>
            <div style="color:#cbd5e0; font-size:.8rem; padding-top:2px;">Upload your study material below</div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:.6rem; margin-bottom:.6rem;">
            <div style="background:#4361ee; color:white; border-radius:50%;
                        width:22px; height:22px; display:flex; align-items:center;
                        justify-content:center; font-size:.7rem; font-weight:700; flex-shrink:0;">2</div>
            <div style="color:#cbd5e0; font-size:.8rem; padding-top:2px;">Go to Quiz Generator tab</div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:.6rem; margin-bottom:.6rem;">
            <div style="background:#4361ee; color:white; border-radius:50%;
                        width:22px; height:22px; display:flex; align-items:center;
                        justify-content:center; font-size:.7rem; font-weight:700; flex-shrink:0;">3</div>
            <div style="color:#cbd5e0; font-size:.8rem; padding-top:2px;">Type a topic & generate AI quiz</div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:.6rem;">
            <div style="background:#06d6a0; color:white; border-radius:50%;
                        width:22px; height:22px; display:flex; align-items:center;
                        justify-content:center; font-size:.7rem; font-weight:700; flex-shrink:0;">4</div>
            <div style="color:#cbd5e0; font-size:.8rem; padding-top:2px;">Track progress & get recommendations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size:.72rem; font-weight:700; color:#a0aec0;
                letter-spacing:1.5px; text-transform:uppercase; margin-bottom:.7rem;">
        📂 Add Study Material
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supports PDF, TXT, and Markdown files"
    )

    url_input = st.text_input("Or paste a URL", placeholder="https://...")

    st.markdown("""
    <div style="display:flex; gap:.4rem; flex-wrap:wrap; margin:.3rem 0 .8rem 0;">
        <span style="background:#1a365d; color:#90cdf4; font-size:.65rem;
                     padding:2px 8px; border-radius:20px;">📄 PDF</span>
        <span style="background:#1a365d; color:#90cdf4; font-size:.65rem;
                     padding:2px 8px; border-radius:20px;">📝 TXT</span>
        <span style="background:#1a365d; color:#90cdf4; font-size:.65rem;
                     padding:2px 8px; border-radius:20px;">🌐 URL</span>
        <span style="background:#1a365d; color:#90cdf4; font-size:.65rem;
                     padding:2px 8px; border-radius:20px;">📋 MD</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ Ingest Material", use_container_width=True):
        sources_processed = 0
        all_docs = []

        if uploaded_files:
            for uf in uploaded_files:
                suffix = "." + uf.name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    docs = load_any(tmp_path)
                    all_docs.extend(docs)
                    sources_processed += 1
                except Exception as e:
                    st.error(f"Error loading {uf.name}: {e}")
                finally:
                    os.unlink(tmp_path)

        if url_input.strip():
            try:
                docs = load_any(url_input.strip())
                all_docs.extend(docs)
                sources_processed += 1
            except Exception as e:
                st.error(f"Error loading URL: {e}")

        if all_docs:
            with st.spinner("🔄 Embedding documents…"):
                if st.session_state.db is None:
                    st.session_state.db = build_vectorstore(all_docs)
                else:
                    st.session_state.db = add_documents(st.session_state.db, all_docs)
            st.success(f"✅ {len(all_docs)} chunks from {sources_processed} source(s) ingested!")
        elif sources_processed == 0:
            st.warning("Please upload a file or enter a URL first.")

    st.markdown("---")

    # Knowledge base status
    st.markdown("""
    <div style="font-size:.72rem; font-weight:700; color:#a0aec0;
                letter-spacing:1.5px; text-transform:uppercase; margin-bottom:.7rem;">
        📊 Knowledge Base
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.db:
        chunks = st.session_state.db.index.ntotal
        st.markdown(f"""
        <div style="background:#1a365d; border-radius:10px; padding:.9rem 1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="color:#90cdf4; font-size:.72rem;">Chunks Indexed</div>
                    <div style="color:white; font-size:1.6rem; font-weight:800;">{chunks}</div>
                </div>
                <div style="font-size:2rem;">🗄️</div>
            </div>
            <div style="margin-top:.5rem; background:#2d3748; border-radius:6px; height:6px;">
                <div style="background:#4cc9f0; width:100%; height:6px; border-radius:6px;"></div>
            </div>
            <div style="color:#68d391; font-size:.72rem; margin-top:.4rem;">✅ Ready for queries</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#2d3748; border-radius:10px; padding:.9rem 1rem;
                    border:1px dashed #4a5568;">
            <div style="text-align:center;">
                <div style="font-size:1.8rem; margin-bottom:.3rem;">📭</div>
                <div style="color:#a0aec0; font-size:.8rem;">No documents yet</div>
                <div style="color:#718096; font-size:.72rem; margin-top:.2rem;">
                    Upload material above to get started
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick stats
    stats        = get_summary_stats()
    topics_count = len(stats)
    weak_count   = len(get_weak_topics())
    weak_color   = "#fc8181" if weak_count else "#68d391"

    st.markdown(f"""
    <div style="font-size:.72rem; font-weight:700; color:#a0aec0;
                letter-spacing:1.5px; text-transform:uppercase; margin-bottom:.7rem;">
        🏆 Quick Stats
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:.5rem;">
        <div style="background:#1a365d; border-radius:8px; padding:.6rem; text-align:center;">
            <div style="color:white; font-size:1.3rem; font-weight:800;">{topics_count}</div>
            <div style="color:#90cdf4; font-size:.65rem;">Topics Studied</div>
        </div>
        <div style="background:#1a365d; border-radius:8px; padding:.6rem; text-align:center;">
            <div style="color:{weak_color}; font-size:1.3rem; font-weight:800;">{weak_count}</div>
            <div style="color:#90cdf4; font-size:.65rem;">Weak Areas</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main area
# ══════════════════════════════════════════════════════════════════════════════

# Welcome banner
st.markdown("""
<div class="welcome-banner">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
        <div>
            <div style="font-size:1.7rem; font-weight:800; margin-bottom:.3rem;">
                🎓 Personalized Learning Assistant
            </div>
            <div style="font-size:.95rem; opacity:.85;">
                Upload your study material → Generate AI quizzes → Track your progress → Get smart recommendations
            </div>
        </div>
        <div style="text-align:center; background:rgba(255,255,255,.15);
                    border-radius:12px; padding:.8rem 1.5rem;">
            <div style="font-size:2rem; font-weight:800;">RAG</div>
            <div style="font-size:.7rem; opacity:.8;">Powered</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature cards — only show when no docs ingested
if not st.session_state.db:
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div style="font-size:2rem; margin-bottom:.5rem;">📂</div>
            <div style="font-weight:700; color:#0f3460; margin-bottom:.3rem;">Upload Material</div>
            <div style="font-size:.82rem; color:#666;">PDF, TXT, or any URL — we chunk & embed it automatically</div>
        </div>
        <div class="feature-card">
            <div style="font-size:2rem; margin-bottom:.5rem;">📝</div>
            <div style="font-weight:700; color:#0f3460; margin-bottom:.3rem;">AI Quiz Generator</div>
            <div style="font-size:.82rem; color:#666;">RAG-powered MCQs generated strictly from your own material</div>
        </div>
        <div class="feature-card">
            <div style="font-size:2rem; margin-bottom:.5rem;">💡</div>
            <div style="font-weight:700; color:#0f3460; margin-bottom:.3rem;">Smart Recommendations</div>
            <div style="font-size:.82rem; color:#666;">AI coach identifies weak areas and builds your study plan</div>
        </div>
    </div>
    <div style="background:#fff3cd; border-radius:10px; padding:1rem 1.2rem;
                border-left:4px solid #ffc107; margin-bottom:1.2rem;">
        <b>👈 Get started:</b> Upload a PDF, TXT file, or paste a URL in the sidebar, then click <b>Ingest Material</b>.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab_quiz, tab_progress, tab_recommend = st.tabs(
    ["📝 Quiz Generator", "📊 My Progress", "💡 Recommendations"]
)

# ── TAB 1: QUIZ ───────────────────────────────────────────────────────────
with tab_quiz:
    st.markdown('<div class="card"><div class="card-header">Generate a Quiz</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "Topic to quiz on",
            placeholder="e.g. neural networks, photosynthesis, World War II…",
        )
    with col2:
        num_q = st.selectbox("Questions", [3, 5, 7, 10], index=1)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Generate Quiz"):
        if not st.session_state.db:
            st.warning("⚠️ Please ingest some study material first (use the sidebar).")
        elif not topic.strip():
            st.warning("⚠️ Please enter a topic.")
        else:
            with st.spinner("✨ Generating quiz with AI…"):
                retriever = get_retriever(st.session_state.db)
                raw = generate_quiz(topic.strip(), retriever, num_questions=num_q)
                questions = parse_quiz(raw)

            if questions:
                st.session_state.quiz_questions = questions
                st.session_state.quiz_topic     = topic.strip()
                st.session_state.quiz_answers   = {}
                st.session_state.quiz_submitted = False
                if topic.strip() not in st.session_state.all_topics:
                    st.session_state.all_topics.append(topic.strip())
            else:
                st.error("Could not parse quiz. Try a different topic or add more material.")

    if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
        st.markdown(f"### Quiz: *{st.session_state.quiz_topic}*")
        for i, q in enumerate(st.session_state.quiz_questions):
            st.markdown(f'<p class="quiz-q">Q{i+1}. {q["question"]}</p>', unsafe_allow_html=True)
            opts = [f"{k}) {v}" for k, v in q["options"].items()]
            ans = st.radio("", opts, key=f"q_{i}", label_visibility="collapsed")
            if ans:
                st.session_state.quiz_answers[i] = ans[0]
            st.markdown("---")

        if st.button("✅ Submit Quiz"):
            st.session_state.quiz_submitted = True
            st.rerun()

    if st.session_state.quiz_submitted and st.session_state.quiz_questions:
        questions = st.session_state.quiz_questions
        answers   = st.session_state.quiz_answers
        score     = sum(
            1 for i, q in enumerate(questions)
            if answers.get(i, "") == q["answer"]
        )
        total = len(questions)
        pct   = score / total * 100

        record_quiz_result(st.session_state.quiz_topic, score, total)

        st.markdown(f"### Results: *{st.session_state.quiz_topic}*")
        color = "green" if pct >= 70 else ("orange" if pct >= 50 else "red")
        st.markdown(
            f'<h2 style="color:{color}">{score}/{total} — {pct:.0f}%</h2>',
            unsafe_allow_html=True,
        )
        st.progress(int(pct))

        for i, q in enumerate(questions):
            user_ans    = answers.get(i, "?")
            correct_ans = q["answer"]
            is_correct  = user_ans == correct_ans
            icon    = "✅" if is_correct else "❌"
            css_cls = "correct" if is_correct else "incorrect"

            st.markdown(
                f'<p class="quiz-q">{icon} Q{i+1}. {q["question"]}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="{css_cls}">Your answer: {user_ans} — Correct: {correct_ans}</p>',
                unsafe_allow_html=True,
            )
            if q["explanation"]:
                st.caption(f"💡 {q['explanation']}")
            st.markdown("---")

        if st.button("🔄 Try Another Quiz"):
            st.session_state.quiz_questions = []
            st.session_state.quiz_submitted = False
            st.rerun()


# ── TAB 2: PROGRESS ───────────────────────────────────────────────────────
with tab_progress:
    st.markdown("### 📈 Your Performance")
    stats = get_summary_stats()

    if not stats:
        st.markdown("""
        <div style="text-align:center; padding:3rem; background:white;
                    border-radius:14px; box-shadow:0 2px 12px rgba(67,97,238,.08);">
            <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
            <div style="font-size:1.1rem; font-weight:700; color:#0f3460; margin-bottom:.5rem;">
                No quiz attempts yet
            </div>
            <div style="color:#666; font-size:.9rem;">
                Take a quiz in the Quiz Generator tab to see your progress here!
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        weak = get_weak_topics()
        all_avgs = [s["avg_score"] for s in stats.values()]
        overall  = sum(all_avgs) / len(all_avgs)

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Average", f"{overall:.1f}%")
        c2.metric("Topics Studied",  len(stats))
        c3.metric("Weak Areas",      len(weak))

        st.markdown("---")

        for topic, s in stats.items():
            bar_color = "#06d6a0" if s["avg_score"] >= 70 else ("#ffd166" if s["avg_score"] >= 50 else "#ef233c")
            label = "⚠️ Needs Work" if topic in weak else "✅ Good"
            st.markdown(f"""
            <div class="card">
                <div class="card-header">{topic}
                    <span style="font-size:.85rem; font-weight:400; margin-left:.5rem;">{label}</span>
                </div>
                <div style="display:flex; gap:2rem; margin-bottom:.6rem; flex-wrap:wrap;">
                    <span>Attempts: <b>{s['attempts']}</b></span>
                    <span>Avg: <b>{s['avg_score']}%</b></span>
                    <span>Best: <b>{s['best']}%</b></span>
                    <span>Last: <b>{s['latest']}%</b></span>
                </div>
                <div style="background:#eee; border-radius:6px; height:10px;">
                    <div style="background:{bar_color}; width:{s['avg_score']}%;
                                height:10px; border-radius:6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── TAB 3: RECOMMENDATIONS ────────────────────────────────────────────────
with tab_recommend:
    st.markdown("### 💡 Personalized Recommendations")

    if not st.session_state.db:
        st.warning("Ingest study material first, then take some quizzes.")
    elif not st.session_state.all_topics:
        st.info("Take at least one quiz to get personalized recommendations.")
    else:
        if st.button("🤖 Generate My Learning Plan"):
            with st.spinner("🧠 Analysing your performance and generating recommendations…"):
                retriever = get_retriever(st.session_state.db)
                recs = get_recommendations(retriever, st.session_state.all_topics)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(recs)
            st.markdown("</div>", unsafe_allow_html=True)

        weak = get_weak_topics()
        if weak:
            st.markdown("#### ⚠️ Topics Below 60% Average")
            for t in weak:
                st.markdown(f"- **{t}**")
