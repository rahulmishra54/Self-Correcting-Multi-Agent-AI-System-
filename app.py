"""
Self-Correcting AI - Streamlit UI
---------------------------------
A front-end for the LangGraph workflow in main.py.

Backend facts this UI is built on (taken from main.py):
  * State fields: topic, answer, revision_count, feedback, decision
  * decision is "PASS" or "REVISE"
  * graph.invoke() returns only the FINAL state (no intermediate events),
    so agent statuses are derived from that final state.
  * On PASS, the reviewer sets feedback to "" (so there is no feedback text then).
"""

import os

import streamlit as st

# ------------------------------------------------------------------
# Page config must be the first Streamlit call
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Self-Correcting AI",
    page_icon="🧠",
    layout="wide",
)

# Import the compiled LangGraph from main.py. If it fails (missing package,
# bad import, etc.) we keep the UI alive and show the error instead of crashing.
try:
    from backend import graph, MAX_REVISIONS

    BACKEND_ERROR = None
except Exception as exc:  # noqa: BLE001
    graph, MAX_REVISIONS = None, 3
    BACKEND_ERROR = str(exc)


# ------------------------------------------------------------------
# Custom CSS (light SaaS look: white/grey backgrounds, blue/purple accents)
# ------------------------------------------------------------------
st.markdown(
    """
<style>
    .stApp { background: #f6f7fb; }
    .block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }
    #MainMenu, footer { visibility: hidden; }

    /* Header */
    .app-header {
        display: flex; justify-content: space-between; align-items: center;
        background: #ffffff; border-radius: 18px; padding: 22px 28px;
        box-shadow: 0 2px 12px rgba(79, 70, 229, 0.08); margin-bottom: 22px;
    }
    .app-title {
        font-size: 1.9rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .app-subtitle { color: #6b7280; margin: 4px 0 0 0; font-size: 0.95rem; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;
    }
    .status-ready { background: #ecfdf5; color: #047857; }
    .status-error { background: #fef2f2; color: #b91c1c; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

    /* Section titles */
    .section-title { font-size: 1.05rem; font-weight: 700; color: #1f2937; margin: 26px 0 10px 0; }

    /* Agent cards */
    .agent-card {
        background: #ffffff; border-radius: 16px; padding: 20px;
        box-shadow: 0 2px 10px rgba(17, 24, 39, 0.06);
        border-top: 4px solid #e5e7eb; height: 100%;
    }
    .agent-icon { font-size: 1.8rem; }
    .agent-name { font-weight: 700; color: #111827; margin: 6px 0 2px 0; }
    .agent-desc { color: #6b7280; font-size: 0.85rem; min-height: 40px; }
    .badge { display: inline-block; margin-top: 10px; padding: 4px 12px;
             border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
    .st-waiting   { border-top-color: #d1d5db; }
    .st-running   { border-top-color: #3b82f6; }
    .st-completed { border-top-color: #10b981; }
    .st-rejected  { border-top-color: #f59e0b; }
    .badge-waiting   { background: #f3f4f6; color: #6b7280; }
    .badge-running   { background: #dbeafe; color: #1d4ed8; }
    .badge-completed { background: #d1fae5; color: #047857; }
    .badge-rejected  { background: #fef3c7; color: #b45309; }

    /* Answer card */
    .answer-card {
        background: #ffffff; border-radius: 16px; padding: 24px 28px;
        box-shadow: 0 2px 12px rgba(79, 70, 229, 0.08); border-left: 5px solid #7c3aed;
        color: #1f2937; line-height: 1.65;
    }
    .metric-card {
        background: #ffffff; border-radius: 14px; padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(17, 24, 39, 0.06);
    }
    .metric-label { color: #6b7280; font-size: 0.8rem; }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #111827; }

    /* Primary button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #7c3aed); border: none;
        color: white; border-radius: 10px; font-weight: 600;
    }
    div.stButton > button { border-radius: 10px; }
    .stTextArea textarea { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of past runs (newest first)
if "result" not in st.session_state:
    st.session_state.result = None  # the latest run, shown in the main area
if "question_input" not in st.session_state:
    st.session_state.question_input = ""


def clear_all():
    """Callback for the Clear button: reset the input box and the current result."""
    st.session_state.question_input = ""
    st.session_state.result = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
# status key -> (card CSS class, badge CSS class, label)
STATUS_STYLES = {
    "waiting": ("st-waiting", "badge-waiting", "⏳ Waiting"),
    "running": ("st-running", "badge-running", "🔄 Running"),
    "completed": ("st-completed", "badge-completed", "✅ Completed"),
    "rejected": ("st-rejected", "badge-rejected", "⚠️ Needs Revision"),
    "skipped": ("st-waiting", "badge-waiting", "➖ Not needed"),
}

AGENTS = [
    ("writer", "✍️", "Writer Agent", "Drafts a beginner-friendly first answer."),
    ("reviewer", "🔍", "Reviewer Agent", "Checks the answer against quality rules."),
    ("reviser", "🛠️", "Reviser Agent", "Improves the answer using reviewer feedback."),
]


def render_agent_cards(statuses: dict):
    """Draw the three agent cards. `statuses` maps agent key -> status key."""
    cols = st.columns(3)
    for col, (key, icon, name, desc) in zip(cols, AGENTS):
        card_cls, badge_cls, label = STATUS_STYLES[statuses[key]]
        col.markdown(
            f"""
            <div class="agent-card {card_cls}">
                <div class="agent-icon">{icon}</div>
                <div class="agent-name">{name}</div>
                <div class="agent-desc">{desc}</div>
                <span class="badge {badge_cls}">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def statuses_from_result(result: dict) -> dict:
    """
    The graph only returns the final state, so we infer what happened:
      * Writer always runs.
      * Reviewer always runs; if the final decision is not PASS it "rejected" the answer.
      * Reviser ran only if revision_count > 0.
    """
    revisions = result["revision_count"]
    return {
        "writer": "completed",
        "reviewer": "completed" if result["decision"] == "PASS" else "rejected",
        "reviser": "completed" if revisions > 0 else "skipped",
    }


def metric(col, label, value):
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# A. Header
# ------------------------------------------------------------------
if BACKEND_ERROR:
    pill = '<span class="status-pill status-error"><span class="status-dot"></span>Backend error</span>'
else:
    pill = '<span class="status-pill status-ready"><span class="status-dot"></span>System Ready</span>'

st.markdown(
    f"""
    <div class="app-header">
        <div>
            <p class="app-title">Self-Correcting AI</p>
            <p class="app-subtitle">Multi-Agent Intelligence &amp; Automated Answer Refinement</p>
        </div>
        {pill}
    </div>
    """,
    unsafe_allow_html=True,
)

if BACKEND_ERROR:
    st.error(f"Could not load the LangGraph backend from main.py: {BACKEND_ERROR}")
elif not os.getenv("GOOGLE_API_KEY"):
    st.warning("GOOGLE_API_KEY was not found. Add it to your .env file, then restart the app.")

# ------------------------------------------------------------------
# B. Question input
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Your Question</div>', unsafe_allow_html=True)

question = st.text_area(
    "Question",
    key="question_input",
    placeholder="Ask anything... Your AI agents will research, generate and refine the answer.",
    height=130,
    label_visibility="collapsed",
)

btn_col1, btn_col2, _ = st.columns([1.3, 1, 6])
generate = btn_col1.button(
    "Generate Answer", type="primary", use_container_width=True, disabled=BACKEND_ERROR is not None
)
btn_col2.button("Clear", on_click=clear_all, use_container_width=True)

# ------------------------------------------------------------------
# C. Agent workflow (placeholder so we can update it while the graph runs)
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Agent Workflow</div>', unsafe_allow_html=True)
workflow_slot = st.empty()

if generate:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        # graph.invoke() blocks until the whole workflow finishes, so we cannot
        # know which agent is active. While waiting, show Writer and Reviewer as
        # running (they always run) and the Reviser as waiting (only runs if needed).
        with workflow_slot.container():
            render_agent_cards({"writer": "running", "reviewer": "running", "reviser": "waiting"})

        # main.py's State uses "topic" (not "question"), and "decision" (not "approved").
        initial_state = {
            "topic": question.strip(),
            "answer": "",
            "revision_count": 0,
            "feedback": "",
            "decision": "",
        }

        try:
            with st.spinner("Agents are writing, reviewing and refining your answer..."):
                result = graph.invoke(initial_state, config={"recursion_limit": 20})

            st.session_state.result = result
            st.session_state.history.insert(0, result)
        except Exception as exc:  # noqa: BLE001  (API key, network, quota, parsing errors...)
            st.session_state.result = None
            st.error(f"The workflow failed: {exc}")

# Draw the cards for the current state (idle or finished)
result = st.session_state.result
with workflow_slot.container():
    if result:
        render_agent_cards(statuses_from_result(result))
    else:
        render_agent_cards({"writer": "waiting", "reviewer": "waiting", "reviser": "waiting"})

# ------------------------------------------------------------------
# D. Final answer
# ------------------------------------------------------------------
if result:
    st.markdown('<div class="section-title">Final Answer</div>', unsafe_allow_html=True)

    approved = result["decision"] == "PASS"
    m1, m2, m3 = st.columns(3)
    metric(m1, "Approval Status", "✅ Approved" if approved else "⚠️ Not approved")
    metric(m2, "Revision Attempts", f"{result['revision_count']} / {MAX_REVISIONS}")
    metric(m3, "Reviewer Decision", result["decision"])

    st.write("")
    with st.container():
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.markdown(result["answer"])
        st.markdown("</div>", unsafe_allow_html=True)

    if not approved:
        st.info("The revision limit was reached before the reviewer approved the answer.")

    with st.expander("Reviewer feedback"):
        if result["feedback"]:
            st.markdown(result["feedback"])
        else:
            # The backend clears feedback on PASS, and earlier rounds' feedback is overwritten.
            st.caption("The reviewer approved the final answer, so there is no outstanding feedback.")

# ------------------------------------------------------------------
# E. Execution history
# ------------------------------------------------------------------
st.markdown('<div class="section-title">Execution History</div>', unsafe_allow_html=True)

if not st.session_state.history:
    st.caption("No executions yet. Your previous questions will appear here.")
else:
    for i, run in enumerate(st.session_state.history):
        icon = "✅" if run["decision"] == "PASS" else "⚠️"
        label = f"{icon} {run['topic'][:80]}  ·  {run['revision_count']} revision(s)"
        with st.expander(label):
            st.markdown(run["answer"])
            st.caption(
                f"Decision: {run['decision']}  |  Revisions: {run['revision_count']}"
            )