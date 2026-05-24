"""
Chat View UI Component
Interactive chat interface for discussing solutions with the local AI.
"""

import streamlit as st
from core.ai_brain import AIBrain


def render_chat_view():
    """Render the chat discussion interface."""

    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);border-radius:16px;'
        'padding:2rem 2.5rem;margin-bottom:1.5rem;border:1px solid rgba(255,255,255,0.08);">'
        '<h2 style="color:#fff;margin:0 0 .5rem 0;font-size:1.6rem;">💬 Discussion & Chat</h2>'
        '<p style="color:rgba(255,255,255,0.6);margin:0;font-size:0.95rem;">'
        "Ask questions about your data, solution strategy, or how to improve results.</p></div>",
        unsafe_allow_html=True,
    )

    brain: AIBrain = st.session_state.get("ai_brain")
    model_name = st.session_state.get("selected_model")

    if brain is None:
        if model_name is None:
            st.info("🤖 **Ollama not connected.** Run `ollama serve` and `ollama pull llama3.2:3b`.")
            return
        try:
            brain = AIBrain(model=model_name)
            brain.set_context(
                rules=st.session_state.get("competition_rules", ""),
                data_summary=st.session_state.get("data_summary_text", ""),
            )
            st.session_state["ai_brain"] = brain
        except Exception as e:
            st.error(f"Failed to connect: {e}")
            return

    if st.session_state.get("pipeline_results"):
        st.caption("🔗 Context loaded — AI knows about your data and results.")
    else:
        st.caption("💡 Tip: Generate a solution first for context-aware chat.")

    # Suggested questions
    if not st.session_state.get("chat_messages"):
        st.markdown("**Quick questions:**")
        suggestions = [
            "What type of problem is this?",
            "How can I improve my score?",
            "Explain the model comparison results.",
            "What feature engineering would help?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💡 {s}", key=f"sug_{i}", use_container_width=True):
                    st.session_state["pending_message"] = s
                    st.rerun()
        st.markdown("---")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.pop("pending_message", None)
    user_input = st.chat_input("Ask about your data, strategy, or results...")
    active = pending or user_input

    if active and brain:
        with st.chat_message("user"):
            st.markdown(active)
        st.session_state["chat_messages"].append({"role": "user", "content": active})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            try:
                for token in brain.chat(active):
                    full += token
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
            except Exception as e:
                full = f"⚠️ Error: {e}. Make sure Ollama is running."
                placeholder.markdown(full)

        st.session_state["chat_messages"].append({"role": "assistant", "content": full})

    if st.session_state.get("chat_messages"):
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["chat_messages"] = []
                if brain:
                    brain.clear_history()
                st.rerun()
