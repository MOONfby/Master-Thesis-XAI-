"""
Chat panel — streaming conversation area.
"""
from __future__ import annotations

import streamlit as st

from phase3.llm.claude_client import ClaudeExplanationClient


_PERSONA_BADGE_COLORS = {
    "ML Engineer":        "#3498db",
    "Domain Expert":      "#27ae60",
    "Regulator / Auditor": "#8e44ad",
}


def render_chat_panel(persona: str, api_key: str | None) -> None:
    """
    Render the right-hand chat column.

    Reads from st.session_state:
        conversation_history, system_prompt, current_explanation

    Writes to st.session_state:
        conversation_history
    """
    # Header
    color = _PERSONA_BADGE_COLORS.get(persona, "#7f8c8d")
    st.markdown(
        f"### AI Explanation Assistant\n"
        f"<span style='background-color:{color};color:white;"
        f"padding:2px 8px;border-radius:4px;font-size:0.8em'>"
        f"{persona}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Chat history display
    history = st.session_state.get("conversation_history", [])
    chat_container = st.container(height=400)

    with chat_container:
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                with st.chat_message("user"):
                    # Strip the grounding context block for display
                    display_content = _strip_context_block(content)
                    st.markdown(display_content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)

    # Input row
    col_input, col_send, col_reset = st.columns([5, 1, 1])

    with col_input:
        user_input = st.text_input(
            "Ask a follow-up question",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="Ask a follow-up...",
        )

    with col_send:
        send_clicked = st.button("Send", key="send_btn", use_container_width=True)

    with col_reset:
        reset_clicked = st.button("Reset", key="reset_btn", use_container_width=True)

    # Handle reset
    if reset_clicked:
        st.session_state["conversation_history"] = []
        st.rerun()

    # Handle send
    if send_clicked and user_input.strip():
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
            return

        _stream_followup(user_input.strip(), api_key)


def stream_opening_message(
    context_block: str,
    opening_question: str,
    system_prompt: str,
    api_key: str,
) -> None:
    """
    Generate and stream the initial explanation message.

    This is called from app.py when "Run Explanation" is clicked.
    Resets conversation history and injects grounded context.
    """
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        return

    client = ClaudeExplanationClient(api_key=api_key)

    first_user_msg = ClaudeExplanationClient.build_first_user_message(
        context_block, opening_question
    )

    messages = [{"role": "user", "content": first_user_msg}]

    # Collect response silently (no direct rendering here — render happens
    # via chat_container in render_chat_panel after st.rerun())
    with st.spinner("Generating explanation..."):
        full_text = ""
        for chunk in client.stream_response(system_prompt, messages):
            full_text += chunk

    # Commit to history and rerun so chat_container renders in the right place
    messages = client.append_turn(messages, "assistant", full_text)
    st.session_state["conversation_history"] = messages
    st.rerun()


def _stream_followup(user_input: str, api_key: str) -> None:
    """Append a follow-up turn and stream the response."""
    client = ClaudeExplanationClient(api_key=api_key)
    system_prompt = st.session_state.get("system_prompt", "")
    history = st.session_state.get("conversation_history", [])

    messages = client.append_turn(history, "user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for chunk in client.stream_response(system_prompt, messages):
            full_text += chunk
            placeholder.markdown(full_text + " ▌")
        placeholder.markdown(full_text)

    messages = client.append_turn(messages, "assistant", full_text)
    st.session_state["conversation_history"] = messages
    st.rerun()


def _strip_context_block(content: str) -> str:
    """Remove the grounded context block from display (only show the question)."""
    marker = "=== YOUR QUESTION ==="
    idx = content.find(marker)
    if idx >= 0:
        return content[idx + len(marker):].strip()
    return content
