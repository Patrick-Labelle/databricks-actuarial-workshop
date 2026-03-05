import streamlit as st

from chatbot.agent import chat


def _escape_latex(text: str) -> str:
    """Escape LaTeX delimiters so Streamlit renders them as plain text.

    Streamlit renders $...$, $$...$$, \\(...\\), and \\[...\\] as LaTeX.
    LLM responses use $ for currency and sometimes produce LaTeX math
    notation, causing garbled rendering. Escape all delimiters since
    the chatbot never needs math rendering.
    """
    # Escape \(...\) and \[...\] BEFORE escaping $ signs
    text = text.replace("\\(", "\\\\(")
    text = text.replace("\\)", "\\\\)")
    text = text.replace("\\[", "\\\\[")
    text = text.replace("\\]", "\\\\]")
    # Escape dollar signs so KaTeX doesn't trigger
    text = text.replace("$", "\\$")
    return text


EXAMPLE_QUESTIONS = [
    "What are the current Solvency Capital Requirements for the portfolio?",
    "Show me the top 5 segments by average monthly claims",
    "Generate a 12-month claims forecast",
    "What would happen to capital requirements if property losses doubled?",
    "Explain how the GARCH model captures volatility clustering",
    "Compare the stress test scenarios — which one has the highest SCR?",
    "Which provinces have the highest projected claims for the next year?",
    "Show the claims trend for Commercial Property over time",
]


def render(tab):
    with tab:
        st.subheader("Actuarial Risk Assistant")
        st.caption(
            "Ask questions about claims data, forecasts, capital requirements, "
            "stress scenarios, or actuarial concepts"
        )

        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Example questions (collapsed once there's a conversation)
        with st.expander("Example questions", expanded=not st.session_state.chat_messages):
            cols = st.columns(2)
            for i, q in enumerate(EXAMPLE_QUESTIONS):
                if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    st.rerun()

        # Use a container for all chat messages so they render above the input
        chat_container = st.container()

        # Chat input — always at the bottom of the page
        prompt = st.chat_input("Ask about claims, forecasts, capital, or risk...")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.rerun()

        # Render chat history inside the container (above the input)
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(_escape_latex(msg["content"]))

            # Generate response for the latest user message
            if (
                st.session_state.chat_messages
                and st.session_state.chat_messages[-1]["role"] == "user"
            ):
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""

                    agent_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages
                        if m["role"] in ("user", "assistant")
                    ]

                    for chunk in chat(agent_messages):
                        if chunk.startswith("_Calling"):
                            st.caption(chunk.strip("_\n"))
                        else:
                            full_response += chunk
                            response_placeholder.markdown(_escape_latex(full_response))

                    if not full_response:
                        full_response = "I wasn't able to generate a response. Please try again."
                        response_placeholder.markdown(_escape_latex(full_response))

                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": full_response}
                )

            # Clear chat button
            if st.session_state.chat_messages:
                if st.button("Clear conversation", type="secondary"):
                    st.session_state.chat_messages = []
                    st.rerun()
