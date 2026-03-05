import streamlit as st

from chatbot.agent import chat


EXAMPLE_QUESTIONS = [
    "What are the current Solvency Capital Requirements for the portfolio?",
    "Show me the top 5 segments by average monthly claims",
    "Generate a 12-month claims forecast",
    "What would happen to capital requirements if property losses doubled?",
    "Explain how the GARCH model captures volatility clustering",
    "What stress scenarios have been tested and how do they compare?",
    "Show recent analyst annotations for commercial_auto_ontario",
    "What is the VaR trend over the next 12 months?",
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

        # Example questions
        with st.expander("Example questions", expanded=not st.session_state.chat_messages):
            cols = st.columns(2)
            for i, q in enumerate(EXAMPLE_QUESTIONS):
                if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    st.rerun()

        # Render chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about claims, forecasts, capital, or risk..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

        # Generate response for the latest user message
        if (
            st.session_state.chat_messages
            and st.session_state.chat_messages[-1]["role"] == "user"
        ):
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                # Build message history for the agent (exclude tool status messages)
                agent_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_messages
                    if m["role"] in ("user", "assistant")
                ]

                for chunk in chat(agent_messages):
                    if chunk.startswith("_Calling"):
                        # Show tool call status inline
                        st.caption(chunk.strip("_\n"))
                    else:
                        full_response += chunk
                        response_placeholder.markdown(full_response)

                if not full_response:
                    full_response = "I wasn't able to generate a response. Please try again."
                    response_placeholder.markdown(full_response)

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": full_response}
            )

        # Clear chat button
        if st.session_state.chat_messages:
            if st.button("Clear conversation", type="secondary"):
                st.session_state.chat_messages = []
                st.rerun()
