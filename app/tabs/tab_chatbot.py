import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from chatbot.agent import chat


def _escape_latex(text: str) -> str:
    """Escape LaTeX delimiters so Streamlit renders them as plain text."""
    text = text.replace("\\(", "\\\\(")
    text = text.replace("\\)", "\\\\)")
    text = text.replace("\\[", "\\\\[")
    text = text.replace("\\]", "\\\\]")
    text = text.replace("$", "\\$")
    return text


def _fmt_metric_value(v: float) -> str:
    """Format a numeric metric for large display."""
    if abs(v) >= 1000:
        return f"${v / 1000:.1f}B"
    if abs(v) >= 1:
        return f"${v:.1f}M"
    return f"${v:.2f}M"


# ── Metric display labels and colours ─────────────────────────────────────
_METRIC_LABELS = {
    "best_estimate_M":        ("Best Estimate IBNR", "#2196F3"),
    "var_99_M":               ("VaR 99%",            "#FF9800"),
    "var_995_M":              ("Reserve Risk Capital (VaR 99.5%)", "#F44336"),
    "cvar_99_M":              ("CVaR 99%",           "#9C27B0"),
    "reserve_risk_capital_M": ("Reserve Risk Capital", "#F44336"),
    "max_ibnr_M":             ("Max IBNR",           "#795548"),
    "n_replications_used":    ("Replications",       "#607D8B"),
}


def _render_metrics(attachment: dict):
    """Render reserve metrics as large coloured cards."""
    items = attachment.get("items", [])
    scenario = attachment.get("scenario", "")

    if scenario:
        st.caption(f"Scenario: **{scenario}**")

    # Separate monetary metrics from non-monetary
    monetary = [m for m in items if m["label"].endswith("_M")]
    other = [m for m in items if not m["label"].endswith("_M")]

    if not monetary and not other:
        return

    # Show top-line metrics as large cards
    display_items = monetary if monetary else items
    # Limit to 4 columns for readability
    cols = st.columns(min(len(display_items), 4))
    for i, m in enumerate(display_items[:4]):
        label_info = _METRIC_LABELS.get(m["label"], (m["label"].replace("_", " ").title(), "#666"))
        label, color = label_info
        val = m["value"]
        with cols[i % len(cols)]:
            st.markdown(
                f'<div style="text-align:center;padding:12px 8px;'
                f'background:linear-gradient(135deg,#F8FAFC,#F1F5F9);'
                f'border-radius:10px;border:1px solid #E2E8F0;'
                f'box-shadow:0 1px 3px rgba(0,0,0,0.06)">'
                f'<div style="font-size:0.78em;color:#64748B;text-transform:uppercase;'
                f'letter-spacing:0.04em;margin-bottom:4px">{label}</div>'
                f'<div style="font-size:2em;font-weight:700;color:{color}">'
                f'{_fmt_metric_value(val)}</div></div>',
                unsafe_allow_html=True,
            )

    # Show remaining metrics in a second row
    if len(display_items) > 4:
        cols2 = st.columns(min(len(display_items) - 4, 4))
        for i, m in enumerate(display_items[4:8]):
            label_info = _METRIC_LABELS.get(m["label"], (m["label"].replace("_", " ").title(), "#666"))
            label, color = label_info
            with cols2[i]:
                st.metric(label=label, value=_fmt_metric_value(m["value"]))


def _render_dataframe(attachment: dict):
    """Render a dataframe with optional line chart."""
    df = attachment.get("df")
    if df is None or df.empty:
        return

    chart_spec = attachment.get("chart")
    if chart_spec:
        _render_chart(df, chart_spec)

    # Convert numeric columns for better display
    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors="ignore")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _render_chart(df: pd.DataFrame, spec: dict):
    """Render a Plotly chart from a chart specification."""
    x = spec.get("x")
    y = spec.get("y")
    lo = spec.get("lo")
    hi = spec.get("hi")
    title = spec.get("title", "")

    if not x or not y or x not in df.columns or y not in df.columns:
        return

    fig = go.Figure()

    # Confidence band
    if lo and hi and lo in df.columns and hi in df.columns:
        df_sorted = df.sort_values(x)
        fig.add_trace(go.Scatter(
            x=pd.concat([df_sorted[x], df_sorted[x][::-1]]),
            y=pd.concat([
                pd.to_numeric(df_sorted[hi], errors="coerce"),
                pd.to_numeric(df_sorted[lo][::-1], errors="coerce"),
            ]),
            fill="toself", fillcolor="rgba(33, 150, 243, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True,
        ))

    # Main line
    fig.add_trace(go.Scatter(
        x=df[x],
        y=pd.to_numeric(df[y], errors="coerce"),
        mode="lines+markers",
        name=y.replace("_", " ").title(),
        line=dict(color="#1B3A5C", width=2),
    ))

    fig.update_layout(
        title=title, height=300,
        margin=dict(l=0, r=0, t=40, b=30),
        xaxis_title="", yaxis_title="",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_attachments(attachments: list):
    """Render structured tool results as rich Streamlit components."""
    for att in attachments:
        att_type = att.get("type", "")
        if att_type == "metrics":
            _render_metrics(att)
        elif att_type == "dataframe":
            _render_dataframe(att)


EXAMPLE_QUESTIONS = [
    "What is the current best estimate IBNR for the portfolio?",
    "Show me the top 5 segments by average monthly claims",
    "Generate a 12-month frequency forecast",
    "What would happen to reserve risk if LDFs deteriorated by 20%?",
    "Explain how the Bootstrap Chain Ladder works",
    "Compare the reserve scenarios -- which one has the highest VaR 99.5%?",
    "Which provinces have the highest projected claim frequency?",
    "What is the MCT ratio and how is it calculated?",
]


def render(tab):
    with tab:
        st.subheader("Reserve Assistant")
        st.caption(
            "Ask questions about reserve adequacy, IBNR, frequency forecasts, "
            "reserve scenarios, or actuarial concepts"
        )

        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Use a container for all chat messages so they render above the input
        chat_container = st.container()

        # ── Example questions as a popover above the input ───────────────
        with st.popover("Example questions", use_container_width=True):
            for i, q in enumerate(EXAMPLE_QUESTIONS):
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": q})

        # Chat input -- always at the bottom
        prompt = st.chat_input("Ask about reserves, IBNR, forecasts, or risk...")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Render chat history inside the container (above the input)
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(_escape_latex(msg["content"]))
                    # Re-render stored attachments
                    if msg.get("attachments"):
                        _render_attachments(msg["attachments"])

            # Generate response for the latest user message
            if (
                st.session_state.chat_messages
                and st.session_state.chat_messages[-1]["role"] == "user"
            ):
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    collected_attachments = []

                    agent_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_messages
                        if m["role"] in ("user", "assistant")
                    ]

                    tool_status = st.empty()
                    for chunk in chat(agent_messages):
                        if isinstance(chunk, dict) and chunk.get("type") == "attachments":
                            collected_attachments = chunk["data"]
                        elif isinstance(chunk, str) and chunk.startswith("_Calling"):
                            tool_status.caption(chunk.strip("_\n"))
                        elif isinstance(chunk, str):
                            tool_status.empty()
                            full_response += chunk
                            response_placeholder.markdown(_escape_latex(full_response))

                    if not full_response:
                        full_response = "I wasn't able to generate a response. Please try again."
                        response_placeholder.markdown(_escape_latex(full_response))

                    # Render rich attachments below the text
                    if collected_attachments:
                        _render_attachments(collected_attachments)

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "attachments": collected_attachments if collected_attachments else None,
                })

            # Clear chat button
            if st.session_state.chat_messages:
                if st.button("Clear conversation", type="secondary"):
                    st.session_state.chat_messages = []
                    st.rerun()
