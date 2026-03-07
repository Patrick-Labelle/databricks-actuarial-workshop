import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from endpoints import call_frequency_forecast_endpoint


def render(tab):
    with tab:
        st.subheader("On-Demand Frequency Forecast")
        st.caption("Get an instant claim frequency projection for any time horizon — results update in seconds")

        with st.expander("ℹ️ How this works", expanded=False):
            st.markdown("""
**What this does:** Generates a claim frequency forecast on demand for any horizon up to 24 months, without re-running the full modelling pipeline.

**When to use it:**
- Quick "what if I need an 18-month forecast?" checks
- Board or management presentations requiring a specific horizon
- Validating that the deployed model is live and responding correctly

**What you get back:**
- A monthly point forecast (most likely outcome)
- Upper and lower bounds of the 95% forecast range
- A note if uncertainty grows significantly at longer horizons (expected — forecasts become less precise the further out you go)

**The model** is the same one trained on all 40 segments — it represents aggregate portfolio behaviour. For segment-specific forecasts with full historical context, use the **Reserve Development Forecast** tab.

_Technical: SARIMA REST endpoint served via Databricks Model Serving, @Champion alias in Unity Catalog._
""")

        horizon = st.slider(
            "How many months ahead to forecast:",
            min_value=1, max_value=24, value=6,
            help="How many months ahead to forecast. Uncertainty (CI width) grows with horizon."
        )

        if st.button("Generate Forecast"):
            with st.spinner("Calling Model Serving endpoint..."):
                _fetched = call_frequency_forecast_endpoint(horizon)
                if not _fetched.empty:
                    st.session_state["ondemand_result"] = _fetched
                    st.session_state["ondemand_horizon"] = horizon
                else:
                    st.warning("Endpoint not available — start the Model Serving endpoint from Module 4")

        result_df = st.session_state.get("ondemand_result", pd.DataFrame())
        _display_horizon = st.session_state.get("ondemand_horizon", horizon)
        if not result_df.empty:
            # Rename for display
            display_cols = {
                "month_offset": "Month Ahead",
                "forecast_mean": "Point Forecast (mean)",
                "forecast_lo95": "Lower 95% CI",
                "forecast_hi95": "Upper 95% CI",
            }
            display_df = result_df.rename(columns={k: v for k, v in display_cols.items() if k in result_df.columns})

            # Numeric formatting
            for col in display_df.columns:
                if col != "Month Ahead":
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(1)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Visualise the forecast
            if all(c in result_df.columns for c in ["month_offset", "forecast_mean", "forecast_lo95", "forecast_hi95"]):
                result_df["forecast_mean"] = pd.to_numeric(result_df["forecast_mean"], errors='coerce')
                result_df["forecast_lo95"] = pd.to_numeric(result_df["forecast_lo95"], errors='coerce')
                result_df["forecast_hi95"] = pd.to_numeric(result_df["forecast_hi95"], errors='coerce')

                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=result_df["month_offset"], y=result_df["forecast_mean"],
                    mode="lines+markers", name="Projected claims",
                    line=dict(color="#FF3419"),
                    hovertemplate="Month +%{x}<br>Projected: %{y:,.1f} claims<extra></extra>",
                ))
                fig4.add_trace(go.Scatter(
                    x=pd.concat([result_df["month_offset"], result_df["month_offset"][::-1]]),
                    y=pd.concat([result_df["forecast_hi95"], result_df["forecast_lo95"][::-1]]),
                    fill="toself", fillcolor="rgba(255,52,25,0.15)",
                    line=dict(color="rgba(255,0,0,0)"),
                    name="Forecast range",
                    hoverinfo="skip",
                ))
                fig4.update_layout(
                    title=f"Frequency Forecast — Next {_display_horizon} Months",
                    xaxis_title="Month",
                    yaxis_title="Projected Monthly Claims",
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig4, use_container_width=True)

                # Flag widening CI
                ci_width_start = float(result_df["forecast_hi95"].iloc[0] - result_df["forecast_lo95"].iloc[0])
                ci_width_end   = float(result_df["forecast_hi95"].iloc[-1] - result_df["forecast_lo95"].iloc[-1])
                if ci_width_end > ci_width_start * 1.5:
                    st.info(
                        "The forecast range widens significantly at longer horizons — this is expected. "
                        "Uncertainty compounds over time; use shorter-horizon forecasts for operational decisions "
                        "and longer horizons for strategic planning only."
                    )

            st.download_button(
                "Download CSV",
                result_df.to_csv(index=False),
                file_name=f"frequency_forecast_{_display_horizon}m.csv",
                mime="text/csv",
            )
