import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from db import load_segments, load_forecasts, load_segment_stats, load_garch_volatility
from endpoints import email_from_token
from lakebase import save_scenario_annotation, load_annotations
from constants import SCENARIO_TYPES, APPROVAL_STATUSES


def render(tab):
    with tab:
        st.subheader("Claims Volume Forecast by Product & Region")

        with st.expander("ℹ️ About this forecast", expanded=False):
            st.markdown("""
**What this shows:** Projected monthly claim volumes for the selected product line and region, based on 7 years of historical claims data (Jan 2019 – Dec 2025).

**The shaded band** is the forecast uncertainty range — the model expects 95% of actual future months to fall within this range. A wider band means higher uncertainty, which is normal for longer forecast horizons.

**The orange area** (right axis) shows GARCH conditional volatility — when it rises, the confidence band widens accordingly.

**The dashed line** shows the most likely (point estimate) outcome each month.

**Important:** This model assumes the historical seasonal pattern continues. Use the analyst note tool below to flag known upcoming changes — new products, regulatory changes, or unusual loss events — that the model cannot anticipate automatically.

_Technical details: SARIMA(1,1,1)(1,1,1)₁₂ fitted per segment using statsmodels.SARIMAX._
""")

        segments = load_segments()
        if segments:
            selected = st.selectbox("Select product line & region:", segments, index=0)

            if selected:
                # Load segment history stats alongside the forecast
                stats_df = load_segment_stats(selected)
                if not stats_df.empty:
                    for col in stats_df.columns:
                        stats_df[col] = stats_df[col].apply(
                            lambda x: pd.to_numeric(x, errors='ignore') if x is not None else x
                        )
                    s = stats_df.iloc[0]
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric(
                        "History",
                        f"{str(s['first_month'])[:7]} – {str(s['last_month'])[:7]}",
                        help="Date range of historical claims data used to build this forecast"
                    )
                    sc2.metric(
                        "Avg Monthly Claims",
                        f"{float(s['avg_monthly_claims']):,.0f}",
                        help="Average number of claims filed per month over the historical period"
                    )
                    sc3.metric(
                        "Month-to-Month Variability",
                        f"{float(s['stddev_claims']):,.0f}",
                        help="How much monthly claims typically vary. Higher values mean less predictable months."
                    )
                    sc4.metric(
                        "Range",
                        f"{int(float(s['min_claims'])):,} – {int(float(s['max_claims'])):,}",
                        help="Minimum and maximum observed monthly claims count"
                    )

                df = load_forecasts(selected)

                if not df.empty:
                    for col in ["claims_count", "forecast_mean", "forecast_lo95", "forecast_hi95"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    actuals   = df[df["record_type"] == "actual"]
                    forecasts = df[df["record_type"] == "forecast"]

                    # Load GARCH volatility (actuals + forecasts)
                    garch_df = load_garch_volatility(selected)
                    has_garch = not garch_df.empty

                    # ── Integrated dual-axis chart ────────────────────────────
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    # Left y-axis: claims
                    fig.add_trace(go.Scatter(
                        x=actuals["month"], y=actuals["claims_count"],
                        mode="lines+markers", name="Actual claims",
                        line=dict(color="#1f77b4"),
                        hovertemplate="<b>%{x}</b><br>Actual: %{y:,.0f} claims<extra></extra>",
                    ), secondary_y=False)

                    fig.add_trace(go.Scatter(
                        x=forecasts["month"], y=forecasts["forecast_mean"],
                        mode="lines+markers", name="Projected claims",
                        line=dict(color="#FF3419", dash="dash"),
                        hovertemplate="<b>%{x}</b><br>Forecast: %{y:,.0f} claims<extra></extra>",
                    ), secondary_y=False)

                    # CI band
                    fig.add_trace(go.Scatter(
                        x=pd.concat([forecasts["month"], forecasts["month"][::-1]]),
                        y=pd.concat([forecasts["forecast_hi95"], forecasts["forecast_lo95"][::-1]]),
                        fill="toself", fillcolor="rgba(255,52,25,0.15)",
                        line=dict(color="rgba(255,0,0,0)"),
                        name="Forecast range (95% CI)",
                        hoverinfo="skip",
                    ), secondary_y=False)

                    # Right y-axis: GARCH conditional volatility as filled area
                    if has_garch:
                        fig.add_trace(go.Scatter(
                            x=garch_df["month"],
                            y=garch_df["cond_volatility"],
                            mode="lines", name="GARCH Volatility (σ_t)",
                            line=dict(color="#FF6B35", width=1.5),
                            fill="tozeroy",
                            fillcolor="rgba(255,107,53,0.15)",
                            hovertemplate="<b>%{x}</b><br>Volatility: %{y:,.1f}<extra></extra>",
                        ), secondary_y=True)
                        _garch_note = ""
                    else:
                        _garch_note = " (CIs are constant — no significant ARCH effects detected)"

                    # Vertical forecast-start marker
                    if not actuals.empty and not forecasts.empty:
                        cutoff = str(actuals["month"].max())
                        fig.add_shape(
                            type="line",
                            x0=cutoff, x1=cutoff, y0=0, y1=1, yref="paper",
                            line=dict(dash="dot", color="grey", width=1),
                        )
                        fig.add_annotation(
                            x=cutoff, y=1, yref="paper",
                            text="Forecast start", showarrow=False,
                            yanchor="bottom", font=dict(color="grey", size=11),
                        )

                    fig.update_layout(
                        title=f"{selected} — Claims Forecast with GARCH Volatility",
                        height=460,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode="x unified",
                    )
                    fig.update_xaxes(title_text="Month")
                    fig.update_yaxes(title_text="Monthly Claims", secondary_y=False)
                    fig.update_yaxes(
                        title_text="Conditional Volatility (σ_t)",
                        secondary_y=True,
                        showgrid=False,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    if _garch_note:
                        st.caption(f"Note{_garch_note}")

                    col1, col2, col3 = st.columns(3)
                    if "mape" in df.columns:
                        mape_vals = pd.to_numeric(df[df["record_type"] == "forecast"]["mape"], errors='coerce')
                        col1.metric(
                            "Forecast Accuracy",
                            f"{mape_vals.mean():.1f}%",
                            help="Average error rate on the out-of-sample test period. Lower is better — under 10% is strong for insurance data."
                        )
                    if not forecasts.empty:
                        col2.metric(
                            "Projected Monthly Claims",
                            f"{int(forecasts['forecast_mean'].mean()):,}",
                            help="Average projected claims per month over the 12-month forecast horizon"
                        )
                        half_width = int((forecasts['forecast_hi95'] - forecasts['forecast_lo95']).mean() / 2)
                        col3.metric(
                            "Forecast Uncertainty Range",
                            f"±{half_width:,}",
                            help="Average margin of uncertainty around the monthly forecast. Wider ranges reflect higher uncertainty — normal as the forecast extends further into the future."
                        )

                    # GARCH diagnostic details (collapsed)
                    with st.expander("📊 GARCH Diagnostics", expanded=False):
                        if has_garch:
                            _pval = garch_df['arch_lm_pvalue'].iloc[0] if 'arch_lm_pvalue' in garch_df.columns else None
                            _alpha = garch_df['garch_alpha'].iloc[0] if 'garch_alpha' in garch_df.columns else None
                            _beta = garch_df['garch_beta'].iloc[0] if 'garch_beta' in garch_df.columns else None

                            if pd.notna(_pval):
                                _sig = "significant" if _pval < 0.10 else "not significant"
                                st.caption(f"ARCH-LM p-value: {_pval:.4f} ({_sig} at 10% level)")
                            if pd.notna(_alpha) and pd.notna(_beta):
                                st.caption(
                                    f"GARCH parameters: α={_alpha:.3f} (news impact), "
                                    f"β={_beta:.3f} (persistence), α+β={_alpha+_beta:.3f}"
                                )
                                st.caption(
                                    "α measures how quickly new shocks affect volatility; "
                                    "β measures how long high-volatility regimes persist. "
                                    "α+β close to 1.0 indicates highly persistent volatility."
                                )
                        else:
                            st.info("No significant ARCH effects detected for this segment, or Module 4 has not been run yet.")

                    with st.expander("📋 Raw forecast data"):
                        st.markdown("""
| Column | Description |
|---|---|
| `month` | Calendar month (YYYY-MM-DD, first of month) |
| `record_type` | `actual` = observed history; `forecast` = SARIMA prediction |
| `claims_count` | Observed monthly claims count (actuals only) |
| `forecast_mean` | Point forecast — mean of the predictive distribution |
| `forecast_lo95` | Lower bound of 95% prediction interval |
| `forecast_hi95` | Upper bound of 95% prediction interval |
""")
                        display_df = df.copy()
                        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Scenario annotation
            with st.expander("📝 Add Analyst Note"):
                st.caption(
                    "Record any known factors that could affect this forecast — new products, regulatory changes, "
                    "loss events, or expert judgment overrides. Notes are saved and visible to all team members."
                )
                # Pre-populate analyst name from the forwarded user token if available
                _user_token = st.context.headers.get("X-Forwarded-Access-Token", "")
                _default_analyst = email_from_token(_user_token) if _user_token else ""

                _an_col1, _an_col2 = st.columns(2)
                with _an_col1:
                    analyst       = st.text_input("Analyst:", value=_default_analyst)
                    scenario_type = st.selectbox("Type:", SCENARIO_TYPES)
                with _an_col2:
                    approval_status = st.selectbox("Status:", APPROVAL_STATUSES)
                    adjustment_pct  = st.number_input(
                        "Recommended Forecast Adjustment (%):",
                        min_value=-50.0, max_value=50.0, value=0.0, step=0.5,
                        help="Enter a positive % to revise the forecast upward, or negative to revise downward. Use 0 if no adjustment is needed.",
                    )
                note = st.text_area("Notes:")
                if st.button("Save Note"):
                    adj = adjustment_pct if adjustment_pct != 0.0 else None
                    if save_scenario_annotation(selected, note, analyst,
                                                scenario_type, adj, approval_status):
                        st.success("Note saved")

            with st.expander("📋 View Previous Notes"):
                if selected:
                    annotations = load_annotations(selected)
                    if not annotations.empty:
                        st.dataframe(annotations, use_container_width=True, hide_index=True)
                    else:
                        st.info("No annotations yet for this segment.")
        else:
            st.warning("No data found. The pipeline may still be initialising — please check back in a few minutes.")
