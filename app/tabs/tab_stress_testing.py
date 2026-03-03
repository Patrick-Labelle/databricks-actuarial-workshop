import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from db import load_monte_carlo_summary
from endpoints import call_monte_carlo_endpoint


def render(tab):
    with tab:
        st.subheader("Custom Stress Test")
        st.caption("Model the capital impact of any loss scenario in seconds")

        with st.expander("ℹ️ How this works", expanded=False):
            st.markdown("""
**What this does:** Lets you change the loss assumptions for any line of business and immediately see how your capital requirements change. Use it to answer questions like:

- "What happens to our SCR if property losses increase 20%?"
- "How much extra capital do we need if losses become more volatile?"
- "What's the capital impact of a widespread event that hits Property and Auto at the same time?"

**The three groups of inputs:**
- **Expected Annual Losses** — Your best estimate of what each line will cost in the scenario. Raise these to model a hard market, higher exposure, or adverse claims trends.
- **Loss Volatility** — How unpredictable each line is. Higher volatility means wider loss distributions and higher capital requirements.
- **How Lines Move Together** — Whether multiple lines tend to have bad years at the same time. Higher values mean more losses pile up during stress events, increasing required capital.

**Results** show how your capital requirements (Expected Loss, 1-in-100, SCR, Tail Risk) change versus the pre-computed baseline.

_Technical: t-Copula Monte Carlo endpoint, same model as the Portfolio Risk tab._
""")

        # ── Parameter inputs ──────────────────────────────────────────────────────
        st.markdown("#### Loss Assumptions")

        col_means, col_cv, col_corr = st.columns(3)

        with col_means:
            st.markdown("**Expected Annual Losses ($M)**")
            mean_prop = st.number_input("Commercial Property", value=12.5, min_value=0.1, max_value=500.0, step=0.5,
                                        help="Baseline: $12.5M. Increase to model higher property losses — hard market, increased building values, or adverse claims trends.")
            mean_auto = st.number_input("Commercial Auto",     value=8.3,  min_value=0.1, max_value=500.0, step=0.5,
                                        help="Baseline: $8.3M. Increase to model more frequent or severe auto claims.")
            mean_liab = st.number_input("Liability",           value=5.7,  min_value=0.1, max_value=500.0, step=0.5,
                                        help="Baseline: $5.7M. Increase to model higher liability exposure or adverse reserve development.")

        with col_cv:
            st.markdown("**Loss Volatility (unpredictability)**")
            cv_prop = st.slider("Property Volatility", min_value=0.05, max_value=2.0, value=0.35, step=0.05,
                                help="Baseline: 0.35. Higher values mean more unpredictable losses and higher capital requirements.")
            cv_auto = st.slider("Auto Volatility",     min_value=0.05, max_value=2.0, value=0.28, step=0.05,
                                help="Baseline: 0.28. Auto claims tend to be the most stable line.")
            cv_liab = st.slider("Liability Volatility", min_value=0.05, max_value=2.0, value=0.42, step=0.05,
                                help="Baseline: 0.42. Liability is naturally more volatile due to long development tails and legal uncertainty.")

        with col_corr:
            st.markdown("**How Lines Move Together**")
            corr_pa = st.slider("Property ↔ Auto",      min_value=0.0, max_value=0.95, value=0.40, step=0.05,
                                help="Baseline: 0.40. How often Property and Auto have bad years at the same time. Raise to 0.6–0.8 to model a widespread event like a storm affecting both.")
            corr_pl = st.slider("Property ↔ Liability", min_value=0.0, max_value=0.95, value=0.20, step=0.05,
                                help="Baseline: 0.20. Property and Liability losses are relatively independent in normal conditions.")
            corr_al = st.slider("Auto ↔ Liability",     min_value=0.0, max_value=0.95, value=0.30, step=0.05,
                                help="Baseline: 0.30. Auto and Liability sometimes move together — e.g., an economic downturn affecting both claims frequency and litigation.")

        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            n_scen = st.select_slider(
                "Simulation Precision",
                options=[1_000, 5_000, 10_000, 25_000, 50_000],
                value=10_000,
                help="More paths = more accurate results, takes slightly longer. 10,000 is good for most analyses.",
            )
        with col_sim2:
            copula_df_val = st.select_slider(
                "Tail Risk Sensitivity",
                options=[3, 4, 5, 10, 20, 30],
                value=4,
                help="Controls how severely multiple lines are affected simultaneously during extreme events. Lower = more conservative (heavier simultaneous losses). Default of 4 is the actuarial calibration; 30 approximates independent lines.",
            )

        # ── Baseline reference (read from Delta table, cached) ────────────────────
        _baseline_summary = load_monte_carlo_summary()

        # ── Run button ────────────────────────────────────────────────────────────
        if st.button("Run Scenario", type="primary"):
            _scenario = {
                "mean_property_M":  mean_prop,
                "mean_auto_M":      mean_auto,
                "mean_liability_M": mean_liab,
                "cv_property":      cv_prop,
                "cv_auto":          cv_auto,
                "cv_liability":     cv_liab,
                "corr_prop_auto":   corr_pa,
                "corr_prop_liab":   corr_pl,
                "corr_auto_liab":   corr_al,
                "n_scenarios":      n_scen,
                "copula_df":        copula_df_val,
            }
            with st.spinner(f"Running Monte Carlo ({n_scen:,} scenarios)..."):
                _fetched = call_monte_carlo_endpoint(_scenario)
            if _fetched:
                st.session_state["scenario_result"] = _fetched
            else:
                st.warning(
                    "Monte Carlo endpoint not available. "
                    "Start it from Module 6 or wait for the setup job to complete."
                )

        _result = st.session_state.get("scenario_result")
        if _result:
            st.success("Scenario complete")
            st.divider()

            # ── Risk metrics comparison ───────────────────────────────────────
            st.markdown("#### Capital Impact: Scenario vs. Baseline")

            _b = _baseline_summary  # Series from load_monte_carlo_summary()

            def _delta(scenario_val, baseline_val):
                if baseline_val and baseline_val != 0:
                    d = scenario_val - float(baseline_val)
                    return f"{'+' if d >= 0 else ''}${d:.1f}M"
                return None

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric(
                "Expected Annual Loss",
                f"${_result['expected_loss_M']:.1f}M",
                delta=_delta(_result['expected_loss_M'], _b.get('expected_loss', 0)),
                help="What the portfolio is expected to cost per year under these assumptions.",
            )
            rc2.metric(
                "1-in-100 Year Loss",
                f"${_result['var_99_M']:.1f}M",
                delta=_delta(_result['var_99_M'], _b.get('var_99', 0)),
                help="Loss level exceeded only once a century under this scenario.",
            )
            rc3.metric(
                "Solvency Capital Requirement",
                f"${_result['var_995_M']:.1f}M",
                delta=_delta(_result['var_995_M'], _b.get('var_995', 0)),
                help="Required capital under Solvency II for this scenario.",
            )
            rc4.metric(
                "Tail Risk Estimate",
                f"${_result['cvar_99_M']:.1f}M",
                delta=_delta(_result['cvar_99_M'], _b.get('cvar_99', 0)),
                help="Average loss in the very worst outcomes — beyond the 1-in-100 threshold.",
            )

            # ── Waterfall chart: Scenario vs Baseline ─────────────────────────
            _metrics_labels = ["Expected\nAnnual Loss", "VaR 95%", "1-in-100\nYear Loss", "Solvency Capital\nRequirement", "Tail Risk\nEstimate"]
            _baseline_vals = [
                float(_b.get("expected_loss", 0)),
                float(_b.get("var_99", 0)) * 0.85,   # approximate VaR95 from VaR99
                float(_b.get("var_99", 0)),
                float(_b.get("var_995", 0)),
                float(_b.get("cvar_99", 0)),
            ]
            _scenario_vals = [
                _result["expected_loss_M"],
                _result["var_95_M"],
                _result["var_99_M"],
                _result["var_995_M"],
                _result["cvar_99_M"],
            ]

            _fig_cmp = go.Figure()
            _fig_cmp.add_trace(go.Bar(
                name="Baseline",
                x=_metrics_labels,
                y=_baseline_vals,
                marker_color="rgba(31,119,180,0.7)",
                hovertemplate="%{x}<br>Baseline: $%{y:.1f}M<extra></extra>",
            ))
            _fig_cmp.add_trace(go.Bar(
                name="Stress Scenario",
                x=_metrics_labels,
                y=_scenario_vals,
                marker_color="rgba(214,39,40,0.7)",
                hovertemplate="%{x}<br>Scenario: $%{y:.1f}M<extra></extra>",
            ))
            _fig_cmp.update_layout(
                title="Capital Requirement Change vs. Baseline",
                yaxis_title="Annual Portfolio Loss ($M)",
                barmode="group",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(_fig_cmp, use_container_width=True)

            # ── Raw metrics table ──────────────────────────────────────────────
            with st.expander("📋 Detailed results"):
                _display = {
                    "Metric": ["Expected Loss", "VaR (95%)", "1-in-100 Year Loss", "Solvency Capital Req.", "Tail Risk Estimate", "Max Loss"],
                    "Scenario ($M)": [
                        f"${_result['expected_loss_M']:.2f}",
                        f"${_result['var_95_M']:.2f}",
                        f"${_result['var_99_M']:.2f}",
                        f"${_result['var_995_M']:.2f}",
                        f"${_result['cvar_99_M']:.2f}",
                        f"${_result['max_loss_M']:.2f}",
                    ],
                    "Copula": [_result.get("copula", ""), "", "", "", "", ""],
                    "Scenarios": [f"{int(_result.get('n_scenarios_used', n_scen)):,}", "", "", "", "", ""],
                }
                st.dataframe(pd.DataFrame(_display), use_container_width=True, hide_index=True)
