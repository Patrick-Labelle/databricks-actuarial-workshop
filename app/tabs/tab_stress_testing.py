import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from db import load_bootstrap_summary
from endpoints import call_bootstrap_endpoint
from constants import fmt_dollars
from chart_theme import apply_default_layout


def render(tab):
    with tab:
        st.subheader("Reserve Scenario Analysis")
        st.caption("Model the reserve impact of any deterioration scenario in seconds")

        with st.expander("ℹ️ How this works", expanded=False):
            st.markdown("""
**What this does:** Lets you change reserve assumptions for any product line and immediately see how your IBNR distribution changes. Use it to answer questions like:

- "What happens to reserve risk if development factors deteriorate by 20%?"
- "How much additional reserve do we need if claim severity inflation accelerates?"
- "What's the reserve impact of a scenario where all lines deteriorate simultaneously?"

**The three groups of inputs:**
- **LDF Adjustments** — Multipliers on development factors. Values above 1.0 mean claims develop worse than expected (adverse development).
- **Reserve Volatility** — How unpredictable each line's reserves are. Higher volatility means wider IBNR distributions.
- **Calendar-Year Inflation** — Superimposed inflation affecting all future development periods.

**Results** show how your reserve requirements (Best Estimate, VaR 99%, Reserve Risk Capital, CVaR) change versus the pre-computed baseline.

_Technical: Bootstrap Chain Ladder endpoint with scenario-adjusted parameters._
""")

        # ── Parameter inputs ──────────────────────────────────────────────────────
        st.markdown("#### Reserve Assumptions")

        col_ldf, col_cv, col_infl = st.columns(3)

        with col_ldf:
            st.markdown("**LDF Multipliers (development)**")
            ldf_mult = st.slider("LDF Multiplier (all lines)", min_value=1.0, max_value=2.0, value=1.0, step=0.05,
                                  help="Baseline: 1.0. Values above 1.0 inflate late-lag development factors — models adverse reserve development.")

        with col_cv:
            st.markdown("**Reserve Volatility**")
            cv_pers_auto = st.slider("Personal Auto CV", 0.05, 1.0, 0.15, 0.05, key="st_cv_pa",
                                      help="Baseline: 0.15. Higher = more uncertainty in reserve estimates.")
            cv_comm_auto = st.slider("Commercial Auto CV", 0.05, 1.0, 0.18, 0.05, key="st_cv_ca",
                                      help="Baseline: 0.18.")
            cv_home = st.slider("Homeowners CV", 0.05, 1.0, 0.12, 0.05, key="st_cv_ho",
                                 help="Baseline: 0.12.")
            cv_comm_prop = st.slider("Commercial Property CV", 0.05, 1.0, 0.20, 0.05, key="st_cv_cp",
                                      help="Baseline: 0.20.")

        with col_infl:
            st.markdown("**Calendar-Year Inflation**")
            inflation_adj = st.slider("Superimposed Inflation", 0.0, 0.10, 0.0, 0.005, key="st_infl",
                                       format="%.3f",
                                       help="Additional calendar-year trend on top of base development. 0.03 = CPI + 3%.")
            scenario_type = st.selectbox("Scenario type:",
                                          ["baseline", "adverse_development", "judicial_inflation",
                                           "pandemic_tail", "superimposed_inflation"],
                                          key="st_scenario")

        n_rep = st.select_slider(
            "Bootstrap Replications",
            options=[5_000, 10_000, 25_000, 50_000],
            value=10_000,
            help="More replications = more accurate results. 10,000 is good for most analyses.",
        )

        # ── Baseline reference ────────────────────────────────────────────────────
        _baseline_summary = load_bootstrap_summary()

        # ── Run button ────────────────────────────────────────────────────────────
        if st.button("Run Reserve Scenario", type="primary"):
            _scenario = {
                "scenario":        scenario_type,
                "ldf_multiplier":  ldf_mult,
                "inflation_adj":   inflation_adj,
                "cv_personal_auto": cv_pers_auto,
                "cv_commercial_auto": cv_comm_auto,
                "cv_homeowners":   cv_home,
                "cv_commercial_property": cv_comm_prop,
                "n_replications":  n_rep,
            }
            with st.spinner(f"Running Bootstrap ({n_rep:,} replications)..."):
                _fetched = call_bootstrap_endpoint(_scenario)
            if _fetched:
                st.session_state["scenario_result"] = _fetched
            else:
                st.warning(
                    "Bootstrap endpoint not available. "
                    "Start it from Module 4 or wait for the setup job to complete."
                )

        _result = st.session_state.get("scenario_result")
        if _result:
            st.success("Scenario complete")
            st.divider()

            # ── Risk metrics comparison ───────────────────────────────────────
            st.markdown("#### Reserve Impact: Scenario vs. Baseline")

            _b = _baseline_summary

            def _delta(scenario_val, baseline_val):
                if baseline_val and baseline_val != 0:
                    d = scenario_val - float(baseline_val)
                    return f"{'+' if d >= 0 else ''}{fmt_dollars(d)}"
                return None

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric(
                "Best Estimate IBNR",
                fmt_dollars(_result['best_estimate_M']),
                delta=_delta(_result['best_estimate_M'], _b.get('best_estimate', 0)),
                help="Expected total IBNR under this scenario.",
            )
            rc2.metric(
                "VaR 99%",
                fmt_dollars(_result['var_99_M']),
                delta=_delta(_result['var_99_M'], _b.get('var_99', 0)),
                help="1-in-100 year reserve level under this scenario.",
            )
            rc3.metric(
                "Reserve Risk Capital (VaR 99.5%)",
                fmt_dollars(_result['var_995_M']),
                delta=_delta(_result['var_995_M'], _b.get('var_995', 0)),
                help="1-in-200 year reserve level under this scenario.",
            )
            rc4.metric(
                "Tail Risk (CVaR 99%)",
                fmt_dollars(_result['cvar_99_M']),
                delta=_delta(_result['cvar_99_M'], _b.get('cvar_99', 0)),
                help="Average reserve across the worst 1% of outcomes.",
            )

            # ── Comparison chart ──────────────────────────────────────────────
            _metrics_labels = ["Best\nEstimate", "VaR 95%", "VaR 99%", "Reserve Risk\n(VaR 99.5%)", "CVaR 99%"]
            _baseline_vals = [
                float(_b.get("best_estimate", 0)),
                float(_b.get("var_99", 0)) * 0.85,
                float(_b.get("var_99", 0)),
                float(_b.get("var_995", 0)),
                float(_b.get("cvar_99", 0)),
            ]
            _scenario_vals = [
                _result["best_estimate_M"],
                _result.get("var_95_M", _result["var_99_M"] * 0.85),
                _result["var_99_M"],
                _result["var_995_M"],
                _result["cvar_99_M"],
            ]

            _st_max = max(max(_baseline_vals), max(_scenario_vals))
            _st_b = _st_max >= 1000
            _st_dv = 1000.0 if _st_b else 1.0
            _st_un = "B" if _st_b else "M"

            _fig_cmp = go.Figure()
            _fig_cmp.add_trace(go.Bar(
                name="Baseline",
                x=_metrics_labels,
                y=[v / _st_dv for v in _baseline_vals],
                marker_color="rgba(31,119,180,0.7)",
                text=[fmt_dollars(v) for v in _baseline_vals],
                textposition="outside",
                hovertemplate=f"%{{x}}<br>Baseline: $%{{y:.1f}}{_st_un}<extra></extra>",
            ))
            _fig_cmp.add_trace(go.Bar(
                name="Stress Scenario",
                x=_metrics_labels,
                y=[v / _st_dv for v in _scenario_vals],
                marker_color="rgba(214,39,40,0.7)",
                text=[fmt_dollars(v) for v in _scenario_vals],
                textposition="outside",
                hovertemplate=f"%{{x}}<br>Scenario: $%{{y:.1f}}{_st_un}<extra></extra>",
            ))
            apply_default_layout(
                _fig_cmp,
                title="Capital Requirement Change vs. Baseline",
                height=380,
                yaxis_title=f"Annual Portfolio Loss (${_st_un})",
                barmode="group",
            )
            st.plotly_chart(_fig_cmp, use_container_width=True)

            # ── Raw metrics table ──────────────────────────────────────────────
            with st.expander("📋 Detailed results"):
                _display = {
                    "Metric": ["Best Estimate IBNR", "VaR (95%)", "1-in-100 Year Reserve", "Reserve Risk Capital 99.5%", "Tail Risk (CVaR 99%)", "Max IBNR"],
                    "Scenario": [
                        fmt_dollars(_result['best_estimate_M']),
                        fmt_dollars(_result.get('var_95_M', _result['var_99_M'] * 0.85)),
                        fmt_dollars(_result['var_99_M']),
                        fmt_dollars(_result['var_995_M']),
                        fmt_dollars(_result['cvar_99_M']),
                        fmt_dollars(_result.get('max_ibnr_M', _result['var_995_M'] * 1.1)),
                    ],
                    "Replications": [f"{int(_result.get('n_replications_used', n_rep)):,}", "", "", "", "", ""],
                }
                st.dataframe(pd.DataFrame(_display), use_container_width=True, hide_index=True)
