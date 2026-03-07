import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from db import load_bootstrap_summary, load_reserve_scenarios, load_reserve_evolution, load_reserve_triangle, load_runoff_projection
from endpoints import call_bootstrap_endpoint
from auth import email_from_token
from lakebase import save_scenario_annotation, load_annotations
from constants import RESERVE_SCENARIO_PRESETS, SEVERITY_LEVELS, CANADIAN_PROVINCES, fmt_dollars


def render(tab):
    with tab:
        st.subheader("Scenario Analysis")
        st.caption(
            "Pre-modelled scenarios, custom stress tests with audit logging, reserve evolution, and triangle analysis"
        )

        # ── Section 1: Pre-computed reserve scenarios ───────────────────────────────
        st.markdown("### Pre-Modelled Reserve Scenarios")
        st.caption(
            "Pre-run at deployment — adverse development, judicial inflation, pandemic tail, "
            "and superimposed inflation scenarios."
        )

        _baseline_smry = load_bootstrap_summary()
        _scenario_df   = load_reserve_scenarios()

        if not _scenario_df.empty:
            _b_var995  = float(_baseline_smry.get('var_995', 0))
            _b_var99   = float(_baseline_smry.get('var_99', 0))
            _b_cvar99  = float(_baseline_smry.get('cvar_99', 0))
            _b_best_est = float(_baseline_smry.get('best_estimate', 0))

            _disp_rows = [{"Scenario": "Baseline", "Best Estimate": fmt_dollars(_b_best_est),
                           "VaR(99%)": fmt_dollars(_b_var99), "Reserve Risk 99.5%": fmt_dollars(_b_var995),
                           "CVaR(99%)": fmt_dollars(_b_cvar99), "Δ VaR 99.5%": "—"}]
            _scenario_df = _scenario_df[_scenario_df["scenario_label"] != "Baseline"]
            for _, row in _scenario_df.iterrows():
                _disp_rows.append({
                    "Scenario":       row["scenario_label"],
                    "Best Estimate":  fmt_dollars(row['best_estimate_M']),
                    "VaR(99%)":       fmt_dollars(row['var_99_M']),
                    "Reserve Risk 99.5%": fmt_dollars(row['var_995_M']),
                    "CVaR(99%)":      fmt_dollars(row['cvar_99_M']),
                    "Δ VaR 99.5%":   f"{row['var_995_vs_baseline']:+.1f}%",
                })
            st.dataframe(pd.DataFrame(_disp_rows), use_container_width=True, hide_index=True)

            _all_labels = ["Baseline"] + _scenario_df["scenario_label"].tolist()
            _all_var995 = [_b_var995] + _scenario_df["var_995_M"].tolist()
            _all_cvar99 = [_b_cvar99] + _scenario_df["cvar_99_M"].tolist()
            _use_b = _b_var995 >= 1000
            _dv = 1000.0 if _use_b else 1.0
            _un = "B" if _use_b else "M"
            _bar_colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#e377c2"]

            _fig_scenario = go.Figure()
            _fig_scenario.add_trace(go.Bar(
                name="Reserve Risk (VaR 99.5%)",
                x=_all_labels, y=[v / _dv for v in _all_var995],
                marker_color=_bar_colors[:len(_all_labels)],
                text=[fmt_dollars(v) for v in _all_var995], textposition="outside",
                hovertemplate=f"%{{x}}<br>VaR(99.5%): $%{{y:.1f}}{_un}<extra></extra>",
            ))
            _fig_scenario.add_trace(go.Bar(
                name="CVaR(99%)",
                x=_all_labels, y=[v / _dv for v in _all_cvar99],
                marker_color=[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.5)" for c in _bar_colors[:len(_all_labels)]],
                text=[fmt_dollars(v) for v in _all_cvar99], textposition="outside",
                hovertemplate=f"%{{x}}<br>CVaR(99%): $%{{y:.1f}}{_un}<extra></extra>",
            ))
            _fig_scenario.update_layout(
                title="Reserve Risk by Scenario — VaR(99.5%) and CVaR(99%)",
                yaxis_title=f"Total IBNR (${_un})",
                barmode="group", height=400, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, max(_all_var995 + _all_cvar99) / _dv * 1.25]),
            )
            st.plotly_chart(_fig_scenario, use_container_width=True)
        else:
            st.info("Reserve scenario data not yet available — run the setup job.")

        # ── Section 2: Reserve evolution timeline ─────────────────────────────────
        st.markdown("### Reserve Adequacy Outlook — Next 12 Months")
        _evolution_df = load_reserve_evolution()
        if not _evolution_df.empty:
            _tl_max = max(_evolution_df["var_995_M"].max(), _evolution_df["var_99_M"].max())
            _tl_b = _tl_max >= 1000
            _tl_dv = 1000.0 if _tl_b else 1.0
            _tl_un = "B" if _tl_b else "M"

            _fig_tl = go.Figure()
            _fig_tl.add_trace(go.Scatter(
                x=_evolution_df["month_idx"], y=_evolution_df["var_995_M"] / _tl_dv,
                mode="lines+markers", name="Reserve Risk (VaR 99.5%)", line=dict(color="#d62728"),
                hovertemplate=f"Month +%{{x}}<br>VaR(99.5%): $%{{y:.1f}}{_tl_un}<extra></extra>",
            ))
            _fig_tl.add_trace(go.Scatter(
                x=_evolution_df["month_idx"], y=_evolution_df["var_99_M"] / _tl_dv,
                mode="lines+markers", name="VaR(99%)", line=dict(color="#ff7f0e", dash="dash"),
                hovertemplate=f"Month +%{{x}}<br>VaR(99%): $%{{y:.1f}}{_tl_un}<extra></extra>",
            ))
            _fig_tl.add_trace(go.Scatter(
                x=_evolution_df["month_idx"], y=_evolution_df["best_estimate_M"] / _tl_dv,
                mode="lines+markers", name="Best Estimate", line=dict(color="#2ca02c", dash="dot"),
                hovertemplate=f"Month +%{{x}}<br>Best Est: $%{{y:.1f}}{_tl_un}<extra></extra>",
            ))
            _b_v995 = float(_baseline_smry.get('var_995', 0))
            if _b_v995 > 0:
                _fig_tl.add_hline(
                    y=_b_v995 / _tl_dv, line_dash="dot", line_color="grey",
                    annotation_text=f"Current VaR(99.5%): {fmt_dollars(_b_v995)}",
                    annotation_position="bottom right",
                )
            _fig_tl.update_layout(
                title="How Reserve Risk Is Projected to Change Over the Next 12 Months",
                xaxis_title="Month", yaxis_title=f"Reserve Requirement (${_tl_un})",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(_fig_tl, use_container_width=True)
        else:
            st.info("Reserve evolution not yet available. Run the full setup job to generate.")

        st.divider()

        # ── Section 3: Custom reserve scenario submission ─────────────────────────
        st.markdown("### Model a Custom Reserve Scenario")
        st.caption(
            "Configure reserve deterioration assumptions and immediately see the impact "
            "on IBNR distribution. Results are saved to the audit log."
        )

        _sc_top1, _sc_top2 = st.columns(2)
        with _sc_top1:
            _sc_type = st.selectbox(
                "Scenario type:",
                list(RESERVE_SCENARIO_PRESETS.keys()),
                help="Each scenario pre-fills realistic reserve assumptions. You can adjust them below.",
            )
            st.caption(f"_{RESERVE_SCENARIO_PRESETS[_sc_type]['desc']}_")
            _severity = st.selectbox(
                "Severity level:",
                list(SEVERITY_LEVELS.keys()),
                index=1,
                help="Scales the scenario's LDF and inflation adjustments.",
            )
        with _sc_top2:
            _affected_lines = st.multiselect(
                "Affected Lines of Business:",
                ["Personal Auto", "Commercial Auto", "Homeowners", "Commercial Property"],
                default=["Personal Auto", "Commercial Auto"],
                help="Which lines are most affected — used for documentation.",
            )

        _preset = RESERVE_SCENARIO_PRESETS[_sc_type]
        _sev_mult = SEVERITY_LEVELS[_severity]

        with st.expander("⚙️ Fine-tune reserve assumptions", expanded=False):
            _adj1, _adj2 = st.columns(2)
            with _adj1:
                st.markdown("**Development Factor Adjustments**")
                _sc_ldf_mult = st.slider("LDF Multiplier", 1.0, 2.0,
                                          min(float(_preset['ldf_multiplier'] * _sev_mult), 2.0),
                                          0.05, key="sc_ldf")
                _sc_cv_mult = st.slider("CV Multiplier", 1.0, 3.0,
                                         min(float(_preset['cv_mult'] * _sev_mult), 3.0),
                                         0.1, key="sc_cv")
            with _adj2:
                st.markdown("**Inflation Adjustments**")
                _sc_inflation = st.slider("Calendar-Year Inflation", 0.0, 0.10,
                                           float(_preset['inflation_adj'] * _sev_mult),
                                           0.005, key="sc_infl", format="%.3f")

            st.markdown("**Per-Line Reserve Volatility (CV)**")
            _cv1, _cv2, _cv3, _cv4 = st.columns(4)
            with _cv1:
                _sc_cv_pa = st.slider("Personal Auto", 0.05, 1.0,
                                       min(0.15 * _sc_cv_mult, 1.0), 0.05, key="sc_cv_pa")
            with _cv2:
                _sc_cv_ca = st.slider("Commercial Auto", 0.05, 1.0,
                                       min(0.18 * _sc_cv_mult, 1.0), 0.05, key="sc_cv_ca")
            with _cv3:
                _sc_cv_ho = st.slider("Homeowners", 0.05, 1.0,
                                       min(0.12 * _sc_cv_mult, 1.0), 0.05, key="sc_cv_ho")
            with _cv4:
                _sc_cv_cp = st.slider("Commercial Prop", 0.05, 1.0,
                                       min(0.20 * _sc_cv_mult, 1.0), 0.05, key="sc_cv_cp")

        _sc_n_rep = st.select_slider(
            "Bootstrap replications:",
            options=[5_000, 10_000, 25_000, 50_000],
            value=10_000,
            key="sc_nrep",
        )

        _user_tok5 = st.context.headers.get("X-Forwarded-Access-Token", "")
        _sc_analyst = email_from_token(_user_tok5) if _user_tok5 else ""
        _sc_analyst_in = st.text_input("Analyst:", value=_sc_analyst, key="sc_analyst")
        _sc_note_in    = st.text_area(
            "Scenario rationale / assumptions:",
            placeholder=f"Describe the {_sc_type} scenario — basis for assumptions, regulatory context...",
            key="sc_note",
        )

        if st.button("🔍 Run Reserve Scenario", type="primary"):
            _sc_params = {
                "scenario":        _preset['scenario'],
                "ldf_multiplier":  _sc_ldf_mult,
                "inflation_adj":   _sc_inflation,
                "cv_personal_auto":     _sc_cv_pa,
                "cv_commercial_auto":   _sc_cv_ca,
                "cv_homeowners":        _sc_cv_ho,
                "cv_commercial_property": _sc_cv_cp,
                "n_replications":  _sc_n_rep,
            }
            with st.spinner(f"Running {_sc_type} ({_severity}) — {_sc_n_rep:,} replications..."):
                _sc_result = call_bootstrap_endpoint(_sc_params)

            if _sc_result:
                st.success(f"✅ {_sc_type} scenario complete")

                _b5 = _baseline_smry

                def _sc_delta(sc_val, base_key):
                    bv = float(_b5.get(base_key, 0))
                    d  = sc_val - bv
                    return f"{'+' if d >= 0 else ''}{fmt_dollars(d)}"

                _cr1, _cr2, _cr3, _cr4 = st.columns(4)
                _cr1.metric("Best Estimate IBNR",     fmt_dollars(_sc_result['best_estimate_M']),
                            delta=_sc_delta(_sc_result['best_estimate_M'], 'best_estimate'))
                _cr2.metric("VaR 99%",          fmt_dollars(_sc_result['var_99_M']),
                            delta=_sc_delta(_sc_result['var_99_M'], 'var_99'))
                _cr3.metric("Reserve Risk 99.5%",  fmt_dollars(_sc_result['var_995_M']),
                            delta=_sc_delta(_sc_result['var_995_M'], 'var_995'))
                _cr4.metric("CVaR 99%",         fmt_dollars(_sc_result['cvar_99_M']),
                            delta=_sc_delta(_sc_result['cvar_99_M'], 'cvar_99'))

                # Save annotation to Lakebase
                _var_lift = ((_sc_result['var_995_M'] / max(float(_b5.get('var_995', 1)), 0.01)) - 1.0) * 100
                _full_note = (
                    f"[{_sc_type} | {_severity}] "
                    f"Lines: {', '.join(_affected_lines) if _affected_lines else 'N/A'} | "
                    f"LDF×{_sc_ldf_mult:.2f}, CV×{_sc_cv_mult:.1f}, Infl={_sc_inflation:.1%} | "
                    f"VaR(99.5%)={fmt_dollars(_sc_result['var_995_M'])}, CVaR={fmt_dollars(_sc_result['cvar_99_M'])}"
                    + (f" | {_sc_note_in}" if _sc_note_in else "")
                )
                if save_scenario_annotation(
                    segment_id="RESERVE_SCENARIO",
                    note=_full_note,
                    analyst=_sc_analyst_in,
                    scenario_type=f"Reserve: {_sc_type}",
                    adjustment_pct=round(_var_lift, 1),
                    approval_status="Draft",
                ):
                    st.caption("✓ Scenario saved to audit log")
            else:
                st.warning(
                    "Bootstrap endpoint not available. "
                    "Start it from Module 4 or wait for the setup job to complete."
                )

        # Recent scenarios
        st.divider()
        st.markdown("### Recent Reserve Scenario Audit Log")
        _sc_history = load_annotations("RESERVE_SCENARIO")
        if _sc_history.empty:
            # Try old key for backward compatibility
            _sc_history = load_annotations("CAT_SCENARIO")
        if not _sc_history.empty:
            st.dataframe(_sc_history, use_container_width=True, hide_index=True)
        else:
            st.info("No reserve scenarios submitted yet. Use the form above to run one.")

        # ── Section 4: Run-off projection ─────────────────────────────────────────
        st.divider()
        st.markdown("### Run-Off Projection — 12-Month Outlook")
        st.caption(
            "Multi-period reserve development with 2-state regime-switching (Normal/Crisis). "
            "Shows how surplus evolves as reserves develop, premiums are collected, and claims are paid."
        )
        _runoff_df = load_runoff_projection()
        if not _runoff_df.empty:
            _s_max = _runoff_df[["surplus_p95", "surplus_p05", "surplus_p50"]].abs().max().max()
            _s_b = _s_max >= 1000
            _s_dv = 1000.0 if _s_b else 1.0
            _s_un = "B" if _s_b else "M"

            _fig_runoff = go.Figure()

            _fig_runoff.add_trace(go.Scatter(
                x=_runoff_df["month"], y=_runoff_df["surplus_p95"] / _s_dv,
                mode="lines", line=dict(width=0), showlegend=False,
                hovertemplate=f"Month %{{x}}<br>95th pctl: $%{{y:.1f}}{_s_un}<extra></extra>",
            ))
            _fig_runoff.add_trace(go.Scatter(
                x=_runoff_df["month"], y=_runoff_df["surplus_p05"] / _s_dv,
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(31,119,180,0.15)", name="5th–95th percentile",
                hovertemplate=f"Month %{{x}}<br>5th pctl: $%{{y:.1f}}{_s_un}<extra></extra>",
            ))

            _fig_runoff.add_trace(go.Scatter(
                x=_runoff_df["month"], y=_runoff_df["surplus_p75"] / _s_dv,
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            _fig_runoff.add_trace(go.Scatter(
                x=_runoff_df["month"], y=_runoff_df["surplus_p25"] / _s_dv,
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(31,119,180,0.30)", name="25th–75th percentile",
            ))

            _fig_runoff.add_trace(go.Scatter(
                x=_runoff_df["month"], y=_runoff_df["surplus_p50"] / _s_dv,
                mode="lines+markers", name="Median surplus",
                line=dict(color="#1f77b4", width=2),
                hovertemplate=f"Month %{{x}}<br>Median: $%{{y:.1f}}{_s_un}<extra></extra>",
            ))

            _fig_runoff.add_hline(y=0, line_dash="dot", line_color="red",
                                   annotation_text="Ruin threshold",
                                   annotation_position="bottom right")

            _fig_runoff.update_layout(
                title="Run-Off Surplus Trajectory with Regime-Switching (50,000 scenarios)",
                xaxis_title="Month", yaxis_title=f"Surplus (${_s_un})", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(_fig_runoff, use_container_width=True)

            _ruin_final = _runoff_df["ruin_probability"].iloc[-1]
            if _ruin_final > 0:
                st.metric("12-Month Ruin Probability", f"{_ruin_final:.4%}",
                          help="Probability that surplus drops below zero within 12 months.")
            else:
                st.info("No ruin events observed in 50,000 scenarios — surplus remains healthy.")
        else:
            st.info("Run-off projection data not yet available. Run Module 3 to generate.")

        # ── Section 5: Reserve Development Triangle ──────────────────────────────
        st.divider()
        st.markdown("### Loss Development Triangle")
        st.caption(
            "Standard actuarial exhibit showing how claims reserves develop over time. "
            "Each row is an accident month; columns show cumulative paid at each development lag."
        )
        _triangle_data = load_reserve_triangle()
        if not _triangle_data.empty:
            _tri_products = sorted(_triangle_data["product_line"].unique())
            _tri_selected = st.selectbox("Product line:", _tri_products, key="tri_prod")
            _tri_filtered = _triangle_data[_triangle_data["product_line"] == _tri_selected]

            _tri_agg = (
                _tri_filtered
                .groupby(["accident_month", "dev_lag"])
                .agg({"cumulative_paid": "sum", "cumulative_incurred": "sum", "case_reserve": "sum"})
                .reset_index()
            )

            if not _tri_agg.empty:
                _pivot = _tri_agg.pivot_table(
                    index="accident_month", columns="dev_lag",
                    values="cumulative_paid", aggfunc="sum",
                )
                _pivot.columns = [f"Lag {int(c)}" for c in _pivot.columns]
                _pivot.index.name = "Accident Month"

                _display_tri = _pivot.map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                st.dataframe(_display_tri, use_container_width=True)

                with st.expander("📊 Development Factors (Link Ratios)"):
                    st.caption(
                        "Weighted link ratios — the core of the chain ladder method. "
                        "f_k = Σ C(i,k+1) / Σ C(i,k) across accident periods."
                    )
                    _dev_factors = {}
                    _cols = sorted(_tri_agg["dev_lag"].unique())
                    for i in range(len(_cols) - 1):
                        _lag_from = _cols[i]
                        _lag_to = _cols[i + 1]
                        _from_vals = _tri_agg[_tri_agg["dev_lag"] == _lag_from].set_index("accident_month")["cumulative_paid"]
                        _to_vals = _tri_agg[_tri_agg["dev_lag"] == _lag_to].set_index("accident_month")["cumulative_paid"]
                        _common = _from_vals.index.intersection(_to_vals.index)
                        if len(_common) > 0:
                            _ratios = _to_vals[_common] / _from_vals[_common].replace(0, np.nan)
                            _dev_factors[f"{int(_lag_from)}→{int(_lag_to)}"] = round(_ratios.mean(), 3)
                    if _dev_factors:
                        st.dataframe(
                            pd.DataFrame([_dev_factors], index=["Avg Link Ratio"]),
                            use_container_width=True,
                        )
        else:
            st.info("Reserve triangle data not yet available. Run the declarative pipeline to generate.")
