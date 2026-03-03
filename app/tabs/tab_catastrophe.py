import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from db import load_monte_carlo_summary, load_stress_scenarios, load_var_timeline, load_regional_forecast, load_reserve_triangle
from endpoints import call_monte_carlo_endpoint, email_from_token
from lakebase import save_scenario_annotation, load_annotations
from constants import CAT_PRESETS, RETURN_PERIOD_MULT, CANADIAN_PROVINCES


def render(tab):
    with tab:
        st.subheader("Catastrophe Event Analysis")
        st.caption(
            "Pre-modelled natural disasters and market events, plus custom catastrophe simulation"
        )

        # ── Section 1: Pre-computed stress scenarios ───────────────────────────────
        st.markdown("### Pre-Modelled Catastrophe Stress Tests")
        st.caption(
            "Pre-run at deployment across 3 major stress scenarios — 120 million simulated loss paths each."
        )

        _baseline_smry = load_monte_carlo_summary()
        _stress_df     = load_stress_scenarios()

        if not _stress_df.empty:
            _b_var995  = float(_baseline_smry.get('var_995', 0))
            _b_var99   = float(_baseline_smry.get('var_99', 0))
            _b_cvar99  = float(_baseline_smry.get('cvar_99', 0))
            _b_mean    = float(_baseline_smry.get('expected_loss', 0))

            _disp_rows = [{"Scenario": "Baseline", "Exp. Loss": f"${_b_mean:.1f}M",
                           "VaR(99%)": f"${_b_var99:.1f}M", "VaR(99.5%) SCR": f"${_b_var995:.1f}M",
                           "CVaR(99%)": f"${_b_cvar99:.1f}M", "Δ SCR": "—"}]
            for _, row in _stress_df.iterrows():
                _disp_rows.append({
                    "Scenario":       row["scenario_label"],
                    "Exp. Loss":      f"${row['total_mean_M']:.1f}M",
                    "VaR(99%)":       f"${row['var_99_M']:.1f}M",
                    "VaR(99.5%) SCR": f"${row['var_995_M']:.1f}M",
                    "CVaR(99%)":      f"${row['cvar_99_M']:.1f}M",
                    "Δ SCR":   f"{row['var_995_vs_baseline']:+.1f}%",
                })
            st.dataframe(pd.DataFrame(_disp_rows), use_container_width=True, hide_index=True)

            _all_labels = ["Baseline"] + _stress_df["scenario_label"].tolist()
            _all_var995 = [_b_var995] + _stress_df["var_995_M"].tolist()
            _all_cvar99 = [_b_cvar99] + _stress_df["cvar_99_M"].tolist()
            _bar_colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
            _bar_colors_muted = [
                f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.5)"
                for c in _bar_colors
            ]

            _fig_stress = go.Figure()
            _fig_stress.add_trace(go.Bar(
                name="VaR(99.5%) — SCR",
                x=_all_labels, y=_all_var995,
                marker_color=_bar_colors,
                text=[f"${v:.1f}M" for v in _all_var995], textposition="outside",
                hovertemplate="%{x}<br>VaR(99.5%): $%{y:.1f}M<extra></extra>",
            ))
            _fig_stress.add_trace(go.Bar(
                name="CVaR(99%)",
                x=_all_labels, y=_all_cvar99,
                marker_color=_bar_colors_muted,
                text=[f"${v:.1f}M" for v in _all_cvar99], textposition="outside",
                hovertemplate="%{x}<br>CVaR(99%): $%{y:.1f}M<extra></extra>",
            ))
            _fig_stress.update_layout(
                title="VaR(99.5%) and CVaR(99%) — Baseline vs. Stress Scenarios",
                yaxis_title="Annual Portfolio Loss ($M)",
                barmode="group", height=400, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, max(_all_var995 + _all_cvar99) * 1.25]),
            )
            st.plotly_chart(_fig_stress, use_container_width=True)
        else:
            st.info("Stress scenario data not yet available — run the setup job (e2-demo-ray target).")

        # ── Section 2: VaR evolution timeline ─────────────────────────────────────
        st.markdown("### Capital Requirement Outlook — Next 12 Months")
        _timeline_df = load_var_timeline()
        if not _timeline_df.empty:
            _fig_tl = go.Figure()
            _fig_tl.add_trace(go.Scatter(
                x=_timeline_df["month_idx"], y=_timeline_df["var_995_M"],
                mode="lines+markers", name="VaR(99.5%)", line=dict(color="#d62728"),
                hovertemplate="Month +%{x}<br>VaR(99.5%): $%{y:.1f}M<extra></extra>",
            ))
            _fig_tl.add_trace(go.Scatter(
                x=_timeline_df["month_idx"], y=_timeline_df["var_99_M"],
                mode="lines+markers", name="VaR(99%)", line=dict(color="#ff7f0e", dash="dash"),
                hovertemplate="Month +%{x}<br>VaR(99%): $%{y:.1f}M<extra></extra>",
            ))
            _b_v995 = float(_baseline_smry.get('var_995', 0))
            if _b_v995 > 0:
                _fig_tl.add_hline(
                    y=_b_v995, line_dash="dot", line_color="grey",
                    annotation_text=f"Current VaR(99.5%): ${_b_v995:.1f}M",
                    annotation_position="bottom right",
                )
            _fig_tl.update_layout(
                title="How Capital Requirements Are Projected to Change Over the Next 12 Months",
                xaxis_title="Month", yaxis_title="Required Capital ($M)",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(_fig_tl, use_container_width=True)
        else:
            st.info("Capital outlook not yet available. Run the full setup job to generate this view.")

        st.divider()

        # ── Section 3: Custom CAT scenario submission ──────────────────────────────
        st.markdown("### Model a Custom Catastrophe Event")
        st.caption(
            "Configure a natural disaster or market event and immediately see its capital impact. Results are saved to the audit log."
        )

        _cat_top1, _cat_top2 = st.columns(2)
        with _cat_top1:
            _cat_type = st.selectbox(
                "Event type:",
                list(CAT_PRESETS.keys()),
                help="Each event type pre-fills realistic loss assumptions for that type of disaster. You can adjust them in the parameters section below.",
            )
            st.caption(f"_{CAT_PRESETS[_cat_type]['desc']}_")
            _return_period = st.selectbox(
                "Event Severity:",
                list(RETURN_PERIOD_MULT.keys()),
                index=1,
                help="How rare and severe the event is. 1-in-250yr is the Solvency II catastrophe benchmark; 1-in-500yr represents an extreme stress test.",
            )
        with _cat_top2:
            _affected_regions = st.multiselect(
                "Affected Regions:",
                CANADIAN_PROVINCES,
                default=["Ontario", "Quebec"],
                help="Which provinces are exposed to this event — used for documentation and the audit log.",
            )
            _affected_lines = st.multiselect(
                "Affected Lines of Business:",
                ["Commercial Property", "Commercial Auto", "Personal Auto", "Homeowners", "Liability"],
                default=["Commercial Property", "Commercial Auto"],
                help="Which lines of business are most directly hit — used for documentation.",
            )

        # Compute default params from preset + return period
        _preset   = CAT_PRESETS[_cat_type]
        _rp_mult  = RETURN_PERIOD_MULT[_return_period]
        _bm, _bc, _bco = [12.5, 8.3, 5.7], [0.35, 0.28, 0.42], [0.40, 0.20, 0.30]
        _def_means = [min(round(m * mu * _rp_mult, 1), 2000.0) for m, mu in zip(_bm, _preset["means_mult"])]
        _def_cvs   = [min(round(c * cu, 2), 3.0) for c, cu in zip(_bc, _preset["cv_mult"])]
        _def_corrs = [min(c + _preset["corr_add"], 0.95) for c in _bco]

        with st.expander("⚙️ Fine-tune loss assumptions", expanded=False):
            _adj1, _adj2, _adj3 = st.columns(3)
            with _adj1:
                st.markdown("**Expected Annual Losses ($M)**")
                _cat_mean_prop = st.number_input("Property", value=_def_means[0], min_value=0.1, max_value=2000.0, step=1.0, key="cat_mp")
                _cat_mean_auto = st.number_input("Auto",     value=_def_means[1], min_value=0.1, max_value=2000.0, step=1.0, key="cat_ma")
                _cat_mean_liab = st.number_input("Liability",value=_def_means[2], min_value=0.1, max_value=2000.0, step=1.0, key="cat_ml")
            with _adj2:
                st.markdown("**Coefficients of Variation**")
                _cat_cv_prop = st.slider("Property CV", 0.05, 3.0, _def_cvs[0], 0.05, key="cat_cvp")
                _cat_cv_auto = st.slider("Auto CV",     0.05, 3.0, _def_cvs[1], 0.05, key="cat_cva")
                _cat_cv_liab = st.slider("Liability CV",0.05, 3.0, _def_cvs[2], 0.05, key="cat_cvl")
            with _adj3:
                st.markdown("**Inter-Line Correlations**")
                _cat_corr_pa = st.slider("Property ↔ Auto",      0.0, 0.95, _def_corrs[0], 0.05, key="cat_cpa")
                _cat_corr_pl = st.slider("Property ↔ Liability", 0.0, 0.95, _def_corrs[1], 0.05, key="cat_cpl")
                _cat_corr_al = st.slider("Auto ↔ Liability",     0.0, 0.95, _def_corrs[2], 0.05, key="cat_cal")
        _cat_n_scen = st.select_slider(
            "Simulation paths:",
            options=[5_000, 10_000, 25_000, 50_000],
            value=25_000,
            help="25,000 recommended for catastrophe scenarios — good balance of accuracy and speed.",
            key="cat_nscen",
        )

        _user_tok5 = st.context.headers.get("X-Forwarded-Access-Token", "")
        _cat_analyst = email_from_token(_user_tok5) if _user_tok5 else ""
        _cat_analyst_in = st.text_input("Analyst:", value=_cat_analyst, key="cat_analyst")
        _cat_note_in    = st.text_area(
            "Scenario rationale / assumptions:",
            placeholder=f"Describe the {_cat_type} scenario — affected exposure, basis for severity, any expert judgment applied...",
            key="cat_note",
        )

        if st.button("🌪️ Run Catastrophe Scenario", type="primary"):
            _cat_scenario_params = {
                "mean_property_M":  _cat_mean_prop,
                "mean_auto_M":      _cat_mean_auto,
                "mean_liability_M": _cat_mean_liab,
                "cv_property":      _cat_cv_prop,
                "cv_auto":          _cat_cv_auto,
                "cv_liability":     _cat_cv_liab,
                "corr_prop_auto":   _cat_corr_pa,
                "corr_prop_liab":   _cat_corr_pl,
                "corr_auto_liab":   _cat_corr_al,
                "n_scenarios":      _cat_n_scen,
                "copula_df":        4,
            }
            with st.spinner(f"Running {_cat_type} ({_return_period}) — {_cat_n_scen:,} scenarios..."):
                _cat_result = call_monte_carlo_endpoint(_cat_scenario_params)

            if _cat_result:
                st.success(f"✅ {_cat_type} scenario complete")

                _b5 = _baseline_smry

                def _cat_delta(sc_val, base_key):
                    bv = float(_b5.get(base_key, 0))
                    d  = sc_val - bv
                    return f"{'+' if d >= 0 else ''}${d:.1f}M"

                _cr1, _cr2, _cr3, _cr4 = st.columns(4)
                _cr1.metric("Expected Annual Loss",     f"${_cat_result['expected_loss_M']:.1f}M",
                            delta=_cat_delta(_cat_result['expected_loss_M'], 'expected_loss'))
                _cr2.metric("1-in-100 Year Loss",          f"${_cat_result['var_99_M']:.1f}M",
                            delta=_cat_delta(_cat_result['var_99_M'], 'var_99'))
                _cr3.metric("Solvency Capital Requirement",  f"${_cat_result['var_995_M']:.1f}M",
                            delta=_cat_delta(_cat_result['var_995_M'], 'var_995'))
                _cr4.metric("Tail Risk Estimate",         f"${_cat_result['cvar_99_M']:.1f}M",
                            delta=_cat_delta(_cat_result['cvar_99_M'], 'cvar_99'))

                # Save annotation to Lakebase
                _var_lift = ((_cat_result['var_995_M'] / max(float(_b5.get('var_995', 1)), 0.01)) - 1.0) * 100
                _full_note = (
                    f"[{_cat_type} | {_return_period}] "
                    f"Regions: {', '.join(_affected_regions) if _affected_regions else 'N/A'} | "
                    f"Lines: {', '.join(_affected_lines) if _affected_lines else 'N/A'} | "
                    f"Prop ${_cat_mean_prop:.1f}M/CV={_cat_cv_prop:.2f}, "
                    f"Auto ${_cat_mean_auto:.1f}M/CV={_cat_cv_auto:.2f}, "
                    f"Liab ${_cat_mean_liab:.1f}M/CV={_cat_cv_liab:.2f} | "
                    f"VaR(99.5%)=${_cat_result['var_995_M']:.1f}M, CVaR=${_cat_result['cvar_99_M']:.1f}M"
                    + (f" | {_cat_note_in}" if _cat_note_in else "")
                )
                if save_scenario_annotation(
                    segment_id="CAT_SCENARIO",
                    note=_full_note,
                    analyst=_cat_analyst_in,
                    scenario_type=f"Catastrophe: {_cat_type}",
                    adjustment_pct=round(_var_lift, 1),
                    approval_status="Draft",
                ):
                    st.caption("✓ Scenario saved to audit log")
            else:
                st.warning(
                    "Monte Carlo endpoint not available. "
                    "Start it from Module 6 or wait for the setup job to complete."
                )

        # Recent CAT scenarios
        st.divider()
        st.markdown("### Recent Catastrophe Scenario Audit Log")
        _cat_history = load_annotations("CAT_SCENARIO")
        if not _cat_history.empty:
            st.dataframe(_cat_history, use_container_width=True, hide_index=True)
        else:
            st.info("No catastrophe scenarios submitted yet. Use the form above to run one.")

        # ── Section 4: Regional Forecast Context ─────────────────────────────────
        st.divider()
        st.markdown("### Regional Claims Forecast")
        st.caption(
            "Geographic breakdown of projected claims over the next 12 months — shows where "
            "exposure is concentrated and helps contextualize catastrophe stress scenarios."
        )
        _reg_forecast = load_regional_forecast()
        if not _reg_forecast.empty:
            # Pivot to get regions as rows, months as value
            _reg_total = _reg_forecast.groupby("region")["total_forecast_claims"].sum().reset_index()
            _reg_total = _reg_total.sort_values("total_forecast_claims", ascending=True)
            _fig_reg = px.bar(
                _reg_total, x="total_forecast_claims", y="region",
                orientation="h",
                title="12-Month Cumulative Projected Claims by Region",
                labels={"total_forecast_claims": "Total Projected Claims", "region": "Region"},
            )
            _fig_reg.update_layout(height=400)
            st.plotly_chart(_fig_reg, use_container_width=True)

            with st.expander("📋 Regional forecast detail"):
                st.dataframe(_reg_forecast, use_container_width=True, hide_index=True)
        else:
            st.info("Regional forecast data not yet available. Run Module 4 (Ray target) to generate.")

        # ── Section 5: Reserve Development Triangle ──────────────────────────────
        st.divider()
        st.markdown("### Loss Development Triangle")
        st.caption(
            "Standard actuarial exhibit showing how claims reserves develop over time. "
            "Each row is an accident month; columns show cumulative paid at each development lag."
        )
        _triangle_data = load_reserve_triangle()
        if not _triangle_data.empty:
            # Let user select a product line to view
            _tri_products = sorted(_triangle_data["product_line"].unique())
            _tri_selected = st.selectbox("Product line:", _tri_products, key="tri_prod")
            _tri_filtered = _triangle_data[_triangle_data["product_line"] == _tri_selected]

            # Aggregate across regions for the selected product line
            _tri_agg = (
                _tri_filtered
                .groupby(["accident_month", "dev_lag"])
                .agg({"cumulative_paid": "sum", "cumulative_incurred": "sum", "case_reserve": "sum"})
                .reset_index()
            )

            # Pivot to triangle format: accident_month × dev_lag → cumulative_paid
            if not _tri_agg.empty:
                _pivot = _tri_agg.pivot_table(
                    index="accident_month", columns="dev_lag",
                    values="cumulative_paid", aggfunc="sum",
                )
                _pivot.columns = [f"Lag {int(c)}" for c in _pivot.columns]
                _pivot.index.name = "Accident Month"

                # Format as currency
                _display_tri = _pivot.map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                st.dataframe(_display_tri, use_container_width=True)

                # Development factor summary
                with st.expander("📊 Development Factors"):
                    st.caption(
                        "Link ratios (cumulative paid at lag N / cumulative paid at lag N-1) — "
                        "used by actuaries to estimate ultimate losses via the chain ladder method."
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
            st.info("Reserve triangle data not yet available. Run the DLT pipeline to generate.")
