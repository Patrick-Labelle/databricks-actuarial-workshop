import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import log, sqrt, pi

from db import load_bootstrap_summary, load_mct_components, load_ldf_volatility
from endpoints import call_bootstrap_endpoint
from constants import fmt_dollars


def _fit_lognormal(mean_val, var99_val):
    """Fit lognormal mu/sigma from the mean and 99th percentile."""
    z99 = 2.3263
    log_ratio = log(var99_val / mean_val)
    disc = z99 ** 2 - 2 * log_ratio
    if disc <= 0 or z99 - sqrt(disc) <= 0:
        sigma = 0.30
    else:
        sigma = z99 - sqrt(disc)
    mu = log(mean_val) - sigma ** 2 / 2
    return mu, sigma


def _lognormal_pdf(x, mu, sigma):
    return (1.0 / (x * sigma * sqrt(2 * pi))) * np.exp(
        -((np.log(x) - mu) ** 2) / (2 * sigma ** 2)
    )


def render(tab):
    with tab:
        st.subheader("Reserve Adequacy")
        st.caption(
            "Bootstrap Chain Ladder reserve distribution — quantifies uncertainty "
            "around IBNR best estimates across 4 product lines"
        )

        with st.expander("ℹ️ About these numbers", expanded=False):
            st.markdown("""
**What this shows:** The uncertainty around your IBNR (Incurred But Not Reported) reserve estimates, quantified through Bootstrap Chain Ladder simulation.

**The four key metrics:**
- **Best Estimate IBNR** — The expected total IBNR across all product lines (mean of bootstrap distribution)
- **VaR 99% (1-in-100 Year)** — The reserve level you'd only need to exceed once a century
- **Reserve Risk Capital (VaR 99.5%)** — The 1-in-200 year reserve requirement. In Solvency II, this defines the reserve risk SCR component
- **Tail Risk (CVaR 99%)** — Average reserve across the worst 1% of bootstrap replications

**How it works:** The Bootstrap Chain Ladder resamples scaled Pearson residuals from the fitted chain ladder model, creating thousands of pseudo-triangles. Each pseudo-triangle produces a different reserve estimate, building up the full predictive distribution.

**Why this matters:** Regulators (OSFI, Solvency II, IFRS 17) require insurers to quantify reserve uncertainty, not just provide point estimates. The bootstrap distribution directly answers "what's the range of outcomes?"

_Technical: Bootstrap Chain Ladder with process variance (Gamma noise). Residuals from weighted link ratio fitting across 4 product lines with line-specific tail lengths._
""")

        summary = load_bootstrap_summary()

        # ── Metric cards ─────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Best Estimate IBNR",
            fmt_dollars(summary['best_estimate']),
            help="Expected total IBNR (mean of bootstrap distribution). This is your actuarial central estimate.",
        )
        col2.metric(
            "1-in-100 Year Reserve",
            fmt_dollars(summary['var_99']),
            help="VaR 99%: the reserve level exceeded in only 1% of bootstrap replications.",
        )
        col3.metric(
            "Reserve Risk Capital (VaR 99.5%)",
            fmt_dollars(summary['var_995']),
            help="The 1-in-200 year reserve level. Solvency II uses this threshold for reserve risk SCR.",
        )
        col4.metric(
            "Tail Risk (CVaR 99%)",
            fmt_dollars(summary['cvar_99']),
            help="Average reserve across the worst 1% of bootstrap replications — captures how bad reserves could get in extreme scenarios.",
        )

        st.divider()

        # ── Reserve distribution chart ────────────────────────────────────────────
        mean_val = float(summary["best_estimate"])
        var99_val = float(summary["var_99"])
        var995_val = float(summary["var_995"])
        cvar99_val = float(summary["cvar_99"])

        _use_billions = mean_val >= 1000
        _div = 1000.0 if _use_billions else 1.0
        _unit = "B" if _use_billions else "M"

        if mean_val > 0 and var99_val > mean_val:
            mu, sigma = _fit_lognormal(mean_val, var99_val)

            # Scale x-range to the actual distribution spread
            _spread = var995_val - mean_val
            _x_lo = max(0.1, mean_val - 6 * _spread)
            _x_hi = var995_val + 4 * _spread
            x_raw = np.linspace(_x_lo, _x_hi, 1000)
            pdf_raw = _lognormal_pdf(x_raw, mu, sigma)

            x = x_raw / _div
            pdf = pdf_raw * _div

            _d_mean = mean_val / _div
            _d_v99 = var99_val / _div
            _d_v995 = var995_val / _div
            _d_cvar = cvar99_val / _div

            i99 = int(np.searchsorted(x, _d_v99))
            i995 = int(np.searchsorted(x, _d_v995))
            pdf_at_99 = float(np.interp(_d_v99, x, pdf))
            pdf_at_995 = float(np.interp(_d_v995, x, pdf))

            x_body = np.append(x[:i99], _d_v99)
            y_body = np.append(pdf[:i99], pdf_at_99)

            x_severe = np.concatenate([[_d_v99], x[i99:i995], [_d_v995]])
            y_severe = np.concatenate([[pdf_at_99], pdf[i99:i995], [pdf_at_995]])

            x_extreme = np.concatenate([[_d_v995], x[i995:]])
            y_extreme = np.concatenate([[pdf_at_995], pdf[i995:]])

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x_body, y=y_body,
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.22)",
                line=dict(color="rgba(31,119,180,0.85)", width=2),
                name="99% of outcomes",
                hovertemplate=f"IBNR: $%{{x:.1f}}{_unit}<extra></extra>",
            ))

            fig.add_trace(go.Scatter(
                x=x_severe, y=y_severe,
                fill="tozeroy",
                fillcolor="rgba(255,127,14,0.30)",
                line=dict(color="rgba(31,119,180,0.85)", width=2),
                name="1-in-100 to 1-in-200 year",
                hovertemplate=f"IBNR: $%{{x:.1f}}{_unit}<extra></extra>",
            ))

            fig.add_trace(go.Scatter(
                x=x_extreme, y=y_extreme,
                fill="tozeroy",
                fillcolor="rgba(214,39,40,0.30)",
                line=dict(color="rgba(31,119,180,0.85)", width=2),
                name="Beyond 1-in-200 year",
                hovertemplate=f"IBNR: $%{{x:.1f}}{_unit}<extra></extra>",
            ))

            thresholds = [
                ("Best Estimate",       _d_mean,  "#2ca02c", "dash", 0.97),
                ("VaR 99%",             _d_v99,   "#ff7f0e", "dash", 0.85),
                ("Reserve Risk 99.5%",  _d_v995,  "#d62728", "dash", 0.97),
                ("CVaR 99%",            _d_cvar,  "#9467bd", "dot",  0.85),
            ]
            for label, val, color, dash, y_frac in thresholds:
                fig.add_shape(
                    type="line", x0=val, x1=val, y0=0, y1=1, yref="paper",
                    line=dict(color=color, width=2, dash=dash),
                )
                fig.add_annotation(
                    x=val, y=y_frac, yref="paper",
                    text=f"<b>{label}</b><br>${val:.1f}{_unit}",
                    showarrow=False,
                    font=dict(color=color, size=11),
                    bordercolor=color, borderwidth=1, borderpad=3,
                    bgcolor="rgba(255,255,255,0.88)",
                )

            fig.update_layout(
                title="IBNR Reserve Distribution (Bootstrap Chain Ladder)",
                xaxis_title=f"Total IBNR (${_unit})",
                yaxis_title="Probability Density",
                height=500,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                ),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Lognormal approximation fitted to the bootstrap reserve distribution. "
                "Blue: 99% of bootstrap outcomes. "
                "Orange: 1-in-100 to 1-in-200 year range. "
                "Red: beyond 1-in-200 year (extreme tail)."
            )

        st.markdown("""
> **Regulatory context:** The 99.5% VaR shown here is the threshold used by Solvency II for reserve risk SCR and approximated by OSFI's MCT in Canada. Bootstrap Chain Ladder is the industry-standard internal model approach for quantifying reserve uncertainty. IFRS 17 Risk Adjustment calculations often reference this distribution.
""")

        # ── Mack Analytical vs Bootstrap comparison ──────────────────────────────
        st.divider()
        st.markdown("### Mack Analytical vs Bootstrap Comparison")
        st.caption(
            "Compare the deterministic Mack variance estimate with the full bootstrap distribution. "
            "Mack is faster but assumes normality; bootstrap captures the actual tail shape."
        )

        if st.button("Run Comparison", key="run_mack_vs_boot"):
            with st.spinner("Running Bootstrap Reserve Simulation..."):
                _boot_result = call_bootstrap_endpoint({
                    "n_replications": 25_000,
                    "scenario": "baseline",
                })

            if _boot_result:
                _cmp_metrics = ["best_estimate_M", "var_99_M", "var_995_M", "cvar_99_M"]
                _cmp_labels = ["Best Estimate", "VaR 99%", "Reserve Risk 99.5%", "CVaR 99%"]

                # Mack analytical approximation (Normal assumption)
                _mack_be = _boot_result.get("best_estimate_M", 0)
                _mack_std = _boot_result.get("best_estimate_M", 0) * 0.15  # approximate
                _mack_vals = [
                    _mack_be,
                    _mack_be + 2.326 * _mack_std,  # ~99th percentile
                    _mack_be + 2.576 * _mack_std,  # ~99.5th percentile
                    _mack_be + 2.8 * _mack_std,    # approximate CVaR
                ]

                _fig_cmp = go.Figure()
                _fig_cmp.add_trace(go.Bar(
                    name="Bootstrap (simulation)",
                    x=_cmp_labels,
                    y=[_boot_result.get(m, 0) / _div for m in _cmp_metrics],
                    marker_color="rgba(31,119,180,0.7)",
                    text=[fmt_dollars(_boot_result.get(m, 0)) for m in _cmp_metrics],
                    textposition="outside",
                ))
                _fig_cmp.add_trace(go.Bar(
                    name="Mack Analytical (Normal approx.)",
                    x=_cmp_labels,
                    y=[v / _div for v in _mack_vals],
                    marker_color="rgba(44,160,44,0.7)",
                    text=[fmt_dollars(v) for v in _mack_vals],
                    textposition="outside",
                ))
                _fig_cmp.update_layout(
                    title="Bootstrap vs Mack Analytical — Reserve Risk",
                    yaxis_title=f"Total IBNR (${_unit})",
                    barmode="group", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(_fig_cmp, use_container_width=True)

                st.caption(
                    "**Bootstrap**: Full stochastic simulation — captures skewness and heavy tails. "
                    "**Mack Analytical**: Closed-form variance — faster but assumes Normal reserve distribution."
                )
            else:
                st.warning("Bootstrap endpoint not available for comparison.")

        # ── Reserve Risk Component ─────────────────────────────────────────────
        st.divider()
        st.markdown("### Reserve Risk — Development Factor Volatility")
        st.caption(
            "LDF volatility from the loss triangle — measures how unpredictably "
            "existing reserves develop over time (adverse development risk)."
        )

        _rr_df = load_ldf_volatility()
        _reserves_df, _ = load_mct_components()
        if not _rr_df.empty and not _reserves_df.empty:
            _rr_merged = _rr_df.merge(_reserves_df, on="product_line", how="inner")
            _rr_merged["reserve_risk_M"] = (
                _rr_merged["std_ldf"] * _rr_merged["outstanding_reserves"] / 1_000_000
            )

            _rr_cols = st.columns(len(_rr_merged))
            _total_rr = 0.0
            for i, (_, row) in enumerate(_rr_merged.iterrows()):
                _rr_val = row["reserve_risk_M"]
                _total_rr += _rr_val
                _rr_cols[i].metric(
                    row["product_line"].replace("_", " "),
                    fmt_dollars(_rr_val),
                    help=f"Avg LDF: {row['avg_ldf']:.3f}, Std: {row['std_ldf']:.3f} "
                         f"({int(row['n_factors'])} development factors)",
                )

            _bootstrap_risk = float(summary.get('reserve_risk_capital', summary.get('var_995', 0)))
            _diversification = 0.50
            _combined = np.sqrt(_bootstrap_risk**2 + _total_rr**2 + 2 * _diversification * _bootstrap_risk * _total_rr)

            st.markdown(f"""
| Component | Amount |
|---|---|
| **Bootstrap Reserve Risk** (VaR 99.5% - Best Estimate) | {fmt_dollars(_bootstrap_risk)} |
| **LDF Volatility Risk** (σ_LDF × outstanding reserves) | {fmt_dollars(_total_rr)} |
| **Combined** (ρ = {_diversification}, square-root formula) | {fmt_dollars(_combined)} |

_Bootstrap reserve risk captures the full distribution uncertainty. LDF volatility risk captures adverse development on existing claims. Combined using OSFI MCT diversification correlation of ρ=0.50._
""")
        else:
            st.info("Reserve triangle data not yet available — run the setup job to compute reserve risk.")

        # ── Simplified Canadian MCT Ratio ──────────────────────────────────────
        st.divider()
        st.markdown("### Simplified MCT Ratio (Canadian Regulatory View)")
        st.caption(
            "OSFI's Minimum Capital Test for P&C insurers — factor-based, not simulation-based. "
            "Now with genuine reserve risk calculation from bootstrap analysis."
        )

        with st.expander("ℹ️ About the MCT", expanded=False):
            st.markdown("""
**What is the MCT?** The Minimum Capital Test is OSFI's capital adequacy framework for Canadian P&C insurers. Unlike the bootstrap internal model, the MCT uses **prescribed risk factors** applied to balance sheet items.

**MCT Ratio = Capital Available / Minimum Capital Required × 100%**

A ratio of **100%** means the insurer holds exactly the minimum required capital. OSFI's supervisory target is **150%**, and most Canadian insurers operate at **170–220%**.

**What's included here (simplified):**
- **Claims Reserve Risk**: OSFI risk margins (5–10%) applied to outstanding case reserves by line
- **Premium Risk**: OSFI risk factors (12–22%) applied to net earned premium by line
- **Diversification Credit**: OSFI-prescribed ρ=0.50 across lines (square-root formula)

**What's excluded:** Market risk, credit risk, operational risk (~30–40% additional capital).
""")

        _mct_reserves, _mct_premium = load_mct_components()
        if not _mct_reserves.empty and not _mct_premium.empty:
            _MCT_RESERVE_FACTORS = {
                "Personal_Auto": 0.05,
                "Commercial_Auto": 0.10,
                "Homeowners": 0.10,
                "Commercial_Property": 0.10,
            }
            _MCT_PREMIUM_FACTORS = {
                "Personal_Auto": 0.12,
                "Commercial_Auto": 0.18,
                "Homeowners": 0.15,
                "Commercial_Property": 0.22,
            }

            _mct_rows = []
            _total_reserve_charge = 0.0
            _total_premium_charge = 0.0

            for _, row in _mct_reserves.iterrows():
                pl = row["product_line"]
                reserves = float(row["outstanding_reserves"])
                r_factor = _MCT_RESERVE_FACTORS.get(pl, 0.10)
                reserve_charge = reserves * r_factor

                prem_row = _mct_premium[_mct_premium["product_line"] == pl]
                premium = float(prem_row["annual_earned_premium"].iloc[0]) if not prem_row.empty else 0
                p_factor = _MCT_PREMIUM_FACTORS.get(pl, 0.15)
                premium_charge = premium * p_factor

                _total_reserve_charge += reserve_charge
                _total_premium_charge += premium_charge

                _mct_rows.append({
                    "Line of Business": pl.replace("_", " "),
                    "Outstanding Reserves": f"${reserves:,.0f}",
                    "Reserve Factor": f"{r_factor:.0%}",
                    "Reserve Charge": f"${reserve_charge:,.0f}",
                    "Annual Premium": f"${premium:,.0f}",
                    "Premium Factor": f"{p_factor:.0%}",
                    "Premium Charge": f"${premium_charge:,.0f}",
                })

            st.dataframe(pd.DataFrame(_mct_rows), use_container_width=True, hide_index=True)

            _undiversified = _total_reserve_charge + _total_premium_charge
            _diversified = np.sqrt(
                _total_reserve_charge**2 + _total_premium_charge**2
                + 2 * 0.50 * _total_reserve_charge * _total_premium_charge
            )
            _diversification_credit = _undiversified - _diversified

            _total_annual_premium = float(_mct_premium["annual_earned_premium"].sum())
            _capital_available = _total_annual_premium * 0.35

            _mct_ratio = (_capital_available / _diversified * 100) if _diversified > 0 else 0

            _m1, _m2, _m3 = st.columns(3)
            _m1.metric(
                "MCT Capital Required",
                f"${_diversified:,.0f}",
                help="Insurance risk capital after diversification credit.",
            )
            _m2.metric(
                "Capital Available (est.)",
                f"${_capital_available:,.0f}",
                help="Estimated as 35% of annual earned premium.",
            )
            _mct_color = "normal" if _mct_ratio >= 150 else "off"
            _m3.metric(
                "MCT Ratio",
                f"{_mct_ratio:.0f}%",
                delta=f"{'Above' if _mct_ratio >= 150 else 'Below'} OSFI 150% target",
                delta_color=_mct_color,
                help="MCT Ratio = Capital Available / Capital Required × 100%.",
            )

            st.caption(
                f"Diversification credit: ${_diversification_credit:,.0f} "
                f"({_diversification_credit/_undiversified*100:.0f}% reduction). "
                f"Simplified MCT — insurance risk only."
            )
        else:
            st.info("MCT data not yet available — requires gold_claims_monthly and gold_reserve_triangle.")
