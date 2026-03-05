import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import log, sqrt, pi

from db import load_monte_carlo_summary


def _fit_lognormal(mean_val, var99_val):
    """Fit lognormal mu/sigma from the portfolio mean and 99th percentile.

    The portfolio loss is a sum of correlated lognormals — not itself
    lognormal — but a lognormal approximation captures the right-skew shape
    well enough for visualization purposes.
    """
    z99 = 2.3263  # norm.ppf(0.99)
    log_ratio = log(var99_val / mean_val)
    disc = z99 ** 2 - 2 * log_ratio
    if disc <= 0 or z99 - sqrt(disc) <= 0:
        sigma = 0.30  # safe fallback
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
        st.subheader("Annual Capital Requirements")
        st.caption(
            "Based on 40 million simulated loss scenarios across "
            "Commercial Property, Commercial Auto, and Liability"
        )

        with st.expander("ℹ️ About these numbers", expanded=False):
            st.markdown("""
**What this shows:** How much capital your portfolio needs to hold at different risk tolerance levels, based on 40 million simulated loss scenarios.

**The four risk levels:**
- **Expected Annual Loss** — What you should budget for in a typical year (long-run average)
- **1-in-100 Year Loss** — The loss level you'd only expect to exceed once a century. A common internal risk appetite threshold.
- **Solvency Capital Requirement (1-in-200 Year)** — The Solvency II regulatory capital benchmark. Insurers must hold capital sufficient to survive this level with 99.5% confidence.
- **Tail Risk Estimate** — The average loss across the very worst scenarios (beyond the 1-in-100 threshold). More conservative than VaR because it captures how bad things get in extreme years.

**Why losses cluster and spike:** Most years sit near the expected loss, but a small number of scenarios produce losses many times larger. This "heavy right tail" is normal in insurance — it's why catastrophe reinsurance and capital buffers exist.

**How lines of business interact:** Losses are modelled as correlated — a widespread event (e.g., a major storm) can hit Property and Auto simultaneously, compounding total losses beyond what either line would produce alone.

_Technical: t-Copula (df=4) + lognormal marginals. Correlation matrix: ρ(Prop,Auto)=0.40, ρ(Prop,Liab)=0.20, ρ(Auto,Liab)=0.30._
""")

        summary = load_monte_carlo_summary()

        # ── Metric cards ─────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Expected Annual Loss",
            f"${summary['expected_loss']:.1f}M",
            help="What the portfolio is expected to cost in a typical year — the long-run average across all simulated scenarios. Use this as your base budget assumption.",
        )
        col2.metric(
            "1-in-100 Year Loss",
            f"${summary['var_99']:.1f}M",
            help="The loss level exceeded in only 1% of scenarios. Think of it as the worst year in a century. A common risk appetite benchmark for internal capital planning.",
        )
        col3.metric(
            "Solvency Capital Requirement",
            f"${summary['var_995']:.1f}M",
            help="The Solvency II regulatory capital threshold (SCR) — the 1-in-200 year loss level. Insurers must hold enough capital to cover losses at this level with 99.5% confidence.",
        )
        col4.metric(
            "Tail Risk Estimate",
            f"${summary['cvar_99']:.1f}M",
            help="The average loss across the worst 1% of scenarios — what you'd expect to lose when things go really wrong, beyond the 1-in-100 threshold. More conservative than VaR and used in IFRS 17 risk margin calculations.",
        )

        st.divider()

        # ── Fitted loss distribution with threshold markers ──────────────────────
        mean_val = float(summary["expected_loss"])
        var99_val = float(summary["var_99"])
        var995_val = float(summary["var_995"])
        cvar99_val = float(summary["cvar_99"])

        if mean_val > 0 and var99_val > mean_val:
            mu, sigma = _fit_lognormal(mean_val, var99_val)

            x_max = max(var995_val * 1.5, cvar99_val * 1.3)
            x = np.linspace(max(0.1, mean_val * 0.15), x_max, 1000)
            pdf = _lognormal_pdf(x, mu, sigma)

            # Split curve at thresholds for colored fill regions
            i99 = int(np.searchsorted(x, var99_val))
            i995 = int(np.searchsorted(x, var995_val))
            pdf_at_99 = float(np.interp(var99_val, x, pdf))
            pdf_at_995 = float(np.interp(var995_val, x, pdf))

            x_body = np.append(x[:i99], var99_val)
            y_body = np.append(pdf[:i99], pdf_at_99)

            x_severe = np.concatenate([[var99_val], x[i99:i995], [var995_val]])
            y_severe = np.concatenate([[pdf_at_99], pdf[i99:i995], [pdf_at_995]])

            x_extreme = np.concatenate([[var995_val], x[i995:]])
            y_extreme = np.concatenate([[pdf_at_995], pdf[i995:]])

            fig = go.Figure()

            # Body of the distribution (up to VaR 99%)
            fig.add_trace(go.Scatter(
                x=x_body, y=y_body,
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.22)",
                line=dict(color="rgba(31,119,180,0.85)", width=2),
                name="99% of years",
                hovertemplate="Loss: $%{x:.1f}M<extra></extra>",
            ))

            # Severe tail (VaR 99% → VaR 99.5%)
            fig.add_trace(go.Scatter(
                x=x_severe, y=y_severe,
                fill="tozeroy",
                fillcolor="rgba(255,127,14,0.30)",
                line=dict(color="rgba(31,119,180,0.85)", width=2),
                name="1-in-100 to 1-in-200 year",
                hovertemplate="Loss: $%{x:.1f}M<extra></extra>",
            ))

            # Extreme tail (beyond VaR 99.5%)
            fig.add_trace(go.Scatter(
                x=x_extreme, y=y_extreme,
                fill="tozeroy",
                fillcolor="rgba(214,39,40,0.30)",
                line=dict(color="rgba(31,119,180,0.85)", width=2),
                name="Beyond 1-in-200 year",
                hovertemplate="Loss: $%{x:.1f}M<extra></extra>",
            ))

            # ── Threshold markers ────────────────────────────────────────────────
            # Stagger annotation y-positions so labels don't overlap.
            thresholds = [
                ("Expected Loss",   mean_val,   "#2ca02c", "dash", 0.97),
                ("VaR 99%",         var99_val,  "#ff7f0e", "dash", 0.85),
                ("SCR (VaR 99.5%)", var995_val, "#d62728", "dash", 0.97),
                ("CVaR 99%",        cvar99_val, "#9467bd", "dot",  0.85),
            ]
            for label, val, color, dash, y_frac in thresholds:
                fig.add_shape(
                    type="line", x0=val, x1=val, y0=0, y1=1, yref="paper",
                    line=dict(color=color, width=2, dash=dash),
                )
                fig.add_annotation(
                    x=val, y=y_frac, yref="paper",
                    text=f"<b>{label}</b><br>${val:.1f}M",
                    showarrow=False,
                    font=dict(color=color, size=11),
                    bordercolor=color, borderwidth=1, borderpad=3,
                    bgcolor="rgba(255,255,255,0.88)",
                )

            fig.update_layout(
                title="Portfolio Loss Distribution",
                xaxis_title="Annual Portfolio Loss ($M)",
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
                "Lognormal approximation fitted to the 40M-path Monte Carlo results "
                "(t-Copula, df=4). Blue: 99% of simulated years. "
                "Orange: the 1-in-100 to 1-in-200 year range. "
                "Red: beyond 1-in-200 year (extreme tail). "
                "Dashed lines mark percentile thresholds; "
                "dotted line marks the conditional tail expectation (CVaR)."
            )

        st.markdown("""
> **Regulatory context:** The Solvency Capital Requirement (SCR) is the amount of capital a Solvency II insurer must hold to survive a 1-in-200 year loss event. The Tail Risk Estimate goes further — capturing the average severity of scenarios beyond that threshold, and is increasingly referenced in IFRS 17 risk margin calculations.
""")
