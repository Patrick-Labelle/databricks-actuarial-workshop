import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from db import load_monte_carlo_summary, load_monte_carlo_distribution


def render(tab):
    with tab:
        st.subheader("Annual Capital Requirements")
        st.caption("Based on 40 million simulated loss scenarios across Commercial Property, Commercial Auto, and Liability")

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

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Expected Annual Loss",
            f"${summary['expected_loss']:.1f}M",
            help="What the portfolio is expected to cost in a typical year — the long-run average across all simulated scenarios. Use this as your base budget assumption."
        )
        col2.metric(
            "1-in-100 Year Loss",
            f"${summary['var_99']:.1f}M",
            help="The loss level exceeded in only 1% of scenarios. Think of it as the worst year in a century. A common risk appetite benchmark for internal capital planning."
        )
        col3.metric(
            "Solvency Capital Requirement",
            f"${summary['var_995']:.1f}M",
            help="The Solvency II regulatory capital threshold (SCR) — the 1-in-200 year loss level. Insurers must hold enough capital to cover losses at this level with 99.5% confidence."
        )
        col4.metric(
            "Tail Risk Estimate",
            f"${summary['cvar_99']:.1f}M",
            help="The average loss across the worst 1% of scenarios — what you'd expect to lose when things go really wrong, beyond the 1-in-100 threshold. More conservative than VaR and used in IFRS 17 risk margin calculations."
        )

        st.divider()

        # Risk metric comparison chart
        metrics = {
            "Expected\nAnnual Loss": float(summary["expected_loss"]),
            "1-in-100\nYear Loss": float(summary["var_99"]),
            "Solvency Capital\nRequirement": float(summary["var_995"]),
            "Tail Risk\nEstimate": float(summary["cvar_99"]),
        }
        colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
        fig2 = go.Figure(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=colors,
            text=[f"${v:.1f}M" for v in metrics.values()],
            textposition="outside",
            hovertemplate="%{x}<br><b>$%{y:.1f}M</b><extra></extra>",
        ))
        fig2.update_layout(
            title="Capital Required at Each Risk Level",
            yaxis_title="Annual Loss ($M)",
            height=380,
            showlegend=False,
            yaxis=dict(range=[0, max(metrics.values()) * 1.25]),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Simulated distribution using the per-row mean losses as a proxy histogram
        dist_df = load_monte_carlo_distribution()
        if not dist_df.empty:
            for col in dist_df.columns:
                dist_df[col] = pd.to_numeric(dist_df[col], errors='coerce')

            with st.expander("📊 How Losses Are Distributed Across Scenarios", expanded=True):
                st.markdown("""
Each bar shows how many of the 40 million simulated years produced losses in that range. The vertical lines mark key capital thresholds.

Most simulated years cluster near the expected loss — but a long tail of rare, high-loss years stretches to the right. This is the fundamental reason insurance companies hold capital buffers well above their average annual loss.
""")
                fig3 = go.Figure()
                fig3.add_trace(go.Histogram(
                    x=dist_df["mean_loss_M"],
                    nbinsx=60,
                    name="Simulated scenarios",
                    marker_color="rgba(31,119,180,0.6)",
                    hovertemplate="Loss: $%{x:.1f}M<br>Count: %{y}<extra></extra>",
                ))
                for label, val, color in [
                    ("Expected Loss", float(summary["expected_loss"]), "#2ca02c"),
                    ("1-in-100yr", float(summary["var_99"]), "#ff7f0e"),
                    ("SCR (1-in-200yr)", float(summary["var_995"]), "#d62728"),
                ]:
                    fig3.add_vline(
                        x=val, line_dash="dash", line_color=color,
                        annotation_text=f"{label}: ${val:.1f}M",
                        annotation_position="top right",
                    )
                fig3.update_layout(
                    xaxis_title="Annual Portfolio Loss ($M)",
                    yaxis_title="Number of Scenarios",
                    height=360,
                    showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
> **Regulatory context:** The Solvency Capital Requirement (SCR) is the amount of capital a Solvency II insurer must hold to survive a 1-in-200 year loss event. The Tail Risk Estimate goes further — capturing the average severity of scenarios beyond that threshold, and is increasingly referenced in IFRS 17 risk margin calculations.
""")
