import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from db import load_regional_summary, load_regional_product_breakdown
from chart_theme import apply_default_layout

# Province centroids (lat, lon) for the bubble map
_COORDS = {
    "Ontario":              (50.0, -85.0),
    "Quebec":               (52.0, -72.0),
    "British_Columbia":     (53.7, -127.6),
    "Alberta":              (53.9, -116.6),
    "Manitoba":             (55.0, -98.8),
    "Saskatchewan":         (54.0, -106.0),
    "New_Brunswick":        (46.5, -66.2),
    "Nova_Scotia":          (44.7, -63.0),
    "Prince_Edward_Island": (46.2, -63.0),
    "Newfoundland":         (53.1, -57.7),
}

_METRIC_OPTIONS = {
    "Total Claims":          "total_claims",
    "Avg Monthly Claims":    "avg_monthly_claims",
    "Total Incurred ($M)":   "total_incurred",
    "Avg Severity ($)":      "avg_severity",
    "Earned Premium ($M)":   "total_premium",
}


def _display_label(region: str) -> str:
    return region.replace("_", " ")


def render(tab):
    with tab:
        st.subheader("Geographical Breakdown")
        st.caption("Claims and reserve metrics across Canadian provinces")

        summary = load_regional_summary()
        if summary.empty:
            st.warning("No regional data available. Ensure gold_claims_monthly is populated.")
            return

        # ── Controls ──────────────────────────────────────────────────────
        col_metric, col_product = st.columns([1, 1])
        with col_metric:
            metric_label = st.selectbox(
                "Colour by metric", list(_METRIC_OPTIONS.keys()), index=0
            )
        metric_col = _METRIC_OPTIONS[metric_label]

        breakdown = load_regional_product_breakdown()
        product_lines = sorted(breakdown["product_line"].unique()) if not breakdown.empty else []
        with col_product:
            selected_product = st.selectbox(
                "Filter by product line", ["All"] + product_lines, index=0
            )

        # Filter breakdown if product selected
        if selected_product != "All" and not breakdown.empty:
            filtered = (
                breakdown[breakdown["product_line"] == selected_product]
                .drop(columns=["product_line"])
                .reset_index(drop=True)
            )
        else:
            filtered = summary.copy()

        # ── Bubble map ────────────────────────────────────────────────────
        map_data = filtered.copy()
        map_data["lat"] = map_data["region"].map(lambda r: _COORDS.get(r, (0, 0))[0])
        map_data["lon"] = map_data["region"].map(lambda r: _COORDS.get(r, (0, 0))[1])
        map_data["display_name"] = map_data["region"].map(_display_label)

        vals = pd.to_numeric(map_data[metric_col], errors="coerce").fillna(0)
        max_val = vals.max() if vals.max() > 0 else 1

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lat=map_data["lat"],
            lon=map_data["lon"],
            text=map_data.apply(
                lambda r: f"<b>{_display_label(r['region'])}</b><br>{metric_label}: {r[metric_col]:,.0f}",
                axis=1,
            ),
            hoverinfo="text",
            marker=dict(
                size=10 + (vals / max_val) * 40,
                color=vals,
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title=metric_label, thickness=15),
                line=dict(width=1, color="white"),
                opacity=0.85,
            ),
        ))
        fig.update_geos(
            scope="north america",
            showland=True, landcolor="#f0f0f0",
            showlakes=True, lakecolor="white",
            showcountries=True, countrycolor="#cccccc",
            showsubunits=True, subunitcolor="#cccccc",
            lonaxis=dict(range=[-142, -50]),
            lataxis=dict(range=[41, 62]),
            bgcolor="white",
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=500,
            title=dict(
                text=f"{metric_label} by Province"
                + (f" — {selected_product}" if selected_product != "All" else ""),
                font=dict(size=16, color="#1E1E1E"),
            ),
            paper_bgcolor="white",
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#444"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Bar chart comparison ──────────────────────────────────────────
        st.subheader("Regional Comparison")

        sorted_df = filtered.sort_values(metric_col, ascending=True)
        bar_fig = go.Figure(go.Bar(
            x=pd.to_numeric(sorted_df[metric_col], errors="coerce"),
            y=sorted_df["region"].map(_display_label),
            orientation="h",
            marker_color=pd.to_numeric(sorted_df[metric_col], errors="coerce"),
            marker_colorscale="YlOrRd",
            text=pd.to_numeric(sorted_df[metric_col], errors="coerce").apply(lambda v: f"{v:,.0f}"),
            textposition="outside",
        ))
        apply_default_layout(
            bar_fig,
            height=400,
            xaxis_title=metric_label,
            yaxis_title="",
            showlegend=False,
            margin=dict(l=0, r=40, t=10, b=40),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # ── Product line heatmap (all products) ──────────────────────────
        if not breakdown.empty and selected_product == "All":
            st.subheader("Product Line x Province Heatmap")
            pivot = breakdown.pivot_table(
                index="region", columns="product_line",
                values=metric_col, aggfunc="sum",
            ).fillna(0)
            pivot.index = pivot.index.map(_display_label)

            heat_fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="YlOrRd",
                texttemplate="%{z:,.0f}",
                hovertemplate="%{y} / %{x}<br>%{z:,.0f}<extra></extra>",
            ))
            apply_default_layout(
                heat_fig,
                height=400,
                margin=dict(l=0, r=0, t=10, b=40),
            )
            st.plotly_chart(heat_fig, use_container_width=True)

        # ── Data table ────────────────────────────────────────────────────
        with st.expander("Raw data"):
            display_df = filtered.copy()
            display_df["region"] = display_df["region"].map(_display_label)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
