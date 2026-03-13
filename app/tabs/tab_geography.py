import json
import os
from datetime import date

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from db import load_regional_summary, load_regional_product_breakdown
from chart_theme import apply_default_layout

# ── GeoJSON for Canadian province choropleth ─────────────────────────────────
_GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "..", "canada_provinces.geojson")


@st.cache_resource
def _load_geojson():
    with open(_GEOJSON_PATH) as f:
        return json.load(f)


# Map app region names (underscored) → GeoJSON "name" property values
_REGION_TO_GEOJSON = {
    "Ontario":              "Ontario",
    "Quebec":               "Quebec",
    "British_Columbia":     "British Columbia",
    "Alberta":              "Alberta",
    "Manitoba":             "Manitoba",
    "Saskatchewan":         "Saskatchewan",
    "New_Brunswick":        "New Brunswick",
    "Nova_Scotia":          "Nova Scotia",
    "Prince_Edward_Island": "Prince Edward Island",
    "Newfoundland":         "Newfoundland and Labrador",
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

        # ── Date range filter ─────────────────────────────────────────
        col_start, col_end, _ = st.columns([1, 1, 2])
        with col_start:
            start_dt = st.date_input(
                "Start month", value=date(2025, 1, 1),
                min_value=date(2019, 1, 1), max_value=date(2025, 12, 1),
                key="geo_start",
            )
        with col_end:
            end_dt = st.date_input(
                "End month", value=date(2025, 12, 31),
                min_value=date(2019, 1, 1), max_value=date(2025, 12, 31),
                key="geo_end",
            )
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        summary = load_regional_summary(start_str, end_str)
        if summary.empty:
            st.warning("No regional data available for the selected period.")
            return

        # ── Controls ──────────────────────────────────────────────────────
        col_metric, col_product = st.columns([1, 1])
        with col_metric:
            metric_label = st.selectbox(
                "Metric", list(_METRIC_OPTIONS.keys()), index=2
            )
        metric_col = _METRIC_OPTIONS[metric_label]

        breakdown = load_regional_product_breakdown(start_str, end_str)
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

        # ── Choropleth map ───────────────────────────────────────────────
        geojson = _load_geojson()
        map_data = filtered.copy()
        map_data["geojson_name"] = map_data["region"].map(_REGION_TO_GEOJSON)
        map_data["display_name"] = map_data["region"].map(_display_label)
        vals = pd.to_numeric(map_data[metric_col], errors="coerce").fillna(0)

        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson,
            locations=map_data["geojson_name"],
            z=vals,
            featureidkey="properties.name",
            colorscale="YlOrRd",
            marker_opacity=0.75,
            marker_line_width=1,
            marker_line_color="white",
            showscale=True,
            colorbar=dict(title=metric_label, thickness=15),
            text=map_data["display_name"],
            customdata=vals,
            hovertemplate="<b>%{text}</b><br>" + metric_label + ": %{customdata:,.0f}<extra></extra>",
        ))

        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=56, lon=-96),
                zoom=2.8,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=550,
            title=dict(
                text=f"{metric_label} by Province"
                + (f" — {_display_label(selected_product)}" if selected_product != "All" else ""),
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
