"""Unified Plotly chart theme for consistent styling across all tabs."""

# Databricks-inspired palette
COLORS = {
    "primary":    "#1B3A5C",
    "accent":     "#FF3621",
    "blue":       "#1f77b4",
    "orange":     "#FF6B35",
    "green":      "#2ca02c",
    "red":        "#d62728",
    "purple":     "#9467bd",
    "grey":       "#636363",
    "light_grey": "#E8ECF0",
}

SEQUENCE = [
    "#1B3A5C", "#FF3621", "#2ca02c", "#FF6B35",
    "#9467bd", "#1f77b4", "#d62728", "#636363",
]


def apply_default_layout(fig, title="", height=420, **kwargs):
    """Apply consistent layout defaults to any Plotly figure."""
    layout = dict(
        title=dict(text=title, font=dict(size=16, color="#1E1E1E")),
        height=height,
        margin=dict(l=0, r=20, t=50 if title else 20, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#444"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
        xaxis=dict(gridcolor="#F0F0F0", zeroline=False),
        yaxis=dict(gridcolor="#F0F0F0", zeroline=False),
    )
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig
