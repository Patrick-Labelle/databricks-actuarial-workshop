import streamlit as st

# ─── Page config (must be the first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="Stochastic Reserve Analytics",
    page_icon="📊",
    layout="wide",
)

# ─── Global CSS for polished metric cards and layout ─────────────────────────
st.markdown("""
<style>
/* Metric card polish */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}
[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: #1B3A5C !important;
}
/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 20px;
    font-weight: 500;
}
/* Divider subtlety */
hr {
    border-color: #E8ECF0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Eagerly probe auth so the warning appears early ─────────────────────────
from auth import get_workspace_client, get_auth_init_error

get_workspace_client()
if get_auth_init_error() is not None:
    st.warning(
        f"Databricks SDK could not initialise: {get_auth_init_error()}. "
        "Data features will be unavailable until the issue is resolved."
    )

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    import base64, pathlib
    _logo_path = pathlib.Path(__file__).parent / "logo.svg"
    try:
        _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
        _logo_html = (
            f'<img src="data:image/svg+xml;base64,{_logo_b64}" '
            'style="width:120px;margin-bottom:8px" alt="Logo">'
        )
    except FileNotFoundError:
        _logo_html = '<span style="font-size:2.2em">📊</span>'
    st.markdown(
        '<div style="text-align:center;padding:8px 0 16px 0">'
        f'{_logo_html}<br>'
        '<span style="font-size:1.1em;font-weight:700;color:#1B3A5C">'
        'Stochastic Reserve Analytics</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Portfolio Data**")
    st.markdown("""
Insurance reserves across **52 segments** (product line x Canadian province/territory):

| | |
|---|---|
| **Product lines** | Personal Auto, Commercial Auto, Homeowners, Commercial Property |
| **Regions** | All 10 provinces + 3 territories |
| **Period** | Jan 2019 – Dec 2025 (84 months) |
| **Triangle** | Loss development with line-specific tail lengths (36–60 months) |
| **Macro data** | StatCan unemployment, housing price index, housing starts |
""")

    st.divider()
    st.markdown("**What the models do**")
    st.markdown("""
| Model | Business purpose |
|---|---|
| **Frequency Forecasting** | SARIMAX projects future accident period claim counts — the exposure base for new reserves |
| **Chain Ladder** | Deterministic best estimate IBNR from weighted link ratios and Mack variance |
| **Bootstrap Chain Ladder** | 40K+ bootstrap replications produce the reserve risk distribution — VaR 99.5%, CVaR, reserve risk capital |
| **GARCH(1,1)** | Time-varying frequency volatility from SARIMAX residuals |
""")

    st.divider()
    st.markdown("**How it's built**")
    st.markdown("""
```
Raw CDC events
  → Bronze (Spark Declarative Pipelines)
  → Silver (SCD Type 2 reserves)
  → Gold (reserve triangle + monthly claims)
  → Feature Store (point-in-time joins)
  → SARIMAX frequency forecasting
  → Chain Ladder + Bootstrap (Ray)
  → UC Model Registry (@Champion)
     ├── Frequency Forecaster endpoint
     └── Bootstrap Reserve endpoint
  → This App
```
All assets are version-controlled and reproducible.
Model artifacts are logged to MLflow and promoted via
the Unity Catalog Model Registry.
""")

    st.divider()
    st.caption("Powered by Databricks Apps + Streamlit")

# ─── App Header + Tabs ───────────────────────────────────────────────────────
st.markdown(
    '<h1 style="display:flex;align-items:center;gap:12px">'
    f'<img src="data:image/svg+xml;base64,{_logo_b64}" style="height:1.1em" alt="">'
    'Stochastic Reserve Analytics</h1>',
    unsafe_allow_html=True,
)
st.caption("Powered by Databricks | Claims Forecasting · Reserve Adequacy · Scenario Stress Testing")

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "💬 Risk Assistant",
    "📈 Claims Forecast",
    "💰 Reserve Adequacy",
    "📋 Scenario Analysis",
    "⚡ On-Demand Forecast",
    "🗺️ Geography",
    "🏗️ Architecture",
    "📖 Glossary",
])

from tabs import tab_chatbot, tab_claims_forecast, tab_capital, tab_catastrophe, tab_quick_forecast, tab_geography, tab_architecture, tab_glossary

tab_chatbot.render(tab0)
tab_claims_forecast.render(tab1)
tab_capital.render(tab2)
tab_catastrophe.render(tab3)
tab_quick_forecast.render(tab4)
tab_geography.render(tab5)
tab_architecture.render(tab6)
tab_glossary.render(tab7)
