import streamlit as st

# ─── Page config (must be the first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="Stochastic Reserve Analytics",
    page_icon="📊",
    layout="wide",
)

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
    st.header("About")

    st.markdown("**Portfolio Data**")
    st.markdown("""
Insurance reserves across **40 segments** (product line × Canadian province):

| | |
|---|---|
| **Product lines** | Personal Auto, Commercial Auto, Homeowners, Commercial Property |
| **Provinces** | All 10 Canadian provinces |
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
st.title("📊 Stochastic Reserve Analytics")
st.caption("Powered by Databricks | Claims Forecasting · Reserve Adequacy · Scenario Stress Testing")

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Risk Assistant",
    "📈 Claims Forecast",
    "💰 Reserve Adequacy",
    "📋 Scenario Analysis",
    "⚡ On-Demand Forecast",
    "🗺️ Geography",
    "📖 Glossary",
])

from tabs import tab_chatbot, tab_claims_forecast, tab_capital, tab_catastrophe, tab_quick_forecast, tab_geography, tab_glossary

tab_chatbot.render(tab0)
tab_claims_forecast.render(tab1)
tab_capital.render(tab2)
tab_catastrophe.render(tab3)
tab_quick_forecast.render(tab4)
tab_geography.render(tab5)
tab_glossary.render(tab6)
