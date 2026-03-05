import streamlit as st

# ─── Page config (must be the first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="Insurance Portfolio Risk Intelligence",
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
Insurance claims across **40 segments** (product line × Canadian province):

| | |
|---|---|
| **Product lines** | Personal Auto, Commercial Auto, Homeowners, Commercial Property |
| **Provinces** | All 10 Canadian provinces |
| **Period** | Jan 2019 – Dec 2025 (84 months) |
| **Macro data** | StatCan unemployment, housing price index, housing starts |
""")

    st.divider()
    st.markdown("**What the models do**")
    st.markdown("""
| Model | Business purpose |
|---|---|
| **Claims Forecasting** | Projects monthly claim volumes per segment — used in the Claims Forecast and Quick Forecast tabs |
| **Volatility Model** | GARCH(1,1) on SARIMA residuals — captures time-varying forecast uncertainty, feeds into Monte Carlo capital calculations |
| **Monte Carlo Simulation** | Runs millions of loss scenarios to compute capital requirements (Expected Loss, SCR, Tail Risk) — powers the Capital Requirements and Stress Testing tabs |
""")

    st.divider()
    st.markdown("**How it's built**")
    st.markdown("""
```
Raw CDC events
  → Bronze (Spark Declarative Pipelines)
  → Silver (SCD Type 2 policies)
  → Gold (monthly segment stats)
  → Feature Store (point-in-time joins)
  → SARIMA+GARCH per segment
  → Monte Carlo portfolio simulation
  → UC Model Registry (@Champion)
     ├── SARIMA endpoint (Forecasts tab)
     └── Monte Carlo endpoint (Scenario tab)
  → This App
```
All assets are version-controlled and reproducible.
Model artifacts are logged to MLflow and promoted via
the Unity Catalog Model Registry.
""")

    st.divider()
    st.caption("Powered by Databricks Apps + Streamlit")

# ─── App Header + Tabs ───────────────────────────────────────────────────────
st.title("📊 Insurance Portfolio Risk Intelligence")
st.caption("Powered by Databricks | Claims Forecasting · Capital Planning · Catastrophe Analysis")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Risk Assistant",
    "📈 Claims Forecast",
    "💰 Capital Requirements",
    "⚡ Quick Forecast",
    "🎲 Stress Testing",
    "🌪️ Catastrophe & Reserves",
])

from tabs import tab_chatbot, tab_claims_forecast, tab_capital, tab_quick_forecast, tab_stress_testing, tab_catastrophe

tab_chatbot.render(tab0)
tab_claims_forecast.render(tab1)
tab_capital.render(tab2)
tab_quick_forecast.render(tab3)
tab_stress_testing.render(tab4)
tab_catastrophe.render(tab5)
