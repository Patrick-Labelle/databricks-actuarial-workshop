# Databricks notebook source
# MAGIC %md
# MAGIC # Bonus: Databricks Apps + Lakebase
# MAGIC ## Building Governed Internal Actuary Tools
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC
# MAGIC ---
# MAGIC ### What We've Built — The Serving Gap
# MAGIC
# MAGIC By the end of Module 6, you have:
# MAGIC - ✅ Reliable data pipelines (DLT + Medallion)
# MAGIC - ✅ Scaled statistical models (SARIMA, GARCH, Monte Carlo)
# MAGIC - ✅ Governed models in UC Registry with REST endpoints
# MAGIC - ✅ CI/CD with DABs + Azure DevOps
# MAGIC
# MAGIC **What's missing**: A way for **non-technical stakeholders** (pricing committee,
# MAGIC reserving actuaries, risk officers) to interact with these models and results
# MAGIC *without* touching notebooks or REST APIs.
# MAGIC
# MAGIC **Databricks Apps** fills this gap: governed web applications running on the same
# MAGIC platform as your models, with the same Unity Catalog permissions.
# MAGIC
# MAGIC ---
# MAGIC ### Architecture: Apps + Model Serving + Feature Store
# MAGIC
# MAGIC ```
# MAGIC  Actuary's Browser
# MAGIC        ↓
# MAGIC  Databricks App (Streamlit)   ← SSO via Databricks identity
# MAGIC        ↓                            ↓
# MAGIC  UC Tables (gold_segment_*)   Model Serving endpoint (SARIMA)
# MAGIC  monte_carlo_results          segment_features_online (Online Table)
# MAGIC        ↓
# MAGIC  Lakebase (Postgres)          ← Scenario annotations, user comments
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: The Streamlit App
# MAGIC
# MAGIC Below is the **complete Streamlit application code** that would be deployed as a Databricks App.
# MAGIC In this notebook we display and explain the code; the deployment section shows how to package it.
# MAGIC
# MAGIC The app provides:
# MAGIC 1. **Forecast Dashboard**: Select a segment, view SARIMA forecasts with confidence intervals
# MAGIC 2. **Monte Carlo Risk Summary**: VaR, CVaR, loss distribution from Module 4's simulation
# MAGIC 3. **On-Demand Forecast**: Call the Module 5 Model Serving endpoint in real time
# MAGIC 4. **Scenario Annotation**: Save comments/assumptions to Lakebase (Postgres)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `app.py` — Full Streamlit Application Source

# COMMAND ----------

APP_CODE = '''
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import psycopg2
from databricks import sdk

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Actuarial Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

# ─── Authentication ───────────────────────────────────────────────────────────
# Databricks Apps automatically injects DATABRICKS_HOST and uses the
# app's service principal identity — no manual token management needed.
w = sdk.WorkspaceClient()
TOKEN = w.config.token
WORKSPACE_HOST = os.environ.get("DATABRICKS_HOST", "")
ENDPOINT_NAME  = os.environ.get("ENDPOINT_NAME", "actuarial-workshop-sarima-forecaster")

# ─── Spark connection via Databricks Connect ─────────────────────────────────
from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.serverless(True).getOrCreate()

# All workspace-specific values come from env vars injected by the bundle app
# resource (resources/app.yml → config.env). No hardcoded values here.
CATALOG = os.environ.get("CATALOG", "my_catalog")
SCHEMA  = os.environ.get("SCHEMA",  "actuarial_workshop")

# ─── Helper functions ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_segments():
    return [row["segment_id"] for row in
            spark.sql(f"SELECT DISTINCT segment_id FROM {CATALOG}.{SCHEMA}.claims_time_series ORDER BY 1").collect()]

@st.cache_data(ttl=300)
def load_forecasts(segment_id: str):
    return spark.sql(f"""
        SELECT month, record_type, claims_count, forecast_mean, forecast_lo95, forecast_hi95
        FROM {CATALOG}.{SCHEMA}.sarima_forecasts
        WHERE segment_id = \'{segment_id}\'
        ORDER BY month
    """).toPandas()

@st.cache_data(ttl=600)
def load_monte_carlo_summary():
    return spark.sql(f"""
        SELECT
            AVG(mean_loss_M)  AS expected_loss,
            AVG(var_99_M)     AS var_99,
            AVG(var_995_M)    AS var_995,
            AVG(cvar_99_M)    AS cvar_99,
            MAX(max_loss_M)   AS max_loss
        FROM {CATALOG}.{SCHEMA}.monte_carlo_results
    """).toPandas().iloc[0]

def call_serving_endpoint(horizon: int) -> pd.DataFrame:
    """Call Model Serving endpoint for on-demand forecast."""
    resp = requests.post(
        f"{WORKSPACE_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"dataframe_records": [{"horizon": horizon}]},
        timeout=15,
    )
    if resp.status_code == 200:
        return pd.DataFrame(resp.json().get("predictions", []))
    return pd.DataFrame()

# ─── Lakebase: Scenario annotations ──────────────────────────────────────────

def get_lakebase_conn():
    """Connect to Lakebase Postgres for scenario annotations."""
    return psycopg2.connect(
        host     = os.environ.get("LAKEBASE_HOST", ""),
        port     = int(os.environ.get("LAKEBASE_PORT", "5432")),
        database = os.environ.get("LAKEBASE_DB", "actuarial_workshop"),
        user     = os.environ.get("LAKEBASE_USER", ""),
        password = os.environ.get("LAKEBASE_PASSWORD", ""),
    )

def save_scenario_annotation(segment: str, note: str, analyst: str):
    try:
        conn = get_lakebase_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO scenario_annotations (segment_id, note, analyst, created_at) "
            "VALUES (%s, %s, %s, NOW())",
            (segment, note, analyst)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"Could not save annotation: {e}")
        return False

# ─── App Layout ───────────────────────────────────────────────────────────────

st.title("📊 Actuarial Risk Dashboard")
st.caption("Powered by Databricks | SARIMA Forecasting + Monte Carlo Portfolio Risk")

tab1, tab2, tab3 = st.tabs(["🔮 Forecasts", "📉 Portfolio Risk", "⚡ On-Demand Forecast"])

# ── Tab 1: Segment Forecasts ──────────────────────────────────────────────────
with tab1:
    st.subheader("SARIMA Claims Forecasts by Segment")

    segments = load_segments()
    selected = st.selectbox("Select segment:", segments, index=0)

    if selected:
        df = load_forecasts(selected)

        if not df.empty:
            actuals  = df[df["record_type"] == "actual"]
            forecasts = df[df["record_type"] == "forecast"]

            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actuals["month"], y=actuals["claims_count"],
                                     mode="lines+markers", name="Actual", line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=forecasts["month"], y=forecasts["forecast_mean"],
                                     mode="lines+markers", name="Forecast", line=dict(color="#FF3419", dash="dash")))
            fig.add_trace(go.Scatter(
                x=pd.concat([forecasts["month"], forecasts["month"][::-1]]),
                y=pd.concat([forecasts["forecast_hi95"], forecasts["forecast_lo95"][::-1]]),
                fill="toself", fillcolor="rgba(255,52,25,0.15)", line=dict(color="rgba(255,0,0,0)"),
                name="95% CI",
            ))
            fig.update_layout(title=f"{selected} — SARIMA Forecast (12 months)",
                              xaxis_title="Month", yaxis_title="Claims Count", height=400)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            mape = df[df["record_type"]=="forecast"]["mape"].mean()
            col1.metric("Avg MAPE", f"{mape:.1f}%")
            col2.metric("12-Month Forecast", f"{int(forecasts['forecast_mean'].mean()):,}")
            col3.metric("Forecast Range", f"±{int((forecasts['forecast_hi95']-forecasts['forecast_lo95']).mean()/2):,}")

    # Scenario annotation
    with st.expander("📝 Add Scenario Note"):
        analyst = st.text_input("Analyst name:")
        note    = st.text_area("Assumptions / adjustments:")
        if st.button("Save Note"):
            if save_scenario_annotation(selected, note, analyst):
                st.success("Note saved to Lakebase")

# ── Tab 2: Portfolio Risk ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Monte Carlo Portfolio Risk Summary")
    st.caption("100,000 correlated lognormal scenarios | Property + Auto + Liability")

    summary = load_monte_carlo_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Annual Loss", f"${summary['expected_loss']:.1f}M")
    col2.metric("VaR (99%)", f"${summary['var_99']:.1f}M")
    col3.metric("VaR (99.5%)", f"${summary['var_995']:.1f}M", help="Solvency II SCR level")
    col4.metric("CVaR (99%)", f"${summary['cvar_99']:.1f}M", help="Expected Shortfall")

    st.info("ℹ️ VaR(99.5%) is the Solvency Capital Requirement (SCR) level under Solvency II standard formula.")

# ── Tab 3: On-Demand Forecast ─────────────────────────────────────────────────
with tab3:
    st.subheader("On-Demand Forecast via Model Serving")
    st.caption("Calls the deployed SARIMA REST endpoint in real time")

    horizon = st.slider("Forecast horizon (months):", 1, 24, 6)

    if st.button("Generate Forecast"):
        with st.spinner("Calling Model Serving endpoint..."):
            result_df = call_serving_endpoint(horizon)
            if not result_df.empty:
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    result_df.to_csv(index=False),
                    file_name=f"forecast_{horizon}m.csv",
                )
            else:
                st.warning("Endpoint not available — start the Model Serving endpoint from Module 5")
'''

print("App code ready for deployment (see deployment section below)")
print(f"Lines: {len(APP_CODE.splitlines())}")
print(f"Size: {len(APP_CODE)} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B: App Configuration (`app.yaml`)
# MAGIC
# MAGIC The `app.yaml` file is what Databricks Apps uses to configure the deployment.

# COMMAND ----------

APP_YAML = """
# app.yaml — Databricks App configuration
command:
  - "streamlit"
  - "run"
  - "app.py"
  - "--server.port=8080"
  - "--server.address=0.0.0.0"

env:
  # These are injected automatically by Databricks Apps
  # - DATABRICKS_HOST
  # - DATABRICKS_TOKEN (service principal)

  # Lakebase connection (set these as App secrets in the Databricks console)
  - name: LAKEBASE_HOST
    valueFrom:
      secret: lakebase-host
  - name: LAKEBASE_PASSWORD
    valueFrom:
      secret: lakebase-password
  - name: LAKEBASE_USER
    valueFrom:
      secret: lakebase-user
  - name: LAKEBASE_DB
    valueFrom:
      secret: lakebase-db
"""

print(APP_YAML)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part C: Lakebase — Managed Postgres for Scenario Annotations
# MAGIC
# MAGIC The app uses **Lakebase** (managed Postgres) to store user-facing operational data
# MAGIC (scenario annotations, analyst comments, assumption overrides) that needs transactional writes.
# MAGIC
# MAGIC Why not just write to Delta tables?
# MAGIC - Analysts need sub-second write latency from the web app
# MAGIC - Comments need to be updated/deleted easily (transactional)
# MAGIC - Delta Lake is optimized for analytical reads, not point-writes
# MAGIC
# MAGIC Lakebase syncs its tables back to Delta Lake, so the annotations are also available
# MAGIC in the Lakehouse for analytical queries.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Lakebase Schema
# MAGIC
# MAGIC Once a Lakebase instance is provisioned (via the Databricks console or API),
# MAGIC connect with standard psycopg2 and create the schema.

# COMMAND ----------

LAKEBASE_SCHEMA_SQL = """
-- Create tables in Lakebase Postgres for scenario annotations
-- Connect with: psycopg2.connect(host=..., port=5432, database='actuarial_workshop', ...)

CREATE TABLE IF NOT EXISTS scenario_annotations (
    id            SERIAL PRIMARY KEY,
    segment_id    VARCHAR(100) NOT NULL,
    note          TEXT,
    analyst       VARCHAR(100),
    created_at    TIMESTAMP DEFAULT NOW(),
    tags          JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_segment_id ON scenario_annotations (segment_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON scenario_annotations (created_at);

-- Assumption overrides: actuaries can apply manual adjustments to model outputs
CREATE TABLE IF NOT EXISTS model_assumption_overrides (
    id              SERIAL PRIMARY KEY,
    segment_id      VARCHAR(100) NOT NULL,
    override_type   VARCHAR(50),   -- 'trend_adjustment', 'volatility_cap', 'seasonal_factor'
    override_value  DECIMAL(10,4),
    rationale       TEXT,
    effective_from  DATE,
    effective_to    DATE,
    approved_by     VARCHAR(100),
    created_at      TIMESTAMP DEFAULT NOW()
);
"""

print("Lakebase schema DDL:")
print(LAKEBASE_SCHEMA_SQL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part D: Deploy the App
# MAGIC
# MAGIC Deployment follows these steps. In production, add this to the DABs bundle (Module 6).

# COMMAND ----------

DEPLOY_COMMANDS = """
# 1. Create app directory
mkdir -p ~/actuarial-workshop-app
cd ~/actuarial-workshop-app

# 2. Write app.py and app.yaml (as shown above)

# 3. Create requirements.txt
cat > requirements.txt << EOF
streamlit>=1.35
plotly>=5.0
databricks-sdk>=0.28
databricks-connect==17.0.1
psycopg2-binary>=2.9
pandas>=2.0
numpy>=1.24
requests>=2.31
EOF

# 4. Deploy via Databricks CLI
databricks apps create actuarial-risk-dashboard \\
  --description "SARIMA forecasts + Monte Carlo risk dashboard for actuaries" \\
  --profile e2-demo-west

# 5. Deploy code
databricks apps deploy actuarial-risk-dashboard \\
  --source-code-path ~/actuarial-workshop-app \\
  --profile e2-demo-west

# 6. View app URL
databricks apps get actuarial-risk-dashboard --profile e2-demo-west
"""

print(deploy_commands_display := DEPLOY_COMMANDS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part E: Including Apps in a DABs Bundle
# MAGIC
# MAGIC Add the App to your `databricks.yml` so it's deployed alongside Jobs and Model Serving:
# MAGIC
# MAGIC ```yaml
# MAGIC # resources/apps.yml
# MAGIC resources:
# MAGIC   apps:
# MAGIC     actuarial_risk_dashboard:
# MAGIC       name: "actuarial-risk-dashboard"
# MAGIC       description: "SARIMA forecasts + Monte Carlo risk dashboard"
# MAGIC       source_code_path: ./app/
# MAGIC       config:
# MAGIC         env:
# MAGIC           - name: CATALOG
# MAGIC             value: "${var.catalog}"
# MAGIC           - name: SCHEMA
# MAGIC             value: "${var.schema}"
# MAGIC           - name: ENDPOINT_NAME
# MAGIC             value: "${var.endpoint_name}"
# MAGIC       resources:
# MAGIC         # Grant the App's service principal read access to UC tables
# MAGIC         - name: serving-endpoint
# MAGIC           serving_endpoint:
# MAGIC             name: "${var.endpoint_name}"
# MAGIC             permission: CAN_QUERY
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: The Complete Workshop Stack
# MAGIC
# MAGIC You've now built a **production-grade actuarial modeling platform** on Databricks:
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │  DATABRICKS APPS (Streamlit)                                │
# MAGIC │  Internal actuary dashboard — governed, SSO, serverless     │
# MAGIC └────────────────────┬────────────────────────────────────────┘
# MAGIC                      │
# MAGIC          ┌───────────┴───────────┐
# MAGIC          │                       │
# MAGIC ┌────────▼───────┐    ┌─────────▼────────┐
# MAGIC │ MODEL SERVING  │    │  LAKEBASE         │
# MAGIC │ SARIMA REST    │    │  Postgres — annot.│
# MAGIC │ @Champion alias│    └──────────────────┘
# MAGIC └────────┬───────┘
# MAGIC          │
# MAGIC ┌────────▼──────────────────────────────────────────────┐
# MAGIC │  UNITY CATALOG                                         │
# MAGIC │  Feature Store + Model Registry + Tables + Lineage     │
# MAGIC └────────┬──────────────────────────────────────────────┘
# MAGIC          │
# MAGIC ┌────────▼──────────────────────────────────────────────┐
# MAGIC │  GOLD TABLES              SARIMA/GARCH forecasts       │
# MAGIC │  Segment monthly stats    Monte Carlo results          │
# MAGIC └────────┬──────────────────────────────────────────────┘
# MAGIC          │
# MAGIC ┌────────▼──────────────────────────────────────────────┐
# MAGIC │  DLT PIPELINE  (Bronze → Silver → Gold)               │
# MAGIC │  CDC Apply Changes · Expectations · Serverless         │
# MAGIC └───────────────────────────────────────────────────────┘
# MAGIC          │
# MAGIC ┌────────▼──────────────────────────────────────────────┐
# MAGIC │  CI/CD: Databricks Asset Bundles + Azure DevOps        │
# MAGIC │  Git-native · Dev/Staging/Prod · Zero stored secrets   │
# MAGIC └───────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Everything is governed, reproducible, and production-ready** —
# MAGIC meeting actuarial standards for audit trail, data lineage, and model governance.