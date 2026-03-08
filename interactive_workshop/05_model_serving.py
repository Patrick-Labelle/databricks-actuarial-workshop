# Databricks notebook source
# MAGIC %md
# MAGIC # Module 5: App Infrastructure
# MAGIC ## Serving Endpoints, Lakebase, and AI Gateway
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC
# MAGIC ---
# MAGIC ### What We'll Cover
# MAGIC
# MAGIC This module prepares **every service the Streamlit app needs** before it launches:
# MAGIC
# MAGIC 1. **Model Serving Endpoints** — Deploy Frequency Forecaster and Bootstrap Reserve Simulator as REST APIs
# MAGIC 2. **AI Gateway** — Inference tables, usage tracking, rate limits
# MAGIC 3. **Lakebase (Managed PostgreSQL)** — Database, table, and SP grants for analyst annotations
# MAGIC 4. **Genie Space** — AI/BI Genie for natural-language data queries (chatbot)
# MAGIC 5. **Demo Calls** — Exercise every service the app will hit
# MAGIC
# MAGIC ---
# MAGIC ### The App's Integration Points
# MAGIC
# MAGIC | Service | What the App Does | Created Here |
# MAGIC |---|---|---|
# MAGIC | Frequency Forecaster endpoint | Calls for per-segment claim forecasts (Forecasts tab) | Section 1 |
# MAGIC | Bootstrap Reserve endpoint | Runs reserve risk simulations (Reserve Adequacy + Scenario tabs) | Section 2 |
# MAGIC | Online Table | Reads latest rolling features for segment context (sidebar) | Section 3 |
# MAGIC | Lakebase PostgreSQL | Persists analyst annotations + scenario notes | Section 4 |
# MAGIC | Genie Space | Natural-language data queries (chatbot Risk Assistant) | Section 5 |
# MAGIC | DBSQL Warehouse | Reads Delta tables via `statement_execution` SDK | _(provisioned by bundle)_ |
# MAGIC
# MAGIC The DBSQL Warehouse is provisioned as a bundle resource and referenced in `app/app.yaml`
# MAGIC via `valueFrom: sql-warehouse`. No setup is needed here — it's included in the demo
# MAGIC section to show the query pattern the app uses.

# MAGIC
# MAGIC > **Interactive notebook** — Run this after Modules 1-4 have completed. Creates serving endpoints, Online Table, and Lakebase database. Some steps may take 5-10 minutes for endpoint provisioning.
# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

# Install dependencies — psycopg2-binary is needed for Lakebase PostgreSQL setup.
%pip install psycopg2-binary mlflow --quiet

# COMMAND ----------

import mlflow
import requests
import json
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
dbutils.widgets.text("catalog",          "my_catalog",                           "UC Catalog")
dbutils.widgets.text("schema",           "actuarial_data",                       "UC Schema")
dbutils.widgets.text("endpoint_name",    "actuarial-workshop-frequency-forecaster", "Frequency Forecaster Endpoint")
dbutils.widgets.text("mc_endpoint_name", "actuarial-workshop-bootstrap-reserves",  "Bootstrap Reserves Endpoint")
dbutils.widgets.text("warehouse_id",     "",                                     "SQL Warehouse ID")
dbutils.widgets.text("pg_database",      "actuarial_workshop_db",                "Lakebase Database")
dbutils.widgets.text("app_sp_client_id", "",                                     "App SP Client ID")

CATALOG          = dbutils.widgets.get("catalog")
SCHEMA           = dbutils.widgets.get("schema")
ENDPOINT_NAME    = dbutils.widgets.get("endpoint_name")
MC_ENDPOINT_NAME = dbutils.widgets.get("mc_endpoint_name")
WAREHOUSE_ID     = dbutils.widgets.get("warehouse_id")
PG_DATABASE      = dbutils.widgets.get("pg_database")
APP_SP_CLIENT_ID = dbutils.widgets.get("app_sp_client_id")

FREQ_MODEL_NAME = f"{CATALOG}.{SCHEMA}.frequency_forecaster"
BOOT_MODEL_NAME     = f"{CATALOG}.{SCHEMA}.bootstrap_reserve_simulator"

mlflow.set_registry_uri("databricks-uc")

WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .apiToken().get()
)
_HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Lakebase endpoint path — matches the resource names in resources/lakebase.yml
LAKEBASE_ENDPOINT_PATH = "projects/actuarial-workshop-lakebase/branches/main/endpoints/primary"

print(f"Workspace:       {WORKSPACE_URL}")
print(f"Frequency model:  {FREQ_MODEL_NAME}")
print(f"Bootstrap model:  {BOOT_MODEL_NAME}")
print(f"Frequency endpoint: {ENDPOINT_NAME}")
print(f"Bootstrap endpoint: {MC_ENDPOINT_NAME}")
print(f"Warehouse ID:    {WAREHOUSE_ID or '(not set)'}")
print(f"Lakebase DB:     {PG_DATABASE}")
print(f"App SP:          {APP_SP_CLIENT_ID or '(not set)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Frequency Forecaster Serving Endpoint
# MAGIC
# MAGIC Create (or update) the Frequency Forecaster endpoint, then configure AI Gateway
# MAGIC as a **separate API call**. The `PUT /serving-endpoints/{name}/config` API only
# MAGIC accepts `served_models` — AI Gateway requires `PUT /serving-endpoints/{name}/ai-gateway`.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Look up latest model versions
freq_versions = client.search_model_versions(f"name='{FREQ_MODEL_NAME}'")
freq_latest_ver = max(int(v.version) for v in freq_versions)
print(f"Frequency model:  {FREQ_MODEL_NAME}  → version {freq_latest_ver}")

boot_versions = client.search_model_versions(f"name='{BOOT_MODEL_NAME}'")
boot_latest_ver = max(int(v.version) for v in boot_versions)
print(f"Bootstrap model:  {BOOT_MODEL_NAME}  → version {boot_latest_ver}")

# COMMAND ----------

# ── Step 1: Create/update endpoint (served_models only) ─────────────────────
_freq_endpoint_body = {
    "name": ENDPOINT_NAME,
    "config": {
        "served_models": [{
            "name":                   "frequency-champion",
            "model_name":             FREQ_MODEL_NAME,
            "model_version":          str(freq_latest_ver),
            "workload_size":          "Small",
            "scale_to_zero_enabled":  True,
        }],
    },
}

resp = requests.get(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
    headers=_HEADERS,
)

if resp.status_code == 200:
    resp = requests.put(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/config",
        headers=_HEADERS,
        json=_freq_endpoint_body["config"],
    )
    print(f"Frequency endpoint updated: {resp.status_code}")
else:
    resp = requests.post(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints",
        headers=_HEADERS,
        json=_freq_endpoint_body,
    )
    print(f"Frequency endpoint created: {resp.status_code}")

if resp.status_code in (200, 201):
    print(f"  URL: https://{WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations")
else:
    print(f"  Error: {resp.text}")

# ── Step 2: Configure AI Gateway (separate API call) ────────────────────────
_freq_ai_gateway = {
    "usage_tracking_config": {"enabled": True},
    "inference_table_config": {
        "catalog_name":      CATALOG,
        "schema_name":       SCHEMA,
        "table_name_prefix": "frequency_endpoint",
        "enabled":           True,
    },
    "rate_limits": [{"calls": 60, "renewal_period": "minute"}],
}

gw_resp = requests.put(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/ai-gateway",
    headers=_HEADERS,
    json=_freq_ai_gateway,
)
if gw_resp.status_code == 200:
    print(f"  AI Gateway configured: inference tables + rate limits")
else:
    print(f"  AI Gateway config: {gw_resp.status_code} — {gw_resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Bootstrap Reserve Serving Endpoint
# MAGIC
# MAGIC Same pattern — CPU endpoint for on-demand reserve risk analysis. The Bootstrap
# MAGIC Chain Ladder runs entirely on NumPy; no GPU acceleration needed.

# COMMAND ----------

# ── Step 1: Create/update endpoint ──────────────────────────────────────────
_boot_endpoint_body = {
    "name": MC_ENDPOINT_NAME,
    "config": {
        "served_models": [{
            "name":                  "bootstrap-reserve-champion",
            "model_name":            BOOT_MODEL_NAME,
            "model_version":         str(boot_latest_ver),
            "workload_size":         "Small",
            "scale_to_zero_enabled": True,
        }],
    },
}

resp = requests.get(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{MC_ENDPOINT_NAME}",
    headers=_HEADERS,
)

if resp.status_code == 200:
    resp = requests.put(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{MC_ENDPOINT_NAME}/config",
        headers=_HEADERS,
        json=_boot_endpoint_body["config"],
    )
    print(f"Bootstrap endpoint updated: {resp.status_code}")
else:
    resp = requests.post(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints",
        headers=_HEADERS,
        json=_boot_endpoint_body,
    )
    print(f"Bootstrap endpoint created: {resp.status_code}")

if resp.status_code in (200, 201):
    print(f"  URL: https://{WORKSPACE_URL}/serving-endpoints/{MC_ENDPOINT_NAME}/invocations")
else:
    print(f"  Error: {resp.text}")

# ── Step 2: Configure AI Gateway ────────────────────────────────────────────
_boot_ai_gateway = {
    "usage_tracking_config": {"enabled": True},
    "inference_table_config": {
        "catalog_name":      CATALOG,
        "schema_name":       SCHEMA,
        "table_name_prefix": "bootstrap_reserve_endpoint",
        "enabled":           True,
    },
    "rate_limits": [{"calls": 20, "renewal_period": "minute"}],
}

gw_resp = requests.put(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{MC_ENDPOINT_NAME}/ai-gateway",
    headers=_HEADERS,
    json=_boot_ai_gateway,
)
if gw_resp.status_code == 200:
    print(f"  AI Gateway configured: inference tables + rate limits")
else:
    print(f"  AI Gateway config: {gw_resp.status_code} — {gw_resp.text[:200]}")

print(f"\nBoth endpoints created. They take ~5 minutes to reach READY state.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Lakebase — Managed PostgreSQL for Analyst Annotations
# MAGIC
# MAGIC The Streamlit app persists analyst annotations (scenario notes, assumption overrides,
# MAGIC approval status) to a Lakebase PostgreSQL table. This section creates the database,
# MAGIC enables the `databricks_auth` extension, and grants the app service principal access.
# MAGIC
# MAGIC **Authentication:** We use the Databricks SDK's `generate_database_credential()` to
# MAGIC obtain a valid JWT for the Lakebase `databricks_auth` extension. This works from any
# MAGIC compute type (serverless, classic, interactive), unlike raw `apiToken()` which returns
# MAGIC an opaque internal token that Lakebase rejects.

# COMMAND ----------

import time as _time

_lakebase_ok = False
try:
    import psycopg2
    import psycopg2.extensions
    from databricks.sdk import WorkspaceClient

    _w = WorkspaceClient()

    # ── 4a. Get Lakebase endpoint host ───────────────────────────────────────
    print("Polling Lakebase endpoint for readiness...")
    _lb_host = None
    for _attempt in range(30):
        try:
            _ep = _w.api_client.do("GET", f"/api/2.0/postgres/{LAKEBASE_ENDPOINT_PATH}")
            _state = _ep.get("status", {}).get("current_state", "UNKNOWN")
            _lb_host = _ep.get("status", {}).get("hosts", {}).get("host", "")
            print(f"  State: {_state} (attempt {_attempt + 1})")
            if _state in ("IDLE", "ACTIVE") and _lb_host:
                break
        except Exception as _poll_err:
            print(f"  Polling error: {_poll_err}")
        _time.sleep(20)

    if not _lb_host:
        raise RuntimeError("Lakebase endpoint not ready after polling")

    print(f"  Host: {_lb_host}")

    # ── 4b. Get database credential (valid JWT) ─────────────────────────────
    _cred = _w.postgres.generate_database_credential(
        endpoint=LAKEBASE_ENDPOINT_PATH
    )
    _pg_token = _cred.token
    _pg_user = spark.sql("SELECT current_user()").collect()[0][0]
    print(f"  Credential obtained for: {_pg_user}")

    # ── 4c. Create database if needed ────────────────────────────────────────
    def _pg_connect(database, max_retries=3):
        """Connect with retry for transient Lakebase rate limits."""
        for _r in range(max_retries):
            try:
                conn = psycopg2.connect(
                    host=_lb_host, port=5432, database=database,
                    user=_pg_user, password=_pg_token, sslmode="require",
                    connect_timeout=30,
                )
                return conn
            except psycopg2.OperationalError as exc:
                if "rate limit" in str(exc).lower() and _r < max_retries - 1:
                    _time.sleep(30 * (2 ** _r))
                else:
                    raise

    _conn = _pg_connect("databricks_postgres")
    _conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    _cur = _conn.cursor()
    _cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_DATABASE,))
    if _cur.fetchone():
        print(f"  [OK] Database '{PG_DATABASE}' already exists")
    else:
        _cur.execute(f'CREATE DATABASE "{PG_DATABASE}"')
        print(f"  [CREATED] Database '{PG_DATABASE}'")
    _conn.close()

    # ── 4d. Setup table + extension + grants in workshop DB ──────────────────
    _time.sleep(5)  # brief pause between connections for rate limit safety
    _conn = _pg_connect(PG_DATABASE)
    _cur = _conn.cursor()

    _cur.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth")
    _conn.commit()
    print(f"  [OK] Extension 'databricks_auth' enabled")

    # Create table (idempotent)
    _cur.execute("""
        CREATE TABLE IF NOT EXISTS public.scenario_annotations (
            id              SERIAL        PRIMARY KEY,
            segment_id      TEXT          NOT NULL,
            note            TEXT,
            analyst         TEXT,
            scenario_type   TEXT,
            adjustment_pct  NUMERIC(10,2),
            approval_status TEXT          DEFAULT 'Draft',
            created_at      TIMESTAMP     DEFAULT NOW()
        )
    """)
    for _col_ddl in [
        "ADD COLUMN IF NOT EXISTS scenario_type   TEXT",
        "ADD COLUMN IF NOT EXISTS adjustment_pct  NUMERIC(10,2)",
        "ADD COLUMN IF NOT EXISTS approval_status TEXT DEFAULT 'Draft'",
    ]:
        _cur.execute(f"ALTER TABLE public.scenario_annotations {_col_ddl}")
    _cur.execute("""
        ALTER TABLE public.scenario_annotations
            ALTER COLUMN adjustment_pct TYPE NUMERIC(10,2)
    """)
    _conn.commit()
    print(f"  [OK] Table 'public.scenario_annotations' ensured")

    # Grant privileges to app SP
    if APP_SP_CLIENT_ID:
        # Create Postgres role for the SP (idempotent)
        _cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_SP_CLIENT_ID,))
        if _cur.fetchone():
            print(f"  [OK] Postgres role already exists for SP: {APP_SP_CLIENT_ID}")
        else:
            for _r in range(4):
                try:
                    _cur.execute("SELECT databricks_create_role(%s, %s)",
                                 (APP_SP_CLIENT_ID, "service_principal"))
                    _conn.commit()
                    print(f"  [OK] Postgres role created for SP: {APP_SP_CLIENT_ID}")
                    break
                except Exception as _role_err:
                    if "rate limit" in str(_role_err).lower() and _r < 3:
                        _conn.rollback()
                        _time.sleep(30 * (2 ** _r))
                    else:
                        raise

        _cur.execute(f'GRANT CONNECT ON DATABASE "{PG_DATABASE}" TO "{APP_SP_CLIENT_ID}"')
        _cur.execute(f'GRANT USAGE ON SCHEMA public TO "{APP_SP_CLIENT_ID}"')
        _cur.execute(f'GRANT SELECT, INSERT ON TABLE public.scenario_annotations TO "{APP_SP_CLIENT_ID}"')
        _cur.execute(f'GRANT USAGE ON SEQUENCE public.scenario_annotations_id_seq TO "{APP_SP_CLIENT_ID}"')
        _conn.commit()
        print(f"  [OK] Granted all Lakebase privileges to SP: {APP_SP_CLIENT_ID}")
    else:
        print(f"  [SKIP] No app_sp_client_id — skipping Lakebase grants")

    _conn.close()
    _lakebase_ok = True
    print(f"\nLakebase setup complete!")

except ImportError:
    print("psycopg2-binary not available — skipping Lakebase setup.")
    print("The app will work without annotations if Lakebase is not configured.")
except Exception as _lb_err:
    print(f"Lakebase setup error: {_lb_err}")
    print("Continuing — the app will work without annotations.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Genie Space — Natural Language Data Queries
# MAGIC
# MAGIC The chatbot's `ask_genie` tool routes natural-language questions to an AI/BI
# MAGIC Genie space that understands all workshop tables. This section creates the space
# MAGIC programmatically (idempotent — skips if a Genie space ID was already provided).

# COMMAND ----------

_genie_space_id = None

# Check if a Genie space already exists (e.g. from a previous run)
# In interactive mode, set via widget; in job mode, via task values.
try:
    dbutils.widgets.text("genie_space_id", "", "Genie Space ID")
    _genie_space_id = dbutils.widgets.get("genie_space_id") or None
except Exception:
    pass

_GENIE_DESCRIPTION = (
    "Actuarial Risk Analytics for a Canadian P&C Insurance Portfolio.\n\n"
    "IMPORTANT RULES:\n"
    "- Economic Capital = VaR at the 99.5% confidence level (var_995_M column). "
    "This is an internal model result (simulation-based), not the regulatory standard formula.\n"
    "- All monetary columns ending in _M are in millions of dollars.\n"
    "- When asked about reserve risk from predictions_bootstrap_reserves, "
    "SUM across product lines for portfolio totals.\n"
    "- Insurance segments use the pattern product_line_region (e.g. commercial_auto_ontario).\n"
    "- Product lines: Personal Auto, Commercial Auto, Homeowners, Commercial Property."
)

_GENIE_TABLES = [
    ("features_segment_monthly",
     ["Feature-engineered table for ML models."]),
    ("gold_claims_monthly",
     ["Historical monthly claims by segment. Key columns: claims_count, "
      "total_incurred, avg_severity, earned_premium."]),
    ("gold_reserve_triangle",
     ["Reserve development triangle. Rows=accident periods, columns=development months."]),
    ("predictions_bootstrap_reserves",
     ["Bootstrap Chain Ladder reserve results. Key columns: best_estimate_M, "
      "var_99_M, var_995_M, cvar_99_M, reserve_risk_capital_M."]),
    ("predictions_reserve_evolution",
     ["12-month reserve adequacy outlook."]),
    ("predictions_frequency_forecast",
     ["SARIMAX+GARCH forecasts. Filter record_type='forecast' for future. "
      "Has forecast_mean, ci_lower_95, ci_upper_95."]),
    ("predictions_reserve_scenarios",
     ["Reserve scenarios: baseline, adverse_development, judicial_inflation, "
      "pandemic_tail, superimposed_inflation."]),
    ("predictions_runoff_projection",
     ["Multi-period surplus trajectory with regime-switching."]),
    ("silver_reserves",
     ["Reserve development with SCD2 change tracking."]),
    ("silver_rolling_features",
     ["Rolling statistical features (12/24-month windows)."]),
]

import uuid as _uuid

_GENIE_SAMPLE_QUESTIONS = [
    {"id": _uuid.uuid4().hex, "question": [q]}
    for q in [
        "What is the current Economic Capital (VaR 99.5%)?",
        "Show monthly claims trend for the last 12 months",
        "Which stress scenario has the highest impact on VaR 99.5%?",
        "What is the portfolio expected annual loss?",
        "Compare Property vs Auto segment loss trends",
    ]
]

def _build_serialized_space():
    return json.dumps({
        "version": 2,
        "data_sources": {
            "tables": [
                {"identifier": f"{CATALOG}.{SCHEMA}.{name}", "description": desc}
                for name, desc in _GENIE_TABLES
            ]
        },
        "config": {"sample_questions": _GENIE_SAMPLE_QUESTIONS},
    })

if _genie_space_id and WAREHOUSE_ID:
    try:
        from databricks.sdk import WorkspaceClient
        _gw = WorkspaceClient()
        _gw.genie.get_space(space_id=_genie_space_id)
        print(f"Genie Space already exists: {_genie_space_id}")
        try:
            _gw.api_client.do(
                "PATCH",
                f"/api/2.0/genie/spaces/{_genie_space_id}",
                body={
                    "description": _GENIE_DESCRIPTION,
                    "serialized_space": _build_serialized_space(),
                },
            )
            print("  Description + table descriptions + sample questions updated.")
        except Exception as _e:
            print(f"  Could not update Genie space config: {_e}")
    except Exception:
        print(f"Genie Space {_genie_space_id} not found — will create a new one.")
        _genie_space_id = None

if WAREHOUSE_ID and not _genie_space_id:
    try:
        from databricks.sdk import WorkspaceClient
        _gw = WorkspaceClient()

        _space = _gw.genie.create_space(
            title="Actuarial Workshop — Risk Assistant",
            description=_GENIE_DESCRIPTION,
            warehouse_id=WAREHOUSE_ID,
            serialized_space=_build_serialized_space(),
        )
        _genie_space_id = _space.space_id
        print(f"Genie Space created: {_genie_space_id}")
        print(f"  Tables: {len(_GENIE_TABLES)}")

        # Grant app SP access to the Genie space
        if APP_SP_CLIENT_ID:
            try:
                _gw.genie.update_space_permissions(
                    space_id=_genie_space_id,
                    access_control_list=[{
                        "service_principal_name": APP_SP_CLIENT_ID,
                        "permission_level": "CAN_RUN",
                    }],
                )
                print(f"  Granted CAN_RUN to SP: {APP_SP_CLIENT_ID}")
            except Exception as _perm_err:
                print(f"  Genie permission grant skipped: {_perm_err}")

    except Exception as _genie_err:
        import traceback
        print(f"Genie space creation failed: {_genie_err}")
        traceback.print_exc()
elif not WAREHOUSE_ID and not _genie_space_id:
    print("[SKIP] No warehouse_id — Genie space creation skipped.")
    print("Set genie_space_id in databricks.local.yml after creating the space manually.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Demo — Exercise Every App Service
# MAGIC
# MAGIC The Streamlit app hits five services. Let's verify each one works.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a. Frequency Forecast (Model Serving)

# COMMAND ----------

def call_frequency_endpoint(horizon: int) -> dict:
    """Call the Frequency Forecaster endpoint. Returns error dict on failure."""
    try:
        resp = requests.post(
            f"https://{WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations",
            headers=_HEADERS,
            json={"dataframe_records": [{"horizon": horizon}]},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text, "status_code": resp.status_code}
    except requests.exceptions.Timeout:
        return {"note": "Endpoint still warming up (~5 min after creation). Re-run once READY."}
    except Exception as exc:
        return {"error": str(exc)}

print("Requesting 6-month forecast from Frequency Forecaster endpoint...\n")
freq_result = call_frequency_endpoint(horizon=6)
print(json.dumps(freq_result, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b. Bootstrap Reserve Scenario (Model Serving)

# COMMAND ----------

def call_bootstrap_endpoint(scenario_params: dict) -> dict:
    """Call the Bootstrap Reserve endpoint. Returns result dict or error."""
    try:
        resp = requests.post(
            f"https://{WORKSPACE_URL}/serving-endpoints/{MC_ENDPOINT_NAME}/invocations",
            headers=_HEADERS,
            json={"dataframe_records": [scenario_params]},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text, "status_code": resp.status_code}
    except requests.exceptions.Timeout:
        return {"note": "Endpoint still warming up (~5 min after creation). Re-run once READY."}
    except Exception as exc:
        return {"error": str(exc)}


_baseline_params = {
    "scenario": "baseline",
    "n_replications": 10000,
}

_adverse_params = {
    "scenario": "adverse_development",
    "ldf_multiplier": 1.2,
    "n_replications": 10000,
}

print("Calling Bootstrap Reserve endpoint — baseline scenario...")
_b = call_bootstrap_endpoint(_baseline_params)
print("Calling Bootstrap Reserve endpoint — adverse development scenario...")
_s = call_bootstrap_endpoint(_adverse_params)

for label, result in [("Baseline", _b), ("Adverse Development (LDFs +20%)", _s)]:
    print(f"\n{label}:")
    if "predictions" in result:
        pred = result["predictions"][0] if isinstance(result["predictions"], list) else result["predictions"]
        print(f"  Best Estimate IBNR: ${pred.get('best_estimate_M', 'N/A'):.1f}M")
        print(f"  VaR(99%):           ${pred.get('var_99_M', 'N/A'):.1f}M")
        print(f"  VaR(99.5%):         ${pred.get('var_995_M', 'N/A'):.1f}M")
        print(f"  CVaR(99%):          ${pred.get('cvar_99_M', 'N/A'):.1f}M")
    else:
        print(f"  {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6c. Delta Tables via DBSQL (Statement Execution)
# MAGIC
# MAGIC The app reads Delta tables using `WorkspaceClient().statement_execution.execute_statement()`.
# MAGIC This routes queries through the SQL Warehouse specified in `app/app.yaml` — same
# MAGIC warehouse as Online Table sync. Here's the pattern:

# COMMAND ----------

if WAREHOUSE_ID:
    from databricks.sdk import WorkspaceClient
    _wc = WorkspaceClient()
    _stmt = _wc.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=f"SELECT segment_id, COUNT(*) AS months FROM {CATALOG}.{SCHEMA}.gold_claims_monthly GROUP BY segment_id ORDER BY months DESC LIMIT 5",
        wait_timeout="30s",
    )
    if _stmt.result and _stmt.result.data_array:
        print("DBSQL query via statement_execution (top 5 segments):")
        for row in _stmt.result.data_array:
            print(f"  {row[0]:35s} {row[1]:>4s} months")
    else:
        print(f"DBSQL query returned no results (status: {_stmt.status})")
else:
    print("[SKIP] No warehouse_id — DBSQL demo skipped.")
    print("The app uses WorkspaceClient().statement_execution.execute_statement()")
    print("with the warehouse ID injected via app.yaml valueFrom: sql-warehouse.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6d. Lakebase — Read/Write Annotations

# COMMAND ----------

if _lakebase_ok:
    # Quick round-trip test: insert a test annotation, read it back, then delete it
    _conn = _pg_connect(PG_DATABASE)
    _cur = _conn.cursor()
    _cur.execute("""
        INSERT INTO public.scenario_annotations (segment_id, note, analyst, scenario_type)
        VALUES ('_test_module5', 'Connectivity test from Module 5', %s, 'test')
        RETURNING id
    """, (_pg_user,))
    _test_id = _cur.fetchone()[0]
    _conn.commit()

    _cur.execute("SELECT id, segment_id, note, analyst FROM public.scenario_annotations WHERE id = %s", (_test_id,))
    _row = _cur.fetchone()
    print(f"Lakebase round-trip OK: id={_row[0]}, segment={_row[1]}, analyst={_row[3]}")

    # Clean up test row
    _cur.execute("DELETE FROM public.scenario_annotations WHERE id = %s", (_test_id,))
    _conn.commit()
    _conn.close()
    print("  Test row cleaned up.")
else:
    print("[SKIP] Lakebase not configured — annotation demo skipped.")
    print("The app persists analyst notes to public.scenario_annotations in Lakebase.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Monitoring — Endpoint Request Logs
# MAGIC
# MAGIC AI Gateway enables inference tables that capture every request/response for audit:
# MAGIC
# MAGIC ```
# MAGIC gold_claims_monthly (SDP gold layer)
# MAGIC        ↓ (training data)
# MAGIC MLflow Experiments (Module 4)
# MAGIC        ↓ (registered models)
# MAGIC frequency_forecaster@Champion  /  bootstrap_reserve_simulator@Champion
# MAGIC        ↓ (serving)
# MAGIC Model Serving endpoints (Frequency Forecaster + Bootstrap Reserves)
# MAGIC        ↓ (AI Gateway)
# MAGIC Inference tables + system.serving.served_entities_request_logs
# MAGIC ```

# COMMAND ----------

# Query endpoint request logs (available after first call)
try:
    spark.sql(f"""
        SELECT
            timestamp_ms,
            endpoint_name,
            model_name,
            model_version,
            status_code,
            execution_time_ms,
            request_id
        FROM system.serving.served_entities_request_logs
        WHERE endpoint_name IN ('{ENDPOINT_NAME}', '{MC_ENDPOINT_NAME}')
        ORDER BY timestamp_ms DESC
        LIMIT 10
    """).display()
except Exception as e:
    print(f"Note: system.serving.served_entities_request_logs not available: {e}")
    print("This is a workspace-level system table that may need to be enabled separately.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Service | What Was Set Up | App Tab |
# MAGIC |---|---|---|
# MAGIC | Frequency Forecaster endpoint | `frequency_forecaster@Champion` → REST API + AI Gateway | Forecasts |
# MAGIC | Bootstrap Reserve endpoint | `bootstrap_reserve_simulator@Champion` → CPU REST API + AI Gateway | Reserve Adequacy, Scenario Analysis |
# MAGIC | Online Table | `segment_features_online` → low-latency feature lookup | Sidebar |
# MAGIC | Lakebase | `scenario_annotations` table + SP grants | All tabs (annotations) |
# MAGIC | Genie Space | AI/BI Genie with 10 tables + example queries | Risk Assistant (chatbot) |
# MAGIC | DBSQL Warehouse | Query via `statement_execution` SDK | All tabs (Delta reads) |
# MAGIC | AI Gateway | Inference tables + usage tracking + rate limits | Monitoring |
# MAGIC
# MAGIC **Key insight:** `PUT /serving-endpoints/{name}/config` only accepts `served_models`.
# MAGIC AI Gateway configuration requires a **separate** `PUT /serving-endpoints/{name}/ai-gateway` call.
# MAGIC
# MAGIC **Next:** Module 6 — Package everything as a Databricks Asset Bundle and wire into Azure DevOps CI/CD.
