# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: App Infrastructure
# MAGIC ## Serving Endpoints, Online Table, Lakebase, and Genie Space
# MAGIC
# MAGIC Creates all services the Streamlit app needs: Frequency Forecaster and Bootstrap Reserve
# MAGIC serving endpoints with AI Gateway, Online Table, Lakebase PostgreSQL, and Genie Space.

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
dbutils.widgets.text("data_schema",     "actuarial_data",                       "Data Schema")
dbutils.widgets.text("models_schema",   "actuarial_models",                     "Models Schema")
dbutils.widgets.text("endpoint_name",    "actuarial-workshop-frequency-forecaster", "Frequency Forecaster Endpoint")
dbutils.widgets.text("mc_endpoint_name", "actuarial-workshop-bootstrap-reserves",  "Bootstrap Reserves Endpoint")
dbutils.widgets.text("warehouse_id",     "",                                     "SQL Warehouse ID")
dbutils.widgets.text("pg_database",      "actuarial_workshop_db",                "Lakebase Database")
dbutils.widgets.text("app_sp_client_id", "",                                     "App SP Client ID")
dbutils.widgets.text("genie_space_id",  "",                                     "Genie Space ID")

CATALOG          = dbutils.widgets.get("catalog")
DATA_SCHEMA      = dbutils.widgets.get("data_schema")
MODELS_SCHEMA    = dbutils.widgets.get("models_schema")
ENDPOINT_NAME    = dbutils.widgets.get("endpoint_name")
MC_ENDPOINT_NAME = dbutils.widgets.get("mc_endpoint_name")
WAREHOUSE_ID     = dbutils.widgets.get("warehouse_id")
PG_DATABASE      = dbutils.widgets.get("pg_database")
APP_SP_CLIENT_ID = dbutils.widgets.get("app_sp_client_id")

SARIMA_MODEL_NAME = f"{CATALOG}.{MODELS_SCHEMA}.frequency_forecaster"
MC_MODEL_NAME     = f"{CATALOG}.{MODELS_SCHEMA}.bootstrap_reserve_simulator"
FEATURE_TABLE     = f"{CATALOG}.{DATA_SCHEMA}.features_segment_monthly"

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
print(f"Frequency model:  {SARIMA_MODEL_NAME}")
print(f"Bootstrap model:  {MC_MODEL_NAME}")
print(f"Frequency endpoint: {ENDPOINT_NAME}")
print(f"Bootstrap endpoint: {MC_ENDPOINT_NAME}")
print(f"Feature table:   {FEATURE_TABLE}")
print(f"Warehouse ID:    {WAREHOUSE_ID or '(not set)'}")
print(f"Lakebase DB:     {PG_DATABASE}")
print(f"App SP:          {APP_SP_CLIENT_ID or '(not set)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Frequency Forecaster Serving Endpoint
# MAGIC
# MAGIC Create/update endpoint + configure AI Gateway (inference tables, rate limits).

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Look up latest model versions
sarima_versions = client.search_model_versions(f"name='{SARIMA_MODEL_NAME}'")
sarima_latest_ver = max(int(v.version) for v in sarima_versions)
print(f"Frequency model:  {SARIMA_MODEL_NAME}  → version {sarima_latest_ver}")

mc_versions = client.search_model_versions(f"name='{MC_MODEL_NAME}'")
mc_latest_ver = max(int(v.version) for v in mc_versions)
print(f"Bootstrap model:  {MC_MODEL_NAME}  → version {mc_latest_ver}")

# COMMAND ----------

# ── Step 1: Create/update endpoint (served_models only) ─────────────────────
_sarima_endpoint_body = {
    "name": ENDPOINT_NAME,
    "config": {
        "served_models": [{
            "name":                   "frequency-champion",
            "model_name":             SARIMA_MODEL_NAME,
            "model_version":          str(sarima_latest_ver),
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
        json=_sarima_endpoint_body["config"],
    )
    print(f"Frequency endpoint updated: {resp.status_code}")
else:
    resp = requests.post(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints",
        headers=_HEADERS,
        json=_sarima_endpoint_body,
    )
    print(f"Frequency endpoint created: {resp.status_code}")

if resp.status_code in (200, 201):
    print(f"  URL: https://{WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations")
else:
    print(f"  Error: {resp.text}")

# ── Step 2: Configure AI Gateway (separate API call) ────────────────────────
_sarima_ai_gateway = {
    "usage_tracking_config": {"enabled": True},
    "inference_table_config": {
        "catalog_name":      CATALOG,
        "schema_name":       MODELS_SCHEMA,
        "table_name_prefix": "frequency_endpoint",
        "enabled":           True,
    },
    "rate_limits": [{"calls": 60, "renewal_period": "minute"}],
}

gw_resp = requests.put(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/ai-gateway",
    headers=_HEADERS,
    json=_sarima_ai_gateway,
)
if gw_resp.status_code == 200:
    print(f"  AI Gateway configured: inference tables + rate limits")
else:
    print(f"  AI Gateway config: {gw_resp.status_code} — {gw_resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Bootstrap Reserve Serving Endpoint
# MAGIC
# MAGIC CPU endpoint for on-demand reserve scenario analysis (NumPy).

# COMMAND ----------

# ── Step 1: Create/update endpoint ──────────────────────────────────────────
_mc_endpoint_body = {
    "name": MC_ENDPOINT_NAME,
    "config": {
        "served_models": [{
            "name":                  "bootstrap-reserve-champion",
            "model_name":            MC_MODEL_NAME,
            "model_version":         str(mc_latest_ver),
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
        json=_mc_endpoint_body["config"],
    )
    print(f"Bootstrap endpoint updated: {resp.status_code}")
else:
    resp = requests.post(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints",
        headers=_HEADERS,
        json=_mc_endpoint_body,
    )
    print(f"Bootstrap endpoint created: {resp.status_code}")

if resp.status_code in (200, 201):
    print(f"  URL: https://{WORKSPACE_URL}/serving-endpoints/{MC_ENDPOINT_NAME}/invocations")
else:
    print(f"  Error: {resp.text}")

# ── Step 2: Configure AI Gateway ────────────────────────────────────────────
_mc_ai_gateway = {
    "usage_tracking_config": {"enabled": True},
    "inference_table_config": {
        "catalog_name":      CATALOG,
        "schema_name":       MODELS_SCHEMA,
        "table_name_prefix": "bootstrap_endpoint",
        "enabled":           True,
    },
    "rate_limits": [{"calls": 20, "renewal_period": "minute"}],
}

gw_resp = requests.put(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{MC_ENDPOINT_NAME}/ai-gateway",
    headers=_HEADERS,
    json=_mc_ai_gateway,
)
if gw_resp.status_code == 200:
    print(f"  AI Gateway configured: inference tables + rate limits")
else:
    print(f"  AI Gateway config: {gw_resp.status_code} — {gw_resp.text[:200]}")

print(f"\nBoth endpoints created/updated. They take ~5 minutes to reach READY state.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Online Table — Low-Latency Feature Serving
# MAGIC
# MAGIC Syncs `features_segment_monthly` for sub-millisecond lookups at inference time.

# COMMAND ----------

ONLINE_TABLE_NAME = f"{CATALOG}.{MODELS_SCHEMA}.segment_features_online"

online_table_spec = {
    "name": ONLINE_TABLE_NAME,
    "spec": {
        "source_table_full_name": FEATURE_TABLE,
        "primary_key_columns":    [{"name": "segment_id"}, {"name": "month"}],
        "run_triggered": {
            "triggered_update_spec": {}
        },
    },
}

resp = requests.post(
    f"https://{WORKSPACE_URL}/api/2.0/online-tables",
    headers=_HEADERS,
    json=online_table_spec,
)

if resp.status_code in (200, 201):
    print(f"Online Table created: {ONLINE_TABLE_NAME}")
    print(f"Syncing from: {FEATURE_TABLE}")
    print(f"Note: initial sync takes ~2-5 minutes")
elif resp.status_code == 409:
    print(f"Online Table already exists: {ONLINE_TABLE_NAME}")
else:
    print(f"Online Table creation response ({resp.status_code}): {resp.text[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Lakebase — Managed PostgreSQL for Analyst Annotations
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
# MAGIC ## 5. Genie Space — Natural Language Data Queries
# MAGIC
# MAGIC The chatbot's `ask_genie` tool routes natural-language questions to an AI/BI
# MAGIC Genie space that understands all workshop tables. This section creates the space
# MAGIC programmatically (idempotent — skips if a Genie space ID was already provided).

# COMMAND ----------

_genie_space_id = dbutils.widgets.get("genie_space_id")

_GENIE_DESCRIPTION = (
    "Stochastic Reserve Analytics for a Canadian P&C Insurance Portfolio.\n\n"
    "IMPORTANT RULES:\n"
    "- Reserve Risk Capital = VaR at the 99.5% confidence level (var_995_M column). "
    "This is computed via Bootstrap Chain Ladder (internal model).\n"
    "- All monetary columns ending in _M are in millions of dollars.\n"
    "- IBNR = Incurred But Not Reported reserves. Best estimate is the mean of the bootstrap distribution.\n"
    "- Insurance segments use the pattern product_line_region (e.g. commercial_auto_ontario).\n"
    "- 4 product lines: Personal_Auto, Commercial_Auto, Homeowners, Commercial_Property.\n"
    "- 10 Canadian provinces. 40 total segments."
)

# Per-table descriptions (shown to Genie as context for each table)
# Tuples: (schema_var, table_name, description)
_GENIE_TABLES_DATA = [
    (DATA_SCHEMA, "features_segment_monthly",
     ["Feature-engineered table for ML models."]),
    (DATA_SCHEMA, "gold_claims_monthly",
     ["Historical monthly claims by segment. Key columns: claims_count, "
      "total_incurred, avg_severity, earned_premium."]),
    (DATA_SCHEMA, "gold_reserve_triangle",
     ["Reserve development triangle. Rows=accident periods, columns=development months. "
      "Includes incremental_paid and incremental_incurred."]),
    (MODELS_SCHEMA, "predictions_bootstrap_reserves",
     ["Bootstrap Chain Ladder reserve distribution. "
      "best_estimate_M = mean IBNR, var_995_M = Reserve Risk Capital at 99.5%."]),
    (MODELS_SCHEMA, "predictions_frequency_forecast",
     ["SARIMAX+GARCH frequency forecasts. Filter record_type='forecast' for future. "
      "Has forecast_mean, forecast_lo95, forecast_hi95, cond_volatility."]),
    (MODELS_SCHEMA, "predictions_ldf_volatility",
     ["Development factor volatility per product line. avg_ldf, std_ldf, n_factors."]),
    (MODELS_SCHEMA, "predictions_reserve_evolution",
     ["12-month reserve adequacy outlook with var_995_vs_baseline = % change."]),
    (MODELS_SCHEMA, "predictions_reserve_scenarios",
     ["Reserve deterioration scenarios: adverse_development, judicial_inflation, "
      "pandemic_tail, superimposed_inflation. var_995_vs_baseline = % impact."]),
    (MODELS_SCHEMA, "predictions_runoff_projection",
     ["Multi-period run-off surplus trajectory with ruin probability."]),
    (DATA_SCHEMA, "silver_reserves",
     ["Reserve development with SCD2 change tracking."]),
    (DATA_SCHEMA, "silver_rolling_features",
     ["Rolling statistical features (12/24-month windows)."]),
]

import uuid as _uuid

_GENIE_SAMPLE_QUESTIONS = [
    {"id": _uuid.uuid4().hex, "question": [q]}
    for q in [
        "What is the current best estimate IBNR and Reserve Risk Capital (VaR 99.5%)?",
        "Show monthly claims frequency trend for the last 12 months",
        "Which reserve scenario has the highest impact on VaR 99.5%?",
        "What are the development factor volatilities by product line?",
        "Compare Personal Auto vs Commercial Auto reserve adequacy",
    ]
]

def _build_serialized_space():
    return json.dumps({
        "version": 2,
        "data_sources": {
            "tables": [
                {"identifier": f"{CATALOG}.{schema}.{name}", "description": desc}
                for schema, name, desc in _GENIE_TABLES_DATA
            ]
        },
        "config": {"sample_questions": _GENIE_SAMPLE_QUESTIONS},
    })

# If a Genie space ID was provided, verify it still exists and update config
if _genie_space_id and WAREHOUSE_ID:
    try:
        from databricks.sdk import WorkspaceClient as _GC
        _gw = _GC()
        _gw.genie.get_space(space_id=_genie_space_id)
        print(f"Genie Space already exists: {_genie_space_id}")
        # Update description + serialized_space via raw API (SDK update_space
        # doesn't reliably persist description changes)
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
        from databricks.sdk import WorkspaceClient as _GC
        _gw = _GC()

        _space = _gw.genie.create_space(
            title="Actuarial Workshop — Reserve Assistant",
            description=_GENIE_DESCRIPTION,
            warehouse_id=WAREHOUSE_ID,
            serialized_space=_build_serialized_space(),
        )
        _genie_space_id = _space.space_id
        print(f"Genie Space created: {_genie_space_id}")
        print(f"  Tables: {len(_GENIE_TABLES_DATA)}")

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

# Pass Genie space ID to downstream tasks via task values
if _genie_space_id:
    print(f"\n  ⚠️  Genie space instructions must be added manually via the UI.")
    print(f"  Open the space in AI/BI Genie, click Edit, and paste the contents of")
    print(f"  resources/genie_space_instructions.txt into the 'General Instructions' box.")
    try:
        dbutils.jobs.taskValues.set(key="genie_space_id", value=_genie_space_id)
        print(f"  Task value set: genie_space_id={_genie_space_id}")
    except Exception:
        pass  # Not running in a job context (interactive notebook)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC All app infrastructure created: Frequency Forecaster + Bootstrap Reserve serving
# MAGIC endpoints, AI Gateway, Online Table, Lakebase database, and Genie Space.
