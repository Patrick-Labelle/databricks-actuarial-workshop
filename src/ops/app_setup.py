# Databricks notebook source
# MAGIC %md
# MAGIC # App Setup Notebook
# MAGIC ## Lakebase Autoscaling Project Provisioning + UC Permission Grants
# MAGIC
# MAGIC **Run once as part of the `actuarial_workshop_setup` job (Task 7).**
# MAGIC
# MAGIC ### What this notebook does
# MAGIC
# MAGIC | Step | Action |
# MAGIC |---|---|
# MAGIC | 1 | Poll the Lakebase endpoint until it reaches READY state |
# MAGIC | 2 | Obtain an endpoint credential (OAuth token for Postgres auth) |
# MAGIC | 3 | Create the `{pg_database}` database inside the project |
# MAGIC | 4 | Enable the `databricks_auth` extension (required for OAuth login) |
# MAGIC | 5 | Create a Postgres role for the app SP via `databricks_create_role()` |
# MAGIC | 6 | Create the `scenario_annotations` table (idempotent) |
# MAGIC | 7 | Grant schema + table + sequence privileges to the app SP role |
# MAGIC | 8 | Grant `USE CATALOG`, `USE SCHEMA`, and `SELECT` on all workshop tables to the app SP |
# MAGIC | 9 | Grant `CAN_QUERY` on the model serving endpoint to the app SP |

# COMMAND ----------

# ─── Configuration ─────────────────────────────────────────────────────────────
dbutils.widgets.text("catalog",               "my_catalog",                                                      "UC Catalog")
dbutils.widgets.text("schema",                "actuarial_workshop",                                               "UC Schema")
dbutils.widgets.text("pg_database",           "actuarial_workshop_db",                                            "Lakebase DB name")
dbutils.widgets.text("lakebase_endpoint_path","projects/actuarial-workshop-lakebase/branches/main/endpoints/primary", "Lakebase endpoint resource path")
dbutils.widgets.text("app_sp_client_id",      "",                                                                "App SP client ID")
dbutils.widgets.text("endpoint_name",         "actuarial-workshop-sarima-forecaster",                             "Model Serving endpoint name")

CATALOG               = dbutils.widgets.get("catalog")
SCHEMA                = dbutils.widgets.get("schema")
PG_DATABASE           = dbutils.widgets.get("pg_database")
LAKEBASE_ENDPOINT_PATH = dbutils.widgets.get("lakebase_endpoint_path")
APP_SP_CLIENT_ID      = dbutils.widgets.get("app_sp_client_id")
ENDPOINT_NAME         = dbutils.widgets.get("endpoint_name")

WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .apiToken().get()
)
CURRENT_USER = spark.sql("SELECT current_user()").collect()[0][0]

print(f"Workspace:        {WORKSPACE_URL}")
print(f"Catalog/Schema:   {CATALOG}.{SCHEMA}")
print(f"Lakebase path:    {LAKEBASE_ENDPOINT_PATH} / {PG_DATABASE}")
print(f"Endpoint:         {ENDPOINT_NAME}")
print(f"App SP client ID: {APP_SP_CLIENT_ID or '(not provided)'}")
print(f"Running as:       {CURRENT_USER}")

# COMMAND ----------

%pip install psycopg2-binary --quiet

# COMMAND ----------

import requests, time
import psycopg2, psycopg2.extensions

# ─── 1. Poll endpoint until READY (up to 10 min) ──────────────────────────────
# The Lakebase project and endpoint are provisioned by bundle deploy, which runs
# before this setup job. Provisioning typically completes in 2–5 minutes.
_MAX_WAIT_S = 600
_POLL_S     = 20
_waited     = 0
HOST        = None

while _waited <= _MAX_WAIT_S:
    ep_resp = requests.get(
        f"https://{WORKSPACE_URL}/api/2.0/postgres/{LAKEBASE_ENDPOINT_PATH}",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    ep = ep_resp.json()
    # Autoscaling endpoints use current_state (IDLE / ACTIVE / STARTING / PROVISIONING)
    state = ep.get("status", {}).get("current_state", "UNKNOWN")
    # Host is nested under status.hosts.host
    HOST  = ep.get("status", {}).get("hosts", {}).get("host", "")
    print(f"Endpoint state: {state} (waited {_waited}s)")
    # IDLE = scale-to-zero paused but operational; ACTIVE = serving queries
    if state in ("IDLE", "ACTIVE"):
        break
    if state not in ("PROVISIONING", "STARTING", "PENDING", "UNKNOWN"):
        raise RuntimeError(f"Lakebase endpoint in unexpected state: {state}. Response: {ep}")
    time.sleep(_POLL_S)
    _waited += _POLL_S

assert HOST, f"Lakebase endpoint not operational after {_MAX_WAIT_S}s. Last state: {state}"
print("Host:", HOST)

# COMMAND ----------

# ─── 2. Get an endpoint credential for Postgres authentication ─────────────────
# For Lakebase Autoscaling, the standard Databricks access token is used directly
# as the Postgres password. No separate credential API is required.
# The connecting user (CURRENT_USER) is automatically a superuser on the project.
PG_TOKEN = TOKEN
print(f"Using notebook context token for Postgres authentication (user: {CURRENT_USER})")

# COMMAND ----------

# ─── 3. Create database if it doesn't exist ────────────────────────────────────
# Connect to the default 'databricks_postgres' database first, then create
# the custom workshop database.
conn = psycopg2.connect(
    host=HOST, port=5432, database="databricks_postgres",
    user=CURRENT_USER, password=PG_TOKEN, sslmode="require",
)
conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_DATABASE,))
if cur.fetchone():
    print(f"[OK] Database '{PG_DATABASE}' already exists.")
else:
    cur.execute(f'CREATE DATABASE "{PG_DATABASE}"')
    print(f"[CREATED] Database '{PG_DATABASE}' created.")

# Grant CONNECT on the custom database to the app SP so it can log in.
if APP_SP_CLIENT_ID:
    cur.execute(f'GRANT CONNECT ON DATABASE "{PG_DATABASE}" TO "{APP_SP_CLIENT_ID}"')
    conn.commit()
    print(f"[OK] Granted CONNECT on database '{PG_DATABASE}' to SP: {APP_SP_CLIENT_ID}")

conn.close()

# COMMAND ----------

# ─── 4–7. Enable databricks_auth, create SP role, table, and grants ────────────
# Connect to the custom database as the project owner (CURRENT_USER).
conn = psycopg2.connect(
    host=HOST, port=5432, database=PG_DATABASE,
    user=CURRENT_USER, password=PG_TOKEN, sslmode="require",
)
cur = conn.cursor()

# 4. Enable the databricks_auth extension — required for Databricks OAuth login.
#    This allows Postgres roles to authenticate using Databricks access tokens.
cur.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth")
conn.commit()
print("[OK] Extension 'databricks_auth' enabled.")

# 5. Create a Postgres role for the app SP (idempotent — safe to re-run).
#    databricks_create_role() creates a role that accepts Databricks tokens as
#    its password, matching the client_id as the Postgres username.
if APP_SP_CLIENT_ID:
    cur.execute(
        "SELECT databricks_create_role(%s, %s)",
        (APP_SP_CLIENT_ID, "service_principal"),
    )
    conn.commit()
    print(f"[OK] Created Postgres role for SP: {APP_SP_CLIENT_ID}")

# 6. Create the scenario_annotations table (idempotent).
#    Explicit public schema qualifier: Lakebase sets each user's search_path to
#    "$user", public — without it the table would land in the user's personal schema.
cur.execute("""
    CREATE TABLE IF NOT EXISTS public.scenario_annotations (
        id          SERIAL      PRIMARY KEY,
        segment_id  TEXT        NOT NULL,
        note        TEXT,
        analyst     TEXT,
        created_at  TIMESTAMP   DEFAULT NOW()
    )
""")
conn.commit()
print("[OK] Table 'public.scenario_annotations' ensured.")

# 7. Grant schema + table + sequence privileges to the app SP.
if APP_SP_CLIENT_ID:
    cur.execute(f'GRANT USAGE ON SCHEMA public TO "{APP_SP_CLIENT_ID}"')
    cur.execute(f'GRANT SELECT, INSERT ON TABLE public.scenario_annotations TO "{APP_SP_CLIENT_ID}"')
    # SERIAL columns create a backing sequence — INSERT alone does not include sequence access.
    cur.execute(f'GRANT USAGE ON SEQUENCE public.scenario_annotations_id_seq TO "{APP_SP_CLIENT_ID}"')
    conn.commit()
    print(f"[OK] Granted public schema + scenario_annotations + sequence to SP: {APP_SP_CLIENT_ID}")
else:
    print("[SKIP] No app_sp_client_id — skipping Lakebase PostgreSQL grants.")

conn.close()

# COMMAND ----------

# ─── 8. Grant UC permissions to app service principal ──────────────────────────
if not APP_SP_CLIENT_ID:
    print("[SKIP] No app_sp_client_id provided — skipping UC grants.")
    dbutils.notebook.exit("skipped UC grants: no app_sp_client_id")

# The app's service principal needs:
#   USE CATALOG  — to enumerate the catalog
#   USE SCHEMA   — to enumerate the schema
#   SELECT       — on each table the app queries
print(f"Granting UC permissions to SP: {APP_SP_CLIENT_ID}")

spark.sql(f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{APP_SP_CLIENT_ID}`")
print(f"[OK] GRANT USE CATALOG ON {CATALOG}")

spark.sql(f"GRANT USE SCHEMA ON SCHEMA {CATALOG}.{SCHEMA} TO `{APP_SP_CLIENT_ID}`")
print(f"[OK] GRANT USE SCHEMA ON {CATALOG}.{SCHEMA}")

tables = [
    row["tableName"]
    for row in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
]
print(f"Granting SELECT on {len(tables)} tables...")
for t in tables:
    try:
        spark.sql(f"GRANT SELECT ON TABLE {CATALOG}.{SCHEMA}.{t} TO `{APP_SP_CLIENT_ID}`")
        print(f"  [OK] {t}")
    except Exception as e:
        print(f"  [WARN] {t}: {e}")

print("\nApp setup complete.")

# COMMAND ----------

# ─── 9. Grant CAN_QUERY on the serving endpoint to the app SP ──────────────────
# The serving endpoint is created by the setup job (Task 6) before this task runs.
if not APP_SP_CLIENT_ID:
    print("[SKIP] No app_sp_client_id — skipping serving endpoint grant.")
    dbutils.notebook.exit("skipped endpoint grant: no app_sp_client_id")

endpoint_resp = requests.get(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
    headers={"Authorization": f"Bearer {TOKEN}"},
)
assert endpoint_resp.status_code == 200, \
    f"Could not find endpoint '{ENDPOINT_NAME}': {endpoint_resp.text[:200]}"
endpoint_id = endpoint_resp.json()["id"]

perms_resp = requests.patch(
    f"https://{WORKSPACE_URL}/api/2.0/permissions/serving-endpoints/{endpoint_id}",
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json={"access_control_list": [
        {"service_principal_name": APP_SP_CLIENT_ID, "permission_level": "CAN_QUERY"},
    ]},
)
assert perms_resp.status_code == 200, \
    f"Failed to grant CAN_QUERY on endpoint: {perms_resp.text[:200]}"
print(f"[OK] Granted CAN_QUERY on endpoint '{ENDPOINT_NAME}' to SP: {APP_SP_CLIENT_ID}")
