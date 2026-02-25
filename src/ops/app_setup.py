# Databricks notebook source
# MAGIC %md
# MAGIC # App Setup Notebook
# MAGIC ## UC Permission Grants + Model Serving Endpoint Grant
# MAGIC
# MAGIC **Run once as part of the `actuarial_workshop_setup` job (Task 7).**
# MAGIC
# MAGIC The Lakebase PostgreSQL database setup (create DB, enable databricks_auth
# MAGIC extension, create SP role, create table, grant Lakebase privileges) runs
# MAGIC locally from `deploy.sh` using the CLI's OAuth JWT, which is the only
# MAGIC credential type accepted by Lakebase Autoscaling's `databricks_auth`
# MAGIC extension. Internal Databricks cluster tokens are not valid for direct
# MAGIC PostgreSQL authentication.
# MAGIC
# MAGIC ### What this notebook does
# MAGIC
# MAGIC | Step | Action |
# MAGIC |---|---|
# MAGIC | 1 | Grant `USE CATALOG`, `USE SCHEMA`, and `SELECT` on all workshop tables to the app SP |
# MAGIC | 2 | Grant `CAN_QUERY` on the model serving endpoint to the app SP |

# COMMAND ----------

# ─── Configuration ─────────────────────────────────────────────────────────────
dbutils.widgets.text("catalog",          "my_catalog",                           "UC Catalog")
dbutils.widgets.text("schema",           "actuarial_workshop",                   "UC Schema")
dbutils.widgets.text("app_sp_client_id", "",                                     "App SP client ID")
dbutils.widgets.text("endpoint_name",    "actuarial-workshop-sarima-forecaster", "SARIMA endpoint name")
dbutils.widgets.text("mc_endpoint_name", "actuarial-workshop-monte-carlo",       "Monte Carlo endpoint name")

CATALOG          = dbutils.widgets.get("catalog")
SCHEMA           = dbutils.widgets.get("schema")
APP_SP_CLIENT_ID = dbutils.widgets.get("app_sp_client_id")
ENDPOINT_NAME    = dbutils.widgets.get("endpoint_name")
MC_ENDPOINT_NAME = dbutils.widgets.get("mc_endpoint_name")

WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
import os, requests

# Any cluster token works for Databricks REST API calls (steps below).
# The Lakebase PostgreSQL setup ran locally in deploy.sh with a proper OAuth JWT.
TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
if not TOKEN:
    try:
        TOKEN = (
            dbutils.notebook.entry_point.getDbutils().notebook().getContext()
            .apiToken().get()
        )
    except Exception:
        TOKEN = ""

CURRENT_USER = spark.sql("SELECT current_user()").collect()[0][0]

print(f"Workspace:          {WORKSPACE_URL}")
print(f"Catalog/Schema:     {CATALOG}.{SCHEMA}")
print(f"SARIMA endpoint:    {ENDPOINT_NAME}")
print(f"MC endpoint:        {MC_ENDPOINT_NAME}")
print(f"App SP client ID:   {APP_SP_CLIENT_ID or '(not provided)'}")
print(f"Running as:         {CURRENT_USER}")
print(f"Token:              {'present' if TOKEN else 'MISSING'}")

# COMMAND ----------

# ─── 1. Grant UC permissions to app service principal ──────────────────────────
if not APP_SP_CLIENT_ID:
    print("[SKIP] No app_sp_client_id provided — skipping UC grants.")
    dbutils.notebook.exit("skipped: no app_sp_client_id")

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

print("\nUC grants complete.")

# COMMAND ----------

# ─── 2. Grant CAN_QUERY on serving endpoints to the app SP ────────────────────
# Both endpoints are created by Tasks 6a/6b before this task runs.
_endpoints_to_grant = [ep for ep in [ENDPOINT_NAME, MC_ENDPOINT_NAME] if ep]

for ep_name in _endpoints_to_grant:
    endpoint_resp = requests.get(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ep_name}",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    if endpoint_resp.status_code != 200:
        print(f"[WARN] Could not find endpoint '{ep_name}': {endpoint_resp.text[:200]}")
        continue
    endpoint_id = endpoint_resp.json()["id"]

    perms_resp = requests.patch(
        f"https://{WORKSPACE_URL}/api/2.0/permissions/serving-endpoints/{endpoint_id}",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"access_control_list": [
            {"service_principal_name": APP_SP_CLIENT_ID, "permission_level": "CAN_QUERY"},
        ]},
    )
    if perms_resp.status_code == 200:
        print(f"[OK] Granted CAN_QUERY on endpoint '{ep_name}' to SP: {APP_SP_CLIENT_ID}")
    else:
        print(f"[WARN] Failed CAN_QUERY on '{ep_name}': {perms_resp.text[:200]}")

print("\nApp setup complete.")
