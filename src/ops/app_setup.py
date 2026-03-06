# Databricks notebook source
# MAGIC %md
# MAGIC # App Setup — UC Grants + Endpoint Permissions
# MAGIC
# MAGIC Grants `USE CATALOG`, `USE SCHEMA`, `SELECT` on all tables, and `CAN_QUERY`
# MAGIC on serving endpoints to the app service principal. Run as job Task 6.

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
# All endpoints are created by Task 5 (prepare_app_infrastructure) before this task runs.
# The agent endpoint (if it exists) is created by Task 7 (register_chatbot_agent).
AGENT_ENDPOINT_NAME = "actuarial-workshop-chatbot-agent"
_endpoints_to_grant = [ep for ep in [ENDPOINT_NAME, MC_ENDPOINT_NAME, AGENT_ENDPOINT_NAME] if ep]

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

# COMMAND ----------

# ─── 3. Grant CAN_RUN on Genie space to the app SP ────────────────────────────
# The Genie space is created by Module 4 (src/04_model_serving.py, Section 5).
# The app SP needs CAN_RUN to use the Genie space for natural-language queries.
dbutils.widgets.text("genie_space_id", "", "Genie Space ID")
GENIE_SPACE_ID = dbutils.widgets.get("genie_space_id")
if not GENIE_SPACE_ID:
    try:
        GENIE_SPACE_ID = dbutils.jobs.taskValues.get(
            taskKey="prepare_app_infrastructure", key="genie_space_id"
        )
        print(f"  Resolved genie_space_id from upstream task: {GENIE_SPACE_ID}")
    except Exception:
        pass

if GENIE_SPACE_ID and APP_SP_CLIENT_ID:
    genie_perms_resp = requests.patch(
        f"https://{WORKSPACE_URL}/api/2.0/permissions/genie/{GENIE_SPACE_ID}",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"access_control_list": [
            {"service_principal_name": APP_SP_CLIENT_ID, "permission_level": "CAN_RUN"},
        ]},
    )
    if genie_perms_resp.status_code == 200:
        print(f"[OK] Granted CAN_RUN on Genie space '{GENIE_SPACE_ID}' to SP: {APP_SP_CLIENT_ID}")
    else:
        print(f"[WARN] Failed CAN_RUN on Genie space: {genie_perms_resp.text[:200]}")
else:
    if not GENIE_SPACE_ID:
        print("[SKIP] No genie_space_id provided — skipping Genie space grant.")
    if not APP_SP_CLIENT_ID:
        print("[SKIP] No app_sp_client_id provided — skipping Genie space grant.")

print("\nApp setup complete.")
