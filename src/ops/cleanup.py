# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Cleanup — Remove All Workshop Assets
# MAGIC
# MAGIC Drops the UC schema (CASCADE), Online Table, serving endpoints, registered models,
# MAGIC MLflow experiments, SDP pipeline, jobs, and Lakebase database.
# MAGIC Bundle-managed resources are removed separately via `databricks bundle destroy`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

# ─── Configuration ────────────────────────────────────────────────────────────
# All values default to the workshop defaults but can be overridden via widgets
# (or via job base_parameters) to clean up any target catalog/schema.
dbutils.widgets.text("catalog",           "my_catalog",                           "UC Catalog")
dbutils.widgets.text("data_schema",      "actuarial_data",                       "Data Schema")
dbutils.widgets.text("models_schema",    "actuarial_models",                     "Models Schema")
dbutils.widgets.text("app_schema",       "actuarial_app",                        "App Schema")
dbutils.widgets.text("endpoint_name",     "actuarial-workshop-frequency-forecaster", "Serving Endpoint")
dbutils.widgets.text("pg_database",       "actuarial_workshop_db",                "Lakebase DB name")
dbutils.widgets.text("lakebase_instance", "actuarial-workshop-lakebase",          "Lakebase instance name")
CATALOG           = dbutils.widgets.get("catalog")
DATA_SCHEMA       = dbutils.widgets.get("data_schema")
MODELS_SCHEMA     = dbutils.widgets.get("models_schema")
APP_SCHEMA        = dbutils.widgets.get("app_schema")
ENDPOINT_NAME     = dbutils.widgets.get("endpoint_name")
PG_DATABASE       = dbutils.widgets.get("pg_database")
LAKEBASE_INSTANCE = dbutils.widgets.get("lakebase_instance")

ALL_SCHEMAS = [APP_SCHEMA, MODELS_SCHEMA, DATA_SCHEMA]  # drop order: app first, data last

WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
import requests, mlflow, time, os

# Serverless-compatible token acquisition.
# Method 1: notebook context (works on classic clusters and scheduled job runs).
# Method 2: DATABRICKS_TOKEN env var (set automatically on serverless compute).
TOKEN = None
try:
    TOKEN = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        .apiToken().get()
    )
except Exception:
    TOKEN = None

if not TOKEN:
    TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

print(f"Workspace: {WORKSPACE_URL}")
print(f"Target catalog: {CATALOG}")
print(f"  Schemas: {', '.join(ALL_SCHEMAS)}")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Lakebase: {LAKEBASE_INSTANCE} / {PG_DATABASE}")
print(f"Token acquired: {'yes' if TOKEN else 'no'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Drop Unity Catalog Schema and All Tables

# COMMAND ----------

# List tables before dropping (for verification)
for _schema in ALL_SCHEMAS:
    try:
        tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{_schema}").collect()
        print(f"Tables in {_schema} ({len(tables)}):")
        for t in tables:
            print(f"  - {CATALOG}.{_schema}.{t['tableName']}")
    except Exception:
        print(f"Schema {CATALOG}.{_schema} does not exist (skipping)")

# COMMAND ----------

# Drop all schemas and their contained objects
for _schema in ALL_SCHEMAS:
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{_schema} CASCADE")
    print(f"Schema {CATALOG}.{_schema} dropped (CASCADE).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Remove the Online Table

# COMMAND ----------

ONLINE_TABLE_NAME = f"{CATALOG}.{MODELS_SCHEMA}.segment_features_online"

resp = requests.delete(
    f"https://{WORKSPACE_URL}/api/2.0/online-tables/{ONLINE_TABLE_NAME}",
    headers={"Authorization": f"Bearer {TOKEN}"},
)

if resp.status_code == 200:
    print(f"Online Table deleted: {ONLINE_TABLE_NAME}")
elif resp.status_code == 404:
    print(f"Online Table not found (already deleted or never created): {ONLINE_TABLE_NAME}")
else:
    print(f"Unexpected response ({resp.status_code}): {resp.text[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Delete Model Serving Endpoint

# COMMAND ----------

resp = requests.delete(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
    headers={"Authorization": f"Bearer {TOKEN}"},
)

if resp.status_code == 200:
    print(f"Serving endpoint deleted: {ENDPOINT_NAME}")
elif resp.status_code == 404:
    print(f"Endpoint not found (already deleted or never created): {ENDPOINT_NAME}")
else:
    print(f"Unexpected response ({resp.status_code}): {resp.text[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Delete Registered Model from UC Registry

# COMMAND ----------

MODEL_NAME = f"{CATALOG}.{MODELS_SCHEMA}.frequency_forecaster"

mlflow.set_registry_uri("databricks-uc")
client = mlflow.tracking.MlflowClient()

try:
    client.delete_registered_model(name=MODEL_NAME)
    print(f"UC model deleted: {MODEL_NAME}")
except Exception as e:
    print(f"Could not delete model (may not exist): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Delete MLflow Experiment

# COMMAND ----------

_current_user = spark.sql("SELECT current_user()").collect()[0][0]
_EXPERIMENT_NAMES = [
    f"/Users/{_current_user}/actuarial_workshop_frequency_forecast",
    f"/Users/{_current_user}/actuarial_workshop_bootstrap_reserves",
]

for exp_name in _EXPERIMENT_NAMES:
    try:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment:
            mlflow.delete_experiment(experiment.experiment_id)
            print(f"MLflow experiment deleted: {exp_name}")
        else:
            print(f"Experiment not found (skipping): {exp_name}")
    except Exception as e:
        print(f"Could not delete experiment {exp_name}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Delete SDP Pipeline

# COMMAND ----------

# List all pipelines to find the workshop pipeline
resp = requests.get(
    f"https://{WORKSPACE_URL}/api/2.0/pipelines",
    headers={"Authorization": f"Bearer {TOKEN}"},
    params={"max_results": 100},
)
pipelines = resp.json().get("statuses", [])
workshop_pipelines = [p for p in pipelines if "actuarial" in p.get("name", "").lower()]

print(f"Found {len(workshop_pipelines)} actuarial workshop pipeline(s):")
for p in workshop_pipelines:
    print(f"  - {p['name']} (ID: {p['pipeline_id']}, state: {p.get('state','unknown')})")

# COMMAND ----------

# Delete each workshop pipeline found
for p in workshop_pipelines:
    pid = p["pipeline_id"]
    del_resp = requests.delete(
        f"https://{WORKSPACE_URL}/api/2.0/pipelines/{pid}",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    if del_resp.status_code == 200:
        print(f"Pipeline deleted: {p['name']} ({pid})")
    else:
        print(f"Could not delete pipeline {p['name']}: {del_resp.status_code} — {del_resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Delete Databricks Job

# COMMAND ----------

# List jobs to find workshop jobs
resp = requests.get(
    f"https://{WORKSPACE_URL}/api/2.1/jobs/list",
    headers={"Authorization": f"Bearer {TOKEN}"},
    params={"name": "Actuarial Workshop"},
)
jobs = resp.json().get("jobs", [])

print(f"Found {len(jobs)} workshop job(s):")
for j in jobs:
    print(f"  - {j['settings']['name']} (ID: {j['job_id']})")

# COMMAND ----------

# Delete each workshop job
for j in jobs:
    jid = j["job_id"]
    del_resp = requests.post(
        f"https://{WORKSPACE_URL}/api/2.1/jobs/delete",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"job_id": jid},
    )
    if del_resp.status_code == 200:
        print(f"Job deleted: {j['settings']['name']} ({jid})")
    else:
        print(f"Could not delete job {j['settings']['name']}: {del_resp.status_code} — {del_resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Drop Lakebase PostgreSQL Database
# MAGIC
# MAGIC Drops the database inside the Lakebase instance. The instance itself is
# MAGIC removed by `databricks bundle destroy`.

# COMMAND ----------

%pip install psycopg2-binary --quiet

# COMMAND ----------

import psycopg2, psycopg2.extensions

# ─── Get Lakebase hostname ──────────────────────────────────────────────────
inst_resp = requests.get(
    f"https://{WORKSPACE_URL}/api/2.0/database/instances/{LAKEBASE_INSTANCE}",
    headers={"Authorization": f"Bearer {TOKEN}"},
)
inst = inst_resp.json()
print("Instance state:", inst.get("state"))
LB_HOST = inst.get("read_write_dns")

if inst.get("state") != "AVAILABLE":
    print(f"[SKIP] Lakebase instance not AVAILABLE (state={inst.get('state')}). Skipping Lakebase cleanup.")
    LB_HOST = None

# ─── Get Lakebase credential ────────────────────────────────────────────────
if LB_HOST:
    cred_resp = requests.post(
        f"https://{WORKSPACE_URL}/api/2.0/database/credentials",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"request_id": f"cleanup-{int(time.time())}", "instance_names": [LAKEBASE_INSTANCE]},
    )
    if cred_resp.status_code != 200:
        print(f"[SKIP] Could not obtain Lakebase credential: {cred_resp.text[:200]}")
        LB_HOST = None
    else:
        PG_TOKEN = cred_resp.json().get("token")
        print("Lakebase credential obtained:", bool(PG_TOKEN))

# COMMAND ----------

if LB_HOST:
    CURRENT_USER = spark.sql("SELECT current_user()").collect()[0][0]
    conn = psycopg2.connect(
        host=LB_HOST, port=5432, database="postgres",
        user=CURRENT_USER, password=PG_TOKEN, sslmode="require",
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_DATABASE,))
    if cur.fetchone():
        cur.execute(
            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
            (PG_DATABASE,),
        )
        cur.execute(f'DROP DATABASE "{PG_DATABASE}"')
        print(f"[DROPPED] Database '{PG_DATABASE}' dropped.")
    else:
        print(f"[OK] Database '{PG_DATABASE}' does not exist (already removed).")

    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cleanup Verification
# MAGIC
# MAGIC Verify all assets removed. Run `databricks bundle destroy` afterward for bundle resources.

# COMMAND ----------

print("=== Cleanup Verification ===\n")

# Check schemas
for _schema in ALL_SCHEMAS:
    try:
        tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{_schema}").count()
        print(f"[WARN] Schema {CATALOG}.{_schema} still exists with {tables} tables")
    except Exception:
        print(f"[OK] Schema {CATALOG}.{_schema} does not exist")

# Check serving endpoint
resp = requests.get(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
    headers={"Authorization": f"Bearer {TOKEN}"},
)
if resp.status_code == 404:
    print("[OK] Model serving endpoint does not exist")
else:
    print(f"[WARN] Model serving endpoint may still exist: {resp.status_code}")

# Check registered model
try:
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.tracking.MlflowClient()
    model = client.get_registered_model(MODEL_NAME)
    print(f"[WARN] UC model still exists: {MODEL_NAME}")
except Exception:
    print(f"[OK] UC model does not exist: {MODEL_NAME}")

print("\nCleanup complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC All workshop assets removed. Run `databricks bundle destroy --target <target>`
# MAGIC to clean up bundle-managed resources (App, Lakebase instance, jobs, pipeline).