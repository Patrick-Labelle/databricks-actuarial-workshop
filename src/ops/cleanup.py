# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Cleanup Notebook
# MAGIC ## Remove All Assets Created by the Actuarial Workshop Demo
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC
# MAGIC ---
# MAGIC > **⚠️ WARNING**: This notebook permanently deletes workshop assets.
# MAGIC > Review each section carefully before running. Run cells individually, not all at once.
# MAGIC >
# MAGIC > **Do NOT run this notebook during the workshop.**
# MAGIC > It is intended for post-workshop cleanup only.
# MAGIC
# MAGIC ### Assets This Notebook Will Remove
# MAGIC
# MAGIC | Asset | Type | Created By |
# MAGIC |---|---|---|
# MAGIC | `{catalog}.{schema}` | UC Schema + all tables | Modules 1–5 |
# MAGIC | `segment_features_online` | Online Table | Module 3 |
# MAGIC | `actuarial-workshop-sarima-forecaster` | Model Serving endpoint | Module 5 |
# MAGIC | `actuarial_workshop_sarima_claims_forecaster` | MLflow experiment | Module 5 |
# MAGIC | `{catalog}.{schema}.sarima_claims_forecaster` | UC Model | Module 5 |
# MAGIC | `Actuarial Workshop — DLT Pipeline` | DLT Pipeline | Module 1 |
# MAGIC | `Actuarial Workshop — Orchestration Demo` | Databricks Job | Module 1 |
# MAGIC | `actuarial_workshop_db` | Lakebase PostgreSQL database | Bonus (App setup) |
# MAGIC
# MAGIC > **Note:** The Databricks App, Lakebase instance, bundle jobs, and DLT pipeline managed
# MAGIC > by the Asset Bundle are removed by `databricks bundle destroy --target <target>`.
# MAGIC > This notebook removes assets created *inside* the workspace by the workshop notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

# ─── Configuration ────────────────────────────────────────────────────────────
# All values default to the workshop defaults but can be overridden via widgets
# (or via job base_parameters) to clean up any target catalog/schema.
dbutils.widgets.text("catalog",           "my_catalog",                           "UC Catalog")
dbutils.widgets.text("schema",            "actuarial_workshop",                   "UC Schema")
dbutils.widgets.text("endpoint_name",     "actuarial-workshop-sarima-forecaster", "Serving Endpoint")
dbutils.widgets.text("pg_database",       "actuarial_workshop_db",                "Lakebase DB name")
dbutils.widgets.text("lakebase_instance", "actuarial-workshop-lakebase",          "Lakebase instance name")
CATALOG           = dbutils.widgets.get("catalog")
SCHEMA            = dbutils.widgets.get("schema")
ENDPOINT_NAME     = dbutils.widgets.get("endpoint_name")
PG_DATABASE       = dbutils.widgets.get("pg_database")
LAKEBASE_INSTANCE = dbutils.widgets.get("lakebase_instance")

WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
import requests, json, mlflow, time, os

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
print(f"Target catalog/schema: {CATALOG}.{SCHEMA}")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Lakebase: {LAKEBASE_INSTANCE} / {PG_DATABASE}")
print(f"Token acquired: {'yes' if TOKEN else 'no'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Drop Unity Catalog Schema and All Tables
# MAGIC
# MAGIC This drops **all** tables, views, functions, and volumes in `{catalog}.{schema}`,
# MAGIC then drops the schema itself.

# COMMAND ----------

# List tables before dropping (for verification)
tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
print(f"Tables to be dropped ({len(tables)}):")
for t in tables:
    print(f"  - {CATALOG}.{SCHEMA}.{t['tableName']}")

# COMMAND ----------

# Drop the schema and all contained objects
spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
print(f"Schema {CATALOG}.{SCHEMA} dropped (CASCADE — all tables, views, functions, volumes removed).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Remove the Online Table
# MAGIC
# MAGIC The Online Table (`segment_features_online`) was created in Module 3 via the REST API.
# MAGIC It must also be removed via REST API.

# COMMAND ----------

ONLINE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.segment_features_online"

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
# MAGIC
# MAGIC The endpoint `actuarial-workshop-sarima-forecaster` was created in Module 5.

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
# MAGIC
# MAGIC Removes the model `{catalog}.{schema}.sarima_claims_forecaster`
# MAGIC and all its versions from Unity Catalog.

# COMMAND ----------

MODEL_NAME = f"{CATALOG}.{SCHEMA}.sarima_claims_forecaster"

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
# MAGIC
# MAGIC Removes the MLflow experiment and all its run data.

# COMMAND ----------

_current_user = spark.sql("SELECT current_user()").collect()[0][0]
# Flat paths used since run 4 fix (nested paths fail on fresh workspaces)
EXPERIMENT_NAME   = f"/Users/{_current_user}/actuarial_workshop_sarima_claims_forecaster"
EXPERIMENT_NAME_4 = f"/Users/{_current_user}/actuarial_workshop_claims_sarima"
# Also handle old nested paths (in case workspace was seeded before the fix)
EXPERIMENT_NAME_OLD   = f"/Users/{_current_user}/actuarial-workshop/champion-model"
EXPERIMENT_NAME_4_OLD = f"/Users/{_current_user}/actuarial-workshop/sarima-per-segment"

for exp_name in [EXPERIMENT_NAME, EXPERIMENT_NAME_OLD, EXPERIMENT_NAME_4, EXPERIMENT_NAME_4_OLD]:
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
# MAGIC ## 6. Delete DLT Pipeline
# MAGIC
# MAGIC Find and delete the DLT pipeline named `Actuarial Workshop — DLT Pipeline`.

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
# MAGIC
# MAGIC Find and delete jobs named `Actuarial Workshop — Orchestration Demo`.

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
# MAGIC Drops the `actuarial_workshop_db` database (and all tables in it) from the
# MAGIC Lakebase instance. This removes `public.scenario_annotations` and all analyst annotations.
# MAGIC
# MAGIC > The Lakebase **instance** itself is removed by `databricks bundle destroy`.
# MAGIC > This step only drops the database inside the instance.

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
# MAGIC Verify all assets have been removed.
# MAGIC
# MAGIC > After running this notebook, run `databricks bundle destroy --target <target>` to
# MAGIC > remove the Databricks App, Lakebase instance, jobs, and DLT pipeline managed by the bundle.

# COMMAND ----------

print("=== Cleanup Verification ===\n")

# Check schema
try:
    tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").count()
    print(f"[WARN] Schema {CATALOG}.{SCHEMA} still exists with {tables} tables")
except Exception:
    print(f"[OK] Schema {CATALOG}.{SCHEMA} does not exist")

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
# MAGIC | Asset | Action |
# MAGIC |---|---|
# MAGIC | `{catalog}.{schema}` schema | Dropped (CASCADE) |
# MAGIC | Online Table `segment_features_online` | Deleted via REST API |
# MAGIC | Model Serving endpoint | Deleted via REST API |
# MAGIC | UC registered model | Deleted via MLflow client |
# MAGIC | MLflow experiments | Deleted |
# MAGIC | DLT pipeline | Deleted via REST API |
# MAGIC | Databricks Job | Deleted via REST API |
# MAGIC | Lakebase database `actuarial_workshop_db` | Dropped (PostgreSQL DROP DATABASE) |
# MAGIC | Databricks App + Lakebase instance + bundle jobs | Run `databricks bundle destroy` |