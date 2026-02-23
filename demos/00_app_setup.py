# Databricks notebook source
# MAGIC %md
# MAGIC # App Setup Notebook
# MAGIC ## Lakebase Database Provisioning + UC Permission Grants
# MAGIC
# MAGIC **Run once as part of the `actuarial_workshop_setup` job (Task 7).**
# MAGIC
# MAGIC ### What this notebook does
# MAGIC
# MAGIC | Step | Action |
# MAGIC |---|---|
# MAGIC | 1 | Obtain a Lakebase-scoped credential (JWT) via the Databricks credential API |
# MAGIC | 2 | Create the `{pg_database}` PostgreSQL database inside the Lakebase instance |
# MAGIC | 3 | Create the `scenario_annotations` table (idempotent) |
# MAGIC | 4 | Grant `USE CATALOG`, `USE SCHEMA`, and `SELECT` on all workshop tables to the app SP |

# COMMAND ----------

# ─── Configuration ─────────────────────────────────────────────────────────────
dbutils.widgets.text("catalog",           "my_catalog",                  "UC Catalog")
dbutils.widgets.text("schema",            "actuarial_workshop",           "UC Schema")
dbutils.widgets.text("pg_database",       "actuarial_workshop_db",        "Lakebase DB name")
dbutils.widgets.text("lakebase_instance", "actuarial-workshop-lakebase",  "Lakebase instance name")
dbutils.widgets.text("app_sp_client_id",  "",                             "App SP client ID")

CATALOG           = dbutils.widgets.get("catalog")
SCHEMA            = dbutils.widgets.get("schema")
PG_DATABASE       = dbutils.widgets.get("pg_database")
LAKEBASE_INSTANCE = dbutils.widgets.get("lakebase_instance")
APP_SP_CLIENT_ID  = dbutils.widgets.get("app_sp_client_id")

WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .apiToken().get()
)
CURRENT_USER = spark.sql("SELECT current_user()").collect()[0][0]

print(f"Workspace:        {WORKSPACE_URL}")
print(f"Catalog/Schema:   {CATALOG}.{SCHEMA}")
print(f"Lakebase:         {LAKEBASE_INSTANCE} / {PG_DATABASE}")
print(f"App SP client ID: {APP_SP_CLIENT_ID or '(not provided)'}")
print(f"Running as:       {CURRENT_USER}")

# COMMAND ----------

%pip install psycopg2-binary --quiet

# COMMAND ----------

import requests, time
import psycopg2, psycopg2.extensions

# ─── 1. Get Lakebase instance hostname (wait up to 10 min for AVAILABLE) ──────
# The Lakebase instance is provisioned by bundle deploy, which runs before this
# setup job. Provisioning typically completes in 2–5 minutes. We poll with
# backoff so the job doesn't fail if the instance is still starting up.
_MAX_WAIT_S = 600   # 10 minutes
_POLL_S     = 30
_waited     = 0
HOST        = None

while _waited <= _MAX_WAIT_S:
    inst_resp = requests.get(
        f"https://{WORKSPACE_URL}/api/2.0/database/instances/{LAKEBASE_INSTANCE}",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    inst  = inst_resp.json()
    state = inst.get("state")
    HOST  = inst.get("read_write_dns")
    print(f"Instance state: {state} (waited {_waited}s)")
    if state == "AVAILABLE":
        break
    if state not in ("PROVISIONING", "PENDING"):
        raise RuntimeError(f"Lakebase instance in unexpected state: {state}. Response: {inst}")
    time.sleep(_POLL_S)
    _waited += _POLL_S

assert HOST, f"Lakebase instance not AVAILABLE after {_MAX_WAIT_S}s. Last state: {state}"
print("Host:", HOST)

# ─── 2. Get Lakebase-scoped credential ────────────────────────────────────────
# The /api/2.0/database/credentials endpoint issues a JWT scoped for PostgreSQL
# authentication. This is different from a general Databricks PAT or API token.
cred_resp = requests.post(
    f"https://{WORKSPACE_URL}/api/2.0/database/credentials",
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json={"request_id": f"setup-{int(time.time())}", "instance_names": [LAKEBASE_INSTANCE]},
)
print("Credential API status:", cred_resp.status_code)
assert cred_resp.status_code == 200, f"Credential API failed: {cred_resp.text[:200]}"
PG_TOKEN = cred_resp.json().get("token")
print("Lakebase credential obtained:", bool(PG_TOKEN))

# COMMAND ----------

# ─── 3. Create database if it doesn't exist ────────────────────────────────────
conn = psycopg2.connect(
    host=HOST, port=5432, database="postgres",
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
conn.close()

# COMMAND ----------

# ─── 4. Create scenario_annotations table in public schema (idempotent) ────────
# Explicit public schema qualifier is critical: Lakebase sets each user's
# search_path to "$user", public — without public. the table would be created
# in the connecting user's personal schema and invisible to the app SP.
conn = psycopg2.connect(
    host=HOST, port=5432, database=PG_DATABASE,
    user=CURRENT_USER, password=PG_TOKEN, sslmode="require",
)
cur = conn.cursor()
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

# Grant the app SP access to the public schema and the table so it can
# INSERT/SELECT using its own PostgreSQL role (its Databricks client_id).
if APP_SP_CLIENT_ID:
    cur.execute(f'GRANT USAGE ON SCHEMA public TO "{APP_SP_CLIENT_ID}"')
    cur.execute(f'GRANT SELECT, INSERT ON TABLE public.scenario_annotations TO "{APP_SP_CLIENT_ID}"')
    # SERIAL columns create a backing sequence that requires a separate USAGE grant —
    # INSERT on the table alone does not include sequence access.
    cur.execute(f'GRANT USAGE ON SEQUENCE public.scenario_annotations_id_seq TO "{APP_SP_CLIENT_ID}"')
    conn.commit()
    print(f"[OK] Granted public schema + scenario_annotations + sequence to SP: {APP_SP_CLIENT_ID}")
else:
    print("[SKIP] No app_sp_client_id — skipping Lakebase PostgreSQL grants.")

conn.close()
print("[OK] Table 'public.scenario_annotations' ensured.")

# COMMAND ----------

# ─── 5. Grant UC permissions to app service principal ──────────────────────────
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
