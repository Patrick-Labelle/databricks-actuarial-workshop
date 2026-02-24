# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Foundations
# MAGIC ## Delta Live Tables Pipeline + Databricks Workflows
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC *Audience: Actuaries, Data Scientists, Financial Analysts*
# MAGIC
# MAGIC ---
# MAGIC ### The Actuarial Data Problem
# MAGIC
# MAGIC Before we can fit SARIMA or GARCH models, run Monte Carlo simulations, or build features for ML,
# MAGIC we need **reliable, versioned, auditable data** flowing through a pipeline. The classic actuarial
# MAGIC challenges:
# MAGIC
# MAGIC | Challenge | Without Databricks | With DLT + Medallion |
# MAGIC |---|---|---|
# MAGIC | CDC / policy updates | Hand-rolled MERGE SQL | Declarative Apply Changes |
# MAGIC | Data quality enforcement | Ad-hoc checks, late discovery | Expectations: fail/warn/quarantine |
# MAGIC | Reprocessing history | Manual, error-prone | Re-run from Bronze, full lineage |
# MAGIC | Pipeline orchestration | Cron + shell scripts | Databricks Jobs DAG |
# MAGIC | Streaming + batch unification | Two separate codebases | One DLT pipeline |
# MAGIC
# MAGIC ---
# MAGIC ### Architecture for Today
# MAGIC
# MAGIC ```
# MAGIC Raw CDC feed (policy changes)
# MAGIC        ↓
# MAGIC   BRONZE  — append-only raw records (full history preserved)
# MAGIC        ↓  Apply Changes (SCD Type 2)
# MAGIC   SILVER  — current + history of each policy, DLT expectations enforced
# MAGIC        ↓  Aggregation
# MAGIC   GOLD    — segment-level monthly loss statistics (ready for SARIMA/GARCH)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: DLT Pipeline Definition
# MAGIC
# MAGIC **This notebook is the DLT pipeline source.** When attached to a Delta Live Tables pipeline,
# MAGIC Databricks automatically:
# MAGIC - Determines execution order from `@dlt.table` dependencies
# MAGIC - Manages incremental processing (only new/changed data)
# MAGIC - Tracks data quality metrics (expectations)
# MAGIC - Handles SCD logic via `dlt.apply_changes()`
# MAGIC
# MAGIC > **How to run**: Create a DLT pipeline in the Workflows UI, point it at this notebook,
# MAGIC > and click Start. The pipeline handles the rest.

# COMMAND ----------

try:
    import dlt
    IN_DLT = True
except Exception:
    # Running as a regular notebook (not inside a DLT pipeline context)
    # Data generation and Job creation sections still work normally
    # Catching Exception (not just ImportError) because on some runtimes
    # importing dlt outside a DLT pipeline raises a Py4J/Java error
    dlt = None
    IN_DLT = False

import pyspark.sql.functions as F
from pyspark.sql.types import *

# ─── Configuration ────────────────────────────────────────────────────────────
# When running as a DLT pipeline, values come from the pipeline's configuration
# block (set in databricks.yml → resources/pipeline.yml).
# When running as a job task or interactively, values come from widgets
# (base_parameters in resources/jobs.yml, or the defaults below).
if IN_DLT:
    CATALOG            = spark.conf.get("catalog",             "my_catalog")
    SCHEMA             = spark.conf.get("schema",              "actuarial_workshop")
    NOTIFICATION_EMAIL = spark.conf.get("notification_email", "")
else:
    dbutils.widgets.text("catalog",            "my_catalog",         "UC Catalog")
    dbutils.widgets.text("schema",             "actuarial_workshop", "UC Schema")
    dbutils.widgets.text("notification_email", "",                   "Notification Email")
    CATALOG            = dbutils.widgets.get("catalog")
    SCHEMA             = dbutils.widgets.get("schema")
    NOTIFICATION_EMAIL = dbutils.widgets.get("notification_email")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 — Generate Raw CDC Data (simulates a Bronze landing zone)
# MAGIC
# MAGIC In production, Bronze is fed by Auto Loader reading from S3/ADLS.
# MAGIC For this workshop we generate a synthetic CDC stream of insurance policy records:
# MAGIC - Policy events: NEW, ENDORSEMENT (mid-term change), CANCELLATION, REINSTATEMENT
# MAGIC - Each event captures: policy_id, effective_date, premium, coverage_amount, loss_ratio

# COMMAND ----------

# ── Helper: generate synthetic policy CDC data ────────────────────────────────
# Run this cell OUTSIDE the DLT pipeline to seed the landing zone with data

def generate_bronze_cdc_data(spark, catalog: str, schema: str, n_policies: int = 5000):
    """
    Generate synthetic insurance policy CDC records and write to a Delta table
    that acts as the Bronze source for the DLT pipeline.
    """
    import numpy as np
    import pandas as pd
    from datetime import date, timedelta

    np.random.seed(2024)

    PRODUCT_LINES  = ["Personal_Auto", "Commercial_Auto", "Homeowners", "Commercial_Property"]
    REGIONS        = ["Ontario", "Quebec", "British_Columbia", "Alberta", "Atlantic"]
    EVENT_TYPES    = ["NEW", "ENDORSEMENT", "CANCELLATION", "REINSTATEMENT"]
    EVENT_WEIGHTS  = [0.50,   0.30,          0.12,            0.08]

    rows = []
    event_seq = 1

    for policy_id in range(1001, 1001 + n_policies):
        product = np.random.choice(PRODUCT_LINES)
        region  = np.random.choice(REGIONS)
        base_premium = np.random.uniform(800, 4500)

        # Each policy has 1-5 CDC events spanning 3 years
        n_events = np.random.randint(1, 6)
        start_date = date(2021, 1, 1) + timedelta(days=np.random.randint(0, 730))

        current_premium = base_premium
        for ev_idx in range(n_events):
            event_type = (
                "NEW" if ev_idx == 0
                else np.random.choice(EVENT_TYPES[1:], p=[w/0.50 for w in EVENT_WEIGHTS[1:]])
            )
            effective_date = start_date + timedelta(days=ev_idx * np.random.randint(30, 180))

            if event_type == "ENDORSEMENT":
                current_premium *= np.random.uniform(0.90, 1.15)
            elif event_type == "CANCELLATION":
                current_premium = 0.0

            rows.append({
                "event_id":         event_seq,
                "policy_id":        policy_id,
                "event_type":       event_type,
                "effective_date":   effective_date.isoformat(),
                "product_line":     product,
                "region":           region,
                "annual_premium":   round(current_premium, 2),
                "coverage_amount":  round(current_premium * np.random.uniform(15, 50), 2),
                "loss_ratio":       round(np.random.uniform(0.45, 0.95), 4),
                "ingested_at":      pd.Timestamp.now().isoformat(),
                "_is_deleted":      (event_type == "CANCELLATION"),
            })
            event_seq += 1

    pdf = pd.DataFrame(rows)

    schema_str = StructType([
        StructField("event_id",        IntegerType(), False),
        StructField("policy_id",       IntegerType(), False),
        StructField("event_type",      StringType(),  False),
        StructField("effective_date",  StringType(),  False),
        StructField("product_line",    StringType(),  False),
        StructField("region",          StringType(),  False),
        StructField("annual_premium",  DoubleType(),  True),
        StructField("coverage_amount", DoubleType(),  True),
        StructField("loss_ratio",      DoubleType(),  True),
        StructField("ingested_at",     StringType(),  False),
        StructField("_is_deleted",     BooleanType(), False),
    ])

    sdf = spark.createDataFrame(pdf, schema=schema_str)

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    (sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.{schema}.policy_cdc_raw"))

    count = sdf.count()
    print(f"Generated {count:,} CDC events for {n_policies} policies → {catalog}.{schema}.policy_cdc_raw")
    return sdf

# Run OUTSIDE the DLT context to seed data
if not IN_DLT:  # Use the IN_DLT flag set at top of notebook (avoids inaccessible DLT spark conf)
    raw_df = generate_bronze_cdc_data(spark, CATALOG, SCHEMA)
    display(raw_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 — Bronze: Raw Ingestion Layer
# MAGIC
# MAGIC The Bronze table is **append-only** and preserves the complete CDC history.
# MAGIC No transformations — just ingest exactly what arrived, exactly when it arrived.
# MAGIC
# MAGIC In production, replace `read_table` with Auto Loader (`cloud_files`):
# MAGIC ```python
# MAGIC spark.readStream.format("cloudFiles").option("cloudFiles.format", "parquet").load(landing_path)
# MAGIC ```

# COMMAND ----------

if IN_DLT:
    @dlt.table(
        name="bronze_policy_cdc",
        comment="Raw CDC events from policy administration system. Append-only, schema-on-read, full history.",
        table_properties={
            "quality":      "bronze",
            "delta.enableChangeDataFeed": "true",
        },
    )
    @dlt.expect_or_drop("valid_premium",    "annual_premium >= 0")
    @dlt.expect_or_drop("valid_loss_ratio", "loss_ratio BETWEEN 0.0 AND 2.5")
    @dlt.expect_or_drop("valid_product",    "product_line IN ('Personal_Auto','Commercial_Auto','Homeowners','Commercial_Property')")
    @dlt.expect("valid_coverage",           "coverage_amount > annual_premium")
    def bronze_policy_cdc():
        """
        Bronze: Read raw policy CDC records from the landing zone.
        Append-only — preserves complete history including corrections and deletions.
        """
        return (
            spark.readStream
                 .format("delta")
                 .table(f"{CATALOG}.{SCHEMA}.policy_cdc_raw")
        )
else:
    print("ℹ️  DLT not active — Bronze/Silver/Gold tables are created by the DLT pipeline.")
    print("   Create a DLT pipeline pointing at this notebook to materialize these tables.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 — Silver: Cleansed Policies (SCD Type 2 via Apply Changes)
# MAGIC
# MAGIC `dlt.apply_changes()` replaces hand-rolled MERGE SQL with a declarative pattern:
# MAGIC - **SCD Type 2**: Every version of a policy is retained with `__START_AT` and `__END_AT` timestamps
# MAGIC - **Sequence**: `effective_date` determines ordering (handles late-arriving events)
# MAGIC - **Soft deletes**: `_is_deleted = true` marks cancelled policies without physical deletion
# MAGIC - **Data quality**: Expectations are applied at Bronze (Step 2) — architecturally correct since
# MAGIC   Bronze should capture every raw record and quality enforcement happens before CDC merge.
# MAGIC
# MAGIC This is the layer actuaries query for historical exposure analysis.

# COMMAND ----------

# ── Silver target table + Apply Changes ───────────────────────────────────────
# Use create_streaming_table() to declare the SCD2 target, then populate it
# with apply_changes(). This is the correct pattern for DLT runtimes ≥ 2024-Q3:
# the old @dlt.table + apply_changes() on the same name produces a duplicate-
# query error in newer runtimes.
#
# Data quality expectations are enforced at the Bronze layer (above), which is
# architecturally correct: Bronze should capture every record; Silver applies
# CDC merge logic on top of validated Bronze records.
if IN_DLT:
    dlt.create_streaming_table(
        name="silver_policies",
        comment="Current and historical policy states with full SCD Type 2 history.",
        table_properties={"quality": "silver"},
    )

    # Apply Changes: CDC → SCD Type 2
    dlt.apply_changes(
        target       = "silver_policies",
        source       = "bronze_policy_cdc",
        keys         = ["policy_id"],
        sequence_by  = F.col("effective_date"),
        apply_as_deletes = F.col("_is_deleted") == True,
        stored_as_scd_type = 2,
        column_list  = ["policy_id", "event_type", "effective_date", "product_line",
                        "region", "annual_premium", "coverage_amount", "loss_ratio"],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 — Gold: Segment Aggregates (Materialized View)
# MAGIC
# MAGIC Gold is where actuarial consumers (pricing analysts, reserving actuaries, ML pipelines) read from.
# MAGIC We aggregate monthly claims statistics by segment — exactly the data that feeds Module 4's SARIMA models.
# MAGIC
# MAGIC DLT **materialized views** keep this table incrementally up-to-date as Silver changes.
# MAGIC No scheduled jobs needed — DLT handles the refresh automatically.

# COMMAND ----------

if IN_DLT:
    @dlt.table(
        name="gold_segment_monthly_stats",
        comment="Monthly aggregated loss statistics by product line × region. Primary input for SARIMA/GARCH modeling.",
        table_properties={"quality": "gold"},
    )
    def gold_segment_monthly_stats():
        """
        Gold: Monthly segment statistics aggregated from Silver.
        - segment_id: product_line × region key (matches Module 4 naming)
        - avg_loss_ratio: simple average (use earned premium weighted in production)
        - exposure_count: policy-months in force
        """
        silver = dlt.read("silver_policies")

        return (
            silver
            .withColumn("month", F.date_trunc("month", F.col("effective_date").cast("date")))
            .withColumn("segment_id", F.concat_ws("__", "product_line", "region"))
            .groupBy("segment_id", "product_line", "region", "month")
            .agg(
                F.count("*").alias("exposure_count"),
                F.avg("loss_ratio").alias("avg_loss_ratio"),
                F.sum("annual_premium").alias("total_premium"),
                F.avg("annual_premium").alias("avg_premium"),
                F.countDistinct("policy_id").alias("unique_policies"),
            )
            .withColumn("claims_estimate", (F.col("avg_loss_ratio") * F.col("total_premium") / 12).cast("int"))
            .orderBy("segment_id", "month")
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part B: Orchestrate with Databricks Workflows
# MAGIC
# MAGIC Now that our pipeline is defined, we wire it into a **multi-task Job** that:
# MAGIC 1. Runs the DLT pipeline (Bronze → Silver → Gold)
# MAGIC 2. Runs Module 4's SARIMA modeling notebook downstream (depends on Gold being ready)
# MAGIC 3. Sends an email on failure
# MAGIC
# MAGIC The Job is defined as code via Databricks Asset Bundles (see Module 6).
# MAGIC Here we show the equivalent via the Jobs API for illustration.

# COMMAND ----------

import requests, json

# ── Get workspace URL and token ───────────────────────────────────────────────
WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .apiToken().get()
)

def get_pipeline_id(pipeline_name: str) -> str | None:
    """Find an existing DLT pipeline by name."""
    resp = requests.get(
        f"https://{WORKSPACE_URL}/api/2.0/pipelines",
        headers={"Authorization": f"Bearer {TOKEN}"},
        params={"filter": f"name LIKE '{pipeline_name}'"},
    )
    pipelines = resp.json().get("statuses", [])
    return pipelines[0]["pipeline_id"] if pipelines else None


def get_job_id(job_name: str) -> int | None:
    """Find an existing job by exact name. Returns the job_id or None."""
    resp = requests.get(
        f"https://{WORKSPACE_URL}/api/2.1/jobs/list",
        headers={"Authorization": f"Bearer {TOKEN}"},
        params={"name": job_name, "limit": 5},
    )
    jobs = resp.json().get("jobs", [])
    matches = [j for j in jobs if j.get("settings", {}).get("name") == job_name]
    return matches[0]["job_id"] if matches else None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Multi-Task Job
# MAGIC
# MAGIC This code creates a Job with two tasks:
# MAGIC 1. `refresh_medallion_pipeline` — triggers the DLT pipeline
# MAGIC 2. `fit_sarima_models` — runs Module 4 notebook, depends on DLT completing successfully

# COMMAND ----------

PIPELINE_NAME = "actuarial-workshop-medallion"
pipeline_id = get_pipeline_id(PIPELINE_NAME)

# Derive the current user's workspace home for notebook path resolution
_current_user = spark.sql("SELECT current_user()").collect()[0][0]

JOB_NAME = "Actuarial Workshop — Monthly Model Refresh"

if pipeline_id:
    existing_job_id = get_job_id(JOB_NAME)
    if existing_job_id:
        print(f"Job '{JOB_NAME}' already exists (job_id={existing_job_id}) — skipping creation.")
        print(f"https://{WORKSPACE_URL}/jobs/{existing_job_id}")
    else:
        job_config = {
            "name": JOB_NAME,
            "schedule": {
                "quartz_cron_expression": "0 0 3 1 * ?",   # 3am on the 1st of each month
                "timezone_id": "America/Toronto",
                "pause_status": "PAUSED",                   # Unpaused in production
            },
            "email_notifications": {
                "on_failure": [NOTIFICATION_EMAIL] if NOTIFICATION_EMAIL else [],
            },
            "tasks": [
                {
                    "task_key":    "refresh_medallion_pipeline",
                    "description": "Refresh Bronze→Silver→Gold via DLT Apply Changes",
                    "pipeline_task": {
                        "pipeline_id": pipeline_id,
                        "full_refresh": False,
                    },
                },
                {
                    "task_key":    "fit_sarima_models",
                    "description": "Fit SARIMA/GARCH per segment; log to MLflow",
                    "depends_on":  [{"task_key": "refresh_medallion_pipeline"}],
                    "notebook_task": {
                        "notebook_path": f"/Users/{_current_user}/actuarial-workshop/04_classical_stats_at_scale",
                        "base_parameters": {"catalog": CATALOG, "schema": SCHEMA},
                    },
                    "environment_key": "ml_env",
                },
            ],
            "environments": [{
                "environment_key": "ml_env",
                "spec": {
                    "client": "4",
                    "dependencies": ["statsmodels>=0.14", "arch>=7.0", "mlflow>=2.14"],
                },
            }],
        }

        resp = requests.post(
            f"https://{WORKSPACE_URL}/api/2.1/jobs/create",
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
            json=job_config,
        )
        if resp.status_code == 200:
            job_id = resp.json()["job_id"]
            print(f"Job created: {job_id}")
            print(f"https://{WORKSPACE_URL}/jobs/{job_id}")
        else:
            print(f"Could not create job: {resp.text}")
            print("Tip: Create the DLT pipeline first via Workflows UI, then re-run this cell.")
else:
    print(f"Pipeline '{PIPELINE_NAME}' not found — create it via Workflows UI first.")
    print("Steps:")
    print("  1. Go to Workflows → Delta Live Tables → Create Pipeline")
    print(f"  2. Name: {PIPELINE_NAME}")
    print("  3. Source: this notebook")
    print(f"  4. Catalog: {CATALOG} | Schema: {SCHEMA}")
    print("  5. Click Start")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Job Observability Features
# MAGIC
# MAGIC Once the Job is running, Databricks provides:
# MAGIC
# MAGIC **Task-level status**: Each task shows its own run duration, logs, and exit status.
# MAGIC
# MAGIC **Repair & rerun**: If `fit_sarima_models` fails (e.g., due to a library conflict),
# MAGIC you can repair just that task without re-running the DLT pipeline:
# MAGIC ```python
# MAGIC # Repair a specific failed task in a run
# MAGIC requests.post(f"https://{WORKSPACE_URL}/api/2.1/jobs/runs/repair", json={
# MAGIC     "run_id": <failed_run_id>,
# MAGIC     "rerun_tasks": ["fit_sarima_models"],
# MAGIC })
# MAGIC ```
# MAGIC
# MAGIC **System tables**: Query job run history in SQL:
# MAGIC ```sql
# MAGIC SELECT job_id, run_name, result_state, execution_duration_ms
# MAGIC FROM system.lakeflow.job_runs
# MAGIC WHERE workspace_id = current_workspace_id()
# MAGIC ORDER BY start_time DESC LIMIT 20
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review: DLT Pipeline Event Log
# MAGIC
# MAGIC After the pipeline runs, the event log is queryable as a Delta table.
# MAGIC This is your audit trail for data quality and pipeline health.

# COMMAND ----------

# Query the DLT pipeline event log (available after first pipeline run)
try:
    event_log = spark.sql(f"""
        SELECT
            timestamp,
            event_type,
            origin.flow_name AS table_name,
            details:flow_progress.metrics.num_output_rows AS rows_written,
            details:flow_progress.data_quality.dropped_records AS records_dropped,
            message
        FROM {CATALOG}.{SCHEMA}.`_dlt_event_log`
        WHERE event_type IN ('flow_progress', 'create_table', 'flow_definition')
        ORDER BY timestamp DESC
        LIMIT 30
    """)
    display(event_log)
except Exception:
    print("Event log available after the DLT pipeline runs.")
    print("Run the pipeline via Workflows UI first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review: Data Quality Metrics
# MAGIC
# MAGIC DLT tracks expectation pass/fail counts — queryable as metrics, viewable in the Pipeline UI.

# COMMAND ----------

try:
    quality_metrics = spark.sql(f"""
        SELECT
            origin.flow_name AS table_name,
            details:flow_progress.data_quality.expectations[*].name AS expectation_name,
            details:flow_progress.data_quality.expectations[*].passed_records AS passed,
            details:flow_progress.data_quality.expectations[*].failed_records AS failed
        FROM {CATALOG}.{SCHEMA}.`_dlt_event_log`
        WHERE event_type = 'flow_progress'
          AND details:flow_progress.data_quality IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    display(quality_metrics)
except Exception:
    print("Quality metrics available after pipeline run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Layer | Table | Pattern |
# MAGIC |---|---|---|
# MAGIC | Bronze | `bronze_policy_cdc` | Append-only, full CDC history, Auto Loader in production |
# MAGIC | Silver | `silver_policies` | SCD Type 2 via Apply Changes; expectations enforced |
# MAGIC | Gold | `gold_segment_monthly_stats` | Materialized view; monthly segment stats for SARIMA |
# MAGIC
# MAGIC The Gold table now feeds directly into Module 4's SARIMA/GARCH pipeline with `segment_id` and `claims_estimate` columns matching exactly.
# MAGIC
# MAGIC **Next:** Module 2 — making decisions about compute and scale before we fit those models.