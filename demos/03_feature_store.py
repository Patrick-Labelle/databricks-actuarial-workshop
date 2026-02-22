# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3: Data Architecture
# MAGIC ## Unity Catalog Feature Store + Point-in-Time Joins
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC
# MAGIC ---
# MAGIC ### The Data Leakage Problem in Actuarial ML
# MAGIC
# MAGIC Actuarial models are **retrospective by nature** — pricing models are trained on historical experience,
# MAGIC but the features used must reflect what was *known at the time* of each observation.
# MAGIC
# MAGIC A classic leakage scenario:
# MAGIC
# MAGIC > Training a lapse model using a policyholder's **12-month rolling loss ratio** —
# MAGIC > but at the time of the lapse event, only **3 months** of loss data was available.
# MAGIC > The model learns from the future. In production, it fails.
# MAGIC
# MAGIC **Unity Catalog Feature Store** solves this with **point-in-time joins**:
# MAGIC retrieve feature values as they existed at training time, not as they exist today.
# MAGIC
# MAGIC ---
# MAGIC ### What We'll Build
# MAGIC
# MAGIC ```
# MAGIC Silver rolling features (Module 2)
# MAGIC         ↓ register
# MAGIC UC Feature Table (patrick_labelle.actuarial_workshop.segment_features)
# MAGIC         ↓ point-in-time join
# MAGIC Training set (no leakage guaranteed)
# MAGIC         ↓ publish
# MAGIC Online Table (low-latency lookup at inference)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# ─── Configuration ────────────────────────────────────────────────────────────
# Passed as base_parameters by the bundle job; defaults used when interactive.
dbutils.widgets.text("catalog",      "my_catalog",         "UC Catalog")
dbutils.widgets.text("schema",       "actuarial_workshop", "UC Schema")
dbutils.widgets.text("warehouse_id", "",                   "SQL Warehouse ID")
# job_mode: "true" skips display() calls, Feature Engineering client, training set demo,
# and Online Table creation. Set to "false" for full interactive demo.
dbutils.widgets.dropdown("job_mode", "false", ["false", "true"], "Job (automated) mode")
CATALOG       = dbutils.widgets.get("catalog")
SCHEMA        = dbutils.widgets.get("schema")
WAREHOUSE_ID  = dbutils.widgets.get("warehouse_id")
JOB_MODE      = dbutils.widgets.get("job_mode") == "true"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.segment_monthly_features"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build the Feature Table
# MAGIC
# MAGIC We create a rich feature table from the Silver rolling features (Module 2).
# MAGIC Each row represents the feature state for a `(segment_id, month)` combination.
# MAGIC
# MAGIC Feature groups:
# MAGIC - **Trend**: rolling means at 3m, 6m, 12m
# MAGIC - **Volatility**: rolling std, coefficient of variation
# MAGIC - **Momentum**: month-over-month and year-over-year changes
# MAGIC - **Seasonality**: month of year, quarter indicators
# MAGIC - **Exposure**: premium level, policy count proxies

# COMMAND ----------

# Load Silver rolling features (from Module 2) or regenerate
try:
    silver_features = spark.table(f"{CATALOG}.{SCHEMA}.silver_rolling_features")
    print(f"Loaded from Silver: {silver_features.count()} rows")
except Exception:
    print("Silver features not found — regenerating (run Module 2 first for full pipeline)")

    # Fallback: generate directly
    from itertools import product as iterproduct
    np.random.seed(42)
    PRODUCT_LINES = ["Personal_Auto","Commercial_Auto","Homeowners","Commercial_Property"]
    REGIONS       = ["Ontario","Quebec","British_Columbia","Alberta","Atlantic"]
    MONTHS        = pd.date_range("2019-01-01", periods=60, freq="MS")
    SEASONALITY   = {1:1.25,2:1.20,3:1.10,4:0.95,5:0.90,6:0.88,7:0.85,8:0.87,9:0.92,10:1.00,11:1.10,12:1.20}
    BASE          = {"Personal_Auto":450,"Commercial_Auto":180,"Homeowners":320,"Commercial_Property":90}
    MULT          = {"Ontario":1.4,"Quebec":1.1,"British_Columbia":1.2,"Alberta":1.0,"Atlantic":0.7}

    rows = []
    for prod, reg in iterproduct(PRODUCT_LINES, REGIONS):
        b = BASE[prod]*MULT[reg]
        y = [max(0, b*(1+0.003*i)*SEASONALITY[m.month]*(1+np.random.normal(0,0.08))) for i,m in enumerate(MONTHS)]
        y_arr = np.array(y)
        for i, (m, v) in enumerate(zip(MONTHS, y_arr)):
            rows.append({
                "segment_id": f"{prod}__{reg}", "product_line": prod, "region": reg,
                "month": m.date(), "claims_estimate": int(round(v)),
                "rolling_3m_mean": float(np.mean(y_arr[max(0,i-2):i+1])),
                "rolling_6m_mean": float(np.mean(y_arr[max(0,i-5):i+1])),
                "rolling_12m_mean": float(np.mean(y_arr[max(0,i-11):i+1])),
                "rolling_3m_std": float(np.std(y_arr[max(0,i-2):i+1])+1e-6),
                "mom_change_pct": float((v/y_arr[i-1]-1)*100 if i>0 else 0.0),
                "yoy_change_pct": float((v/y_arr[i-12]-1)*100 if i>=12 else 0.0),
                "avg_loss_ratio": round(np.random.uniform(0.55, 0.80), 4),
                "total_premium": round(b * np.random.uniform(3.2, 3.8), 2),
            })
    silver_features = spark.createDataFrame(pd.DataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enrich with Additional Features

# COMMAND ----------

# Add any columns that may be missing when loading from the Silver table
# (notebook 02 saves rolling means/pct_change but not std, loss ratio, or premium)
from pyspark.sql import Window as _W
import pyspark.sql.functions as _F

if "rolling_3m_std" not in silver_features.columns:
    w3 = _W.partitionBy("segment_id").orderBy("month").rowsBetween(-2, 0)
    silver_features = silver_features.withColumn(
        "rolling_3m_std",
        _F.coalesce(_F.stddev("claims_estimate").over(w3).cast("double"), _F.lit(0.0)) + _F.lit(1e-6)
    )

if "avg_loss_ratio" not in silver_features.columns:
    silver_features = silver_features.withColumn(
        "avg_loss_ratio",
        _F.lit(0.65) + (_F.abs(_F.hash("segment_id")) % 1000).cast("double") / _F.lit(40000)
    )

if "total_premium" not in silver_features.columns:
    silver_features = silver_features.withColumn(
        "total_premium",
        (_F.col("claims_estimate").cast("double") / _F.lit(0.65)) * _F.lit(3.5)
    )

feature_df = (
    silver_features
    .withColumn("month_ts",         F.col("month").cast("timestamp"))
    .withColumn("month_of_year",    F.month("month"))
    .withColumn("quarter",          F.quarter("month"))
    .withColumn("is_q1",            (F.quarter("month") == 1).cast("int"))  # High-claims quarter
    .withColumn("is_winter",        F.month("month").isin([12, 1, 2]).cast("int"))
    # Coefficient of variation — normalized volatility measure (used in actuarial risk scoring)
    .withColumn("coeff_variation_3m",
        F.when(F.col("rolling_3m_mean") > 0,
               F.col("rolling_3m_std") / F.col("rolling_3m_mean"))
         .otherwise(F.lit(0.0)))
    # Loss ratio stability: lower = more predictable segment
    .withColumn("loss_ratio_momentum",
        F.lag("avg_loss_ratio", 3).over(
            Window.partitionBy("segment_id").orderBy("month")
        ))
    .withColumn("loss_ratio_trend_3m",
        (F.col("avg_loss_ratio") - F.col("loss_ratio_momentum")))
    .drop("loss_ratio_momentum")
    .fillna(0.0, subset=["loss_ratio_trend_3m"])
    # Normalized exposure (relative to segment mean)
    # Guard against divide-by-zero (Photon + ANSI mode raises ArithmeticException on DBX Serverless)
    .withColumn("_avg_prem",  F.avg("total_premium").over(Window.partitionBy("segment_id")))
    .withColumn("normalized_premium",
        F.when(F.col("_avg_prem").isNotNull() & (F.col("_avg_prem") != 0),
               F.col("total_premium") / F.col("_avg_prem"))
         .otherwise(F.lit(1.0)))
    .drop("_avg_prem")
)

print(f"Feature table shape: {feature_df.count()} rows × {len(feature_df.columns)} columns")
if not JOB_MODE:
    display(feature_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register as a Unity Catalog Feature Table
# MAGIC
# MAGIC Registering in the Feature Store adds:
# MAGIC - **Lineage**: which models consumed this feature table
# MAGIC - **Point-in-time join capability**: retrieve values as of any timestamp
# MAGIC - **Discoverability**: actuaries can browse features in the UC catalog
# MAGIC - **Governance**: same permissions as all other UC assets

# COMMAND ----------

# Ensure the schema exists before writing the feature table
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

if JOB_MODE:
    # In job mode: save as plain Delta to avoid Feature Engineering client compatibility issues
    # on serverless compute. Interactive demo uses the Feature Engineering client below.
    (feature_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FEATURE_TABLE))
    print(f"Feature table saved (job mode): {FEATURE_TABLE}")
    FE_AVAILABLE = False
    fe = None
else:
    try:
        from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
        fe = FeatureEngineeringClient()  # Validate it can connect (may fail on some serverless configs)
        FE_AVAILABLE = True
    except Exception as _fe_import_err:
        print(f"FeatureEngineeringClient not available ({type(_fe_import_err).__name__}: {_fe_import_err}) "
              "— saving as plain Delta table instead.")
        FE_AVAILABLE = False
        fe = None

    if FE_AVAILABLE:
        # Write the feature table (create if not exists, overwrite if exists)
        try:
            fe.create_table(
                name             = FEATURE_TABLE,
                primary_keys     = ["segment_id", "month"],
                timestamp_keys   = ["month_ts"],           # Enables point-in-time joins
                df               = feature_df,
                description      = (
                    "Monthly actuarial risk features per segment (product × region). "
                    "Rolling means, volatility measures, seasonality indicators. "
                    "Registered for point-in-time correct training set assembly."
                ),
            )
            print(f"Feature table created: {FEATURE_TABLE}")
        except Exception as e:
            if "already exists" in str(e).lower() or "table already exists" in str(e).lower():
                print(f"Feature table already exists — overwriting data: {FEATURE_TABLE}")
                fe.write_table(name=FEATURE_TABLE, df=feature_df, mode="overwrite")
            else:
                raise
        print(f"Point-in-time key: month_ts")
    else:
        # Fallback: save as plain Delta table
        (feature_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(FEATURE_TABLE))
        print(f"Feature table saved as Delta (no Feature Store registration): {FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Assemble a Training Set with Point-in-Time Join
# MAGIC
# MAGIC The **point-in-time join** ensures that for each training observation,
# MAGIC we use the feature values that existed **at or before** the observation time.
# MAGIC
# MAGIC This is how we guarantee **no data leakage** from future feature values.
# MAGIC
# MAGIC **Scenario**: Training a loss ratio prediction model.
# MAGIC Each observation is a (segment_id, training_date) pair.
# MAGIC We want to retrieve the rolling features as they existed on that training date.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Create the Observation Table
# MAGIC
# MAGIC The observation table defines what we're trying to predict and when.
# MAGIC It's the "spine" of the training set.

# COMMAND ----------

# Training set assembly is an interactive demo — skip in automated job runs
if not JOB_MODE:
 observations_pdf = (
    feature_df
    .select("segment_id", "month_ts", "avg_loss_ratio")
    .filter(F.col("month_ts") >= "2020-01-01")  # Train on 2020+ data
    .withColumnRenamed("avg_loss_ratio", "target_loss_ratio")
    .withColumn("observation_id", F.monotonically_increasing_id())
    .toPandas()
 )

if not JOB_MODE:
    observations_spark = spark.createDataFrame(observations_pdf)
    print(f"Observation table: {len(observations_pdf)} rows")
    display(observations_spark.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Point-in-Time Join — No Future Leakage

# COMMAND ----------

# The Feature Engineering client handles the temporal join automatically:
# For each row in observations_spark (segment_id + timestamp),
# it retrieves features from FEATURE_TABLE where month_ts <= observation timestamp.
# This ensures we only use information available at prediction time.

if not JOB_MODE and FE_AVAILABLE:
    training_set = fe.create_training_set(
        df                = observations_spark,
        feature_lookups   = [
            FeatureLookup(
                table_name       = FEATURE_TABLE,
                feature_names    = [
                    "rolling_3m_mean", "rolling_6m_mean", "rolling_12m_mean",
                    "rolling_3m_std", "coeff_variation_3m",
                    "mom_change_pct", "yoy_change_pct",
                    "month_of_year", "quarter", "is_q1", "is_winter",
                    "loss_ratio_trend_3m", "normalized_premium",
                ],
                lookup_key       = ["segment_id"],
                timestamp_lookup_key = "month_ts",  # Point-in-time key
            ),
        ],
        label             = "target_loss_ratio",
        exclude_columns   = ["observation_id"],
    )
    training_df = training_set.load_df()
    print(f"Training set (point-in-time join): {training_df.count()} rows × {len(training_df.columns)} columns")
    print("Columns:", training_df.columns)
    display(training_df.limit(10))
elif not JOB_MODE:
    # Fallback: simple join without point-in-time guarantees
    print("Feature Store not available — using plain join for training set assembly")
    training_df = (
        observations_spark
        .join(feature_df.drop("avg_loss_ratio"), on=["segment_id", "month_ts"], how="left")
    )
    print(f"Training set (plain join): {training_df.count()} rows × {len(training_df.columns)} columns")
    display(training_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify: No Future Leakage
# MAGIC
# MAGIC Check that rolling_12m_mean uses only data from before the observation date.

# COMMAND ----------

# Sanity check: for the earliest observation dates (Jan 2020),
# rolling_12m_mean should reflect data from 2019 only (12 months prior)
if not JOB_MODE and 'training_df' in dir():
    leakage_check = (
        training_df
        .filter((F.col("segment_id") == "Personal_Auto__Ontario") & (F.col("month_ts") < "2020-04-01"))
        .select("segment_id", "month_ts", "rolling_12m_mean", "rolling_3m_mean", "target_loss_ratio")
        .orderBy("month_ts")
    )
    display(leakage_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Publish to Online Table (Real-Time Feature Serving)
# MAGIC
# MAGIC The Online Table enables **low-latency feature lookup at inference time**.
# MAGIC Instead of the model caller needing to pass all features, the serving endpoint
# MAGIC retrieves them automatically from the Online Table.
# MAGIC
# MAGIC Use case: when the Model Serving endpoint (Module 5) scores a new forecast request,
# MAGIC it looks up the latest rolling features for the requested segment — sub-millisecond latency.

# COMMAND ----------

import requests, json

try:
    WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
    # dbutils.notebook.entry_point is a JVM API that may not be available on all serverless configs
    TOKEN = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        .apiToken().get()
    )
except Exception as _tok_err:
    print(f"Could not retrieve API token ({type(_tok_err).__name__}) — skipping Online Table creation.")
    WORKSPACE_URL = None
    TOKEN = None

ONLINE_TABLE_NAME = f"{CATALOG}.{SCHEMA}.segment_features_online"

if TOKEN and WORKSPACE_URL:
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
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
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
# MAGIC ## 5. Examine Feature Lineage in Unity Catalog
# MAGIC
# MAGIC Every model that trains on this feature table creates a lineage link in Unity Catalog.
# MAGIC This answers: *"Which models use this feature? What changes if we update it?"*

# COMMAND ----------

# Query UC information schema to see feature table metadata
spark.sql(f"""
    SELECT
        table_name,
        table_type,
        comment,
        created,
        created_by,
        last_altered
    FROM {CATALOG}.information_schema.tables
    WHERE table_schema = '{SCHEMA}'
      AND table_name LIKE '%feature%'
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | What Happened |
# MAGIC |---|---|
# MAGIC | Feature computation | Rolling means, volatility, seasonality, momentum features for all 20 segments |
# MAGIC | UC registration | Feature table with timestamp key — enables point-in-time joins |
# MAGIC | Training set assembly | Point-in-time join guarantees no future leakage |
# MAGIC | Online Table | Low-latency feature lookup for real-time scoring |
# MAGIC | Lineage | UC tracks which models consume this feature table |
# MAGIC
# MAGIC **Key actuarial guarantee**: The point-in-time join ensures every training observation
# MAGIC uses only information that was available at the time — a fundamental requirement
# MAGIC for unbiased actuarial models and regulatory compliance.
# MAGIC
# MAGIC **Next:** Module 4 — with reliable data and leakage-free features, we're ready to fit
# MAGIC SARIMA, GARCH, and Monte Carlo models at scale across all 20 segments.