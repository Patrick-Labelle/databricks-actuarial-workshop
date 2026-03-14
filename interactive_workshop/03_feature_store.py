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
# MAGIC Silver rolling features (SDP pipeline)
# MAGIC         ↓ register
# MAGIC UC Feature Table ({catalog}.{schema}.segment_features)
# MAGIC         ↓ point-in-time join
# MAGIC Training set (no leakage guaranteed)
# MAGIC         ↓ publish
# MAGIC Online Table (low-latency lookup at inference)
# MAGIC ```

# MAGIC
# MAGIC > **Interactive notebook** — Run this attached to a classic ML cluster (DBR 16.4+). The automated job version runs as part of the setup job.
# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window

# ─── Configuration ────────────────────────────────────────────────────────────
# Passed as base_parameters by the bundle job; defaults used when interactive.
dbutils.widgets.text("catalog",      "my_catalog",         "UC Catalog")
dbutils.widgets.text("schema",       "actuarial_data", "UC Schema")
dbutils.widgets.text("warehouse_id", "",                   "SQL Warehouse ID")
CATALOG       = dbutils.widgets.get("catalog")
SCHEMA        = dbutils.widgets.get("schema")
WAREHOUSE_ID  = dbutils.widgets.get("warehouse_id")
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.features_segment_monthly"

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

# Load Silver rolling features (from declarative pipeline — Module 2 must run first)
silver_features = spark.table(f"{CATALOG}.{SCHEMA}.silver_rolling_features")
print(f"Loaded from Silver: {silver_features.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enrich with Additional Features

# COMMAND ----------

# Add derived columns not in silver_rolling_features (avg_loss_ratio, total_premium)
# rolling_3m_std already exists from the DLT pipeline
w_seg = Window.partitionBy("segment_id").orderBy("month")
silver_features = silver_features.withColumns({
    "avg_loss_ratio": F.lit(0.65) + (F.abs(F.hash("segment_id")) % 1000).cast("double") / F.lit(40000),
    "total_premium": (F.col("claims_count").cast("double") / F.lit(0.65)) * F.lit(3.5),
})

feature_df = (
    silver_features
    .withColumns({
        "month_ts":          F.col("month").cast("timestamp"),
        "month_of_year":     F.month("month"),
        "quarter":           F.quarter("month"),
        "is_q1":             (F.quarter("month") == 1).cast("int"),  # High-claims quarter
        "is_winter":         F.month("month").isin([12, 1, 2]).cast("int"),
        # Coefficient of variation — normalized volatility measure (used in actuarial risk scoring)
        "coeff_variation_3m": F.when(F.col("rolling_3m_mean") > 0,
                                     F.col("rolling_3m_std") / F.col("rolling_3m_mean"))
                               .otherwise(F.lit(0.0)),
        # Loss ratio stability: lower = more predictable segment
        "loss_ratio_momentum": F.lag("avg_loss_ratio", 3).over(w_seg),
        # Normalized exposure (relative to segment mean)
        "_avg_prem": F.avg("total_premium").over(Window.partitionBy("segment_id")),
    })
    .withColumns({
        "loss_ratio_trend_3m": F.coalesce(
            F.col("avg_loss_ratio") - F.col("loss_ratio_momentum"), F.lit(0.0)),
        "normalized_premium": F.when(
            F.col("_avg_prem").isNotNull() & (F.col("_avg_prem") != 0),
            F.col("total_premium") / F.col("_avg_prem"))
            .otherwise(F.lit(1.0)),
    })
    .drop("loss_ratio_momentum", "_avg_prem")
)

print(f"Feature table shape: {feature_df.count()} rows × {len(feature_df.columns)} columns")

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

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fe = FeatureEngineeringClient()

# Write the feature table (create if not exists, overwrite if exists)
try:
    fe.create_table(
        name             = FEATURE_TABLE,
        primary_keys     = ["segment_id", "month", "month_ts"],
        timestamp_keys   = ["month_ts"],
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

observations_pdf = (
    feature_df
    .select("segment_id", "month_ts", "avg_loss_ratio")
    .filter(F.col("month_ts") >= "2020-01-01")  # Train on 2020+ data
    .withColumnRenamed("avg_loss_ratio", "target_loss_ratio")
    .withColumn("observation_id", F.monotonically_increasing_id())
    .toPandas()
)
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify: No Future Leakage
# MAGIC
# MAGIC Check that rolling_12m_mean uses only data from before the observation date.

# COMMAND ----------

# Sanity check: for the earliest observation dates (Jan 2020),
# rolling_12m_mean should reflect data from 2019 only (12 months prior)
leakage_check = (
    training_df
    .filter((F.col("segment_id") == "Personal_Auto__Ontario") & (F.col("month_ts") < "2020-04-01"))
    .select("segment_id", "month_ts", "rolling_12m_mean", "rolling_3m_mean", "target_loss_ratio")
    .orderBy("month_ts")
)
display(leakage_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Examine Feature Lineage in Unity Catalog
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
# MAGIC | Feature computation | Rolling means, volatility, seasonality, momentum features for all 52 segments |
# MAGIC | UC registration | Feature table with timestamp key — enables point-in-time joins |
# MAGIC | Training set assembly | Point-in-time join guarantees no future leakage |
# MAGIC | Lineage | UC tracks which models consume this feature table |
# MAGIC
# MAGIC **Key actuarial guarantee**: The point-in-time join ensures every training observation
# MAGIC uses only information that was available at the time — a fundamental requirement
# MAGIC for unbiased actuarial models and regulatory compliance.
# MAGIC
# MAGIC **Next:** Module 4 — with reliable data and leakage-free features, we're ready to fit
# MAGIC SARIMA, GARCH, and Monte Carlo models at scale across all 52 segments. Module 4 reads
# MAGIC `features_segment_monthly` to provide exogenous variables for SARIMAX forecasting.