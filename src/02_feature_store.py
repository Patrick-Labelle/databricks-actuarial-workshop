# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: Data Architecture — UC Feature Store
# MAGIC
# MAGIC Registers Silver rolling features as a Unity Catalog Feature Table with
# MAGIC point-in-time join capability. Enriches with seasonality, volatility, and
# MAGIC exposure features for downstream SARIMAX consumption in Module 3.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window

# ─── Configuration ────────────────────────────────────────────────────────────
# Passed as base_parameters by the bundle job; defaults used when interactive.
dbutils.widgets.text("catalog",      "my_catalog",         "UC Catalog")
dbutils.widgets.text("schema",       "actuarial_workshop", "UC Schema")
dbutils.widgets.text("warehouse_id", "",                   "SQL Warehouse ID")
CATALOG       = dbutils.widgets.get("catalog")
SCHEMA        = dbutils.widgets.get("schema")
WAREHOUSE_ID  = dbutils.widgets.get("warehouse_id")
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.features_segment_monthly"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build the Feature Table
# MAGIC
# MAGIC Enrich Silver rolling features with trend, volatility, momentum, seasonality,
# MAGIC and exposure features per `(segment_id, month)`.

# COMMAND ----------

# Load Silver rolling features (from declarative pipeline — Module 1 must run first)
silver_features = spark.table(f"{CATALOG}.{SCHEMA}.silver_rolling_features")
print(f"Loaded from Silver: {silver_features.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enrich with Additional Features

# COMMAND ----------

# Add any columns that may be missing when loading from the Silver table
# (notebook 02 saves rolling means/pct_change but not std, loss ratio, or premium)
if "rolling_3m_std" not in silver_features.columns:
    w3 = Window.partitionBy("segment_id").orderBy("month").rowsBetween(-2, 0)
    silver_features = silver_features.withColumn(
        "rolling_3m_std",
        F.coalesce(F.stddev("claims_count").over(w3).cast("double"), F.lit(0.0)) + F.lit(1e-6)
    )

if "avg_loss_ratio" not in silver_features.columns:
    silver_features = silver_features.withColumn(
        "avg_loss_ratio",
        F.lit(0.65) + (F.abs(F.hash("segment_id")) % 1000).cast("double") / F.lit(40000)
    )

if "total_premium" not in silver_features.columns:
    silver_features = silver_features.withColumn(
        "total_premium",
        (F.col("claims_count").cast("double") / F.lit(0.65)) * F.lit(3.5)
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register as a Unity Catalog Feature Table
# MAGIC
# MAGIC Writes to UC with `timestamp_keys=["month_ts"]` for point-in-time joins.

# COMMAND ----------

# Ensure the schema exists before writing the feature table
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

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


# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Feature table registered with point-in-time join capability. Module 3 reads
# MAGIC `features_segment_monthly` for SARIMAX exogenous variables.