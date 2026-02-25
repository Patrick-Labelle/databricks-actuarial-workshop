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
    REGIONS        = [
        "Ontario", "Quebec", "British_Columbia", "Alberta",
        "Manitoba", "Saskatchewan", "New_Brunswick", "Nova_Scotia",
        "Prince_Edward_Island", "Newfoundland",
    ]
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

# ── Helper: generate synthetic claim incident data ────────────────────────────
def generate_claims_events_data(spark, catalog: str, schema: str):
    """
    Generate individual claim incident records for Jan 2019 – Dec 2024.

    Volume calibrated to Alberta Open Data 2013 loss ratios and IBC provincial
    market shares. Each row is one claim event. Writes to claims_events_raw,
    which feeds the DLT bronze_claims → gold_claims_monthly pipeline.

    Returns the Spark DataFrame.
    """
    import numpy as np
    import pandas as pd
    from itertools import product as iterproduct

    np.random.seed(2025)

    PRODUCT_LINES = ["Personal_Auto", "Commercial_Auto", "Homeowners", "Commercial_Property"]
    REGIONS = [
        "Ontario", "Quebec", "British_Columbia", "Alberta",
        "Manitoba", "Saskatchewan", "New_Brunswick", "Nova_Scotia",
        "Prince_Edward_Island", "Newfoundland",
    ]
    MONTHS = pd.date_range("2019-01-01", periods=72, freq="MS")

    # Monthly claim counts for Alberta (region multiplier = 1.00)
    BASE_CLAIMS = {
        "Personal_Auto":       450,
        "Commercial_Auto":     180,
        "Homeowners":          320,
        "Commercial_Property":  90,
    }
    # Alberta Open Data 2013: loss ratio calibration
    LOSS_RATIO_TARGET = {
        "Personal_Auto":       0.70,
        "Commercial_Auto":     0.67,
        "Homeowners":          0.62,
        "Commercial_Property": 0.65,
    }
    # Average claim severity by product line
    AVG_SEVERITY = {
        "Personal_Auto":       6_500.0,
        "Commercial_Auto":     9_200.0,
        "Homeowners":          8_400.0,
        "Commercial_Property": 14_000.0,
    }
    # IBC provincial market shares (relative volume multiplier)
    REGION_MULTIPLIER = {
        "Ontario":              1.40,
        "Quebec":               1.10,
        "British_Columbia":     1.20,
        "Alberta":              1.00,
        "Manitoba":             0.85,
        "Saskatchewan":         0.80,
        "New_Brunswick":        0.70,
        "Nova_Scotia":          0.75,
        "Prince_Edward_Island": 0.60,
        "Newfoundland":         0.65,
    }
    CLAIM_TYPES = {
        "Personal_Auto":       ["Collision", "Comprehensive", "Bodily_Injury"],
        "Commercial_Auto":     ["Collision", "Comprehensive", "Bodily_Injury"],
        "Homeowners":          ["Fire", "Water", "Theft", "Wind"],
        "Commercial_Property": ["Fire", "Water", "Wind", "Equipment"],
    }

    segment_frames = []
    for prod, region in iterproduct(PRODUCT_LINES, REGIONS):
        base       = BASE_CLAIMS[prod] * REGION_MULTIPLIER[region]
        loss_ratio = LOSS_RATIO_TARGET[prod]
        avg_sev    = AVG_SEVERITY[prod]
        ctypes     = CLAIM_TYPES[prod]

        # Poisson number of claims per month (vectorized for all 72 months at once)
        n_per_month = np.random.poisson(base, size=len(MONTHS))
        total_n     = int(n_per_month.sum())
        if total_n == 0:
            continue

        # Lognormal claim amounts (CV ≈ 0.5)
        sigma2  = np.log(1.0 + 0.5 ** 2)
        mu_ln   = np.log(avg_sev) - sigma2 / 2.0
        amounts = np.random.lognormal(mu_ln, np.sqrt(sigma2), size=total_n)

        # Premium exposure derived from amount and loss ratio (with small noise)
        prems    = amounts / loss_ratio * np.random.uniform(0.85, 1.15, size=total_n)
        type_idx = np.random.randint(0, len(ctypes), size=total_n)

        # Assign loss_date = first of the claim's month
        loss_dates = np.repeat([m.strftime("%Y-%m-%d") for m in MONTHS], n_per_month)

        segment_frames.append(pd.DataFrame({
            "product_line":          prod,
            "region":                region,
            "loss_date":             loss_dates,
            "claim_amount":          np.round(amounts, 2),
            "monthly_prem_exposure": np.round(prems, 2),
            "claim_type":            [ctypes[i] for i in type_idx],
            "ingested_at":           pd.Timestamp.now().isoformat(),
        }))

    pdf = pd.concat(segment_frames, ignore_index=True)
    pdf.insert(0, "claim_id", range(1, len(pdf) + 1))

    claims_schema = StructType([
        StructField("claim_id",              IntegerType(), False),
        StructField("product_line",          StringType(),  False),
        StructField("region",                StringType(),  False),
        StructField("loss_date",             StringType(),  False),
        StructField("claim_amount",          DoubleType(),  True),
        StructField("monthly_prem_exposure", DoubleType(),  True),
        StructField("claim_type",            StringType(),  True),
        StructField("ingested_at",           StringType(),  False),
    ])

    sdf = spark.createDataFrame(pdf, schema=claims_schema)
    (sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.{schema}.claims_events_raw"))

    count = sdf.count()
    print(f"Generated {count:,} claim events → {catalog}.{schema}.claims_events_raw")
    return sdf


# Run OUTSIDE the DLT context to seed data
if not IN_DLT:  # Use the IN_DLT flag set at top of notebook (avoids inaccessible DLT spark conf)
    raw_df = generate_bronze_cdc_data(spark, CATALOG, SCHEMA)
    display(raw_df.limit(20))
    # Generate claims events data (feeds bronze_claims → gold_claims_monthly DLT tables)
    generate_claims_events_data(spark, CATALOG, SCHEMA)
    # Create empty macro_indicators_raw landing zone if not yet present.
    # fetch_macro_data.py populates it before the DLT pipeline runs (see jobs.yml).
    # Creating the schema here ensures DLT streaming sources don't fail on first run
    # when this notebook is executed interactively without running fetch_macro_data first.
    _macro_table = f"{CATALOG}.{SCHEMA}.macro_indicators_raw"
    if not spark.catalog.tableExists(_macro_table):
        _empty_macro_schema = StructType([
            StructField("source_table",   StringType(), False),
            StructField("province",       StringType(), False),
            StructField("ref_date",       StringType(), False),
            StructField("indicator_name", StringType(), False),
            StructField("value",          DoubleType(), True),
            StructField("unit",           StringType(), True),
            StructField("ingested_at",    StringType(), False),
            StructField("batch_id",       StringType(), False),
        ])
        (spark.createDataFrame([], _empty_macro_schema).write
             .format("delta")
             .saveAsTable(_macro_table))
        print(f"Created empty macro_indicators_raw → {_macro_table}")

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
# MAGIC ### Step 5 — Claims Events: Bronze → Gold
# MAGIC
# MAGIC A second streaming medallion pipeline ingests individual **claim incidents** (one row per claim)
# MAGIC generated by `generate_claims_events_data()` and aggregates them to a monthly grain.
# MAGIC
# MAGIC | Table | Layer | Pattern |
# MAGIC |---|---|---|
# MAGIC | `bronze_claims` | Bronze | Append-only, streaming, data quality expectations |
# MAGIC | `gold_claims_monthly` | Gold | Materialized view — segment × month aggregate |
# MAGIC
# MAGIC `gold_claims_monthly` is the **primary input for Module 4** (SARIMAX, GARCH, Monte Carlo).
# MAGIC It provides real `claims_count`, `loss_ratio`, and `earned_premium` columns that replace
# MAGIC Module 4's former synthetic data generation loop.

# COMMAND ----------

if IN_DLT:
    @dlt.table(
        name="bronze_claims",
        comment="Raw claim incidents stream. Append-only — one row per claim event.",
        table_properties={
            "quality":      "bronze",
            "delta.enableChangeDataFeed": "true",
        },
    )
    @dlt.expect_or_drop("valid_amount",  "claim_amount >= 0")
    @dlt.expect_or_drop("valid_product", "product_line IN ('Personal_Auto','Commercial_Auto','Homeowners','Commercial_Property')")
    def bronze_claims():
        return (
            spark.readStream
                 .format("delta")
                 .table(f"{CATALOG}.{SCHEMA}.claims_events_raw")
        )

# COMMAND ----------

if IN_DLT:
    @dlt.table(
        name="gold_claims_monthly",
        comment="Monthly claims aggregate by product line × province. Primary input for SARIMAX/GARCH in Module 4.",
        table_properties={"quality": "gold"},
    )
    def gold_claims_monthly():
        """
        Gold: Aggregate claim events to segment × month grain.
        Computes claims_count, total incurred, average severity, earned premium, and loss ratio.
        """
        return (
            dlt.read("bronze_claims")
            .withColumn("segment_id", F.concat_ws("__", "product_line", "region"))
            .withColumn("month", F.date_trunc("month", F.to_date(F.col("loss_date"))))
            .groupBy("segment_id", "product_line", "region", "month")
            .agg(
                F.count("*").alias("claims_count"),
                F.sum("claim_amount").alias("total_incurred"),
                F.avg("claim_amount").alias("avg_severity"),
                F.sum("monthly_prem_exposure").alias("earned_premium"),
            )
            # Guard against divide-by-zero (Photon + ANSI mode on Serverless)
            .withColumn("loss_ratio",
                F.when(F.col("earned_premium") > 0,
                       F.col("total_incurred") / F.col("earned_premium"))
                 .otherwise(F.lit(0.0)))
            .orderBy("segment_id", "month")
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 — Macro Indicators: Bronze → Silver (SCD2) → Gold
# MAGIC
# MAGIC Real macroeconomic data from Statistics Canada flows through the same medallion pattern,
# MAGIC demonstrating that the architecture applies equally to external reference data:
# MAGIC
# MAGIC - **`bronze_macro_indicators`** — append-only raw ingestion from `macro_indicators_raw`
# MAGIC - **`silver_macro_indicators`** — SCD Type 2 via `apply_changes` (captures StatCan revisions)
# MAGIC - **`gold_macro_features`** — pivoted wide table, current versions only, adds `hpi_growth`
# MAGIC
# MAGIC `gold_macro_features` is joined to claims data in Module 4 to provide exogenous variables
# MAGIC for SARIMAX: **unemployment_rate** and **hpi_growth** (month-over-month HPI change).
# MAGIC
# MAGIC **Demo narrative:** *"The DLT pipeline processes your incoming claims stream. The gold layer
# MAGIC feeds directly into our SARIMAX models, which improve their forecasts using real macro
# MAGIC signals from Statistics Canada — watch the MAPE drop when we add provincial unemployment rates."*

# COMMAND ----------

if IN_DLT:
    @dlt.table(
        name="bronze_macro_indicators",
        comment="Raw StatCan macro indicator ingestion (append-only). One row per province × ref_date × indicator × batch.",
        table_properties={
            "quality":      "bronze",
            "delta.enableChangeDataFeed": "true",
        },
    )
    @dlt.expect_or_drop("valid_value",     "value IS NOT NULL")
    @dlt.expect_or_drop("valid_indicator", "indicator_name IN ('unemployment_rate','hpi_index','housing_starts')")
    def bronze_macro_indicators():
        return (
            spark.readStream
                 .format("delta")
                 .table(f"{CATALOG}.{SCHEMA}.macro_indicators_raw")
        )

# COMMAND ----------

if IN_DLT:
    # SCD Type 2 target table: tracks every StatCan revision over time.
    # A StatCan release that revises a prior month's unemployment figure creates a new version.
    dlt.create_streaming_table(
        name="silver_macro_indicators",
        comment="StatCan macro indicators with SCD Type 2 revision tracking. Current records have __END_AT IS NULL.",
        table_properties={"quality": "silver"},
    )

    dlt.apply_changes(
        target             = "silver_macro_indicators",
        source             = "bronze_macro_indicators",
        keys               = ["province", "ref_date", "indicator_name"],  # natural key
        sequence_by        = F.col("ingested_at"),                         # latest ingestion wins
        stored_as_scd_type = 2,                                            # retain all revisions
    )

# COMMAND ----------

# Macro province → workshop region mapping (StatCan uses spaces; workshop uses underscores)
_PROVINCE_MAP = {
    "Ontario":                   "Ontario",
    "Quebec":                    "Quebec",
    "British Columbia":          "British_Columbia",
    "Alberta":                   "Alberta",
    "Manitoba":                  "Manitoba",
    "Saskatchewan":              "Saskatchewan",
    "New Brunswick":             "New_Brunswick",
    "Nova Scotia":               "Nova_Scotia",
    "Prince Edward Island":      "Prince_Edward_Island",
    "Newfoundland and Labrador": "Newfoundland",
}

if IN_DLT:
    from pyspark.sql import Window as _Window

    @dlt.table(
        name="gold_macro_features",
        comment="Pivoted macro features for SARIMAX — current SCD2 versions only. Columns: region, month, unemployment_rate, hpi_index, hpi_growth, housing_starts.",
        table_properties={"quality": "gold"},
    )
    def gold_macro_features():
        """
        Gold: Current-version macro features pivoted to a wide format.
        - Filters silver to current SCD2 records (__END_AT IS NULL)
        - Pivots indicator_name rows → columns (unemployment_rate, hpi_index, housing_starts)
        - Maps StatCan province names to workshop region names
        - Adds hpi_growth = month-over-month HPI % change (ANSI-safe division)
        """
        from pyspark.sql import Window

        # Current SCD2 records only (end timestamp is NULL for active rows)
        current = (
            dlt.read("silver_macro_indicators")
               .filter(F.col("__END_AT").isNull())
        )

        # Pivot: one row per (province, ref_date), columns = indicator values
        pivoted = (
            current
            .groupBy("province", "ref_date")
            .pivot("indicator_name", ["unemployment_rate", "hpi_index", "housing_starts"])
            .agg(F.first("value"))
            # Convert "YYYY-MM" ref_date to a proper date column
            .withColumn("month", F.to_date(F.concat(F.col("ref_date"), F.lit("-01"))))
        )

        # Map province names to workshop region names using a Spark map literal
        _prov_pairs = [item for pair in _PROVINCE_MAP.items() for item in pair]
        _map_expr   = F.create_map(*[F.lit(x) for x in _prov_pairs])
        pivoted = pivoted.withColumn("region", _map_expr[F.col("province")])

        # hpi_growth = month-over-month HPI % change
        # ANSI-safe: guard against lag = 0 with F.when
        w = Window.partitionBy("province").orderBy("month")
        hpi_lag = F.lag("hpi_index", 1).over(w)
        pivoted = pivoted.withColumn(
            "hpi_growth",
            F.when(hpi_lag > 0,
                   (F.col("hpi_index") - hpi_lag) / hpi_lag * 100.0)
             .otherwise(F.lit(None).cast("double"))
        )

        return (
            pivoted
            .select("region", "month", "unemployment_rate", "hpi_index", "hpi_growth", "housing_starts")
            .filter(F.col("region").isNotNull())   # exclude non-province GEO rows
            .orderBy("region", "month")
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
# MAGIC Three medallion pipelines run in a single DLT notebook:
# MAGIC
# MAGIC | Layer | Table | Pattern | Consumer |
# MAGIC |---|---|---|---|
# MAGIC | Bronze | `bronze_policy_cdc` | Append-only CDC stream | Silver SCD2 |
# MAGIC | Silver | `silver_policies` | SCD Type 2 via Apply Changes | Gold aggregates |
# MAGIC | Gold | `gold_segment_monthly_stats` | Materialized view; monthly policy stats | Module 2, 3 |
# MAGIC | Bronze | `bronze_claims` | Append-only claim events stream | Gold claims |
# MAGIC | Gold | `gold_claims_monthly` | Materialized view; segment × month claims | **Module 4** |
# MAGIC | Bronze | `bronze_macro_indicators` | Append-only StatCan macro stream | Silver SCD2 |
# MAGIC | Silver | `silver_macro_indicators` | SCD Type 2; tracks StatCan revisions | Gold features |
# MAGIC | Gold | `gold_macro_features` | Pivoted macro features; current versions | **Module 4 exog** |
# MAGIC
# MAGIC `gold_claims_monthly` and `gold_macro_features` feed directly into Module 4's SARIMAX pipeline.
# MAGIC Real macro signals from Statistics Canada (unemployment, HPI growth) improve forecast accuracy
# MAGIC over baseline SARIMA — watch the MAPE improvement in the MLflow experiment comparison.
# MAGIC
# MAGIC **Next:** Module 2 — making decisions about compute and scale before we fit those models.