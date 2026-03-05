# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Foundations — Lakeflow Declarative Pipelines
# MAGIC
# MAGIC Medallion architecture (Bronze → Silver → Gold) for reserve CDC, claims events,
# MAGIC and StatCan macro indicators. Generates synthetic data, then defines the
# MAGIC declarative pipeline tables with SCD2, expectations, and materialized views.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A: Declarative Pipeline Definition
# MAGIC
# MAGIC This notebook is the pipeline source. Attach it to a Lakeflow Declarative Pipeline
# MAGIC to materialize Bronze/Silver/Gold tables automatically.

# COMMAND ----------

try:
    import dlt
    IN_PIPELINE = True
except Exception:
    # Running as a regular notebook (not inside a declarative pipeline context)
    # Data generation and Job creation sections still work normally
    # Catching Exception (not just ImportError) because on some runtimes
    # importing dlt outside a declarative pipeline raises a Py4J/Java error
    dlt = None
    IN_PIPELINE = False

import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

# ─── Configuration ────────────────────────────────────────────────────────────
# When running as a declarative pipeline, values come from the pipeline's configuration
# block (set in databricks.yml → resources/pipeline.yml).
# When running as a job task, values come from widgets (base_parameters in resources/jobs.yml).
if IN_PIPELINE:
    CATALOG = spark.conf.get("catalog", "my_catalog")
    SCHEMA  = spark.conf.get("schema",  "actuarial_workshop")
else:
    dbutils.widgets.text("catalog", "my_catalog",         "UC Catalog")
    dbutils.widgets.text("schema",  "actuarial_workshop", "UC Schema")
    CATALOG = dbutils.widgets.get("catalog")
    SCHEMA  = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 — Generate Raw CDC Data
# MAGIC
# MAGIC Synthetic reserve development CDC records and claim events seeded into
# MAGIC the landing zone. In production, replace with Auto Loader from S3/ADLS.

# COMMAND ----------

# ── Helper: generate synthetic reserve development CDC data ───────────────────
# Run this cell OUTSIDE the declarative pipeline to seed the landing zone with data

def generate_reserve_development_data(spark, catalog: str, schema: str):
    """
    Generate synthetic claims reserve development CDC records and write to a Delta
    table that acts as the Bronze source for the declarative pipeline.

    Reserve development is core actuarial data: case reserves are set when a claim
    is first reported, then revised as the claim develops (additional information,
    partial payments, settlements). The SCD2 pattern captures how reserve estimates
    change over time — a standard actuarial workflow for loss triangle construction.

    Each (segment_id, accident_month) has multiple development snapshots at
    increasing development lags (1, 2, …, 12+). Earlier lags have higher case
    reserves (claims are uncertain); later lags converge as claims settle.
    """
    import numpy as np
    import pandas as pd
    from itertools import product as iterproduct

    np.random.seed(2024)

    PRODUCT_LINES = ["Personal_Auto", "Commercial_Auto", "Homeowners", "Commercial_Property"]
    REGIONS = [
        "Ontario", "Quebec", "British_Columbia", "Alberta",
        "Manitoba", "Saskatchewan", "New_Brunswick", "Nova_Scotia",
        "Prince_Edward_Island", "Newfoundland",
    ]
    # Accident months: Jan 2019 – Dec 2025 (84 months, matching claims data)
    ACCIDENT_MONTHS = pd.date_range("2019-01-01", periods=84, freq="MS")

    # Base ultimate incurred by product line (per accident month, Alberta reference)
    # Scaled ~50x to match ~500K claims/month volume
    BASE_ULTIMATE = {
        "Personal_Auto":       185_000_000,
        "Commercial_Auto":     105_000_000,
        "Homeowners":          155_000_000,
        "Commercial_Property":  73_000_000,
    }
    # Population-weighted province multipliers (2025 Stats Canada, relative to Alberta = 1.00)
    REGION_MULT = {
        "Ontario": 3.19, "Quebec": 1.85, "British_Columbia": 1.15,
        "Alberta": 1.00, "Manitoba": 0.30, "Saskatchewan": 0.25,
        "New_Brunswick": 0.17, "Nova_Scotia": 0.22,
        "Prince_Edward_Island": 0.04, "Newfoundland": 0.11,
    }

    # Development pattern: cumulative paid % of ultimate at each dev lag (months)
    # Realistic pattern — fast-settling personal lines, slower commercial/liability
    DEV_PATTERN = {
        "Personal_Auto":       [0.15, 0.30, 0.45, 0.58, 0.68, 0.76, 0.82, 0.87, 0.91, 0.94, 0.97, 0.99],
        "Commercial_Auto":     [0.10, 0.22, 0.35, 0.46, 0.55, 0.63, 0.70, 0.77, 0.83, 0.88, 0.93, 0.97],
        "Homeowners":          [0.12, 0.28, 0.42, 0.54, 0.64, 0.73, 0.80, 0.86, 0.90, 0.94, 0.97, 0.99],
        "Commercial_Property": [0.08, 0.18, 0.30, 0.40, 0.50, 0.59, 0.67, 0.74, 0.81, 0.87, 0.92, 0.96],
    }

    rows = []
    event_seq = 1

    for prod, region in iterproduct(PRODUCT_LINES, REGIONS):
        segment_id = f"{prod}__{region}"
        base_ult = BASE_ULTIMATE[prod] * REGION_MULT[region]
        dev_pct = DEV_PATTERN[prod]

        for acc_month in ACCIDENT_MONTHS:
            # Ultimate incurred with noise (±15%)
            ultimate = base_ult * np.random.uniform(0.85, 1.15)

            # Slight trend: +0.3% per month
            month_idx = (acc_month.year - 2019) * 12 + (acc_month.month - 1)
            ultimate *= (1 + 0.003 * month_idx)

            # Max development lags available depends on how old the accident month is
            # (newer months have fewer lags observed)
            max_lag = min(12, len(ACCIDENT_MONTHS) - month_idx)
            if max_lag < 1:
                continue

            for lag in range(1, max_lag + 1):
                cum_paid_pct = dev_pct[lag - 1] * np.random.uniform(0.92, 1.08)
                cum_paid = ultimate * min(cum_paid_pct, 1.0)
                incurred = ultimate * np.random.uniform(0.95, 1.05)  # incurred fluctuates

                # Case reserve = incurred - cumulative paid (what's still expected)
                case_reserve = max(0, incurred - cum_paid)

                # Development month = accident month + lag months
                dev_month = acc_month + pd.DateOffset(months=lag)

                rows.append({
                    "event_id":           event_seq,
                    "segment_id":         segment_id,
                    "product_line":       prod,
                    "region":             region,
                    "accident_month":     acc_month.strftime("%Y-%m-%d"),
                    "development_month":  dev_month.strftime("%Y-%m-%d"),
                    "dev_lag":            lag,
                    "cumulative_paid":    round(cum_paid, 2),
                    "cumulative_incurred": round(incurred, 2),
                    "case_reserve":       round(case_reserve, 2),
                    "op":                 "INSERT" if lag == 1 else "UPDATE",
                    "ingested_at":        pd.Timestamp.now().isoformat(),
                })
                event_seq += 1

    pdf = pd.DataFrame(rows)

    reserve_schema = StructType([
        StructField("event_id",            IntegerType(), False),
        StructField("segment_id",          StringType(),  False),
        StructField("product_line",        StringType(),  False),
        StructField("region",              StringType(),  False),
        StructField("accident_month",      StringType(),  False),
        StructField("development_month",   StringType(),  False),
        StructField("dev_lag",             IntegerType(), False),
        StructField("cumulative_paid",     DoubleType(),  True),
        StructField("cumulative_incurred", DoubleType(),  True),
        StructField("case_reserve",        DoubleType(),  True),
        StructField("op",                  StringType(),  False),
        StructField("ingested_at",         StringType(),  False),
    ])

    sdf = spark.createDataFrame(pdf, schema=reserve_schema)

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    (sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.{schema}.raw_reserve_development"))

    count = sdf.count()
    print(f"Generated {count:,} reserve development CDC events → {catalog}.{schema}.raw_reserve_development")
    return sdf

# ── Helper: generate synthetic claim incident data ────────────────────────────
def generate_claims_events_data(spark, catalog: str, schema: str):
    """
    Generate individual claim incident records for Jan 2019 – Dec 2025.

    Volume calibrated to Alberta Open Data 2013 loss ratios with population-weighted
    provincial multipliers (~500K claims/month, ~42M total). Each row is one claim
    event. Writes to raw_claims_events per-segment batch to manage memory,
    which feeds the SDP bronze_claims → gold_claims_monthly pipeline.

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
    MONTHS = pd.date_range("2019-01-01", periods=84, freq="MS")

    # Monthly claim counts for Alberta (region multiplier = 1.00)
    # Scaled ~58x for ~500K claims/month across all provinces
    BASE_CLAIMS = {
        "Personal_Auto":       26_000,
        "Commercial_Auto":     10_500,
        "Homeowners":          18_500,
        "Commercial_Property":  5_200,
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
    # Population-weighted province multipliers (2025 Stats Canada, relative to Alberta = 1.00)
    REGION_MULTIPLIER = {
        "Ontario":              3.19,
        "Quebec":               1.85,
        "British_Columbia":     1.15,
        "Alberta":              1.00,
        "Manitoba":             0.30,
        "Saskatchewan":         0.25,
        "New_Brunswick":        0.17,
        "Nova_Scotia":          0.22,
        "Prince_Edward_Island": 0.04,
        "Newfoundland":         0.11,
    }
    CLAIM_TYPES = {
        "Personal_Auto":       ["Collision", "Comprehensive", "Bodily_Injury"],
        "Commercial_Auto":     ["Collision", "Comprehensive", "Bodily_Injury"],
        "Homeowners":          ["Fire", "Water", "Theft", "Wind"],
        "Commercial_Property": ["Fire", "Water", "Wind", "Equipment"],
    }

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

    # Write per-segment batch to avoid >10 GB in-memory DataFrame for ~42M rows
    claim_id_offset = 0
    first = True
    total_count = 0
    for prod, region in iterproduct(PRODUCT_LINES, REGIONS):
        base       = BASE_CLAIMS[prod] * REGION_MULTIPLIER[region]
        loss_ratio = LOSS_RATIO_TARGET[prod]
        avg_sev    = AVG_SEVERITY[prod]
        ctypes     = CLAIM_TYPES[prod]

        # Poisson number of claims per month (vectorized for all months at once)
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

        segment_pdf = pd.DataFrame({
            "claim_id":              range(claim_id_offset + 1, claim_id_offset + total_n + 1),
            "product_line":          prod,
            "region":                region,
            "loss_date":             loss_dates,
            "claim_amount":          np.round(amounts, 2),
            "monthly_prem_exposure": np.round(prems, 2),
            "claim_type":            [ctypes[i] for i in type_idx],
            "ingested_at":           pd.Timestamp.now().isoformat(),
        })
        claim_id_offset += total_n
        total_count += total_n

        sdf = spark.createDataFrame(segment_pdf, schema=claims_schema)
        mode = "overwrite" if first else "append"
        (sdf.write
            .format("delta")
            .mode(mode)
            .option("overwriteSchema", "true" if first else "false")
            .saveAsTable(f"{catalog}.{schema}.raw_claims_events"))
        first = False

    print(f"Generated {total_count:,} claim events → {catalog}.{schema}.raw_claims_events")
    return spark.table(f"{catalog}.{schema}.raw_claims_events")


# Run OUTSIDE the declarative pipeline context to seed data
if not IN_PIPELINE:  # Use the IN_PIPELINE flag set at top of notebook (avoids inaccessible SDP spark conf)
    generate_reserve_development_data(spark, CATALOG, SCHEMA)
    # Generate claims events data (feeds bronze_claims → gold_claims_monthly SDP tables)
    generate_claims_events_data(spark, CATALOG, SCHEMA)
    # Create empty raw_macro_indicators landing zone if not yet present.
    # fetch_macro_data.py populates it before the declarative pipeline runs (see jobs.yml).
    # Creating the schema here ensures SDP streaming sources don't fail on first run
    # when this notebook is executed interactively without running fetch_macro_data first.
    _macro_table = f"{CATALOG}.{SCHEMA}.raw_macro_indicators"
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
        print(f"Created empty raw_macro_indicators → {_macro_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 — Bronze: Raw Ingestion Layer
# MAGIC
# MAGIC Append-only raw CDC history. Data quality expectations enforce validity at ingestion.

# COMMAND ----------

if IN_PIPELINE:
    @dlt.table(
        name="bronze_reserve_cdc",
        comment="Raw reserve development CDC events. Append-only — tracks how case reserves evolve over time.",
        table_properties={
            "quality":      "bronze",
            "delta.enableChangeDataFeed": "true",
        },
    )
    @dlt.expect_or_drop("valid_paid",      "cumulative_paid >= 0")
    @dlt.expect_or_drop("valid_incurred",  "cumulative_incurred >= 0")
    @dlt.expect_or_drop("valid_reserve",   "case_reserve >= 0")
    @dlt.expect_or_drop("valid_product",   "product_line IN ('Personal_Auto','Commercial_Auto','Homeowners','Commercial_Property')")
    @dlt.expect("valid_dev_lag",           "dev_lag BETWEEN 1 AND 120")
    def bronze_reserve_cdc():
        """
        Bronze: Read raw reserve development CDC records from the landing zone.
        Append-only — preserves complete history of reserve estimate revisions.
        """
        return (
            spark.readStream
                 .format("delta")
                 .table(f"{CATALOG}.{SCHEMA}.raw_reserve_development")
        )
else:
    print("ℹ️  SDP not active — Bronze/Silver/Gold tables are created by the declarative pipeline.")
    print("   Create a declarative pipeline pointing at this notebook to materialize these tables.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 — Silver: Reserve History (SCD Type 2 via Apply Changes)
# MAGIC
# MAGIC `dlt.apply_changes()` with SCD Type 2 on key `(segment_id, accident_month, dev_lag)`.
# MAGIC Sequenced by `development_month`; quality expectations enforced at Bronze.

# COMMAND ----------

# ── Silver target table + Apply Changes ───────────────────────────────────────
# create_streaming_table() + apply_changes() pattern (SDP runtimes ≥ 2024-Q3).
if IN_PIPELINE:
    dlt.create_streaming_table(
        name="silver_reserves",
        comment="Reserve development history with SCD Type 2 tracking. Each (segment, accident_month, dev_lag) tracks how estimates evolve.",
        table_properties={"quality": "silver"},
    )

    # Apply Changes: CDC → SCD Type 2
    dlt.apply_changes(
        target       = "silver_reserves",
        source       = "bronze_reserve_cdc",
        keys         = ["segment_id", "accident_month", "dev_lag"],
        sequence_by  = F.col("development_month"),
        stored_as_scd_type = 2,
        column_list  = ["segment_id", "product_line", "region", "accident_month",
                        "development_month", "dev_lag", "cumulative_paid",
                        "cumulative_incurred", "case_reserve"],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 — Gold: Loss Development Triangle (Materialized View)
# MAGIC
# MAGIC Aggregates Silver reserve history into the standard actuarial loss development
# MAGIC triangle (accident month x dev lag). Auto-refreshed as Silver changes.

# COMMAND ----------

if IN_PIPELINE:
    @dlt.table(
        name="gold_reserve_triangle",
        comment="Loss development triangle by segment × accident month × development lag. Core actuarial exhibit for reserve adequacy analysis.",
        table_properties={"quality": "gold"},
    )
    def gold_reserve_triangle():
        """
        Gold: Loss development triangle from Silver reserve history.
        Uses current SCD2 versions only (__END_AT IS NULL) to get the latest
        reserve estimate at each development lag.
        """
        silver = dlt.read("silver_reserves")

        return (
            silver
            .filter(F.col("__END_AT").isNull())  # current SCD2 records only
            .groupBy("segment_id", "product_line", "region", "accident_month", "dev_lag")
            .agg(
                F.sum("cumulative_paid").alias("cumulative_paid"),
                F.sum("cumulative_incurred").alias("cumulative_incurred"),
                F.sum("case_reserve").alias("case_reserve"),
            )
            .orderBy("segment_id", "accident_month", "dev_lag")
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 — Claims Events: Bronze → Gold
# MAGIC
# MAGIC Individual claim incidents → `bronze_claims` (append-only) → `gold_claims_monthly`
# MAGIC (segment x month aggregate). Primary input for Modules 2 and 3.

# COMMAND ----------

if IN_PIPELINE:
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
                 .table(f"{CATALOG}.{SCHEMA}.raw_claims_events")
        )

# COMMAND ----------

if IN_PIPELINE:
    @dlt.table(
        name="gold_claims_monthly",
        comment="Monthly claims aggregate by product line × province. Primary input for SARIMAX/GARCH in Module 3.",
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

if IN_PIPELINE:
    from pyspark.sql import Window as _RollingWindow

    @dlt.table(
        name="silver_rolling_features",
        comment="Rolling statistical features per segment. Feeds Feature Store (Module 2).",
        table_properties={"quality": "silver"},
    )
    def silver_rolling_features():
        claims = dlt.read("gold_claims_monthly")
        w3  = _RollingWindow.partitionBy("segment_id").orderBy("month").rowsBetween(-2, 0)
        w6  = _RollingWindow.partitionBy("segment_id").orderBy("month").rowsBetween(-5, 0)
        w12 = _RollingWindow.partitionBy("segment_id").orderBy("month").rowsBetween(-11, 0)
        lag_w = _RollingWindow.partitionBy("segment_id").orderBy("month")
        return (
            claims
            .withColumn("rolling_3m_mean",  F.avg("claims_count").over(w3))
            .withColumn("rolling_6m_mean",  F.avg("claims_count").over(w6))
            .withColumn("rolling_12m_mean", F.avg("claims_count").over(w12))
            .withColumn("rolling_3m_std",   F.stddev("claims_count").over(w3))
            .withColumn("_prev",  F.lag("claims_count", 1).over(lag_w))
            .withColumn("_prev12", F.lag("claims_count", 12).over(lag_w))
            .withColumn("mom_change_pct",
                F.when(F.col("_prev") != 0,
                       (F.col("claims_count") - F.col("_prev")) / F.col("_prev") * 100)
                 .otherwise(F.lit(0.0)))
            .withColumn("yoy_change_pct",
                F.when(F.col("_prev12") != 0,
                       (F.col("claims_count") - F.col("_prev12")) / F.col("_prev12") * 100)
                 .otherwise(F.lit(0.0)))
            .drop("_prev", "_prev12")
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 — Macro Indicators: Bronze → Silver (SCD2) → Gold
# MAGIC
# MAGIC StatCan macro data (unemployment, HPI, housing starts) through the same medallion
# MAGIC pattern. `gold_macro_features` provides SARIMAX exogenous variables in Module 3.

# COMMAND ----------

if IN_PIPELINE:
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
                 .table(f"{CATALOG}.{SCHEMA}.raw_macro_indicators")
        )

# COMMAND ----------

if IN_PIPELINE:
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

if IN_PIPELINE:
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
# MAGIC ## Summary
# MAGIC
# MAGIC Three medallion pipelines run in a single declarative pipeline notebook:
# MAGIC
# MAGIC | Layer | Table | Pattern | Consumer |
# MAGIC |---|---|---|---|
# MAGIC | Bronze | `bronze_reserve_cdc` | Append-only reserve development CDC | Silver SCD2 |
# MAGIC | Silver | `silver_reserves` | SCD Type 2 via Apply Changes | Gold triangle |
# MAGIC | Gold | `gold_reserve_triangle` | Materialized view; loss development triangle | **Module 3, App** |
# MAGIC | Bronze | `bronze_claims` | Append-only claim events stream | Gold claims |
# MAGIC | Gold | `gold_claims_monthly` | Materialized view; segment × month claims | **interactive Module 2, Module 2, 3** |
# MAGIC | Silver | `silver_rolling_features` | Rolling window features per segment | **Module 2 Feature Store** |
# MAGIC | Bronze | `bronze_macro_indicators` | Append-only StatCan macro stream | Silver SCD2 |
# MAGIC | Silver | `silver_macro_indicators` | SCD Type 2; tracks StatCan revisions | Gold features |
# MAGIC | Gold | `gold_macro_features` | Pivoted macro features; current versions | **Module 3 exog** |
# MAGIC
# MAGIC `gold_claims_monthly` feeds interactive Module 2 (performance comparison), Module 2 (feature store), and Module 3 (SARIMAX).
# MAGIC `silver_rolling_features` computes rolling window statistics from `gold_claims_monthly` and feeds Module 2.
# MAGIC `gold_reserve_triangle` provides the loss development triangle for reserve adequacy validation in Module 3
# MAGIC and display in the Streamlit app.
# MAGIC
# MAGIC **Next:** Module 2 — Feature Store with point-in-time joins for leakage-free training sets.