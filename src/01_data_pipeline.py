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
    # On newer DBRs, `import dlt` succeeds even outside a pipeline context.
    # Detect actual pipeline execution by checking if pipeline-injected Spark
    # conf keys exist (these are only set when running inside a DLT pipeline).
    IN_PIPELINE = spark.conf.get("pipelines.id", None) is not None
except Exception:
    dlt = None
    IN_PIPELINE = False

if not IN_PIPELINE:
    dlt = None

import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

# ─── Configuration ────────────────────────────────────────────────────────────
# When running as a declarative pipeline, values come from the pipeline's configuration
# block (set in databricks.yml → resources/pipeline.yml).
# When running as a job task, values come from widgets (base_parameters in resources/jobs.yml).
if IN_PIPELINE:
    CATALOG = spark.conf.get("catalog", "my_catalog")
    SCHEMA  = spark.conf.get("schema",  "actuarial_data")
else:
    dbutils.widgets.text("catalog", "my_catalog",         "UC Catalog")
    dbutils.widgets.text("schema",  "actuarial_data", "UC Schema")
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
    Generate synthetic claims reserve development CDC records and write to Delta.

    Uses Spark-native generation: a small control table (3,360 rows) is exploded
    to individual development lag entries with random values via Spark SQL functions.
    Single Delta write.
    """
    import numpy as np
    import pandas as pd

    np.random.seed(2024)

    PRODUCT_LINES = ["Personal_Auto", "Commercial_Auto", "Homeowners", "Commercial_Property"]
    REGIONS = [
        "Ontario", "Quebec", "British_Columbia", "Alberta",
        "Manitoba", "Saskatchewan", "New_Brunswick", "Nova_Scotia",
        "Prince_Edward_Island", "Newfoundland",
    ]
    ACCIDENT_MONTHS = pd.date_range("2019-01-01", periods=84, freq="MS")

    import math

    BASE_ULTIMATE = {
        "Personal_Auto":       185_000_000,
        "Commercial_Auto":     105_000_000,
        "Homeowners":          155_000_000,
        "Commercial_Property":  73_000_000,
    }
    REGION_MULT = {
        "Ontario": 3.19, "Quebec": 1.85, "British_Columbia": 1.15,
        "Alberta": 1.00, "Manitoba": 0.30, "Saskatchewan": 0.25,
        "New_Brunswick": 0.17, "Nova_Scotia": 0.22,
        "Prince_Edward_Island": 0.04, "Newfoundland": 0.11,
    }

    # Line-specific maximum development lags (months)
    MAX_DEV_LAGS = {
        "Personal_Auto":       60,   # 5-year tail
        "Commercial_Auto":     60,   # 5-year tail
        "Homeowners":          36,   # 3-year tail
        "Commercial_Property": 48,   # 4-year tail
    }
    MAX_LAG_GLOBAL = max(MAX_DEV_LAGS.values())  # 60

    # Weibull CDF parameters for cumulative development: F(k) = 1 - exp(-(k/lambda)^theta)
    WEIBULL_PARAMS = {
        "Personal_Auto":       (15.0, 1.2),   # medium tail
        "Commercial_Auto":     (20.0, 1.3),   # longer tail
        "Homeowners":          (10.0, 1.1),   # short tail
        "Commercial_Property": (15.0, 1.4),   # medium-long tail
    }

    # Calendar-year superimposed inflation (annual rate, varying by line)
    CAL_YEAR_INFLATION = {
        "Personal_Auto":       0.03,   # 3% per year
        "Commercial_Auto":     0.04,   # 4% per year
        "Homeowners":          0.025,  # 2.5% per year
        "Commercial_Property": 0.05,   # 5% per year (construction costs)
    }

    # Generate parametric development patterns using Weibull CDF
    # Padded to MAX_LAG_GLOBAL (60) so all product lines use same-length arrays
    DEV_PATTERN = {}
    for _prod, (_lam, _theta) in WEIBULL_PARAMS.items():
        _max_lag = MAX_DEV_LAGS[_prod]
        _pattern = []
        for _k in range(1, MAX_LAG_GLOBAL + 1):
            if _k <= _max_lag:
                _val = 1.0 - math.exp(-(_k / _lam) ** _theta)
            else:
                _val = 1.0  # fully developed beyond line's max lag
            _pattern.append(round(_val, 6))
        DEV_PATTERN[_prod] = _pattern

    n_months = len(ACCIDENT_MONTHS)
    month_strs = [m.strftime("%Y-%m-%d") for m in ACCIDENT_MONTHS]

    # Control table: one row per (segment, month) with pre-drawn ultimate (3,360 rows)
    # Line-specific max development lags create a realistic staircase pattern
    control_rows = []
    for prod in PRODUCT_LINES:
        _line_max_lag = MAX_DEV_LAGS[prod]
        _inflation_rate = CAL_YEAR_INFLATION[prod]
        for region in REGIONS:
            base_ult = BASE_ULTIMATE[prod] * REGION_MULT[region]
            ult_noise = np.random.uniform(0.85, 1.15, size=n_months)
            control_rows.extend(
                (prod, region, f"{prod}__{region}",
                 month_strs[mi], min(_line_max_lag, n_months - mi),
                 float(base_ult * ult_noise[mi] * (1 + 0.003 * mi)),
                 float(_inflation_rate))
                for mi in range(n_months)
                if min(_line_max_lag, n_months - mi) >= 1
            )

    control_schema = StructType([
        StructField("product_line",   StringType(),  False),
        StructField("region",         StringType(),  False),
        StructField("segment_id",     StringType(),  False),
        StructField("accident_month", StringType(),  False),
        StructField("max_lag",        IntegerType(), False),
        StructField("ultimate",       DoubleType(),  False),
        StructField("inflation_rate", DoubleType(),  False),
    ])
    control_sdf = spark.createDataFrame(control_rows, schema=control_schema)

    # Dev pattern lookup: product_line → array of cumulative paid %
    # Arrays are padded to MAX_LAG_GLOBAL (60) so all lines use the same structure
    dev_pct_map = F.create_map(*[
        x for prod in PRODUCT_LINES for x in [
            F.lit(prod),
            F.array(*[F.lit(float(v)) for v in DEV_PATTERN[prod]])
        ]
    ])

    # Explode to individual lag entries, compute values in Spark
    # Calendar-year inflation: (1 + rate)^(calendar_year_offset)
    # ODP-consistent noise: wider range (0.85–1.15) for realistic development variability
    reserve_sdf = (
        control_sdf
        .withColumn("dev_lag", F.explode(F.sequence(F.lit(1), F.col("max_lag"))))
        .withColumns({
            "development_month": F.date_format(
                F.add_months(F.to_date("accident_month"), F.col("dev_lag")),
                "yyyy-MM-dd"),
            "_cal_year_offset": (
                F.year(F.to_date("accident_month")) + F.col("dev_lag") / F.lit(12.0)
                - F.lit(2019.0)),
        })
        # _inflation_factor depends on _cal_year_offset → separate withColumns
        .withColumn("_inflation_factor",
            F.pow(F.lit(1.0) + F.col("inflation_rate"), F.col("_cal_year_offset")))
        .withColumns({
            "cumulative_paid": F.round(F.col("ultimate") * F.least(
                F.element_at(dev_pct_map[F.col("product_line")], F.col("dev_lag"))
                * (F.lit(0.85) + F.rand(seed=2024) * F.lit(0.30))
                * F.col("_inflation_factor"),
                F.lit(1.0)), 2),
            "cumulative_incurred": F.round(F.col("ultimate")
                * (F.lit(0.95) + F.rand(seed=2025) * F.lit(0.10))
                * F.col("_inflation_factor"), 2),
            "op": F.when(F.col("dev_lag") == 1, "INSERT").otherwise("UPDATE"),
            "event_id": F.monotonically_increasing_id(),
            "ingested_at": F.current_timestamp().cast("string"),
        })
        # case_reserve depends on cumulative_incurred and cumulative_paid
        .withColumn("case_reserve",
            F.round(F.greatest(F.lit(0.0),
                F.col("cumulative_incurred") - F.col("cumulative_paid")), 2))
        .select("event_id", "segment_id", "product_line", "region",
                "accident_month", "development_month", "dev_lag",
                "cumulative_paid", "cumulative_incurred", "case_reserve",
                "op", "ingested_at")
    )

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    (reserve_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.{schema}.raw_reserve_development"))

    print(f"Reserve development CDC events written → {catalog}.{schema}.raw_reserve_development")
    return spark.table(f"{catalog}.{schema}.raw_reserve_development")

# ── Helper: generate synthetic claim incident data ────────────────────────────
def generate_claims_events_data(spark, catalog: str, schema: str):
    """
    Generate individual claim incident records for Jan 2019 – Dec 2025.

    Volume calibrated to Alberta Open Data 2013 loss ratios with population-weighted
    provincial multipliers (~500K claims/month, ~42M total). Uses Spark-native
    generation: a small control table (3,360 rows) is exploded to ~42M claim rows
    with random values computed via Spark SQL functions — single Delta write.
    """
    import numpy as np
    import pandas as pd

    np.random.seed(2025)

    PRODUCT_LINES = ["Personal_Auto", "Commercial_Auto", "Homeowners", "Commercial_Property"]
    REGIONS = [
        "Ontario", "Quebec", "British_Columbia", "Alberta",
        "Manitoba", "Saskatchewan", "New_Brunswick", "Nova_Scotia",
        "Prince_Edward_Island", "Newfoundland",
    ]
    MONTHS = pd.date_range("2019-01-01", periods=84, freq="MS")

    BASE_CLAIMS = {
        "Personal_Auto":       26_000,
        "Commercial_Auto":     10_500,
        "Homeowners":          18_500,
        "Commercial_Property":  5_200,
    }
    LOSS_RATIO_TARGET = {
        "Personal_Auto":       0.70,
        "Commercial_Auto":     0.67,
        "Homeowners":          0.62,
        "Commercial_Property": 0.65,
    }
    AVG_SEVERITY = {
        "Personal_Auto":       6_500.0,
        "Commercial_Auto":     9_200.0,
        "Homeowners":          8_400.0,
        "Commercial_Property": 14_000.0,
    }
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

    sigma2 = np.log(1.0 + 0.5 ** 2)
    month_strs = [m.strftime("%Y-%m-%d") for m in MONTHS]

    # Control table: one row per (segment, month) with pre-drawn Poisson counts (3,360 rows)
    # Poisson is drawn in numpy (no Spark SQL equivalent); everything else is Spark-native
    control_rows = []
    for prod in PRODUCT_LINES:
        mu_ln = float(np.log(AVG_SEVERITY[prod]) - sigma2 / 2.0)
        sigma_ln = float(np.sqrt(sigma2))
        loss_ratio = float(LOSS_RATIO_TARGET[prod])
        for region in REGIONS:
            base = BASE_CLAIMS[prod] * REGION_MULTIPLIER[region]
            n_per_month = np.random.poisson(base, size=len(MONTHS))
            control_rows.extend(
                (prod, region, month_strs[mi], int(n_per_month[mi]),
                 mu_ln, sigma_ln, loss_ratio)
                for mi in range(len(MONTHS))
            )

    control_schema = StructType([
        StructField("product_line", StringType(),  False),
        StructField("region",       StringType(),  False),
        StructField("loss_date",    StringType(),  False),
        StructField("n_claims",     IntegerType(), False),
        StructField("mu_ln",        DoubleType(),  False),
        StructField("sigma_ln",     DoubleType(),  False),
        StructField("loss_ratio",   DoubleType(),  False),
    ])
    control_sdf = (spark.createDataFrame(control_rows, schema=control_schema)
                       .repartition(40))

    # Claim type lookup: product_line → array of types
    ctype_map = F.create_map(*[
        x for prod in PRODUCT_LINES for x in [
            F.lit(prod),
            F.array(*[F.lit(t) for t in CLAIM_TYPES[prod]])
        ]
    ])

    # Explode to individual claims, compute random values with Spark
    claims_sdf = (
        control_sdf
        .withColumn("_seq", F.explode(F.sequence(F.lit(1), F.col("n_claims"))))
        .withColumns({
            "claim_amount": F.round(
                F.exp(F.col("mu_ln") + F.col("sigma_ln") * F.randn(seed=42)), 2),
            "_ctypes": ctype_map[F.col("product_line")],
        })
        # monthly_prem_exposure depends on claim_amount; claim_type depends on _ctypes
        .withColumns({
            "monthly_prem_exposure": F.round(
                F.col("claim_amount") / F.col("loss_ratio")
                * (F.lit(0.85) + F.rand(seed=43) * F.lit(0.30)), 2),
            "claim_type": F.element_at(F.col("_ctypes"),
                F.floor(F.rand(seed=44) * F.size(F.col("_ctypes"))).cast("int") + 1),
            "claim_id": F.monotonically_increasing_id(),
            "ingested_at": F.current_timestamp().cast("string"),
        })
        .select("claim_id", "product_line", "region", "loss_date",
                "claim_amount", "monthly_prem_exposure", "claim_type", "ingested_at")
    )

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    (claims_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.{schema}.raw_claims_events"))

    print(f"Claim events written → {catalog}.{schema}.raw_claims_events")
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
    from pyspark.sql import Window as _TriWindow

    @dlt.table(
        name="gold_reserve_triangle",
        comment="Loss development triangle by segment × accident month × development lag. Core actuarial exhibit for reserve adequacy, chain ladder, and bootstrap analysis. Includes incremental values for bootstrap resampling.",
        table_properties={"quality": "gold"},
    )
    def gold_reserve_triangle():
        """
        Gold: Loss development triangle from Silver reserve history.
        Uses current SCD2 versions only (__END_AT IS NULL) to get the latest
        reserve estimate at each development lag. Adds incremental values
        (cumulative at lag k minus cumulative at lag k-1) for bootstrap chain ladder.
        """
        silver = dlt.read("silver_reserves")

        agg = (
            silver
            .filter(F.col("__END_AT").isNull())  # current SCD2 records only
            .groupBy("segment_id", "product_line", "region", "accident_month", "dev_lag")
            .agg(
                F.sum("cumulative_paid").alias("cumulative_paid"),
                F.sum("cumulative_incurred").alias("cumulative_incurred"),
                F.sum("case_reserve").alias("case_reserve"),
            )
        )

        # Compute incremental values for bootstrap chain ladder
        _w = _TriWindow.partitionBy("segment_id", "accident_month").orderBy("dev_lag")
        return (
            agg
            .withColumns({
                "incremental_paid": F.col("cumulative_paid")
                    - F.coalesce(F.lag("cumulative_paid", 1).over(_w), F.lit(0.0)),
                "incremental_incurred": F.col("cumulative_incurred")
                    - F.coalesce(F.lag("cumulative_incurred", 1).over(_w), F.lit(0.0)),
            })
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
            .withColumns({
                "segment_id": F.concat_ws("__", "product_line", "region"),
                "month": F.date_trunc("month", F.to_date(F.col("loss_date"))),
            })
            .groupBy("segment_id", "product_line", "region", "month")
            .agg(
                F.count("*").alias("claims_count"),
                F.sum("claim_amount").alias("total_incurred"),
                F.avg("claim_amount").alias("avg_severity"),
                F.sum("monthly_prem_exposure").alias("earned_premium"),
            )
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
            .withColumns({
                "rolling_3m_mean":  F.avg("claims_count").over(w3),
                "rolling_6m_mean":  F.avg("claims_count").over(w6),
                "rolling_12m_mean": F.avg("claims_count").over(w12),
                "rolling_3m_std":   F.stddev("claims_count").over(w3),
                "_prev":  F.lag("claims_count", 1).over(lag_w),
                "_prev12": F.lag("claims_count", 12).over(lag_w),
            })
            # mom/yoy depend on _prev/_prev12
            .withColumns({
                "mom_change_pct": F.when(F.col("_prev") != 0,
                    (F.col("claims_count") - F.col("_prev")) / F.col("_prev") * 100)
                    .otherwise(F.lit(0.0)),
                "yoy_change_pct": F.when(F.col("_prev12") != 0,
                    (F.col("claims_count") - F.col("_prev12")) / F.col("_prev12") * 100)
                    .otherwise(F.lit(0.0)),
            })
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
        _prov_pairs = [item for pair in _PROVINCE_MAP.items() for item in pair]
        _map_expr   = F.create_map(*[F.lit(x) for x in _prov_pairs])
        w = Window.partitionBy("province").orderBy("month")
        hpi_lag = F.lag("hpi_index", 1).over(w)

        pivoted = (
            current
            .groupBy("province", "ref_date")
            .pivot("indicator_name", ["unemployment_rate", "hpi_index", "housing_starts"])
            .agg(F.first("value"))
            .withColumns({
                "month": F.to_date(F.concat(F.col("ref_date"), F.lit("-01"))),
                "region": _map_expr[F.col("province")],
            })
            .withColumn("hpi_growth",
                F.when(hpi_lag > 0,
                       (F.col("hpi_index") - hpi_lag) / hpi_lag * 100.0)
                 .otherwise(F.lit(None).cast("double")))
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