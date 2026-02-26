# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: Scale Decisions
# MAGIC ## Spark vs Ray, Pandas API on Spark & Vectorized UDFs
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC
# MAGIC ---
# MAGIC ### The Decision We're Making
# MAGIC
# MAGIC You have **reliable Gold data** from Module 1. Now you need to scale your statistical work.
# MAGIC The most common question actuaries ask when moving to Databricks:
# MAGIC
# MAGIC > *"I have working pandas/statsmodels code for one segment. How do I run it across 500 segments?"*
# MAGIC
# MAGIC Three patterns — choose based on your workload type:
# MAGIC
# MAGIC | Pattern | When to use | Module 2 demo |
# MAGIC |---|---|---|
# MAGIC | **Pandas API on Spark** | Scale existing pandas code with minimal rewrite | Part A |
# MAGIC | **`applyInPandas`** (grouped map) | Same Python logic applied per group (data-parallel) | Part B |
# MAGIC | **Ray tasks** | Many independent Python computations (task-parallel) | Part C |
# MAGIC
# MAGIC ---
# MAGIC ### Key Mental Model
# MAGIC
# MAGIC ```
# MAGIC Data-parallel  (Spark is ideal):  [row₁,row₂,...,rowₙ]  →  same fn applied to all rows/groups
# MAGIC Task-parallel  (Ray is ideal):    [task₁][task₂]...[taskₙ]  →  independent Python calls
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Load the Gold Data from Module 1

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import *

# ─── Configuration ────────────────────────────────────────────────────────────
# Passed as base_parameters by the bundle job; defaults used when interactive.
dbutils.widgets.text("catalog",   "my_catalog",         "UC Catalog")
dbutils.widgets.text("schema",    "actuarial_workshop", "UC Schema")
# run_ray: "auto" tries Ray when running interactively; "skip" disables it for
# automated job runs (Ray on Spark is not supported on serverless compute).
dbutils.widgets.dropdown("run_ray", "auto", ["auto", "skip"], "Run Ray section")
CATALOG  = dbutils.widgets.get("catalog")
SCHEMA   = dbutils.widgets.get("schema")
RUN_RAY  = dbutils.widgets.get("run_ray")

# Load segment data from Module 1's Gold table; fall back to synthetic data if unavailable.
# Note: gold_claims_monthly is a DLT materialized view — reading it in a regular
# notebook context works via Delta, but we guard with try/except for robustness.
try:
    _gold_tbl = f"{CATALOG}.{SCHEMA}.gold_claims_monthly"
    gold_df = spark.table(_gold_tbl)
    # Trigger a lightweight Spark action to verify the table is readable before proceeding
    _schema_ok = gold_df.schema  # schema-only fetch, no data scan
    print(f"Gold table schema verified: {_gold_tbl}")
    USE_GOLD = True
except Exception as _e:
    print(f"Gold table not available ({_e!r}) — using synthetic fallback")
    USE_GOLD = False

# COMMAND ----------

# Fallback: regenerate if Module 1 hasn't been run
if not USE_GOLD:
    from itertools import product as iterproduct

    np.random.seed(42)
    PRODUCT_LINES = ["Personal_Auto", "Commercial_Auto", "Homeowners", "Commercial_Property"]
    REGIONS       = [
        "Ontario", "Quebec", "British_Columbia", "Alberta",
        "Manitoba", "Saskatchewan", "New_Brunswick", "Nova_Scotia",
        "Prince_Edward_Island", "Newfoundland",
    ]
    MONTHS        = pd.date_range("2019-01-01", periods=72, freq="MS")
    SEASONALITY   = {1:1.25,2:1.20,3:1.10,4:0.95,5:0.90,6:0.88,7:0.85,8:0.87,9:0.92,10:1.00,11:1.10,12:1.20}
    BASE_CLAIMS   = {"Personal_Auto":450,"Commercial_Auto":180,"Homeowners":320,"Commercial_Property":90}
    REGION_MULT   = {"Ontario":1.4,"Quebec":1.1,"British_Columbia":1.2,"Alberta":1.0,
                     "Manitoba":0.85,"Saskatchewan":0.80,"New_Brunswick":0.70,
                     "Nova_Scotia":0.75,"Prince_Edward_Island":0.60,"Newfoundland":0.65}

    rows = []
    for prod, region in iterproduct(PRODUCT_LINES, REGIONS):
        base = BASE_CLAIMS[prod] * REGION_MULT[region]
        for i, m in enumerate(MONTHS):
            claims_count = max(0, base * (1+0.003*i) * SEASONALITY[m.month] * (1+np.random.normal(0,0.08)))
            avg_sev = np.random.uniform(5000, 12000)
            total_incurred = claims_count * avg_sev
            earned_premium = total_incurred / np.random.uniform(0.55, 0.80)
            rows.append({
                "segment_id": f"{prod}__{region}", "product_line": prod, "region": region,
                "month": m.date(), "claims_count": int(round(claims_count)),
                "total_incurred": round(total_incurred, 2),
                "avg_severity": round(avg_sev, 2),
                "earned_premium": round(earned_premium, 2),
                "loss_ratio": round(total_incurred / earned_premium, 4),
            })

    gold_df = spark.createDataFrame(pd.DataFrame(rows))
    print(f"Generated synthetic data: {len(rows)} rows (40 segments × 72 months)")

if not RUN_RAY == "skip":  # Skip display in automated job runs
    display(gold_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part A: Pandas API on Spark
# MAGIC ### Scale Without Rewrites
# MAGIC
# MAGIC **Scenario**: You have a pandas script that computes rolling window features on one segment.
# MAGIC The same logic needs to run on the full dataset (all segments × all months).
# MAGIC
# MAGIC **Pandas API on Spark** (`pyspark.pandas`) lets you use familiar pandas syntax on a distributed DataFrame.
# MAGIC The syntax is nearly identical — Spark handles the distribution transparently.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Original pandas code (single segment, single machine)

# COMMAND ----------

# Original pandas code — works fine for one segment, but single-threaded
one_segment_pdf = (
    gold_df
    .filter(F.col("segment_id") == "Personal_Auto__Ontario")
    .orderBy("month")
    .toPandas()
)

def compute_rolling_features_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling window features used as inputs to the SARIMA/GARCH models.
    Originally written for single-segment, single-node analysis.
    """
    pdf = pdf.sort_values("month").copy()
    claims = pdf["claims_count"].astype(float)

    pdf["rolling_3m_mean"]  = claims.rolling(3, min_periods=1).mean()
    pdf["rolling_6m_mean"]  = claims.rolling(6, min_periods=1).mean()
    pdf["rolling_12m_mean"] = claims.rolling(12, min_periods=1).mean()
    pdf["rolling_3m_std"]   = claims.rolling(3, min_periods=1).std().fillna(0)
    pdf["mom_change_pct"]   = claims.pct_change().fillna(0) * 100  # month-over-month %
    pdf["yoy_change_pct"]   = claims.pct_change(12).fillna(0) * 100  # year-over-year %
    pdf["trend_index"]      = np.arange(len(pdf)) / len(pdf)

    return pdf

result_single = compute_rolling_features_pandas(one_segment_pdf)
print(f"Single-segment result: {len(result_single)} rows")
print(result_single[["segment_id", "month", "claims_count", "rolling_3m_mean", "mom_change_pct"]].tail(5).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scale with Pandas API on Spark — minimal code change

# COMMAND ----------

# ── Native Spark implementation (reliable on all compute including serverless) ─
# The equivalent pyspark.pandas code is shown below in the comparison block.
from pyspark.sql import Window

seg_time_win = Window.partitionBy("segment_id").orderBy("month")

features_spark_df = (
    gold_df
    .withColumn("rolling_3m_mean",  F.avg("claims_count").over(seg_time_win.rowsBetween(-2, 0)))
    .withColumn("rolling_6m_mean",  F.avg("claims_count").over(seg_time_win.rowsBetween(-5, 0)))
    .withColumn("rolling_12m_mean", F.avg("claims_count").over(seg_time_win.rowsBetween(-11, 0)))
    .withColumn("_prev1",   F.lag("claims_count", 1).over(seg_time_win))
    .withColumn("_prev12",  F.lag("claims_count", 12).over(seg_time_win))
    # Guard against divide-by-zero (Photon + ANSI mode raises ArithmeticException on DBX Serverless)
    .withColumn("mom_change_pct",
        F.when(F.col("_prev1").isNotNull() & (F.col("_prev1") != 0),
               (F.col("claims_count") - F.col("_prev1")) / F.col("_prev1") * 100
        ).otherwise(F.lit(0.0)))
    .withColumn("yoy_change_pct",
        F.when(F.col("_prev12").isNotNull() & (F.col("_prev12") != 0),
               (F.col("claims_count") - F.col("_prev12")) / F.col("_prev12") * 100
        ).otherwise(F.lit(0.0)))
    .drop("_prev1", "_prev12")
)

print(f"All-segment rolling features computed across {gold_df.select('segment_id').distinct().count()} segments")
display(features_spark_df.limit(30))

# ── Equivalent pyspark.pandas code (demonstrates API parity) ──────────────────
# Uncomment to run interactively — requires non-serverless compute for .transform():
#
# import pyspark.pandas as ps
# ps.set_option("compute.ops_on_diff_frames", True)
# ps_df = gold_df.to_pandas_on_spark().sort_values(["segment_id", "month"])
# ps_df["rolling_3m_mean"]  = ps_df.groupby("segment_id")["claims_count"].transform(
#     lambda x: x.rolling(3, min_periods=1).mean())
# ps_df["mom_change_pct"]   = ps_df.groupby("segment_id")["claims_count"].transform(
#     lambda x: x.pct_change().fillna(0) * 100)
# features_spark_df = ps_df.to_spark()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save as Feature Table in Silver (feeds Module 3)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

(features_spark_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.silver_rolling_features"))

print(f"Saved rolling features → {CATALOG}.{SCHEMA}.silver_rolling_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### When to Use Pandas API on Spark vs Native Spark
# MAGIC
# MAGIC ```
# MAGIC ✅ Use Pandas API on Spark when:
# MAGIC    - Migrating existing pandas scripts to run at scale
# MAGIC    - EDA on data too large for single-node memory
# MAGIC    - Team is more familiar with pandas than Spark
# MAGIC    - ~85% of the pandas API is sufficient
# MAGIC
# MAGIC ⚠️  Switch to Native Spark when:
# MAGIC    - Performance is critical (native Spark is 2-5x faster for many operations)
# MAGIC    - Using operations not in the 85% coverage (in-place ops, positional indexing)
# MAGIC    - Need fine-grained control over partitioning
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part B: `applyInPandas` — Per-Group Model Fitting
# MAGIC ### Data-Parallel: Same Function, Every Group
# MAGIC
# MAGIC **Scenario**: Fit a statsmodels OLS trend model per segment.
# MAGIC `applyInPandas` distributes each segment's data to a separate Spark task.
# MAGIC Each task gets a pandas DataFrame — your statsmodels code runs unchanged.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

TREND_SCHEMA = StructType([
    StructField("segment_id",    StringType(), False),
    StructField("month_idx",     IntegerType(), False),
    StructField("claims_actual", DoubleType(), True),
    StructField("trend_fitted",  DoubleType(), True),
    StructField("detrended",     DoubleType(), True),
    StructField("slope",         DoubleType(), True),
    StructField("r_squared",     DoubleType(), True),
    StructField("annualized_growth_pct", DoubleType(), True),
])

def fit_trend_per_segment(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit an OLS trend model per segment using statsmodels.
    Standard Python/statsmodels code — no Spark imports needed.

    This function runs inside a Spark task, but receives a plain pandas DataFrame.
    """
    import statsmodels.api as sm
    import warnings
    warnings.filterwarnings("ignore")

    segment_id = pdf["segment_id"].iloc[0]
    pdf = pdf.sort_values("month").reset_index(drop=True)
    y = pdf["claims_count"].astype(float).values
    n = len(y)
    t = np.arange(n)

    try:
        X = sm.add_constant(t)
        model = sm.OLS(y, X).fit()

        slope     = model.params[1]
        intercept = model.params[0]
        fitted    = model.fittedvalues
        r_sq      = model.rsquared
        ann_growth = (slope * 12 / np.mean(y)) * 100  # annualized % growth

    except Exception:
        fitted    = np.full(n, np.nan)
        slope     = r_sq = ann_growth = np.nan

    return pd.DataFrame({
        "segment_id":             segment_id,
        "month_idx":              list(range(n)),
        "claims_actual":          list(y),
        "trend_fitted":           list(fitted),
        "detrended":              list(y - fitted),
        "slope":                  slope,
        "r_squared":              r_sq,
        "annualized_growth_pct":  ann_growth,
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribute Across All 40 Segments Simultaneously

# COMMAND ----------

trend_results = (
    gold_df
    .select("segment_id", "month", "claims_count")
    .groupby("segment_id")
    .applyInPandas(fit_trend_per_segment, schema=TREND_SCHEMA)
)

# trend_results.cache()  # Not supported on serverless compute
n_segments = trend_results.select("segment_id").distinct().count()
avg_r2 = trend_results.agg(F.mean("r_squared")).collect()[0][0]

print(f"Fitted trend models: {n_segments} segments | Avg R²: {avg_r2:.3f}")
display(
    trend_results
    .filter(F.col("segment_id") == "Personal_Auto__Ontario")
    .limit(30)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Print Trend Summary by Segment

# COMMAND ----------

display(
    trend_results
    .groupBy("segment_id")
    .agg(
        F.first("slope").alias("monthly_trend"),
        F.first("r_squared").alias("r_squared"),
        F.first("annualized_growth_pct").alias("annual_growth_pct"),
    )
    .orderBy(F.col("annual_growth_pct").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part C: Ray Tasks — Task-Parallel Computation
# MAGIC ### Independent Work Units: No Shared State Needed
# MAGIC
# MAGIC **Scenario**: Run a hyperparameter grid search for SARIMA orders across selected segments.
# MAGIC Each (segment, p, q) combination is an **independent experiment** — perfect for Ray task parallelism.
# MAGIC
# MAGIC Compare to Part B:
# MAGIC
# MAGIC | | `applyInPandas` (Part B) | Ray tasks (Part C) |
# MAGIC |---|---|---|
# MAGIC | Unit of work | Group of rows (one segment's data) | Arbitrary Python function |
# MAGIC | Data movement | Spark shuffles data to executors | Ray passes function args |
# MAGIC | Best for | Same fn over many groups of rows | Many independent Python calls |
# MAGIC | Our use case | Fit one model per segment | Grid search: many models per segment |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Ray on Spark

# COMMAND ----------

if RUN_RAY == "skip":
    print("Ray section skipped (run_ray=skip — set to 'auto' for interactive use)")
    RAY_AVAILABLE = False
else:
    try:
        import ray

        # On classic multi-node clusters Ray on Spark can distribute work across executors.
        # On serverless compute setup_ray_cluster is not supported; fall back to local Ray.
        try:
            from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
            try:
                shutdown_ray_cluster()
            except Exception:
                pass
            _n_executors = int(spark.conf.get("spark.executor.instances", "0"))
            if _n_executors > 0:
                setup_ray_cluster(max_worker_nodes=min(4, _n_executors), num_cpus_worker_node=4)
            else:
                raise RuntimeError("Serverless detected — skipping Ray on Spark setup")
        except Exception as _spark_ray_err:
            print(f"Ray on Spark not available ({_spark_ray_err!r}) — using local Ray cluster")

        ray.init(ignore_reinit_error=True)
        print(f"Ray initialized | Resources: {ray.cluster_resources()}")
        RAY_AVAILABLE = True

    except (ImportError, Exception) as e:
        print(f"Ray not available: {e}")
        RAY_AVAILABLE = False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Ray Task: Single (p, q) Fit for One Segment

# COMMAND ----------

# Collect data to driver for distribution to Ray workers (small dataset — only 60 months per segment)
segment_data = {
    row_s["segment_id"]: [r["claims_count"] for r in
        gold_df.filter(F.col("segment_id") == row_s["segment_id"])
               .orderBy("month")
               .select("claims_count").collect()]
    for row_s in gold_df.select("segment_id").distinct().limit(4).collect()  # Top 4 segments for demo
}

# Define the Ray remote function (only if Ray is available)
if RAY_AVAILABLE:
    @ray.remote
    def fit_arima_order(segment_id: str, y: list, p: int, d: int, q: int) -> dict:
        """
        Fit ARIMA(p,d,q) for a given segment and order combination.
        Each call is independent — perfect task-parallel fit for Ray.
        """
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore")

        from statsmodels.tsa.arima.model import ARIMA

        try:
            model = ARIMA(y, order=(p, d, q))
            fit   = model.fit()
            aic   = fit.aic
            bic   = fit.bic
            mape  = np.mean(np.abs(np.diff(y))) / np.mean(y) * 100  # simplified
        except Exception as e:
            aic = bic = mape = float("inf")

        return {
            "segment_id": segment_id,
            "p": p, "d": d, "q": q,
            "order_str": f"ARIMA({p},{d},{q})",
            "aic": aic,
            "bic": bic,
            "approx_mape": mape,
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fan Out Grid Search Across All Segments × All Orders

# COMMAND ----------

import itertools

# Hyperparameter grid
P_VALUES = [0, 1, 2]
D_VALUES = [0, 1]
Q_VALUES = [0, 1, 2]
ORDERS   = list(itertools.product(P_VALUES, D_VALUES, Q_VALUES))

if RAY_AVAILABLE:
    # Launch all tasks asynchronously
    futures = []
    for seg_id, y in segment_data.items():
        for p, d, q in ORDERS:
            futures.append(fit_arima_order.remote(seg_id, y, p, d, q))

    total_tasks = len(futures)
    print(f"Launched {total_tasks} Ray tasks ({len(segment_data)} segments × {len(ORDERS)} orders)")

    # Collect all results
    results = ray.get(futures)
    results_df = pd.DataFrame(results)

    # Find best order per segment (by AIC)
    best_per_segment = (
        results_df[results_df["aic"] < float("inf")]
        .sort_values("aic")
        .groupby("segment_id")
        .first()
        .reset_index()
        [["segment_id", "order_str", "aic", "bic"]]
    )

    print(f"\nBest ARIMA orders per segment:")
    print(best_per_segment.to_string(index=False))

    # Results displayed interactively — no need to persist since this is a technique demo.
    # The SARIMA model fitting in Module 04 uses its own optimized orders.

else:
    # ── Single-node fallback ──────────────────────────────────────────────────
    print("\nRunning grid search single-node (Ray not available)\n")
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")

    results = []
    for seg_id, y in list(segment_data.items())[:2]:  # 2 segments for speed
        for p, d, q in itertools.product([0,1], [0,1], [0,1]):
            try:
                fit = ARIMA(y, order=(p,d,q)).fit()
                results.append({"segment_id": seg_id, "order_str": f"ARIMA({p},{d},{q})",
                                "aic": fit.aic, "bic": fit.bic})
            except Exception:
                pass

    best = pd.DataFrame(results).sort_values("aic").groupby("segment_id").first()
    print(best[["order_str","aic","bic"]].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary: Choosing Your Scale Pattern
# MAGIC
# MAGIC | Workload | Recommended Pattern | Why |
# MAGIC |---|---|---|
# MAGIC | Scale existing pandas code | Pandas API on Spark | Minimal rewrite, ~85% coverage |
# MAGIC | Fit one model per segment | `applyInPandas` | Data-parallel; Spark manages distribution |
# MAGIC | Hyperparameter search | Ray tasks | Task-parallel; independent, no shared state |
# MAGIC | Monte Carlo simulation | Ray tasks | Task-parallel; pure Python; millions of trials |
# MAGIC | SQL/aggregation queries | SQL Warehouses + Photon | Vectorized columnar engine; 10x+ speedup |
# MAGIC
# MAGIC **Next:** Module 3 — now that we have reliable, scaled data and features across 40 segments,
# MAGIC let's register them in the Unity Catalog Feature Store with point-in-time correctness.
# MAGIC The features from `silver_rolling_features` will feed Module 4's SARIMAX models as exogenous variables.
# MAGIC