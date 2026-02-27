# Databricks notebook source
# MAGIC %md
# MAGIC # Module 2: Performance at Scale
# MAGIC ## Choosing the Right Spark Pattern
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC *Audience: Actuaries, Data Scientists, Financial Analysts*
# MAGIC
# MAGIC ---
# MAGIC ### The Question
# MAGIC
# MAGIC You have **reliable Gold data** from Module 1. Before building features and models,
# MAGIC you need to understand **which Spark pattern to use when**.
# MAGIC
# MAGIC > *"I have working pandas code. How do I scale it — and what traps should I avoid?"*
# MAGIC
# MAGIC This notebook compares multiple approaches side-by-side, **with timing**, so you can see
# MAGIC the performance trade-offs yourself. **No persistent outputs** — everything here is a
# MAGIC learning exercise.
# MAGIC
# MAGIC ---
# MAGIC ### What We'll Cover
# MAGIC
# MAGIC | Section | What | Why |
# MAGIC |---|---|---|
# MAGIC | **1. Scaling ETL** | 4 approaches to rolling features | Understand when pandas, Spark, or UDFs win |
# MAGIC | **2. Run Many Models** | 3 approaches to OLS per segment | Data-parallel vs sequential vs built-in |
# MAGIC | **3. For-Loop Anti-Patterns** | `withColumn` loop vs `select()` | The #1 Spark performance trap |
# MAGIC | **4. Decision Framework** | Summary table | Quick reference for production choices |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import time
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
from contextlib import contextmanager

# ─── Configuration ────────────────────────────────────────────────────────────
dbutils.widgets.text("catalog", "my_catalog",         "UC Catalog")
dbutils.widgets.text("schema",  "actuarial_workshop", "UC Schema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")

# ─── Timer helper ─────────────────────────────────────────────────────────────
@contextmanager
def timed(label: str):
    """Context manager that prints elapsed wall-clock time for a block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"⏱  {label}: {elapsed:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Gold Data from Module 1

# COMMAND ----------

# Load segment data from Module 1's Gold table; fall back to synthetic data if unavailable.
try:
    _gold_tbl = f"{CATALOG}.{SCHEMA}.gold_claims_monthly"
    gold_df = spark.table(_gold_tbl)
    _count = gold_df.count()
    print(f"Loaded {_gold_tbl}: {_count:,} rows")
    USE_GOLD = True
except Exception as _e:
    print(f"Gold table not available ({_e!r}) — using synthetic fallback")
    USE_GOLD = False

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

display(gold_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1: Scaling ETL — Rolling Features
# MAGIC
# MAGIC **Task**: Compute rolling window features (3m/6m/12m means, std, MoM/YoY %) for each segment.
# MAGIC This is the exact computation that the DLT pipeline runs in production (`silver_rolling_features`).
# MAGIC
# MAGIC We compare **four approaches** on the real 2,880-row dataset, then scale to ~300K rows
# MAGIC to see where each approach shines.
# MAGIC
# MAGIC | # | Approach | Description |
# MAGIC |---|---|---|
# MAGIC | A | Plain pandas | `toPandas()` → `groupby().rolling()` on the driver |
# MAGIC | B | Native Spark | `Window.partitionBy().orderBy().rowsBetween()` — pure Spark SQL |
# MAGIC | C | Pandas API on Spark | `ps.DataFrame` + `groupby().rolling()` — pandas syntax on Spark |
# MAGIC | D | `applyInPandas` | `groupBy().applyInPandas(fn, schema)` — pandas UDF per group |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Approach A: Plain Pandas (baseline — collect to driver)

# COMMAND ----------

def compute_rolling_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    """Standard pandas rolling features — works but runs on a single core."""
    result_frames = []
    for seg_id, group in pdf.groupby("segment_id"):
        g = group.sort_values("month").copy()
        claims = g["claims_count"].astype(float)
        g["rolling_3m_mean"]  = claims.rolling(3, min_periods=1).mean()
        g["rolling_6m_mean"]  = claims.rolling(6, min_periods=1).mean()
        g["rolling_12m_mean"] = claims.rolling(12, min_periods=1).mean()
        g["rolling_3m_std"]   = claims.rolling(3, min_periods=1).std().fillna(0)
        g["mom_change_pct"]   = claims.pct_change().fillna(0) * 100
        g["yoy_change_pct"]   = claims.pct_change(12).fillna(0) * 100
        result_frames.append(g)
    return pd.concat(result_frames, ignore_index=True)

with timed("Approach A — Plain pandas (2,880 rows)"):
    pdf = gold_df.toPandas()
    result_a = compute_rolling_pandas(pdf)

print(f"  Result: {len(result_a)} rows, {len(result_a.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Approach B: Native Spark Window Functions (production pattern)

# COMMAND ----------

def compute_rolling_spark(df):
    """Pure Spark — scales to any size, optimized by Catalyst, no serialization."""
    seg_w = Window.partitionBy("segment_id").orderBy("month")
    return (
        df
        .withColumn("rolling_3m_mean",  F.avg("claims_count").over(seg_w.rowsBetween(-2, 0)))
        .withColumn("rolling_6m_mean",  F.avg("claims_count").over(seg_w.rowsBetween(-5, 0)))
        .withColumn("rolling_12m_mean", F.avg("claims_count").over(seg_w.rowsBetween(-11, 0)))
        .withColumn("rolling_3m_std",   F.stddev("claims_count").over(seg_w.rowsBetween(-2, 0)))
        .withColumn("_prev",  F.lag("claims_count", 1).over(seg_w))
        .withColumn("_prev12", F.lag("claims_count", 12).over(seg_w))
        .withColumn("mom_change_pct",
            F.when(F.col("_prev").isNotNull() & (F.col("_prev") != 0),
                   (F.col("claims_count") - F.col("_prev")) / F.col("_prev") * 100)
             .otherwise(F.lit(0.0)))
        .withColumn("yoy_change_pct",
            F.when(F.col("_prev12").isNotNull() & (F.col("_prev12") != 0),
                   (F.col("claims_count") - F.col("_prev12")) / F.col("_prev12") * 100)
             .otherwise(F.lit(0.0)))
        .drop("_prev", "_prev12")
    )

with timed("Approach B — Native Spark (2,880 rows)"):
    result_b = compute_rolling_spark(gold_df)
    _count_b = result_b.count()  # force materialization

print(f"  Result: {_count_b} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Approach C: Pandas API on Spark (familiar syntax, distributed execution)

# COMMAND ----------

# Pandas API on Spark — familiar pandas syntax, but executed on Spark behind the scenes.
# Note: .transform(lambda: rolling()) may crash on Serverless; we use a safe subset here.
import pyspark.pandas as ps

with timed("Approach C — Pandas API on Spark (2,880 rows)"):
    ps.set_option("compute.ops_on_diff_frames", True)
    ps_df = gold_df.to_pandas_on_spark().sort_values(["segment_id", "month"])
    # Pandas API on Spark supports rolling on the whole DataFrame (not grouped transform on all runtimes)
    ps_df = ps_df.set_index(["segment_id", "month"]).sort_index()
    # Use a simpler approach: convert to Spark and use window functions
    # (demonstrating that the API parity has limits for grouped rolling)
    ps_result = ps_df.reset_index()
    _count_c = len(ps_result)

print(f"  Result: {_count_c} rows")
print("  Note: Grouped rolling via .transform() has limited support on Serverless.")
print("  In practice, switch to Native Spark (Approach B) for grouped window operations.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Approach D: `applyInPandas` (per-group pandas UDF)

# COMMAND ----------

ROLLING_SCHEMA = StructType(
    gold_df.schema.fields + [
        StructField("rolling_3m_mean",  DoubleType(), True),
        StructField("rolling_6m_mean",  DoubleType(), True),
        StructField("rolling_12m_mean", DoubleType(), True),
        StructField("rolling_3m_std",   DoubleType(), True),
        StructField("mom_change_pct",   DoubleType(), True),
        StructField("yoy_change_pct",   DoubleType(), True),
    ]
)

def rolling_features_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    """Per-group UDF: receives one segment's data as a pandas DataFrame."""
    pdf = pdf.sort_values("month").copy()
    claims = pdf["claims_count"].astype(float)
    pdf["rolling_3m_mean"]  = claims.rolling(3, min_periods=1).mean()
    pdf["rolling_6m_mean"]  = claims.rolling(6, min_periods=1).mean()
    pdf["rolling_12m_mean"] = claims.rolling(12, min_periods=1).mean()
    pdf["rolling_3m_std"]   = claims.rolling(3, min_periods=1).std().fillna(0)
    pdf["mom_change_pct"]   = claims.pct_change().fillna(0) * 100
    pdf["yoy_change_pct"]   = claims.pct_change(12).fillna(0) * 100
    return pdf

with timed("Approach D — applyInPandas (2,880 rows)"):
    result_d = gold_df.groupBy("segment_id").applyInPandas(rolling_features_udf, schema=ROLLING_SCHEMA)
    _count_d = result_d.count()

print(f"  Result: {_count_d} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scale Test: 300K Rows
# MAGIC
# MAGIC At 2,880 rows, pandas wins (no Spark overhead). What happens at 100× scale?

# COMMAND ----------

# Create a ~300K row dataset by replicating the Gold data 100×
# Each replica gets a suffix so segment_ids remain unique within replicas
replicas = []
for i in range(100):
    replicas.append(
        gold_df.withColumn("segment_id",
            F.concat(F.col("segment_id"), F.lit(f"_r{i:03d}")))
    )
from functools import reduce
large_df = reduce(lambda a, b: a.unionAll(b), replicas)
large_count = large_df.count()
print(f"Scaled dataset: {large_count:,} rows ({large_df.select('segment_id').distinct().count()} segments)")

# COMMAND ----------

with timed("Approach A — Plain pandas (300K rows)"):
    pdf_large = large_df.toPandas()
    result_a_large = compute_rolling_pandas(pdf_large)
print(f"  Result: {len(result_a_large)} rows")

# COMMAND ----------

with timed("Approach B — Native Spark (300K rows)"):
    result_b_large = compute_rolling_spark(large_df)
    _count_b_large = result_b_large.count()
print(f"  Result: {_count_b_large} rows")

# COMMAND ----------

with timed("Approach D — applyInPandas (300K rows)"):
    result_d_large = large_df.groupBy("segment_id").applyInPandas(rolling_features_udf, schema=ROLLING_SCHEMA)
    _count_d_large = result_d_large.count()
print(f"  Result: {_count_d_large} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 1 Summary
# MAGIC
# MAGIC | Rows | Plain Pandas | Native Spark | Pandas API on Spark | applyInPandas |
# MAGIC |------|-------------|-------------|---------------------|---------------|
# MAGIC | 2,880 | Fastest (no overhead) | Spark startup cost | Limited grouped ops | UDF overhead |
# MAGIC | 300K | **Slows down** (single core) | **Scales flat** | — | **Scales well** (parallel groups) |
# MAGIC
# MAGIC **Takeaway**: At small scale, pandas is fine. At production scale, **Native Spark window functions**
# MAGIC are the clear winner for ETL transforms — no serialization, fully optimized by Catalyst.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2: Run Many Models — OLS Trend per Segment
# MAGIC
# MAGIC **Task**: Fit a simple OLS trend (`claims_count ~ time_index`) per segment (40 segments).
# MAGIC
# MAGIC | # | Approach | Description |
# MAGIC |---|---|---|
# MAGIC | A | Plain pandas for-loop | `toPandas()`, loop over segments, `np.polyfit()` |
# MAGIC | B | `applyInPandas` | `groupBy("segment_id").applyInPandas(ols_fn, schema)` |
# MAGIC | C | Native Spark built-ins | `F.regr_slope()`, `F.regr_intercept()` — pure SQL |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Approach A: Pandas For-Loop (sequential on driver)

# COMMAND ----------

with timed("Approach A — Pandas for-loop OLS (40 segments)"):
    pdf_gold = gold_df.toPandas()
    ols_results_a = []
    for seg_id, group in pdf_gold.groupby("segment_id"):
        group = group.sort_values("month")
        y = group["claims_count"].astype(float).values
        t = np.arange(len(y), dtype=float)
        if len(y) > 1:
            slope, intercept = np.polyfit(t, y, 1)
            fitted = intercept + slope * t
            ss_res = np.sum((y - fitted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            slope = intercept = r2 = 0.0
        ols_results_a.append({
            "segment_id": seg_id,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r2,
            "annualized_growth_pct": (slope * 12 / np.mean(y)) * 100 if np.mean(y) > 0 else 0.0,
        })
    ols_df_a = pd.DataFrame(ols_results_a)

print(f"  Fitted {len(ols_df_a)} segments")
display(spark.createDataFrame(ols_df_a).orderBy(F.col("annualized_growth_pct").desc()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Approach B: `applyInPandas` (data-parallel on Spark workers)

# COMMAND ----------

OLS_SCHEMA = StructType([
    StructField("segment_id",            StringType(), False),
    StructField("slope",                 DoubleType(), True),
    StructField("intercept",             DoubleType(), True),
    StructField("r_squared",             DoubleType(), True),
    StructField("annualized_growth_pct", DoubleType(), True),
])

def ols_per_segment(pdf: pd.DataFrame) -> pd.DataFrame:
    """Fit OLS trend per segment — runs in a Spark task, receives plain pandas."""
    seg_id = pdf["segment_id"].iloc[0]
    pdf = pdf.sort_values("month")
    y = pdf["claims_count"].astype(float).values
    t = np.arange(len(y), dtype=float)

    if len(y) > 1:
        slope, intercept = np.polyfit(t, y, 1)
        fitted = intercept + slope * t
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope = intercept = r2 = 0.0

    return pd.DataFrame([{
        "segment_id": seg_id,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "annualized_growth_pct": float((slope * 12 / np.mean(y)) * 100) if np.mean(y) > 0 else 0.0,
    }])

with timed("Approach B — applyInPandas OLS (40 segments)"):
    ols_result_b = (
        gold_df
        .select("segment_id", "month", "claims_count")
        .groupBy("segment_id")
        .applyInPandas(ols_per_segment, schema=OLS_SCHEMA)
    )
    _count_ols_b = ols_result_b.count()

print(f"  Fitted {_count_ols_b} segments")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Approach C: Spark Built-in Analytical Functions (no UDF)

# COMMAND ----------

# Spark has built-in regression functions: regr_slope, regr_intercept, regr_r2
# These run entirely inside the Catalyst engine — no Python serialization at all.

with timed("Approach C — Spark regr_slope/regr_intercept (40 segments)"):
    # Create a numeric time index per segment
    seg_w = Window.partitionBy("segment_id").orderBy("month")
    gold_with_idx = gold_df.withColumn("time_idx", (F.row_number().over(seg_w) - 1).cast("double"))

    ols_result_c = (
        gold_with_idx
        .groupBy("segment_id")
        .agg(
            F.regr_slope("claims_count", "time_idx").alias("slope"),
            F.regr_intercept("claims_count", "time_idx").alias("intercept"),
            F.regr_r2("claims_count", "time_idx").alias("r_squared"),
            F.avg("claims_count").alias("avg_claims"),
        )
        .withColumn("annualized_growth_pct",
            F.when(F.col("avg_claims") > 0,
                   F.col("slope") * 12 / F.col("avg_claims") * 100)
             .otherwise(F.lit(0.0)))
        .drop("avg_claims")
    )
    _count_ols_c = ols_result_c.count()

print(f"  Fitted {_count_ols_c} segments")
display(ols_result_c.orderBy(F.col("annualized_growth_pct").desc()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section 2 Summary
# MAGIC
# MAGIC | Approach | How it works | Best for |
# MAGIC |---|---|---|
# MAGIC | **Pandas for-loop** | Sequential on driver | Prototyping, <10 groups |
# MAGIC | **`applyInPandas`** | Parallel on Spark workers | Arbitrary Python per group (statsmodels, scipy) |
# MAGIC | **Spark built-ins** | Pure SQL engine, no UDF | Simple aggregates (`regr_slope`, window fns) |
# MAGIC
# MAGIC **Takeaway**: For simple models (OLS, averages), Spark built-ins are fastest.
# MAGIC For complex models (SARIMA, GARCH — see Module 4), `applyInPandas` is the way to go.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 3: For-Loop Anti-Patterns on Spark
# MAGIC
# MAGIC This is the **#1 performance trap** for teams migrating from pandas to Spark.
# MAGIC
# MAGIC ### The Trap
# MAGIC
# MAGIC ```python
# MAGIC # DON'T DO THIS — each withColumn adds a node to the Catalyst logical plan
# MAGIC for i in range(N):
# MAGIC     df = df.withColumn(f"col_{i}", F.col("base") * F.lit(i))
# MAGIC ```
# MAGIC
# MAGIC Catalyst plan resolution is **O(N²)** in the number of columns. At N=200 it's slow;
# MAGIC at N=400 it's painful; at N=600 it may hang.
# MAGIC
# MAGIC ### The Fix
# MAGIC
# MAGIC ```python
# MAGIC # DO THIS — single select() builds one plan node
# MAGIC df = df.select(
# MAGIC     "*",
# MAGIC     *[(F.col("base") * F.lit(i)).alias(f"col_{i}") for i in range(N)]
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demo: 400 Columns — For-Loop vs Select

# COMMAND ----------

# Generate a base DataFrame (100K rows, 1 column)
base_df = spark.range(100_000).withColumn("base", (F.col("id") % 100).cast("double"))

N_COLS = 400

# COMMAND ----------

# ── Anti-pattern: withColumn in a for-loop ────────────────────────────────────
with timed(f"FOR LOOP — {N_COLS} withColumn calls"):
    df_loop = base_df
    for i in range(N_COLS):
        df_loop = df_loop.withColumn(f"col_{i}", F.col("base") * F.lit(float(i)))
    _loop_count = df_loop.count()  # force materialization

print(f"  Result: {_loop_count:,} rows × {len(df_loop.columns)} columns")

# COMMAND ----------

# ── Correct pattern: select() with list comprehension ─────────────────────────
with timed(f"SELECT — {N_COLS} columns via list comprehension"):
    df_select = base_df.select(
        "*",
        *[(F.col("base") * F.lit(float(i))).alias(f"col_{i}") for i in range(N_COLS)]
    )
    _select_count = df_select.count()

print(f"  Result: {_select_count:,} rows × {len(df_select.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demo: 600 Columns — Where For-Loops Break Down

# COMMAND ----------

N_COLS_LARGE = 600

with timed(f"SELECT — {N_COLS_LARGE} columns via list comprehension"):
    df_select_large = base_df.select(
        "*",
        *[(F.col("base") * F.lit(float(i))).alias(f"col_{i}") for i in range(N_COLS_LARGE)]
    )
    _select_large_count = df_select_large.count()

print(f"  Result: {_select_large_count:,} rows × {len(df_select_large.columns)} columns")
print(f"\n  (The for-loop version at {N_COLS_LARGE} columns would take several minutes or hang.)")
print(f"  Try it yourself if you're curious — but be prepared to cancel the cell!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Rule
# MAGIC
# MAGIC > **Never iterate `withColumn` in a for-loop.**
# MAGIC > Use `select()` with a list comprehension, or `functools.reduce()` for chained transforms.
# MAGIC
# MAGIC ```python
# MAGIC # Pattern 1: select() with list comprehension (preferred)
# MAGIC df.select("*", *[expr.alias(name) for expr, name in column_specs])
# MAGIC
# MAGIC # Pattern 2: functools.reduce for conditional chains
# MAGIC from functools import reduce
# MAGIC df = reduce(lambda d, spec: d.withColumn(spec[0], spec[1]), column_specs, df)
# MAGIC # (Note: reduce still builds O(N) plan nodes — select() is better for large N)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 4: Decision Framework
# MAGIC
# MAGIC | Workload | Best Pattern | Why |
# MAGIC |---|---|---|
# MAGIC | **ETL transforms** (rolling windows, joins, aggregations) | Native Spark | Scales, Catalyst-optimized, no serialization |
# MAGIC | **Per-group modeling** (SARIMA, GARCH, custom Python) | `applyInPandas` | Arbitrary Python per group, data-parallel on workers |
# MAGIC | **Simple aggregates** (OLS, correlation, variance) | Spark built-in functions | `regr_slope`, `corr`, `stddev` — no UDF overhead |
# MAGIC | **Explore / prototype** | Pandas API on Spark | Familiar pandas syntax, easy migration path |
# MAGIC | **Heavy task-parallel compute** (Monte Carlo, grid search) | Ray (see Module 4) | Independent Python tasks, GPU optional |
# MAGIC | **Multi-column generation** | `select()` + list comprehension | Avoids O(N²) Catalyst plan resolution |
# MAGIC
# MAGIC **Next:** Module 3 — with reliable, scaled Gold data and rolling features from the DLT pipeline,
# MAGIC we register them in the Unity Catalog Feature Store with point-in-time correctness.
# MAGIC The features from `silver_rolling_features` feed Module 4's SARIMAX models as exogenous variables.
