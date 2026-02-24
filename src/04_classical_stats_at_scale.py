# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: Classical Statistical Models at Scale
# MAGIC ## SARIMA/GARCH per Segment + Monte Carlo with Ray
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC *Audience: Actuaries, Data Scientists, Financial Analysts*
# MAGIC
# MAGIC ---
# MAGIC ### What We'll Cover
# MAGIC 1. **Data Setup** — Synthetic monthly claims time series (20 product × region segments)
# MAGIC 2. **SARIMA at Scale** — Per-segment forecasting with `applyInPandas` + statsmodels
# MAGIC 3. **GARCH at Scale** — Per-segment volatility modeling with the `arch` library
# MAGIC 4. **Monte Carlo with Ray** — Task-parallel portfolio loss simulation (100k+ paths)
# MAGIC 5. **MLflow Integration** — Experiment tracking, metrics, artifacts
# MAGIC
# MAGIC ---
# MAGIC ### Why `applyInPandas`?
# MAGIC
# MAGIC We have **20 segments**, each needing its own ARIMA/GARCH fit. These are **independent** per-group operations.
# MAGIC `applyInPandas` lets Spark distribute this work: each executor receives a pandas DataFrame for one segment,
# MAGIC runs standard Python/statsmodels code, and returns results. No Spark ML required — just familiar Python libraries.
# MAGIC
# MAGIC ```
# MAGIC df.groupby("segment_id").applyInPandas(fit_sarima_fn, schema=output_schema)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install Required Libraries
# MAGIC
# MAGIC These libraries are available in Databricks Runtime ML. On standard DBR, install via the cluster library UI
# MAGIC or with `%pip install` below.

# COMMAND ----------

# On Serverless (Spark Connect), %pip install ensures packages are available on
# both the driver and the compute environment where applyInPandas UDFs execute.
# The ml_env spec in resources/jobs.yml only covers the driver side.
%pip install statsmodels arch --quiet

# COMMAND ----------

# ── GPU detection ─────────────────────────────────────────────────────────────
# PyTorch (pre-installed on DBR GPU ML) is used for GPU Monte Carlo simulation.
# No extra pip install needed — torch.cuda is available on all GPU ML nodes.
import subprocess as _subprocess
_gpu_check = _subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True, timeout=5,
)
HAS_GPU = _gpu_check.returncode == 0 and bool(_gpu_check.stdout.strip())
if HAS_GPU:
    print(f"GPU detected on driver: {_gpu_check.stdout.strip().splitlines()[0]}")
else:
    print("No GPU on driver (normal for multi-node clusters) — Ray workers will use torch.cuda")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Insurance Claims Time Series
# MAGIC
# MAGIC We simulate **60 months** (5 years) of monthly claims for **20 segments** (4 product lines × 5 regions).
# MAGIC Each segment has its own:
# MAGIC - **Level** (expected monthly claims)
# MAGIC - **Seasonality** (winter peak — higher claims in Q1)
# MAGIC - **Trend** (slow upward drift in loss ratios)
# MAGIC - **Volatility** (GARCH-style clustering)

# COMMAND ----------

import numpy as np
import pandas as pd
from itertools import product as iterproduct
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# ─── Configuration ────────────────────────────────────────────────────────────
# Passed as base_parameters by the bundle job; defaults used when interactive.
dbutils.widgets.text("catalog",  "my_catalog",         "UC Catalog")
dbutils.widgets.text("schema",   "actuarial_workshop", "UC Schema")
dbutils.widgets.text("run_ray",  "skip",               "Run Ray (auto/skip)")
# job_mode=true: use driver-side pandas groupby.apply() for SARIMA/GARCH instead of
# applyInPandas. On Serverless (Spark Connect), applyInPandas UDFs run on the remote
# compute which has a separate Python environment — packages installed via %pip only
# reach the notebook kernel (client side), not the remote compute (server side).
# Driver-side fitting produces identical outputs and works reliably on all runtimes.
dbutils.widgets.text("job_mode", "false",              "Job (automated) mode")
CATALOG   = dbutils.widgets.get("catalog")
SCHEMA    = dbutils.widgets.get("schema")
# run_ray="skip" disables Ray setup entirely, using the single-node NumPy fallback.
# Pass run_ray=skip from the bundle job on Serverless where setup_ray_cluster is
# not supported (Spark Connect does not expose the Ray-on-Spark RPC interface).
RUN_RAY   = dbutils.widgets.get("run_ray")
JOB_MODE  = dbutils.widgets.get("job_mode") == "true"

np.random.seed(42)

# ─── Segment definitions ─────────────────────────────────────────────────────
PRODUCT_LINES = ["Personal_Auto", "Commercial_Auto", "Homeowners", "Commercial_Property"]
REGIONS       = [
    "Ontario", "Quebec", "British_Columbia", "Alberta", "Atlantic",
    "Manitoba", "Saskatchewan", "Nova_Scotia", "New_Brunswick", "Newfoundland",
    "Prince_Edward_Island", "Northwest_Territories", "Yukon",
]
MONTHS        = pd.date_range("2019-01-01", periods=72, freq="MS")  # Jan 2019 – Dec 2024

# Base claim levels (expected monthly claims per segment)
BASE_CLAIMS = {
    "Personal_Auto":         450,
    "Commercial_Auto":       180,
    "Homeowners":            320,
    "Commercial_Property":   90,
}

REGION_MULTIPLIER = {
    "Ontario":              1.40,
    "Quebec":               1.10,
    "British_Columbia":     1.20,
    "Alberta":              1.00,
    "Atlantic":             0.70,   # composite Atlantic region
    "Manitoba":             0.85,
    "Saskatchewan":         0.80,
    "Nova_Scotia":          0.75,
    "New_Brunswick":        0.70,
    "Newfoundland":         0.65,
    "Prince_Edward_Island": 0.60,
    "Northwest_Territories":0.55,
    "Yukon":                0.50,
}

# Winter seasonality factors by month (1=Jan, 12=Dec)
SEASONALITY = {1: 1.25, 2: 1.20, 3: 1.10, 4: 0.95, 5: 0.90, 6: 0.88,
               7: 0.85, 8: 0.87, 9: 0.92, 10: 1.00, 11: 1.10, 12: 1.20}

rows = []
for (prod, region), seg_idx in zip(
    iterproduct(PRODUCT_LINES, REGIONS),
    range(len(PRODUCT_LINES) * len(REGIONS))
):
    segment_id = f"{prod}__{region}"
    base = BASE_CLAIMS[prod] * REGION_MULTIPLIER[region]

    # Simulate GARCH-like volatility clustering (one innovation per month in MONTHS)
    vol = 0.08  # base volatility
    innovations = []
    h = vol ** 2
    for _ in range(len(MONTHS)):
        e = np.random.normal(0, np.sqrt(h))
        innovations.append(e)
        h = 0.02 + 0.15 * e**2 + 0.80 * h  # GARCH(1,1) parameters

    for i, (month, innov) in enumerate(zip(MONTHS, innovations)):
        trend    = 1.0 + 0.003 * i            # 0.3% monthly upward trend in loss ratio
        seasonal = SEASONALITY[month.month]
        claims   = max(0, base * trend * seasonal * (1 + innov))

        rows.append({
            "segment_id":   segment_id,
            "product_line": prod,
            "region":       region,
            "month":        month.date(),
            "claims_count": int(round(claims)),
            "earned_premium": round(base * trend * seasonal * np.random.uniform(3.2, 3.8), 2),
        })

pdf = pd.DataFrame(rows)
pdf["loss_ratio"] = pdf["claims_count"] / pdf["earned_premium"]

# Convert to Spark DataFrame
schema = StructType([
    StructField("segment_id",     StringType(),  False),
    StructField("product_line",   StringType(),  False),
    StructField("region",         StringType(),  False),
    StructField("month",          DateType(),    False),
    StructField("claims_count",   IntegerType(), False),
    StructField("earned_premium", DoubleType(),  False),
    StructField("loss_ratio",     DoubleType(),  False),
])

claims_df = spark.createDataFrame(pdf, schema=schema)
claims_df.createOrReplaceTempView("claims_ts")

print(f"Segments: {claims_df.select('segment_id').distinct().count()}")
print(f"Rows:     {claims_df.count()}")
display(claims_df.orderBy("segment_id", "month").limit(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Persist to Unity Catalog (Silver Layer)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

(claims_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.claims_time_series"))

print(f"Saved to {CATALOG}.{SCHEMA}.claims_time_series")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SARIMA at Scale — Per-Segment Forecasting
# MAGIC
# MAGIC ### Strategy
# MAGIC We use `df.groupby("segment_id").applyInPandas(fit_fn, schema)`.
# MAGIC
# MAGIC Each Spark task receives **one segment's pandas DataFrame** and:
# MAGIC 1. Fits `statsmodels.SARIMAX` with seasonal order `(1,0,1)(1,1,0,12)` — monthly seasonality
# MAGIC 2. Produces a 12-month forecast with confidence intervals
# MAGIC 3. Returns a standardized pandas DataFrame
# MAGIC
# MAGIC The output schema must be declared upfront so Spark can parallelize safely.

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, DoubleType, IntegerType, LongType
)
import pyspark.sql.functions as F

# Output schema for SARIMA results
SARIMA_SCHEMA = StructType([
    StructField("segment_id",   StringType(),  False),
    StructField("month",        DateType(),    False),
    StructField("record_type",  StringType(),  False),   # "actual" or "forecast"
    StructField("claims_count", DoubleType(),  True),
    StructField("forecast_mean",DoubleType(),  True),
    StructField("forecast_lo95",DoubleType(),  True),
    StructField("forecast_hi95",DoubleType(),  True),
    StructField("aic",          DoubleType(),  True),
    StructField("mape",         DoubleType(),  True),
])

def fit_sarima_per_segment(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit SARIMAX(1,0,1)(1,1,0,12) for one segment.
    Called by Spark for each group — standard pandas/statsmodels code inside.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    warnings.filterwarnings("ignore")

    segment_id = pdf["segment_id"].iloc[0]
    pdf = pdf.sort_values("month").reset_index(drop=True)

    y = pdf["claims_count"].astype(float).values
    months = pd.to_datetime(pdf["month"])

    # ── Fit SARIMAX ──────────────────────────────────────────────────────────
    try:
        model = SARIMAX(
            y,
            order=(1, 0, 1),
            seasonal_order=(1, 1, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=100)
        aic = fit.aic

        # In-sample MAPE (exclude first 12 months for seasonal differencing warmup)
        fitted_vals = fit.fittedvalues[12:]
        actual_vals = y[12:]
        mape = np.mean(np.abs((actual_vals - fitted_vals) / np.clip(actual_vals, 1, None))) * 100

        # 12-month forecast
        forecast = fit.get_forecast(steps=12)
        fcast_mean = forecast.predicted_mean
        fcast_ci_raw = forecast.conf_int(alpha=0.05)
        # Normalize to DataFrame regardless of statsmodels version
        if hasattr(fcast_ci_raw, 'iloc'):
            fcast_ci = fcast_ci_raw
        else:
            fcast_ci = pd.DataFrame(fcast_ci_raw, columns=["lower", "upper"])

    except Exception as e:
        # On failure, return NaN forecasts — allows the pipeline to continue
        fcast_mean = np.full(12, np.nan)
        fcast_ci   = pd.DataFrame({"lower claims_count": np.full(12, np.nan),
                                   "upper claims_count": np.full(12, np.nan)})
        aic, mape = np.nan, np.nan

    # ── Build output DataFrame ────────────────────────────────────────────────
    last_month = months.max()
    forecast_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=12, freq="MS")

    actuals_rows = pd.DataFrame({
        "segment_id":    segment_id,
        "month":         months.dt.date,
        "record_type":   "actual",
        "claims_count":  y.tolist(),
        "forecast_mean": [None] * len(y),
        "forecast_lo95": [None] * len(y),
        "forecast_hi95": [None] * len(y),
        "aic":           [None] * len(y),
        "mape":          [None] * len(y),
    })

    forecast_rows = pd.DataFrame({
        "segment_id":    segment_id,
        "month":         forecast_months.date,
        "record_type":   "forecast",
        "claims_count":  [None] * 12,
        "forecast_mean": list(fcast_mean),
        "forecast_lo95": list(np.asarray(fcast_ci)[:, 0]),
        "forecast_hi95": list(np.asarray(fcast_ci)[:, 1]),
        "aic":           [aic] * 12,
        "mape":          [mape] * 12,
    })

    return pd.concat([actuals_rows, forecast_rows], ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run SARIMA Across All 20 Segments
# MAGIC
# MAGIC Spark distributes one segment per task. Each task runs pure Python/statsmodels independently.

# COMMAND ----------

import mlflow
_current_user = spark.sql("SELECT current_user()").collect()[0][0]
# Use a flat path under /Users/<email>/ — avoid nested subdirectories which
# require the parent to pre-exist (fails on fresh workspaces).
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_claims_sarima")

# Register input dataset for UC lineage — all three models (SARIMA, GARCH, Monte Carlo)
# are trained from the same claims_time_series source table.  mlflow.log_input()
# creates a lineage edge visible in the UC Explorer → Lineage tab.
_claims_dataset = mlflow.data.load_delta(
    table_name=f"{CATALOG}.{SCHEMA}.claims_time_series",
    name="claims_time_series",
)

with mlflow.start_run(run_name="sarima_all_segments") as run:
    mlflow.log_input(_claims_dataset, context="training")
    mlflow.set_tags({
        "workshop_module": "4",
        "model_type":      "SARIMAX(1,0,1)(1,1,0,12)",
        "segments":        "20",
        "horizon_months":  "12",
        "audience":        "actuarial-workshop",
    })

    if JOB_MODE:
        # Driver-side: collect to pandas, fit in a Python loop.
        # Avoids applyInPandas which requires statsmodels on the remote Serverless
        # compute — a separate Python environment that %pip install cannot reach.
        print("job_mode=true: using driver-side pandas groupby.apply() for SARIMA")
        claims_pdf = claims_df.toPandas()
        sarima_result_pdf = (
            claims_pdf
            .groupby("segment_id", group_keys=False)
            .apply(fit_sarima_per_segment)
            .reset_index(drop=True)
        )
        sarima_results_df = spark.createDataFrame(sarima_result_pdf, schema=SARIMA_SCHEMA)
    else:
        # Distributed: Spark sends one segment per task to applyInPandas workers.
        # Requires statsmodels on the compute environment (ml_env spec in bundle jobs).
        sarima_results_df = (
            claims_df
            .groupby("segment_id")
            .applyInPandas(fit_sarima_per_segment, schema=SARIMA_SCHEMA)
        )

    # Trigger execution and compute metrics
    total_rows = sarima_results_df.count()

    # Compute average MAPE across segments
    avg_mape = (
        sarima_results_df
        .filter(F.col("record_type") == "forecast")
        .agg(F.mean("mape").alias("avg_mape"))
        .collect()[0]["avg_mape"]
    )

    mlflow.log_metric("avg_mape_pct", round(avg_mape, 2))
    mlflow.log_metric("total_output_rows", total_rows)
    mlflow.log_metric("segments_fitted", 20)

    print(f"SARIMA complete | Total rows: {total_rows} | Avg MAPE: {avg_mape:.1f}%")
    print(f"MLflow run: {run.info.run_id}")

if not JOB_MODE:
    display(sarima_results_df.filter(F.col("record_type") == "forecast").orderBy("segment_id", "month").limit(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save SARIMA Results

# COMMAND ----------

(sarima_results_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.sarima_forecasts"))

print(f"Saved to {CATALOG}.{SCHEMA}.sarima_forecasts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. GARCH at Scale — Volatility Modeling
# MAGIC
# MAGIC GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models **variance clustering** in loss ratios —
# MAGIC the tendency for volatile periods to cluster together. This is critical for:
# MAGIC - **Risk capital** calculations (Solvency II internal models)
# MAGIC - **Reinsurance pricing** (tail risk quantification)
# MAGIC - **Premium rate adequacy** stress testing
# MAGIC
# MAGIC We fit a **GARCH(1,1)** to the loss ratio of each segment and extract conditional volatility forecasts.

# COMMAND ----------

GARCH_SCHEMA = StructType([
    StructField("segment_id",          StringType(), False),
    StructField("month",               DateType(),   False),
    StructField("record_type",         StringType(), False),
    StructField("loss_ratio",          DoubleType(), True),
    StructField("cond_volatility",     DoubleType(), True),
    StructField("forecast_vol_1m",     DoubleType(), True),
    StructField("forecast_vol_12m",    DoubleType(), True),
    StructField("omega",               DoubleType(), True),
    StructField("alpha",               DoubleType(), True),
    StructField("beta",                DoubleType(), True),
    StructField("log_likelihood",      DoubleType(), True),
])

def fit_garch_per_segment(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit GARCH(1,1) to loss ratio returns for one segment.
    Uses the `arch` library — standard actuarial Python tooling.
    """
    from arch import arch_model
    import warnings
    warnings.filterwarnings("ignore")

    segment_id = pdf["segment_id"].iloc[0]
    pdf = pdf.sort_values("month").reset_index(drop=True)

    lr = pdf["loss_ratio"].astype(float).values
    months = pd.to_datetime(pdf["month"])

    # Work on log-returns of loss ratio (stationarity)
    lr_returns = np.diff(np.log(np.clip(lr, 0.01, None))) * 100  # in pct

    result_rows = []
    omega_val = alpha_val = beta_val = ll_val = np.nan
    cond_vols = [np.nan] + [np.nan] * len(lr_returns)

    try:
        am = arch_model(lr_returns, vol="Garch", p=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)

        omega_val = float(res.params["omega"])
        alpha_val = float(res.params["alpha[1]"])
        beta_val  = float(res.params["beta[1]"])
        ll_val    = float(res.loglikelihood)

        # Conditional volatility (annualized, scaled to loss ratio units)
        cond_vols = [np.nan] + list(res.conditional_volatility)

        # Forecast: 1-month and 12-month ahead volatility
        fc = res.forecast(horizon=12, reindex=False)
        # Handle both DataFrame and ndarray variance output across arch versions
        var_arr = np.asarray(fc.variance)
        vol_1m  = float(np.sqrt(var_arr.flat[0]))
        vol_12m = float(np.sqrt(var_arr.flat[11] if var_arr.size > 11 else var_arr.flat[-1]))

    except Exception:
        vol_1m = vol_12m = np.nan

    for i, (m, lr_val, cv) in enumerate(zip(months, lr, cond_vols)):
        result_rows.append({
            "segment_id":       segment_id,
            "month":            m.date(),
            "record_type":      "actual",
            "loss_ratio":       float(lr_val),
            "cond_volatility":  float(cv) if not np.isnan(cv) else None,
            "forecast_vol_1m":  vol_1m if i == len(months) - 1 else None,
            "forecast_vol_12m": vol_12m if i == len(months) - 1 else None,
            "omega":            omega_val,
            "alpha":            alpha_val,
            "beta":             beta_val,
            "log_likelihood":   ll_val,
        })

    return pd.DataFrame(result_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run GARCH Across All Segments

# COMMAND ----------

with mlflow.start_run(run_name="garch_all_segments") as run:
    mlflow.log_input(_claims_dataset, context="training")
    n_segments = claims_df.select("segment_id").distinct().count()
    mlflow.set_tags({
        "workshop_module": "4",
        "model_type":      "GARCH(1,1)",
        "segments":        str(n_segments),
        "audience":        "actuarial-workshop",
    })

    if JOB_MODE:
        print("job_mode=true: using driver-side pandas groupby.apply() for GARCH")
        garch_result_pdf = (
            claims_df.toPandas()
            .groupby("segment_id", group_keys=False)
            .apply(fit_garch_per_segment)
            .reset_index(drop=True)
        )
        garch_results_df = spark.createDataFrame(garch_result_pdf, schema=GARCH_SCHEMA)
    else:
        garch_results_df = (
            claims_df
            .groupby("segment_id")
            .applyInPandas(fit_garch_per_segment, schema=GARCH_SCHEMA)
        )

    garch_count = garch_results_df.count()

    # Average persistence (alpha + beta) — key GARCH risk metric; close to 1 = long memory
    avg_persistence = (
        garch_results_df
        .filter(F.col("record_type") == "actual")
        .agg(F.mean((F.col("alpha") + F.col("beta"))).alias("avg_persistence"))
        .collect()[0]["avg_persistence"]
    )

    mlflow.log_metric("avg_garch_persistence", round(avg_persistence, 4))
    mlflow.log_metric("segments_fitted", n_segments)
    print(f"GARCH complete | Rows: {garch_count} | Segments: {n_segments} | Avg persistence (α+β): {avg_persistence:.4f}")

if not JOB_MODE:
    display(garch_results_df.orderBy("segment_id", "month").limit(30))

# COMMAND ----------

(garch_results_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.garch_volatility"))

print(f"Saved to {CATALOG}.{SCHEMA}.garch_volatility")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Monte Carlo Portfolio Simulation with Ray
# MAGIC
# MAGIC ### Why Ray (Not Spark) for This?
# MAGIC
# MAGIC Each Monte Carlo trial is an **independent Python computation** — no data-parallel transformation over rows.
# MAGIC This is the textbook case for **task parallelism**:
# MAGIC
# MAGIC | Characteristic | Spark | Ray |
# MAGIC |---|---|---|
# MAGIC | Parallelism model | Data-parallel | Task-parallel |
# MAGIC | Scheduling unit | Partition of rows | Arbitrary Python function |
# MAGIC | Best for | Same op over large dataset | Many independent computations |
# MAGIC | Monte Carlo fit | Poor | **Excellent** |
# MAGIC
# MAGIC We simulate a **3-segment correlated portfolio** (Property, Auto, Liability):
# MAGIC - **100,000 annual loss scenarios** using a **t-Copula + lognormal marginals** (see below)
# MAGIC - Each Ray task handles **1,000 scenarios** → 100 tasks run in parallel on GPU workers
# MAGIC - Aggregate: VaR(99.5%), CVaR, Expected Loss — standard Solvency II / IFRS 17 metrics
# MAGIC
# MAGIC ### Why t-Copula, Not Gaussian Copula?
# MAGIC
# MAGIC Insurance portfolios are exposed to **common shocks** — catastrophic events, judicial trends,
# MAGIC or macro-economic shocks (e.g. inflation) that affect multiple lines of business simultaneously.
# MAGIC These common shocks create **tail dependence**: the probability that multiple lines experience
# MAGIC extreme losses *at the same time* is higher than average correlations suggest.
# MAGIC
# MAGIC | Copula | Tail Dependence | Best For |
# MAGIC |---|---|---|
# MAGIC | **Gaussian** | None (symmetric, thin tails) | Normal business conditions |
# MAGIC | **t-Copula (df=4)** | Yes — joint extremes are more likely | Insurance with common-shock risk |
# MAGIC | **Vine/Clayton** | Asymmetric (lower/upper tail) | Reinsurance treaty modelling |
# MAGIC
# MAGIC The **Gaussian copula understates risk** (VaR is lower) because it treats severe Property
# MAGIC and Auto losses as roughly independent at extremes. The **t-Copula** (Student-t, df=4) captures
# MAGIC the heavier joint tails expected from catastrophe events, inflation shocks, and judicial trends.
# MAGIC
# MAGIC **Implementation:** t-Copula = correlated t-distributed variables (via Cholesky + chi² scaling)
# MAGIC → apply t-CDF → get uniform marginals → apply lognormal inverse CDF (Sklar's theorem).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Ray on Spark
# MAGIC
# MAGIC `ray.util.spark.setup_ray_cluster` launches Ray workers on top of the existing Spark cluster.
# MAGIC Ray and Spark share the same compute but operate independently.

# COMMAND ----------

if RUN_RAY == "skip":
    # Explicitly disabled — used in bundle jobs on Serverless (Spark Connect does
    # not support setup_ray_cluster; attempting it kills the Python process silently).
    print("Ray skipped (run_ray=skip). Using single-node NumPy fallback for Monte Carlo.")
    RAY_AVAILABLE = False
else:
    try:
        import ray
        from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
        import ray.util.spark

        # Shut down any existing Ray cluster before starting fresh
        try:
            shutdown_ray_cluster()
        except Exception:
            pass

        # Prevent Spark from reserving GPUs for its own tasks so Ray can use them all.
        spark.conf.set("spark.task.resource.gpu.amount", "0")

        # Reserve 2 vCPUs per worker for Spark so that createDataFrame/saveAsTable
        # can schedule tasks even while Ray is running. Without this, Ray occupies
        # all 8 vCPUs on each g4dn.2xlarge worker and subsequent Spark SQL jobs
        # stall at "0 tasks started" because no task slots are available.
        # (ref: Databricks Ray-on-Spark docs — reserve CPUs for hybrid Spark+Ray)
        setup_ray_cluster(
            max_worker_nodes=2,
            num_cpus_worker_node=6,      # leave 2 vCPUs/worker free for Spark tasks
            num_gpus_worker_node=1,      # g4dn.2xlarge: 1 NVIDIA T4 GPU per worker
            collect_log_to_path="/tmp/ray_logs",
        )

        ray.init(ignore_reinit_error=True)
        print(f"Ray initialized | Resources: {ray.cluster_resources()}")
        RAY_AVAILABLE = True

    except Exception as e:
        # Catches ImportError, RuntimeError, and process-level failures from
        # setup_ray_cluster (e.g. on Serverless where Spark Connect is used).
        print(f"Ray not available ({type(e).__name__}: {e}). Using single-node NumPy fallback.")
        RAY_AVAILABLE = False

# COMMAND ----------

# ── GPU diagnostic check ───────────────────────────────────────────────────────
# Run a @ray.remote(num_gpus=1) probe task before the simulation to confirm
# that Ray workers actually see a GPU.  This catches two silent failure modes:
#   1. torch.cuda.is_available() = False on workers despite being a GPU cluster
#      (e.g. CUDA_VISIBLE_DEVICES="", driver issue, or wrong DBR image)
#   2. nvidia-smi unreachable (containerisation / PATH issue)
# The task also runs a small matmul stress-test to verify CUDA execution works
# end-to-end, not just that the driver is reachable.
if RAY_AVAILABLE:
    @ray.remote(num_gpus=1)
    def _check_gpu_worker():
        import torch, subprocess
        result = {'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available()}
        print(f"[check_gpu] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            result['device_count'] = torch.cuda.device_count()
            result['device_name']  = torch.cuda.get_device_name(0)
            print(f"[check_gpu] devices={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}")
            # Quick matmul smoke-test to confirm CUDA execution end-to-end
            _a = torch.randn(1000, 1000, device='cuda', dtype=torch.float32)
            _b = (_a @ _a.T).sum().item()
            torch.cuda.synchronize()
            print(f"[check_gpu] matmul smoke-test passed (sum={_b:.2f})")
            result['matmul_ok'] = True
        try:
            nvsmi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            result['nvidia_smi'] = nvsmi.stdout.strip()
            print(f"[check_gpu] nvidia-smi: {nvsmi.stdout.strip()}")
        except Exception as _e:
            result['nvidia_smi_error'] = str(_e)
        return result

    print("==> Running GPU diagnostic on a Ray worker...")
    _gpu_diag = ray.get(_check_gpu_worker.remote())
    print(f"GPU diagnostic result: {_gpu_diag}")
    if not _gpu_diag.get('cuda_available', False):
        print("WARNING: Ray workers report CUDA not available — Monte Carlo will use CPU fallback.")
    else:
        print(f"GPU confirmed on workers: {_gpu_diag.get('device_name', 'unknown')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Monte Carlo Task (t-Copula + Lognormal Marginals)
# MAGIC
# MAGIC Each `@ray.remote` task simulates **100,000 correlated loss scenarios** using:
# MAGIC 1. **t-Copula (df=4)** for joint dependence — captures tail co-movement between lines
# MAGIC 2. **Lognormal marginals** — standard for insurance aggregate losses (skewed, non-negative)
# MAGIC
# MAGIC **Correlation structure** reflects common-shock drivers (Pearson correlations):
# MAGIC - Property ↔ Auto: 0.40 (shared weather/catastrophe events)
# MAGIC - Property ↔ Liability: 0.20 (moderate — inflation affects both)
# MAGIC - Auto ↔ Liability: 0.30 (judicial/social inflation trend)
# MAGIC
# MAGIC **GPU acceleration:** `@ray.remote(num_gpus=0.25)` — fractional allocation allows
# MAGIC 4 concurrent tasks per T4 GPU (8 tasks total with 2 workers × 1 GPU each).
# MAGIC Each task runs 100,000 scenarios using PyTorch. Random sampling + Cholesky +
# MAGIC lognormal inverse CDF (`erfinv`) run on GPU; the t-CDF step (`betainc`) runs
# MAGIC on CPU (no CUDA kernel in some PyTorch builds) then results move back to GPU.
# MAGIC PyTorch is pre-installed on DBR GPU ML; no pip install needed.
# MAGIC Falls back transparently to NumPy/SciPy if `torch.cuda` is unavailable.

# COMMAND ----------

import numpy as np
import pandas as pd

# Portfolio parameters (in $M expected annual losses)
PORTFOLIO = {
    "segments":   ["Property", "Auto", "Liability"],
    "means":      [12.5, 8.3, 5.7],          # Expected annual loss per segment ($M)
    "cv":         [0.35, 0.28, 0.42],         # Coefficients of variation
    "corr_matrix": np.array([                 # Pearson correlations
        [1.00, 0.40, 0.20],
        [0.40, 1.00, 0.30],
        [0.20, 0.30, 1.00],
    ]),
}

# Cholesky decomposition for correlated draws (done once, shared)
def get_cholesky():
    sigma = np.diag(PORTFOLIO["cv"]) @ PORTFOLIO["corr_matrix"] @ np.diag(PORTFOLIO["cv"])
    return np.linalg.cholesky(sigma)

CHOL = get_cholesky()

# ── Monte Carlo constants (module-level so Ray workers compute once per process) ─
# Importing here avoids repeated import overhead inside the @ray.remote function.
import scipy.special as _scipy_sp

_MC_COPULA_DF  = 4
_MC_MEANS      = np.array(PORTFOLIO["means"],      dtype=np.float32)
_MC_CV         = np.array(PORTFOLIO["cv"],         dtype=np.float32)
_MC_SIGMA2     = np.log(1 + _MC_CV**2)
_MC_MU_LN      = np.log(_MC_MEANS) - _MC_SIGMA2 / 2
_MC_SIGMA_LN   = np.sqrt(_MC_SIGMA2)
_MC_CORR       = np.asarray(PORTFOLIO["corr_matrix"], dtype=np.float32)

# Per-worker GPU tensor cache: populated on first GPU call, reused by all subsequent
# tasks on the same worker process. Avoids re-allocating and re-computing Cholesky
# (an O(n³) op on a 3×3 matrix) for every task call.
_GPU_TENSOR_CACHE: dict = {}

def _get_gpu_tensors(device, dtype):
    """Return (mu_t, sig_t, chol_t) tensors, computing only on first call per worker."""
    import torch
    key = (str(device), str(dtype))
    if key not in _GPU_TENSOR_CACHE:
        _GPU_TENSOR_CACHE[key] = (
            torch.tensor(_MC_MU_LN,   dtype=dtype, device=device),
            torch.tensor(_MC_SIGMA_LN, dtype=dtype, device=device),
            torch.linalg.cholesky(torch.tensor(_MC_CORR, dtype=dtype, device=device)),
        )
    return _GPU_TENSOR_CACHE[key]

# Define the Ray task function (only if Ray is available)
# @ray.remote(num_gpus=0.25): fractional GPU allocation allows 4 concurrent tasks
# per T4 GPU on a g4dn.2xlarge worker (8 tasks total across 2 workers × 1 GPU each).
# GPU path (hybrid): random sampling + Cholesky + lognormal on GPU; betainc on CPU
# (torch.special.betainc may not have a CUDA kernel in all PyTorch builds).
if RAY_AVAILABLE:
    @ray.remote(num_gpus=0.25)
    def simulate_portfolio_losses(n_scenarios: int, seed: int) -> dict:
        """
        t-Copula + lognormal marginals Monte Carlo (Sklar's theorem).

        Upgrade over Gaussian copula: the t-Copula (Student-t, df=4) captures
        *tail dependence* — extreme losses in multiple lines co-occur more
        frequently than linear correlations alone imply. This reflects common
        shocks: catastrophe events, judicial/social inflation trends, and
        macro-economic stress scenarios (e.g. stagflation) that simultaneously
        impact Property, Auto, and Liability books.

        Steps:
          1. Draw correlated standard normals Z ~ N(0, R) using Cholesky(R).   [GPU]
          2. Draw chi2 W ~ chi2(df)/df (mixing variable for t-distribution).   [GPU]
          3. T = Z / sqrt(W) gives multivariate t(df) with correlation R.      [GPU]
          4. Apply t-CDF via scipy.special.betainc to get uniform marginals.   [CPU]
             torch.special.betainc is absent from PyTorch's stable public API
             (AttributeError in PyTorch 2.7 on DBR 17.3). scipy.betainc is fast
             (vectorized C); we convert to numpy for this step then move back.
          5. Apply lognormal inverse CDF via erfinv: Phi^-1(u) = sqrt(2)*erfinv(2u-1)
             torch.special.erfinv is CUDA-accelerated; compute back on GPU.   [GPU]

        GPU path uses PyTorch (pre-installed on DBR GPU ML) for all array ops.
        Uses float32 for throughput (T4 tensor cores optimised for fp32).
        Results are moved back to CPU via .cpu().numpy() before returning (Ray
        cannot serialize CUDA tensors across process boundaries).

        Module-level constants (_MC_MU_LN, _MC_SIGMA_LN, _MC_CORR, _MC_COPULA_DF)
        and _get_gpu_tensors() are captured in the closure and loaded once per
        worker process, avoiding redundant computation on repeated calls.
        """
        import numpy as np
        import torch

        # ── GPU path via PyTorch (hybrid: sampling on GPU, betainc on CPU) ──────
        # torch.device('cuda') lets CUDA_VISIBLE_DEVICES pick the right GPU index
        # (safer than hardcoding 'cuda:0' in multi-GPU or containerised setups).
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available on this Ray worker")

            device = torch.device('cuda')   # let CUDA_VISIBLE_DEVICES pick index
            dtype  = torch.float32          # fp32 maximises T4 throughput

            # Quick matmul smoke-test to confirm CUDA execution works end-to-end
            # before running the full simulation (avoids silent hang on first op).
            _s = torch.randn(64, 64, dtype=dtype, device=device)
            _s = (_s @ _s.T).sum().item()
            torch.cuda.synchronize()
            if seed == 42:
                print(f"[Ray seed=42] GPU smoke-test ok ({torch.cuda.get_device_name(0)}, sum={_s:.1f})")

            # Retrieve precomputed tensors (Cholesky, mu, sigma) — cached per worker
            mu_t, sig_t, chol_t = _get_gpu_tensors(device, dtype)

            # Steps 1-2: correlated standard normals + chi2 mixing variable
            # chi2(df) = sum of df² independent standard normals (exact, GPU-native)
            torch.manual_seed(seed)
            z    = torch.randn(n_scenarios, 3, dtype=dtype, device=device)
            z_nu = torch.randn(n_scenarios, _MC_COPULA_DF, dtype=dtype, device=device)
            chi2 = (z_nu ** 2).sum(dim=1)                              # (n,)
            x_cor = z @ chol_t.T                                       # correlated N(0,1)

            # Step 3: t-distributed with correlation structure
            t_cor = x_cor / (chi2.unsqueeze(1) / _MC_COPULA_DF).sqrt()  # (n, 3)

            # Step 4: t-CDF via scipy.special.betainc on CPU.
            # torch.special.betainc is not in PyTorch's stable public API (absent in
            # PyTorch 2.7.0+cu126 on DBR 17.3 — raises AttributeError).
            # scipy.betainc is vectorized C — fast even for 100k×3; not the bottleneck.
            t_cor_np   = t_cor.cpu().numpy()                         # (n, 3) float32
            x_beta_np  = (_MC_COPULA_DF / (_MC_COPULA_DF + t_cor_np ** 2)).clip(0.0, 1.0)
            ibeta_np   = _scipy_sp.betainc(_MC_COPULA_DF / 2.0, 0.5, x_beta_np).astype(np.float32)
            u_np       = np.where(t_cor_np <= 0, ibeta_np / 2.0, 1.0 - ibeta_np / 2.0)
            u_cpu      = torch.from_numpy(u_np)

            # Step 5: lognormal marginals via inverse normal CDF (back on GPU)
            # Phi^-1(u) = sqrt(2) * erfinv(2u - 1)
            # torch.special.erfinv is CUDA-accelerated (unlike betainc).
            u      = u_cpu.to(device)
            u_clip = u.clamp(1e-6, 1 - 1e-6)
            q      = torch.special.erfinv(2.0 * u_clip - 1.0).mul(2.0 ** 0.5)
            losses = torch.exp(mu_t + sig_t * q)                       # (n, 3)
            total  = losses.sum(dim=1).cpu().numpy().astype(np.float64)
            backend = 'torch-gpu-hybrid'   # sampling on GPU, betainc on CPU

        except Exception as e_gpu:
            # ── CPU fallback (Serverless, non-GPU workers, torch.cuda unavailable) ─
            print(f"[Ray task seed={seed}] GPU path failed ({type(e_gpu).__name__}: {e_gpu}) — using CPU")
            from scipy.stats import t as tdist, norm as scipy_norm
            chol  = np.linalg.cholesky(_MC_CORR)
            rng   = np.random.default_rng(seed)
            z     = rng.standard_normal((n_scenarios, 3))
            chi2  = rng.chisquare(_MC_COPULA_DF, n_scenarios)
            t_cor = (z @ chol.T) / np.sqrt(chi2[:, None] / _MC_COPULA_DF)
            u     = tdist.cdf(t_cor, df=_MC_COPULA_DF)
            q     = scipy_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
            losses = np.exp(_MC_MU_LN + _MC_SIGMA_LN * q)
            total  = losses.sum(axis=1)
            backend = 'numpy-cpu'

        return {
            'seed':            seed,
            'n_scenarios':     n_scenarios,
            'copula':          f't-copula(df={_MC_COPULA_DF})',
            'backend':         backend,
            'total_loss_mean': float(total.mean()),
            'var_95':          float(np.percentile(total, 95)),
            'var_99':          float(np.percentile(total, 99)),
            'var_995':         float(np.percentile(total, 99.5)),
            'cvar_99':         float(total[total >= np.percentile(total, 99)].mean()),
            'max_loss':        float(total.max()),
            'raw_percentiles': list(np.percentile(total, [90, 95, 99, 99.5, 99.9]).tolist()),
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Launch 10 Ray Tasks — 1,000,000 Scenarios (GPU-accelerated)

# COMMAND ----------

# GPU batching: 1M scenarios/task gives the T4 GPU strong occupancy.
# 4 tasks × 1M = 4M total paths. With num_gpus=0.25, all 4 tasks run
# concurrently on a single GPU (4 × 0.25 = 1.0 GPU), keeping it fully utilized.
N_TASKS     = 4
N_PER_TASK  = 1_000_000
TOTAL_PATHS = N_TASKS * N_PER_TASK   # 4,000,000 scenarios

if RAY_AVAILABLE:
    print(f'Launching {N_TASKS} Ray tasks x {N_PER_TASK:,} scenarios = {TOTAL_PATHS:,} total paths')

    with mlflow.start_run(run_name='monte_carlo_portfolio_ray') as run:
        mlflow.log_input(_claims_dataset, context="training")
        mlflow.set_tags({
            'workshop_module': '4',
            'model_type':      'Monte Carlo - t-Copula + Lognormal Marginals',
            'n_scenarios':     str(TOTAL_PATHS),
            'n_segments':      '3',
            'framework':       'Ray + PyTorch GPU',
            'audience':        'actuarial-workshop',
        })
        mlflow.log_params({
            'n_tasks':            N_TASKS,
            'scenarios_per_task': N_PER_TASK,
            'total_scenarios':    TOTAL_PATHS,
            'copula':             't-copula',
            'copula_df':          4,
            'marginals':          'lognormal',
            'correlation_P_A':    0.40,
            'correlation_P_L':    0.20,
            'correlation_A_L':    0.30,
        })
        futures = [
            simulate_portfolio_losses.remote(N_PER_TASK, seed=42 + i)
            for i in range(N_TASKS)
        ]
        results = ray.get(futures)

        # Shut down Ray cluster immediately after all futures resolve so that
        # Spark executors are released back to the cluster. Without this,
        # the subsequent saveAsTable Spark job cannot acquire executor slots
        # (Ray workers hold them) and stalls indefinitely.
        try:
            shutdown_ray_cluster()
            ray.shutdown()
        except Exception:
            pass

        aggregate_var99  = float(sum(r['var_99']  for r in results) / len(results))
        aggregate_var995 = float(sum(r['var_995'] for r in results) / len(results))
        aggregate_cvar99 = float(sum(r['cvar_99'] for r in results) / len(results))
        aggregate_mean   = float(sum(r['total_loss_mean'] for r in results) / len(results))

        mlflow.log_metrics({
            'expected_annual_loss_M':  round(aggregate_mean, 2),
            'VaR_99_pct_M':            round(aggregate_var99, 2),
            'VaR_99_5_pct_M':          round(aggregate_var995, 2),
            'CVaR_99_pct_M':           round(aggregate_cvar99, 2),
            'implied_risk_margin_pct': round((aggregate_cvar99 / aggregate_mean - 1) * 100, 1),
        })

        backends = set(r.get('backend', 'unknown') for r in results)
        print(f'\n' + '='*55)
        print(f'  PORTFOLIO RISK SUMMARY ({TOTAL_PATHS:,} scenarios)')
        print(f'  Backend: {", ".join(sorted(backends))}')
        print('='*55)
        print(f'  Expected Annual Loss:   ${aggregate_mean:.1f}M')
        print(f'  VaR(99%):              ${aggregate_var99:.1f}M')
        print(f'  VaR(99.5%):            ${aggregate_var995:.1f}M <- Solvency II SCR')
        print(f'  CVaR(99%):             ${aggregate_cvar99:.1f}M')
        print(f'  Risk Margin (CVaR/EL): {(aggregate_cvar99/aggregate_mean - 1)*100:.0f}%')
        print('='*55)
        print(f'\nMLflow run: {run.info.run_id}')

        results_pdf = pd.DataFrame([{
            'task_id':        r['seed'] - 42,
            'n_scenarios':    r['n_scenarios'],
            'mean_loss_M':    r['total_loss_mean'],
            'var_95_M':       r['var_95'],
            'var_99_M':       r['var_99'],
            'var_995_M':      r['var_995'],
            'cvar_99_M':      r['cvar_99'],
            'max_loss_M':     r['max_loss'],
            'mlflow_run_id':  run.info.run_id,
        } for r in results])

        mc_df = spark.createDataFrame(results_pdf)
        (mc_df.write
            .format('delta')
            .mode('overwrite')
            .option('overwriteSchema', 'true')
            .saveAsTable(f'{CATALOG}.{SCHEMA}.monte_carlo_results'))
        print(f'\nResults saved to {CATALOG}.{SCHEMA}.monte_carlo_results')

else:
    # Fallback: single-node t-Copula simulation without Ray (Serverless / no GPU)
    print('\nRunning single-node Monte Carlo (Ray not available on this cluster)\n')
    from scipy.stats import t as tdist, norm as scipy_norm
    N_SCENARIOS  = 100_000
    COPULA_DF    = 4
    rng          = np.random.default_rng(42)
    means        = np.array([12.5, 8.3, 5.7])
    cv           = np.array([0.35, 0.28, 0.42])
    sigma2       = np.log(1 + cv**2)
    mu_ln        = np.log(means) - sigma2 / 2
    sigma_ln     = np.sqrt(sigma2)
    corr         = np.array([[1.00, 0.40, 0.20], [0.40, 1.00, 0.30], [0.20, 0.30, 1.00]])
    chol         = np.linalg.cholesky(corr)
    z            = rng.standard_normal((N_SCENARIOS, 3))
    chi2         = rng.chisquare(COPULA_DF, N_SCENARIOS)
    t_cor        = (z @ chol.T) / np.sqrt(chi2[:, None] / COPULA_DF)
    u            = tdist.cdf(t_cor, df=COPULA_DF)
    q            = scipy_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    losses       = np.exp(mu_ln + sigma_ln * q)
    total        = losses.sum(axis=1)

    with mlflow.start_run(run_name='monte_carlo_portfolio_singlenode') as run:
        mlflow.log_input(_claims_dataset, context="training")
        mlflow.log_params({
            'n_scenarios': N_SCENARIOS, 'framework': 'numpy',
            'copula': 't-copula', 'copula_df': COPULA_DF, 'marginals': 'lognormal',
        })
        mlflow.log_metrics({
            'expected_annual_loss_M': round(float(total.mean()), 2),
            'VaR_99_pct_M':           round(float(np.percentile(total, 99)), 2),
            'VaR_99_5_pct_M':         round(float(np.percentile(total, 99.5)), 2),
        })

        print(f'Portfolio Risk Summary ({N_SCENARIOS:,} scenarios)')
        print(f'  Expected Annual Loss: ${total.mean():.1f}M')
        print(f'  VaR(99%):            ${np.percentile(total, 99):.1f}M')
        print(f'  VaR(99.5%):          ${np.percentile(total, 99.5):.1f}M <- Solvency II SCR')
        print(f'  CVaR(99%):           ${total[total >= np.percentile(total, 99)].mean():.1f}M')

        # Write summary result to Delta — same table as the Ray path so downstream
        # consumers (app, module 5) work regardless of which path ran.
        _cvar99 = float(total[total >= np.percentile(total, 99)].mean())
        results_pdf = pd.DataFrame([{
            'task_id':      0,
            'n_scenarios':  N_SCENARIOS,
            'mean_loss_M':  float(total.mean()),
            'var_95_M':     float(np.percentile(total, 95)),
            'var_99_M':     float(np.percentile(total, 99)),
            'var_995_M':    float(np.percentile(total, 99.5)),
            'cvar_99_M':    _cvar99,
            'max_loss_M':   float(total.max()),
            'mlflow_run_id': run.info.run_id,
        }])
        mc_df = spark.createDataFrame(results_pdf)
        (mc_df.write
            .format('delta')
            .mode('overwrite')
            .option('overwriteSchema', 'true')
            .saveAsTable(f'{CATALOG}.{SCHEMA}.monte_carlo_results'))
        print(f'Results saved to {CATALOG}.{SCHEMA}.monte_carlo_results')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Technique | Framework | Scale | Use Case |
# MAGIC |---|---|---|---|
# MAGIC | SARIMA(1,0,1)(1,1,0,12) | statsmodels + applyInPandas | 52 segments × 72 months | Claim volume forecasting |
# MAGIC | GARCH(1,1) | arch + applyInPandas | 52 segments | Loss ratio volatility, risk capital |
# MAGIC | Monte Carlo (t-Copula + Lognormal) | Ray + PyTorch GPU | 4M scenarios (4 tasks × 1M paths) | VaR, CVaR, SCR calculation |
# MAGIC
# MAGIC **Next:** Module 5 — Log the best SARIMA model to MLflow, register in UC, and deploy a Model Serving endpoint.