# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: Classical Statistical Models at Scale
# MAGIC ## SARIMAX/GARCH per Segment + Monte Carlo with Ray
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC *Audience: Actuaries, Data Scientists, Financial Analysts*
# MAGIC
# MAGIC ---
# MAGIC ### What We'll Cover
# MAGIC 1. **Data Setup** — Read `gold_claims_monthly` from the DLT pipeline (40 segments × 72 months)
# MAGIC 2. **Macro Integration** — Join real StatCan macro data; visualize claims vs unemployment
# MAGIC 3. **SARIMAX at Scale** — Per-segment SARIMA + SARIMAX fit; compare MAPE with/without exog
# MAGIC 4. **GARCH at Scale** — Per-segment volatility modeling with the `arch` library
# MAGIC 5. **Monte Carlo with Ray** — Task-parallel portfolio loss simulation (100k+ paths)
# MAGIC 6. **Reserve Validation** — SARIMA forecasts vs. actual development from the loss triangle
# MAGIC 7. **Model Registration** — Register SARIMA + Monte Carlo models to UC with `@Champion` alias
# MAGIC
# MAGIC ---
# MAGIC ### Why `applyInPandas`?
# MAGIC
# MAGIC We have **40 segments** (4 product lines × 10 provinces), each needing its own SARIMAX/GARCH fit.
# MAGIC These are **independent** per-group operations. `applyInPandas` lets Spark distribute this work:
# MAGIC each executor receives a pandas DataFrame for one segment, runs standard Python/statsmodels code,
# MAGIC and returns results. No Spark ML required — just familiar Python libraries.
# MAGIC
# MAGIC ```
# MAGIC df.groupby("segment_id").applyInPandas(fit_sarimax_fn, schema=output_schema)
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

# ── Runtime detection ─────────────────────────────────────────────────────────
# Monte Carlo simulation uses NumPy/SciPy on CPU Ray workers.
# No GPU or PyTorch required — all array ops are CPU-native.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Claims Data from DLT Gold Layer
# MAGIC
# MAGIC Data flows directly from the DLT pipeline (Module 1) — no synthetic generation needed.
# MAGIC The `gold_claims_monthly` table provides **real** claim counts, loss ratios, and premium
# MAGIC exposures for **40 segments** (4 product lines × 10 provinces) × **72 months** (Jan 2019 – Dec 2024).
# MAGIC
# MAGIC ```
# MAGIC DLT Pipeline (Module 1)
# MAGIC   claims_events_raw  →  bronze_claims  →  gold_claims_monthly  ← This module reads here
# MAGIC   macro_indicators_raw → bronze_macro → silver_macro → gold_macro_features  ← exog vars
# MAGIC ```

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# ─── Configuration ────────────────────────────────────────────────────────────
# Passed as base_parameters by the bundle job; defaults used when interactive.
dbutils.widgets.text("catalog",  "my_catalog",         "UC Catalog")
dbutils.widgets.text("schema",   "actuarial_workshop", "UC Schema")
dbutils.widgets.text("run_ray",  "skip",               "Run Ray (auto/skip)")
# job_mode=true: use driver-side pandas groupby.apply() for SARIMAX/GARCH instead of
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
    "Ontario", "Quebec", "British_Columbia", "Alberta",
    "Manitoba", "Saskatchewan", "New_Brunswick", "Nova_Scotia",
    "Prince_Edward_Island", "Newfoundland",
]
MONTHS        = pd.date_range("2019-01-01", periods=72, freq="MS")  # Jan 2019 – Dec 2024

# Base claim levels (monthly, Alberta reference — used by Monte Carlo bridge weight blending)
BASE_CLAIMS = {
    "Personal_Auto":       450,
    "Commercial_Auto":     180,
    "Homeowners":          320,
    "Commercial_Property":  90,
}

# ─── Read from DLT gold layer ─────────────────────────────────────────────────
# gold_claims_monthly is produced by the DLT pipeline (task: run_dlt_pipeline in jobs.yml).
# Expected: 40 segments × 72 months = 2,880 rows with claims_count, loss_ratio, earned_premium.
claims_df = (
    spark.table(f"{CATALOG}.{SCHEMA}.gold_claims_monthly")
    .filter(F.col("month").between("2019-01-01", "2024-12-01"))
)
claims_df.createOrReplaceTempView("claims_ts")

n_segments = claims_df.select("segment_id").distinct().count()
n_rows     = claims_df.count()
print(f"Segments: {n_segments} (expected 40 = 4 product lines × 10 provinces)")
print(f"Rows:     {n_rows} (expected 2,880 = 40 × 72 months)")
if not JOB_MODE:
    display(claims_df.orderBy("segment_id", "month").limit(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Macro Data Integration
# MAGIC
# MAGIC Real macroeconomic data from **Statistics Canada** is joined to the claims series.
# MAGIC Three indicators (fetched by `scripts/fetch_macro_data.py`, processed through the DLT medallion):
# MAGIC
# MAGIC | Indicator | Source | Use in SARIMAX |
# MAGIC |---|---|---|
# MAGIC | `unemployment_rate` | StatCan 14-10-0090-01 | Leading indicator for auto/liability claims |
# MAGIC | `hpi_growth` | Derived from StatCan 18-10-0205-01 | Housing market proxy for homeowners claims |
# MAGIC | `housing_starts` | StatCan 34-10-0035-01 | New-exposure leading indicator |
# MAGIC
# MAGIC **Demo narrative:** *"Watch the MAPE drop when we add provincial unemployment rates as
# MAGIC exogenous variables — the SARIMAX model captures the economic cycle that drives claims."*

# COMMAND ----------

# Load gold_macro_features from the DLT pipeline and join to claims data
try:
    macro_df = spark.table(f"{CATALOG}.{SCHEMA}.gold_macro_features")
    macro_count = macro_df.count()
    print(f"gold_macro_features: {macro_count:,} rows (expected ~720 = 10 provinces × 72 months)")

    claims_with_macro = (
        claims_df
        .join(
            macro_df.select("region", "month", "unemployment_rate", "hpi_index",
                            "hpi_growth", "housing_starts"),
            on=["region", "month"],
            how="left",
        )
    )
    claims_with_macro.createOrReplaceTempView("claims_with_macro")
    HAS_MACRO = macro_count > 0

    if not JOB_MODE and HAS_MACRO:
        # Correlation: claims_count vs macro indicators
        corr_pdf = (
            claims_with_macro
            .select("claims_count", "unemployment_rate", "hpi_growth", "housing_starts")
            .toPandas()
            .corr()
            .round(3)
        )
        print("\nCorrelation with claims_count:")
        print(corr_pdf[["claims_count"]].sort_values("claims_count", ascending=False).to_string())

        # Time-series preview: Ontario Personal_Auto vs unemployment
        _preview = (
            claims_with_macro
            .filter(F.col("segment_id") == "Personal_Auto__Ontario")
            .orderBy("month")
            .select("month", "claims_count", "unemployment_rate")
            .toPandas()
        )
        print(f"\nOntario Personal_Auto — first 6 rows with macro:")
        print(_preview.head(6).to_string(index=False))

except Exception as _e:
    print(f"Note: gold_macro_features not available ({_e}).")
    print("SARIMAX will fall back to baseline SARIMA (no exogenous variables).")
    # Attach null exog columns so fit_sarimax_per_segment handles them gracefully
    for _col in ["unemployment_rate", "hpi_index", "hpi_growth", "housing_starts"]:
        claims_df = claims_df.withColumn(_col, F.lit(None).cast(DoubleType()))
    claims_with_macro = claims_df
    claims_with_macro.createOrReplaceTempView("claims_with_macro")
    HAS_MACRO = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2b. Feature Store Integration — Exogenous Variables from Module 3
# MAGIC
# MAGIC The Feature Store (`segment_monthly_features`) provides rolling statistical features
# MAGIC computed in Module 2 and registered in Module 3. These features — rolling means,
# MAGIC volatility regime, momentum — serve as additional exogenous variables for SARIMAX,
# MAGIC completing the lineage: `gold_claims_monthly` → Module 2 → Module 3 → Module 4.

# COMMAND ----------

# Load Feature Store features and join to claims data
try:
    _fs_table = f"{CATALOG}.{SCHEMA}.segment_monthly_features"
    fs_features = spark.table(_fs_table)
    _fs_count = fs_features.count()
    print(f"segment_monthly_features: {_fs_count:,} rows")

    # Select key features for SARIMAX exogenous variables
    _fs_cols = ["segment_id", "month", "rolling_3m_mean", "rolling_6m_mean",
                "coeff_variation_3m", "mom_change_pct", "normalized_premium"]
    # Only select columns that exist in the feature table
    _available_fs_cols = [c for c in _fs_cols if c in fs_features.columns]
    fs_subset = fs_features.select(*_available_fs_cols)

    claims_with_macro = (
        claims_with_macro
        .join(fs_subset, on=["segment_id", "month"], how="left")
    )
    claims_with_macro.createOrReplaceTempView("claims_with_macro")
    HAS_FS_FEATURES = _fs_count > 0
    print(f"Feature Store features joined: {', '.join(_available_fs_cols[2:])}")

except Exception as _fs_err:
    print(f"Note: segment_monthly_features not available ({_fs_err}).")
    print("SARIMAX will use macro variables only (no Feature Store exog).")
    HAS_FS_FEATURES = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SARIMAX at Scale — Per-Segment Forecasting with Macro + Feature Store Exogenous Variables
# MAGIC
# MAGIC ### Strategy
# MAGIC We use `df.groupby("segment_id").applyInPandas(fit_fn, schema)`.
# MAGIC
# MAGIC Each Spark task receives **one segment's pandas DataFrame** (claims + macro + features joined) and:
# MAGIC 1. Fits baseline `SARIMA(1,0,1)(1,1,0,12)` — monthly seasonality, no exog
# MAGIC 2. Fits `SARIMAX(1,0,1)(1,1,0,12)` with `unemployment_rate` + `hpi_growth` + Feature Store features as exog
# MAGIC 3. Evaluates out-of-sample MAPE on held-out last 12 months (validation set)
# MAGIC 4. Refits final model on all 72 months for the 12-month forecast
# MAGIC 5. Returns a standardized pandas DataFrame with both MAPE metrics for comparison
# MAGIC
# MAGIC The output schema must be declared upfront so Spark can parallelize safely.

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, DoubleType, IntegerType, LongType
)
import pyspark.sql.functions as F

# Output schema for SARIMAX results
# New fields vs baseline SARIMA: mape_baseline, mape_sarimax, exog_vars
SARIMA_SCHEMA = StructType([
    StructField("segment_id",    StringType(), False),
    StructField("month",         DateType(),   False),
    StructField("record_type",   StringType(), False),   # "actual" or "forecast"
    StructField("claims_count",  DoubleType(), True),
    StructField("forecast_mean", DoubleType(), True),
    StructField("forecast_lo95", DoubleType(), True),
    StructField("forecast_hi95", DoubleType(), True),
    StructField("aic",           DoubleType(), True),
    StructField("mape",          DoubleType(), True),    # primary MAPE (SARIMAX if available)
    StructField("mape_baseline", DoubleType(), True),    # SARIMA (no exog)
    StructField("mape_sarimax",  DoubleType(), True),    # SARIMAX (with macro exog)
    StructField("exog_vars",     StringType(), True),    # e.g. "unemployment_rate,hpi_growth"
])

def fit_sarimax_per_segment(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit baseline SARIMA and SARIMAX with macro exogenous variables for one segment.

    Train/validation split: first 60 months for training, last 12 for out-of-sample MAPE.
    Final model is refit on all 72 months for the 12-month forecast.

    Exogenous forecast: last 3-month average held flat (appropriate for a 12-month horizon;
    in production, use Bank of Canada / StatCan projections for the exog forecast).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    warnings.filterwarnings("ignore")

    segment_id = pdf["segment_id"].iloc[0]
    pdf        = pdf.sort_values("month").reset_index(drop=True)

    y      = pdf["claims_count"].astype(float).values
    months = pd.to_datetime(pdf["month"])

    # Exog columns: macro variables + Feature Store features (if available)
    exog_cols = ["unemployment_rate", "hpi_growth"]
    _fs_exog = ["rolling_3m_mean", "coeff_variation_3m", "mom_change_pct"]
    for _fc in _fs_exog:
        if _fc in pdf.columns and not pdf[_fc].isna().all():
            exog_cols.append(_fc)

    # Fill NaN exog values (hpi_growth is NaN for the first month due to lag;
    # forward-fill then back-fill covers edge cases at either end of the series)
    exog_data = pdf[exog_cols].copy() if all(c in pdf.columns for c in exog_cols) \
                else pd.DataFrame(np.nan, index=pdf.index, columns=exog_cols)
    exog_data = exog_data.ffill().bfill()
    has_exog  = not exog_data.isna().all().any()
    exog_arr  = exog_data.values.astype(float) if has_exog else None

    # Train/validation split (60 train, 12 validation)
    n_train, n_val = 60, 12
    y_train = y[:n_train]
    y_val   = y[n_train:]

    aic = mape_baseline = mape_sarimax = np.nan
    fcast_mean = np.full(12, np.nan)
    fcast_ci   = pd.DataFrame({"lower": np.full(12, np.nan), "upper": np.full(12, np.nan)})
    exog_vars_str = ",".join(exog_cols) if has_exog else ""

    try:
        # ── Model 1: Baseline SARIMA (no exog) ───────────────────────────────
        m_base   = SARIMAX(y_train, order=(1,0,1), seasonal_order=(1,1,0,12),
                           enforce_stationarity=False, enforce_invertibility=False)
        fit_base = m_base.fit(disp=False, maxiter=100)
        aic      = fit_base.aic
        fc_base  = fit_base.forecast(steps=n_val)
        mape_baseline = float(
            np.mean(np.abs((y_val - fc_base) / np.clip(y_val, 1, None))) * 100
        )

        # ── Model 2: SARIMAX with macro exog (if data is available) ──────────
        if has_exog:
            exog_train = exog_arr[:n_train]
            exog_val   = exog_arr[n_train:]
            try:
                m_sx   = SARIMAX(y_train, exog=exog_train, order=(1,0,1),
                                 seasonal_order=(1,1,0,12),
                                 enforce_stationarity=False, enforce_invertibility=False)
                fit_sx = m_sx.fit(disp=False, maxiter=100)
                fc_sx  = fit_sx.forecast(steps=n_val, exog=exog_val)
                mape_sarimax = float(
                    np.mean(np.abs((y_val - fc_sx) / np.clip(y_val, 1, None))) * 100
                )
            except Exception:
                mape_sarimax = mape_baseline   # fall back gracefully

        # ── Final model: refit on full 72 months for forecasting ─────────────
        if has_exog:
            m_final = SARIMAX(y, exog=exog_arr, order=(1,0,1), seasonal_order=(1,1,0,12),
                              enforce_stationarity=False, enforce_invertibility=False)
        else:
            m_final = SARIMAX(y, order=(1,0,1), seasonal_order=(1,1,0,12),
                              enforce_stationarity=False, enforce_invertibility=False)
        fit_final = m_final.fit(disp=False, maxiter=100)

        # 12-month forecast; extrapolate exog as last 3-month average held flat
        if has_exog:
            exog_fcast = np.tile(exog_arr[-3:].mean(axis=0), (12, 1))
            forecast   = fit_final.get_forecast(steps=12, exog=exog_fcast)
        else:
            forecast   = fit_final.get_forecast(steps=12)

        fcast_mean   = forecast.predicted_mean
        fcast_ci_raw = forecast.conf_int(alpha=0.05)
        fcast_ci     = fcast_ci_raw if hasattr(fcast_ci_raw, 'iloc') \
                       else pd.DataFrame(fcast_ci_raw, columns=["lower", "upper"])

    except Exception:
        # Return NaN forecasts — allows the pipeline to continue for all segments
        pass

    # ── Build output DataFrame ────────────────────────────────────────────────
    last_month     = months.max()
    forecast_months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1), periods=12, freq="MS"
    )
    # Primary MAPE = SARIMAX when available, else baseline
    _primary_mape = mape_sarimax if has_exog and not np.isnan(mape_sarimax) else mape_baseline

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
        "mape_baseline": [None] * len(y),
        "mape_sarimax":  [None] * len(y),
        "exog_vars":     [None] * len(y),
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
        "mape":          [_primary_mape] * 12,
        "mape_baseline": [mape_baseline] * 12,
        "mape_sarimax":  [mape_sarimax]  * 12,
        "exog_vars":     [exog_vars_str] * 12,
    })

    return pd.concat([actuals_rows, forecast_rows], ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run SARIMAX Across All 40 Segments
# MAGIC
# MAGIC Spark distributes one segment per task. Each task fits two models (SARIMA + SARIMAX)
# MAGIC and returns MAPE for both — enabling a direct accuracy comparison in MLflow.

# COMMAND ----------

import mlflow
_current_user = spark.sql("SELECT current_user()").collect()[0][0]
# Use a flat path under /Users/<email>/ — avoid nested subdirectories which
# require the parent to pre-exist (fails on fresh workspaces).
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_claims_sarima")

# Register input dataset for UC lineage — all models (SARIMAX, GARCH, Monte Carlo) are
# trained from gold_claims_monthly (the DLT gold layer, not a synthetic table).
try:
    _claims_dataset = mlflow.data.load_delta(
        table_name=f"{CATALOG}.{SCHEMA}.gold_claims_monthly",
        name="gold_claims_monthly",
    )
    _log_input = True
except Exception:
    _log_input = False   # DLT table may not exist in interactive session before first pipeline run

with mlflow.start_run(run_name="sarimax_all_segments") as run:
    if _log_input:
        mlflow.log_input(_claims_dataset, context="training")
    mlflow.set_tags({
        "workshop_module": "4",
        "model_type":      "SARIMAX(1,0,1)(1,1,0,12)",
        "segments":        "40",
        "horizon_months":  "12",
        "exog_vars":       "unemployment_rate,hpi_growth" if HAS_MACRO else "none",
        "audience":        "actuarial-workshop",
    })

    if JOB_MODE:
        # Driver-side: collect to pandas, fit in a Python loop.
        # Avoids applyInPandas which requires statsmodels on the remote Serverless
        # compute — a separate Python environment that %pip install cannot reach.
        print("job_mode=true: using driver-side pandas groupby.apply() for SARIMAX")
        claims_pdf = claims_with_macro.toPandas()
        sarima_result_pdf = (
            claims_pdf
            .groupby("segment_id", group_keys=False)
            .apply(fit_sarimax_per_segment)
            .reset_index(drop=True)
        )
        sarima_results_df = spark.createDataFrame(sarima_result_pdf, schema=SARIMA_SCHEMA)
    else:
        # Distributed: Spark sends one segment per task to applyInPandas workers.
        # Requires statsmodels on the compute environment (ml_env spec in bundle jobs).
        sarima_results_df = (
            claims_with_macro
            .groupby("segment_id")
            .applyInPandas(fit_sarimax_per_segment, schema=SARIMA_SCHEMA)
        )

    # Trigger execution and compute MAPE metrics
    total_rows = sarima_results_df.count()

    mape_row = (
        sarima_results_df
        .filter(F.col("record_type") == "forecast")
        .agg(
            F.mean("mape").alias("avg_mape"),
            F.mean("mape_baseline").alias("avg_mape_baseline"),
            F.mean("mape_sarimax").alias("avg_mape_sarimax"),
        )
        .collect()[0]
    )
    avg_mape          = mape_row["avg_mape"]          or 0.0
    avg_mape_baseline = mape_row["avg_mape_baseline"] or avg_mape
    avg_mape_sarimax  = mape_row["avg_mape_sarimax"]  or avg_mape
    avg_mape_improve  = avg_mape_baseline - avg_mape_sarimax

    mlflow.log_metrics({
        "avg_mape_pct":             round(avg_mape, 2),
        "avg_mape_baseline_pct":    round(avg_mape_baseline, 2),
        "avg_mape_sarimax_pct":     round(avg_mape_sarimax, 2),
        "avg_mape_improvement_pct": round(avg_mape_improve, 2),
        "total_output_rows":        total_rows,
        "segments_fitted":          40,
    })

    print(f"SARIMAX complete | Total rows: {total_rows}")
    print(f"  Baseline SARIMA MAPE:  {avg_mape_baseline:.1f}%")
    print(f"  SARIMAX MAPE:          {avg_mape_sarimax:.1f}%")
    print(f"  Improvement:           {avg_mape_improve:+.1f}%  ({'↓ better' if avg_mape_improve > 0 else '↑ worse or unchanged'})")
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
    if _log_input:
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

        # 4 workers × 6 Ray CPUs = 24 Ray CPU slots; Spark retains 2 vCPUs
        # per worker (8 total) for Delta writes after Ray shuts down.
        # Without reserving CPUs for Spark, Ray occupies all cores and
        # subsequent saveAsTable calls stall at "0 tasks started".
        setup_ray_cluster(
            max_worker_nodes=4,
            num_cpus_worker_node=6,      # leave 2 of 8 vCPUs free for Spark per worker
            num_gpus_worker_node=0,      # pure CPU cluster
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
# MAGIC **Ray-distributed:** `@ray.remote(num_cpus=1)` — each task uses 1 CPU slot; 24 Ray CPUs
# MAGIC across 4 workers allow up to 24 concurrent tasks. Each task runs 10M scenarios
# MAGIC using NumPy/SciPy: random sampling + Cholesky + lognormal inverse CDF (`norm.ppf`)
# MAGIC + t-CDF (`scipy.stats.t.cdf`). Falls back to single-node NumPy when Ray is unavailable.

# COMMAND ----------

# ── SARIMA → Monte Carlo Bridge ──────────────────────────────────────────────
# Connects SARIMA segment-level forecasts to the Monte Carlo simulation:
# for each of the 12 forecast months the product-line claims growth factor
# (forecast / trailing-12-month actual average) is used to scale the baseline
# PORTFOLIO means.  This gives VaR a forward-looking dimension — capital
# requirements evolve in line with the SARIMA predictions rather than staying
# at static calibration values.
#
# Product-line → MC-segment mapping (weighted by BASE_CLAIMS volume):
#   Property  = Homeowners (320) + Commercial_Property (90)
#   Auto      = Personal_Auto (450) + Commercial_Auto (180)
#   Liability = 0.5×Property + 0.5×Auto growth (proxied; no direct SARIMA data)
#
# Regional dimension: SARIMA forecasts are also aggregated by region → saved to
# `regional_claims_forecast` Delta table, giving a region × line breakdown of
# expected claims over the 12-month horizon without requiring a 52-dim MC.
try:
    _sarima_pd = sarima_results_df.toPandas()
    _sarima_pd["product_line"] = _sarima_pd["segment_id"].str.split("__").str[0]
    _sarima_pd["region"]       = _sarima_pd["segment_id"].str.split("__").str[1]

    # Trailing-12-month average of region-aggregated actuals per product line
    _baseline_monthly = (
        _sarima_pd[_sarima_pd["record_type"] == "actual"]
        .groupby(["product_line", "month"])["claims_count"]
        .sum()
        .reset_index()
        .sort_values(["product_line", "month"])
    )
    _baseline_by_line = (
        _baseline_monthly
        .groupby("product_line")
        .apply(lambda g: g.tail(12)["claims_count"].mean())
    )

    # Monthly aggregate forecast: sum across all regions per product line
    _forecast_agg = (
        _sarima_pd[_sarima_pd["record_type"] == "forecast"]
        .groupby(["product_line", "month"])["forecast_mean"]
        .sum()
        .reset_index()
        .sort_values(["product_line", "month"])
    )

    # Growth factor: forecast / baseline, clipped to [0.5, 2.0] to dampen outliers
    _baseline_lookup = _baseline_by_line.to_dict()
    _forecast_agg["growth"] = _forecast_agg.apply(
        lambda row: float(np.clip(
            row["forecast_mean"] / max(_baseline_lookup.get(row["product_line"], 1.0), 1e-6),
            0.5, 2.0,
        )),
        axis=1,
    )

    # Build nested dict: {product_line: {month: growth_factor}}
    _growth_by_line = {
        line: grp.set_index("month")["growth"].to_dict()
        for line, grp in _forecast_agg.groupby("product_line")
    }
    _forecast_months = sorted(_forecast_agg["month"].unique())  # 12 datetime.date values

    # Weights for product-line → MC segment blending
    _W_PROP = {"Homeowners": 320, "Commercial_Property": 90}
    _W_AUTO = {"Personal_Auto": 450, "Commercial_Auto": 180}

    def _sarima_growth_for_month(month_date) -> np.ndarray:
        """Return growth vector [property, auto, liability] for a forecast month."""
        def _wblend(weights):
            total = sum(weights.values())
            return sum(w * _growth_by_line[line].get(month_date, 1.0)
                       for line, w in weights.items()) / total
        prop_g = _wblend(_W_PROP)
        auto_g = _wblend(_W_AUTO)
        liab_g = 0.5 * (prop_g + auto_g)
        return np.array([prop_g, auto_g, liab_g], dtype=np.float32)

    print(f"SARIMA bridge ready | {len(_forecast_months)} forecast months")
    if not JOB_MODE and _forecast_months:
        g1  = _sarima_growth_for_month(_forecast_months[0])
        g12 = _sarima_growth_for_month(_forecast_months[-1])
        print(f"  Growth +1 mo:  Property={g1[0]:.3f}  Auto={g1[1]:.3f}  Liability={g1[2]:.3f}")
        print(f"  Growth +12 mo: Property={g12[0]:.3f}  Auto={g12[1]:.3f}  Liability={g12[2]:.3f}")

    # Regional breakdown preview (top 5 regions by share of month-12 forecast)
    if not JOB_MODE and _forecast_months:
        _reg_agg = (
            _sarima_pd[_sarima_pd["record_type"] == "forecast"]
            .groupby(["region", "month"])["forecast_mean"]
            .sum()
            .reset_index()
        )
        _last_month = _forecast_months[-1]
        _reg_last = _reg_agg[_reg_agg["month"] == _last_month].copy()
        _reg_last["share_pct"] = (_reg_last["forecast_mean"] / _reg_last["forecast_mean"].sum() * 100).round(1)
        print(f"\n  Regional claims share (month +12):")
        for _, row in _reg_last.sort_values("share_pct", ascending=False).head(5).iterrows():
            print(f"    {row['region']:<25} {row['share_pct']:>5.1f}%")

except NameError:
    # sarima_results_df not in scope (running section 4 in isolation)
    _forecast_months = []
    def _sarima_growth_for_month(_): return np.ones(3, dtype=np.float32)
    print("SARIMA bridge: sarima_results_df not available — using static means (growth=1.0)")

# COMMAND ----------

import numpy as np
import pandas as pd

# ── GARCH → Monte Carlo Bridge: derive CVs from fitted volatilities ──────────
# Instead of hardcoded CVs, aggregate GARCH conditional volatilities by line of
# business to produce data-driven coefficients of variation for the MC simulation.
# This connects the GARCH output directly to Monte Carlo input.
_DEFAULT_CV = [0.35, 0.28, 0.42]  # fallback if GARCH results unavailable
try:
    _garch_pd = garch_results_df.toPandas()
    _garch_pd["product_line"] = _garch_pd["segment_id"].str.split("__").str[0]

    # Aggregate conditional volatility by product line → MC segment mapping
    _garch_vol = (
        _garch_pd[_garch_pd["cond_volatility"].notna()]
        .groupby("product_line")["cond_volatility"]
        .mean()
    )

    # Map product lines to MC segments (Property, Auto, Liability proxy)
    _prop_vol = np.mean([_garch_vol.get("Homeowners", 0.35),
                          _garch_vol.get("Commercial_Property", 0.42)])
    _auto_vol = np.mean([_garch_vol.get("Personal_Auto", 0.28),
                          _garch_vol.get("Commercial_Auto", 0.30)])
    # Liability proxied as average of Property and Auto volatility
    _liab_vol = 0.5 * (_prop_vol + _auto_vol)

    # Scale GARCH volatility (in log-return %) to CV space (typically 0.1–0.6)
    # GARCH conditional vol is in % of log-returns; scale to CV by dividing by 100
    # and applying a floor/ceiling for stability
    _garch_cvs = np.clip(
        np.array([_prop_vol, _auto_vol, _liab_vol]) / 100.0 * 3.0,  # empirical scaling
        0.15, 0.60,
    ).tolist()
    print(f"GARCH-derived CVs: Property={_garch_cvs[0]:.3f}, Auto={_garch_cvs[1]:.3f}, Liability={_garch_cvs[2]:.3f}")
    _MC_USE_GARCH_CV = True
except (NameError, Exception) as _garch_cv_err:
    print(f"GARCH CVs not available ({_garch_cv_err}) — using default CVs: {_DEFAULT_CV}")
    _garch_cvs = _DEFAULT_CV
    _MC_USE_GARCH_CV = False

# Portfolio parameters (in $M expected annual losses)
PORTFOLIO = {
    "segments":   ["Property", "Auto", "Liability"],
    "means":      [12.5, 8.3, 5.7],          # Expected annual loss per segment ($M)
    "cv":         _garch_cvs,                 # GARCH-derived CVs (or defaults)
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
_MC_COPULA_DF  = 4
_MC_MEANS      = np.array(PORTFOLIO["means"],      dtype=np.float32)
_MC_CV         = np.array(PORTFOLIO["cv"],         dtype=np.float32)
_MC_SIGMA2     = np.log(1 + _MC_CV**2)
_MC_MU_LN      = np.log(_MC_MEANS) - _MC_SIGMA2 / 2
_MC_SIGMA_LN   = np.sqrt(_MC_SIGMA2)
_MC_CORR       = np.asarray(PORTFOLIO["corr_matrix"], dtype=np.float32)

# Stress scenario correlation matrices (module-level for Ray worker closures).
# stress_corr: systemic/contagion regime — correlations spike during market crises
# cat_event:   post-catastrophe regime — geographic concentration amplifies co-movement
_STRESS_CORR = np.array([[1.00, 0.75, 0.60],
                          [0.75, 1.00, 0.65],
                          [0.60, 0.65, 1.00]], dtype=np.float32)
_CAT_CORR    = np.array([[1.00, 0.55, 0.40],
                          [0.55, 1.00, 0.45],
                          [0.40, 0.45, 1.00]], dtype=np.float32)

# Define the Ray task function (only if Ray is available)
# @ray.remote(num_cpus=1): 24 concurrent tasks across 4 workers (4 × 6 = 24 Ray CPUs).
# NumPy/SciPy path: random sampling + Cholesky + lognormal + t-CDF all on CPU.
if RAY_AVAILABLE:
    @ray.remote(num_cpus=1)
    def simulate_portfolio_losses(n_scenarios: int, seed: int, means_override=None,
                                   scenario: str = 'baseline') -> dict:
        """
        t-Copula + lognormal marginals Monte Carlo (Sklar's theorem).

        scenario options
        ─────────────────────────────────────────────────────────────────
        'baseline'        Standard portfolio with static means (or SARIMA means
                          when means_override is supplied for time-series runs).
        'stress_corr'     Systemic / contagion risk: correlations spike to 0.65–0.75,
                          modelling financial crisis or industry-wide loss event.
        'cat_event'       1-in-250yr catastrophe: elevated means (Property 3.5×,
                          Auto 1.8×, Liability 1.4×), stressed correlations, and a
                          Poisson(λ=0.05) jump process for discrete CAT hits.
        'inflation_shock' Sustained 30 % loss-cost inflation with +15 % CV uncertainty.

        Steps (NumPy/SciPy):
          1. Draw correlated standard normals Z ~ N(0, R) using Cholesky(R).
          2. Draw chi2 W ~ chi2(df)/df (mixing variable for t-distribution).
          3. T = Z / sqrt(W) gives multivariate t(df) with correlation R.
          4. Apply t-CDF via scipy.stats.t.cdf().
          5. Apply lognormal inverse CDF via scipy.stats.norm.ppf().
          6. (cat_event only) Poisson jump shocks for discrete CAT events.

        Module-level constants (_MC_MEANS, _MC_CV, _MC_CORR, etc.) are captured
        in the closure and available on each Ray worker without serialisation.
        """
        import numpy as np
        from scipy.stats import t as tdist, norm as scipy_norm

        # ── Scenario parameter resolution ─────────────────────────────────────
        # Each scenario adjusts the effective means, CV, and/or correlation matrix.
        _sc_means_mult = np.ones(3, dtype=np.float32)   # multiplier on top of base means
        _sc_cv_mult    = np.ones(3, dtype=np.float32)   # multiplier on top of base CV
        _sc_corr_arr   = None                            # None → use _MC_CORR (base)
        _add_cat_jump  = False

        if scenario == 'stress_corr':
            # Systemic/contagion risk: correlations spike under market-wide stress
            _sc_corr_arr = _STRESS_CORR
        elif scenario == 'cat_event':
            # 1-in-250yr catastrophe: major natural disaster (hurricane / earthquake)
            # Property bears the brunt (3.5×), Auto elevated (1.8×), Liability (1.4×)
            _sc_means_mult = np.array([3.5, 1.8, 1.4], dtype=np.float32)
            _sc_cv_mult    = np.array([1.5, 1.3, 1.2], dtype=np.float32)
            _sc_corr_arr   = _CAT_CORR
            _add_cat_jump  = True   # Poisson jump process added after copula step
        elif scenario == 'inflation_shock':
            # +30 % loss-cost inflation with elevated uncertainty
            _sc_means_mult = np.array([1.30, 1.30, 1.30], dtype=np.float32)
            _sc_cv_mult    = np.array([1.15, 1.15, 1.15], dtype=np.float32)

        # Apply scenario multipliers on top of means_override (SARIMA) or base means
        _base_means = (np.array(means_override, dtype=np.float32)
                       if means_override is not None else _MC_MEANS)
        _eff_means  = _base_means * _sc_means_mult
        _eff_cv     = _MC_CV * _sc_cv_mult
        _eff_sig2   = np.log(1.0 + _eff_cv**2)
        _eff_mu_ln  = np.log(_eff_means) - _eff_sig2 / 2.0
        _eff_sig_ln = np.sqrt(_eff_sig2)

        # ── NumPy/SciPy simulation ────────────────────────────────────────────
        _cpu_chol = np.linalg.cholesky(
            _sc_corr_arr if _sc_corr_arr is not None else _MC_CORR
        )
        rng      = np.random.default_rng(seed)
        z        = rng.standard_normal((n_scenarios, 3))
        chi2     = rng.chisquare(_MC_COPULA_DF, n_scenarios)
        t_cor    = (z @ _cpu_chol.T) / np.sqrt(chi2[:, None] / _MC_COPULA_DF)
        u        = tdist.cdf(t_cor, df=_MC_COPULA_DF)
        q        = scipy_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        losses   = np.exp(_eff_mu_ln + _eff_sig_ln * q)

        # Catastrophe jump process (cat_event scenario only).
        # Poisson(λ=0.05): each scenario has a 5 % chance of a discrete CAT
        # event.  When it hits, losses are amplified by line-specific factors
        # (Property ×8, Auto ×3, Liability ×1.5) on top of the copula result.
        if _add_cat_jump:
            _cat_n   = rng.poisson(0.05, n_scenarios)
            _cat_amp = np.stack(
                [_cat_n * 8.0, _cat_n * 3.0, _cat_n * 1.5], axis=1
            ).astype(np.float32)
            losses   = losses * (1.0 + _cat_amp)
        total    = losses.sum(axis=1)

        return {
            'seed':            seed,
            'n_scenarios':     n_scenarios,
            'scenario':        scenario,
            'copula':          f't-copula(df={_MC_COPULA_DF})',
            'backend':         'numpy-cpu',
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
# MAGIC ### Monte Carlo — Baseline + VaR Evolution + Stress Tests (Ray-distributed)
# MAGIC
# MAGIC Three complementary simulations dispatched together in a single Ray batch:
# MAGIC
# MAGIC 1. **Baseline** (static `PORTFOLIO` means, 40M paths) — current-period capital
# MAGIC    position; written to `monte_carlo_results` for downstream Modules 5–7.
# MAGIC 2. **12-month VaR evolution** (SARIMA-driven means, 480M paths) — for each
# MAGIC    forecast month means are scaled by the SARIMA growth factor, giving a
# MAGIC    **forward capital requirement curve** aligned to claims expectations.
# MAGIC 3. **Stress tests** (3 scenarios × 40M paths = 120M paths):
# MAGIC    - `cat_event`: 1-in-250yr catastrophe — elevated means (Property 3.5×,
# MAGIC      Auto 1.8×), stressed correlations, **Poisson(λ=0.05) jump shocks**
# MAGIC    - `stress_corr`: systemic/contagion risk — correlations spike to 0.65–0.75
# MAGIC    - `inflation_shock`: +30 % loss-cost inflation, +15 % CV uncertainty
# MAGIC
# MAGIC All `(1 + 12 + 3) × 4 = 64` tasks are dispatched before any are collected.
# MAGIC Ray queues them across 24 CPU slots (4 workers × 6 CPUs), running up to 24
# MAGIC tasks concurrently. 640M total paths in ~2-3 minutes on 4 × m5.2xlarge.

# COMMAND ----------

# 4 tasks per run, 10M scenarios each → distributed across 24 Ray CPU slots.
# 10M paths/task: ~600 MB RAM per task; 24 concurrent tasks across 4 workers.
N_TASKS          = 4
N_PER_TASK       = 10_000_000      # 10× — GPU stress demo; total ~640M paths
STRESS_SCENARIOS = ['stress_corr', 'cat_event', 'inflation_shock']
N_RUNS           = 1 + len(_forecast_months) + len(STRESS_SCENARIOS)

if RAY_AVAILABLE:
    print(f'Launching {N_RUNS} runs × {N_TASKS} tasks × {N_PER_TASK:,} paths '
          f'= {N_RUNS * N_TASKS * N_PER_TASK:,} total paths')

    with mlflow.start_run(run_name='monte_carlo_portfolio_ray') as run:
        if _log_input:
            mlflow.log_input(_claims_dataset, context="training")
        mlflow.set_tags({
            'workshop_module': '4',
            'model_type':      'Monte Carlo - t-Copula + Lognormal Marginals (SARIMAX-driven)',
            'n_scenarios':     str(N_RUNS * N_TASKS * N_PER_TASK),
            'n_segments':      '3',
            'framework':       'Ray + NumPy CPU',
            'audience':        'actuarial-workshop',
        })
        mlflow.log_params({
            'n_tasks':              N_TASKS,
            'scenarios_per_task':   N_PER_TASK,
            'total_scenarios':      N_RUNS * N_TASKS * N_PER_TASK,
            'n_forecast_months':    len(_forecast_months),
            'copula':               't-copula',
            'copula_df':            4,
            'marginals':            'lognormal',
            'correlation_P_A':      0.40,
            'correlation_P_L':      0.20,
            'correlation_A_L':      0.30,
            'stress_scenarios':     ','.join(STRESS_SCENARIOS),
            'n_stress_scenarios':   len(STRESS_SCENARIOS),
        })

        # ── Dispatch all tasks before collecting any ───────────────────────────
        # Baseline: static PORTFOLIO means (seeds 42–45)
        baseline_futures = [
            simulate_portfolio_losses.remote(N_PER_TASK, seed=42 + i)
            for i in range(N_TASKS)
        ]
        # Time-series: 12 months with SARIMA-adjusted means (seeds 100+)
        ts_futures: dict = {}
        for _mi, _month in enumerate(_forecast_months):
            _growth   = _sarima_growth_for_month(_month)
            _means_t  = (_MC_MEANS * _growth).tolist()
            ts_futures[_month] = [
                simulate_portfolio_losses.remote(
                    N_PER_TASK,
                    seed=100 + _mi * N_TASKS + i,
                    means_override=_means_t,
                )
                for i in range(N_TASKS)
            ]

        # Stress scenarios (seeds 300+): 3 scenarios × 4 tasks each.
        # All dispatched before collecting any results — Ray queues all 64 tasks
        # and runs up to 24 concurrently across 4 CPU workers.
        stress_futures: dict = {}
        for _si, _sc in enumerate(STRESS_SCENARIOS):
            stress_futures[_sc] = [
                simulate_portfolio_losses.remote(
                    N_PER_TASK, seed=300 + _si * N_TASKS + i, scenario=_sc,
                )
                for i in range(N_TASKS)
            ]

        # Collect (Ray blocks until all tasks finish)
        baseline_results = ray.get(baseline_futures)
        ts_results       = {m: ray.get(futs) for m, futs in ts_futures.items()}
        stress_results   = {sc: ray.get(futs) for sc, futs in stress_futures.items()}

        # Shut down Ray immediately so Spark can reclaim executor slots for the
        # saveAsTable calls below. Ray workers hold all CPUs while alive.
        try:
            shutdown_ray_cluster()
            ray.shutdown()
        except Exception:
            pass

        # ── Baseline portfolio summary ─────────────────────────────────────────
        def _agg(res, key): return float(sum(r[key] for r in res) / len(res))

        aggregate_var99  = _agg(baseline_results, 'var_99')
        aggregate_var995 = _agg(baseline_results, 'var_995')
        aggregate_cvar99 = _agg(baseline_results, 'cvar_99')
        aggregate_mean   = _agg(baseline_results, 'total_loss_mean')

        mlflow.log_metrics({
            'expected_annual_loss_M':  round(aggregate_mean, 2),
            'VaR_99_pct_M':            round(aggregate_var99, 2),
            'VaR_99_5_pct_M':          round(aggregate_var995, 2),
            'CVaR_99_pct_M':           round(aggregate_cvar99, 2),
            'implied_risk_margin_pct': round((aggregate_cvar99 / aggregate_mean - 1) * 100, 1),
        })

        backends = set(r.get('backend', 'unknown') for r in baseline_results)
        print(f'\n' + '='*55)
        print(f'  PORTFOLIO RISK SUMMARY ({N_TASKS * N_PER_TASK:,} scenarios)')
        print(f'  Backend: {", ".join(sorted(backends))}')
        print('='*55)
        print(f'  Expected Annual Loss:   ${aggregate_mean:.1f}M')
        print(f'  VaR(99%):              ${aggregate_var99:.1f}M')
        print(f'  VaR(99.5%):            ${aggregate_var995:.1f}M <- Solvency II SCR')
        print(f'  CVaR(99%):             ${aggregate_cvar99:.1f}M')
        print(f'  Risk Margin (CVaR/EL): {(aggregate_cvar99/aggregate_mean - 1)*100:.0f}%')
        print('='*55)
        print(f'\nMLflow run: {run.info.run_id}')

        # Baseline rows → monte_carlo_results (backward-compatible with Module 5/app)
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
        } for r in baseline_results])

        mc_df = spark.createDataFrame(results_pdf)
        (mc_df.write
            .format('delta')
            .mode('overwrite')
            .option('overwriteSchema', 'true')
            .saveAsTable(f'{CATALOG}.{SCHEMA}.monte_carlo_results'))
        print(f'\nBaseline saved → {CATALOG}.{SCHEMA}.monte_carlo_results')

        # ── 12-month VaR evolution (SARIMA-driven) ─────────────────────────────
        timeline_rows = []
        for _mi, _month in enumerate(_forecast_months):
            _month_res = ts_results[_month]
            _growth    = _sarima_growth_for_month(_month)
            _means_t   = _MC_MEANS * _growth
            _row = {
                'forecast_month':      str(_month),
                'month_idx':           _mi + 1,
                'property_mean_M':     float(_means_t[0]),
                'auto_mean_M':         float(_means_t[1]),
                'liability_mean_M':    float(_means_t[2]),
                'total_mean_M':        _agg(_month_res, 'total_loss_mean'),
                'var_99_M':            _agg(_month_res, 'var_99'),
                'var_995_M':           _agg(_month_res, 'var_995'),
                'cvar_99_M':           _agg(_month_res, 'cvar_99'),
                'var_995_vs_baseline': (_agg(_month_res, 'var_995') / aggregate_var995 - 1.0) * 100,
            }
            timeline_rows.append(_row)
            mlflow.log_metric(f'VaR_99_5_month_{_mi + 1:02d}', round(_row['var_995_M'], 2))

        # Print VaR evolution table
        print(f'\n  VaR EVOLUTION  (SARIMA-driven, {len(_forecast_months)}-month horizon)')
        print('  ' + '─'*70)
        print(f'  {"Month":<12} {"Exp.Loss":>10} {"VaR(99%)":>10} {"VaR(99.5%)":>12} {"Δ VaR(99.5%)":>14}')
        print('  ' + '─'*70)
        print(f'  {"[baseline]":<12} ${aggregate_mean:>8.1f}M ${aggregate_var99:>8.1f}M ${aggregate_var995:>10.1f}M  {"—":>12}')
        for _row in timeline_rows:
            print(
                f'  {_row["forecast_month"]:<12} '
                f'${_row["total_mean_M"]:>8.1f}M '
                f'${_row["var_99_M"]:>8.1f}M '
                f'${_row["var_995_M"]:>10.1f}M  '
                f'{_row["var_995_vs_baseline"]:>+11.1f}%'
            )
        print('  ' + '─'*70)

        ts_df = spark.createDataFrame(pd.DataFrame(timeline_rows))
        (ts_df.write
            .format('delta')
            .mode('overwrite')
            .option('overwriteSchema', 'true')
            .saveAsTable(f'{CATALOG}.{SCHEMA}.portfolio_risk_timeline'))
        print(f'\nVaR time-series saved → {CATALOG}.{SCHEMA}.portfolio_risk_timeline')

        # Regional claims forecast (SARIMA breakdown by region × month)
        if '_sarima_pd' in dir():
            _reg_df = spark.createDataFrame(
                _sarima_pd[_sarima_pd["record_type"] == "forecast"]
                .groupby(["region", "month"])["forecast_mean"]
                .sum()
                .reset_index()
                .rename(columns={"forecast_mean": "total_forecast_claims",
                                  "month": "forecast_month"})
            )
            (_reg_df.write
                .format('delta')
                .mode('overwrite')
                .option('overwriteSchema', 'true')
                .saveAsTable(f'{CATALOG}.{SCHEMA}.regional_claims_forecast'))
            print(f'Regional breakdown saved → {CATALOG}.{SCHEMA}.regional_claims_forecast')

        # ── Stress test scenario comparison ────────────────────────────────────
        _STRESS_LABELS = {
            'stress_corr':    'Systemic Risk (ρ↑)',
            'cat_event':      'Catastrophe (1-in-250yr)',
            'inflation_shock': 'Inflation Shock (+30%)',
        }
        stress_rows = []
        for _sc in STRESS_SCENARIOS:
            _sr     = stress_results[_sc]
            _sc_row = {
                'scenario':            _sc,
                'scenario_label':      _STRESS_LABELS[_sc],
                'total_mean_M':        _agg(_sr, 'total_loss_mean'),
                'var_99_M':            _agg(_sr, 'var_99'),
                'var_995_M':           _agg(_sr, 'var_995'),
                'cvar_99_M':           _agg(_sr, 'cvar_99'),
                'var_995_vs_baseline': (_agg(_sr, 'var_995') / aggregate_var995 - 1.0) * 100,
            }
            stress_rows.append(_sc_row)
            mlflow.log_metric(f'VaR_99_5_{_sc}_M', round(_sc_row['var_995_M'], 2))

        print(f'\n  STRESS TEST COMPARISON  (vs. baseline — {N_TASKS * N_PER_TASK:,} paths/scenario)')
        print('  ' + '─'*75)
        print(f'  {"Scenario":<28} {"Exp.Loss":>10} {"VaR(99%)":>10} {"VaR(99.5%)":>12} {"Δ VaR(99.5%)":>14}')
        print('  ' + '─'*75)
        print(f'  {"[baseline]":<28} ${aggregate_mean:>8.1f}M ${aggregate_var99:>8.1f}M ${aggregate_var995:>10.1f}M  {"—":>12}')
        for _row in stress_rows:
            print(
                f'  {_row["scenario_label"]:<28} '
                f'${_row["total_mean_M"]:>8.1f}M '
                f'${_row["var_99_M"]:>8.1f}M '
                f'${_row["var_995_M"]:>10.1f}M  '
                f'{_row["var_995_vs_baseline"]:>+11.1f}%'
            )
        print('  ' + '─'*75)

        stress_df = spark.createDataFrame(pd.DataFrame(stress_rows))
        (stress_df.write
            .format('delta')
            .mode('overwrite')
            .option('overwriteSchema', 'true')
            .saveAsTable(f'{CATALOG}.{SCHEMA}.stress_test_scenarios'))
        print(f'\nStress scenarios saved → {CATALOG}.{SCHEMA}.stress_test_scenarios')

else:
    # Fallback: single-node t-Copula simulation without Ray (Serverless / no GPU)
    print('\nRunning single-node Monte Carlo (Ray not available on this cluster)\n')
    from scipy.stats import t as tdist, norm as scipy_norm
    N_SCENARIOS  = 100_000
    COPULA_DF    = 4
    rng          = np.random.default_rng(42)
    means        = np.array([12.5, 8.3, 5.7])
    cv           = np.array(_garch_cvs)  # GARCH-derived CVs (or defaults)
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
        if _log_input:
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
# MAGIC ## 6. Reserve Triangle Validation
# MAGIC
# MAGIC The loss development triangle from the DLT pipeline (`gold_reserve_triangle`) provides an
# MAGIC independent check on SARIMA forecasts. We compare the SARIMA-projected incurred claims against
# MAGIC actual reserve development to assess **reserve adequacy**.
# MAGIC
# MAGIC This connects the new CDC pipeline → Module 4, completing the end-to-end lineage.

# COMMAND ----------

try:
    _triangle_table = f"{CATALOG}.{SCHEMA}.gold_reserve_triangle"
    triangle_df = spark.table(_triangle_table)
    _tri_count = triangle_df.count()
    print(f"gold_reserve_triangle: {_tri_count:,} rows")

    # Aggregate triangle to get latest cumulative incurred per segment × accident month
    # (use the maximum dev_lag available for each accident month as the most mature estimate)
    from pyspark.sql import Window as _RW
    _max_lag_win = _RW.partitionBy("segment_id", "accident_month").orderBy(F.col("dev_lag").desc())
    latest_reserves = (
        triangle_df
        .withColumn("_rn", F.row_number().over(_max_lag_win))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .withColumnRenamed("accident_month", "month")
        .withColumn("month", F.to_date("month"))
    )

    # Compare with SARIMA actuals: join on segment_id + month
    try:
        _sarima_actuals = (
            sarima_results_df
            .filter(F.col("record_type") == "actual")
            .select("segment_id", "month", "claims_count")
        )
        reserve_validation = (
            latest_reserves
            .join(_sarima_actuals, on=["segment_id", "month"], how="inner")
            .withColumn("reserve_adequacy_ratio",
                F.when(F.col("claims_count") > 0,
                       F.col("cumulative_incurred") / F.col("claims_count"))
                 .otherwise(F.lit(None).cast("double")))
            .select("segment_id", "month", "cumulative_incurred", "cumulative_paid",
                    "case_reserve", "claims_count", "reserve_adequacy_ratio", "dev_lag")
        )

        (reserve_validation.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"{CATALOG}.{SCHEMA}.reserve_validation"))

        _avg_adequacy = reserve_validation.agg(F.mean("reserve_adequacy_ratio")).collect()[0][0]
        print(f"Reserve validation saved → {CATALOG}.{SCHEMA}.reserve_validation")
        print(f"  Average reserve adequacy ratio: {_avg_adequacy:.2f}")
        print(f"  (>1.0 = over-reserved, <1.0 = under-reserved, ~1.0 = adequate)")

    except NameError:
        print("SARIMA results not available — skipping reserve adequacy comparison")

except Exception as _tri_err:
    print(f"Note: gold_reserve_triangle not available ({_tri_err})")
    print("Reserve validation skipped — run the DLT pipeline first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Registration — MLflow + Unity Catalog
# MAGIC
# MAGIC We register both production models to Unity Catalog so Module 5 can deploy
# MAGIC serving endpoints without re-training:
# MAGIC
# MAGIC 1. **SARIMA Champion** — fitted SARIMAX(1,0,1)(1,1,0,12) for `Personal_Auto__Ontario`
# MAGIC 2. **Monte Carlo Portfolio** — stateless t-Copula + Lognormal simulation
# MAGIC
# MAGIC Both get an `@Champion` alias, which the serving endpoints reference.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7a. SARIMA Champion Model
# MAGIC
# MAGIC Wrap the best SARIMAX model as `mlflow.pyfunc.PythonModel`, log to MLflow,
# MAGIC and register to Unity Catalog with `@Champion` alias.

# COMMAND ----------

import os, pickle, tempfile, cloudpickle, scipy
import statsmodels as _statsmodels
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from mlflow.tracking import MlflowClient

# ── PyFunc wrapper (same interface as Module 5's serving contract) ────────────
class SARIMAXForecaster(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for a fitted statsmodels SARIMAX model.

    Input:  pandas DataFrame with column `horizon` (int) — months to forecast
    Output: pandas DataFrame with columns: month_offset, forecast_mean, lo95, hi95

    This pattern works for any classical Python model: statsmodels, lifelines,
    scikit-survival, arch — wrap it, log it, serve it.
    """

    def load_context(self, context):
        """Load the pickled SARIMAX model from the MLflow artifact store."""
        import pickle
        import os
        model_path = os.path.join(context.artifacts["sarimax_model"], "model.pkl")
        with open(model_path, "rb") as f:
            self.model_fit = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecast.

        Args:
            model_input: DataFrame with column `horizon` (int, 1–24)

        Returns:
            DataFrame with forecast mean and 95% confidence intervals.
        """
        horizon = int(model_input["horizon"].iloc[0])
        horizon = max(1, min(horizon, 24))  # clamp to [1, 24]

        forecast = self.model_fit.get_forecast(steps=horizon)
        mean_fcst = forecast.predicted_mean
        ci        = np.asarray(forecast.conf_int(alpha=0.05))

        return pd.DataFrame({
            "month_offset":   list(range(1, horizon + 1)),
            "forecast_mean":  list(mean_fcst.round(1)),
            "forecast_lo95":  list(np.round(ci[:, 0], 1)),
            "forecast_hi95":  list(np.round(ci[:, 1], 1)),
        })

# ── Train on Personal_Auto__Ontario (highest volume, primary validation segment)
try:
    _reg_claims_pdf = (
        spark.table(f"{CATALOG}.{SCHEMA}.gold_claims_monthly")
        .filter("segment_id = 'Personal_Auto__Ontario'")
        .orderBy("month")
        .select("month", "claims_count")
        .toPandas()
    )
    print(f"Loaded {len(_reg_claims_pdf)} months from gold_claims_monthly")
except Exception:
    print("Generating sample data (gold_claims_monthly not available)")
    np.random.seed(42)
    _months = pd.date_range("2019-01-01", periods=60, freq="MS")
    _SEASONALITY = {1: 1.25, 2: 1.20, 3: 1.10, 4: 0.95, 5: 0.90, 6: 0.88,
                    7: 0.85, 8: 0.87, 9: 0.92, 10: 1.00, 11: 1.10, 12: 1.20}
    _base = 450 * 1.4
    _y = [max(0, _base * (1+0.003*i) * _SEASONALITY[m.month] * (1+np.random.normal(0, 0.08)))
          for i, m in enumerate(_months)]
    _reg_claims_pdf = pd.DataFrame({"month": _months.date, "claims_count": [int(round(v)) for v in _y]})

_y_train = _reg_claims_pdf["claims_count"].astype(float).values

# ── Switch experiment for SARIMA champion registration ────────────────────────
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_sarima_claims_forecaster")

# Dataset reference for UC lineage
try:
    _sarima_reg_dataset = mlflow.data.load_delta(
        table_name=f"{CATALOG}.{SCHEMA}.gold_claims_monthly",
        name="gold_claims_monthly",
    )
    _sarima_reg_log_ds = True
except Exception:
    _sarima_reg_log_ds = False

with mlflow.start_run(run_name="sarima_personal_auto_ontario_champion") as _sarima_reg_run:
    if _sarima_reg_log_ds:
        mlflow.log_input(_sarima_reg_dataset, context="training")

    # ── Fit model ────────────────────────────────────────────────────────────
    _reg_model = SARIMAX(
        _y_train,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    _reg_fit = _reg_model.fit(disp=False, maxiter=200)

    # ── Compute metrics ───────────────────────────────────────────────────────
    _reg_fitted = _reg_fit.fittedvalues[12:]
    _reg_actual = _y_train[12:]
    _reg_mape = float(np.mean(np.abs((_reg_actual - _reg_fitted) / np.clip(_reg_actual, 1, None))) * 100)
    _reg_rmse = float(np.sqrt(np.mean((_reg_actual - _reg_fitted)**2)))

    # ── Log parameters ────────────────────────────────────────────────────────
    mlflow.set_tags({
        "segment_id":      "Personal_Auto__Ontario",
        "workshop_module": "4",
        "model_class":     "SARIMAX",
        "audience":        "actuarial-workshop",
    })
    mlflow.log_params({
        "order_p": 1, "order_d": 0, "order_q": 1,
        "seasonal_P": 1, "seasonal_D": 1, "seasonal_Q": 0,
        "seasonal_m": 12,
        "training_months": len(_y_train),
        "segment": "Personal_Auto__Ontario",
    })
    mlflow.log_metrics({
        "mape_pct": round(_reg_mape, 2),
        "rmse":     round(_reg_rmse, 1),
        "aic":      round(_reg_fit.aic, 2),
        "bic":      round(_reg_fit.bic, 2),
    })

    # ── Save forecast plot as artifact ───────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(12, 4))
    _ax.plot(range(len(_y_train)), _y_train, label="Actual", lw=1.5)
    _ax.plot(range(12, len(_reg_fit.fittedvalues)), _reg_fit.fittedvalues[12:],
             label="Fitted", lw=1.5, ls="--")
    _fc = _reg_fit.get_forecast(steps=12)
    _fc_mean = _fc.predicted_mean
    _fc_ci   = np.asarray(_fc.conf_int())
    _t_fc = range(len(_y_train), len(_y_train) + 12)
    _ax.plot(_t_fc, _fc_mean, label="Forecast (12m)", color="orange", lw=2)
    _ax.fill_between(_t_fc, _fc_ci[:, 0], _fc_ci[:, 1], alpha=0.2, color="orange")
    _ax.set_title("SARIMA(1,0,1)(1,1,0,12) — Personal Auto Ontario")
    _ax.set_xlabel("Month offset")
    _ax.set_ylabel("Monthly Claims Count")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as _tmpdir:
        _plot_path = os.path.join(_tmpdir, "forecast_plot.png")
        _fig.savefig(_plot_path, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(_plot_path, artifact_path="plots")
    plt.close()

    # ── Save pickled model for PyFunc ─────────────────────────────────────────
    _SARIMA_MODEL_NAME = f"{CATALOG}.{SCHEMA}.sarima_claims_forecaster"
    with tempfile.TemporaryDirectory() as _tmpdir:
        _model_pkl_path = os.path.join(_tmpdir, "model.pkl")
        with open(_model_pkl_path, "wb") as _f:
            pickle.dump(_reg_fit, _f)

        _input_schema  = mlflow.types.Schema([mlflow.types.ColSpec("integer", "horizon")])
        _output_schema = mlflow.types.Schema([
            mlflow.types.ColSpec("integer", "month_offset"),
            mlflow.types.ColSpec("double",  "forecast_mean"),
            mlflow.types.ColSpec("double",  "forecast_lo95"),
            mlflow.types.ColSpec("double",  "forecast_hi95"),
        ])
        _signature = mlflow.models.ModelSignature(inputs=_input_schema, outputs=_output_schema)

        mlflow.pyfunc.log_model(
            artifact_path="sarima_forecaster",
            python_model=SARIMAXForecaster(),
            artifacts={"sarimax_model": _tmpdir},
            signature=_signature,
            registered_model_name=_SARIMA_MODEL_NAME,
            pip_requirements=[
                f"statsmodels=={_statsmodels.__version__}",
                f"numpy=={np.__version__}",
                f"scipy=={scipy.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
            ],
        )

    print(f"\nSARIMA model registered to: {_SARIMA_MODEL_NAME}")
    print(f"MAPE: {_reg_mape:.1f}%  |  RMSE: {_reg_rmse:.0f}  |  AIC: {_reg_fit.aic:.1f}")

# ── Set @Champion alias ──────────────────────────────────────────────────────
_client = MlflowClient()
_sarima_versions = _client.search_model_versions(f"name='{_SARIMA_MODEL_NAME}'")
_sarima_latest_ver = max(int(v.version) for v in _sarima_versions)
_client.set_registered_model_alias(name=_SARIMA_MODEL_NAME, alias="Champion", version=_sarima_latest_ver)
_client.set_model_version_tag(name=_SARIMA_MODEL_NAME, version=str(_sarima_latest_ver),
                              key="approved_by", value="actuarial-workshop-demo")
_client.set_model_version_tag(name=_SARIMA_MODEL_NAME, version=str(_sarima_latest_ver),
                              key="segment", value="Personal_Auto__Ontario")
print(f"Set @Champion → version {_sarima_latest_ver}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7b. Monte Carlo Portfolio Model
# MAGIC
# MAGIC The Monte Carlo simulation is **stateless** — all assumptions arrive in the request.
# MAGIC The PyFunc wraps the t-Copula + Lognormal simulation code; no pickle is needed.
# MAGIC
# MAGIC Default CVs are inlined as static values (`0.35, 0.28, 0.42`). The app always
# MAGIC sends all 11 parameters explicitly, so defaults only serve standalone validation.

# COMMAND ----------

class MonteCarloPyFunc(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for t-Copula + Lognormal Marginals Monte Carlo.

    Parameterised simulation: all assumptions arrive in the request, so analysts
    can run stressed scenarios (hard market, cat event, parameter uncertainty)
    without retraining.

    Actuarial design:
      - t-Copula (df=4): captures tail dependence / common shocks between lines
      - Lognormal marginals: consistent with the collective risk model (Panjer,
        Klugman) for skewed, non-negative insurance losses
      - Cholesky decomposition: enforces positive semi-definite correlation structure

    Input:  DataFrame with one row of scenario parameters
    Output: DataFrame with one row of portfolio risk metrics
    """

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from scipy.stats import t as tdist, norm as scipy_norm

        row = model_input.iloc[0]

        # ── Read scenario parameters (with static defaults) ──────────────────
        means = np.array([
            float(row.get("mean_property_M",  12.5)),
            float(row.get("mean_auto_M",       8.3)),
            float(row.get("mean_liability_M",  5.7)),
        ])
        cv = np.array([
            float(row.get("cv_property",  0.35)),
            float(row.get("cv_auto",      0.28)),
            float(row.get("cv_liability", 0.42)),
        ])
        corr_prop_auto = float(row.get("corr_prop_auto",  0.40))
        corr_prop_liab = float(row.get("corr_prop_liab",  0.20))
        corr_auto_liab = float(row.get("corr_auto_liab",  0.30))
        n_scenarios    = int(row.get("n_scenarios", 10_000))
        copula_df      = int(row.get("copula_df",   4))

        # Safety bounds
        n_scenarios = max(1_000, min(n_scenarios, 100_000))
        copula_df   = max(2,     min(copula_df,   30))
        means       = np.clip(means, 0.01, 1_000.0)
        cv          = np.clip(cv,    0.01, 5.0)
        corr_prop_auto = np.clip(corr_prop_auto, -0.99, 0.99)
        corr_prop_liab = np.clip(corr_prop_liab, -0.99, 0.99)
        corr_auto_liab = np.clip(corr_auto_liab, -0.99, 0.99)

        # ── Lognormal parameters ────────────────────────────────────────────
        sigma2   = np.log(1 + cv**2)
        mu_ln    = np.log(means) - sigma2 / 2
        sigma_ln = np.sqrt(sigma2)

        # ── Correlation matrix (Cholesky decomposition) ─────────────────────
        corr = np.array([
            [1.0,            corr_prop_auto, corr_prop_liab],
            [corr_prop_auto, 1.0,            corr_auto_liab],
            [corr_prop_liab, corr_auto_liab, 1.0           ],
        ])
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = np.maximum(eigvals, 1e-8)
            corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
            np.fill_diagonal(corr, 1.0)
            chol = np.linalg.cholesky(corr)

        # ── t-Copula simulation (Sklar's theorem) ──────────────────────────
        rng   = np.random.default_rng(42)
        z     = rng.standard_normal((n_scenarios, 3))
        chi2  = rng.chisquare(copula_df, n_scenarios)
        x_cor = z @ chol.T
        t_cor = x_cor / np.sqrt(chi2[:, None] / copula_df)
        u     = tdist.cdf(t_cor, df=copula_df)
        q     = scipy_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        losses = np.exp(mu_ln + sigma_ln * q)
        total  = losses.sum(axis=1)

        # ── Risk metrics ────────────────────────────────────────────────────
        var_99_threshold = np.percentile(total, 99)

        return pd.DataFrame([{
            "expected_loss_M":  round(float(total.mean()), 3),
            "var_95_M":         round(float(np.percentile(total, 95)),  3),
            "var_99_M":         round(float(np.percentile(total, 99)),  3),
            "var_995_M":        round(float(np.percentile(total, 99.5)), 3),
            "cvar_99_M":        round(float(total[total >= var_99_threshold].mean()), 3),
            "max_loss_M":       round(float(total.max()), 3),
            "n_scenarios_used": n_scenarios,
            "copula":           f"t-copula(df={copula_df})",
        }])

# ── Validate locally with baseline scenario ──────────────────────────────────
_mc_baseline_input = pd.DataFrame([{
    "mean_property_M": 12.5, "mean_auto_M": 8.3, "mean_liability_M": 5.7,
    "cv_property": _garch_cvs[0] if isinstance(_garch_cvs, list) else float(_garch_cvs[0]),
    "cv_auto":     _garch_cvs[1] if isinstance(_garch_cvs, list) else float(_garch_cvs[1]),
    "cv_liability":_garch_cvs[2] if isinstance(_garch_cvs, list) else float(_garch_cvs[2]),
    "corr_prop_auto": 0.40, "corr_prop_liab": 0.20, "corr_auto_liab": 0.30,
    "n_scenarios": 10_000, "copula_df": 4,
}])

_mc_pyfunc = MonteCarloPyFunc()
_mc_baseline_result = _mc_pyfunc.predict(None, _mc_baseline_input)
print("Baseline Monte Carlo validation (10,000 scenarios):")
print(_mc_baseline_result.to_string(index=False))

# COMMAND ----------

# ── Register Monte Carlo model to UC ─────────────────────────────────────────
_MC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.monte_carlo_portfolio"
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_monte_carlo_portfolio")

try:
    _mc_reg_dataset = mlflow.data.load_delta(
        table_name=f"{CATALOG}.{SCHEMA}.gold_claims_monthly",
        name="gold_claims_monthly",
    )
    _mc_reg_log_ds = True
except Exception:
    _mc_reg_log_ds = False

# ── Define MLflow signature ──────────────────────────────────────────────────
_mc_input_schema = mlflow.types.Schema([
    mlflow.types.ColSpec("double",  "mean_property_M"),
    mlflow.types.ColSpec("double",  "mean_auto_M"),
    mlflow.types.ColSpec("double",  "mean_liability_M"),
    mlflow.types.ColSpec("double",  "cv_property"),
    mlflow.types.ColSpec("double",  "cv_auto"),
    mlflow.types.ColSpec("double",  "cv_liability"),
    mlflow.types.ColSpec("double",  "corr_prop_auto"),
    mlflow.types.ColSpec("double",  "corr_prop_liab"),
    mlflow.types.ColSpec("double",  "corr_auto_liab"),
    mlflow.types.ColSpec("long",    "n_scenarios"),
    mlflow.types.ColSpec("long",    "copula_df"),
])
_mc_output_schema = mlflow.types.Schema([
    mlflow.types.ColSpec("double", "expected_loss_M"),
    mlflow.types.ColSpec("double", "var_95_M"),
    mlflow.types.ColSpec("double", "var_99_M"),
    mlflow.types.ColSpec("double", "var_995_M"),
    mlflow.types.ColSpec("double", "cvar_99_M"),
    mlflow.types.ColSpec("double", "max_loss_M"),
    mlflow.types.ColSpec("long",   "n_scenarios_used"),
    mlflow.types.ColSpec("string", "copula"),
])
_mc_signature = mlflow.models.ModelSignature(inputs=_mc_input_schema, outputs=_mc_output_schema)

with mlflow.start_run(run_name="monte_carlo_portfolio_champion") as _mc_reg_run:
    if _mc_reg_log_ds:
        mlflow.log_input(_mc_reg_dataset, context="training")

    mlflow.set_tags({
        "model_class":     "MonteCarloPyFunc",
        "copula":          "t-copula",
        "marginals":       "lognormal",
        "workshop_module": "4",
        "audience":        "actuarial-workshop",
    })
    mlflow.log_params({
        "copula_df":              4,
        "n_lines":                3,
        "default_n_scenarios":    10_000,
        "mean_property_M_base":   12.5,
        "mean_auto_M_base":        8.3,
        "mean_liability_M_base":   5.7,
        "cv_property_base":       _garch_cvs[0] if isinstance(_garch_cvs, list) else float(_garch_cvs[0]),
        "cv_auto_base":           _garch_cvs[1] if isinstance(_garch_cvs, list) else float(_garch_cvs[1]),
        "cv_liability_base":      _garch_cvs[2] if isinstance(_garch_cvs, list) else float(_garch_cvs[2]),
        "cv_source":              "GARCH(1,1)" if _MC_USE_GARCH_CV else "static",
        "corr_prop_auto_base":    0.40,
        "corr_prop_liab_base":    0.20,
        "corr_auto_liab_base":    0.30,
    })

    _mc_r = _mc_baseline_result.iloc[0]
    mlflow.log_metrics({
        "baseline_expected_loss_M":  float(_mc_r["expected_loss_M"]),
        "baseline_var_95_M":         float(_mc_r["var_95_M"]),
        "baseline_var_99_M":         float(_mc_r["var_99_M"]),
        "baseline_var_995_M":        float(_mc_r["var_995_M"]),
        "baseline_cvar_99_M":        float(_mc_r["cvar_99_M"]),
    })

    mlflow.pyfunc.log_model(
        artifact_path="monte_carlo_pyfunc",
        python_model=MonteCarloPyFunc(),
        signature=_mc_signature,
        registered_model_name=_MC_MODEL_NAME,
        pip_requirements=[
            f"scipy=={scipy.__version__}",
            f"numpy=={np.__version__}",
        ],
    )

    print(f"\nMonte Carlo model registered to: {_MC_MODEL_NAME}")
    print(f"Baseline VaR(99%): ${_mc_r['var_99_M']:.1f}M  |  CVaR(99%): ${_mc_r['cvar_99_M']:.1f}M")

# ── Set @Champion alias ──────────────────────────────────────────────────────
_mc_versions = _client.search_model_versions(f"name='{_MC_MODEL_NAME}'")
_mc_latest_ver = max(int(v.version) for v in _mc_versions)
_client.set_registered_model_alias(name=_MC_MODEL_NAME, alias="Champion", version=_mc_latest_ver)
_client.set_model_version_tag(name=_MC_MODEL_NAME, version=str(_mc_latest_ver),
                              key="approved_by", value="actuarial-workshop-demo")
_client.set_model_version_tag(name=_MC_MODEL_NAME, version=str(_mc_latest_ver),
                              key="simulation_type", value="t-copula-lognormal")
print(f"Set @Champion → version {_mc_latest_ver}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Technique | Framework | Scale | Use Case |
# MAGIC |---|---|---|---|
# MAGIC | SARIMA(1,0,1)(1,1,0,12) | statsmodels + applyInPandas | 40 segments × 72 months | Baseline claim volume forecast |
# MAGIC | SARIMAX(1,0,1)(1,1,0,12) | statsmodels + applyInPandas | 40 segments + macro + FS exog | Forecast with StatCan + Feature Store signals |
# MAGIC | GARCH(1,1) | arch + applyInPandas | 40 segments | Loss ratio volatility → MC CVs |
# MAGIC | Monte Carlo — baseline | Ray + PyTorch GPU | 40M paths (4 tasks × 10M) | VaR(99.5%), CVaR, SCR — GARCH-calibrated |
# MAGIC | Monte Carlo — VaR evolution | Ray + PyTorch GPU (SARIMAX-driven) | 480M paths (12 months × 4 × 10M) | Forward VaR, regional breakdown |
# MAGIC | Monte Carlo — stress tests | Ray + PyTorch GPU (3 scenarios) | 120M paths (3 × 4 × 10M) | CAT event, systemic risk, inflation shock |
# MAGIC | Reserve validation | Spark join | Triangle × SARIMA | Reserve adequacy vs. forecasted claims |
# MAGIC | Model registration | MLflow + UC | 2 models (SARIMA + MC) | `@Champion` alias for serving |
# MAGIC
# MAGIC **Data lineage:** `gold_claims_monthly` (DLT pipeline) → `silver_rolling_features` → Module 3 → `segment_monthly_features` → SARIMAX exog vars
# MAGIC
# MAGIC **GARCH → MC:** Fitted volatilities from GARCH(1,1) provide data-driven CVs for Monte Carlo (replacing hardcoded values)
# MAGIC
# MAGIC **Next:** Module 5 — Create Model Serving endpoints for both models and configure AI Gateway.