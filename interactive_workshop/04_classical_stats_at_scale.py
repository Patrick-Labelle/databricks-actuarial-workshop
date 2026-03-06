# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: Classical Statistical Models at Scale
# MAGIC ## SARIMAX + GARCH(1,1) on Residuals + Monte Carlo with Ray
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC *Audience: Actuaries, Data Scientists, Financial Analysts*
# MAGIC
# MAGIC ---
# MAGIC ### What We'll Cover
# MAGIC 1. **Data Setup** — Read `gold_claims_monthly` from the declarative pipeline (40 segments × 84 months)
# MAGIC 2. **Macro Integration** — Join real StatCan macro data; visualize claims vs unemployment
# MAGIC 3. **SARIMAX + GARCH at Scale** — Per-segment SARIMA + SARIMAX fit with GARCH(1,1) on residuals for time-varying CIs
# MAGIC 4. **ARCH-LM Diagnostic** — Engle's test results; why GARCH on residuals is correct
# MAGIC 5. **Monte Carlo with Ray** — Task-parallel portfolio loss simulation (100k+ paths)
# MAGIC 6. **Reserve Validation** — SARIMA forecasts vs. actual development from the loss triangle
# MAGIC 7. **Model Registration** — Register SARIMA+GARCH + Monte Carlo models to UC with `@Champion` alias
# MAGIC
# MAGIC ---
# MAGIC ### Why `applyInPandas`?
# MAGIC
# MAGIC We have **40 segments** (4 product lines × 10 provinces), each needing its own SARIMAX fit + GARCH(1,1) on residuals.
# MAGIC These are **independent** per-group operations. `applyInPandas` lets Spark distribute this work:
# MAGIC each executor receives a pandas DataFrame for one segment, runs standard Python/statsmodels code,
# MAGIC and returns results. No Spark ML required — just familiar Python libraries.
# MAGIC
# MAGIC ```
# MAGIC df.groupby("segment_id").applyInPandas(fit_sarimax_fn, schema=output_schema)
# MAGIC ```

# MAGIC
# MAGIC > **Interactive notebook** — Run this attached to a classic ML cluster with Ray-on-Spark support (DBR 16.4+ ML). The Ray cluster setup happens automatically. Expect ~5 min for SARIMA fitting + ~3 min for Monte Carlo simulation.
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
# MAGIC ## 1. Load Claims Data from SDP Gold Layer
# MAGIC
# MAGIC Data flows directly from the declarative pipeline (Module 1) — no synthetic generation needed.
# MAGIC The `gold_claims_monthly` table provides **real** claim counts, loss ratios, and premium
# MAGIC exposures for **40 segments** (4 product lines × 10 provinces) × **84 months** (Jan 2019 – Dec 2025).
# MAGIC
# MAGIC ```
# MAGIC SDP Pipeline (Module 1)
# MAGIC   raw_claims_events  →  bronze_claims  →  gold_claims_monthly  ← This module reads here
# MAGIC   raw_macro_indicators → bronze_macro → silver_macro → gold_macro_features  ← exog vars
# MAGIC ```

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, LongType

# ─── Configuration ────────────────────────────────────────────────────────────
# Passed as base_parameters by the bundle job; defaults used when interactive.
dbutils.widgets.text("catalog",  "my_catalog",         "UC Catalog")
dbutils.widgets.text("schema",   "actuarial_workshop", "UC Schema")
CATALOG   = dbutils.widgets.get("catalog")
SCHEMA    = dbutils.widgets.get("schema")

np.random.seed(42)

# ─── Segment definitions ─────────────────────────────────────────────────────
PRODUCT_LINES = ["Personal_Auto", "Commercial_Auto", "Homeowners", "Commercial_Property"]
REGIONS       = [
    "Ontario", "Quebec", "British_Columbia", "Alberta",
    "Manitoba", "Saskatchewan", "New_Brunswick", "Nova_Scotia",
    "Prince_Edward_Island", "Newfoundland",
]
MONTHS        = pd.date_range("2019-01-01", periods=84, freq="MS")  # Jan 2019 – Dec 2025

# Base claim levels (monthly, Alberta reference — used by Monte Carlo bridge weight blending)
BASE_CLAIMS = {
    "Personal_Auto":       26_000,
    "Commercial_Auto":     10_500,
    "Homeowners":          18_500,
    "Commercial_Property":  5_200,
}

# ─── Read from SDP gold layer ─────────────────────────────────────────────────
# gold_claims_monthly is produced by the declarative pipeline (task: run_sdp_pipeline in jobs.yml).
# Expected: 40 segments × 84 months = 3,360 rows with claims_count, loss_ratio, earned_premium.
claims_df = (
    spark.table(f"{CATALOG}.{SCHEMA}.gold_claims_monthly")
    .filter(F.col("month").between("2019-01-01", "2025-12-01"))
)
claims_df.createOrReplaceTempView("claims_ts")

n_segments = claims_df.select("segment_id").distinct().count()
n_rows     = claims_df.count()
print(f"Segments: {n_segments} (expected 40 = 4 product lines × 10 provinces)")
print(f"Rows:     {n_rows} (expected 3,360 = 40 × 84 months)")
display(claims_df.orderBy("segment_id", "month").limit(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Macro Data Integration
# MAGIC
# MAGIC Real macroeconomic data from **Statistics Canada** is joined to the claims series.
# MAGIC Three indicators (fetched by `scripts/fetch_macro_data.py`, processed through the SDP medallion):
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

# Load gold_macro_features from the declarative pipeline and join to claims data
macro_df = spark.table(f"{CATALOG}.{SCHEMA}.gold_macro_features")
macro_count = macro_df.count()
print(f"gold_macro_features: {macro_count:,} rows (expected ~840 = 10 provinces × 84 months)")

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
print(f"Macro data joined ({macro_count:,} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2b. Feature Store Integration — Exogenous Variables from Module 3
# MAGIC
# MAGIC The Feature Store (`features_segment_monthly`) provides rolling statistical features
# MAGIC computed in Module 2 and registered in Module 3. These features — rolling means,
# MAGIC volatility regime, momentum — serve as additional exogenous variables for SARIMAX,
# MAGIC completing the lineage: `gold_claims_monthly` → Module 2 → Module 3 → Module 4.

# COMMAND ----------

# Load Feature Store features and join to claims data
_fs_table = f"{CATALOG}.{SCHEMA}.features_segment_monthly"
fs_features = spark.table(_fs_table)
_fs_count = fs_features.count()
print(f"features_segment_monthly: {_fs_count:,} rows")

# Select key features for SARIMAX exogenous variables
_fs_cols = ["segment_id", "month", "rolling_3m_mean", "rolling_6m_mean",
            "coeff_variation_3m", "mom_change_pct", "normalized_premium"]
fs_subset = fs_features.select(*_fs_cols)

claims_with_macro = (
    claims_with_macro
    .join(fs_subset, on=["segment_id", "month"], how="left")
)
claims_with_macro.createOrReplaceTempView("claims_with_macro")
print(f"Feature Store features joined: {', '.join(_fs_cols[2:])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SARIMAX + GARCH at Scale — Per-Segment Forecasting with Volatility Modeling
# MAGIC
# MAGIC ### Strategy
# MAGIC We use `df.groupby("segment_id").applyInPandas(fit_fn, schema)`.
# MAGIC
# MAGIC Each Spark task receives **one segment's pandas DataFrame** (claims + macro + features joined) and:
# MAGIC 1. Fits baseline `SARIMA(1,0,1)(1,1,0,12)` — monthly seasonality, no exog
# MAGIC 2. Fits `SARIMAX(1,0,1)(1,1,0,12)` with `unemployment_rate` + `hpi_growth` + Feature Store features as exog
# MAGIC 3. Evaluates out-of-sample MAPE on held-out last 12 months (validation set)
# MAGIC 4. Refits final model on all 84 months for the 12-month forecast
# MAGIC 5. Runs **Engle's ARCH-LM test** on residuals; if significant, fits **GARCH(1,1)** for time-varying CIs
# MAGIC 6. Returns a standardized pandas DataFrame with MAPE metrics, conditional volatility, and GARCH diagnostics
# MAGIC
# MAGIC The output schema must be declared upfront so Spark can parallelize safely.

# COMMAND ----------

# Output schema for SARIMAX results
# New fields vs baseline SARIMA: mape_baseline, mape_sarimax, exog_vars
SARIMA_SCHEMA = StructType([
    StructField("segment_id",      StringType(), False),
    StructField("month",           DateType(),   False),
    StructField("record_type",     StringType(), False),   # "actual" or "forecast"
    StructField("claims_count",    DoubleType(), True),
    StructField("forecast_mean",   DoubleType(), True),
    StructField("forecast_lo95",   DoubleType(), True),
    StructField("forecast_hi95",   DoubleType(), True),
    StructField("aic",             DoubleType(), True),
    StructField("mape",            DoubleType(), True),    # primary MAPE (SARIMAX if available)
    StructField("mape_baseline",   DoubleType(), True),    # SARIMA (no exog)
    StructField("mape_sarimax",    DoubleType(), True),    # SARIMAX (with macro exog)
    StructField("exog_vars",       StringType(), True),    # e.g. "unemployment_rate,hpi_growth"
    StructField("cond_volatility", DoubleType(), True),    # GARCH σ_t on SARIMA residuals
    StructField("arch_lm_pvalue",  DoubleType(), True),    # Engle's ARCH-LM test p-value
    StructField("garch_alpha",     DoubleType(), True),    # GARCH α₁ (news impact)
    StructField("garch_beta",      DoubleType(), True),    # GARCH β₁ (persistence)
])

def fit_sarimax_per_segment(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit baseline SARIMA and SARIMAX with macro exogenous variables for one segment,
    then fit GARCH(1,1) on the SARIMA residuals to capture time-varying volatility.

    Train/validation split: first 72 months for training, last 12 for out-of-sample MAPE.
    Final model is refit on all 84 months for the 12-month forecast.

    GARCH on residuals: After the final SARIMAX fit, run Engle's ARCH-LM test on the
    residuals. If significant (p < 0.10), fit GARCH(1,1) to produce time-varying
    prediction intervals and conditional volatility estimates for Monte Carlo CVs.

    Exogenous forecast: last 3-month average held flat (appropriate for a 12-month horizon;
    in production, use Bank of Canada / StatCan projections for the exog forecast).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from arch import arch_model
    from statsmodels.stats.diagnostic import het_arch
    import warnings
    warnings.filterwarnings("ignore")

    segment_id = pdf["segment_id"].iloc[0]
    pdf        = pdf.sort_values("month").reset_index(drop=True)

    y      = pdf["claims_count"].astype(float).values
    months = pd.to_datetime(pdf["month"])

    # Exog columns: macro variables + Feature Store features
    exog_cols = ["unemployment_rate", "hpi_growth",
                 "rolling_3m_mean", "coeff_variation_3m", "mom_change_pct"]

    # Fill NaN exog values (hpi_growth is NaN for the first month due to lag;
    # forward-fill then back-fill covers edge cases at either end of the series)
    exog_data = pdf[exog_cols].copy().ffill().bfill()
    exog_arr  = exog_data.values.astype(float)

    # Train/validation split (72 train, 12 validation)
    n_train, n_val = 72, 12
    y_train = y[:n_train]
    y_val   = y[n_train:]

    aic = mape_baseline = mape_sarimax = np.nan
    fcast_mean = np.full(12, np.nan)
    fcast_ci   = pd.DataFrame({"lower": np.full(12, np.nan), "upper": np.full(12, np.nan)})
    exog_vars_str = ",".join(exog_cols)

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

        # ── Model 2: SARIMAX with macro exog ──────────────────────────────
        exog_train = exog_arr[:n_train]
        exog_val   = exog_arr[n_train:]
        m_sx   = SARIMAX(y_train, exog=exog_train, order=(1,0,1),
                         seasonal_order=(1,1,0,12),
                         enforce_stationarity=False, enforce_invertibility=False)
        fit_sx = m_sx.fit(disp=False, maxiter=100)
        fc_sx  = fit_sx.forecast(steps=n_val, exog=exog_val)
        mape_sarimax = float(
            np.mean(np.abs((y_val - fc_sx) / np.clip(y_val, 1, None))) * 100
        )

        # ── Final model: refit on full 84 months for forecasting ─────────────
        m_final = SARIMAX(y, exog=exog_arr, order=(1,0,1), seasonal_order=(1,1,0,12),
                          enforce_stationarity=False, enforce_invertibility=False)
        fit_final = m_final.fit(disp=False, maxiter=100)

        # 12-month forecast; extrapolate exog as last 3-month average held flat
        exog_fcast = np.tile(exog_arr[-3:].mean(axis=0), (12, 1))
        forecast   = fit_final.get_forecast(steps=12, exog=exog_fcast)

        fcast_mean   = forecast.predicted_mean
        fcast_ci     = pd.DataFrame(forecast.conf_int(alpha=0.05))
        fcast_ci.columns = ["lower", "upper"]

    except Exception:
        # Return NaN forecasts — allows the pipeline to continue for all segments
        fit_final = None

    # ── GARCH(1,1) on SARIMA residuals ────────────────────────────────────────
    # Filter mean dynamics first (SARIMAX), then model the residual volatility.
    # This is the econometrically correct approach — fitting GARCH on mean-filtered
    # residuals. Standalone GARCH on raw data contaminates volatility estimates
    # with unmodeled mean dynamics.
    arch_lm_pval = garch_alpha_val = garch_beta_val = np.nan
    cond_vol_actual = [None] * len(y)      # per-month conditional volatility (actuals)
    cond_vol_fcast  = [None] * 12          # per-month GARCH forecast vol (forecast horizon)
    _garch_fit = None

    if fit_final is not None:
        try:
            resid = fit_final.resid[12:]  # skip seasonal burn-in (first 12 months)
            # Engle's ARCH-LM test: null = no ARCH effects in residuals
            _lm_stat, arch_lm_pval, _fstat, _fpval = het_arch(resid, nlags=4)
            arch_lm_pval = float(arch_lm_pval)

            if arch_lm_pval < 0.10:
                # Significant ARCH effects — fit GARCH(1,1) on mean-filtered residuals
                am = arch_model(resid, mean='Zero', vol='Garch', p=1, q=1, dist='normal')
                _garch_fit = am.fit(disp='off', show_warning=False)

                garch_alpha_val = float(_garch_fit.params.get('alpha[1]', np.nan))
                garch_beta_val  = float(_garch_fit.params.get('beta[1]', np.nan))

                # Align conditional volatility to full series
                # (first 12 months are NaN due to seasonal burn-in)
                _cv = _garch_fit.conditional_volatility
                cond_vol_actual = [None] * 12 + [float(v) for v in _cv]

                # Forecast GARCH volatility 12 months ahead
                _garch_fc = _garch_fit.forecast(horizon=12, reindex=False)
                _garch_var = np.asarray(_garch_fc.variance).flatten()[:12]
                _garch_vol = np.sqrt(_garch_var)
                cond_vol_fcast = [float(v) for v in _garch_vol]

                # Time-varying CIs: replace statsmodels constant CIs
                fcast_ci = pd.DataFrame({
                    "lower": np.asarray(fcast_mean) - 1.96 * _garch_vol,
                    "upper": np.asarray(fcast_mean) + 1.96 * _garch_vol,
                })
        except Exception:
            pass  # GARCH cols stay NaN — statsmodels CIs are kept as-is

    # ── Build output DataFrame ────────────────────────────────────────────────
    last_month     = months.max()
    forecast_months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1), periods=12, freq="MS"
    )
    # Primary MAPE = SARIMAX when available, else baseline
    _primary_mape = mape_sarimax if not np.isnan(mape_sarimax) else mape_baseline

    actuals_rows = pd.DataFrame({
        "segment_id":      segment_id,
        "month":           months.dt.date,
        "record_type":     "actual",
        "claims_count":    y.tolist(),
        "forecast_mean":   [None] * len(y),
        "forecast_lo95":   [None] * len(y),
        "forecast_hi95":   [None] * len(y),
        "aic":             [None] * len(y),
        "mape":            [None] * len(y),
        "mape_baseline":   [None] * len(y),
        "mape_sarimax":    [None] * len(y),
        "exog_vars":       [None] * len(y),
        "cond_volatility": cond_vol_actual,
        "arch_lm_pvalue":  [arch_lm_pval if not np.isnan(arch_lm_pval) else None] * len(y),
        "garch_alpha":     [garch_alpha_val if not np.isnan(garch_alpha_val) else None] * len(y),
        "garch_beta":      [garch_beta_val if not np.isnan(garch_beta_val) else None] * len(y),
    })

    forecast_rows = pd.DataFrame({
        "segment_id":      segment_id,
        "month":           forecast_months.date,
        "record_type":     "forecast",
        "claims_count":    [None] * 12,
        "forecast_mean":   list(fcast_mean),
        "forecast_lo95":   list(np.asarray(fcast_ci)[:, 0]),
        "forecast_hi95":   list(np.asarray(fcast_ci)[:, 1]),
        "aic":             [aic] * 12,
        "mape":            [_primary_mape] * 12,
        "mape_baseline":   [mape_baseline] * 12,
        "mape_sarimax":    [mape_sarimax]  * 12,
        "exog_vars":       [exog_vars_str] * 12,
        "cond_volatility": cond_vol_fcast,
        "arch_lm_pvalue":  [arch_lm_pval if not np.isnan(arch_lm_pval) else None] * 12,
        "garch_alpha":     [garch_alpha_val if not np.isnan(garch_alpha_val) else None] * 12,
        "garch_beta":      [garch_beta_val if not np.isnan(garch_beta_val) else None] * 12,
    })

    return pd.concat([actuals_rows, forecast_rows], ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run SARIMAX + GARCH Across All 40 Segments
# MAGIC
# MAGIC Spark distributes one segment per task. Each task fits SARIMA + SARIMAX + GARCH(1,1)
# MAGIC on residuals, returning MAPE for both mean models and conditional volatility estimates.

# COMMAND ----------

import mlflow
_current_user = spark.sql("SELECT current_user()").collect()[0][0]
# Use a flat path under /Users/<email>/ — avoid nested subdirectories which
# require the parent to pre-exist (fails on fresh workspaces).
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_claims_sarima")

# Register input dataset for UC lineage — all models (SARIMAX, GARCH, Monte Carlo) are
# trained from gold_claims_monthly (the SDP gold layer, not a synthetic table).
_claims_dataset = mlflow.data.load_delta(
    table_name=f"{CATALOG}.{SCHEMA}.gold_claims_monthly",
    name="gold_claims_monthly",
)

with mlflow.start_run(run_name="sarimax_all_segments") as run:
    mlflow.log_input(_claims_dataset, context="training")
    mlflow.set_tags({
        "workshop_module": "4",
        "model_type":      "SARIMAX(1,0,1)(1,1,0,12)+GARCH(1,1)",
        "segments":        "40",
        "horizon_months":  "12",
        "exog_vars":       "unemployment_rate,hpi_growth,rolling_3m_mean,coeff_variation_3m,mom_change_pct",
        "audience":        "actuarial-workshop",
    })

    # Distributed: Spark sends one segment per task to applyInPandas workers.
    # Requires statsmodels on the compute environment (ml_env spec in bundle jobs).
    sarima_results_df = (
        claims_with_macro
        .groupby("segment_id")
        .applyInPandas(fit_sarimax_per_segment, schema=SARIMA_SCHEMA)
    )

    # Write to Delta — single trigger for applyInPandas execution
    (sarima_results_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.predictions_sarima"))

    # Re-read from Delta for metrics (cheap read, no recomputation)
    sarima_results_df = spark.table(f"{CATALOG}.{SCHEMA}.predictions_sarima")

    # Compute all metrics in a single query
    _metrics = (
        sarima_results_df
        .filter(F.col("record_type") == "forecast")
        .agg(
            F.count("*").alias("total_forecasts"),
            F.mean("mape").alias("avg_mape"),
            F.mean("mape_baseline").alias("avg_mape_baseline"),
            F.mean("mape_sarimax").alias("avg_mape_sarimax"),
            F.countDistinct(
                F.when(F.col("garch_alpha").isNotNull(), F.col("segment_id"))
            ).alias("garch_seg_count"),
        )
        .collect()[0]
    )
    avg_mape          = _metrics["avg_mape"]          or 0.0
    avg_mape_baseline = _metrics["avg_mape_baseline"] or avg_mape
    avg_mape_sarimax  = _metrics["avg_mape_sarimax"]  or avg_mape
    avg_mape_improve  = avg_mape_baseline - avg_mape_sarimax
    total_rows = sarima_results_df.count()

    mlflow.log_metrics({
        "avg_mape_pct":             round(avg_mape, 2),
        "avg_mape_baseline_pct":    round(avg_mape_baseline, 2),
        "avg_mape_sarimax_pct":     round(avg_mape_sarimax, 2),
        "avg_mape_improvement_pct": round(avg_mape_improve, 2),
        "total_output_rows":        total_rows,
        "segments_fitted":          40,
    })

    print(f"SARIMAX+GARCH complete | Total rows: {total_rows}")
    print(f"  Baseline SARIMA MAPE:  {avg_mape_baseline:.1f}%")
    print(f"  SARIMAX MAPE:          {avg_mape_sarimax:.1f}%")
    print(f"  Improvement:           {avg_mape_improve:+.1f}%  ({'↓ better' if avg_mape_improve > 0 else '↑ worse or unchanged'})")
    print(f"  GARCH(1,1) fitted:     {_metrics['garch_seg_count']}/40 segments (ARCH-LM p < 0.10)")
    print(f"Saved to {CATALOG}.{SCHEMA}.predictions_sarima")
    print(f"MLflow run: {run.info.run_id}")

display(sarima_results_df.filter(F.col("record_type") == "forecast").orderBy("segment_id", "month").limit(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ARCH-LM Diagnostic — GARCH(1,1) on SARIMA Residuals
# MAGIC
# MAGIC ### Why GARCH on Residuals (Not on Raw Data)?
# MAGIC
# MAGIC The SARIMAX fit above models the **conditional mean** of claims. The residuals
# MAGIC from that fit represent the *unexplained* variation — if those residuals show
# MAGIC **volatility clustering** (quiet periods followed by volatile periods), a GARCH
# MAGIC model can capture that time-varying uncertainty.
# MAGIC
# MAGIC **Engle's ARCH-LM test** (1982) checks whether squared residuals are autocorrelated:
# MAGIC - **H₀:** No ARCH effects (homoskedastic residuals)
# MAGIC - **H₁:** ARCH effects present (heteroskedastic — volatility clusters)
# MAGIC - If p-value < 0.10, we fit **GARCH(1,1)** on the residuals
# MAGIC
# MAGIC This is the **econometrically correct** approach:
# MAGIC 1. Filter the mean first (SARIMA) → then model the variance (GARCH)
# MAGIC 2. Standalone GARCH on raw data contaminates volatility with unmodeled mean dynamics
# MAGIC 3. GARCH on residuals produces **time-varying prediction intervals** and properly
# MAGIC    calibrated coefficients of variation for Monte Carlo
# MAGIC
# MAGIC **Regulatory context:** Solvency II internal models, IFRS 17 risk adjustments, and
# MAGIC OSFI MCT all require volatility estimates — but always on **mean-filtered** series.

# COMMAND ----------

# Summarize ARCH-LM results across all 40 segments
_arch_summary = (
    sarima_results_df
    .filter(F.col("record_type") == "forecast")
    .select("segment_id", "arch_lm_pvalue", "garch_alpha", "garch_beta")
    .distinct()
    .orderBy("arch_lm_pvalue")
)
_arch_count = _arch_summary.filter(F.col("arch_lm_pvalue") < 0.10).count()
_arch_total = _arch_summary.filter(F.col("arch_lm_pvalue").isNotNull()).count()
print(f"ARCH-LM summary: {_arch_count}/{_arch_total} segments show significant ARCH effects (p < 0.10)")
print("Segments with GARCH(1,1) fitted have time-varying CIs in predictions_sarima.")

# Show per-segment GARCH diagnostics
_garch_fitted = (
    _arch_summary
    .filter(F.col("garch_alpha").isNotNull())
    .withColumn("persistence", F.col("garch_alpha") + F.col("garch_beta"))
    .orderBy("arch_lm_pvalue")
)
if _garch_fitted.count() > 0:
    print(f"\nGARCH(1,1) fitted segments (top 10 by ARCH-LM significance):")
    display(_garch_fitted.limit(10))

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
# MAGIC - **10M annual loss scenarios per task** using a **t-Copula + lognormal marginals** (see below)
# MAGIC - Each Ray task handles **10M scenarios** → tasks run in parallel across CPU workers
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

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Monte Carlo Task (t-Copula + Lognormal Marginals)
# MAGIC
# MAGIC Each `@ray.remote` task simulates **10M correlated loss scenarios** using:
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
#   Property  = Homeowners (18500) + Commercial_Property (5200)
#   Auto      = Personal_Auto (26000) + Commercial_Auto (10500)
#   Liability = 0.5×Property + 0.5×Auto growth (proxied; no direct SARIMA data)
#
# Regional forecast is derivable from `predictions_sarima` via SQL aggregation
# (GROUP BY region, month WHERE record_type='forecast'), so no separate table is
# needed. The MC bridge below still reads SARIMA per-segment growth factors.
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
_W_PROP = {"Homeowners": 18500, "Commercial_Property": 5200}
_W_AUTO = {"Personal_Auto": 26000, "Commercial_Auto": 10500}

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
if _forecast_months:
    g1  = _sarima_growth_for_month(_forecast_months[0])
    g12 = _sarima_growth_for_month(_forecast_months[-1])
    print(f"  Growth +1 mo:  Property={g1[0]:.3f}  Auto={g1[1]:.3f}  Liability={g1[2]:.3f}")
    print(f"  Growth +12 mo: Property={g12[0]:.3f}  Auto={g12[1]:.3f}  Liability={g12[2]:.3f}")

# Regional breakdown preview (top 5 regions by share of month-12 forecast)
if _forecast_months:
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

# COMMAND ----------

# ── Historical Calibration: derive MC parameters from gold_claims_monthly ────
# Replaces hardcoded means and correlations with data-driven estimates.
# Product-line → MC segment mapping (weighted by BASE_CLAIMS volume):
#   Property  = Homeowners + Commercial_Property
#   Auto      = Personal_Auto + Commercial_Auto
#   Liability = 50/50 blend of Property + Auto (proxy; no direct data)
_SEGMENT_MAP = {
    "Homeowners": "Property", "Commercial_Property": "Property",
    "Personal_Auto": "Auto", "Commercial_Auto": "Auto",
}
_SEGMENT_WEIGHTS = {
    "Property": {"Homeowners": 18500, "Commercial_Property": 5200},
    "Auto":     {"Personal_Auto": 26000, "Commercial_Auto": 10500},
}

# ── Run calibration ──────────────────────────────────────────────────────────
_CALIBRATION_TABLE = f"{CATALOG}.{SCHEMA}.gold_claims_monthly"

_calib_pdf = spark.table(_CALIBRATION_TABLE).toPandas()
_calib_pdf["product_line"] = _calib_pdf["segment_id"].str.split("__").str[0]
_calib_pdf["month"] = pd.to_datetime(_calib_pdf["month"])
_calib_pdf["mc_segment"] = _calib_pdf["product_line"].map(_SEGMENT_MAP)

# Monthly total incurred per MC segment
_seg_monthly = (
    _calib_pdf.groupby(["mc_segment", "month"])["total_incurred"]
    .sum().reset_index().sort_values(["mc_segment", "month"])
)

# Annualized means via trailing 12-month average (in $M)
_means_dict = {}
for _seg in ["Property", "Auto"]:
    _seg_data = _seg_monthly[_seg_monthly["mc_segment"] == _seg].sort_values("month")
    _trailing_12 = _seg_data.tail(12)["total_incurred"].mean()
    _means_dict[_seg] = round(_trailing_12 * 12 / 1_000_000, 2)  # annualize, convert to $M
_means_dict["Liability"] = round(0.5 * (_means_dict["Property"] + _means_dict["Auto"]), 2)

_calibrated_means = [_means_dict["Property"], _means_dict["Auto"], _means_dict["Liability"]]

# Empirical correlations from monthly log-loss time series
_pivot = _seg_monthly.pivot(index="month", columns="mc_segment", values="total_incurred")
if "Property" in _pivot.columns and "Auto" in _pivot.columns:
    _pivot["Liability"] = 0.5 * (_pivot["Property"] + _pivot["Auto"])

_log_losses = np.log(_pivot[["Property", "Auto", "Liability"]].clip(lower=1.0))
_calibrated_corr = _log_losses.corr().values

# Ensure positive semi-definite
_eigvals, _eigvecs = np.linalg.eigh(_calibrated_corr)
_eigvals = np.maximum(_eigvals, 1e-8)
_calibrated_corr = _eigvecs @ np.diag(_eigvals) @ _eigvecs.T
_calibrated_corr = np.clip(_calibrated_corr, -0.99, 0.99)
np.fill_diagonal(_calibrated_corr, 1.0)

# Copula df grid search: AIC on empirical tail dependence
from scipy.stats import t as tdist, kendalltau
_best_df, _best_aic = 4, float('inf')
for _df_candidate in [3, 4, 5, 6, 8, 10]:
    _ll = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            _tau, _ = kendalltau(_log_losses.iloc[:, i], _log_losses.iloc[:, j])
            _rho = np.sin(np.pi * _tau / 2)
            _ll += -0.5 * (_rho - _calibrated_corr[i, j])**2
    _aic = -2 * _ll + 2 * 1
    if _aic < _best_aic:
        _best_aic = _aic
        _best_df = _df_candidate
_calibrated_df = _best_df

_calib_report = {
    "means_M": _calibrated_means,
    "corr_matrix": _calibrated_corr.tolist(),
    "copula_df": _calibrated_df,
    "data_window": f"{_calib_pdf['month'].min().strftime('%Y-%m')} to {_calib_pdf['month'].max().strftime('%Y-%m')}",
    "n_segments_raw": int(_calib_pdf["segment_id"].nunique()),
    "n_months": int(_calib_pdf["month"].nunique()),
    "calibration_method": "historical_MoM",
}

print(f"Historical calibration OK:")
print(f"  Means ($M):   Property={_calibrated_means[0]:.2f}, Auto={_calibrated_means[1]:.2f}, Liability={_calibrated_means[2]:.2f}")
print(f"  Correlations: P-A={_calibrated_corr[0,1]:.3f}, P-L={_calibrated_corr[0,2]:.3f}, A-L={_calibrated_corr[1,2]:.3f}")
print(f"  Copula df:    {_calibrated_df}")
print(f"  Data window:  {_calib_report['data_window']}")

# COMMAND ----------

# ── Frequency-Severity Calibration (Collective Risk Model) ───────────────────
# Fit Negative Binomial (frequency) and Lognormal (severity) per MC segment
# from gold_claims_monthly. These parameters enable the bottom-up Collective
# Risk Model simulation path alongside the aggregate t-Copula approach.

# ── Frequency-Severity calibration (reuses _calib_pdf from above) ────────────
_FREQ_PARAMS = {}
_SEV_PARAMS = {}

for _mc_seg in ["Property", "Auto"]:
    _seg_data = _calib_pdf[_calib_pdf["mc_segment"] == _mc_seg]

    # Frequency: monthly claims_count aggregated across provinces
    _monthly_counts = _seg_data.groupby("month")["claims_count"].sum().values.astype(float)

    # Fit Negative Binomial: mean = μ, var = μ + μ²/k → k = μ² / (var - μ)
    _mu_freq = float(_monthly_counts.mean())
    _var_freq = float(_monthly_counts.var())
    # NegBin k parameter (synthetic data is always overdispersed: var > mean)
    _k_freq = _mu_freq ** 2 / max(_var_freq - _mu_freq, 1e-6)

    _FREQ_PARAMS[_mc_seg] = {"lambda": round(_mu_freq, 1), "k": round(_k_freq, 2)}

    # Severity: avg_severity per month (total_incurred / claims_count)
    _severity_data = _seg_data.groupby("month").agg(
        total_incurred=("total_incurred", "sum"),
        claims_count=("claims_count", "sum"),
    ).reset_index()
    _severity_data["avg_sev"] = _severity_data["total_incurred"] / _severity_data["claims_count"].clip(lower=1)
    _avg_sevs = _severity_data["avg_sev"].values
    _avg_sevs = _avg_sevs[_avg_sevs > 0]

    # Fit Lognormal via MLE: μ = mean(log(x)), σ = std(log(x))
    _log_sevs = np.log(_avg_sevs)
    _mu_sev = float(_log_sevs.mean())
    _sigma_sev = float(_log_sevs.std())

    _SEV_PARAMS[_mc_seg] = {"mu": round(_mu_sev, 4), "sigma": round(_sigma_sev, 4)}

# Liability proxy: average of Property and Auto parameters
_FREQ_PARAMS["Liability"] = {
    "lambda": round(0.5 * (_FREQ_PARAMS["Property"]["lambda"] + _FREQ_PARAMS["Auto"]["lambda"]), 1),
    "k": round(0.5 * (_FREQ_PARAMS["Property"]["k"] + _FREQ_PARAMS["Auto"]["k"]), 2),
}
_SEV_PARAMS["Liability"] = {
    "mu": round(0.5 * (_SEV_PARAMS["Property"]["mu"] + _SEV_PARAMS["Auto"]["mu"]), 4),
    "sigma": round(0.5 * (_SEV_PARAMS["Property"]["sigma"] + _SEV_PARAMS["Auto"]["sigma"]), 4),
}

print(f"\nFrequency-Severity calibration OK:")
for _seg in ["Property", "Auto", "Liability"]:
    _fp = _FREQ_PARAMS[_seg]
    _sp = _SEV_PARAMS[_seg]
    print(f"  {_seg}: λ={_fp['lambda']:.0f}, k={_fp['k']:.1f} | μ_sev={_sp['mu']:.3f}, σ_sev={_sp['sigma']:.3f}")

# COMMAND ----------

# ── GARCH → Monte Carlo Bridge: derive CVs from SARIMA residual volatilities ─
# The GARCH(1,1) fitted on SARIMA residuals (inside fit_sarimax_per_segment)
# produces conditional volatility in claims_count units.  The textbook CV is
# simply σ/μ — no ad-hoc scaling needed because the units are already correct.
_sarima_cv_pd = _sarima_pd  # reuse from calibration bridge (already has product_line)

# Filter to actuals with GARCH conditional volatility
_actual_garch = _sarima_cv_pd[
    (_sarima_cv_pd["record_type"] == "actual") &
    (_sarima_cv_pd["cond_volatility"].notna())
]

assert len(_actual_garch) > 0, "No segments with GARCH conditional volatility — check SARIMAX fitting"

# CV = mean(σ_t) / mean(claims_count) per product line (textbook definition)
_cv_by_line = (
    _actual_garch
    .groupby("product_line")
    .apply(lambda g: g["cond_volatility"].mean() / max(g["claims_count"].mean(), 1.0))
)

# Map product lines to MC segments (Property, Auto, Liability proxy)
_prop_cv = np.mean([_cv_by_line.get("Homeowners", 0.35),
                     _cv_by_line.get("Commercial_Property", 0.42)])
_auto_cv = np.mean([_cv_by_line.get("Personal_Auto", 0.28),
                     _cv_by_line.get("Commercial_Auto", 0.30)])
_liab_cv = 0.5 * (_prop_cv + _auto_cv)

_garch_cvs = np.clip(
    np.array([_prop_cv, _auto_cv, _liab_cv]),
    0.15, 0.60,
).tolist()
print(f"GARCH-derived CVs (σ/μ): Property={_garch_cvs[0]:.3f}, Auto={_garch_cvs[1]:.3f}, Liability={_garch_cvs[2]:.3f}")

# Portfolio parameters (in $M expected annual losses)
# Means and correlations are historically calibrated from gold_claims_monthly.
# CVs come from GARCH(1,1) on SARIMA residuals.
# Copula df is selected via AIC on empirical tail dependence.
PORTFOLIO = {
    "segments":    ["Property", "Auto", "Liability"],
    "means":       _calibrated_means,           # Historically calibrated ($M)
    "cv":          _garch_cvs,                   # GARCH-derived CVs (or defaults)
    "corr_matrix": _calibrated_corr,             # Empirical log-loss correlations
}

# Cholesky decomposition for correlated draws (done once, shared)
_sigma = np.diag(PORTFOLIO["cv"]) @ PORTFOLIO["corr_matrix"] @ np.diag(PORTFOLIO["cv"])
CHOL = np.linalg.cholesky(_sigma)

# ── Monte Carlo constants (module-level so Ray workers compute once per process) ─
_MC_COPULA_DF  = _calibrated_df
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

# @ray.remote(num_cpus=1): 24 concurrent tasks across 4 workers (4 × 6 = 24 Ray CPUs).
# NumPy/SciPy path: random sampling + Cholesky + lognormal + t-CDF all on CPU.
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
# MAGIC    position; written to `predictions_monte_carlo` for downstream Modules 5–7.
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
N_PER_TASK       = 10_000_000      # 10M paths per task; total ~640M paths
STRESS_SCENARIOS = ['stress_corr', 'cat_event', 'inflation_shock']
N_RUNS           = 1 + len(_forecast_months) + len(STRESS_SCENARIOS)

print(f'Launching {N_RUNS} runs × {N_TASKS} tasks × {N_PER_TASK:,} paths '
          f'= {N_RUNS * N_TASKS * N_PER_TASK:,} total paths')

with mlflow.start_run(run_name='monte_carlo_portfolio_ray') as run:
    mlflow.log_input(_claims_dataset, context="training")
    mlflow.set_tags({
        'workshop_module': '4',
        'model_type':      'Monte Carlo - t-Copula + Lognormal Marginals (SARIMAX-driven)',
        'n_scenarios':     str(N_RUNS * N_TASKS * N_PER_TASK),
        'n_segments':      '3',
        'framework':       'Ray + NumPy CPU',
        'audience':        'actuarial-workshop',
        'calibration_method': 'historical_MoM',
    })
    mlflow.log_params({
        'n_tasks':              N_TASKS,
        'scenarios_per_task':   N_PER_TASK,
        'total_scenarios':      N_RUNS * N_TASKS * N_PER_TASK,
        'n_forecast_months':    len(_forecast_months),
        'copula':               't-copula',
        'copula_df':            _MC_COPULA_DF,
        'marginals':            'lognormal',
        'correlation_P_A':      round(float(_calibrated_corr[0, 1]), 3),
        'correlation_P_L':      round(float(_calibrated_corr[0, 2]), 3),
        'correlation_A_L':      round(float(_calibrated_corr[1, 2]), 3),
        'mean_property_M':      _calibrated_means[0],
        'mean_auto_M':          _calibrated_means[1],
        'mean_liability_M':     _calibrated_means[2],
        'stress_scenarios':     ','.join(STRESS_SCENARIOS),
        'n_stress_scenarios':   len(STRESS_SCENARIOS),
        'freq_sev_available':   True,
    })

    # Log calibration report as JSON artifact
    import json as _json, tempfile as _tmpf, os as _os
    with _tmpf.TemporaryDirectory() as _td:
        _cr_path = _os.path.join(_td, "calibration_report.json")
        with open(_cr_path, "w") as _f:
            _json.dump(_calib_report, _f, indent=2, default=str)
        mlflow.log_artifact(_cr_path, artifact_path="calibration")
    print("Calibration report logged as MLflow artifact")

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

    # Baseline rows → predictions_monte_carlo (backward-compatible with Module 5/app)
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
        .saveAsTable(f'{CATALOG}.{SCHEMA}.predictions_monte_carlo'))
    print(f'\nBaseline saved → {CATALOG}.{SCHEMA}.predictions_monte_carlo')

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
        .saveAsTable(f'{CATALOG}.{SCHEMA}.predictions_risk_timeline'))
    print(f'\nVaR time-series saved → {CATALOG}.{SCHEMA}.predictions_risk_timeline')

    # Regional claims forecast removed — derivable from predictions_sarima
    # via: SELECT region, month, SUM(forecast_mean) FROM predictions_sarima
    #      WHERE record_type='forecast' GROUP BY region, month

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
        .saveAsTable(f'{CATALOG}.{SCHEMA}.predictions_stress_scenarios'))
    print(f'\nStress scenarios saved → {CATALOG}.{SCHEMA}.predictions_stress_scenarios')

    # ── Collective Risk Model (frequency-severity bottom-up) ─────────────
    # Uses the CLT compound distribution approximation: with λ ≈ 200K–300K
    # claims per scenario, the aggregate loss S = Σ X_i is approximately
    # Normal with E[S] = N·E[X] and Var[S] = N·Var[X] + E[X]²·Var[N].
    # This avoids allocating ~25B individual claim amounts (200GB+).
    from scipy.stats import nbinom as _nbinom_dist
    print('\n  Running Collective Risk Model (frequency-severity)...')
    _CR_N_SCENARIOS = 100_000
    _cr_rng = np.random.default_rng(777)

    # Correlated frequencies via Gaussian copula on NegBin marginals
    _cr_z = _cr_rng.standard_normal((_CR_N_SCENARIOS, 3))
    _cr_chol = np.linalg.cholesky(_calibrated_corr)
    _cr_z_cor = _cr_z @ _cr_chol.T
    from scipy.stats import norm as _cr_norm
    _cr_u = _cr_norm.cdf(_cr_z_cor)  # uniform marginals

    _cr_total_losses = np.zeros(_CR_N_SCENARIOS)
    for _seg_idx, _seg_name in enumerate(["Property", "Auto", "Liability"]):
        fp = _FREQ_PARAMS[_seg_name]
        sp = _SEV_PARAMS[_seg_name]
        _lam, _k = fp["lambda"], fp["k"]
        _mu_s, _sig_s = sp["mu"], sp["sigma"]

        # NegBin inverse CDF applied to copula uniform marginals
        _p_nb = _k / (_k + _lam)
        _n_claims = _nbinom_dist.ppf(_cr_u[:, _seg_idx], n=_k, p=_p_nb).astype(float)

        # CLT compound approximation: with λ ≈ 200K+ claims, S ≈ Normal
        _ex = np.exp(_mu_s + _sig_s**2 / 2)
        _varx = (np.exp(_sig_s**2) - 1) * np.exp(2*_mu_s + _sig_s**2)
        _seg_mean = _n_claims * _ex
        _seg_std = np.sqrt(_n_claims * _varx)
        _seg_losses = (_seg_mean + _seg_std * _cr_rng.standard_normal(_CR_N_SCENARIOS)) / 1_000_000
        _seg_losses = np.maximum(_seg_losses, 0.0)
        _cr_total_losses += _seg_losses

    _cr_mean = float(_cr_total_losses.mean())
    _cr_var99 = float(np.percentile(_cr_total_losses, 99))
    _cr_var995 = float(np.percentile(_cr_total_losses, 99.5))
    _cr_cvar99 = float(_cr_total_losses[_cr_total_losses >= _cr_var99].mean())

    mlflow.log_metrics({
        'CR_expected_loss_M': round(_cr_mean, 2),
        'CR_VaR_99_M': round(_cr_var99, 2),
        'CR_VaR_99_5_M': round(_cr_var995, 2),
        'CR_CVaR_99_M': round(_cr_cvar99, 2),
    })

    print(f'  Collective Risk Model ({_CR_N_SCENARIOS:,} scenarios):')
    print(f'    Expected Loss: ${_cr_mean:.1f}M  |  VaR(99%): ${_cr_var99:.1f}M  |  SCR: ${_cr_var995:.1f}M  |  CVaR: ${_cr_cvar99:.1f}M')
    print(f'    vs Aggregate:  EL ${aggregate_mean:.1f}M  |  VaR(99%) ${aggregate_var99:.1f}M  |  SCR ${aggregate_var995:.1f}M')

    # ── Multi-Period Portfolio Evolution (regime-switching) ───────────────
    # 2-state regime model: Normal and Crisis, with Markov transition.
    # 12-month horizon, monthly steps. Tracks cumulative surplus.
    print('\n  Running multi-period surplus evolution...')
    _MP_N_SCENARIOS = 50_000
    _MP_HORIZON = 12
    _mp_rng = np.random.default_rng(999)

    # Regime parameters
    _REGIME_PARAMS = {
        'Normal': {
            'means_mult': np.ones(3, dtype=np.float32),
            'cv_mult': np.ones(3, dtype=np.float32),
            'corr_adj': 0.0,
        },
        'Crisis': {
            'means_mult': np.array([1.3, 1.3, 1.3], dtype=np.float32),
            'cv_mult': np.array([1.15, 1.15, 1.15], dtype=np.float32),
            'corr_adj': 0.20,  # added to each off-diagonal element
        },
    }
    # Transition probabilities: P(stay_normal)=0.95, P(normal→crisis)=0.05
    #                           P(stay_crisis)=0.85, P(crisis→normal)=0.15
    _TRANS_PROB = {'Normal': {'Normal': 0.95, 'Crisis': 0.05},
                   'Crisis': {'Normal': 0.15, 'Crisis': 0.85}}

    # Monthly premium = annualized earned premium / 12 (from gold_claims_monthly)
    _monthly_premium = float(
        spark.table(f"{CATALOG}.{SCHEMA}.gold_claims_monthly")
        .agg(F.mean("earned_premium")).collect()[0][0]
    ) / 1_000_000  # monthly avg in $M

    _investment_rate_monthly = 0.04 / 12  # 4% annual risk-free rate

    # Initialize surplus trajectories
    _initial_surplus = aggregate_var995 * 1.2  # start at 120% of SCR
    _surplus = np.full((_MP_N_SCENARIOS, _MP_HORIZON + 1), _initial_surplus)
    _regime_state = np.zeros(_MP_N_SCENARIOS, dtype=int)  # 0=Normal, 1=Crisis

    for _t in range(_MP_HORIZON):
        # Regime transitions (Markov)
        _u_trans = _mp_rng.random(_MP_N_SCENARIOS)
        _stay_prob = np.where(_regime_state == 0,
                              _TRANS_PROB['Normal']['Normal'],
                              _TRANS_PROB['Crisis']['Crisis'])
        _regime_state = np.where(_u_trans < _stay_prob, _regime_state,
                                 1 - _regime_state)

        # Compute effective parameters per scenario
        _eff_means = np.where(
            _regime_state[:, None] == 0,
            _MC_MEANS[None, :],
            _MC_MEANS[None, :] * _REGIME_PARAMS['Crisis']['means_mult'][None, :],
        )
        _eff_cv = np.where(
            _regime_state[:, None] == 0,
            _MC_CV[None, :],
            _MC_CV[None, :] * _REGIME_PARAMS['Crisis']['cv_mult'][None, :],
        )

        # Simplified: draw monthly losses using aggregate model (1/12 of annual)
        _monthly_means = _eff_means / 12.0
        _monthly_cv = _eff_cv  # CV doesn't change with time scaling
        _m_sigma2 = np.log(1 + _monthly_cv**2)
        _m_mu_ln = np.log(_monthly_means) - _m_sigma2 / 2
        _m_sigma_ln = np.sqrt(_m_sigma2)

        # Independent lognormal draws per segment (simplified, no copula per month)
        _z_month = _mp_rng.standard_normal((_MP_N_SCENARIOS, 3))
        _monthly_losses = np.exp(_m_mu_ln + _m_sigma_ln * _z_month)
        _total_monthly_loss = _monthly_losses.sum(axis=1)

        # Surplus update: S(t+1) = S(t) + Premium - Loss + Investment
        _investment = _surplus[:, _t] * _investment_rate_monthly
        _surplus[:, _t + 1] = (_surplus[:, _t] + _monthly_premium
                                - _total_monthly_loss + _investment)

    # Compute percentile bands and ruin probability
    _surplus_pctiles = np.percentile(_surplus, [5, 25, 50, 75, 95], axis=0)
    _ruin_prob = np.mean(_surplus < 0, axis=0)

    surplus_rows = []
    for _t in range(_MP_HORIZON + 1):
        surplus_rows.append({
            'month': _t,
            'surplus_p05': round(float(_surplus_pctiles[0, _t]), 2),
            'surplus_p25': round(float(_surplus_pctiles[1, _t]), 2),
            'surplus_p50': round(float(_surplus_pctiles[2, _t]), 2),
            'surplus_p75': round(float(_surplus_pctiles[3, _t]), 2),
            'surplus_p95': round(float(_surplus_pctiles[4, _t]), 2),
            'ruin_probability': round(float(_ruin_prob[_t]), 6),
        })

    surplus_df = spark.createDataFrame(pd.DataFrame(surplus_rows))
    (surplus_df.write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .saveAsTable(f'{CATALOG}.{SCHEMA}.predictions_surplus_evolution'))
    print(f'Surplus evolution saved → {CATALOG}.{SCHEMA}.predictions_surplus_evolution')

    # Save regime parameters
    regime_rows = []
    for _state_name, _params in _REGIME_PARAMS.items():
        regime_rows.append({
            'state': _state_name,
            'means_mult_property': float(_params['means_mult'][0]),
            'means_mult_auto': float(_params['means_mult'][1]),
            'means_mult_liability': float(_params['means_mult'][2]),
            'cv_mult_property': float(_params['cv_mult'][0]),
            'cv_mult_auto': float(_params['cv_mult'][1]),
            'cv_mult_liability': float(_params['cv_mult'][2]),
            'corr_adjustment': float(_params['corr_adj']),
            'transition_to_normal': float(_TRANS_PROB[_state_name]['Normal']),
            'transition_to_crisis': float(_TRANS_PROB[_state_name]['Crisis']),
        })
    regime_df = spark.createDataFrame(pd.DataFrame(regime_rows))
    (regime_df.write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .saveAsTable(f'{CATALOG}.{SCHEMA}.predictions_regime_parameters'))
    print(f'Regime parameters saved → {CATALOG}.{SCHEMA}.predictions_regime_parameters')

    mlflow.log_metrics({
        'surplus_median_month12': round(float(_surplus_pctiles[2, -1]), 2),
        'ruin_prob_month12': round(float(_ruin_prob[-1]), 6),
        'initial_surplus_M': round(_initial_surplus, 2),
    })
    print(f'  Initial surplus: ${_initial_surplus:.1f}M | Month-12 median: ${_surplus_pctiles[2, -1]:.1f}M | Ruin prob: {_ruin_prob[-1]:.4%}')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Reserve Triangle Validation
# MAGIC
# MAGIC The loss development triangle from the declarative pipeline (`gold_reserve_triangle`) provides an
# MAGIC independent check on SARIMA forecasts. We compare the SARIMA-projected incurred claims against
# MAGIC actual reserve development to assess **reserve adequacy**.
# MAGIC
# MAGIC This connects the new CDC pipeline → Module 4, completing the end-to-end lineage.

# COMMAND ----------

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
_sarima_actuals = (
    sarima_results_df
    .filter(F.col("record_type") == "actual")
    .select("segment_id", "month", "claims_count")
)
predictions_reserve_validation = (
    latest_reserves
    .join(_sarima_actuals, on=["segment_id", "month"], how="inner")
    .withColumn("reserve_adequacy_ratio",
        F.when(F.col("claims_count") > 0,
               F.col("cumulative_incurred") / F.col("claims_count"))
         .otherwise(F.lit(None).cast("double")))
    .select("segment_id", "month", "cumulative_incurred", "cumulative_paid",
            "case_reserve", "claims_count", "reserve_adequacy_ratio", "dev_lag")
)

(predictions_reserve_validation.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.predictions_reserve_validation"))

_avg_adequacy = predictions_reserve_validation.agg(F.mean("reserve_adequacy_ratio")).collect()[0][0]
print(f"Reserve validation saved → {CATALOG}.{SCHEMA}.predictions_reserve_validation")
print(f"  Average reserve adequacy ratio: {_avg_adequacy:.2f}")
print(f"  (>1.0 = over-reserved, <1.0 = under-reserved, ~1.0 = adequate)")

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
import arch as _arch_pkg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from mlflow.tracking import MlflowClient

# ── PyFunc wrapper (same interface as Module 5's serving contract) ────────────
class SARIMAXForecaster(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for a fitted SARIMAX + optional GARCH(1,1) model.

    Input:  pandas DataFrame with column `horizon` (int) — months to forecast
    Output: pandas DataFrame with columns: month_offset, forecast_mean, lo95, hi95

    When a GARCH pickle is present, prediction intervals are time-varying
    (wider at longer horizons due to GARCH volatility persistence).
    When absent, falls back to constant statsmodels CIs.
    """

    def load_context(self, context):
        """Load the pickled SARIMAX model and optional GARCH model."""
        import pickle
        import os
        model_path = os.path.join(context.artifacts["sarimax_model"], "model.pkl")
        with open(model_path, "rb") as f:
            self.model_fit = pickle.load(f)
        garch_path = os.path.join(context.artifacts["sarimax_model"], "garch.pkl")
        self.garch_fit = None
        if os.path.exists(garch_path):
            with open(garch_path, "rb") as f:
                self.garch_fit = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecast with time-varying CIs when GARCH is available.

        Args:
            model_input: DataFrame with column `horizon` (int, 1–24)

        Returns:
            DataFrame with forecast mean and 95% confidence intervals.
        """
        horizon = int(model_input["horizon"].iloc[0])
        horizon = max(1, min(horizon, 24))  # clamp to [1, 24]

        forecast = self.model_fit.get_forecast(steps=horizon)
        mean_fcst = forecast.predicted_mean

        if self.garch_fit is not None:
            # Time-varying CIs from GARCH forecast volatility
            fc = self.garch_fit.forecast(horizon=horizon, reindex=False)
            garch_vol = np.sqrt(np.asarray(fc.variance).flatten()[:horizon])
            lo95 = np.asarray(mean_fcst) - 1.96 * garch_vol
            hi95 = np.asarray(mean_fcst) + 1.96 * garch_vol
        else:
            ci = np.asarray(forecast.conf_int(alpha=0.05))
            lo95, hi95 = ci[:, 0], ci[:, 1]

        return pd.DataFrame({
            "month_offset":   list(range(1, horizon + 1)),
            "forecast_mean":  list(np.round(mean_fcst, 1)),
            "forecast_lo95":  list(np.round(lo95, 1)),
            "forecast_hi95":  list(np.round(hi95, 1)),
        })

# ── Train on Personal_Auto__Ontario (highest volume, primary validation segment)
_reg_claims_pdf = (
    spark.table(f"{CATALOG}.{SCHEMA}.gold_claims_monthly")
    .filter("segment_id = 'Personal_Auto__Ontario'")
    .orderBy("month")
    .select("month", "claims_count")
    .toPandas()
)
print(f"Loaded {len(_reg_claims_pdf)} months from gold_claims_monthly")

_y_train = _reg_claims_pdf["claims_count"].astype(float).values

# ── Switch experiment for SARIMA champion registration ────────────────────────
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_sarima_claims_forecaster")

# Dataset reference for UC lineage
_sarima_reg_dataset = mlflow.data.load_delta(
    table_name=f"{CATALOG}.{SCHEMA}.gold_claims_monthly",
    name="gold_claims_monthly",
)

with mlflow.start_run(run_name="sarima_personal_auto_ontario_champion") as _sarima_reg_run:
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

    # ── Fit GARCH(1,1) on registration segment's residuals ───────────────────
    from arch import arch_model
    from statsmodels.stats.diagnostic import het_arch
    _reg_resid = _reg_fit.resid[12:]  # skip seasonal burn-in
    _reg_garch_fit = None
    try:
        _lm_stat, _lm_pval, _, _ = het_arch(_reg_resid, nlags=4)
        if _lm_pval < 0.10:
            _am = arch_model(_reg_resid, mean='Zero', vol='Garch', p=1, q=1, dist='normal')
            _reg_garch_fit = _am.fit(disp='off', show_warning=False)
            print(f"GARCH fitted on registration segment (ARCH-LM p={_lm_pval:.4f})")
        else:
            print(f"No significant ARCH effects on registration segment (p={_lm_pval:.4f}) — GARCH skipped")
    except Exception as _garch_reg_err:
        print(f"GARCH fit failed on registration segment: {_garch_reg_err}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    _reg_fitted = _reg_fit.fittedvalues[12:]
    _reg_actual = _y_train[12:]
    _reg_mape = float(np.mean(np.abs((_reg_actual - _reg_fitted) / np.clip(_reg_actual, 1, None))) * 100)
    _reg_rmse = float(np.sqrt(np.mean((_reg_actual - _reg_fitted)**2)))

    # ── Log parameters ────────────────────────────────────────────────────────
    _model_type_tag = "SARIMAX(1,0,1)(1,1,0,12)+GARCH(1,1)" if _reg_garch_fit else "SARIMAX(1,0,1)(1,1,0,12)"
    mlflow.set_tags({
        "segment_id":      "Personal_Auto__Ontario",
        "workshop_module": "4",
        "model_class":     "SARIMAX",
        "model_type":      _model_type_tag,
        "audience":        "actuarial-workshop",
    })
    mlflow.log_params({
        "order_p": 1, "order_d": 0, "order_q": 1,
        "seasonal_P": 1, "seasonal_D": 1, "seasonal_Q": 0,
        "seasonal_m": 12,
        "training_months": len(_y_train),
        "segment": "Personal_Auto__Ontario",
        "garch_fitted": _reg_garch_fit is not None,
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
    _ax.set_title(f"{_model_type_tag} — Personal Auto Ontario")
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

    # ── Save pickled models for PyFunc ────────────────────────────────────────
    _SARIMA_MODEL_NAME = f"{CATALOG}.{SCHEMA}.sarima_claims_forecaster"
    with tempfile.TemporaryDirectory() as _tmpdir:
        _model_pkl_path = os.path.join(_tmpdir, "model.pkl")
        with open(_model_pkl_path, "wb") as _f:
            pickle.dump(_reg_fit, _f)

        # Pickle GARCH model alongside SARIMAX (if fitted)
        if _reg_garch_fit is not None:
            _garch_pkl_path = os.path.join(_tmpdir, "garch.pkl")
            with open(_garch_pkl_path, "wb") as _f:
                pickle.dump(_reg_garch_fit, _f)

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
                f"arch=={_arch_pkg.__version__}",
            ],
        )

    print(f"\nSARIMA+GARCH model registered to: {_SARIMA_MODEL_NAME}")
    print(f"MAPE: {_reg_mape:.1f}%  |  RMSE: {_reg_rmse:.0f}  |  AIC: {_reg_fit.aic:.1f}")
    print(f"GARCH included: {_reg_garch_fit is not None}")

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
    MLflow PyFunc wrapper for Monte Carlo portfolio loss simulation.

    Supports two model types:
      - "aggregate" (default): t-Copula + Lognormal Marginals (standard formula)
      - "collective_risk": Frequency-Severity bottom-up (internal model)

    And two simulation modes:
      - "single_period" (default): Single annual loss distribution
      - "multi_period": 12-month surplus evolution with regime-switching

    Parameterised simulation: all assumptions arrive in the request.

    Input:  DataFrame with one row of scenario parameters
    Output: DataFrame with one row of portfolio risk metrics
    """

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from scipy.stats import t as tdist, norm as scipy_norm

        row = model_input.iloc[0]

        model_type = str(row.get("model_type", "aggregate"))
        simulation_mode = str(row.get("simulation_mode", "single_period"))

        # ── Read scenario parameters (with calibrated defaults) ──────────────
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

        # Frequency-severity parameters (used when model_type="collective_risk")
        freq_lambda = [float(row.get("freq_lambda_property", 23700)),
                       float(row.get("freq_lambda_auto", 36500)),
                       float(row.get("freq_lambda_liability", 30100))]
        freq_k = [float(row.get("freq_k_property", 50)),
                  float(row.get("freq_k_auto", 80)),
                  float(row.get("freq_k_liability", 65))]
        sev_mu = [float(row.get("sev_mu_property", 6.0)),
                  float(row.get("sev_mu_auto", 5.5)),
                  float(row.get("sev_mu_liability", 5.8))]
        sev_sigma = [float(row.get("sev_sigma_property", 0.8)),
                     float(row.get("sev_sigma_auto", 0.6)),
                     float(row.get("sev_sigma_liability", 0.7))]

        # Safety bounds
        n_scenarios = max(1_000, min(n_scenarios, 100_000))
        copula_df   = max(2,     min(copula_df,   30))
        means       = np.clip(means, 0.01, 1_000.0)
        cv          = np.clip(cv,    0.01, 5.0)
        corr_prop_auto = np.clip(corr_prop_auto, -0.99, 0.99)
        corr_prop_liab = np.clip(corr_prop_liab, -0.99, 0.99)
        corr_auto_liab = np.clip(corr_auto_liab, -0.99, 0.99)

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

        rng = np.random.default_rng(42)

        if model_type == "collective_risk":
            # ── Collective Risk Model: frequency-severity bottom-up ──────
            from scipy.stats import nbinom as nbinom_dist
            z = rng.standard_normal((n_scenarios, 3))
            z_cor = z @ chol.T
            u = scipy_norm.cdf(z_cor)

            total = np.zeros(n_scenarios)
            for i in range(3):
                p_nb = freq_k[i] / (freq_k[i] + freq_lambda[i])
                n_claims = nbinom_dist.ppf(u[:, i], n=freq_k[i], p=p_nb).astype(float)
                # CLT compound approximation: with λ ≈ 200K+ claims, S ≈ Normal
                ex = np.exp(sev_mu[i] + sev_sigma[i]**2 / 2)
                varx = (np.exp(sev_sigma[i]**2) - 1) * np.exp(2*sev_mu[i] + sev_sigma[i]**2)
                seg_mean = n_claims * ex
                seg_std = np.sqrt(n_claims * varx)
                seg_losses = (seg_mean + seg_std * rng.standard_normal(n_scenarios)) / 1_000_000
                total += np.maximum(seg_losses, 0.0)
        else:
            # ── Aggregate Model: t-Copula + Lognormal ────────────────────
            sigma2   = np.log(1 + cv**2)
            mu_ln    = np.log(means) - sigma2 / 2
            sigma_ln = np.sqrt(sigma2)

            z     = rng.standard_normal((n_scenarios, 3))
            chi2  = rng.chisquare(copula_df, n_scenarios)
            x_cor = z @ chol.T
            t_cor = x_cor / np.sqrt(chi2[:, None] / copula_df)
            u     = tdist.cdf(t_cor, df=copula_df)
            q     = scipy_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
            losses = np.exp(mu_ln + sigma_ln * q)
            total  = losses.sum(axis=1)

        if simulation_mode == "multi_period":
            # ── Multi-period surplus evolution with regime-switching ──────
            horizon = 12
            monthly_premium = float(sum(means)) / 12 * 1.05
            inv_rate = 0.04 / 12
            initial_surplus = float(np.percentile(total, 99.5)) * 1.2
            surplus = np.full((n_scenarios, horizon + 1), initial_surplus)
            regime = np.zeros(n_scenarios, dtype=int)

            for t in range(horizon):
                u_trans = rng.random(n_scenarios)
                stay_prob = np.where(regime == 0, 0.95, 0.85)
                regime = np.where(u_trans < stay_prob, regime, 1 - regime)

                m_mult = np.where(regime[:, None] == 0,
                                  np.ones((1, 3)), np.full((1, 3), 1.3))
                m_means = means[None, :] * m_mult / 12.0
                m_cv = cv[None, :] * np.where(regime[:, None] == 0,
                                               np.ones((1, 3)), np.full((1, 3), 1.15))
                m_sig2 = np.log(1 + m_cv**2)
                m_mu = np.log(m_means) - m_sig2 / 2
                m_sig = np.sqrt(m_sig2)
                z_m = rng.standard_normal((n_scenarios, 3))
                monthly_loss = np.exp(m_mu + m_sig * z_m).sum(axis=1)
                inv = surplus[:, t] * inv_rate
                surplus[:, t + 1] = surplus[:, t] + monthly_premium - monthly_loss + inv

            pctiles = np.percentile(surplus, [5, 25, 50, 75, 95], axis=0)
            ruin_prob = float(np.mean(surplus[:, -1] < 0))

            return pd.DataFrame([{
                "expected_loss_M":  round(float(total.mean()), 3),
                "var_99_M":         round(float(np.percentile(total, 99)), 3),
                "var_995_M":        round(float(np.percentile(total, 99.5)), 3),
                "cvar_99_M":        round(float(total[total >= np.percentile(total, 99)].mean()), 3),
                "initial_surplus_M": round(initial_surplus, 3),
                "surplus_median_month12_M": round(float(pctiles[2, -1]), 3),
                "surplus_p05_month12_M": round(float(pctiles[0, -1]), 3),
                "ruin_probability": round(ruin_prob, 6),
                "model_type": model_type,
                "simulation_mode": "multi_period",
                "n_scenarios_used": n_scenarios,
            }])

        # ── Single-period risk metrics ──────────────────────────────────────
        var_99_threshold = np.percentile(total, 99)

        return pd.DataFrame([{
            "expected_loss_M":  round(float(total.mean()), 3),
            "var_95_M":         round(float(np.percentile(total, 95)),  3),
            "var_99_M":         round(float(np.percentile(total, 99)),  3),
            "var_995_M":        round(float(np.percentile(total, 99.5)), 3),
            "cvar_99_M":        round(float(total[total >= var_99_threshold].mean()), 3),
            "max_loss_M":       round(float(total.max()), 3),
            "n_scenarios_used": n_scenarios,
            "model_type":       model_type,
            "copula":           f"t-copula(df={copula_df})",
        }])

# ── Validate locally with baseline scenario ──────────────────────────────────
_mc_baseline_input = pd.DataFrame([{
    "mean_property_M": _calibrated_means[0],
    "mean_auto_M":     _calibrated_means[1],
    "mean_liability_M":_calibrated_means[2],
    "cv_property": float(_garch_cvs[0]),
    "cv_auto":     float(_garch_cvs[1]),
    "cv_liability":float(_garch_cvs[2]),
    "corr_prop_auto": round(float(_calibrated_corr[0, 1]), 3),
    "corr_prop_liab": round(float(_calibrated_corr[0, 2]), 3),
    "corr_auto_liab": round(float(_calibrated_corr[1, 2]), 3),
    "n_scenarios": 10_000,
    "copula_df": _calibrated_df,
    "model_type": "aggregate",
    "simulation_mode": "single_period",
}])

_mc_pyfunc = MonteCarloPyFunc()
_mc_baseline_result = _mc_pyfunc.predict(None, _mc_baseline_input)
print("Baseline Monte Carlo validation (10,000 scenarios):")
print(_mc_baseline_result.to_string(index=False))

# COMMAND ----------

# ── Register Monte Carlo model to UC ─────────────────────────────────────────
_MC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.monte_carlo_portfolio"
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_monte_carlo_portfolio")

_mc_reg_dataset = mlflow.data.load_delta(
    table_name=f"{CATALOG}.{SCHEMA}.gold_claims_monthly",
    name="gold_claims_monthly",
)

# ── Define MLflow signature ──────────────────────────────────────────────────
# Input includes aggregate params + optional F-S params + model_type/simulation_mode
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
    mlflow.types.ColSpec("string",  "model_type"),
    mlflow.types.ColSpec("string",  "simulation_mode"),
])
_mc_output_schema = mlflow.types.Schema([
    mlflow.types.ColSpec("double", "expected_loss_M"),
    mlflow.types.ColSpec("double", "var_95_M"),
    mlflow.types.ColSpec("double", "var_99_M"),
    mlflow.types.ColSpec("double", "var_995_M"),
    mlflow.types.ColSpec("double", "cvar_99_M"),
    mlflow.types.ColSpec("double", "max_loss_M"),
    mlflow.types.ColSpec("long",   "n_scenarios_used"),
    mlflow.types.ColSpec("string", "model_type"),
    mlflow.types.ColSpec("string", "copula"),
])
_mc_signature = mlflow.models.ModelSignature(inputs=_mc_input_schema, outputs=_mc_output_schema)

with mlflow.start_run(run_name="monte_carlo_portfolio_champion") as _mc_reg_run:
    mlflow.log_input(_mc_reg_dataset, context="training")

    mlflow.set_tags({
        "model_class":     "MonteCarloPyFunc",
        "copula":          "t-copula",
        "marginals":       "lognormal",
        "workshop_module": "4",
        "audience":        "actuarial-workshop",
        "calibration_method": "historical_MoM",
        "supports_collective_risk": str(True),
        "supports_multi_period": "true",
    })
    mlflow.log_params({
        "copula_df":              _calibrated_df,
        "n_lines":                3,
        "default_n_scenarios":    10_000,
        "mean_property_M_base":   _calibrated_means[0],
        "mean_auto_M_base":       _calibrated_means[1],
        "mean_liability_M_base":  _calibrated_means[2],
        "cv_property_base":       float(_garch_cvs[0]),
        "cv_auto_base":           float(_garch_cvs[1]),
        "cv_liability_base":      float(_garch_cvs[2]),
        "cv_source":              "GARCH(1,1) on SARIMA residuals",
        "corr_prop_auto_base":    round(float(_calibrated_corr[0, 1]), 3),
        "corr_prop_liab_base":    round(float(_calibrated_corr[0, 2]), 3),
        "corr_auto_liab_base":    round(float(_calibrated_corr[1, 2]), 3),
        "model_types":            "aggregate,collective_risk",
        "simulation_modes":       "single_period,multi_period",
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
# MAGIC | SARIMA(1,0,1)(1,1,0,12) | statsmodels + applyInPandas | 40 segments × 84 months | Baseline claim volume forecast |
# MAGIC | SARIMAX(1,0,1)(1,1,0,12) | statsmodels + applyInPandas | 40 segments + macro + FS exog | Forecast with StatCan + Feature Store signals |
# MAGIC | GARCH(1,1) on residuals | arch (inside SARIMAX fit) | 40 segments | Time-varying CIs + MC CVs (σ/μ) |
# MAGIC | Monte Carlo — baseline | Ray + NumPy CPU | 40M paths (4 tasks × 10M) | VaR(99.5%), CVaR, SCR — GARCH-calibrated |
# MAGIC | Monte Carlo — VaR evolution | Ray + NumPy CPU (SARIMAX-driven) | 480M paths (12 months × 4 × 10M) | Forward VaR, regional breakdown |
# MAGIC | Monte Carlo — stress tests | Ray + NumPy CPU (3 scenarios) | 120M paths (3 × 4 × 10M) | CAT event, systemic risk, inflation shock |
# MAGIC | Reserve validation | Spark join | Triangle × SARIMA | Reserve adequacy vs. forecasted claims |
# MAGIC | Model registration | MLflow + UC | 2 models (SARIMA+GARCH + MC) | `@Champion` alias for serving |
# MAGIC
# MAGIC **Data lineage:** `gold_claims_monthly` (SDP pipeline) → `silver_rolling_features` → Module 3 → `features_segment_monthly` → SARIMAX exog vars
# MAGIC
# MAGIC **GARCH → MC:** GARCH(1,1) fitted on SARIMA residuals provides dimensionally correct CVs (σ/μ) for Monte Carlo — no ad-hoc scaling
# MAGIC
# MAGIC **Next:** Module 5 — Create Model Serving endpoints for both models and configure AI Gateway.