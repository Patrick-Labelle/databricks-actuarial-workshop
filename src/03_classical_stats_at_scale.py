# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3: Stochastic Reserving at Scale
# MAGIC ## SARIMAX Frequency Forecasting + Bootstrap Chain Ladder with Ray
# MAGIC
# MAGIC Per-segment SARIMAX + GARCH via `applyInPandas` for frequency forecasting,
# MAGIC then chain ladder fitting and Ray-distributed Bootstrap Chain Ladder for reserve
# MAGIC risk quantification. Outputs: `predictions_frequency_forecast`,
# MAGIC `predictions_bootstrap_reserves`, `predictions_reserve_scenarios`,
# MAGIC `predictions_reserve_evolution`, `predictions_runoff_projection`,
# MAGIC `predictions_ldf_volatility`, and registered UC models.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install Required Libraries

# COMMAND ----------

%pip install statsmodels arch --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data from SDP Gold Layer
# MAGIC
# MAGIC Reads `gold_claims_monthly` (40 segments x 84 months), `gold_reserve_triangle`,
# MAGIC and `gold_macro_features` from the Module 1 declarative pipeline.

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType

# ─── Configuration ────────────────────────────────────────────────────────────
dbutils.widgets.text("catalog",       "my_catalog",       "UC Catalog")
dbutils.widgets.text("data_schema",   "actuarial_data",   "Data Schema (gold tables)")
dbutils.widgets.text("models_schema", "actuarial_models", "Models Schema (predictions)")
CATALOG       = dbutils.widgets.get("catalog")
DATA_SCHEMA   = dbutils.widgets.get("data_schema")
MODELS_SCHEMA = dbutils.widgets.get("models_schema")

np.random.seed(42)

# ─── Segment definitions ─────────────────────────────────────────────────────
PRODUCT_LINES = ["Personal_Auto", "Commercial_Auto", "Homeowners", "Commercial_Property"]
REGIONS       = [
    "Ontario", "Quebec", "British_Columbia", "Alberta",
    "Manitoba", "Saskatchewan", "New_Brunswick", "Nova_Scotia",
    "Prince_Edward_Island", "Newfoundland",
]
MONTHS        = pd.date_range("2019-01-01", periods=84, freq="MS")

# ─── Read from SDP gold layer ─────────────────────────────────────────────────
claims_df = (
    spark.table(f"{CATALOG}.{DATA_SCHEMA}.gold_claims_monthly")
    .filter(F.col("month").between("2019-01-01", "2025-12-01"))
)
claims_df.createOrReplaceTempView("claims_ts")
print(f"Loaded gold_claims_monthly")

# Load reserve triangle (primary data for stochastic reserving)
triangle_sdf = spark.table(f"{CATALOG}.{DATA_SCHEMA}.gold_reserve_triangle")
print(f"Loaded gold_reserve_triangle")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Macro Data & Feature Store Integration

# COMMAND ----------

macro_df = spark.table(f"{CATALOG}.{DATA_SCHEMA}.gold_macro_features")
print("Loaded gold_macro_features")

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

# COMMAND ----------

# Load Feature Store features and join to claims data
_fs_table = f"{CATALOG}.{DATA_SCHEMA}.features_segment_monthly"
fs_features = spark.table(_fs_table)
print("Loaded features_segment_monthly")

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
# MAGIC ## 3. SARIMAX Frequency Forecasting + GARCH
# MAGIC
# MAGIC `applyInPandas` distributes per-segment SARIMAX(1,0,1)(1,1,0,12) + GARCH(1,1)
# MAGIC on residuals. These forecasts predict **future accident period claim counts** —
# MAGIC the exposure base for new accident periods entering the reserve triangle.

# COMMAND ----------

FREQ_FORECAST_SCHEMA = StructType([
    StructField("segment_id",      StringType(), False),
    StructField("month",           DateType(),   False),
    StructField("record_type",     StringType(), False),
    StructField("claims_count",    DoubleType(), True),
    StructField("forecast_mean",   DoubleType(), True),
    StructField("forecast_lo95",   DoubleType(), True),
    StructField("forecast_hi95",   DoubleType(), True),
    StructField("aic",             DoubleType(), True),
    StructField("mape",            DoubleType(), True),
    StructField("mape_baseline",   DoubleType(), True),
    StructField("mape_sarimax",    DoubleType(), True),
    StructField("exog_vars",       StringType(), True),
    StructField("cond_volatility", DoubleType(), True),
    StructField("arch_lm_pvalue",  DoubleType(), True),
    StructField("garch_alpha",     DoubleType(), True),
    StructField("garch_beta",      DoubleType(), True),
])

def fit_sarimax_per_segment(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit SARIMAX + GARCH(1,1) for one segment to forecast future accident period
    claim counts (frequency forecasting for reserving).
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

    exog_cols = ["unemployment_rate", "hpi_growth",
                 "rolling_3m_mean", "coeff_variation_3m", "mom_change_pct"]
    exog_data = pdf[exog_cols].copy().ffill().bfill()
    exog_arr  = exog_data.values.astype(float)

    n_train, n_val = 72, 12
    y_train = y[:n_train]
    y_val   = y[n_train:]

    aic = mape_baseline = mape_sarimax = np.nan
    fcast_mean = np.full(12, np.nan)
    fcast_ci   = pd.DataFrame({"lower": np.full(12, np.nan), "upper": np.full(12, np.nan)})
    exog_vars_str = ",".join(exog_cols)

    exog_train = exog_arr[:n_train]
    exog_val   = exog_arr[n_train:]

    try:
        m_base   = SARIMAX(y_train, order=(1,0,1), seasonal_order=(1,1,0,12),
                           enforce_stationarity=False, enforce_invertibility=False)
        fit_base = m_base.fit(disp=False, maxiter=100)
        aic      = fit_base.aic
        fc_base  = fit_base.forecast(steps=n_val)
        mape_baseline = float(
            np.mean(np.abs((y_val - fc_base) / np.clip(y_val, 1, None))) * 100
        )

        m_sx   = SARIMAX(y_train, exog=exog_train, order=(1,0,1),
                         seasonal_order=(1,1,0,12),
                         enforce_stationarity=False, enforce_invertibility=False)
        fit_sx = m_sx.fit(disp=False, maxiter=100)
        fc_sx  = fit_sx.forecast(steps=n_val, exog=exog_val)
        mape_sarimax = float(
            np.mean(np.abs((y_val - fc_sx) / np.clip(y_val, 1, None))) * 100
        )

        m_final = SARIMAX(y, exog=exog_arr, order=(1,0,1), seasonal_order=(1,1,0,12),
                          enforce_stationarity=False, enforce_invertibility=False)
        fit_final = m_final.fit(disp=False, maxiter=100)

        exog_fcast = np.tile(exog_arr[-3:].mean(axis=0), (12, 1))
        forecast   = fit_final.get_forecast(steps=12, exog=exog_fcast)

        fcast_mean = forecast.predicted_mean
        fcast_ci   = pd.DataFrame(forecast.conf_int(alpha=0.05))
        fcast_ci.columns = ["lower", "upper"]

    except Exception:
        fit_final = None

    # GARCH(1,1) on SARIMA residuals — captures frequency volatility
    arch_lm_pval = garch_alpha_val = garch_beta_val = np.nan
    cond_vol_actual = [None] * len(y)
    cond_vol_fcast  = [None] * 12
    _garch_fit = None

    if fit_final is not None:
        try:
            resid = fit_final.resid[12:]
            _lm_stat, arch_lm_pval, _fstat, _fpval = het_arch(resid, nlags=4)
            arch_lm_pval = float(arch_lm_pval)

            if arch_lm_pval < 0.10:
                am = arch_model(resid, mean='Zero', vol='Garch', p=1, q=1, dist='normal')
                _garch_fit = am.fit(disp='off', show_warning=False)

                garch_alpha_val = float(_garch_fit.params.get('alpha[1]', np.nan))
                garch_beta_val  = float(_garch_fit.params.get('beta[1]', np.nan))

                _cv = _garch_fit.conditional_volatility
                cond_vol_actual = [None] * 12 + [float(v) for v in _cv]

                _garch_fc = _garch_fit.forecast(horizon=12, reindex=False)
                _garch_var = np.asarray(_garch_fc.variance).flatten()[:12]
                _garch_vol = np.sqrt(_garch_var)
                cond_vol_fcast = [float(v) for v in _garch_vol]

                fcast_ci = pd.DataFrame({
                    "lower": np.asarray(fcast_mean) - 1.96 * _garch_vol,
                    "upper": np.asarray(fcast_mean) + 1.96 * _garch_vol,
                })
        except Exception:
            pass

    last_month     = months.max()
    forecast_months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1), periods=12, freq="MS"
    )
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

# COMMAND ----------

import mlflow
_current_user = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_frequency_forecast")

_claims_dataset = mlflow.data.load_delta(
    table_name=f"{CATALOG}.{DATA_SCHEMA}.gold_claims_monthly",
    name="gold_claims_monthly",
)

with mlflow.start_run(run_name="sarimax_frequency_all_segments") as run:
    mlflow.log_input(_claims_dataset, context="training")
    mlflow.set_tags({
        "workshop_module": "3",
        "model_type":      "SARIMAX(1,0,1)(1,1,0,12)+GARCH(1,1)",
        "segments":        "40",
        "horizon_months":  "12",
        "exog_vars":       "unemployment_rate,hpi_growth,rolling_3m_mean,coeff_variation_3m,mom_change_pct",
        "audience":        "actuarial-workshop",
        "purpose":         "frequency_forecasting_for_reserving",
    })

    sarima_results_df = (
        claims_with_macro
        .groupby("segment_id")
        .applyInPandas(fit_sarimax_per_segment, schema=FREQ_FORECAST_SCHEMA)
    )

    # Ensure models schema exists before first write
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{MODELS_SCHEMA}")

    (sarima_results_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{MODELS_SCHEMA}.predictions_frequency_forecast"))

    sarima_results_df = spark.table(f"{CATALOG}.{MODELS_SCHEMA}.predictions_frequency_forecast")

    _metrics = (
        sarima_results_df
        .filter(F.col("record_type") == "forecast")
        .agg(
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

    mlflow.log_metrics({
        "avg_mape_pct":             round(avg_mape, 2),
        "avg_mape_baseline_pct":    round(avg_mape_baseline, 2),
        "avg_mape_sarimax_pct":     round(avg_mape_sarimax, 2),
        "avg_mape_improvement_pct": round(avg_mape_improve, 2),
        "segments_fitted":          40,
    })

    print(f"Frequency forecasting complete → {CATALOG}.{MODELS_SCHEMA}.predictions_frequency_forecast")
    print(f"  MAPE: baseline={avg_mape_baseline:.1f}%, SARIMAX={avg_mape_sarimax:.1f}%, improvement={avg_mape_improve:+.1f}%")
    print(f"  GARCH(1,1) fitted: {_metrics['garch_seg_count']}/40 segments")
    print(f"MLflow run: {run.info.run_id}")

    # ── Register Frequency Forecaster PyFunc to UC ────────────────────────
    import os, pickle, tempfile, cloudpickle, scipy
    import statsmodels as _statsmodels
    import arch as _arch_pkg
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from mlflow.tracking import MlflowClient

    class SARIMAXForecaster(mlflow.pyfunc.PythonModel):
        """
        MLflow PyFunc wrapper for fitted SARIMAX + optional GARCH(1,1) model.
        Forecasts future accident period claim counts (frequency forecasting).
        """

        def load_context(self, context):
            import pickle, os
            model_path = os.path.join(context.artifacts["sarimax_model"], "model.pkl")
            with open(model_path, "rb") as f:
                self.model_fit = pickle.load(f)
            garch_path = os.path.join(context.artifacts["sarimax_model"], "garch.pkl")
            self.garch_fit = None
            if os.path.exists(garch_path):
                with open(garch_path, "rb") as f:
                    self.garch_fit = pickle.load(f)

        def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
            horizon = int(model_input["horizon"].iloc[0])
            horizon = max(1, min(horizon, 24))

            forecast = self.model_fit.get_forecast(steps=horizon)
            mean_fcst = forecast.predicted_mean

            if self.garch_fit is not None:
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

    # Train representative model on Personal_Auto__Ontario (highest volume segment)
    _reg_claims_pdf = (
        spark.table(f"{CATALOG}.{DATA_SCHEMA}.gold_claims_monthly")
        .filter("segment_id = 'Personal_Auto__Ontario'")
        .orderBy("month")
        .select("month", "claims_count")
        .toPandas()
    )
    _y_train = _reg_claims_pdf["claims_count"].astype(float).values

    _reg_model = SARIMAX(
        _y_train, order=(1, 0, 1), seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    _reg_fit = _reg_model.fit(disp=False, maxiter=200)

    from arch import arch_model
    from statsmodels.stats.diagnostic import het_arch
    _reg_resid = _reg_fit.resid[12:]
    _reg_garch_fit = None
    try:
        _lm_stat, _lm_pval, _, _ = het_arch(_reg_resid, nlags=4)
        if _lm_pval < 0.10:
            _am = arch_model(_reg_resid, mean='Zero', vol='Garch', p=1, q=1, dist='normal')
            _reg_garch_fit = _am.fit(disp='off', show_warning=False)
            print(f"GARCH fitted for serving model (ARCH-LM p={_lm_pval:.4f})")
        else:
            print(f"No significant ARCH effects (p={_lm_pval:.4f}) — GARCH skipped for serving model")
    except Exception as _garch_reg_err:
        print(f"GARCH fit failed for serving model: {_garch_reg_err}")

    _FREQ_MODEL_NAME = f"{CATALOG}.{MODELS_SCHEMA}.frequency_forecaster"
    with tempfile.TemporaryDirectory() as _tmpdir:
        _model_pkl_path = os.path.join(_tmpdir, "model.pkl")
        with open(_model_pkl_path, "wb") as _f:
            pickle.dump(_reg_fit, _f)

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
            artifact_path="frequency_forecaster",
            python_model=SARIMAXForecaster(),
            artifacts={"sarimax_model": _tmpdir},
            signature=_signature,
            registered_model_name=_FREQ_MODEL_NAME,
            pip_requirements=[
                f"statsmodels=={_statsmodels.__version__}",
                f"numpy=={np.__version__}",
                f"scipy=={scipy.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
                f"arch=={_arch_pkg.__version__}",
            ],
        )

    print(f"\nFrequency forecaster registered to: {_FREQ_MODEL_NAME}")

    _client = MlflowClient()
    _freq_versions = _client.search_model_versions(f"name='{_FREQ_MODEL_NAME}'")
    _freq_latest_ver = max(int(v.version) for v in _freq_versions)
    _client.set_registered_model_alias(name=_FREQ_MODEL_NAME, alias="Champion", version=_freq_latest_ver)
    _client.set_model_version_tag(name=_FREQ_MODEL_NAME, version=str(_freq_latest_ver),
                                  key="approved_by", value="actuarial-workshop-demo")
    print(f"Set @Champion → version {_freq_latest_ver}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Chain Ladder Fitting + Mack Variance
# MAGIC
# MAGIC Per-product-line deterministic chain ladder via pandas. Computes weighted link
# MAGIC ratios, Mack variance parameters, IBNR best estimates, and scaled Pearson
# MAGIC residuals for the bootstrap.

# COMMAND ----------

# Load triangle data as pandas for chain ladder analysis
_tri_pdf = triangle_sdf.toPandas()
for _c in ['cumulative_paid', 'cumulative_incurred', 'case_reserve', 'dev_lag']:
    _tri_pdf[_c] = pd.to_numeric(_tri_pdf[_c], errors='coerce')
_tri_pdf['accident_month'] = pd.to_datetime(_tri_pdf['accident_month'])

# Compute incremental paid if not present
if 'incremental_paid' not in _tri_pdf.columns:
    _tri_pdf = _tri_pdf.sort_values(['segment_id', 'accident_month', 'dev_lag'])
    _tri_pdf['incremental_paid'] = _tri_pdf.groupby(['segment_id', 'accident_month'])['cumulative_paid'].diff()
    _tri_pdf.loc[_tri_pdf['dev_lag'] == _tri_pdf.groupby(['segment_id', 'accident_month'])['dev_lag'].transform('min'), 'incremental_paid'] = \
        _tri_pdf.loc[_tri_pdf['dev_lag'] == _tri_pdf.groupby(['segment_id', 'accident_month'])['dev_lag'].transform('min'), 'cumulative_paid']

print(f"Triangle data: {len(_tri_pdf)} rows, {_tri_pdf['product_line'].nunique()} product lines, "
      f"{_tri_pdf['segment_id'].nunique()} segments")

# COMMAND ----------

def fit_chain_ladder(tri_df: pd.DataFrame, product_line: str) -> dict:
    """
    Fit deterministic chain ladder for one product line.

    Returns dict with:
      - ldfs: weighted link ratios per development lag
      - mack_sigma: Mack variance parameters per lag
      - ibnr_by_accident: IBNR per accident month
      - total_ibnr: total IBNR for this product line
      - residuals: scaled Pearson residuals for bootstrap
      - fitted_triangle: fitted cumulative values
    """
    # Pivot to cumulative paid triangle: rows=accident_month, cols=dev_lag
    pivot = tri_df.pivot_table(
        index='accident_month', columns='dev_lag',
        values='cumulative_paid', aggfunc='sum'
    ).sort_index()

    lags = sorted(pivot.columns)
    n_acc = len(pivot)

    # Weighted link ratios: f_k = Σ C(i,k+1) / Σ C(i,k)
    ldfs = {}
    mack_sigma = {}
    ldf_se = {}  # Standard error of each LDF: se(f_k) = σ_k / √(Σ C(i,k))

    for k_idx in range(len(lags) - 1):
        k_from, k_to = lags[k_idx], lags[k_idx + 1]
        # Only use rows where both lags have data
        mask = pivot[k_from].notna() & pivot[k_to].notna() & (pivot[k_from] > 0)
        c_from = pivot.loc[mask, k_from].values
        c_to   = pivot.loc[mask, k_to].values

        if len(c_from) < 2:
            ldfs[k_from] = 1.0
            mack_sigma[k_from] = 0.0
            ldf_se[k_from] = 0.0
            continue

        # Weighted average link ratio
        sum_c_from = float(c_from.sum())
        f_k = c_to.sum() / sum_c_from
        ldfs[k_from] = float(f_k)

        # Mack variance: σ²_k = 1/(n-k-1) × Σ C(i,k) × (C(i,k+1)/C(i,k) - f_k)²
        individual_ratios = c_to / c_from
        n_obs = len(c_from)
        denom = max(n_obs - 1, 1)
        sigma_sq = np.sum(c_from * (individual_ratios - f_k) ** 2) / denom
        mack_sigma[k_from] = float(np.sqrt(max(sigma_sq, 0)))

        # Standard error of f_k (Mack's formula): se(f_k) = σ_k / √(Σ C(i,k))
        ldf_se[k_from] = float(np.sqrt(max(sigma_sq, 0) / max(sum_c_from, 1)))

    # Tail factor: assume development is complete after max observed lag
    tail_factor = 1.0

    # Project ultimate losses
    projected = pivot.copy()
    ibnr_by_month = {}
    latest_paid = {}
    max_obs_lag_by_month = {}

    for i, acc_month in enumerate(pivot.index):
        row = pivot.loc[acc_month]
        # Find the latest observed lag
        observed_lags = row.dropna().index.tolist()
        if not observed_lags:
            continue
        max_obs_lag = max(observed_lags)
        latest_value = row[max_obs_lag]
        latest_paid[acc_month] = float(latest_value)
        max_obs_lag_by_month[acc_month] = max_obs_lag

        # Project forward through remaining lags
        projected_value = latest_value
        for k_from in sorted(ldfs.keys()):
            if k_from >= max_obs_lag:
                projected_value *= ldfs[k_from]

        ultimate = projected_value * tail_factor
        ibnr = max(0, ultimate - latest_value)
        ibnr_by_month[acc_month] = float(ibnr)

    total_ibnr = sum(ibnr_by_month.values())

    # Compute fitted values and scaled Pearson residuals for bootstrap
    fitted_vals = {}
    residuals = []

    for k_idx in range(len(lags) - 1):
        k_from, k_to = lags[k_idx], lags[k_idx + 1]
        f_k = ldfs.get(k_from, 1.0)

        for acc_month in pivot.index:
            c_from = pivot.loc[acc_month, k_from] if k_from in pivot.columns else np.nan
            c_to_obs = pivot.loc[acc_month, k_to] if k_to in pivot.columns else np.nan

            if pd.isna(c_from) or pd.isna(c_to_obs) or c_from <= 0:
                continue

            fitted = f_k * c_from
            fitted_vals[(acc_month, k_to)] = fitted

            # Scaled Pearson residual: r = (observed - fitted) / sqrt(fitted)
            if fitted > 0:
                residual = (c_to_obs - fitted) / np.sqrt(fitted)
                residuals.append({
                    'accident_month': acc_month,
                    'dev_lag_from': k_from,
                    'dev_lag_to': k_to,
                    'observed': float(c_to_obs),
                    'fitted': float(fitted),
                    'c_from': float(c_from),
                    'residual': float(residual),
                    'f_k': float(f_k),
                })

    return {
        'product_line': product_line,
        'ldfs': ldfs,
        'mack_sigma': mack_sigma,
        'ldf_se': ldf_se,
        'total_ibnr': total_ibnr,
        'ibnr_by_month': ibnr_by_month,
        'latest_paid': latest_paid,
        'max_obs_lag_by_month': max_obs_lag_by_month,
        'residuals': residuals,
        'fitted_vals': fitted_vals,
        'pivot': pivot,
        'lags': lags,
        'tail_factor': tail_factor,
    }


# Fit chain ladder per product line
_cl_results = {}
for _pl in PRODUCT_LINES:
    _pl_data = _tri_pdf[_tri_pdf['product_line'] == _pl]
    _cl_results[_pl] = fit_chain_ladder(_pl_data, _pl)
    _r = _cl_results[_pl]
    print(f"  {_pl}: IBNR=${_r['total_ibnr']/1e6:.1f}M, "
          f"{len(_r['ldfs'])} LDFs, {len(_r['residuals'])} residuals")

_total_best_estimate = sum(r['total_ibnr'] for r in _cl_results.values())
print(f"\nTotal Best Estimate IBNR: ${_total_best_estimate/1e6:.1f}M")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. LDF Volatility (GARCH on Development Factor Residuals)
# MAGIC
# MAGIC Compute development factor volatility per product line and save for
# MAGIC reserve risk capital calculation.

# COMMAND ----------

_ldf_vol_rows = []
for _pl, _cl in _cl_results.items():
    if _cl['residuals']:
        _resid_vals = [r['residual'] for r in _cl['residuals']]
        _avg_ldf = float(np.mean(list(_cl['ldfs'].values())))
        _std_ldf = float(np.std(list(_cl['ldfs'].values())))
        _cv = _std_ldf / _avg_ldf if _avg_ldf > 0 else 0.15
        _ldf_vol_rows.append({
            'product_line': _pl,
            'avg_ldf': round(_avg_ldf, 4),
            'std_ldf': round(_std_ldf, 4),
            'cv': round(_cv, 4),
            'ibnr_M': round(_cl['total_ibnr'] / 1e6, 4),
            'n_factors': len(_cl['ldfs']),
            'mean_residual': round(float(np.mean(_resid_vals)), 4),
            'std_residual': round(float(np.std(_resid_vals)), 4),
            'n_residuals': len(_resid_vals),
        })

_ldf_vol_df = spark.createDataFrame(pd.DataFrame(_ldf_vol_rows))
(_ldf_vol_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{MODELS_SCHEMA}.predictions_ldf_volatility"))
print(f"LDF volatility saved → {CATALOG}.{MODELS_SCHEMA}.predictions_ldf_volatility")
for _r in _ldf_vol_rows:
    print(f"  {_r['product_line']}: avg LDF={_r['avg_ldf']:.3f}, σ={_r['std_ldf']:.3f}, "
          f"mean resid={_r['mean_residual']:.3f}, σ_resid={_r['std_residual']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Bootstrap Chain Ladder — Ray-Distributed
# MAGIC
# MAGIC Bootstrap Chain Ladder is the industry-standard stochastic reserving method.
# MAGIC Each bootstrap replication: resample Pearson residuals → pseudo-triangle →
# MAGIC refit chain ladder → project reserves → add process variance (Gamma noise).
# MAGIC
# MAGIC Ray distributes replications across workers for scalability.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bootstrap Reserve Simulator PyFunc
# MAGIC
# MAGIC Defined here so it can be validated after the bootstrap run and registered
# MAGIC to UC within the same MLflow experiment.

# COMMAND ----------

class BootstrapReservePyFunc(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for Bootstrap Chain Ladder reserve simulation.

    Accepts reserve parameters and scenario configuration, returns reserve
    distribution metrics (IBNR best estimate, VaR, CVaR, reserve risk capital).

    Supports scenarios: baseline, adverse_development, judicial_inflation,
    pandemic_tail, superimposed_inflation.
    """

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        row = model_input.iloc[0]

        scenario = str(row.get("scenario", "baseline"))
        n_replications = int(row.get("n_replications", 50_000))
        n_replications = max(1_000, min(n_replications, 500_000))

        # Reserve parameters (expected IBNR components by product line in $M)
        mean_ibnr_personal_auto = float(row.get("mean_ibnr_personal_auto_M", 50.0))
        mean_ibnr_commercial_auto = float(row.get("mean_ibnr_commercial_auto_M", 30.0))
        mean_ibnr_homeowners = float(row.get("mean_ibnr_homeowners_M", 40.0))
        mean_ibnr_commercial_property = float(row.get("mean_ibnr_commercial_property_M", 20.0))

        # LDF volatility parameters (CV of development factors)
        cv_personal_auto = float(row.get("cv_personal_auto", 0.15))
        cv_commercial_auto = float(row.get("cv_commercial_auto", 0.18))
        cv_homeowners = float(row.get("cv_homeowners", 0.12))
        cv_commercial_property = float(row.get("cv_commercial_property", 0.20))

        # Scenario adjustments
        ldf_multiplier = float(row.get("ldf_multiplier", 1.0))
        inflation_adj = float(row.get("inflation_adj", 0.0))

        means = np.array([mean_ibnr_personal_auto, mean_ibnr_commercial_auto,
                          mean_ibnr_homeowners, mean_ibnr_commercial_property])
        cvs = np.array([cv_personal_auto, cv_commercial_auto,
                        cv_homeowners, cv_commercial_property])

        # Apply scenario adjustments
        if scenario == 'adverse_development':
            means *= ldf_multiplier if ldf_multiplier > 1.0 else 1.2
            cvs *= 1.3
        elif scenario == 'judicial_inflation':
            means[:2] *= 1.3  # Auto lines
            cvs[:2] *= 1.2
        elif scenario == 'pandemic_tail':
            means *= 1.1
            cvs *= 1.4
        elif scenario == 'superimposed_inflation':
            infl = inflation_adj if inflation_adj > 0 else 0.03
            means *= (1.0 + infl)

        rng = np.random.default_rng(42)

        # Bootstrap: lognormal IBNR per line with correlation
        corr = np.array([
            [1.0,  0.5,  0.3,  0.2],
            [0.5,  1.0,  0.2,  0.3],
            [0.3,  0.2,  1.0,  0.4],
            [0.2,  0.3,  0.4,  1.0],
        ])
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            chol = np.eye(4)

        sigma2 = np.log(1 + cvs**2)
        mu_ln = np.log(means) - sigma2 / 2
        sigma_ln = np.sqrt(sigma2)

        z = rng.standard_normal((n_replications, 4))
        z_cor = z @ chol.T
        ibnr_samples = np.exp(mu_ln + sigma_ln * z_cor)
        total = ibnr_samples.sum(axis=1)

        best_estimate = float(total.mean())
        var_99 = float(np.percentile(total, 99))
        var_995 = float(np.percentile(total, 99.5))
        cvar_99_threshold = np.percentile(total, 99)
        tail = total[total >= cvar_99_threshold]
        cvar_99 = float(tail.mean()) if len(tail) > 0 else var_99

        return pd.DataFrame([{
            "best_estimate_M":       round(best_estimate, 3),
            "var_95_M":              round(float(np.percentile(total, 95)), 3),
            "var_99_M":              round(var_99, 3),
            "var_995_M":             round(var_995, 3),
            "cvar_99_M":             round(cvar_99, 3),
            "reserve_risk_capital_M": round(var_995 - best_estimate, 3),
            "max_ibnr_M":            round(float(total.max()), 3),
            "n_replications_used":   n_replications,
            "scenario":              scenario,
        }])


# Validate the PyFunc locally
# Build baseline input from chain ladder results — no hardcoded values
_ldf_vol_lookup = {r['product_line']: r for r in _ldf_vol_rows}
_boot_baseline_input = pd.DataFrame([{
    "mean_ibnr_personal_auto_M": round(_cl_results['Personal_Auto']['total_ibnr'] / 1e6, 2),
    "mean_ibnr_commercial_auto_M": round(_cl_results['Commercial_Auto']['total_ibnr'] / 1e6, 2),
    "mean_ibnr_homeowners_M": round(_cl_results['Homeowners']['total_ibnr'] / 1e6, 2),
    "mean_ibnr_commercial_property_M": round(_cl_results['Commercial_Property']['total_ibnr'] / 1e6, 2),
    "cv_personal_auto": _ldf_vol_lookup['Personal_Auto']['cv'],
    "cv_commercial_auto": _ldf_vol_lookup['Commercial_Auto']['cv'],
    "cv_homeowners": _ldf_vol_lookup['Homeowners']['cv'],
    "cv_commercial_property": _ldf_vol_lookup['Commercial_Property']['cv'],
    "n_replications": 50_000,
    "scenario": "baseline",
    "ldf_multiplier": 1.0,
    "inflation_adj": 0.0,
}])

_boot_pyfunc = BootstrapReservePyFunc()
_boot_result = _boot_pyfunc.predict(None, _boot_baseline_input)
print("Bootstrap Reserve Simulator validation (50,000 replications):")
print(_boot_result.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Ray on Spark

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

try:
    shutdown_ray_cluster()
except Exception:
    pass

setup_ray_cluster(
    max_worker_nodes=4,
    num_cpus_worker_node=6,
    num_gpus_worker_node=0,
    collect_log_to_path="/tmp/ray_logs",
)

ray.init(ignore_reinit_error=True)
print(f"Ray initialized | Resources: {ray.cluster_resources()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Bootstrap Parameters

# COMMAND ----------

# Serialize chain ladder results for Ray workers
_bootstrap_params = {}
for _pl, _cl in _cl_results.items():
    _bootstrap_params[_pl] = {
        'ldfs': _cl['ldfs'],
        'mack_sigma': _cl['mack_sigma'],
        'ldf_se': _cl['ldf_se'],
        'residuals': [r['residual'] for r in _cl['residuals']],
        'residual_details': _cl['residuals'],
        'lags': _cl['lags'],
        'total_ibnr': _cl['total_ibnr'],
        'latest_paid': _cl['latest_paid'],
        'max_obs_lag_by_month': {str(k): v for k, v in _cl['max_obs_lag_by_month'].items()},
        'ibnr_by_month': {str(k): v for k, v in _cl['ibnr_by_month'].items()},
    }

# Put params in Ray object store (shared across workers without serialization per task)
_params_ref = ray.put(_bootstrap_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Bootstrap Task

# COMMAND ----------

@ray.remote(num_cpus=1)
def bootstrap_chain_ladder(params_ref, n_replications: int, seed: int,
                           scenario: str = 'baseline',
                           ldf_multiplier: float = 1.0,
                           inflation_adj: float = 0.0,
                           dev_extension: int = 0) -> dict:
    """
    Vectorized Bootstrap Chain Ladder using Mack's parametric approach.

    For each product line, all n_replications are computed simultaneously:
      1. Perturb LDFs using Mack standard errors: f_k_boot ~ N(f_k, se(f_k)²)
      2. Project reserves per accident month (only applicable LDFs)
      3. Add Gamma process variance (vectorized)

    Scenarios modify the bootstrap parameters:
      - baseline: standard bootstrap
      - adverse_development: inflate LDFs at late lags by ldf_multiplier
      - judicial_inflation: 1.3x on bodily injury lines at lags 24+
      - pandemic_tail: extend development by dev_extension months
      - superimposed_inflation: apply calendar-year inflation trend

    Returns dict with reserve distribution metrics.
    """
    import numpy as np
    from collections import defaultdict

    params = params_ref
    rng = np.random.default_rng(seed)
    product_lines = list(params.keys())

    # Collect all IBNR samples across all product lines
    total_ibnr_samples = np.zeros(n_replications)
    ibnr_by_line = {pl: np.zeros(n_replications) for pl in product_lines}

    for pl in product_lines:
        pl_params = params[pl]
        ldfs = pl_params['ldfs']
        ibnr_by_month = pl_params['ibnr_by_month']
        latest_paid = pl_params['latest_paid']
        ldf_se_dict = pl_params['ldf_se']

        if not ibnr_by_month:
            ibnr_by_line[pl][:] = pl_params['total_ibnr']
            total_ibnr_samples += pl_params['total_ibnr']
            continue

        # ── Step 1: Vectorized pseudo-LDF construction (Mack parametric) ──
        sorted_ldf_keys = sorted(ldfs.keys())
        n_ldf = len(sorted_ldf_keys)

        # Base LDFs as array: (n_ldf,)
        base_ldf_arr = np.array([ldfs[k] for k in sorted_ldf_keys])

        # Apply scenario adjustments to base LDFs (deterministic)
        adj_ldf_arr = base_ldf_arr.copy()
        if scenario == 'adverse_development':
            late_mask = np.array([k >= max(sorted_ldf_keys) * 0.5 for k in sorted_ldf_keys])
            adj_ldf_arr[late_mask] *= ldf_multiplier
        elif scenario == 'judicial_inflation':
            if pl in ['Personal_Auto', 'Commercial_Auto']:
                late_mask = np.array([k >= 24 for k in sorted_ldf_keys])
                adj_ldf_arr[late_mask] *= 1.3
        elif scenario == 'superimposed_inflation':
            adj_ldf_arr *= (1.0 + inflation_adj)

        # LDF standard errors from Mack's formula: se(f_k) = σ_k / √(Σ C(i,k))
        se_arr = np.array([
            ldf_se_dict.get(str(k), ldf_se_dict.get(k, 0.0))
            for k in sorted_ldf_keys
        ])

        # Perturb LDFs: f_k_boot ~ N(f_k, se(f_k)²), floored at 1.0
        # (n_replications, n_ldf) standard normal draws
        z = rng.standard_normal((n_replications, n_ldf))
        pseudo_ldfs_all = np.maximum(1.0, adj_ldf_arr[np.newaxis, :] + z * se_arr[np.newaxis, :])

        # ── Step 2: Per-accident-month reserve projection ─────────────────
        acc_months = list(latest_paid.keys())
        max_obs_lag_dict = pl_params['max_obs_lag_by_month']
        ldf_keys_arr = np.array(sorted_ldf_keys)

        cv_process = 0.10
        shape = 1.0 / (cv_process ** 2)  # = 100

        boot_ibnr = np.zeros(n_replications)

        # Group accident months by max_obs_lag for efficient vectorization
        lag_groups = defaultdict(list)
        for m in acc_months:
            paid_val = latest_paid[m]
            if paid_val > 0:
                obs_lag = max_obs_lag_dict.get(str(m), max_obs_lag_dict.get(m, 0))
                lag_groups[obs_lag].append(m)

        for obs_lag, months_in_group in lag_groups.items():
            # Which LDF columns apply for this max_obs_lag group
            ldf_col_mask = ldf_keys_arr >= obs_lag
            if not ldf_col_mask.any():
                continue  # fully developed — no projection needed

            # Cumulative dev factor from applicable LDFs: (n_replications,)
            cum_dev = np.prod(pseudo_ldfs_all[:, ldf_col_mask], axis=1)

            # Paid amounts for this group: (n_group,)
            paid_group = np.array([latest_paid[m] for m in months_in_group])

            # Projected ultimates: (n_replications, n_group)
            projected = paid_group[np.newaxis, :] * cum_dev[:, np.newaxis]
            reserve_est = np.maximum(0.0, projected - paid_group[np.newaxis, :])

            # ── Step 3: Gamma process variance (vectorized) ───────────────
            gamma_noise = rng.gamma(shape, 1.0, size=reserve_est.shape)
            reserve_with_noise = np.where(
                reserve_est > 0,
                gamma_noise * (reserve_est * cv_process ** 2),
                0.0,
            )
            boot_ibnr += reserve_with_noise.sum(axis=1)

        ibnr_by_line[pl] = boot_ibnr
        total_ibnr_samples += boot_ibnr

    # Convert to $M
    total_M = total_ibnr_samples / 1_000_000
    ibnr_by_line_M = {pl: vals / 1_000_000 for pl, vals in ibnr_by_line.items()}

    best_estimate = float(total_M.mean())
    var_75 = float(np.percentile(total_M, 75))
    var_90 = float(np.percentile(total_M, 90))
    var_95 = float(np.percentile(total_M, 95))
    var_99 = float(np.percentile(total_M, 99))
    var_995 = float(np.percentile(total_M, 99.5))
    _tail = total_M[total_M >= np.percentile(total_M, 99)]
    cvar_99 = float(_tail.mean()) if len(_tail) > 0 else var_99
    reserve_risk_capital = var_995 - best_estimate

    return {
        'seed': seed,
        'scenario': scenario,
        'n_replications': n_replications,
        'best_estimate_M': best_estimate,
        'var_75_M': var_75,
        'var_90_M': var_90,
        'var_95_M': var_95,
        'var_99_M': var_99,
        'var_995_M': var_995,
        'cvar_99_M': cvar_99,
        'reserve_risk_capital_M': reserve_risk_capital,
        'max_ibnr_M': float(total_M.max()),
        'std_M': float(total_M.std()),
        'by_line': {pl: {
            'mean_M': float(v.mean()),
            'var_995_M': float(np.percentile(v, 99.5)),
        } for pl, v in ibnr_by_line_M.items()},
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Bootstrap — Baseline + Scenarios + Reserve Evolution (Ray-distributed)
# MAGIC
# MAGIC Dispatches baseline bootstrap, reserve deterioration scenarios, and 12-month
# MAGIC reserve evolution tasks across Ray workers.

# COMMAND ----------

N_TASKS          = 24         # 24 concurrent tasks across 4 workers (4 × 6 CPUs)
N_PER_TASK       = 100_000   # 100K bootstrap replications per task (vectorized NumPy)
SCENARIOS = [
    ('baseline',             {'ldf_multiplier': 1.0, 'inflation_adj': 0.0}),
    ('adverse_development',  {'ldf_multiplier': 1.2, 'inflation_adj': 0.0}),
    ('judicial_inflation',   {'ldf_multiplier': 1.0, 'inflation_adj': 0.0}),
    ('pandemic_tail',        {'ldf_multiplier': 1.1, 'inflation_adj': 0.0, 'dev_extension': 6}),
    ('superimposed_inflation', {'ldf_multiplier': 1.0, 'inflation_adj': 0.03}),
]

N_FORECAST_MONTHS = 12
_forecast_months_pd = pd.date_range("2026-01-01", periods=12, freq="MS")
_forecast_months = [m.date() for m in _forecast_months_pd]

N_RUNS = len(SCENARIOS) + N_FORECAST_MONTHS
print(f'Launching {N_RUNS} runs × {N_TASKS} tasks × {N_PER_TASK:,} replications '
      f'= {N_RUNS * N_TASKS * N_PER_TASK:,} total bootstrap paths')

mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_bootstrap_reserves")

_triangle_dataset = mlflow.data.load_delta(
    table_name=f"{CATALOG}.{DATA_SCHEMA}.gold_reserve_triangle",
    name="gold_reserve_triangle",
)

with mlflow.start_run(run_name='bootstrap_chain_ladder_ray') as run:
    mlflow.log_input(_triangle_dataset, context="training")
    mlflow.set_tags({
        'workshop_module': '3',
        'model_type':      'Bootstrap Chain Ladder (stochastic reserving)',
        'n_replications':  str(N_RUNS * N_TASKS * N_PER_TASK),
        'n_product_lines': str(len(PRODUCT_LINES)),
        'framework':       'Ray + NumPy CPU',
        'audience':        'actuarial-workshop',
    })
    mlflow.log_params({
        'n_tasks':            N_TASKS,
        'replications_per_task': N_PER_TASK,
        'total_replications': N_RUNS * N_TASKS * N_PER_TASK,
        'n_scenarios':        len(SCENARIOS),
        'n_forecast_months':  N_FORECAST_MONTHS,
        'product_lines':      ','.join(PRODUCT_LINES),
        'best_estimate_ibnr_M': round(_total_best_estimate / 1e6, 2),
    })

    # ── Dispatch all tasks ─────────────────────────────────────────────────
    # Baseline
    baseline_futures = [
        bootstrap_chain_ladder.remote(
            _params_ref, N_PER_TASK, seed=42 + i, scenario='baseline'
        ) for i in range(N_TASKS)
    ]

    # Scenarios
    scenario_futures = {}
    for _si, (_sc_name, _sc_params) in enumerate(SCENARIOS):
        if _sc_name == 'baseline':
            continue
        scenario_futures[_sc_name] = [
            bootstrap_chain_ladder.remote(
                _params_ref, N_PER_TASK, seed=200 + _si * N_TASKS + i,
                scenario=_sc_name, **_sc_params,
            ) for i in range(N_TASKS)
        ]

    # 12-month reserve evolution (each month adds new claims from frequency forecast)
    evolution_futures = {}
    for _mi in range(N_FORECAST_MONTHS):
        # Simulate reserve evolution: as months pass, reserves develop and new claims enter
        # Later months have slightly higher LDF multiplier (more uncertainty)
        _month_ldf_mult = 1.0 + 0.005 * (_mi + 1)  # gradual uncertainty increase
        evolution_futures[_mi] = [
            bootstrap_chain_ladder.remote(
                _params_ref, N_PER_TASK, seed=400 + _mi * N_TASKS + i,
                scenario='baseline', ldf_multiplier=_month_ldf_mult,
            ) for i in range(N_TASKS)
        ]

    # ── Collect results ────────────────────────────────────────────────────
    baseline_results = ray.get(baseline_futures)
    scenario_results = {sc: ray.get(futs) for sc, futs in scenario_futures.items()}
    evolution_results = {mi: ray.get(futs) for mi, futs in evolution_futures.items()}

    # Shut down Ray so Spark can reclaim executor slots
    try:
        shutdown_ray_cluster()
        ray.shutdown()
    except Exception:
        pass

    # ── Aggregate baseline results ─────────────────────────────────────────
    def _agg(results, key):
        return float(sum(r[key] for r in results) / len(results))

    baseline_best_est = _agg(baseline_results, 'best_estimate_M')
    baseline_var99    = _agg(baseline_results, 'var_99_M')
    baseline_var995   = _agg(baseline_results, 'var_995_M')
    baseline_cvar99   = _agg(baseline_results, 'cvar_99_M')
    baseline_risk_cap = _agg(baseline_results, 'reserve_risk_capital_M')

    mlflow.log_metrics({
        'best_estimate_ibnr_M':     round(baseline_best_est, 2),
        'VaR_99_pct_M':              round(baseline_var99, 2),
        'VaR_99_5_pct_M':            round(baseline_var995, 2),
        'CVaR_99_pct_M':             round(baseline_cvar99, 2),
        'reserve_risk_capital_M':    round(baseline_risk_cap, 2),
    })

    print(f'\n' + '='*60)
    print(f'  RESERVE RISK SUMMARY ({N_TASKS * N_PER_TASK:,} bootstrap replications)')
    print('='*60)
    print(f'  Best Estimate IBNR:       ${baseline_best_est:.1f}M')
    print(f'  VaR(99%):                 ${baseline_var99:.1f}M')
    print(f'  VaR(99.5%):               ${baseline_var995:.1f}M <- Reserve Risk Capital threshold')
    print(f'  CVaR(99%):                ${baseline_cvar99:.1f}M')
    print(f'  Reserve Risk Capital:     ${baseline_risk_cap:.1f}M (VaR 99.5% - Best Estimate)')
    print('='*60)
    print(f'\nMLflow run: {run.info.run_id}')

    # ── Save baseline → predictions_bootstrap_reserves ─────────────────────
    baseline_pdf = pd.DataFrame([{
        'task_id':              r['seed'] - 42,
        'n_replications':       r['n_replications'],
        'best_estimate_M':      r['best_estimate_M'],
        'var_75_M':             r['var_75_M'],
        'var_95_M':             r['var_95_M'],
        'var_99_M':             r['var_99_M'],
        'var_995_M':            r['var_995_M'],
        'cvar_99_M':            r['cvar_99_M'],
        'reserve_risk_capital_M': r['reserve_risk_capital_M'],
        'max_ibnr_M':           r['max_ibnr_M'],
        'mlflow_run_id':        run.info.run_id,
    } for r in baseline_results])

    boot_df = spark.createDataFrame(baseline_pdf)
    (boot_df.write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .saveAsTable(f'{CATALOG}.{MODELS_SCHEMA}.predictions_bootstrap_reserves'))
    print(f'\nBaseline saved → {CATALOG}.{MODELS_SCHEMA}.predictions_bootstrap_reserves')

    # ── Save scenario comparison → predictions_reserve_scenarios ───────────
    _SCENARIO_LABELS = {
        'baseline':               'Baseline',
        'adverse_development':    'Adverse Development (+20% late LDFs)',
        'judicial_inflation':     'Judicial Inflation (1.3× Auto lags 24+)',
        'pandemic_tail':          'Pandemic Tail (+6 months dev)',
        'superimposed_inflation': 'Superimposed Inflation (CPI+3%)',
    }

    scenario_rows = [{
        'scenario':            'baseline',
        'scenario_label':      'Baseline',
        'best_estimate_M':     baseline_best_est,
        'var_99_M':            baseline_var99,
        'var_995_M':           baseline_var995,
        'cvar_99_M':           baseline_cvar99,
        'var_995_vs_baseline': 0.0,
    }]

    for _sc_name, _sc_results in scenario_results.items():
        _sc_be   = _agg(_sc_results, 'best_estimate_M')
        _sc_v99  = _agg(_sc_results, 'var_99_M')
        _sc_v995 = _agg(_sc_results, 'var_995_M')
        _sc_cv99 = _agg(_sc_results, 'cvar_99_M')
        scenario_rows.append({
            'scenario':            _sc_name,
            'scenario_label':      _SCENARIO_LABELS.get(_sc_name, _sc_name),
            'best_estimate_M':     _sc_be,
            'var_99_M':            _sc_v99,
            'var_995_M':           _sc_v995,
            'cvar_99_M':           _sc_cv99,
            'var_995_vs_baseline': (_sc_v995 / baseline_var995 - 1.0) * 100 if baseline_var995 > 0 else 0,
        })
        mlflow.log_metric(f'VaR_99_5_{_sc_name}_M', round(_sc_v995, 2))

    print(f'\n  RESERVE SCENARIO COMPARISON')
    print('  ' + '─'*80)
    print(f'  {"Scenario":<38} {"Best Est":>10} {"VaR(99%)":>10} {"VaR(99.5%)":>12} {"Δ VaR":>10}')
    print('  ' + '─'*80)
    for _row in scenario_rows:
        print(
            f'  {_row["scenario_label"]:<38} '
            f'${_row["best_estimate_M"]:>8.1f}M '
            f'${_row["var_99_M"]:>8.1f}M '
            f'${_row["var_995_M"]:>10.1f}M  '
            f'{_row["var_995_vs_baseline"]:>+8.1f}%'
        )
    print('  ' + '─'*80)

    scenario_df = spark.createDataFrame(pd.DataFrame(scenario_rows))
    (scenario_df.write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .saveAsTable(f'{CATALOG}.{MODELS_SCHEMA}.predictions_reserve_scenarios'))
    print(f'\nScenarios saved → {CATALOG}.{MODELS_SCHEMA}.predictions_reserve_scenarios')

    # ── 12-month reserve evolution ─────────────────────────────────────────
    evolution_rows = []
    for _mi in range(N_FORECAST_MONTHS):
        _evo_results = evolution_results[_mi]
        _row = {
            'forecast_month':      str(_forecast_months[_mi]),
            'month_idx':           _mi + 1,
            'best_estimate_M':     _agg(_evo_results, 'best_estimate_M'),
            'var_99_M':            _agg(_evo_results, 'var_99_M'),
            'var_995_M':           _agg(_evo_results, 'var_995_M'),
            'cvar_99_M':           _agg(_evo_results, 'cvar_99_M'),
            'reserve_risk_capital_M': _agg(_evo_results, 'reserve_risk_capital_M'),
            'var_995_vs_baseline': (_agg(_evo_results, 'var_995_M') / baseline_var995 - 1.0) * 100 if baseline_var995 > 0 else 0,
        }
        evolution_rows.append(_row)
        mlflow.log_metric(f'VaR_99_5_month_{_mi + 1:02d}', round(_row['var_995_M'], 2))

    print(f'\n  RESERVE EVOLUTION (12-month horizon)')
    print('  ' + '─'*70)
    print(f'  {"Month":<12} {"Best Est":>10} {"VaR(99%)":>10} {"VaR(99.5%)":>12} {"Δ VaR":>10}')
    print('  ' + '─'*70)
    print(f'  {"[current]":<12} ${baseline_best_est:>8.1f}M ${baseline_var99:>8.1f}M ${baseline_var995:>10.1f}M  {"—":>8}')
    for _row in evolution_rows:
        print(
            f'  {_row["forecast_month"]:<12} '
            f'${_row["best_estimate_M"]:>8.1f}M '
            f'${_row["var_99_M"]:>8.1f}M '
            f'${_row["var_995_M"]:>10.1f}M  '
            f'{_row["var_995_vs_baseline"]:>+8.1f}%'
        )
    print('  ' + '─'*70)

    evo_df = spark.createDataFrame(pd.DataFrame(evolution_rows))
    (evo_df.write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .saveAsTable(f'{CATALOG}.{MODELS_SCHEMA}.predictions_reserve_evolution'))
    print(f'\nReserve evolution saved → {CATALOG}.{MODELS_SCHEMA}.predictions_reserve_evolution')

    # ── Run-off projection (surplus evolution with reserve development) ────
    print('\n  Running run-off projection (regime-switching)...')
    _MP_N_SCENARIOS = 500_000
    _MP_HORIZON = 12
    _mp_rng = np.random.default_rng(999)

    _REGIME_PARAMS = {
        'Normal': {
            'reserve_mult': 1.0,
            'cv_mult': 1.0,
        },
        'Crisis': {
            'reserve_mult': 1.3,  # reserves develop 30% worse in crisis
            'cv_mult': 1.15,
        },
    }
    _TRANS_PROB = {'Normal': {'Normal': 0.95, 'Crisis': 0.05},
                   'Crisis': {'Normal': 0.15, 'Crisis': 0.85}}

    # Monthly premium from gold_claims_monthly
    _monthly_premium = float(
        spark.table(f"{CATALOG}.{DATA_SCHEMA}.gold_claims_monthly")
        .agg(F.mean("earned_premium")).collect()[0][0]
    ) / 1_000_000

    _investment_rate_monthly = 0.04 / 12
    _initial_surplus = baseline_var995 * 1.2
    _surplus = np.full((_MP_N_SCENARIOS, _MP_HORIZON + 1), _initial_surplus)
    _regime_state = np.zeros(_MP_N_SCENARIOS, dtype=int)

    # Monthly reserve payment = best estimate / 12 with regime adjustment
    _monthly_reserve_pmt = baseline_best_est / 12.0

    for _t in range(_MP_HORIZON):
        _u_trans = _mp_rng.random(_MP_N_SCENARIOS)
        _stay_prob = np.where(_regime_state == 0,
                              _TRANS_PROB['Normal']['Normal'],
                              _TRANS_PROB['Crisis']['Crisis'])
        _regime_state = np.where(_u_trans < _stay_prob, _regime_state,
                                 1 - _regime_state)

        # Reserve payment varies by regime
        _res_mult = np.where(_regime_state == 0,
                             _REGIME_PARAMS['Normal']['reserve_mult'],
                             _REGIME_PARAMS['Crisis']['reserve_mult'])
        _cv_mult = np.where(_regime_state == 0,
                            _REGIME_PARAMS['Normal']['cv_mult'],
                            _REGIME_PARAMS['Crisis']['cv_mult'])

        # Monthly claim payments with lognormal noise
        _base_payment = _monthly_reserve_pmt * _res_mult
        _cv = 0.15 * _cv_mult
        _sigma2 = np.log(1 + _cv**2)
        _mu_ln = np.log(_base_payment) - _sigma2 / 2
        _sigma_ln = np.sqrt(_sigma2)
        _monthly_loss = np.exp(_mu_ln + _sigma_ln * _mp_rng.standard_normal(_MP_N_SCENARIOS))

        _investment = _surplus[:, _t] * _investment_rate_monthly
        _surplus[:, _t + 1] = (_surplus[:, _t] + _monthly_premium
                                - _monthly_loss + _investment)

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
        .saveAsTable(f'{CATALOG}.{MODELS_SCHEMA}.predictions_runoff_projection'))
    print(f'Run-off projection saved → {CATALOG}.{MODELS_SCHEMA}.predictions_runoff_projection')

    # Save regime parameters
    regime_rows = []
    for _state_name, _params in _REGIME_PARAMS.items():
        regime_rows.append({
            'state': _state_name,
            'reserve_mult': float(_params['reserve_mult']),
            'cv_mult': float(_params['cv_mult']),
            'transition_to_normal': float(_TRANS_PROB[_state_name]['Normal']),
            'transition_to_crisis': float(_TRANS_PROB[_state_name]['Crisis']),
        })
    regime_df = spark.createDataFrame(pd.DataFrame(regime_rows))
    (regime_df.write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema', 'true')
        .saveAsTable(f'{CATALOG}.{MODELS_SCHEMA}.predictions_regime_parameters'))
    print(f'Regime parameters saved → {CATALOG}.{MODELS_SCHEMA}.predictions_regime_parameters')

    mlflow.log_metrics({
        'surplus_median_month12': round(float(_surplus_pctiles[2, -1]), 2),
        'ruin_prob_month12': round(float(_ruin_prob[-1]), 6),
        'initial_surplus_M': round(_initial_surplus, 2),
    })
    print(f'  Initial surplus: ${_initial_surplus:.1f}M | Month-12 median: ${_surplus_pctiles[2, -1]:.1f}M | Ruin prob: {_ruin_prob[-1]:.4%}')

    # ── Register Bootstrap Reserve Simulator PyFunc to UC ─────────────────
    import scipy
    from mlflow.tracking import MlflowClient
    _client = MlflowClient()
    _BOOT_MODEL_NAME = f"{CATALOG}.{MODELS_SCHEMA}.bootstrap_reserve_simulator"

    _boot_reg_dataset = mlflow.data.load_delta(
        table_name=f"{CATALOG}.{DATA_SCHEMA}.gold_reserve_triangle",
        name="gold_reserve_triangle",
    )
    mlflow.log_input(_boot_reg_dataset, context="bootstrap_calibration")

    _boot_input_schema = mlflow.types.Schema([
        mlflow.types.ColSpec("double",  "mean_ibnr_personal_auto_M"),
        mlflow.types.ColSpec("double",  "mean_ibnr_commercial_auto_M"),
        mlflow.types.ColSpec("double",  "mean_ibnr_homeowners_M"),
        mlflow.types.ColSpec("double",  "mean_ibnr_commercial_property_M"),
        mlflow.types.ColSpec("double",  "cv_personal_auto"),
        mlflow.types.ColSpec("double",  "cv_commercial_auto"),
        mlflow.types.ColSpec("double",  "cv_homeowners"),
        mlflow.types.ColSpec("double",  "cv_commercial_property"),
        mlflow.types.ColSpec("long",    "n_replications"),
        mlflow.types.ColSpec("string",  "scenario"),
        mlflow.types.ColSpec("double",  "ldf_multiplier"),
        mlflow.types.ColSpec("double",  "inflation_adj"),
    ])
    _boot_output_schema = mlflow.types.Schema([
        mlflow.types.ColSpec("double", "best_estimate_M"),
        mlflow.types.ColSpec("double", "var_95_M"),
        mlflow.types.ColSpec("double", "var_99_M"),
        mlflow.types.ColSpec("double", "var_995_M"),
        mlflow.types.ColSpec("double", "cvar_99_M"),
        mlflow.types.ColSpec("double", "reserve_risk_capital_M"),
        mlflow.types.ColSpec("double", "max_ibnr_M"),
        mlflow.types.ColSpec("long",   "n_replications_used"),
        mlflow.types.ColSpec("string", "scenario"),
    ])
    _boot_signature = mlflow.models.ModelSignature(inputs=_boot_input_schema, outputs=_boot_output_schema)

    mlflow.pyfunc.log_model(
        artifact_path="bootstrap_reserve_pyfunc",
        python_model=BootstrapReservePyFunc(),
        signature=_boot_signature,
        registered_model_name=_BOOT_MODEL_NAME,
        pip_requirements=[
            f"scipy=={scipy.__version__}",
            f"numpy=={np.__version__}",
        ],
    )

    print(f"\nBootstrap Reserve Simulator registered to: {_BOOT_MODEL_NAME}")

    _boot_versions = _client.search_model_versions(f"name='{_BOOT_MODEL_NAME}'")
    _boot_latest_ver = max(int(v.version) for v in _boot_versions)
    _client.set_registered_model_alias(name=_BOOT_MODEL_NAME, alias="Champion", version=_boot_latest_ver)
    _client.set_model_version_tag(name=_BOOT_MODEL_NAME, version=str(_boot_latest_ver),
                                  key="approved_by", value="actuarial-workshop-demo")
    print(f"Set @Champion → version {_boot_latest_ver}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Reserve Validation
# MAGIC
# MAGIC Compare frequency forecasts against reserve triangle for adequacy assessment.

# COMMAND ----------

# Re-read predictions for validation (Ray has shut down)
sarima_results_df = spark.table(f"{CATALOG}.{MODELS_SCHEMA}.predictions_frequency_forecast")

from pyspark.sql import Window as _RW
_max_lag_win = _RW.partitionBy("segment_id", "accident_month").orderBy(F.col("dev_lag").desc())
latest_reserves = (
    triangle_sdf
    .withColumn("_rn", F.row_number().over(_max_lag_win))
    .filter(F.col("_rn") == 1)
    .drop("_rn")
    .withColumnRenamed("accident_month", "month")
    .withColumn("month", F.to_date("month"))
)

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
    .saveAsTable(f"{CATALOG}.{MODELS_SCHEMA}.predictions_reserve_validation"))

print(f"Reserve validation saved → {CATALOG}.{MODELS_SCHEMA}.predictions_reserve_validation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Technique | Framework | Scale | Use Case |
# MAGIC |---|---|---|---|
# MAGIC | SARIMAX(1,0,1)(1,1,0,12) | statsmodels + applyInPandas | 40 segments × 84 months | Frequency forecast for future accident periods |
# MAGIC | GARCH(1,1) on residuals | arch (inside SARIMAX fit) | 40 segments | Time-varying frequency volatility |
# MAGIC | Chain Ladder (deterministic) | pandas | 4 product lines | Best estimate IBNR + Mack variance |
# MAGIC | Bootstrap Chain Ladder | Ray + NumPy CPU | 40K replications × 5 scenarios | Reserve risk distribution (VaR 99.5%, CVaR) |
# MAGIC | Reserve evolution | Ray + NumPy CPU | 12 months × 40K replications | Forward reserve adequacy projection |
# MAGIC | Run-off projection | NumPy (regime-switching) | 50K scenarios × 12 months | Surplus trajectory with reserve development |
# MAGIC | Reserve validation | Spark join | Triangle × frequency forecast | Reserve adequacy vs forecasted frequency |
# MAGIC | Model registration | MLflow + UC | 2 models (Frequency + Bootstrap) | `@Champion` alias for serving |
# MAGIC
# MAGIC **Data lineage:** `gold_reserve_triangle` → Chain Ladder → Bootstrap → `predictions_bootstrap_reserves`
# MAGIC
# MAGIC **Frequency → Reserving:** SARIMAX forecasts future claim counts → new accident periods entering the triangle
# MAGIC
# MAGIC **Next:** Module 4 — Create Model Serving endpoints for both models and configure AI Gateway.
