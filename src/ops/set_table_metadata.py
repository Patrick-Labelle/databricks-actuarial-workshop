# Databricks notebook source
# MAGIC %md
# MAGIC # Set UC Table & Column Descriptions
# MAGIC
# MAGIC Adds human-readable descriptions to all workshop tables and columns in Unity Catalog.
# MAGIC These descriptions are inherited by AI/BI Genie spaces, data exploration UIs, and
# MAGIC lineage tools. Run after all tables have been created (post Module 03).

# COMMAND ----------

dbutils.widgets.text("catalog", "", "UC Catalog")
dbutils.widgets.text("data_schema", "actuarial_data", "Data Schema")
dbutils.widgets.text("models_schema", "actuarial_models", "Models Schema")

CATALOG = dbutils.widgets.get("catalog")
DATA_SCHEMA = dbutils.widgets.get("data_schema")
MODELS_SCHEMA = dbutils.widgets.get("models_schema")

# COMMAND ----------

# ── Table and column metadata ───────────────────────────────────────────────
# Each entry: (table_name, table_comment, {column: comment})

TABLE_METADATA = [
    (
        "gold_claims_monthly",
        "Monthly claims aggregate by product line and province. Primary table for trend analysis, seasonality, SARIMAX modeling, and reserve calibration.",
        {
            "segment_id": "Unique segment identifier: product_line_region (e.g. personal_auto_ontario)",
            "product_line": "Insurance product: Personal_Auto, Commercial_Auto, Homeowners, Commercial_Property",
            "region": "Canadian province (e.g. Ontario, Quebec, British_Columbia)",
            "month": "Calendar month (first day of month)",
            "claims_count": "Total number of claims reported in this segment-month",
            "total_incurred": "Total incurred losses in dollars (paid + case reserves)",
            "avg_severity": "Average cost per claim (total_incurred / claims_count)",
            "earned_premium": "Premium earned for this segment-month (exposure-weighted)",
            "loss_ratio": "Loss ratio: total_incurred / earned_premium",
        },
    ),
    (
        "gold_reserve_triangle",
        "Loss development triangle by segment, accident month, and development lag. Core actuarial exhibit for chain ladder IBNR estimation and bootstrap reserve analysis.",
        {
            "segment_id": "Unique segment identifier: product_line_region",
            "product_line": "Insurance product line",
            "region": "Canadian province",
            "accident_month": "Month in which the loss event occurred",
            "dev_lag": "Development lag in months since the accident month (0 = first evaluation)",
            "cumulative_paid": "Cumulative paid losses at this development lag (dollars)",
            "cumulative_incurred": "Cumulative incurred losses at this development lag (paid + case reserve, dollars)",
            "case_reserve": "Outstanding case reserve at this development lag (dollars)",
            "incremental_paid": "Incremental paid losses for this development period (dollars)",
            "incremental_incurred": "Incremental incurred losses for this development period (dollars)",
        },
    ),
    (
        "predictions_frequency_forecast",
        "Per-segment SARIMAX+GARCH frequency forecasts. 52 segments x 84 months of history + 12-month forecast horizon. Use record_type to filter actuals vs forecasts.",
        {
            "segment_id": "Unique segment identifier: product_line_region",
            "month": "Calendar month (date)",
            "record_type": "Row type: 'actual' for historical observations, 'forecast' for model projections",
            "claims_count": "Observed claims count (actuals only; NULL for forecasts)",
            "forecast_mean": "SARIMAX point forecast for claims count",
            "forecast_lo95": "Lower bound of 95% prediction interval",
            "forecast_hi95": "Upper bound of 95% prediction interval",
            "aic": "Akaike Information Criterion of the fitted SARIMAX model (lower is better)",
            "mape": "Mean Absolute Percentage Error on in-sample fit",
            "mape_baseline": "MAPE of a naive seasonal baseline for comparison",
            "mape_sarimax": "MAPE of the SARIMAX model with exogenous macro features",
            "exog_vars": "Exogenous macro variables used in SARIMAX (comma-separated)",
            "cond_volatility": "GARCH(1,1) conditional standard deviation of SARIMAX residuals",
            "arch_lm_pvalue": "ARCH-LM test p-value; < 0.05 confirms heteroskedasticity in residuals",
            "garch_alpha": "GARCH alpha parameter: short-run shock persistence",
            "garch_beta": "GARCH beta parameter: long-run volatility persistence",
        },
    ),
    (
        "predictions_bootstrap_reserves",
        "Portfolio-level Bootstrap Chain Ladder reserve distribution. Contains IBNR best estimate, VaR, CVaR, and reserve risk capital metrics from bootstrap simulation.",
        {
            "best_estimate_M": "Best estimate IBNR: mean of bootstrap distribution in millions of dollars (actuarial central estimate)",
            "var_99_M": "Value at Risk at 99% confidence (1-in-100 year reserve level) in $M",
            "var_995_M": "Reserve Risk Capital: VaR at 99.5% (1-in-200 year reserve level) in $M. Equivalent to Solvency II reserve risk SCR threshold.",
            "cvar_99_M": "Conditional VaR (Tail Risk): average IBNR in the worst 1% of bootstrap replications in $M",
            "reserve_risk_capital_M": "Reserve risk capital = VaR 99.5% minus Best Estimate in $M",
            "max_ibnr_M": "Maximum simulated IBNR across all bootstrap replications in $M",
        },
    ),
    (
        "predictions_reserve_scenarios",
        "Pre-computed reserve deterioration scenarios. Each row is a scenario with IBNR metrics and percentage change vs baseline. Use var_995_vs_baseline for relative impact.",
        {
            "scenario": "Scenario identifier key (e.g. baseline, adverse_development, judicial_inflation)",
            "scenario_label": "Human-readable scenario name for display",
            "best_estimate_M": "Best estimate IBNR under this scenario in $M",
            "var_99_M": "VaR 99% (1-in-100 year reserve) under this scenario in $M",
            "var_995_M": "Reserve Risk Capital: VaR 99.5% (1-in-200 year reserve) under this scenario in $M",
            "cvar_99_M": "CVaR 99% (tail risk) under this scenario in $M",
            "var_995_vs_baseline": "Percentage change in Reserve Risk Capital (VaR 99.5%) vs the baseline scenario",
        },
    ),
    (
        "predictions_reserve_evolution",
        "12-month forward-looking reserve adequacy evolution. Each row projects how Reserve Risk Capital (VaR 99.5%) will change in a future month, driven by frequency forecasts.",
        {
            "forecast_month": "Projected calendar month (date string)",
            "month_idx": "Month offset from current (1 = next month, 12 = one year ahead)",
            "best_estimate_M": "Projected best estimate IBNR in $M",
            "var_99_M": "Projected VaR 99% (1-in-100 year reserve) in $M",
            "var_995_M": "Projected Reserve Risk Capital: VaR 99.5% (1-in-200 year reserve) in $M",
            "cvar_99_M": "Projected CVaR 99% (tail risk) in $M",
            "reserve_risk_capital_M": "Projected reserve risk capital (VaR 99.5% minus best estimate) in $M",
            "var_995_vs_baseline": "Percentage change in Reserve Risk Capital vs current baseline",
        },
    ),
    (
        "predictions_runoff_projection",
        "Multi-period run-off surplus trajectory from 50,000 scenarios with 2-state regime-switching (Normal/Crisis). Shows percentile bands of cumulative surplus over 12 months.",
        {
            "month": "Month in the projection horizon (1-12)",
            "surplus_p05": "5th percentile of surplus distribution in $M (near-worst case)",
            "surplus_p25": "25th percentile of surplus distribution in $M",
            "surplus_p50": "Median surplus in $M (50th percentile)",
            "surplus_p75": "75th percentile of surplus distribution in $M",
            "surplus_p95": "95th percentile of surplus distribution in $M (near-best case)",
            "ruin_probability": "Fraction of scenarios where surplus drops below zero by this month",
        },
    ),
    (
        "predictions_ldf_volatility",
        "Development factor volatility and chain ladder IBNR per product line from the reserve triangle. Used for reserve risk capital calculation and as input parameters for the bootstrap reserve simulator endpoint.",
        {
            "product_line": "Insurance product line (Personal_Auto, Commercial_Auto, Homeowners, Commercial_Property)",
            "avg_ldf": "Average link ratio (loss development factor) across all accident months and lag transitions",
            "std_ldf": "Standard deviation of link ratios — measures reserve development volatility",
            "cv": "Coefficient of variation of development factors (std_ldf / avg_ldf) — used as reserve volatility parameter",
            "ibnr_M": "Chain ladder best estimate IBNR for this product line in millions of dollars",
            "n_factors": "Number of development factor observations used in the calculation",
        },
    ),
    (
        "features_segment_monthly",
        "Feature-engineered table for ML models. Combines rolling statistical features with macro-economic indicators per segment. Registered in UC Feature Store.",
        {
            "segment_id": "Unique segment identifier: product_line_region",
            "month": "Calendar month (feature observation date)",
            "claims_count": "Raw claims count for this segment-month",
        },
    ),
    (
        "silver_reserves",
        "Reserve development history with SCD Type 2 change tracking. Filter is_current = true for the latest reserve position per segment.",
        {
            "reserve_id": "Unique identifier for the reserve record",
            "segment_id": "Unique segment identifier: product_line_region",
            "accident_month": "Month of the loss event",
            "dev_lag": "Development lag in months since accident",
            "paid_cumulative": "Cumulative paid losses in dollars",
            "incurred_cumulative": "Cumulative incurred losses (paid + case reserve) in dollars",
            "case_reserve": "Outstanding case reserve in dollars",
            "effective_date": "SCD2: date this version became effective",
            "end_date": "SCD2: date this version was superseded (NULL if current)",
            "is_current": "SCD2: true if this is the latest version of this record",
        },
    ),
    (
        "silver_rolling_features",
        "Rolling statistical features per segment for trend analysis and ML feature engineering. Includes 3-month and 6-month rolling windows with year-over-year comparisons.",
        {
            "segment_id": "Unique segment identifier: product_line_region",
            "month": "Calendar month",
            "claims_count": "Raw claims count for this segment-month",
            "rolling_mean_3m": "3-month rolling average of claims count",
            "rolling_std_3m": "3-month rolling standard deviation of claims count",
            "rolling_mean_6m": "6-month rolling average of claims count",
            "rolling_std_6m": "6-month rolling standard deviation of claims count",
            "yoy_change": "Year-over-year change in claims count (current vs 12 months ago)",
        },
    ),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply descriptions

# COMMAND ----------

# Route tables to the correct schema based on naming convention
_MODELS_PREFIXES = ("predictions_",)

for table_name, table_comment, columns in TABLE_METADATA:
    schema = MODELS_SCHEMA if table_name.startswith(_MODELS_PREFIXES) else DATA_SCHEMA
    fqn = f"`{CATALOG}`.`{schema}`.`{table_name}`"

    # Check table exists before setting comments
    try:
        spark.sql(f"DESCRIBE TABLE {fqn}").collect()
    except Exception:
        print(f"  [SKIP] {table_name} — table does not exist yet")
        continue

    # Set table comment (SDP streaming/materialized tables don't allow COMMENT ON TABLE)
    try:
        safe_comment = table_comment.replace("'", "\\'")
        spark.sql(f"COMMENT ON TABLE {fqn} IS '{safe_comment}'")
        print(f"  [OK] {table_name} — table comment set")
    except Exception as e:
        if "STREAMING_TABLE" in str(e) or "MATERIALIZED_VIEW" in str(e):
            print(f"  [SKIP] {table_name} — SDP-managed table (description set by pipeline)")
        else:
            print(f"  [WARN] {table_name} — table comment failed: {e}")
        continue  # skip column comments too for SDP tables

    # Set column comments
    for col_name, col_comment in columns.items():
        try:
            safe_col_comment = col_comment.replace("'", "\\'")
            spark.sql(f"ALTER TABLE {fqn} ALTER COLUMN `{col_name}` COMMENT '{safe_col_comment}'")
        except Exception as e:
            # Column may not exist (e.g. features table has dynamic columns)
            print(f"    [SKIP] {table_name}.{col_name} — {e}")

    print(f"    {len(columns)} column descriptions applied")

# COMMAND ----------

print("Table metadata setup complete.")
