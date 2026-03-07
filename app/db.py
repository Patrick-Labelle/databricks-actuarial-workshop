import streamlit as st
import pandas as pd

from auth import get_workspace_client, get_auth_init_error
from config import WAREHOUSE_ID, CATALOG, DATA_SCHEMA, MODELS_SCHEMA, APP_SCHEMA


def execute_sql(statement: str) -> pd.DataFrame:
    """Execute SQL via the Databricks SDK StatementExecution — handles auth automatically."""
    w = get_workspace_client()
    if w is None:
        st.error(f"Databricks SDK unavailable: {get_auth_init_error() or 'unknown error'}")
        return pd.DataFrame()
    try:
        result = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=statement,
            wait_timeout="30s",
        )
        state = result.status.state.value if result.status and result.status.state else "UNKNOWN"
        if state == "SUCCEEDED":
            columns = []
            if result.manifest and result.manifest.schema and result.manifest.schema.columns:
                columns = [c.name for c in result.manifest.schema.columns]
            rows = []
            if result.result and result.result.data_array:
                rows = result.result.data_array
            if columns and rows:
                return pd.DataFrame(rows, columns=columns)
            return pd.DataFrame()
        else:
            error_msg = "Unknown SQL error"
            if result.status and result.status.error:
                error_msg = result.status.error.message or error_msg
            st.error(f"SQL Error ({state}): {error_msg}")
            return pd.DataFrame()
    except Exception as exc:
        st.error(f"SQL execution failed: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_segments():
    df = execute_sql(f"SELECT DISTINCT segment_id FROM {CATALOG}.{APP_SCHEMA}.predictions_frequency_forecast ORDER BY 1")
    if not df.empty:
        return df["segment_id"].tolist()
    return []


@st.cache_data(ttl=300)
def load_forecasts(segment_id: str):
    return execute_sql(f"""
        SELECT month, record_type, claims_count, forecast_mean, forecast_lo95, forecast_hi95
        FROM {CATALOG}.{APP_SCHEMA}.predictions_frequency_forecast
        WHERE segment_id = '{segment_id}'
        ORDER BY month
    """)


@st.cache_data(ttl=600)
def load_bootstrap_summary():
    """Load bootstrap reserve distribution summary."""
    df = execute_sql(f"""
        SELECT
            AVG(best_estimate_M)        AS best_estimate,
            AVG(var_99_M)               AS var_99,
            AVG(var_995_M)              AS var_995,
            AVG(cvar_99_M)              AS cvar_99,
            AVG(reserve_risk_capital_M) AS reserve_risk_capital,
            MAX(max_ibnr_M)             AS max_ibnr
        FROM {CATALOG}.{APP_SCHEMA}.predictions_bootstrap_reserves
    """)
    if not df.empty:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.iloc[0]
    return pd.Series({
        'best_estimate': 0, 'var_99': 0, 'var_995': 0,
        'cvar_99': 0, 'reserve_risk_capital': 0, 'max_ibnr': 0,
    })


@st.cache_data(ttl=600)
def load_segment_stats(segment_id: str):
    """Summary statistics for the selected segment's history."""
    return execute_sql(f"""
        SELECT
            MIN(month)                          AS first_month,
            MAX(month)                          AS last_month,
            COUNT(*)                            AS num_months,
            ROUND(AVG(claims_count), 1)         AS avg_monthly_claims,
            ROUND(STDDEV(claims_count), 1)      AS stddev_claims,
            MIN(claims_count)                   AS min_claims,
            MAX(claims_count)                   AS max_claims
        FROM {CATALOG}.{APP_SCHEMA}.gold_claims_monthly
        WHERE segment_id = '{segment_id}'
    """)


@st.cache_data(ttl=600)
def load_reserve_scenarios():
    """Load pre-computed reserve deterioration scenarios."""
    df = execute_sql(f"""
        SELECT scenario_label, best_estimate_M, var_99_M, var_995_M, cvar_99_M, var_995_vs_baseline
        FROM {CATALOG}.{APP_SCHEMA}.predictions_reserve_scenarios
        ORDER BY var_995_M DESC
    """)
    if not df.empty:
        for col in ['best_estimate_M', 'var_99_M', 'var_995_M', 'cvar_99_M', 'var_995_vs_baseline']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_reserve_evolution():
    """Load 12-month reserve adequacy evolution."""
    df = execute_sql(f"""
        SELECT forecast_month, month_idx, best_estimate_M, var_99_M, var_995_M,
               cvar_99_M, reserve_risk_capital_M, var_995_vs_baseline
        FROM {CATALOG}.{APP_SCHEMA}.predictions_reserve_evolution
        ORDER BY month_idx
    """)
    if not df.empty:
        for col in ['best_estimate_M', 'var_99_M', 'var_995_M', 'cvar_99_M',
                     'reserve_risk_capital_M', 'var_995_vs_baseline', 'month_idx']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_garch_volatility(segment_id: str):
    """Load GARCH conditional volatility from frequency forecast residuals."""
    df = execute_sql(f"""
        SELECT month, record_type, cond_volatility, arch_lm_pvalue, garch_alpha, garch_beta
        FROM {CATALOG}.{APP_SCHEMA}.predictions_frequency_forecast
        WHERE segment_id = '{segment_id}'
          AND cond_volatility IS NOT NULL
        ORDER BY month
    """)
    if not df.empty:
        for col in ['cond_volatility', 'arch_lm_pvalue', 'garch_alpha', 'garch_beta']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_reserve_triangle():
    """Load the loss development triangle from the declarative pipeline."""
    df = execute_sql(f"""
        SELECT segment_id, product_line, region, accident_month, dev_lag,
               cumulative_paid, cumulative_incurred, case_reserve
        FROM {CATALOG}.{APP_SCHEMA}.gold_reserve_triangle
        ORDER BY segment_id, accident_month, dev_lag
    """)
    if not df.empty:
        for col in ['cumulative_paid', 'cumulative_incurred', 'case_reserve', 'dev_lag']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_runoff_projection():
    """Load run-off surplus trajectory."""
    df = execute_sql(f"""
        SELECT month, surplus_p05, surplus_p25, surplus_p50, surplus_p75, surplus_p95, ruin_probability
        FROM {CATALOG}.{APP_SCHEMA}.predictions_runoff_projection
        ORDER BY month
    """)
    if not df.empty:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_mct_components():
    """Load data for simplified Canadian MCT ratio calculation."""
    reserves_df = execute_sql(f"""
        WITH latest AS (
            SELECT product_line, accident_month, case_reserve, cumulative_incurred,
                   ROW_NUMBER() OVER (
                       PARTITION BY product_line, accident_month
                       ORDER BY dev_lag DESC
                   ) AS rn
            FROM {CATALOG}.{APP_SCHEMA}.gold_reserve_triangle
        )
        SELECT
            product_line,
            SUM(case_reserve)        AS outstanding_reserves,
            SUM(cumulative_incurred) AS total_incurred
        FROM latest
        WHERE rn = 1
        GROUP BY product_line
    """)
    if not reserves_df.empty:
        for col in ['outstanding_reserves', 'total_incurred']:
            reserves_df[col] = pd.to_numeric(reserves_df[col], errors='coerce')

    premium_df = execute_sql(f"""
        SELECT
            product_line,
            AVG(monthly_premium) * 12 AS annual_earned_premium
        FROM (
            SELECT product_line, month, SUM(earned_premium) AS monthly_premium
            FROM {CATALOG}.{APP_SCHEMA}.gold_claims_monthly
            WHERE month >= ADD_MONTHS(
                (SELECT MAX(month) FROM {CATALOG}.{APP_SCHEMA}.gold_claims_monthly), -11
            )
            GROUP BY product_line, month
        )
        GROUP BY product_line
    """)
    if not premium_df.empty:
        premium_df['annual_earned_premium'] = pd.to_numeric(
            premium_df['annual_earned_premium'], errors='coerce'
        )

    return reserves_df, premium_df


@st.cache_data(ttl=600)
def load_ldf_volatility():
    """Load development factor volatility per product line."""
    df = execute_sql(f"""
        SELECT product_line, avg_ldf, std_ldf, n_factors
        FROM {CATALOG}.{APP_SCHEMA}.predictions_ldf_volatility
    """)
    if not df.empty:
        for col in ['avg_ldf', 'std_ldf', 'n_factors']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
