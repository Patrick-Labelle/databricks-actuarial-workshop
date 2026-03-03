import streamlit as st
import pandas as pd

from auth import get_workspace_client, get_auth_init_error
from config import WAREHOUSE_ID, CATALOG, SCHEMA


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
    df = execute_sql(f"SELECT DISTINCT segment_id FROM {CATALOG}.{SCHEMA}.sarima_forecasts ORDER BY 1")
    if not df.empty:
        return df["segment_id"].tolist()
    return []


@st.cache_data(ttl=300)
def load_forecasts(segment_id: str):
    return execute_sql(f"""
        SELECT month, record_type, claims_count, forecast_mean, forecast_lo95, forecast_hi95
        FROM {CATALOG}.{SCHEMA}.sarima_forecasts
        WHERE segment_id = '{segment_id}'
        ORDER BY month
    """)


@st.cache_data(ttl=600)
def load_monte_carlo_summary():
    df = execute_sql(f"""
        SELECT
            AVG(mean_loss_M)  AS expected_loss,
            AVG(var_99_M)     AS var_99,
            AVG(var_995_M)    AS var_995,
            AVG(cvar_99_M)    AS cvar_99,
            MAX(max_loss_M)   AS max_loss
        FROM {CATALOG}.{SCHEMA}.monte_carlo_results
    """)
    if not df.empty:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.iloc[0]
    return pd.Series({
        'expected_loss': 0, 'var_99': 0, 'var_995': 0, 'cvar_99': 0, 'max_loss': 0
    })


@st.cache_data(ttl=600)
def load_monte_carlo_distribution():
    """Load per-simulation total loss for portfolio loss distribution chart."""
    return execute_sql(f"""
        SELECT mean_loss_M, var_99_M, var_995_M, cvar_99_M, max_loss_M
        FROM {CATALOG}.{SCHEMA}.monte_carlo_results
        ORDER BY mean_loss_M
    """)


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
        FROM {CATALOG}.{SCHEMA}.gold_claims_monthly
        WHERE segment_id = '{segment_id}'
    """)


@st.cache_data(ttl=600)
def load_stress_scenarios():
    """Load pre-computed stress test scenario comparison from Module 4."""
    df = execute_sql(f"""
        SELECT scenario_label, total_mean_M, var_99_M, var_995_M, cvar_99_M, var_995_vs_baseline
        FROM {CATALOG}.{SCHEMA}.stress_test_scenarios
        ORDER BY var_995_M DESC
    """)
    if not df.empty:
        for col in ['total_mean_M', 'var_99_M', 'var_995_M', 'cvar_99_M', 'var_995_vs_baseline']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_var_timeline():
    """Load the 12-month SARIMA-driven VaR evolution from Module 4."""
    df = execute_sql(f"""
        SELECT forecast_month, month_idx, total_mean_M, var_99_M, var_995_M, cvar_99_M, var_995_vs_baseline
        FROM {CATALOG}.{SCHEMA}.portfolio_risk_timeline
        ORDER BY month_idx
    """)
    if not df.empty:
        for col in ['total_mean_M', 'var_99_M', 'var_995_M', 'cvar_99_M', 'var_995_vs_baseline', 'month_idx']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_garch_volatility(segment_id: str):
    """Load GARCH conditional volatility (actuals + forecasts) from SARIMA residuals for a segment."""
    df = execute_sql(f"""
        SELECT month, record_type, cond_volatility, arch_lm_pvalue, garch_alpha, garch_beta
        FROM {CATALOG}.{SCHEMA}.sarima_forecasts
        WHERE segment_id = '{segment_id}'
          AND cond_volatility IS NOT NULL
        ORDER BY month
    """)
    if not df.empty:
        for col in ['cond_volatility', 'arch_lm_pvalue', 'garch_alpha', 'garch_beta']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_regional_forecast():
    """Load regional claims forecast from Module 4."""
    df = execute_sql(f"""
        SELECT region, forecast_month, total_forecast_claims
        FROM {CATALOG}.{SCHEMA}.regional_claims_forecast
        ORDER BY region, forecast_month
    """)
    if not df.empty:
        df['total_forecast_claims'] = pd.to_numeric(df['total_forecast_claims'], errors='coerce')
    return df


def load_reserve_triangle():
    """Load the loss development triangle from the DLT pipeline."""
    df = execute_sql(f"""
        SELECT segment_id, product_line, region, accident_month, dev_lag,
               cumulative_paid, cumulative_incurred, case_reserve
        FROM {CATALOG}.{SCHEMA}.gold_reserve_triangle
        ORDER BY segment_id, accident_month, dev_lag
    """)
    if not df.empty:
        for col in ['cumulative_paid', 'cumulative_incurred', 'case_reserve', 'dev_lag']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
