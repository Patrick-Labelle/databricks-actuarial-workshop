import streamlit as st
import pandas as pd

from auth import get_workspace_client, get_auth_init_error
from config import WAREHOUSE_ID, CATALOG, DATA_SCHEMA, MODELS_SCHEMA, APP_SCHEMA


def _safe_sql(statement: str):
    """Execute SQL silently — returns DataFrame or None on failure (no st.error).

    Use inside @st.cache_data functions to avoid widget tree pollution from
    cached st.error() calls, which cause 'Bad setIn index' reconnection loops.
    """
    w = get_workspace_client()
    if w is None:
        return None
    try:
        result = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=statement,
            wait_timeout="30s",
        )
        state = result.status.state.value if result.status and result.status.state else "UNKNOWN"
        if state != "SUCCEEDED":
            return None
        columns = []
        if result.manifest and result.manifest.schema and result.manifest.schema.columns:
            columns = [c.name for c in result.manifest.schema.columns]
        rows = []
        if result.result and result.result.data_array:
            rows = result.result.data_array
        if columns and rows:
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame()
    except Exception:
        return None


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
    df = _safe_sql(f"SELECT DISTINCT segment_id FROM {CATALOG}.{APP_SCHEMA}.predictions_frequency_forecast ORDER BY 1")
    if df is not None and not df.empty:
        return df["segment_id"].tolist()
    return []


@st.cache_data(ttl=300)
def load_forecasts(segment_id: str):
    df = _safe_sql(f"""
        SELECT month, record_type, claims_count, forecast_mean, forecast_lo95, forecast_hi95
        FROM {CATALOG}.{APP_SCHEMA}.predictions_frequency_forecast
        WHERE segment_id = '{segment_id}'
        ORDER BY month
    """)
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=600)
def load_bootstrap_summary():
    """Load bootstrap reserve distribution summary."""
    df = _safe_sql(f"""
        SELECT
            AVG(best_estimate_M)        AS best_estimate,
            AVG(var_99_M)               AS var_99,
            AVG(var_995_M)              AS var_995,
            AVG(cvar_99_M)              AS cvar_99,
            AVG(reserve_risk_capital_M) AS reserve_risk_capital,
            MAX(max_ibnr_M)             AS max_ibnr
        FROM {CATALOG}.{APP_SCHEMA}.predictions_bootstrap_reserves
    """)
    if df is not None and not df.empty:
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
    df = _safe_sql(f"""
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
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=600)
def load_reserve_scenarios():
    """Load pre-computed reserve deterioration scenarios."""
    df = _safe_sql(f"""
        SELECT scenario_label, best_estimate_M, var_99_M, var_995_M, cvar_99_M, var_995_vs_baseline
        FROM {CATALOG}.{APP_SCHEMA}.predictions_reserve_scenarios
        ORDER BY var_995_M DESC
    """)
    if df is None:
        return pd.DataFrame()
    if not df.empty:
        for col in ['best_estimate_M', 'var_99_M', 'var_995_M', 'cvar_99_M', 'var_995_vs_baseline']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_reserve_evolution():
    """Load 12-month reserve adequacy evolution."""
    df = _safe_sql(f"""
        SELECT forecast_month, month_idx, best_estimate_M, var_99_M, var_995_M,
               cvar_99_M, reserve_risk_capital_M, var_995_vs_baseline
        FROM {CATALOG}.{APP_SCHEMA}.predictions_reserve_evolution
        ORDER BY month_idx
    """)
    if df is None:
        return pd.DataFrame()
    if not df.empty:
        for col in ['best_estimate_M', 'var_99_M', 'var_995_M', 'cvar_99_M',
                     'reserve_risk_capital_M', 'var_995_vs_baseline', 'month_idx']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_garch_volatility(segment_id: str):
    """Load GARCH conditional volatility from frequency forecast residuals."""
    df = _safe_sql(f"""
        SELECT month, record_type, cond_volatility, arch_lm_pvalue, garch_alpha, garch_beta
        FROM {CATALOG}.{APP_SCHEMA}.predictions_frequency_forecast
        WHERE segment_id = '{segment_id}'
          AND cond_volatility IS NOT NULL
        ORDER BY month
    """)
    if df is None:
        return pd.DataFrame()
    if not df.empty:
        for col in ['cond_volatility', 'arch_lm_pvalue', 'garch_alpha', 'garch_beta']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_reserve_triangle():
    """Load the loss development triangle from the declarative pipeline."""
    df = _safe_sql(f"""
        SELECT segment_id, product_line, region, accident_month, dev_lag,
               cumulative_paid, cumulative_incurred, case_reserve
        FROM {CATALOG}.{APP_SCHEMA}.gold_reserve_triangle
        ORDER BY segment_id, accident_month, dev_lag
    """)
    if df is None:
        return pd.DataFrame()
    if not df.empty:
        for col in ['cumulative_paid', 'cumulative_incurred', 'case_reserve', 'dev_lag']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_runoff_projection():
    """Load run-off surplus trajectory."""
    df = _safe_sql(f"""
        SELECT month, surplus_p05, surplus_p25, surplus_p50, surplus_p75, surplus_p95, ruin_probability
        FROM {CATALOG}.{APP_SCHEMA}.predictions_runoff_projection
        ORDER BY month
    """)
    if df is None:
        return pd.DataFrame()
    if not df.empty:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_mct_components():
    """Load data for simplified Canadian MCT ratio calculation."""
    reserves_df = _safe_sql(f"""
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
    if reserves_df is None:
        reserves_df = pd.DataFrame()
    if not reserves_df.empty:
        for col in ['outstanding_reserves', 'total_incurred']:
            reserves_df[col] = pd.to_numeric(reserves_df[col], errors='coerce')

    premium_df = _safe_sql(f"""
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
    if premium_df is None:
        premium_df = pd.DataFrame()
    if not premium_df.empty:
        premium_df['annual_earned_premium'] = pd.to_numeric(
            premium_df['annual_earned_premium'], errors='coerce'
        )

    return reserves_df, premium_df


@st.cache_data(ttl=600)
def load_regional_summary(start_date: str = None, end_date: str = None):
    """Load claims summary aggregated by region, optionally filtered by date."""
    where = ""
    if start_date and end_date:
        where = f"WHERE month >= '{start_date}' AND month <= '{end_date}'"
    df = _safe_sql(f"""
        SELECT region,
               SUM(claims_count)              AS total_claims,
               ROUND(AVG(claims_count), 1)    AS avg_monthly_claims,
               ROUND(SUM(total_incurred), 2)  AS total_incurred,
               ROUND(AVG(avg_severity), 2)    AS avg_severity,
               ROUND(SUM(earned_premium), 2)  AS total_premium
        FROM {CATALOG}.{APP_SCHEMA}.gold_claims_monthly
        {where}
        GROUP BY region
        ORDER BY total_claims DESC
    """)
    if df is None:
        return pd.DataFrame()
    if not df.empty:
        for col in ['total_claims', 'avg_monthly_claims', 'total_incurred',
                     'avg_severity', 'total_premium']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_regional_product_breakdown(start_date: str = None, end_date: str = None):
    """Load claims breakdown by region and product line, optionally filtered by date."""
    where = ""
    if start_date and end_date:
        where = f"WHERE month >= '{start_date}' AND month <= '{end_date}'"
    df = _safe_sql(f"""
        SELECT region, product_line,
               SUM(claims_count)              AS total_claims,
               ROUND(AVG(claims_count), 1)    AS avg_monthly_claims,
               ROUND(SUM(total_incurred), 2)  AS total_incurred,
               ROUND(AVG(avg_severity), 2)    AS avg_severity,
               ROUND(SUM(earned_premium), 2)  AS total_premium
        FROM {CATALOG}.{APP_SCHEMA}.gold_claims_monthly
        {where}
        GROUP BY region, product_line
        ORDER BY region, product_line
    """)
    if df is None:
        return pd.DataFrame()
    if not df.empty:
        for col in ['total_claims', 'avg_monthly_claims', 'total_incurred',
                     'avg_severity', 'total_premium']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_ldf_volatility():
    """Load development factor volatility per product line."""
    df = _safe_sql(f"""
        SELECT product_line, avg_ldf, std_ldf, n_factors
        FROM {CATALOG}.{APP_SCHEMA}.predictions_ldf_volatility
    """)
    if df is None:
        return pd.DataFrame()
    if not df.empty:
        for col in ['avg_ldf', 'std_ldf', 'n_factors']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data(ttl=600)
def load_chain_ladder_params() -> dict:
    """Load per-line IBNR and CV from the LDF volatility table (chain ladder output).

    Returns a dict with keys like 'mean_ibnr_personal_auto_M', 'cv_personal_auto', etc.
    Falls back to deriving CV from std_ldf/avg_ldf if new columns aren't available yet.
    """
    # Try the full query first; fall back if ibnr_M/cv columns don't exist yet
    df = _safe_sql(f"""
        SELECT product_line, ibnr_M, cv
        FROM {CATALOG}.{APP_SCHEMA}.predictions_ldf_volatility
    """)
    if df is not None and not df.empty and 'cv' in df.columns:
        for col in ['ibnr_M', 'cv']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        # Derive CV from std_ldf / avg_ldf (always available)
        df = _safe_sql(f"""
            SELECT product_line, std_ldf / NULLIF(avg_ldf, 0) AS cv
            FROM {CATALOG}.{APP_SCHEMA}.predictions_ldf_volatility
        """)
        if df is None or df.empty:
            return {}
        df['cv'] = pd.to_numeric(df['cv'], errors='coerce')

    _pl_map = {
        'Personal_Auto': 'personal_auto',
        'Commercial_Auto': 'commercial_auto',
        'Homeowners': 'homeowners',
        'Commercial_Property': 'commercial_property',
    }
    params = {}
    for _, row in df.iterrows():
        key = _pl_map.get(row['product_line'])
        if key:
            if 'ibnr_M' in df.columns and pd.notna(row.get('ibnr_M')):
                params[f'mean_ibnr_{key}_M'] = float(row['ibnr_M'])
            params[f'cv_{key}'] = float(row['cv']) if pd.notna(row.get('cv')) else 0.15
    return params
