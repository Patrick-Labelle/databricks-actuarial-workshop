"""Agent tools for the actuarial chatbot.

Each tool is a plain function with type hints and a docstring.
The agent calls these via OpenAI-style tool calling.

Tools may return either:
  - A plain string (text for the LLM + displayed as markdown)
  - A tuple (text_for_llm, display_attachment) where display_attachment
    is a dict with rendering instructions for the chat UI.

Adding a new tool:
  1. Define a function with type hints and a Google-style docstring.
  2. Append it to the TOOLS list at the bottom of this file.
  That's it -- the agent auto-discovers tools from that list.
"""

import pandas as pd

from auth import get_workspace_client
from config import (
    CATALOG, DATA_SCHEMA, MODELS_SCHEMA, APP_SCHEMA, WAREHOUSE_ID,
    ENDPOINT_NAME, MC_ENDPOINT_NAME,
    GENIE_SPACE_ID,
)


# ── Data query tool (SQL via warehouse) ──────────────────────────────────────

# Tables organized by schema -- the chatbot can query across all schemas
_DATA_TABLES = {
    "gold_claims_monthly": "Historical monthly claims aggregated by segment. Columns: segment_id, product_line, region, month, claims_count, total_incurred, avg_severity, earned_premium.",
    "gold_reserve_triangle": "Loss development triangle. Columns: segment_id, product_line, region, accident_month, dev_lag, cumulative_paid, cumulative_incurred, case_reserve, incremental_paid, incremental_incurred.",
    "silver_reserves": "SCD Type 2 reserve development. Columns: reserve_id, segment_id, accident_month, dev_lag, paid_cumulative, incurred_cumulative, case_reserve, effective_date, end_date, is_current.",
    "silver_rolling_features": "Rolling statistical features per segment. Columns: segment_id, month, claims_count, rolling_mean_3m, rolling_std_3m, rolling_mean_6m, rolling_std_6m, yoy_change.",
    "features_segment_monthly": "Feature Store table with macro features. Columns: segment_id, month, claims_count, plus rolling and macro-economic features.",
}

_MODELS_TABLES = {
    "predictions_frequency_forecast": "Monthly SARIMAX+GARCH frequency forecasts per segment. Columns: segment_id, month, record_type (actual/forecast), claims_count, forecast_mean, forecast_lo95, forecast_hi95, cond_volatility, arch_lm_pvalue, garch_alpha, garch_beta.",
    "predictions_bootstrap_reserves": "Portfolio-level Bootstrap Chain Ladder reserve distribution. Columns: best_estimate_M, var_99_M, var_995_M, cvar_99_M, reserve_risk_capital_M, max_ibnr_M.",
    "predictions_reserve_scenarios": "Pre-computed reserve deterioration scenarios. Columns: scenario_label, best_estimate_M, var_99_M, var_995_M, cvar_99_M, var_995_vs_baseline.",
    "predictions_reserve_evolution": "12-month reserve adequacy evolution. Columns: forecast_month, month_idx, best_estimate_M, var_99_M, var_995_M, cvar_99_M, reserve_risk_capital_M, var_995_vs_baseline.",
    "predictions_runoff_projection": "Multi-period run-off surplus trajectory with regime-switching. Columns: month, surplus_p05, surplus_p25, surplus_p50, surplus_p75, surplus_p95, ruin_probability.",
    "predictions_ldf_volatility": "Development factor volatility per product line. Columns: product_line, avg_ldf, std_ldf, n_factors.",
    "predictions_reserve_validation": "Reserve adequacy validation. Columns: segment_id, accident_month, reserve_adequacy_ratio.",
}

# Build fully-qualified table map for the chatbot
AVAILABLE_TABLES = {}
for name, desc in _DATA_TABLES.items():
    AVAILABLE_TABLES[f"{CATALOG}.{DATA_SCHEMA}.{name}"] = desc
for name, desc in _MODELS_TABLES.items():
    AVAILABLE_TABLES[f"{CATALOG}.{MODELS_SCHEMA}.{name}"] = desc


def query_data(sql_query: str) -> str:
    """FALLBACK ONLY: Execute a read-only SQL query. Use ask_genie first for any data question -- only call this if ask_genie fails or returns no results.

    Args:
        sql_query: A SELECT query using fully qualified table names.
            Data tables (in {catalog}.{data_schema}):
            - gold_claims_monthly, gold_reserve_triangle, silver_reserves,
              silver_rolling_features, features_segment_monthly
            Model output tables (in {catalog}.{models_schema}):
            - predictions_frequency_forecast, predictions_bootstrap_reserves,
              predictions_reserve_scenarios, predictions_reserve_evolution,
              predictions_runoff_projection, predictions_ldf_volatility

    Returns:
        Query results as a formatted string table, or an error message.
    """
    w = get_workspace_client()
    if w is None:
        return "Error: Databricks SDK not available."

    # Safety: only allow SELECT
    stripped = sql_query.strip().upper()
    if not stripped.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."

    try:
        result = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=sql_query,
            wait_timeout="30s",
        )
        state = result.status.state.value if result.status and result.status.state else "UNKNOWN"
        if state != "SUCCEEDED":
            error_msg = "Unknown error"
            if result.status and result.status.error:
                error_msg = result.status.error.message or error_msg
            return f"SQL Error ({state}): {error_msg}"

        columns = []
        if result.manifest and result.manifest.schema and result.manifest.schema.columns:
            columns = [c.name for c in result.manifest.schema.columns]
        rows = []
        if result.result and result.result.data_array:
            rows = result.result.data_array

        if not columns or not rows:
            return "Query returned no results."

        df = pd.DataFrame(rows, columns=columns)
        text = df.head(50).to_markdown(index=False)
        if len(df) > 50:
            text = f"Showing first 50 of {len(df)} rows:\n\n{text}"

        # Return structured attachment for rich rendering
        return (text, {"type": "dataframe", "df": df.head(50)})
    except Exception as e:
        return f"SQL execution error: {e}"


# ── Frequency forecast tool ──────────────────────────────────────────────────

def run_frequency_forecast(horizon: int) -> str:
    """Generate an on-demand SARIMAX frequency forecast for the portfolio.

    Args:
        horizon: Number of months ahead to forecast (1-24).

    Returns:
        Forecast results with point estimates and 95% confidence intervals.
    """
    if horizon < 1 or horizon > 24:
        return "Error: horizon must be between 1 and 24 months."

    w = get_workspace_client()
    if w is None:
        return "Error: Databricks SDK not available."
    try:
        response = w.serving_endpoints.query(
            name=ENDPOINT_NAME,
            dataframe_records=[{"horizon": horizon}],
        )
        if response.predictions:
            df = pd.DataFrame(response.predictions)
            text = f"Frequency Forecast ({horizon} months):\n\n{df.to_markdown(index=False)}"
            return (text, {"type": "dataframe", "df": df, "chart": {
                "x": "month", "y": "forecast_mean",
                "lo": "forecast_lo95", "hi": "forecast_hi95",
                "title": f"{horizon}-Month Frequency Forecast",
            }})
        return "Endpoint returned no predictions."
    except Exception as e:
        return f"Frequency forecast endpoint error: {e}"


# ── Bootstrap reserve simulation tool ────────────────────────────────────────

def run_bootstrap_reserve(
    scenario: str = "baseline",
    ldf_multiplier: float = 1.0,
    inflation_adj: float = 0.0,
    cv_personal_auto: float = 0.15,
    cv_commercial_auto: float = 0.18,
    cv_homeowners: float = 0.12,
    cv_commercial_property: float = 0.20,
    n_replications: int = 50_000,
) -> str:
    """Run a Bootstrap Chain Ladder reserve simulation with custom parameters.

    Use this to answer "what if" questions about reserve adequacy under
    different reserve deterioration assumptions. Default values represent the
    baseline scenario.

    Supports scenarios:
    - "baseline" (default): Standard reserve development
    - "adverse_development": LDFs inflated -- reserves develop worse than expected
    - "judicial_inflation": Social inflation / nuclear verdicts on Auto lines
    - "pandemic_tail": Extended development periods due to delayed settlements
    - "superimposed_inflation": Calendar-year trend (CPI + X%) across all lines

    Args:
        scenario: Reserve scenario type (default: baseline).
        ldf_multiplier: LDF multiplier -- values above 1.0 inflate development factors (default: 1.0).
        inflation_adj: Calendar-year superimposed inflation rate (default: 0.0).
        cv_personal_auto: Reserve volatility for Personal Auto (default: 0.15).
        cv_commercial_auto: Reserve volatility for Commercial Auto (default: 0.18).
        cv_homeowners: Reserve volatility for Homeowners (default: 0.12).
        cv_commercial_property: Reserve volatility for Commercial Property (default: 0.20).
        n_replications: Number of bootstrap replications (default: 10000).

    Returns:
        Reserve risk metrics including Best Estimate IBNR, VaR 99%, Reserve Risk
        Capital (VaR 99.5%), CVaR 99%, and maximum simulated IBNR.
    """
    w = get_workspace_client()
    if w is None:
        return "Error: Databricks SDK not available."

    params = {
        "scenario": scenario,
        "ldf_multiplier": ldf_multiplier,
        "inflation_adj": inflation_adj,
        "cv_personal_auto": cv_personal_auto,
        "cv_commercial_auto": cv_commercial_auto,
        "cv_homeowners": cv_homeowners,
        "cv_commercial_property": cv_commercial_property,
        "n_replications": n_replications,
        "mean_ibnr_personal_auto_M": 7200.0,
        "mean_ibnr_commercial_auto_M": 4000.0,
        "mean_ibnr_homeowners_M": 5900.0,
        "mean_ibnr_commercial_property_M": 2700.0,
    }

    try:
        response = w.serving_endpoints.query(
            name=MC_ENDPOINT_NAME,
            dataframe_records=[params],
        )
        if response.predictions:
            p = response.predictions[0] if isinstance(response.predictions, list) else response.predictions
            lines = ["Bootstrap Reserve Simulation Results:", ""]
            metrics = []
            for k, v in p.items():
                if isinstance(v, (int, float)):
                    formatted = '${:.1f}B'.format(v / 1000) if abs(v) >= 1000 else '${:.2f}M'.format(v)
                    lines.append(f"- **{k}**: {formatted}")
                    metrics.append({"label": k, "value": v})
                else:
                    lines.append(f"- **{k}**: {v}")

            text = "\n".join(lines)
            return (text, {"type": "metrics", "items": metrics, "scenario": scenario})
        return "Endpoint returned no predictions."
    except Exception as e:
        return f"Bootstrap reserve endpoint error: {e}"


# ── Lakebase annotations tool ────────────────────────────────────────────────

def query_annotations(segment_id: str = "") -> str:
    """Query analyst annotations from the Lakebase database.

    Args:
        segment_id: Filter by segment (e.g. 'commercial_auto_ontario').
            Leave empty to get the most recent annotations across all segments.

    Returns:
        Recent annotations with analyst, type, adjustment, status, and notes.
    """
    from lakebase import get_lakebase_conn

    try:
        conn = get_lakebase_conn()
        cur = conn.cursor()
        if segment_id:
            cur.execute(
                "SELECT segment_id, analyst, scenario_type, adjustment_pct, "
                "approval_status, note, created_at "
                "FROM public.scenario_annotations "
                "WHERE segment_id = %s ORDER BY created_at DESC LIMIT 20",
                (segment_id,),
            )
        else:
            cur.execute(
                "SELECT segment_id, analyst, scenario_type, adjustment_pct, "
                "approval_status, note, created_at "
                "FROM public.scenario_annotations "
                "ORDER BY created_at DESC LIMIT 20",
            )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return "No annotations found."
        df = pd.DataFrame(
            rows,
            columns=["Segment", "Analyst", "Type", "Adj %", "Status", "Note", "Created"],
        )
        text = df.to_markdown(index=False)
        return (text, {"type": "dataframe", "df": df})
    except Exception as e:
        return f"Lakebase query error: {e}"


# ── Genie space tool (natural language -> SQL) ────────────────────────────────

def ask_genie(question: str) -> str:
    """Ask a natural-language question about the insurance portfolio data using the AI/BI Genie space.

    This queries the AI/BI Genie space which understands all workshop tables
    and can generate SQL automatically. Use this for data exploration questions --
    trend analysis, comparisons, aggregations, or "show me" questions.

    Args:
        question: A natural-language question about the data, e.g.
            "What are the top 5 segments by average claims?" or
            "Show claims trend by product line over the last year".

    Returns:
        The Genie response including any generated SQL and query results.
    """
    if not GENIE_SPACE_ID:
        return "Error: GENIE_SPACE_ID is not configured. The Genie space has not been created yet."

    w = get_workspace_client()
    if w is None:
        return "Error: Databricks SDK not available."

    try:
        msg = w.genie.start_conversation_and_wait(
            space_id=GENIE_SPACE_ID,
            content=question,
        )

        parts = []
        attachment_df = None

        # Extract text reply
        reply = getattr(msg, "reply", None)
        if reply:
            text = getattr(reply, "text", None)
            if text:
                parts.append(text)

        # Extract query results from attachments
        attachments = getattr(msg, "attachments", None) or []
        for att in attachments:
            att_id = getattr(att, "attachment_id", None) or getattr(att, "id", None)
            # Get the generated SQL if available
            query_obj = getattr(att, "query", None)
            if query_obj:
                sql_text = getattr(query_obj, "query", None) or getattr(query_obj, "sql", None)
                if sql_text:
                    parts.append(f"\nGenerated SQL:\n```sql\n{sql_text}\n```")

            # Get the query result data
            if att_id:
                try:
                    qr = w.genie.get_message_query_result(
                        space_id=GENIE_SPACE_ID,
                        conversation_id=msg.conversation_id,
                        message_id=msg.id,
                    )
                    stmt = getattr(qr, "statement_response", None)
                    if stmt:
                        manifest = getattr(stmt, "manifest", None)
                        result = getattr(stmt, "result", None)
                        if manifest and result:
                            schema = getattr(manifest, "schema", None)
                            cols_obj = getattr(schema, "columns", None) if schema else None
                            columns = [c.name for c in cols_obj] if cols_obj else []
                            data = getattr(result, "data_array", None) or []
                            if columns and data:
                                df = pd.DataFrame(data, columns=columns)
                                attachment_df = df.head(50)
                                if len(df) > 30:
                                    parts.append(f"\nShowing first 30 of {len(df)} rows:\n\n{df.head(30).to_markdown(index=False)}")
                                else:
                                    parts.append(f"\n{df.to_markdown(index=False)}")
                except Exception as e:
                    parts.append(f"\n(Could not fetch query results: {e})")

        if not parts:
            return "Genie processed the question but returned no content. Try rephrasing or use query_data as a fallback."

        text = "\n".join(parts)
        if attachment_df is not None:
            return (text, {"type": "dataframe", "df": attachment_df})
        return text
    except Exception as e:
        return f"Genie query error: {e}. You can try query_data as a fallback."


# ── Tool registry ────────────────────────────────────────────────────────────

TOOLS = [ask_genie, query_data, run_frequency_forecast, run_bootstrap_reserve, query_annotations]

TOOL_MAP = {fn.__name__: fn for fn in TOOLS}
