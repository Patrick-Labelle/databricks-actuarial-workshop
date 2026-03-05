"""Agent tools for the actuarial chatbot.

Each tool is a plain function with type hints and a docstring.
The agent calls these via OpenAI-style tool calling.

Adding a new tool:
  1. Define a function with type hints and a Google-style docstring.
  2. Append it to the TOOLS list at the bottom of this file.
  That's it — the agent auto-discovers tools from that list.
"""

import json
import time
import pandas as pd

from auth import get_workspace_client
from config import (
    CATALOG, SCHEMA, WAREHOUSE_ID,
    ENDPOINT_NAME, MC_ENDPOINT_NAME,
    GENIE_SPACE_ID,
)


# ── Data query tool (SQL via warehouse) ──────────────────────────────────────

AVAILABLE_TABLES = {
    "sarima_forecasts": "Monthly SARIMA+GARCH forecasts per segment. Columns: segment_id, month, record_type (actual/forecast), claims_count, forecast_mean, forecast_lo95, forecast_hi95, cond_volatility, arch_lm_pvalue, garch_alpha, garch_beta.",
    "gold_claims_monthly": "Historical monthly claims aggregated by segment. Columns: segment_id, product_line, region, month, claims_count, total_incurred, avg_severity.",
    "monte_carlo_results": "Portfolio-level Monte Carlo simulation results. Columns: mean_loss_M, var_99_M, var_995_M, cvar_99_M, max_loss_M.",
    "stress_test_scenarios": "Pre-computed stress test comparisons. Columns: scenario_label, total_mean_M, var_99_M, var_995_M, cvar_99_M, var_995_vs_baseline.",
    "portfolio_risk_timeline": "12-month SARIMA-driven VaR evolution. Columns: forecast_month, month_idx, total_mean_M, var_99_M, var_995_M, cvar_99_M, var_995_vs_baseline.",
    "regional_claims_forecast": "Regional claims forecast from Module 4. Columns: region, forecast_month, total_forecast_claims.",
    "gold_reserve_triangle": "Loss development triangle. Columns: segment_id, product_line, region, accident_month, dev_lag, cumulative_paid, cumulative_incurred, case_reserve.",
    "silver_reserves": "SCD Type 2 reserve development. Columns: reserve_id, segment_id, accident_month, dev_lag, paid_cumulative, incurred_cumulative, case_reserve, effective_date, end_date, is_current.",
    "silver_rolling_features": "Rolling statistical features per segment. Columns: segment_id, month, claims_count, rolling_mean_3m, rolling_std_3m, rolling_mean_6m, rolling_std_6m, yoy_change.",
    "segment_monthly_features": "Feature Store table with macro features. Columns: segment_id, month, claims_count, plus rolling and macro-economic features.",
}


def query_data(sql_query: str) -> str:
    """Execute a read-only SQL query against the actuarial workshop tables.

    Args:
        sql_query: A SELECT query against the actuarial workshop tables.
            Available tables (all in {catalog}.{schema}):
            - sarima_forecasts: Monthly SARIMA+GARCH forecasts per segment
            - gold_claims_monthly: Historical monthly claims by segment
            - monte_carlo_results: Portfolio Monte Carlo simulation results
            - stress_test_scenarios: Pre-computed stress tests
            - portfolio_risk_timeline: 12-month VaR evolution
            - regional_claims_forecast: Regional forecast data
            - gold_reserve_triangle: Loss development triangle
            - silver_reserves: SCD2 reserve development
            - silver_rolling_features: Rolling stats per segment
            - segment_monthly_features: Feature Store with macro features

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
        # Limit output size
        if len(df) > 50:
            return f"Showing first 50 of {len(df)} rows:\n\n{df.head(50).to_markdown(index=False)}"
        return df.to_markdown(index=False)
    except Exception as e:
        return f"SQL execution error: {e}"


# ── SARIMA forecast tool ─────────────────────────────────────────────────────

def run_sarima_forecast(horizon: int) -> str:
    """Generate an on-demand SARIMA claims forecast for the portfolio.

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
            return f"SARIMA Forecast ({horizon} months):\n\n{df.to_markdown(index=False)}"
        return "Endpoint returned no predictions."
    except Exception as e:
        return f"SARIMA endpoint error: {e}"


# ── Monte Carlo simulation tool ──────────────────────────────────────────────

def run_monte_carlo(
    mean_property_M: float = 12.5,
    mean_auto_M: float = 8.3,
    mean_liability_M: float = 5.7,
    cv_property: float = 0.35,
    cv_auto: float = 0.28,
    cv_liability: float = 0.42,
    corr_prop_auto: float = 0.40,
    corr_prop_liab: float = 0.20,
    corr_auto_liab: float = 0.30,
    n_scenarios: int = 10_000,
    copula_df: int = 4,
) -> str:
    """Run a Monte Carlo portfolio loss simulation with custom parameters.

    Use this to answer "what if" questions about capital requirements under
    different assumptions. Default values represent the baseline scenario.

    Args:
        mean_property_M: Expected annual property loss in $M (default: 12.5).
        mean_auto_M: Expected annual auto loss in $M (default: 8.3).
        mean_liability_M: Expected annual liability loss in $M (default: 5.7).
        cv_property: Coefficient of variation for property (default: 0.35).
        cv_auto: Coefficient of variation for auto (default: 0.28).
        cv_liability: Coefficient of variation for liability (default: 0.42).
        corr_prop_auto: Correlation between property and auto (default: 0.40).
        corr_prop_liab: Correlation between property and liability (default: 0.20).
        corr_auto_liab: Correlation between auto and liability (default: 0.30).
        n_scenarios: Number of simulation paths (default: 10000).
        copula_df: Degrees of freedom for t-copula (default: 4).

    Returns:
        Risk metrics including Expected Loss, VaR 99%, SCR (VaR 99.5%),
        CVaR 99%, and maximum simulated loss.
    """
    w = get_workspace_client()
    if w is None:
        return "Error: Databricks SDK not available."

    scenario = {
        "mean_property_M": mean_property_M,
        "mean_auto_M": mean_auto_M,
        "mean_liability_M": mean_liability_M,
        "cv_property": cv_property,
        "cv_auto": cv_auto,
        "cv_liability": cv_liability,
        "corr_prop_auto": corr_prop_auto,
        "corr_prop_liab": corr_prop_liab,
        "corr_auto_liab": corr_auto_liab,
        "n_scenarios": n_scenarios,
        "copula_df": copula_df,
    }

    try:
        response = w.serving_endpoints.query(
            name=MC_ENDPOINT_NAME,
            dataframe_records=[scenario],
        )
        if response.predictions:
            p = response.predictions[0] if isinstance(response.predictions, list) else response.predictions
            lines = ["Monte Carlo Simulation Results:", ""]
            for k, v in p.items():
                if isinstance(v, (int, float)):
                    lines.append(f"- **{k}**: ${v:.2f}M")
                else:
                    lines.append(f"- **{k}**: {v}")
            return "\n".join(lines)
        return "Endpoint returned no predictions."
    except Exception as e:
        return f"Monte Carlo endpoint error: {e}"


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
        return df.to_markdown(index=False)
    except Exception as e:
        return f"Lakebase query error: {e}"


# ── Genie space tool (natural language → SQL) ────────────────────────────────

def ask_genie(question: str) -> str:
    """Ask a natural-language question about the insurance portfolio data.

    This queries the AI/BI Genie space which understands all workshop tables
    and can generate SQL automatically. Use this for data exploration questions
    where you don't want to write SQL manually — for example, trend analysis,
    comparisons, aggregations, or "show me" questions.

    For precise queries where you already know the SQL, prefer query_data instead.

    Args:
        question: A natural-language question about the data, e.g.
            "What are the top 5 segments by average claims?" or
            "Show claims trend by product line over the last year".

    Returns:
        The Genie response including any generated SQL and query results.
    """
    w = get_workspace_client()
    if w is None:
        return "Error: Databricks SDK not available."

    try:
        msg = w.genie.start_conversation_and_wait(
            space_id=GENIE_SPACE_ID,
            content=question,
        )

        parts = []

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
                    parts.append(f"\n**Generated SQL:**\n```sql\n{sql_text}\n```")

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
                                if len(df) > 30:
                                    parts.append(f"\nShowing first 30 of {len(df)} rows:\n\n{df.head(30).to_markdown(index=False)}")
                                else:
                                    parts.append(f"\n{df.to_markdown(index=False)}")
                except Exception:
                    pass  # Query result fetch is best-effort

        if not parts:
            return "Genie processed the question but returned no content."
        return "\n".join(parts)
    except Exception as e:
        return f"Genie query error: {e}"


# ── Tool registry ────────────────────────────────────────────────────────────

TOOLS = [ask_genie, query_data, run_sarima_forecast, run_monte_carlo, query_annotations]

# OpenAI-compatible tool definitions for the LLM
TOOL_DEFINITIONS = []
for fn in TOOLS:
    TOOL_DEFINITIONS.append({
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": fn.__doc__,
        },
    })

TOOL_MAP = {fn.__name__: fn for fn in TOOLS}
