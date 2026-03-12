"""Multi-agent orchestrator for the actuarial chatbot.

Follows the Databricks multi-agent app pattern:
https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-apps

Architecture:
- DatabricksOpenAI client for automatic SP OAuth authentication
- Chat Completions API with tool calling for orchestration
- function tools: SQL, Frequency Forecast, Bootstrap Reserve, Genie, Lakebase annotations
- MLflow autolog for tracing
"""

import json
import inspect
import re
from typing import Generator

from config import CATALOG, SCHEMA, LLM_ENDPOINT_NAME
from chatbot.tools import TOOL_MAP, AVAILABLE_TABLES

# ── MLflow tracing (best-effort) ───────────────────────────────────────
_mlflow_ok = False
try:
    import mlflow
    mlflow.openai.autolog()
    mlflow.set_experiment("/Shared/actuarial-workshop-app-traces")
    _mlflow_ok = True
except Exception:
    mlflow = None  # tracing is optional

SYSTEM_PROMPT = f"""You are a reserve risk analyst assistant for a Canadian P&C insurance portfolio.
You help users understand reserve adequacy, IBNR distributions, frequency forecasting, and reserve risk management.

## Your capabilities
1. **Ask Genie** — For ANY data question about the portfolio (trends, comparisons, "show me", "top N", "which segments", aggregations), ALWAYS use ask_genie first. It connects to the AI/BI Genie space which understands all workshop tables and generates SQL automatically. This is the primary data exploration tool.
2. **Query data** — Only use this as a fallback if ask_genie fails or returns no results. Use query_data with the fully qualified table name: `{CATALOG}.{SCHEMA}.<table>`.
3. **Frequency forecasting** — Generate on-demand claim frequency forecasts for 1-24 months ahead using the deployed SARIMAX+GARCH model.
4. **Bootstrap reserve simulation** — Run Bootstrap Chain Ladder reserve simulations with custom parameters to compute reserve risk metrics (Best Estimate IBNR, VaR, CVaR, Reserve Risk Capital).
5. **Analyst annotations** — Query scenario annotations that analysts have recorded in the Lakebase database.

## Available tables in {CATALOG}.{SCHEMA}
{chr(10).join(f'- **{name}**: {desc}' for name, desc in AVAILABLE_TABLES.items())}

## Domain knowledge
- **Segments**: 40 segments = 4 product lines (Personal Auto, Commercial Auto, Homeowners, Commercial Property) x 10 Canadian provinces. Segment IDs use the format `product_line_province` (e.g., `personal_auto_ontario`).
- **SARIMAX**: Seasonal ARIMA model fitted per segment on 84 months of history (2019-2025), forecasting 12 months ahead. GARCH(1,1) on residuals captures time-varying volatility. Results in `predictions_frequency_forecast`.
- **Chain Ladder**: Deterministic best-estimate IBNR from weighted link ratios (LDFs). Mack variance provides analytical reserve uncertainty. Fitted per product line.
- **Bootstrap Chain Ladder**: Resamples scaled Pearson residuals from the fitted chain ladder model. Each bootstrap replicate produces a pseudo-triangle and a complete IBNR estimate. The collection of replicate IBNRs forms the predictive reserve distribution. Results in `predictions_bootstrap_reserves`.
- **Reserve deterioration scenarios**: Pre-computed in `predictions_reserve_scenarios`:
  - **adverse_development**: LDFs inflated — reserves develop worse than expected
  - **judicial_inflation**: Social inflation / nuclear verdicts on Auto lines at long lags
  - **pandemic_tail**: Extended development periods due to delayed settlements
  - **superimposed_inflation**: Calendar-year trend (CPI + X%) across all lines
- **Reserve evolution**: 12-month reserve adequacy outlook in `predictions_reserve_evolution`.
- **Run-off projection**: Multi-period surplus trajectory with regime-switching in `predictions_runoff_projection`.
- **LDF volatility**: Development factor volatility per product line in `predictions_ldf_volatility`.
- **Reserve risk metrics** (from Bootstrap Chain Ladder):
  - **Best Estimate IBNR**: Mean of the bootstrap IBNR distribution — the actuarial central estimate.
  - **VaR 99% (1-in-100yr)**: IBNR level exceeded in only 1% of bootstrap replications.
  - **Reserve Risk Capital / VaR 99.5% (1-in-200yr)**: The 99.5% IBNR threshold. Equivalent to the Solvency II reserve risk SCR component. In Canada, OSFI uses the MCT framework.
  - **CVaR 99% (Tail Risk)**: Average IBNR across the worst 1% of replications — used in IFRS 17 risk adjustment.
- **Canadian MCT**: The app shows a simplified MCT ratio using OSFI-prescribed risk factors (5-10% on reserves, 12-22% on premium). MCT Ratio = Capital Available / Capital Required × 100%. OSFI target: 150%.
- **GARCH(1,1)**: Captures volatility clustering in SARIMAX residuals. Alpha = short-run shock persistence, Beta = long-run volatility persistence. ARCH-LM p-value < 0.05 confirms heteroskedasticity.
- **Reserve triangle**: Loss development factors track how claims mature over development lags. Used for IBNR estimation. Data in `gold_reserve_triangle` with incremental_paid and incremental_incurred columns.
- **Feature Store**: `features_segment_monthly` provides rolling statistical and macro-economic features per segment.

## Table naming convention
Tables follow a stage-prefix convention:
- `raw_*` — ingested source data (raw_reserve_development, raw_claims_events, raw_macro_indicators)
- `bronze_*` — SDP streaming append-only tables
- `silver_*` — SDP cleaned/SCD2 tables
- `gold_*` — SDP business-ready aggregations (gold_claims_monthly, gold_reserve_triangle, gold_macro_features)
- `features_*` — Feature Store tables (features_segment_monthly)
- `predictions_*` — Model output tables (predictions_frequency_forecast, predictions_bootstrap_reserves, predictions_reserve_scenarios, predictions_reserve_evolution, predictions_runoff_projection, predictions_ldf_volatility)

## Bootstrap reserve parameters (for run_bootstrap_reserve)
- **Product lines**: Personal Auto, Commercial Auto, Homeowners, Commercial Property
- **IBNR means**: Calibrated from chain ladder fit on gold_reserve_triangle
- **CVs (coefficients of variation)**: LDF volatility from development factor standard deviations
- **Correlation matrix**: Cross-line development factor correlations
- **Replications**: 10,000 default (up to 100,000)

## Reserve scenario parameters (for run_bootstrap_reserve)
1. **baseline**: Standard reserve development — use default parameters.
2. **adverse_development**: LDFs inflated by 20%. Reserves develop worse than expected.
   - ldf_multiplier=1.2, cv adjustments x1.3
3. **judicial_inflation**: Social inflation on Auto lines at long development lags.
   - Auto CVs x1.2, Auto means x1.3
4. **pandemic_tail**: Extended development periods (+6 months) due to delayed settlements.
   - All means x1.1, all CVs x1.4
5. **superimposed_inflation**: Calendar-year trend across all lines.
   - inflation_adj=0.03 (CPI + 3%)

## Routing guidelines — STRICT RULES
- **ANY data lookup or query** (current values, trends, comparisons, "show me", "what are the current", "top N", "which segments", aggregations, looking up pre-computed results) → ALWAYS call ask_genie FIRST. This includes questions about current IBNR, VaR, reserve risk, claims data, forecasts stored in tables, etc. NEVER call query_data as a first choice — only use it if ask_genie returns an error or empty results.
- **On-demand forecasting** ("forecast N months", "predict", "project frequency ahead") → use run_frequency_forecast.
- **Custom "what if" reserve simulations** ("what happens if LDFs deteriorated", "run a reserve scenario with inflation", "simulate adverse development") → use run_bootstrap_reserve with adjusted parameters. Refer to the scenario parameters above.
- **Annotations** ("analyst notes", "scenario comments", "approvals") → use query_annotations.
- **Concepts** (no tool needed — explain from your domain knowledge).

## Formatting rules
- Keep responses concise and business-focused. Explain actuarial concepts when asked.
- IMPORTANT: When writing dollar amounts, write them as plain text like "26.5M" or "26.5 million". Do NOT use LaTeX or math notation with dollar signs. For example, write "VaR is 73.2M" not "VaR is $73.2M".
- Use commas for thousands, percent signs for percentages.
- When showing query results, summarize the key insights rather than just dumping raw data.
"""


def _get_openai_client():
    """Create a DatabricksOpenAI client.

    DatabricksOpenAI (from databricks-openai) auto-handles authentication
    using the Databricks SDK credential chain:
    - In Databricks Apps: SP OAuth via auto-injected DATABRICKS_HOST/CLIENT_ID/CLIENT_SECRET
    - In notebooks: workspace token from dbutils
    - Locally: profile from ~/.databrickscfg

    Reference: https://github.com/databricks/app-templates/tree/main/agent-openai-agents-sdk-multiagent
    """
    try:
        from databricks_openai import DatabricksOpenAI
        return DatabricksOpenAI()
    except ImportError:
        # Fallback: manual OpenAI client construction
        import os
        from openai import OpenAI
        from config import WORKSPACE_HOST

        host = WORKSPACE_HOST.rstrip("/")
        if not host:
            return None

        # Try SDK auth for token
        token = os.environ.get("DATABRICKS_TOKEN", "")
        if not token:
            from auth import get_workspace_client
            w = get_workspace_client()
            if w is not None:
                try:
                    headers = {}
                    w.config.authenticate(headers)
                    if "Authorization" in headers:
                        token = headers["Authorization"].replace("Bearer ", "")
                except Exception:
                    pass

        if not token:
            return None
        return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")
    except Exception:
        return None


def _get_tool_schemas():
    """Build OpenAI-compatible tool schemas from the tool functions."""
    schemas = []
    for name, fn in TOOL_MAP.items():
        sig = inspect.signature(fn)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            ann = param.annotation
            prop = {"description": ""}

            if ann == str:
                prop["type"] = "string"
            elif ann == int:
                prop["type"] = "integer"
            elif ann == float:
                prop["type"] = "number"
            else:
                prop["type"] = "string"

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

            properties[param_name] = prop

        # Extract param descriptions from docstring
        docstring = fn.__doc__ or ""
        for param_name in properties:
            pattern = rf"{param_name}:\s*(.+?)(?:\n\s*\w+:|$)"
            match = re.search(pattern, docstring, re.DOTALL)
            if match:
                properties[param_name]["description"] = match.group(1).strip()

        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (fn.__doc__ or "").split("\n\n")[0].strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return schemas


def chat(messages: list[dict], max_tool_rounds: int = 5) -> Generator[str, None, None]:
    """Run the agent loop: LLM -> tool calls -> LLM -> ... -> final response.

    Yields partial text as it becomes available (for streaming in Streamlit).
    """
    # Start an MLflow trace for the full agent conversation
    _trace_ctx = None
    if _mlflow_ok:
        try:
            _trace_ctx = mlflow.start_span(name="agent_chat", span_type="AGENT")
            _trace_ctx.__enter__()
            _trace_ctx.set_inputs({"messages": messages, "max_tool_rounds": max_tool_rounds})
        except Exception:
            _trace_ctx = None

    client = _get_openai_client()
    if client is None:
        yield (
            "Unable to connect to the LLM endpoint. "
            "Ensure `databricks-openai` is installed and the app has valid "
            "DATABRICKS_HOST / DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET."
        )
        return

    tool_schemas = _get_tool_schemas()
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    for _ in range(max_tool_rounds):
        try:
            response = client.chat.completions.create(
                model=LLM_ENDPOINT_NAME,
                messages=full_messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=4096,
            )
        except Exception as e:
            if _trace_ctx:
                try:
                    _trace_ctx.__exit__(type(e), e, e.__traceback__)
                except Exception:
                    pass
            yield f"Error calling LLM endpoint: {e}"
            return

        choice = response.choices[0]
        assistant_msg = choice.message
        tool_calls = assistant_msg.tool_calls

        if not tool_calls:
            content = assistant_msg.content or ""
            if _trace_ctx:
                try:
                    _trace_ctx.set_outputs({"response": content[:1000] if len(content) > 1000 else content})
                    _trace_ctx.__exit__(None, None, None)
                except Exception:
                    pass
            if content:
                yield content
            return

        # Append assistant message with tool calls — strip fields the
        # Databricks Foundation Model API doesn't accept (e.g. annotations).
        assistant_dict = {
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
        full_messages.append(assistant_dict)

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            fn = TOOL_MAP.get(fn_name)
            if fn is None:
                result = f"Unknown tool: {fn_name}"
            else:
                yield f"_Calling {fn_name}..._\n\n"
                try:
                    if _mlflow_ok:
                        with mlflow.start_span(name=fn_name, span_type="TOOL") as tool_span:
                            tool_span.set_inputs(args)
                            result = fn(**args)
                            tool_span.set_outputs({"result": result[:500] if len(result) > 500 else result})
                    else:
                        result = fn(**args)
                except Exception as e:
                    result = f"Tool error: {e}"

            full_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    if _trace_ctx:
        try:
            _trace_ctx.set_outputs({"response": "max_tool_rounds_exceeded"})
            _trace_ctx.__exit__(None, None, None)
        except Exception:
            pass
    yield "Reached maximum tool call rounds. Please simplify your question."
