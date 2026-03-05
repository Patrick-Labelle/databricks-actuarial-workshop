"""Multi-agent orchestrator for the actuarial chatbot.

Follows the Databricks multi-agent app pattern:
https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-apps

Architecture:
- DatabricksOpenAI client for automatic SP OAuth authentication
- Chat Completions API with tool calling for orchestration
- function tools: SQL, SARIMA, Monte Carlo, Genie, Lakebase annotations
- MLflow autolog for tracing
"""

import json
import inspect
import re
from typing import Generator

from config import CATALOG, SCHEMA, LLM_ENDPOINT_NAME
from chatbot.tools import TOOL_MAP, AVAILABLE_TABLES

# ── MLflow 3 tracing (best-effort) ──────────────────────────────────────
try:
    import mlflow
    mlflow.openai.autolog()
except ImportError:
    pass  # mlflow not installed — tracing is optional

SYSTEM_PROMPT = f"""You are an actuarial risk analyst assistant for a Canadian insurance portfolio.
You help users understand claims forecasting, capital requirements, and risk management.

## Your capabilities
1. **Ask Genie** — For ANY data question about the portfolio (trends, comparisons, "show me", "top N", "which segments", aggregations), ALWAYS use ask_genie first. It connects to the AI/BI Genie space which understands all workshop tables and generates SQL automatically. This is the primary data exploration tool.
2. **Query data** — Only use this as a fallback if ask_genie fails or returns no results. Use query_data with the fully qualified table name: `{CATALOG}.{SCHEMA}.<table>`.
3. **SARIMA forecasting** — Generate on-demand claims forecasts for 1-24 months ahead using the deployed SARIMA+GARCH model.
4. **Monte Carlo simulation** — Run portfolio loss simulations with custom parameters to compute capital requirements (VaR, CVaR, SCR).
5. **Analyst annotations** — Query scenario annotations that analysts have recorded in the Lakebase database.

## Available tables in {CATALOG}.{SCHEMA}
{chr(10).join(f'- **{name}**: {desc}' for name, desc in AVAILABLE_TABLES.items())}

## Domain knowledge
- **Segments**: 40 segments = 4 product lines (Personal Auto, Commercial Auto, Homeowners, Commercial Property) x 10 Canadian provinces. Segment IDs use the format `product_line_province` (e.g., `personal_auto_ontario`).
- **SARIMA**: Seasonal ARIMA model fitted per segment on 84 months of history (2019-2025), forecasting 12 months ahead. GARCH(1,1) on residuals captures time-varying volatility. Results in `predictions_sarima`.
- **Monte Carlo models**: Two approaches available:
  - **Aggregate model** (Standard Formula): t-Copula with lognormal marginals across 3 business lines. Parameters are historically calibrated from gold_claims_monthly. Results in `predictions_monte_carlo`.
  - **Collective Risk Model** (Internal Model): Frequency-Severity bottom-up approach. NegBin(λ,k) for claim counts + Lognormal(μ,σ) per claim. More granular but computationally heavier.
- **Multi-period simulation**: 12-month surplus evolution with 2-state regime-switching (Normal/Crisis). Tracks cumulative surplus: S(t) = S(t-1) + Premium - Loss + Investment. Results in `predictions_surplus_evolution`.
- **Stress scenarios**: Pre-computed stress tests in `predictions_stress_scenarios`. Capital outlook in `predictions_risk_timeline`.
- **Capital metrics**:
  - **Expected Loss**: Mean annual portfolio loss across all simulations.
  - **VaR 99% (1-in-100yr)**: Loss exceeded in only 1% of scenarios.
  - **SCR / VaR 99.5% (1-in-200yr)**: Solvency II regulatory capital requirement.
  - **CVaR 99% (Tail Risk)**: Average loss in the worst 1% of scenarios — more conservative than VaR; used in IFRS 17 risk margin.
- **Historical calibration**: Means and correlations are calibrated from gold_claims_monthly via trailing 12-month averages and empirical log-loss correlations. CVs come from GARCH(1,1) on SARIMA residuals. Copula df selected via AIC on empirical tail dependence.
- **Regime-switching**: 2-state Markov model (Normal → Crisis with 5% transition probability, Crisis → Normal with 15%). Crisis state: 1.3x means, 1.15x CVs, elevated correlations.
- **GARCH(1,1)**: Captures volatility clustering in residuals. Alpha = short-run shock persistence, Beta = long-run volatility persistence. ARCH-LM p-value < 0.05 confirms heteroskedasticity.
- **Reserve triangle**: Loss development factors track how claims mature over development lags. Used for IBNR (Incurred But Not Reported) estimation. Data in `gold_reserve_triangle`.
- **Feature Store**: `features_segment_monthly` provides rolling statistical and macro-economic features per segment.

## Table naming convention
Tables follow a stage-prefix convention showing their position in the data pipeline:
- `raw_*` — ingested source data (raw_reserve_development, raw_claims_events, raw_macro_indicators)
- `bronze_*` — SDP streaming append-only tables
- `silver_*` — SDP cleaned/SCD2 tables
- `gold_*` — SDP business-ready aggregations (gold_claims_monthly, gold_reserve_triangle, gold_macro_features)
- `features_*` — Feature Store tables (features_segment_monthly)
- `predictions_*` — Model output tables (predictions_sarima, predictions_monte_carlo, predictions_stress_scenarios, predictions_risk_timeline, predictions_surplus_evolution)

## Monte Carlo baseline parameters (historically calibrated)
- **Business lines**: Property, Auto, Liability
- **Expected annual losses**: Calibrated from gold_claims_monthly (trailing 12-month average, annualized)
- **CVs (coefficients of variation)**: GARCH-derived from SARIMA residual volatilities
- **Correlation matrix**: Empirical Pearson correlations from monthly log-loss time series
- **Copula**: Student-t with AIC-selected df (typically 3-6)
- **Simulation paths**: 40M (baseline), 10M per task x 4 tasks

## Stress test scenarios (with exact parameters for run_monte_carlo)
1. **baseline**: Standard portfolio — use default parameters (no changes needed).
2. **stress_corr** (Systemic Risk): Correlations spike under market-wide stress.
   - Correlations: Property-Auto 0.75, Property-Liability 0.60, Auto-Liability 0.65
   - Means and CVs unchanged.
3. **cat_event** (1-in-250yr Catastrophe): Major natural disaster (hurricane/earthquake).
   - Means multiplied: Property x3.5, Auto x1.8, Liability x1.4
   - CVs multiplied: Property x1.5, Auto x1.3, Liability x1.2
   - Correlations: Property-Auto 0.55, Property-Liability 0.40, Auto-Liability 0.45
   - Plus Poisson(lambda=0.05) jump shocks for discrete CAT events.
4. **inflation_shock** (Inflation +30%): Sustained loss-cost inflation.
   - All means multiplied by 1.30
   - All CVs multiplied by 1.15

## Model type parameter
When the user asks to compare "standard formula" vs "internal model", run two simulations:
- model_type="aggregate" for the standard formula (t-Copula + Lognormal)
- model_type="collective_risk" for the internal model (frequency-severity)
Use simulation_mode="multi_period" when the user asks about surplus evolution or ruin probability.

## Routing guidelines — STRICT RULES
- **ANY data lookup or query** (current values, trends, comparisons, "show me", "what are the current", "top N", "which segments", aggregations, looking up pre-computed results) → ALWAYS call ask_genie FIRST. This includes questions about current SCR, VaR, capital requirements, claims data, forecasts stored in tables, etc. NEVER call query_data as a first choice — only use it if ask_genie returns an error or empty results.
- **On-demand forecasting** ("forecast N months", "predict", "project claims ahead") → use run_sarima_forecast.
- **Custom "what if" simulations** ("what happens if losses doubled", "run a stress test with custom parameters", "simulate with higher correlations") → use run_monte_carlo with adjusted parameters. Refer to the stress scenario parameters above.
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
            yield f"Error calling LLM endpoint: {e}"
            return

        choice = response.choices[0]
        assistant_msg = choice.message
        tool_calls = assistant_msg.tool_calls

        if not tool_calls:
            content = assistant_msg.content or ""
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
                    result = fn(**args)
                except Exception as e:
                    result = f"Tool error: {e}"

            full_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    yield "Reached maximum tool call rounds. Please simplify your question."
