"""Actuarial chatbot agent using Databricks Foundation Model API.

Uses tool-calling via the OpenAI-compatible chat completions API,
routed through AI Gateway on a Databricks model serving endpoint.
"""

import json
import inspect
import os
import re
from typing import Generator

from config import CATALOG, SCHEMA, WORKSPACE_HOST, LLM_ENDPOINT_NAME
from chatbot.tools import TOOL_MAP, AVAILABLE_TABLES

SYSTEM_PROMPT = f"""You are an actuarial risk analyst assistant for a Canadian insurance portfolio.
You help users understand claims forecasting, capital requirements, and risk management.

## Your capabilities
1. **Query data** — You can run SQL queries against the actuarial workshop tables to answer data questions. Always use the fully qualified table name: `{CATALOG}.{SCHEMA}.<table>`.
2. **SARIMA forecasting** — You can generate on-demand claims forecasts for 1-24 months ahead using the deployed SARIMA+GARCH model.
3. **Monte Carlo simulation** — You can run portfolio loss simulations with custom parameters to compute capital requirements (VaR, CVaR, SCR).
4. **Analyst annotations** — You can query scenario annotations that analysts have recorded in the Lakebase database.

## Available tables in {CATALOG}.{SCHEMA}
{chr(10).join(f'- **{name}**: {desc}' for name, desc in AVAILABLE_TABLES.items())}

## Domain knowledge
- **Segments**: 40 segments = 4 product lines (Personal Auto, Commercial Auto, Homeowners, Commercial Property) x 10 Canadian provinces. Segment IDs use the format `product_line_province` (e.g., `personal_auto_ontario`).
- **SARIMA**: Seasonal ARIMA model fitted per segment on 72 months of history, forecasting 12 months ahead. GARCH(1,1) on residuals captures time-varying volatility.
- **Monte Carlo**: t-Copula (df=4) with lognormal marginals across 3 business lines (Property, Auto, Liability). 40M simulation paths for baseline.
- **Capital metrics**:
  - **Expected Loss**: Mean annual portfolio loss across all simulations.
  - **VaR 99% (1-in-100yr)**: Loss exceeded in only 1% of scenarios.
  - **SCR / VaR 99.5% (1-in-200yr)**: Solvency II regulatory capital requirement.
  - **CVaR 99% (Tail Risk)**: Average loss in the worst 1% of scenarios — more conservative than VaR; used in IFRS 17 risk margin.
- **GARCH(1,1)**: Captures volatility clustering in residuals. Alpha = short-run shock persistence, Beta = long-run volatility persistence. ARCH-LM p-value < 0.05 confirms heteroskedasticity.
- **Reserve triangle**: Loss development factors track how claims mature over development lags. Used for IBNR (Incurred But Not Reported) estimation.
- **Stress scenarios**: Baseline, stress_corr (higher correlations), cat_event (catastrophe), inflation_shock (higher means+CVs).

## Guidelines
- When asked about data, prefer using the query_data tool to fetch actual numbers rather than guessing.
- For forecasting questions, use run_sarima_forecast.
- For "what if" capital questions, use run_monte_carlo with adjusted parameters.
- Keep responses concise and business-focused. Explain actuarial concepts when asked.
- Format numbers clearly: use $M for millions, % for percentages, commas for thousands.
- When showing query results, summarize the key insights rather than just dumping raw data.
"""


def _get_openai_client():
    """Create an OpenAI client pointing to the Databricks workspace."""
    from openai import OpenAI

    # In Databricks Apps, DATABRICKS_HOST and auth are auto-injected
    host = WORKSPACE_HOST.rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")

    # Try to get token from the SDK auth if not set directly
    if not token:
        from auth import get_workspace_client
        w = get_workspace_client()
        if w is not None:
            try:
                if w.config.token:
                    token = w.config.token
                else:
                    headers = {}
                    w.config.authenticate(headers)
                    if "Authorization" in headers:
                        token = headers["Authorization"].replace("Bearer ", "")
            except Exception:
                pass

    if not host or not token:
        return None

    return OpenAI(
        api_key=token,
        base_url=f"{host}/serving-endpoints",
    )


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
        yield "Unable to connect to the LLM endpoint. Check DATABRICKS_HOST and authentication."
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

        # Append assistant message with tool calls
        full_messages.append(assistant_msg.model_dump())

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
