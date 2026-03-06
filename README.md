# Databricks Actuarial Workshop

An end-to-end actuarial modeling solution on the Databricks Lakehouse — synthetic
Canadian insurance portfolio data, a full declarative medallion pipeline, statistical models that
**forecast claims, model volatility, and simulate portfolio loss** (SARIMAX/GARCH/Monte Carlo),
a tool-calling **AI chatbot agent** registered with the Databricks Agent Framework,
and a Streamlit risk dashboard —
packaged as a single **Databricks Asset Bundle** for one-command deployment.

---

## Insurance Portfolio

The synthetic portfolio models a Canadian P&C insurer across **40 segments** (4 product lines x 10 provinces) over an **84-month history** (Jan 2019 - Dec 2025), generating ~42M individual claim incidents.

| Product Line | Business Context | Typical Drivers |
|---|---|---|
| Personal Auto | Mandatory coverage; high frequency, moderate severity | Unemployment (commute patterns), seasonality, fraud cycles |
| Commercial Auto | Fleet/trucking; lower frequency, higher severity | Economic activity, fuel costs, regulatory changes |
| Homeowners | Property damage + liability; weather-sensitive | HPI (rebuild costs), housing starts (exposure growth), CAT events |
| Commercial Property | Large commercial risks; low frequency, heavy tail | Construction activity, inflation, catastrophe concentration |

**Regions:** All 10 Canadian provinces, each with distinct macro conditions (unemployment, housing prices, construction activity) sourced from Statistics Canada.

---

## Statistical Models

### SARIMAX — Claims Forecasting

**Business purpose:** Forecast monthly claim frequency and severity by segment for reserving, pricing, and capital planning.

**Method:** Seasonal ARIMA with exogenous macro variables (unemployment rate, HPI growth, housing starts) from Statistics Canada. One model per segment (40 total), fit via `statsmodels` using `applyInPandas` for distributed execution.

**Output:** 12-month ahead forecasts with 95% confidence intervals per segment. MLflow logs MAPE with/without macro variables for direct comparison.

### GARCH(1,1) — Volatility Modeling

**Business purpose:** Model time-varying uncertainty in claim outcomes for risk capital and reinsurance pricing. Stable loss forecasts are insufficient — actuaries need to know *how uncertain* those forecasts are.

**Method:** Fit GARCH(1,1) to SARIMA residuals per segment via the `arch` library. Extracts conditional variance (time-varying volatility) and computes segment-level coefficients of variation (CV = sigma/mu) that feed the Monte Carlo simulation.

**Output:** Per-segment conditional volatility series, ARCH-LM test p-values, and calibrated CVs bridged to the three Monte Carlo portfolio lines.

### Monte Carlo — Portfolio Loss Simulation

**Business purpose:** Quantify portfolio-level tail risk (VaR, CVaR/TVaR) for regulatory capital (SCR), internal risk limits, and catastrophe scenario planning.

**Method:** Three simulation approaches run as 64 Ray-distributed tasks (~640M total paths):

| Approach | What It Does | Business Use |
|---|---|---|
| **Aggregate (t-Copula)** | Lognormal marginals with t-copula (df=4) for tail dependence between Property, Auto, Liability | Standard formula capital; quick what-if |
| **Collective Risk** | Negative Binomial frequency + Lognormal severity per segment; CLT compound approximation | Bottom-up reserving; frequency-severity decomposition |
| **Multi-Period (Regime-Switching)** | 12-month surplus trajectory with 2-state (Normal/Crisis) Markov regime model | Dynamic Financial Analysis; ruin probability |

**Calibration:** Portfolio means from `gold_claims_monthly`, correlations from empirical log-loss series, CVs from GARCH. No hardcoded assumptions — all parameters are historically calibrated.

**Scenarios:**
- **Baseline:** Static calibrated means (40M paths)
- **12-month VaR evolution:** SARIMA-driven means by month (480M paths)
- **Stress tests** (120M paths each):
  - `cat_event` — 1-in-250yr catastrophe (Property 3.5x, Poisson jump process)
  - `stress_corr` — Systemic contagion (correlations spike to 0.65-0.75)
  - `inflation_shock` — Loss-cost inflation (+30% means, +15% CV)

---

## Data Pipeline

Three data streams flow through a Spark Declarative Pipeline (SDP) medallion architecture:

```
Reserve CDC:   raw_reserve_development → bronze_reserve_cdc → silver_reserves (SCD2)  → gold_reserve_triangle
Claims:        raw_claims_events       → bronze_claims      → gold_claims_monthly      → silver_rolling_features
Macro:         raw_macro_indicators    → bronze_macro_indicators → silver_macro_indicators (SCD2) → gold_macro_features
```

### Tables

| Table | Layer | Description |
|---|---|---|
| `raw_reserve_development` | Landing | Synthetic reserve development CDC events (84 months x 4 products x 10 regions x 12 dev lags) |
| `raw_claims_events` | Landing | ~42M individual claim incidents (Jan 2019 - Dec 2025) |
| `raw_macro_indicators` | Landing | Statistics Canada unemployment, HPI, housing starts (10 provinces, 1976-2026) |
| `bronze_reserve_cdc` | Bronze | Append-only reserve CDC stream with data quality expectations |
| `bronze_claims` | Bronze | Append-only claim incidents stream |
| `bronze_macro_indicators` | Bronze | Append-only macro indicator ingestion |
| `silver_reserves` | Silver | SCD Type 2 reserve history on `(segment_id, accident_month, dev_lag)` |
| `silver_macro_indicators` | Silver | SCD Type 2 macro data tracking StatCan revisions |
| `silver_rolling_features` | Silver | Rolling 3m/6m/12m statistical features per segment |
| `gold_reserve_triangle` | Gold | Loss development triangle by segment x accident month x dev lag |
| `gold_claims_monthly` | Gold | Monthly aggregates: claims_count, total_incurred, avg_severity, earned_premium, loss_ratio (40 segments x 84 months) |
| `gold_macro_features` | Gold | Pivoted macro features: unemployment_rate, hpi_index, hpi_growth, housing_starts by region x month |
| `features_segment_monthly` | Feature Store | Point-in-time correct features with timestamp keys for leakage-free training; Online Table for real-time lookup |
| `predictions_sarima` | Predictions | Per-segment SARIMA+GARCH forecasts (actuals + 12-month ahead) with confidence intervals and volatility |
| `predictions_monte_carlo` | Predictions | Portfolio baseline Monte Carlo results: expected loss, VaR 99%, VaR 99.5% (SCR), CVaR 99% |
| `predictions_risk_timeline` | Predictions | 12-month SARIMA-driven VaR/CVaR evolution |
| `predictions_stress_scenarios` | Predictions | Pre-computed stress test results (CAT, systemic, inflation) with delta-vs-baseline |
| `predictions_surplus_evolution` | Predictions | Multi-period surplus trajectory with regime-switching: percentile bands + ruin probability |
| `predictions_regime_parameters` | Predictions | Regime-switching model parameters (Normal/Crisis state means, CVs, correlations, transition matrix) |
| `predictions_reserve_validation` | Predictions | Reserve adequacy validation: forecast vs actual development by segment x accident month |

### Statistics Canada Macro Integration

The SARIMAX models use real macroeconomic data as exogenous variables:

| StatCan Table | Indicator | Role in SARIMAX |
|---|---|---|
| `14-10-0017-01` | `unemployment_rate` | Leading indicator for auto/liability claims frequency |
| `18-10-0205-01` | `hpi_index` / `hpi_growth` | Housing market proxy for homeowners/commercial property claims |
| `34-10-0158-01` | `housing_starts` | New-exposure leading indicator |

**Network-restricted workspaces:** `scripts/fetch_macro_data.py` is fault-tolerant. If `www150.statcan.gc.ca` is unreachable, it creates an empty table and the model falls back to baseline SARIMA.

---

## UC Models and Serving Endpoints

### Registered Models

| Model | UC Name | Type | Input | Output |
|---|---|---|---|---|
| SARIMA Forecaster | `sarima_claims_forecaster` | MLflow PyFunc | `horizon` (1-24 months) | Per-segment forecast_mean, forecast_lo95, forecast_hi95 by month |
| Monte Carlo Portfolio | `monte_carlo_portfolio` | MLflow PyFunc | Scenario parameters (means, CVs, correlations, model_type, simulation_mode) | mean_loss_M, var_99_M, var_995_M, cvar_99_M, max_loss_M |
| Chatbot Agent | `actuarial_chatbot_agent` | MLflow ChatModel | Chat messages (role/content) | ChatCompletionResponse with tool calls |

### Serving Endpoints

| Endpoint | Model | AI Gateway | Purpose |
|---|---|---|---|
| `actuarial-workshop-sarima-forecaster` | sarima_claims_forecaster @Champion | Inference table, rate limits, usage tracking | Quick Forecast tab, chatbot tool |
| `actuarial-workshop-monte-carlo` | monte_carlo_portfolio @Champion | Inference table, rate limits, usage tracking | Stress Testing tab, chatbot tool |
| `actuarial-workshop-chatbot-agent` | actuarial_chatbot_agent @Champion | Inference table, Review App, MLflow tracing | AI Gateway agents tab, Review App |

The chatbot agent endpoint is deployed via `databricks.agents.deploy()` which automatically provisions:
- Review App for stakeholder feedback collection
- Inference tables for request/response logging
- Real-time MLflow 3 tracing

---

## Setup Job Tasks

The full setup job runs 8 tasks in sequence on a single classic ML cluster (Ray-on-Spark):

| # | Task | Notebook | Depends On | What It Does |
|---|---|---|---|---|
| 1 | `generate_source_data` | `src/01_data_pipeline.py` | — | Generate synthetic reserve CDC + claim incidents |
| 1b | `fetch_macro_data` | `scripts/fetch_macro_data.py` | — | Fetch StatCan unemployment, HPI, housing starts (parallel with task 1) |
| 2 | `run_dlt_pipeline` | SDP pipeline | 1, 1b | Bronze -> Silver (SCD2) -> Gold across all three data streams |
| 3 | `build_feature_store` | `src/02_feature_store.py` | 2 | UC Feature Store registration with point-in-time keys |
| 4 | `fit_statistical_models` | `src/03_classical_stats_at_scale.py` | 3 | SARIMAX/GARCH per segment + Ray-distributed Monte Carlo (640M paths) |
| 5 | `prepare_app_infrastructure` | `src/04_model_serving.py` | 4 | Serving endpoints + AI Gateway, Online Table, Lakebase, Genie Space |
| 6 | `setup_app_dependencies` | `src/ops/app_setup.py` | 5 | UC grants + CAN_QUERY on endpoints + Genie space permissions for app SP |
| 7 | `register_chatbot_agent` | `src/ops/register_agent.py` | 6 | Log ChatModel to MLflow, register to UC, deploy via agents.deploy() |

A **Monthly Model Refresh** job (paused by default) re-runs tasks 1b -> 2 -> 4 on a schedule.

---

## Streamlit App (6 Tabs)

| Tab | Name | Description |
|---|---|---|
| 0 | Risk Assistant | Tool-calling chatbot agent (Genie, SQL, SARIMA, Monte Carlo, Lakebase annotations) |
| 1 | Claims Forecast | Per-segment SARIMA forecast with GARCH volatility bands and analyst annotations (Lakebase) |
| 2 | Capital Requirements | Monte Carlo VaR/CVaR metrics with fitted lognormal PDF visualization |
| 3 | Quick Forecast | On-demand SARIMA+GARCH forecast via live serving endpoint (1-24 month horizon) |
| 4 | Stress Testing | Custom Monte Carlo scenario modeling with parameter sliders and delta-SCR comparison |
| 5 | Catastrophe & Reserves | Pre-computed CAT/systemic/inflation scenarios, reserve triangle heatmap, surplus evolution |

### Chatbot Tools

The Risk Assistant agent uses Llama 3.3 70B with 5 function-calling tools:

| Tool | Purpose | Backend |
|---|---|---|
| `ask_genie` | Natural-language data queries (primary data tool) | AI/BI Genie Space |
| `query_data` | Direct SQL fallback (read-only) | SQL Warehouse |
| `run_sarima_forecast` | On-demand claims forecast (1-24 months) | SARIMA serving endpoint |
| `run_monte_carlo` | Custom portfolio loss simulation | Monte Carlo serving endpoint |
| `query_annotations` | Retrieve analyst scenario notes | Lakebase PostgreSQL |

---

## Deployed Resources

After running `./deploy.sh`, the following resources are created:

| Resource | Name / Location |
|---|---|
| Declarative Pipeline | `actuarial-workshop-medallion` |
| Setup Job | `Actuarial Workshop -- Full Setup` (8 tasks) |
| Monthly Refresh Job | `Actuarial Workshop -- Monthly Model Refresh` (paused) |
| SARIMA Serving Endpoint | `actuarial-workshop-sarima-forecaster` |
| Monte Carlo Serving Endpoint | `actuarial-workshop-monte-carlo` |
| Agent Serving Endpoint | `actuarial-workshop-chatbot-agent` |
| Databricks App | `actuarial-workshop` |
| Lakebase Instance | `actuarial-workshop-lakebase` |
| Lakebase Database | `actuarial_workshop_db` |
| AI/BI Genie Space | `Actuarial Workshop -- Risk Assistant` (auto-created) |
| UC Models (3) | `sarima_claims_forecaster`, `monte_carlo_portfolio`, `actuarial_chatbot_agent` |
| Feature Table | `features_segment_monthly` (+ Online Table) |
| Delta Tables (19) | See [Tables](#tables) section above |

---

## Configuration Reference

All configurable values live in `databricks.yml` under `variables:`.

| Variable | Description | Default |
|---|---|---|
| `catalog` | UC catalog name (must exist) | `my_catalog` |
| `schema` | UC schema (created if missing) | `actuarial_workshop` |
| `endpoint_name` | SARIMA serving endpoint name | `actuarial-workshop-sarima-forecaster` |
| `mc_endpoint_name` | Monte Carlo serving endpoint name | `actuarial-workshop-monte-carlo` |
| `warehouse_id` | SQL Warehouse ID for Genie Space and app queries | _(required)_ |
| `pg_database` | Lakebase PostgreSQL database name | `actuarial_workshop_db` |
| `genie_space_id` | AI/BI Genie space ID (auto-created if empty) | _(empty)_ |
| `llm_endpoint_name` | Foundation Model API endpoint for chatbot LLM | `databricks-meta-llama-3-3-70b-instruct` |
| `notification_email` | Email for job failure alerts | _(empty)_ |

---

## Repository Structure

```
.
├── databricks.yml               # Bundle config — variables, sync, includes
├── databricks.local.yml.example # Template for workspace-specific target config
├── deploy.sh                    # End-to-end deploy (bundle + Lakebase + job + app)
├── destroy.sh                   # Full teardown (workspace assets + bundle destroy)
├── resources/
│   ├── pipeline.yml             # Declarative pipeline (Bronze -> Silver -> Gold)
│   ├── jobs.yml                 # Orchestration jobs (setup + monthly refresh)
│   ├── app.yml                  # Databricks App resource + SP authorizations
│   └── lakebase.yml             # Lakebase (managed PostgreSQL) instance
├── scripts/
│   ├── fetch_macro_data.py      # Fetch StatCan macro data -> raw_macro_indicators
│   └── lakebase_setup.py        # Standalone Lakebase setup utility
├── src/                         # Job-only notebooks
│   ├── 01_data_pipeline.py      # Data generation + SDP pipeline definitions
│   ├── 02_feature_store.py      # UC Feature Store registration
│   ├── 03_classical_stats_at_scale.py  # SARIMAX/GARCH/Monte Carlo + MLflow
│   ├── 04_model_serving.py      # Endpoints, Online Table, Lakebase, Genie Space
│   └── ops/
│       ├── app_setup.py         # UC grants + endpoint permissions (job task)
│       ├── register_agent.py    # Agent registration via agents.deploy() (job task)
│       └── cleanup.py           # Manual cleanup notebook
├── interactive_workshop/        # Interactive versions with visualizations + learning notes
│   ├── 01_data_pipeline.py      # Data generation + SDP (interactive version)
│   ├── 02_performance_at_scale.py  # Spark scaling patterns (interactive-only)
│   ├── 03_feature_store.py      # UC Feature Store (interactive version)
│   ├── 04_classical_stats_at_scale.py  # SARIMAX/GARCH/Monte Carlo (interactive version)
│   ├── 05_model_serving.py      # Serving + Lakebase + Genie (interactive version)
│   ├── 06_dabs_cicd.py          # DABs CI/CD concepts (interactive-only)
│   └── 07_databricks_apps.py    # Databricks Apps + Lakebase (interactive-only)
├── app/
│   ├── app.py                   # Streamlit application (6 tabs)
│   ├── app.yaml                 # App command + valueFrom resource injections
│   ├── requirements.txt
│   ├── tabs/                    # Tab modules (tab_chatbot, tab_claims_forecast, etc.)
│   └── chatbot/                 # Agent module (agent.py, tools.py, responses_agent.py)
└── README.md
```

---

## Notebooks

### Job Pipeline (`src/`)

| # | Notebook | Job Task | Key Concepts |
|---|---|---|---|
| 1 | `01_data_pipeline.py` | `generate_source_data` | SDP, Medallion, SCD Type 2; 3 data streams (reserves, claims, macro) |
| 2 | `02_feature_store.py` | `build_feature_store` | UC Feature Store, point-in-time joins, Online Table |
| 3 | `03_classical_stats_at_scale.py` | `fit_statistical_models` | SARIMAX/GARCH, t-Copula Monte Carlo, Collective Risk, Ray-on-Spark, MLflow |
| 4 | `04_model_serving.py` | `prepare_app_infrastructure` | Serving endpoints + AI Gateway, Online Table, Lakebase, Genie Space |

### Interactive Only (`interactive_workshop/`)

| # | Notebook | Key Concepts |
|---|---|---|
| 2 | `02_performance_at_scale.py` | 4 ETL approaches timed, run-many-models, for-loop anti-patterns |
| 6 | `06_dabs_cicd.py` | DABs CI/CD, Azure DevOps |
| 7 | `07_databricks_apps.py` | Databricks Apps, Lakebase |

Interactive versions of all job pipeline modules (01, 03, 04, 05) are also in `interactive_workshop/` with `display()` calls and learning notes.

---

## Variable Injection — How the App Gets Its Config

`app/app.yaml` is uploaded as source code and does not receive bundle variable substitution.
The app's runtime configuration is provided through three channels:

| Variable | Mechanism | Where Set |
|---|---|---|
| `CATALOG`, `SCHEMA`, `PGDATABASE`, `ENDPOINT_NAME`, `MC_ENDPOINT_NAME`, `GENIE_SPACE_ID`, `LLM_ENDPOINT_NAME` | `app/_bundle_config.py` (generated by `deploy.sh`) | Inlined in `deploy.sh` |
| `PGHOST` | `valueFrom: database` in `app/app.yaml` | Injected at runtime from Lakebase resource |
| `DATABRICKS_WAREHOUSE_ID` | `valueFrom: sql-warehouse` in `app/app.yaml` | Injected at runtime from SQL warehouse resource |
| `DATABRICKS_HOST` | Auto-injected by Databricks Apps runtime | -- |
| `DATABRICKS_CLIENT_ID` / `_SECRET` | Auto-injected by Databricks Apps runtime (SP credentials) | -- |

---

## Distributed Monte Carlo (Ray-on-Spark)

All modules run on a **classic DBR 16.4 ML job cluster** with Ray-on-Spark for distributed Monte Carlo simulation.

> **Requirement:** Classic compute (not serverless-only workspaces).

The simulation uses `@ray.remote(num_cpus=1)` tasks dispatched across 4 workers (24 concurrent tasks). All 64 tasks are submitted in a single batch:

- **Baseline:** 4 tasks x 10M = 40M paths
- **12-month VaR evolution:** 12 months x 4 tasks x 10M = 480M paths
- **Stress scenarios:** 3 scenarios x 4 tasks x 10M = 120M paths

CPU-only cluster; ~2-3 minutes wall time for 640M total paths.

---

## Lakebase Authentication

Lakebase endpoints authenticate via the `databricks_auth` PostgreSQL extension, which validates standard OAuth JWTs.

**What works:**
- `WorkspaceClient().postgres.generate_database_credential()` -- returns a valid JWT from any compute type. Used by the setup job and the app at runtime.
- OAuth JWTs from `databricks auth token` (`eyJ...` prefix) -- used by `scripts/lakebase_setup.py` for standalone local setup.

**What does NOT work:**
- Internal cluster tokens (`apiToken()`, ~36 chars, opaque)
- PATs (`dapi...`, opaque tokens)
- Serverless `DATABRICKS_TOKEN` (opaque internal token)

---

## Quick Start

### 1. Create your local config

```bash
cp databricks.local.yml.example databricks.local.yml
```

Edit `databricks.local.yml` -- fill in your workspace host, CLI profile, catalog,
and warehouse ID. This file is gitignored.

### 2. Install prerequisites

**Databricks CLI >= 0.287.0** required (Lakebase resource support).

```bash
# macOS (download from GitHub releases):
curl -fsSL https://github.com/databricks/cli/releases/latest/download/databricks_cli_darwin_arm64.zip \
  -o /tmp/databricks_cli.zip && unzip -p /tmp/databricks_cli.zip databricks > ~/.local/bin/databricks \
  && chmod +x ~/.local/bin/databricks
```

### 3. Validate

```bash
databricks bundle validate --target my-workspace
```

### 4. Deploy

```bash
./deploy.sh --target my-workspace
```

> **Use `deploy.sh` instead of `databricks bundle deploy` directly.**
> The script handles the full end-to-end sequence:
> 1. Resolves bundle variables and generates `app/_bundle_config.py`
> 2. Runs `databricks bundle deploy` to provision all infrastructure
> 3. Resolves the Lakebase endpoint hostname (async, may take a few minutes)
> 4. Syncs `_bundle_config.py` to the workspace
> 5. Starts app compute if needed
> 6. Runs the full setup job (8 tasks: data gen -> pipeline -> features -> models -> serving -> permissions -> agent)
> 7. Deploys the app source code after infrastructure is ready

---

## Teardown

```bash
./destroy.sh --target my-workspace
```

Removes all deployed resources in the correct order:

| Asset | How |
|---|---|
| UC schema + all tables | Statement Execution API (`DROP SCHEMA ... CASCADE`) |
| Online Table | REST API |
| Serving endpoints (SARIMA, MC, Agent) | REST API |
| UC registered models (3) | REST API |
| Genie Space | REST API |
| Lakebase instance | Bundle destroy (async delete) |
| MLflow experiments | REST API |
| Databricks App | Bundle destroy |
| Setup + Refresh jobs | Bundle destroy |
| Declarative pipeline | Bundle destroy |
| Workspace bundle folder | `databricks workspace delete --recursive` |

> **Re-deploy after destroy:** Lakebase instance deletion is asynchronous. If you run
> `./deploy.sh` immediately after `./destroy.sh` you may see `Instance name is not unique`.
> Wait a few minutes and retry.
