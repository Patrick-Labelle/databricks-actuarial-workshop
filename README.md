# Databricks Actuarial Workshop

An end-to-end actuarial modeling solution on the Databricks Lakehouse — synthetic
Canadian insurance portfolio data, a full declarative medallion pipeline, statistical models that
**forecast claim frequency, model volatility, and quantify reserve risk** (SARIMAX/GARCH/Bootstrap Chain Ladder),
a tool-calling **AI chatbot agent** with MLflow tracing,
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

### Bootstrap Chain Ladder — Reserve Risk Quantification

**Business purpose:** Quantify reserve uncertainty and compute reserve risk capital (VaR 99.5%) for regulatory compliance (OSFI MCT, Solvency II) and IFRS 17 risk adjustment.

**Method:** Bootstrap Chain Ladder resamples scaled Pearson residuals from the fitted chain ladder model, creating thousands of pseudo-triangles. Each pseudo-triangle produces a different IBNR estimate, building the full predictive reserve distribution. Runs as Ray-distributed tasks across 4 workers.

**Output per product line:** best_estimate_M, var_95_M, var_99_M, var_995_M, cvar_99_M, reserve_risk_capital_M

**Scenarios:**
- **baseline** — Standard reserve development with calibrated parameters
- **adverse_development** — LDFs inflated by 20%, CVs x1.3
- **judicial_inflation** — Social inflation on Auto lines at long development lags
- **pandemic_tail** — Extended development periods (+6 months), CVs x1.4
- **superimposed_inflation** — Calendar-year trend (CPI + 3%) across all lines

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
| `predictions_frequency_forecast` | Predictions | Per-segment SARIMAX+GARCH forecasts (actuals + 12-month ahead) with confidence intervals and volatility |
| `predictions_bootstrap_reserves` | Predictions | Bootstrap Chain Ladder reserve results: best estimate IBNR, VaR 99%, VaR 99.5%, CVaR 99% |
| `predictions_reserve_evolution` | Predictions | 12-month reserve adequacy outlook |
| `predictions_reserve_scenarios` | Predictions | Pre-computed reserve scenarios (adverse development, judicial inflation, pandemic tail, superimposed inflation) |
| `predictions_runoff_projection` | Predictions | Multi-period surplus trajectory with regime-switching |
| `predictions_ldf_volatility` | Predictions | Development factor volatility per product line |

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
| Frequency Forecaster | `frequency_forecaster` | MLflow PyFunc | `horizon` (1-24 months) | Per-segment forecast_mean, forecast_lo95, forecast_hi95 by month |
| Bootstrap Reserve Simulator | `bootstrap_reserve_simulator` | MLflow PyFunc | Scenario parameters (means, CVs, n_replications, scenario) | best_estimate_M, var_99_M, var_995_M, cvar_99_M, reserve_risk_capital_M |

### Serving Endpoints

| Endpoint | Model | AI Gateway | Purpose |
|---|---|---|---|
| `actuarial-workshop-frequency-forecaster` | frequency_forecaster @Champion | Inference table, rate limits, usage tracking | On-Demand Forecast tab, chatbot tool |
| `actuarial-workshop-bootstrap-reserves` | bootstrap_reserve_simulator @Champion | Inference table, rate limits, usage tracking | Scenario Analysis tab, chatbot tool |

The chatbot agent runs **in-process** within the Streamlit app (not on a separate serving endpoint), using DatabricksOpenAI with tool calling. MLflow tracing captures all conversations to the `/Shared/actuarial-workshop-app-traces` experiment.

---

## Setup Job Tasks

The full setup job runs 7 tasks in sequence on a single classic ML cluster (Ray-on-Spark):

| # | Task | Notebook | Depends On | What It Does |
|---|---|---|---|---|
| 1 | `generate_source_data` | `src/01_data_pipeline.py` | — | Generate synthetic reserve CDC + claim incidents |
| 1b | `fetch_macro_data` | `scripts/fetch_macro_data.py` | — | Fetch StatCan unemployment, HPI, housing starts (parallel with task 1) |
| 2 | `run_dlt_pipeline` | SDP pipeline | 1, 1b | Bronze -> Silver (SCD2) -> Gold across all three data streams |
| 3 | `build_feature_store` | `src/02_feature_store.py` | 2 | UC Feature Store registration with point-in-time keys |
| 4 | `fit_statistical_models` | `src/03_classical_stats_at_scale.py` | 3 | SARIMAX/GARCH per segment + Ray-distributed Bootstrap Chain Ladder |
| 4b | `set_table_metadata` | `src/ops/set_table_metadata.py` | 4 | Add UC table and column descriptions for Genie / lineage |
| 5 | `prepare_app_infrastructure` | `src/04_model_serving.py` | 4b | Serving endpoints + AI Gateway, Online Table, Lakebase, Genie Space |
| 6 | `setup_app_dependencies` | `src/ops/app_setup.py` | 5 | UC grants + CAN_QUERY on endpoints + Genie space permissions for app SP |

A **Monthly Model Refresh** job (paused by default) re-runs tasks 1b -> 2 -> 4 on a schedule.

---

## Streamlit App (6 Tabs)

| Tab | Name | Description |
|---|---|---|
| 0 | Risk Assistant | Tool-calling chatbot agent (Genie, SQL, frequency forecaster, bootstrap reserves, Lakebase annotations) |
| 1 | Claims Forecast | Per-segment SARIMAX forecast with GARCH volatility bands and analyst annotations (Lakebase) |
| 2 | Reserve Adequacy | Bootstrap Chain Ladder IBNR distribution, VaR/CVaR metrics, MCT ratio |
| 3 | Scenario Analysis | Reserve scenarios (adverse development, judicial inflation, pandemic tail, superimposed inflation), reserve evolution, run-off projection, reserve triangle |
| 4 | On-Demand Forecast | Live SARIMAX+GARCH forecast via serving endpoint (1-24 month horizon) |
| 5 | Glossary | Comprehensive reference for all models, metrics, regulatory frameworks, and scenarios |

### Chatbot Tools

The Risk Assistant agent uses Llama 3.3 70B with 5 function-calling tools:

| Tool | Purpose | Backend |
|---|---|---|
| `ask_genie` | Natural-language data queries (primary data tool) | AI/BI Genie Space |
| `query_data` | Direct SQL fallback (read-only) | SQL Warehouse |
| `run_frequency_forecast` | On-demand claims forecast (1-24 months) | Frequency Forecaster serving endpoint |
| `run_bootstrap_reserve` | Custom reserve risk simulation | Bootstrap Reserve serving endpoint |
| `query_annotations` | Retrieve analyst scenario notes | Lakebase PostgreSQL |

---

## Deployed Resources

After running `./deploy.sh`, the following resources are created:

| Resource | Name / Location |
|---|---|
| Declarative Pipeline | `actuarial-workshop-medallion` |
| Setup Job | `Actuarial Workshop -- Full Setup` (7 tasks) |
| Monthly Refresh Job | `Actuarial Workshop -- Monthly Model Refresh` (paused) |
| Frequency Forecaster Endpoint | `actuarial-workshop-frequency-forecaster` |
| Bootstrap Reserve Endpoint | `actuarial-workshop-bootstrap-reserves` |
| Databricks App | `actuarial-workshop` |
| Lakebase Instance | `actuarial-workshop-lakebase` |
| Lakebase Database | `actuarial_workshop_db` |
| AI/BI Genie Space | `Actuarial Workshop -- Risk Assistant` (auto-created) |
| UC Models (2) | `frequency_forecaster`, `bootstrap_reserve_simulator` |
| Feature Table | `features_segment_monthly` (+ Online Table) |
| Delta Tables (19) | See [Tables](#tables) section above |

---

## Configuration Reference

All configurable values live in `databricks.yml` under `variables:`.

| Variable | Description | Default |
|---|---|---|
| `catalog` | UC catalog name (must exist) | `my_catalog` |
| `schema` | UC schema (created if missing) | `actuarial_workshop` |
| `endpoint_name` | Frequency forecaster serving endpoint name | `actuarial-workshop-frequency-forecaster` |
| `mc_endpoint_name` | Bootstrap reserve serving endpoint name | `actuarial-workshop-bootstrap-reserves` |
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
│       ├── set_table_metadata.py # UC table and column descriptions for Genie / lineage
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
│   └── chatbot/                 # Agent module (agent.py, tools.py)
└── README.md
```

---

## Notebooks

### Job Pipeline (`src/`)

| # | Notebook | Job Task | Key Concepts |
|---|---|---|---|
| 1 | `01_data_pipeline.py` | `generate_source_data` | SDP, Medallion, SCD Type 2; 3 data streams (reserves, claims, macro) |
| 2 | `02_feature_store.py` | `build_feature_store` | UC Feature Store, point-in-time joins, Online Table |
| 3 | `03_classical_stats_at_scale.py` | `fit_statistical_models` | SARIMAX/GARCH, Chain Ladder, Bootstrap Chain Ladder, Ray-on-Spark, MLflow |
| 4 | `04_model_serving.py` | `prepare_app_infrastructure` | Serving endpoints + AI Gateway, Online Table, Lakebase, Genie Space |

### Interactive Only (`interactive_workshop/`)

| # | Notebook | Key Concepts |
|---|---|---|
| 2 | `02_performance_at_scale.py` | 4 ETL approaches timed, run-many-models, for-loop anti-patterns |
| 6 | `06_dabs_cicd.py` | DABs CI/CD, Azure DevOps |
| 7 | `07_databricks_apps.py` | Databricks Apps, Lakebase |

Interactive versions of job pipeline modules (01, 03, 04, 05) are also in `interactive_workshop/` with `display()` calls and learning notes.

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

## Distributed Bootstrap (Ray-on-Spark)

All modules run on a **classic DBR 16.4 ML job cluster** with Ray-on-Spark for distributed Bootstrap Chain Ladder simulation.

> **Requirement:** Classic compute (not serverless-only workspaces).

The simulation uses `@ray.remote(num_cpus=1)` tasks dispatched across 4 workers. Bootstrap replications are distributed across workers for parallel execution.

CPU-only cluster; classic ML runtime required for Ray-on-Spark.

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
> 6. Runs the full setup job (7 tasks: data gen -> pipeline -> features -> models -> metadata -> serving -> permissions)
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
| Serving endpoints (Frequency Forecaster, Bootstrap Reserves) | REST API |
| UC registered models (2) | REST API |
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
