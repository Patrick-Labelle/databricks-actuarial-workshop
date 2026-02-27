# Databricks Actuarial Demo

An end-to-end actuarial modeling solution on the Databricks Lakehouse — synthetic
insurance portfolio data, a full DLT medallion pipeline, and statistical models that
**forecast claims, model volatility, and simulate portfolio loss** (SARIMAX/GARCH/Monte Carlo),
plus MLflow model registry, model serving, and a Streamlit risk dashboard —
packaged as a single **Databricks Asset Bundle** for one-command deployment.

## What's in the box

- **DLT medallion pipeline** — three data streams (reserve development CDC, claims events, Statistics Canada macro indicators) through Bronze → Silver (SCD Type 2) → Gold
- **SARIMAX forecasting** — forecasts **monthly claim frequency and severity by product line and province** for reserving, pricing, and capital planning; 40 segments (4 product lines × 10 Canadian provinces), 72-month history, real macro exogenous variables (unemployment, HPI, housing starts) from Statistics Canada
- **GARCH volatility modeling** — models **time-varying uncertainty in claim outcomes by segment** for risk capital and reinsurance; per-segment conditional variance estimation via the `arch` library
- **t-Copula Monte Carlo** — simulates **portfolio loss distribution** for VaR/CVaR, 12-month capital-at-risk evolution, and catastrophe stress scenarios; 640M-path distributed simulation (Ray-on-Spark + NumPy/SciPy)
- **MLflow + UC Model Registry** — PyFunc-wrapped models, versioned, with `@Champion` alias and AI Gateway-enabled serving endpoints
- **Unity Catalog Feature Store** — point-in-time joins for leakage-free training sets, Online Table for real-time feature lookup
- **Streamlit risk dashboard** — claims forecasts, capital requirements, on-demand scoring, stress testing, and catastrophe scenario simulation backed by Lakebase (managed PostgreSQL) for analyst annotations

---

## Repository structure

```
.
├── databricks.yml            # Bundle config — variables, sync, includes
├── databricks.local.yml.example  # Template for your workspace-specific target
├── deploy.sh                 # End-to-end deploy: bundle deploy → Lakebase setup → setup job → app deploy
├── deploy-ray.sh             # Thin wrapper: deploys to the Ray-enabled target
├── destroy.sh                # Full teardown: workspace assets + bundle destroy
├── resources/
│   ├── pipeline.yml          # DLT pipeline (Bronze → Silver → Gold)
│   ├── jobs.yml              # Orchestration jobs (setup + monthly refresh)
│   ├── app.yml               # Databricks App resource + SP authorizations
│   └── lakebase.yml          # Lakebase (managed PostgreSQL) instance
├── scripts/
│   ├── fetch_macro_data.py   # Fetch StatCan unemployment, HPI, housing starts → macro_indicators_raw
│   └── lakebase_setup.py     # Standalone Lakebase setup utility (not called by deploy.sh; DB setup is in Module 5)
├── src/
│   ├── ops/
│   │   ├── app_setup.py      # UC grants + model serving CAN_QUERY (runs as job task)
│   │   └── cleanup.py        # Schema and resource cleanup notebook
│   └── 01–07_*.py            # Demo notebooks (Modules 1–6 + Bonus)
├── app/
│   ├── app.py                # Streamlit application
│   ├── app.yaml              # App command + valueFrom resource injections
│   └── requirements.txt
└── README.md
```

---

## Quick Start — Deploy to Your Workspace

### 1. Create your local config

```bash
cp databricks.local.yml.example databricks.local.yml
```

Edit `databricks.local.yml` — fill in your workspace host, CLI profile, catalog,
and warehouse ID. This file is gitignored and auto-merged by the bundle, so your
workspace-specific values never touch version control.

### 2. Install prerequisites

`deploy.sh` requires **Databricks CLI >= 0.287.0** (Lakebase resource support).

To upgrade the CLI if needed:
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
> The script handles the full end-to-end deployment in a single command:
> 1. Resolves bundle variable values via `bundle validate` and generates `app/_bundle_config.py`
>    (necessary because `app/app.yaml` is uploaded as source code and does not receive DAB variable substitution).
> 2. Runs `databricks bundle deploy` to provision all bundle-managed infrastructure.
> 3. Starts the app compute if needed (clears a bundle-internal deployment lock that occurs on fresh installs).
> 4. Runs the full setup job, which seeds the data, trains the models, and grants the
>    app's service principal permissions on all UC catalog/schema/tables, Lakebase, and the serving endpoints.
> 5. Deploys the app source code only after all permissions are in place, so the app starts without permission errors.

The setup job runs the following tasks in order:
1. **Generate source data** — synthetic reserve development CDC + claim incidents → `reserve_development_raw`, `claims_events_raw`
2. **Fetch macro data** (parallel with task 1) — Statistics Canada unemployment, HPI, housing starts → `macro_indicators_raw`
3. **Run DLT pipeline** (after tasks 1 + 2) — Bronze → Silver (SCD Type 2) → Gold across three data streams:
   - Reserves: `bronze_reserve_cdc` → `silver_reserves` → `gold_reserve_triangle` (loss development triangle)
   - Claims: `bronze_claims` → `gold_claims_monthly` (40 segments × 72 months)
   - Macro: `bronze_macro_indicators` → `silver_macro_indicators` (SCD2) → `gold_macro_features`
4. **Register Feature Store** — leakage-free training set assembly; reads `silver_rolling_features` from DLT; features feed Module 4 SARIMAX as exogenous variables
5. **Fit SARIMAX / GARCH / Monte Carlo + register models** — fits claim-forecast and volatility models by segment, then portfolio loss simulation; reads `gold_claims_monthly` + Feature Store features + `gold_reserve_triangle`; GARCH-derived CVs feed Monte Carlo; reserve validation compares forecasts to actual development; registers both models (SARIMA + MC) to UC with `@Champion` alias
6. **Prepare app infrastructure** — creates both serving endpoints + AI Gateway, Online Table for low-latency feature lookup, Lakebase PostgreSQL database setup (DB, extension, table, SP role + grants) using `generate_database_credential()`
7. **App setup** — grant UC `USE CATALOG`, `USE SCHEMA`, `SELECT` on all tables, and `CAN_QUERY` on both serving endpoints to the app service principal

---

## Statistics Canada Macro Data Integration

The SARIMAX models in `04_classical_stats_at_scale.py` **forecast monthly claim counts and severity by product line and province** for reserving, pricing, and capital planning. They use real macroeconomic data from Statistics Canada as exogenous variables, flowing through the same DLT medallion pipeline as the claims data:

```
scripts/fetch_macro_data.py  →  macro_indicators_raw
                                   ↓ (DLT streaming)
                            bronze_macro_indicators
                                   ↓ (DLT apply_changes, SCD Type 2)
                            silver_macro_indicators     ← tracks StatCan revisions
                                   ↓ (DLT materialized view)
                            gold_macro_features         ← joined to claims before SARIMAX fitting
```

**Three indicators fetched (public CSV API, no authentication required):**

| StatCan Table | Indicator | Role in SARIMAX |
|---|---|---|
| `14-10-0017-01` | `unemployment_rate` | Leading indicator for auto/liability claims frequency |
| `18-10-0205-01` | `hpi_index` → `hpi_growth` | Housing market proxy for homeowners/commercial property claims |
| `34-10-0158-01` | `housing_starts` | New-exposure leading indicator |

**Coverage:** 10 provinces × full history (1976–2026 depending on series) = ~6,000 rows per fetch.

MLflow logs `avg_mape_baseline_pct`, `avg_mape_sarimax_pct`, and `avg_mape_improvement_pct`
for direct before/after comparison in the experiment UI.

**Network-restricted workspaces:** `scripts/fetch_macro_data.py` is fault-tolerant.
If `www150.statcan.gc.ca` is unreachable, the script creates an empty `macro_indicators_raw`
table and exits cleanly. The DLT pipeline runs on the empty table, and the model falls
back to baseline SARIMA. The full pipeline completes successfully in both cases.

---

## Ray-Enabled Deployment (Distributed Monte Carlo)

`04_classical_stats_at_scale.py` includes Ray-on-Spark code for distributed
Monte Carlo simulation. By default, the setup job skips the Ray section (`run_ray: "skip"`)
so the full pipeline runs on serverless. To run with Ray, use the Ray-enabled variant:

```bash
./deploy-ray.sh        # deploys to the Ray-enabled target (~20 min, includes ~5-10 min cluster spin-up)
```

The Ray target overrides Task 5 (`fit_statistical_models`) from serverless to a
**DBR 16.4-cpu-ml** job cluster (4 × `m5.2xlarge` workers, 8 vCPU / 32 GB each),
and passes `run_ray: "auto"` to the notebook.

### Distributed Monte Carlo details (t-Copula, 640M paths)

The simulation produces **capital-at-risk and tail-risk metrics** (VaR/CVaR) and **stress-test outcomes** (catastrophe, correlation spike, inflation) for regulatory and internal risk management. All 64 Ray tasks are dispatched simultaneously in a single batch:

1. **Baseline** (static means): 4 tasks × 10M = **40M paths** → `monte_carlo_results`
2. **12-month VaR evolution** (SARIMA-driven means): 12 months × 4 tasks × 10M = **480M paths**
   → `portfolio_risk_timeline`
3. **Stress scenarios** (3 × 4 tasks × 10M = **120M paths**) → `stress_test_scenarios`:
   - `cat_event`: 1-in-250yr catastrophe — Property 3.5×, Auto 1.8×, Liability 1.4×, stressed ρ,
     Poisson(λ=0.05) jump process
   - `stress_corr`: systemic/contagion risk — correlations spike to 0.65–0.75
   - `inflation_shock`: +30% loss-cost inflation, +15% CV uncertainty

- **t-Copula (df=4)** captures tail dependence between Property, Auto, and Liability lines
- **NumPy/SciPy path:** `scipy.stats.t.cdf()` + `norm.ppf()` for copula and marginals
- `num_cpus=1` per task: **24 concurrent tasks** across 4 workers (4 × 6 Ray CPUs); 640M paths in ~2-3 minutes
- CPU-only driver + 4 × CPU workers — Ray compute runs on all workers in parallel

> **Note:** Classic ML clusters are not available on serverless-only workspaces.

---

## Configuration Reference

All configurable values live in `databricks.yml` under `variables:`.

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | UC catalog name (must exist) | `my_catalog` |
| `schema` | UC schema (created if missing) | `actuarial_workshop` |
| `endpoint_name` | SARIMA Model Serving endpoint name | `actuarial-workshop-sarima-forecaster` |
| `mc_endpoint_name` | Monte Carlo Model Serving endpoint name | `actuarial-workshop-monte-carlo` |
| `warehouse_id` | SQL Warehouse ID for the app | _(empty)_ |
| `pg_database` | Lakebase PostgreSQL database name | `actuarial_workshop_db` |
| `notification_email` | Email for job failure alerts | _(empty)_ |

The Lakebase instance hostname (`PGHOST`) is injected into the app at runtime
via the `valueFrom: database` resource reference in `app/app.yaml` — no manual
configuration is needed.

---

## Variable Injection — How the App Gets Its Config

`app/app.yaml` is uploaded to the workspace as source code and does not receive
bundle variable substitution, so `${var.*}` references cannot be used there.
The app's runtime configuration is provided through three channels:

| Variable | Mechanism | Where set |
|----------|-----------|-----------|
| `CATALOG`, `SCHEMA`, `PGDATABASE`, `ENDPOINT_NAME` | `app/_bundle_config.py` (generated by `deploy.sh`) | Inlined in `deploy.sh` |
| `PGHOST` | `valueFrom: database` in `app/app.yaml` | Injected at runtime from Lakebase resource |
| `DATABRICKS_WAREHOUSE_ID` | `valueFrom: sql-warehouse` in `app/app.yaml` | Injected at runtime from SQL warehouse resource |
| `DATABRICKS_HOST` | Auto-injected by Databricks Apps runtime | — |
| `DATABRICKS_CLIENT_ID` / `_SECRET` | Auto-injected by Databricks Apps runtime (SP credentials) | — |

`deploy.sh` resolves all variables, writes `app/_bundle_config.py`, deploys the bundle,
waits for the setup job to complete (which grants all SP permissions), then deploys the
app source code last so it starts with full permissions. `app/_bundle_config.py` is
gitignored but force-included in the bundle sync via `databricks.yml`.

---

## Notebooks

| # | Notebook | Key concepts |
|---|----------|-------------|
| 1 | `01_dlt_pipeline_and_jobs.py` | DLT, Medallion, SCD Type 2, Jobs API; 3 data streams (reserves, claims, macro) |
| 2 | `02_performance_at_scale.py` | Performance at Scale — 4 ETL approaches timed, run-many-models, for-loop anti-patterns |
| 3 | `03_feature_store.py` | UC Feature Store, point-in-time joins, leakage-free training sets |
| 4 | `04_classical_stats_at_scale.py` | SARIMAX/GARCH (**claim forecasts and volatility by segment**), t-Copula Monte Carlo (**portfolio loss distribution, VaR, stress tests**), reads DLT gold + StatCan macro, Ray distributed, MLflow, registers both models to UC |
| 5 | `05_model_serving.py` | App Infrastructure — serving endpoints + AI Gateway, Online Table, Lakebase setup, monitoring |
| 6 | `06_dabs_cicd.py` | DABs CI/CD, Azure DevOps |
| Bonus | `07_databricks_apps.py` | Databricks Apps, Lakebase |

All notebooks accept `catalog`, `schema`, and `endpoint_name` as widget parameters
(passed automatically by the bundle jobs). They can also be run interactively by
cloning or uploading the `src/` directory — the widget defaults at the top of each
notebook allow standalone execution without the bundle.

---

## Deployed Resources

After running `./deploy.sh`, the following resources are created:

| Resource | Name |
|----------|------|
| Lakebase instance | `actuarial-workshop-lakebase` |
| Lakebase database | `actuarial_workshop_db` |
| DLT Pipeline | `actuarial-workshop-medallion` |
| Setup Job | `Actuarial Workshop — Full Setup` |
| Monthly Refresh Job | `Actuarial Workshop — Monthly Model Refresh` |
| SARIMA Serving Endpoint | value of `endpoint_name` variable |
| Monte Carlo Serving Endpoint | value of `mc_endpoint_name` variable |
| Databricks App | `actuarial-workshop` |
| Feature Table | `{catalog}.{schema}.segment_monthly_features` |
| UC Model (SARIMA) | `{catalog}.{schema}.sarima_claims_forecaster` |
| UC Model (Monte Carlo) | `{catalog}.{schema}.monte_carlo_portfolio` |
| Monte Carlo results | `{catalog}.{schema}.monte_carlo_results` (40M baseline paths) |
| VaR timeline | `{catalog}.{schema}.portfolio_risk_timeline` (12-month SARIMA-driven) |
| Stress scenarios | `{catalog}.{schema}.stress_test_scenarios` (CAT, systemic risk, inflation) |
| Regional forecast | `{catalog}.{schema}.regional_claims_forecast` |
| Claims landing zone | `{catalog}.{schema}.claims_events_raw` (~1.5M synthetic claim incidents) |
| Claims DLT bronze/gold | `bronze_claims` / `gold_claims_monthly` (40 segments × 72 months) |
| Macro landing zone | `{catalog}.{schema}.macro_indicators_raw` |
| Macro DLT bronze/silver/gold | `bronze_macro_indicators` / `silver_macro_indicators` (SCD2) / `gold_macro_features` |
| SARIMAX forecasts | `{catalog}.{schema}.sarima_forecasts` (actuals + 12-month forecasts, mape_baseline, mape_sarimax) |

---

## Lakebase Authentication

Lakebase Autoscaling endpoints authenticate via the `databricks_auth` PostgreSQL
extension. This extension validates **standard OAuth JWTs** (RFC 7519 tokens issued
by the workspace OIDC endpoint), which it receives as the PostgreSQL password.

**What works:**
- `WorkspaceClient().postgres.generate_database_credential()` — returns a valid JWT
  from any compute type (serverless, classic, interactive). Used by Module 5 (setup job)
  and by the app at runtime.
- OAuth JWTs from `databricks auth token` (eyJ... prefix, ~850 chars) — used by
  `scripts/lakebase_setup.py` for standalone local setup.

**What does NOT work:**
- Internal Databricks cluster tokens (`apiToken()`, ~36 chars, opaque, no JWT claims)
- PATs (`dapi...`, also opaque tokens rejected by JWT validation)
- Serverless `DATABRICKS_TOKEN` (also an opaque internal token)

Module 5 (`prepare_app_infrastructure` task) handles all Lakebase database setup
using `generate_database_credential()`, so no local psycopg2 install is needed.
`scripts/lakebase_setup.py` is retained as a standalone utility for manual setup.

---

## Teardown

```bash
./destroy.sh --target my-workspace
```

Removes all deployed resources in the correct order:

| Asset | How |
|---|---|
| UC schema + all tables | Statement Execution API (`DROP SCHEMA … CASCADE`) |
| Online Table | REST API |
| SARIMA + Monte Carlo serving endpoints | REST API |
| UC registered models + versions | REST API |
| MLflow experiments | REST API |
| Databricks App | Bundle destroy |
| Lakebase instance (+ all databases) | Bundle destroy (async delete) |
| Setup + Monthly Refresh jobs | Bundle destroy |
| DLT pipeline | Bundle destroy |
| Workspace bundle folder | `databricks workspace delete --recursive` |

> **Note:** `destroy.sh` only removes resources deployed in the current target.
> Resources from previous deployments to different targets must be removed manually.

> **Re-deploy after destroy:** Lakebase instance deletion is asynchronous. If you run
> `./deploy.sh` immediately after `./destroy.sh` you may see `Instance name is not unique`.
> Wait a few minutes and retry.
