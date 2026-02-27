# Actuarial Workshop Demo — Statistical Modeling at Scale on Databricks

A complete 1-day workshop for actuaries and data scientists, demonstrating how to move from classical statistical workflows into production-grade pipelines on Databricks.

---

## Workshop Overview

| Module | Title | Key Concepts |
|--------|-------|-------------|
| 1 | DLT Pipeline + Databricks Workflows | Medallion architecture, Delta Live Tables, SCD Type 2, Job DAGs |
| 2 | Performance at Scale: Choosing the Right Spark Pattern | Four ETL approaches timed, run-many-models, for-loop anti-patterns, decision framework |
| 3 | Feature Store + Point-in-Time Joins | UC Feature Store, data leakage prevention, point-in-time joins |
| 4 | Classical Stats at Scale | SARIMA/GARCH per-segment, Ray-distributed Monte Carlo, MLflow logging, registers both models (SARIMA + MC) to UC Model Registry with Champion alias |
| 5 | App Infrastructure | Serving endpoints + AI Gateway, Online Table, Lakebase setup, demo all app services |
| 6 | CI/CD with DABs + Azure DevOps | Asset Bundles, bundle.yml, 3-stage DevOps pipeline |
| Bonus | Databricks Apps + Lakebase | Streamlit on serverless, Postgres-integrated transactional state |

---

## Deployed Resources

After running the bundle and Full Setup job, the following resources will exist in your workspace:

### Data & Models
| Resource | Name |
|----------|------|
| Unity Catalog Schema | `{catalog}.{schema}` (configured via bundle variables) |
| DLT Pipeline | `actuarial-workshop-medallion` |
| UC Model Registry | `{catalog}.{schema}.sarima_claims_forecaster` |
| Model Serving Endpoint | value of `endpoint_name` variable |
| Feature Table | `{catalog}.{schema}.segment_monthly_features` |
| Online Table | `{catalog}.{schema}.segment_features_online` |

### Jobs & App
| Resource | Name |
|----------|------|
| Setup Job | `Actuarial Workshop — Full Setup` |
| Monthly Refresh Job | `Actuarial Workshop — Monthly Model Refresh` |
| Databricks App | `actuarial-workshop` |

---

## Prerequisites

### Compute

All notebooks are written for **Serverless Compute** (recommended) or a cluster with:
- Databricks Runtime 15.4 LTS ML or newer
- Access to Unity Catalog

For Module 4 (Ray), the cluster needs Ray pre-installed (included in ML Runtime) or serverless with Ray support. Notebooks include single-node fallback if Ray is unavailable.

### Python Libraries

The following libraries are used. On serverless, install via notebook-scoped libraries or `%pip install` at the top of each notebook if not already available:

| Library | Used In | Install |
|---------|---------|---------|
| `statsmodels>=0.14` | Module 4 | Pre-installed on ML Runtime |
| `arch>=7.0` | Module 4 | `%pip install arch` |
| `mlflow>=2.14` | Modules 4, 5 (MlflowClient) | Pre-installed on ML Runtime |
| `databricks-feature-engineering` | Module 3 | Pre-installed on DBR 14+ |
| `ray[default]` | Module 4 | Pre-installed on ML Runtime |

### Unity Catalog Permissions

The notebooks write to `<CATALOG>.<SCHEMA>` (default: `my_catalog.actuarial_workshop`). The running user needs:

- `USE CATALOG` on the target catalog
- `CREATE SCHEMA` on the target catalog (to create `actuarial_workshop`)
- `ALL PRIVILEGES` on the target schema (tables, views, feature tables)
- `CREATE MODEL` privilege (for Module 4, UC Model Registry)
- `CAN_USE` on a SQL Warehouse (for Online Table creation in Module 5)

For the **Model Serving endpoints** and **Online Table** (both Module 5), the user needs permission to create serving endpoints and online tables in the workspace (typically Databricks admin or granted via workspace settings).

---

## How to Run

### Recommended Order

Run modules in sequence — each module builds on assets from the previous one. Each module also includes an **inline fallback** that regenerates the required data independently, so you can run any module standalone if needed.

```
Module 1 → Module 2 → Module 3 → Module 4 → Module 5 → Module 6 → Bonus
```

### Running Individual Modules

Each notebook can be run independently:
1. Open the notebook in the Databricks workspace
2. Attach to a serverless cluster or ML Runtime cluster
3. Run all cells (Cell > Run All)

Modules that depend on upstream tables will detect if the upstream data is missing and fall back to inline synthetic data generation.

### Module 1 — DLT Pipeline

Module 1 is split into two parts:
1. **DLT Pipeline** (Cells under "Part A"): Attach this notebook as the source of a Delta Live Tables pipeline. Create a DLT pipeline in the Workflows UI → point it at this notebook → click Start.
2. **Standalone data generation** (Cells under "Part B"): Run the notebook directly to generate synthetic data and the Gold table without DLT.
3. **Job creation** (Cells under "Part C"): Creates a Databricks Job with a multi-task DAG programmatically.

The `IN_DLT` flag at the top of the notebook automatically detects whether it's running inside DLT or as a regular notebook, enabling both modes from a single file.

---

## Asset Inventory

After running all modules, the following assets will exist in your workspace:

### Unity Catalog Tables (`<CATALOG>.<SCHEMA>`)

| Table | Created By | Description |
|-------|------------|-------------|
| `reserve_development_raw` | Module 1 (generator) | Synthetic reserve CDC source |
| `bronze_reserve_cdc` | Module 1 (DLT) | Raw reserve development CDC, append-only |
| `silver_reserves` | Module 1 (DLT) | SCD Type 2 reserve history (Apply Changes) |
| `gold_reserve_triangle` | Module 1 (DLT) | Loss development triangle (accident month × dev lag) |
| `bronze_claims` | Module 1 (DLT) | Raw claim events, append-only |
| `gold_claims_monthly` | Module 1 (DLT) | Segment × month claims aggregate (40 segments × 72 months) |
| `silver_rolling_features` | Module 1 (DLT) | Rolling means, volatility features per segment |
| `segment_monthly_features` | Module 3 | UC Feature Table (feeds Module 4 SARIMAX as exog vars) |
| `sarima_forecasts` | Module 4 | SARIMAX forecasts + confidence intervals for all 40 segments |
| `garch_volatility` | Module 4 | GARCH(1,1) volatility estimates → feeds MC CVs |
| `monte_carlo_results` | Module 4 | Monte Carlo simulation (GARCH-calibrated CVs), VaR, CVaR |
| `reserve_validation` | Module 4 | Reserve adequacy: SARIMA forecasts vs. actual development |
| `regional_claims_forecast` | Module 4 | Regional breakdown of projected claims |
| `portfolio_risk_timeline` | Module 4 | 12-month SARIMA-driven VaR evolution |
| `stress_test_scenarios` | Module 4 | Stress test results (CAT, systemic, inflation) |

### MLflow

| Experiment | Created By |
|------------|------------|
| `actuarial_workshop_claims_sarima` | Module 4 |
| `actuarial_workshop_sarima_claims_forecaster` | Module 4 |
| `actuarial_workshop_monte_carlo_portfolio` | Module 4 |

### UC Model Registry

| Model | Created By | Alias |
|-------|------------|-------|
| `<CATALOG>.<SCHEMA>.sarima_claims_forecaster` | Module 4 | `@Champion` |
| `<CATALOG>.<SCHEMA>.monte_carlo_portfolio` | Module 4 | `@Champion` |

### Serving + Feature Infrastructure

| Asset | Created By |
|-------|------------|
| Model Serving endpoint `actuarial-workshop-sarima-forecaster` | Module 5 |
| Model Serving endpoint `actuarial-workshop-monte-carlo` | Module 5 |
| Online Table `<CATALOG>.<SCHEMA>.segment_features_online` | Module 5 |
| DLT Pipeline `Actuarial Workshop — DLT Pipeline` | Module 1 (manual step) |
| Databricks Job `Actuarial Workshop — Orchestration Demo` | Module 1 |

### Bonus: Databricks App + Lakebase

| Asset | Details |
|-------|---------|
| Databricks App `actuarial-workshop` | Available at your workspace Apps URL after deployment |
| Lakebase instance `actuarial-workshop-lakebase` | Managed PostgreSQL, database `actuarial_workshop_db`, table `public.scenario_annotations` |

---

## Adapting to Your Environment

To run this in a different catalog, schema, or workspace, update the following constants at the top of each notebook:

```python
CATALOG = "your_catalog"   # Change to your Unity Catalog name
SCHEMA  = "your_schema"    # Change to your schema name
```

For Module 5 (Model Serving), also update:
```python
ENDPOINT_NAME = "your-endpoint-name"
```

The `WORKSPACE_URL` is derived automatically from Spark config (`spark.conf.get("spark.databricks.workspaceUrl")`), so no changes are needed there.

For the Bonus app, all connection values (`PGHOST`, `DATABRICKS_HOST`, `CATALOG`, `SCHEMA`, `PGDATABASE`) are injected automatically when deployed via `./deploy.sh`. No manual `app.yaml` edits are needed.

### Other Things to Customize

- **Synthetic data parameters** (Modules 1, 2, 4): Product lines, regions, date ranges, and loss ratios are all configurable at the top of each notebook's data generation cell.
- **SARIMA parameters** (Module 4): `ORDER` and `SEASONAL_ORDER` are set conservatively for speed; adjust for better fit.
- **Monte Carlo paths** (Module 4): `N_PATHS = 10_000` by default; increase for higher-fidelity VaR estimates.
- **Ray cluster size** (Module 4): `num_cpus_worker_node` in `setup_ray_cluster()` defaults to serverless; increase for larger workloads.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'dlt'`
This is expected when running Module 1 outside of a DLT pipeline. The `IN_DLT` guard at the top of the notebook handles this — DLT-specific cells are skipped, and standalone data generation runs normally.

### `AnalysisException: Table not found`
Each module has a fallback to generate data inline. If upstream tables are missing, the fallback will trigger automatically. Check the output for `"not found — regenerating"` messages.

### Feature Store / Online Table creation fails
Ensure your user has the `CAN_USE` privilege on a running SQL Warehouse. Online Table creation requires an active warehouse.

### Model Serving endpoint times out
Model Serving endpoints can take 5–10 minutes to become ready after creation. The notebook polls for readiness, but if it times out, wait and re-run the scoring cell.

### Ray `setup_ray_cluster` fails
Ray cluster setup may fail on some serverless configurations. All Ray cells have a fallback that runs the same computation with Python multiprocessing or sequential execution.

### Serverless: `AnalysisException: .cache() not supported`
On Serverless compute, `.cache()` is not supported for DataFrames in some contexts. The notebooks avoid `.cache()` and use `.persist()` with `StorageLevel.MEMORY_AND_DISK` where caching is needed, or simply re-read from Delta.

### Serverless: `pyspark.pandas` cross-DataFrame operations
If you see `ValueError: Cannot combine the series or dataframe because it comes from a different dataframe` in Module 2, add the following before the operation:
```python
import pyspark.pandas as ps
ps.set_option("compute.ops_on_diff_frames", True)
```

### `Py4JJavaError` on DLT import in non-DLT context
Module 1's `IN_DLT` guard catches the `Py4JJavaError` that can occur when importing the `dlt` module outside of a DLT pipeline context. If you see this error in other modules, the guard is working as intended — DLT cells will be skipped.

### Bonus App: `PGHOST` / `CATALOG` / `SCHEMA` / `ENDPOINT_NAME` not set
These values are injected at deploy time by `./deploy.sh` (which generates `app/_bundle_config.py`) and at runtime via the `resources: database:` declaration in `app/app.yaml` for `PGHOST`. Always deploy using `./deploy.sh --target <target>` rather than `databricks bundle deploy` directly to ensure `_bundle_config.py` is generated with the correct values.

### Bonus App: `Could not save annotation` / Lakebase connection errors
1. **SP auth failing** — Ensure `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` are available in the Apps runtime (they are auto-injected if the app resource has an associated service principal in `resources/app.yml`).
2. **`relation "scenario_annotations" does not exist`** — The table must be created in the `public` schema. Run `src/ops/app_setup.py` as Task 7 of the setup job to create the table and grant the required PostgreSQL privileges to the app SP.
3. **`permission denied for sequence`** — The setup notebook grants `USAGE ON SEQUENCE public.scenario_annotations_id_seq`. If skipped, re-run the setup notebook or grant manually.

---

## Cleanup

A cleanup notebook (`ops/cleanup.py`) is provided to remove all workshop assets. Review it carefully before running — it permanently deletes all tables, the UC model, the serving endpoint, MLflow experiments, and the DLT pipeline.

```
ops/cleanup.py — removes all assets (run post-workshop only)
```

---

## Module-by-Module Reference

### Module 1: DLT Pipeline + Databricks Workflows (`01_dlt_pipeline_and_jobs.py`)

**What it demonstrates:**
- Delta Live Tables declarative ETL: Bronze → Silver → Gold
- `dlt.apply_changes()` for SCD Type 2 (full history + current state)
- DLT expectations (`@dlt.expect_or_drop`, `@dlt.expect_or_warn`)
- Multi-task Databricks Job created programmatically via REST API
- Task value passing between Job tasks

**Key actuarial concept:** Medallion architecture ensures raw reserve data is always preserved, SCD Type 2 tracks how reserve estimates develop over time (a core actuarial workflow), and DLT expectations enforce data quality without boilerplate code.

---

### Module 2: Performance at Scale (`02_performance_at_scale.py`)

**What it demonstrates:**
- Four approaches to rolling-window ETL: plain pandas, Native Spark, Pandas API on Spark, `applyInPandas` — timed side-by-side on small (2,880 rows) and scaled (300K rows) datasets
- Three approaches to run-many-models (OLS per segment): for-loop, `applyInPandas`, Spark built-in `regr_slope`/`regr_intercept` — showing parallelism trade-offs
- For-loop anti-patterns: why `withColumn` in a loop is O(N²) and how `select()` + list comprehension fixes it
- Decision framework: when to use each pattern (Native Spark, `applyInPandas`, Pandas API, Ray, `select()`)

**Key actuarial concept:** Before building models, choose the right Spark pattern. Data transforms scale with native Spark window functions; per-group model fitting scales with `applyInPandas`; simple aggregates use Spark built-ins; and multi-column generation must avoid the `withColumn` loop trap. This module produces no persistent outputs — it's a standalone performance guide.

---

### Module 3: Feature Store + Point-in-Time Joins (`03_feature_store.py`)

**What it demonstrates:**
- UC Feature Store table registration with `FeatureEngineeringClient`
- Point-in-time joins via `timestamp_lookup_key` — no future leakage
- Feature lineage tracking in Unity Catalog

**Key actuarial concept:** Point-in-time joins are the production-grade equivalent of "as-of" pricing — every training observation uses only features available at the observation date.

---

### Module 4: Classical Stats at Scale (`04_classical_stats_at_scale.py`)

**What it demonstrates:**
- SARIMAX fitting across 40 segments using `applyInPandas` with Feature Store exogenous variables and StatCan macro data
- GARCH volatility modeling with the `arch` library — conditional volatilities feed Monte Carlo CVs
- Ray-distributed Monte Carlo portfolio simulation (`@ray.remote`, NumPy/SciPy) using GARCH-calibrated CVs
- Reserve validation: SARIMA forecasts vs. actual development from `gold_reserve_triangle`
- VaR and CVaR computation, results written to Delta + logged to MLflow

**Key actuarial concept:** Classical statistical models (SARIMAX, GARCH) remain the right tool for actuarial time series. Feature Store provides leakage-free exogenous variables, GARCH-derived volatilities calibrate the Monte Carlo simulation, and reserve validation closes the loop between forecasts and actual development. Both production models (SARIMA + Monte Carlo) are registered to UC with `@Champion` alias, ready for serving.

---

### Module 5: App Infrastructure (`05_model_serving.py`)

**What it demonstrates:**
- Creating both SARIMA and Monte Carlo serving endpoints via REST API
- AI Gateway configuration via separate `PUT /serving-endpoints/{name}/ai-gateway` call
- Online Table creation for low-latency feature serving
- Lakebase PostgreSQL setup using `generate_database_credential()` from the Databricks SDK
- Demo calls to every service the Streamlit app uses (endpoints, DBSQL, Lakebase)
- Monitoring via `system.serving.served_entities_request_logs`

**Key actuarial concept:** Before an actuarial review app can launch, every data service it depends on must be provisioned and tested. This module creates the complete integration layer — model endpoints, feature serving, transactional state — so the app starts cleanly with full permissions.

---

### Module 6: CI/CD with DABs + Azure DevOps (`06_dabs_cicd.py`)

**What it demonstrates:**
- Databricks Asset Bundle structure (`databricks.yml`, `resources/`)
- Multi-target bundles (dev → staging → prod)
- `databricks bundle validate / deploy / run` CLI workflow
- Azure DevOps 3-stage pipeline: PR validation → Staging → Production
- Workload Identity Federation for secure CI/CD authentication

**Key actuarial concept:** Every model update, pipeline change, and feature definition should flow through version control with automated validation and staged promotion — the same governance actuaries apply to reserving methodologies.

---

### Bonus: Databricks Apps + Lakebase (`07_databricks_apps.py`)

**What it demonstrates:**
- Streamlit application running on Databricks Apps (serverless, UC-integrated, SSO)
- Lakebase (managed Postgres on Databricks) for transactional analyst annotations
- Reading from Delta tables and calling the live Model Serving endpoint
- Three-tab actuarial review UI: SARIMA Forecasts, Monte Carlo Risk, and Model Serving
- Databricks SDK (`WorkspaceClient`) for SQL execution and model serving — no PAT tokens
- JWT-based user identity extraction for Lakebase row-level attribution

**App features:**
- **Tab 1 — SARIMA Forecasts**: Per-segment historical loss ratios with SARIMA forecast overlay and 95% confidence intervals. Forecast cutoff marker, history statistics, and raw data expander. Analyst can write notes that are persisted to Lakebase.
- **Tab 2 — Monte Carlo Risk**: Portfolio-level VaR (99%), CVaR (99%), and Expected Loss from Monte Carlo simulation. Bar chart of risk metrics, loss distribution histogram, and Solvency II context.
- **Tab 3 — Model Serving**: Live call to the deployed `actuarial-workshop-sarima-forecaster` endpoint via `WorkspaceClient.serving_endpoints.query()`. Renders forecast results with confidence intervals and an explainer on the PyFunc wrapper pattern.
- **Sidebar**: Source data context (Delta tables, Lakebase, model endpoint), pipeline reference, and analyst annotation history.

**Authentication architecture:**
- SQL queries: `WorkspaceClient` with `statement_execution.execute_statement()` — uses the forwarded user token automatically in Databricks Apps context
- Model Serving: `WorkspaceClient.serving_endpoints.query()` — same SDK, same token
- Lakebase: direct `psycopg2` connection using the app's **service principal** identity. The app exchanges `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` (auto-injected by Databricks Apps) for a short-lived OIDC token via `POST /oidc/v1/token`, then connects as the SP. The analyst's email is still extracted from the forwarded `X-Forwarded-Access-Token` JWT (`sub` claim) for display and row attribution — only the Lakebase *connection* uses the SP credential. Tokens are cached with a 60-second pre-expiry buffer.

**URL:** Available at your workspace Apps URL after deployment (`databricks apps get actuarial-workshop`).

**Key actuarial concept:** Actuarial review workflows — annotating forecasts, overriding assumptions, approving model outputs — benefit from interactive applications that sit inside the same governance boundary as the data and models. Databricks Apps inherits UC permissions and SSO automatically; no separate auth layer needed.
