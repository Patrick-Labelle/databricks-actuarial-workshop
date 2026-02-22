# Actuarial Workshop Demo — Statistical Modeling at Scale on Databricks

A complete 1-day workshop for actuaries and data scientists, demonstrating how to move from classical statistical workflows into production-grade pipelines on Databricks.

---

## Workshop Overview

| Module | Title | Key Concepts |
|--------|-------|-------------|
| 1 | DLT Pipeline + Databricks Workflows | Medallion architecture, Delta Live Tables, SCD Type 2, Job DAGs |
| 2 | Spark vs Ray: Choosing the Right Parallelism | Pandas API on Spark, `applyInPandas`, Ray tasks, when to use each |
| 3 | Feature Store + Point-in-Time Joins | UC Feature Store, data leakage prevention, Online Tables |
| 4 | Classical Stats at Scale | SARIMA/GARCH per-segment, Monte Carlo with Ray, MLflow logging |
| 5 | MLflow + UC Model Registry + Serving | PyFunc wrappers, Champion alias, Model Serving REST API |
| 6 | CI/CD with DABs + Azure DevOps | Asset Bundles, bundle.yml, 3-stage DevOps pipeline |
| Bonus | Databricks Apps + Lakebase | Streamlit on serverless, Postgres-integrated transactional state |

---

## Live Deployed Resources

These resources are deployed in the **e2-demo-field-eng** workspace and are ready to use for demos.

### Workspace
| Resource | Link |
|----------|------|
| Databricks Workspace | [e2-demo-field-eng.cloud.databricks.com](https://e2-demo-field-eng.cloud.databricks.com) |
| Unity Catalog Schema | [patrick_labelle.actuarial_workshop](https://e2-demo-field-eng.cloud.databricks.com/explore/data/patrick_labelle/actuarial_workshop) |
| Workshop Notebooks | [/Users/patrick.labelle@databricks.com/actuarial-workshop/](https://e2-demo-field-eng.cloud.databricks.com/browse/folders/1444828305810485) |

### Models & Serving
| Resource | Link |
|----------|------|
| UC Model Registry | [patrick_labelle.actuarial_workshop.sarima_claims_forecaster](https://e2-demo-field-eng.cloud.databricks.com/explore/data/patrick_labelle/actuarial_workshop/sarima_claims_forecaster) |
| Model Serving Endpoint | [actuarial-workshop-sarima-forecaster](https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/actuarial-workshop-sarima-forecaster/invocations) |
| MLflow Experiments | [ML Experiments](https://e2-demo-field-eng.cloud.databricks.com/ml/experiments) (search `actuarial_workshop`) |

### Pipelines & Jobs
| Resource | Link |
|----------|------|
| DLT Pipeline | [Pipelines UI](https://e2-demo-field-eng.cloud.databricks.com/pipelines) (search `Actuarial Workshop`) |
| Databricks Job | [Jobs UI](https://e2-demo-field-eng.cloud.databricks.com/jobs) (search `Actuarial Workshop`) |
| Online Table | [patrick_labelle.actuarial_workshop.segment_features_online](https://e2-demo-field-eng.cloud.databricks.com/explore/data/patrick_labelle/actuarial_workshop/segment_features_online) |

### Databricks App
| Resource | Details |
|----------|---------|
| Live App URL | [actuarial-workshop-1444828305810485.aws.databricksapps.com](https://actuarial-workshop-1444828305810485.aws.databricksapps.com) |
| App Source | `/Users/patrick.labelle/actuarial-workshop-app/` |
| Lakebase Instance | `actuarial-workshop-db` (PostgreSQL, database `actuarial_workshop_db`) |

---

## Prerequisites

### Compute

All notebooks are written for **Serverless Compute** (recommended) or a cluster with:
- Databricks Runtime 15.4 LTS ML or newer
- Access to Unity Catalog

For Modules 2 and 4 (Ray), the cluster needs Ray pre-installed (included in ML Runtime) or serverless with Ray support. Notebooks include single-node fallback if Ray is unavailable.

### Python Libraries

The following libraries are used. On serverless, install via notebook-scoped libraries or `%pip install` at the top of each notebook if not already available:

| Library | Used In | Install |
|---------|---------|---------|
| `statsmodels>=0.14` | Modules 2, 4, 5 | Pre-installed on ML Runtime |
| `arch>=7.0` | Module 4 | `%pip install arch` |
| `mlflow>=2.14` | Modules 4, 5 | Pre-installed on ML Runtime |
| `databricks-feature-engineering` | Module 3 | Pre-installed on DBR 14+ |
| `ray[default]` | Modules 2, 4 | Pre-installed on ML Runtime |

### Unity Catalog Permissions

The notebooks write to `<CATALOG>.<SCHEMA>` (default: `patrick_labelle.actuarial_workshop`). The running user needs:

- `USE CATALOG` on the target catalog
- `CREATE SCHEMA` on the target catalog (to create `actuarial_workshop`)
- `ALL PRIVILEGES` on the target schema (tables, views, feature tables)
- `CREATE MODEL` privilege (for Module 5, UC Model Registry)
- `CAN_USE` on a SQL Warehouse (for Online Table creation in Module 3)

For the **Model Serving endpoint** (Module 5) and **Online Table** (Module 3), the user needs permission to create serving endpoints and online tables in the workspace (typically Databricks admin or granted via workspace settings).

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
| `bronze_policy_cdc` | Module 1 (DLT) | Raw CDC policy events, append-only |
| `silver_policies` | Module 1 (DLT) | SCD Type 2 current + history (Apply Changes) |
| `gold_segment_monthly_stats` | Module 1 (DLT) | Aggregated monthly loss stats by segment |
| `silver_rolling_features` | Module 2 | Rolling means, volatility features per segment |
| `segment_monthly_features` | Module 3 | UC Feature Table (with timestamp key) |
| `sarima_forecasts` | Module 4 | SARIMA forecasts + confidence intervals for all 20 segments |
| `garch_volatility_metrics` | Module 4 | GARCH volatility estimates per segment |
| `monte_carlo_results` | Module 4 | Monte Carlo simulation paths, VaR, CVaR |
| `claims_time_series` | Bonus | Monthly claims time series used by the Databricks App |

### MLflow

| Experiment | Created By |
|------------|------------|
| `actuarial_workshop_claims_sarima` | Module 4 |
| `actuarial_workshop_sarima_claims_forecaster` | Module 5 |

### UC Model Registry

| Model | Created By | Alias |
|-------|------------|-------|
| `<CATALOG>.<SCHEMA>.sarima_claims_forecaster` | Module 5 | `@Champion` |

### Serving + Feature Infrastructure

| Asset | Created By |
|-------|------------|
| Model Serving endpoint `actuarial-workshop-sarima-forecaster` | Module 5 |
| Online Table `<CATALOG>.<SCHEMA>.segment_features_online` | Module 3 |
| DLT Pipeline `Actuarial Workshop — DLT Pipeline` | Module 1 (manual step) |
| Databricks Job `Actuarial Workshop — Orchestration Demo` | Module 1 |

### Bonus: Databricks App + Lakebase

| Asset | Details |
|-------|---------|
| Databricks App `actuarial-workshop` | Live at [actuarial-workshop-1444828305810485.aws.databricksapps.com](https://actuarial-workshop-1444828305810485.aws.databricksapps.com) |
| Lakebase instance `actuarial-workshop-db` | Managed PostgreSQL, database `actuarial_workshop_db`, table `annotations` |

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

For the Bonus app, update `app.yaml` with your Lakebase `PGHOST`, `PGDATABASE`, and `DATABRICKS_HOST` values.

### Other Things to Customize

- **Synthetic data parameters** (Modules 1, 2, 4): Product lines, regions, date ranges, and loss ratios are all configurable at the top of each notebook's data generation cell.
- **SARIMA parameters** (Module 4): `ORDER` and `SEASONAL_ORDER` are set conservatively for speed; adjust for better fit.
- **Monte Carlo paths** (Module 4): `N_PATHS = 10_000` by default; increase for higher-fidelity VaR estimates.
- **Ray cluster size** (Modules 2, 4): `num_cpus` and `num_gpus` in `setup_ray_cluster()` defaults to serverless; increase for larger workloads.

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

### Bonus App: Lakebase `PGHOST not set`
The app reads `PGHOST` from environment variables. Ensure `app.yaml` has the `PGHOST` env var set to your Lakebase instance hostname. Lakebase env vars are **not** automatically injected without an explicit `resources:` declaration in `app.yaml`.

---

## Cleanup

A cleanup notebook (`00_cleanup.py`) is provided to remove all workshop assets. Review it carefully before running — it permanently deletes all tables, the UC model, the serving endpoint, MLflow experiments, and the DLT pipeline.

```
00_cleanup.py — removes all assets (run post-workshop only)
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

**Key actuarial concept:** Medallion architecture ensures raw policy data is always preserved, SCD Type 2 enables "as-of" queries, and DLT expectations enforce data quality without boilerplate code.

---

### Module 2: Spark vs Ray (`02_spark_vs_ray.py`)

**What it demonstrates:**
- Pandas API on Spark (`pyspark.pandas`) — write pandas code, run on a distributed cluster
- `applyInPandas` for per-segment OLS trend fitting (data-parallel)
- Ray tasks for ARIMA grid search (task-parallel, embarrassingly parallel)
- When to use each: data-parallel vs task-parallel parallelism

**Key actuarial concept:** The choice of parallelism framework matters. Data transformations scale with Spark; per-model fitting (SARIMA per segment) scales with `applyInPandas`; independent simulations scale with Ray.

---

### Module 3: Feature Store + Point-in-Time Joins (`03_feature_store.py`)

**What it demonstrates:**
- UC Feature Store table registration with `FeatureEngineeringClient`
- Point-in-time joins via `timestamp_lookup_key` — no future leakage
- Online Table for low-latency feature serving at inference
- Feature lineage tracking in Unity Catalog

**Key actuarial concept:** Point-in-time joins are the production-grade equivalent of "as-of" pricing — every training observation uses only features available at the observation date.

---

### Module 4: Classical Stats at Scale (`04_classical_stats_at_scale.py`)

**What it demonstrates:**
- SARIMA fitting across 20 segments using `applyInPandas` (statsmodels SARIMAX)
- GARCH volatility modeling with the `arch` library
- Monte Carlo portfolio simulation with Ray (`@ray.remote`)
- VaR and CVaR computation, results written to Delta + logged to MLflow

**Key actuarial concept:** Classical statistical models (SARIMA, GARCH) remain the right tool for actuarial time series. Databricks enables fitting them at scale across all segments in minutes instead of hours.

---

### Module 5: MLflow + UC Model Registry + Serving (`05_mlflow_uc_serving.py`)

**What it demonstrates:**
- `mlflow.pyfunc.PythonModel` wrapper for SARIMA — standardized inference interface
- Unity Catalog Model Registry with semantic versioning and `@Champion` alias
- Model Serving endpoint creation via REST API
- Calling the endpoint with a standard REST request
- Monitoring via `system.serving.served_entities_request_logs`

**Key actuarial concept:** MLflow PyFunc wraps any statistical model (SARIMA, GARCH, survival) in a standard interface, enabling deployment, A/B testing, and rollback without changing scoring code.

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

### Bonus: Databricks Apps + Lakebase (`07_databricks_apps_bonus.py`)

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
- Lakebase: direct `psycopg2` connection with `PGPASSWORD` set from the forwarded `X-Forwarded-Access-Token` header; `PGUSER` extracted from the JWT `sub` claim (user email)

**Live URL:** [actuarial-workshop-1444828305810485.aws.databricksapps.com](https://actuarial-workshop-1444828305810485.aws.databricksapps.com)

**Key actuarial concept:** Actuarial review workflows — annotating forecasts, overriding assumptions, approving model outputs — benefit from interactive applications that sit inside the same governance boundary as the data and models. Databricks Apps inherits UC permissions and SSO automatically; no separate auth layer needed.
