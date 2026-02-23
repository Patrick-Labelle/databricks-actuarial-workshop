# Databricks Actuarial Workshop

A complete 1-day actuarial workshop — demo notebooks, DLT pipeline, statistical
models, model serving, and a Streamlit dashboard — packaged as a single
**Databricks Asset Bundle** for one-command deployment to any workspace.

## Repository structure

```
.
├── databricks.yml            # Bundle config — variables, sync, includes
├── databricks.local.yml.example  # Template for your workspace-specific target
├── deploy.sh                 # Deploy wrapper (generates _bundle_config.py first)
├── scripts/
│   └── gen_bundle_config.py  # Writes app/_bundle_config.py with substituted vars
├── resources/
│   ├── pipeline.yml          # DLT pipeline (Bronze → Silver → Gold)
│   ├── jobs.yml              # Orchestration jobs (setup + monthly refresh)
│   ├── app.yml               # Databricks App resource + SP authorizations
│   └── lakebase.yml          # Lakebase (managed PostgreSQL) instance
├── demos/
│   ├── 00_app_setup.py       # App setup: Lakebase DB, table, UC + PG grants
│   ├── 00_cleanup.py         # Post-workshop teardown notebook
│   └── 01–07_*.py            # Workshop notebooks (Modules 1–6 + Bonus)
├── app/
│   ├── app.py                # Streamlit application
│   ├── app.yaml              # App command + valueFrom resource injections
│   └── requirements.txt
└── README.md
```

## Quick Start — Deploy to Your Workspace

### 1. Create your local config

```bash
cp databricks.local.yml.example databricks.local.yml
```

Edit `databricks.local.yml` — fill in your workspace host, CLI profile, catalog,
and warehouse ID. This file is gitignored and auto-merged by the bundle, so your
sensitive values never touch version control.

### 2. Validate

```bash
databricks bundle validate --target my-workspace
```

### 3. Deploy

```bash
./deploy.sh --target my-workspace
```

> **Use `deploy.sh` instead of `databricks bundle deploy` directly.**
> The script resolves bundle variable values via `bundle validate`, generates
> `app/_bundle_config.py` with the actual catalog/schema names, then runs the
> deploy. This is necessary because `app/app.yaml` is uploaded as source code
> and does not receive DAB variable substitution at deploy time.

### 4. Run the setup job

```bash
databricks bundle run actuarial_workshop_setup --target my-workspace
```

This runs all modules in sequence:
1. Generate synthetic policy CDC data
2. Run the DLT pipeline (Bronze → Silver → Gold)
3. Build rolling features (Module 2)
4. Register the Feature Store + Online Table (Module 3)
5. Fit SARIMA / GARCH / Monte Carlo models (Module 4)
6. Register model to UC Registry + create Model Serving endpoint (Module 5)
7. **App setup** — create Lakebase DB, `scenario_annotations` table, grant UC permissions and PostgreSQL privileges to the app service principal

### 5. Start the app

```bash
databricks bundle run actuarial_workshop_app --target my-workspace
```

---

## Targets

| Target | Workspace | Notes |
|--------|-----------|-------|
| `dev` (default) | Your configured profile | Adds `[dev <user>]` prefix to resource names |
| _(your target)_ | Defined in `databricks.local.yml` | gitignored; auto-merged by bundle |

The bundle's `include:` glob (`databricks.local*.yml`) automatically merges your
local target definitions when present, and silently skips if the file is absent.

---

## Configuration Reference

All configurable values live in `databricks.yml` under `variables:`.

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | UC catalog name (must exist) | `my_catalog` |
| `schema` | UC schema (created if missing) | `actuarial_workshop` |
| `endpoint_name` | Model Serving endpoint name | `actuarial-workshop-sarima-forecaster` |
| `warehouse_id` | SQL Warehouse ID for the app | _(empty)_ |
| `pg_database` | Lakebase PostgreSQL database name | `actuarial_workshop_db` |
| `notification_email` | Email for job failure alerts | _(empty)_ |

The Lakebase instance hostname (`PGHOST`) is injected into the app at runtime
via the `valueFrom: database` resource reference in `app/app.yaml` — no manual
configuration is needed.

---

## Workshop Modules

| Module | Notebook | Key Concepts |
|--------|----------|-------------|
| 1 | `01_dlt_pipeline_and_jobs.py` | DLT, Medallion, SCD Type 2, Jobs API |
| 2 | `02_spark_vs_ray.py` | Pandas API on Spark, applyInPandas, Ray |
| 3 | `03_feature_store.py` | UC Feature Store, point-in-time joins, Online Tables |
| 4 | `04_classical_stats_at_scale.py` | SARIMA/GARCH, Monte Carlo, MLflow |
| 5 | `05_mlflow_uc_serving.py` | PyFunc, UC Model Registry, Model Serving |
| 6 | `06_dabs_cicd.py` | DABs CI/CD, Azure DevOps |
| Bonus | `07_databricks_apps_bonus.py` | Databricks Apps, Lakebase |

All notebooks accept `catalog`, `schema`, and `endpoint_name` as widget
parameters (passed automatically by the bundle jobs).

---

## Running Modules Interactively

Notebooks can also be run interactively in the workspace. Clone or upload the
`demos/` directory, then run each notebook in order. The widget defaults at
the top of each notebook allow standalone execution without the bundle.

---

## Deployed Resources

After running the Full Setup job, the following resources will be created:

| Resource | Name |
|----------|------|
| Lakebase instance | `actuarial-workshop-lakebase` (provisioned by bundle deploy) |
| Lakebase database | `actuarial_workshop_db` (created by setup job Task 7) |
| DLT Pipeline | `actuarial-workshop-medallion` |
| Setup Job | `Actuarial Workshop — Full Setup` |
| Monthly Refresh Job | `Actuarial Workshop — Monthly Model Refresh` |
| Model Serving Endpoint | value of `endpoint_name` variable |
| Databricks App | `actuarial-workshop` |
| Feature Table | `{catalog}.{schema}.segment_monthly_features` |
| UC Model | `{catalog}.{schema}.sarima_claims_forecaster` |

---

## Teardown

To remove all workshop assets after the session:

```bash
# 1. Run the cleanup notebook to drop UC assets, MLflow experiments, and Lakebase data
databricks bundle run actuarial_workshop_cleanup --target my-workspace
# (or run demos/00_cleanup.py interactively in the workspace)

# 2. Destroy all bundle-managed infrastructure (app, jobs, pipeline, Lakebase instance)
databricks bundle destroy --target my-workspace
```
