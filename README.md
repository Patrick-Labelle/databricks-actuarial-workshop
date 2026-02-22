# Databricks Actuarial Workshop

A complete 1-day actuarial workshop — demo notebooks, DLT pipeline, statistical
models, model serving, and a Streamlit dashboard — packaged as a single
**Databricks Asset Bundle** for one-command deployment to any workspace.

## Repository structure

```
.
├── databricks.yml       # Bundle config — all variables here
├── resources/
│   ├── pipeline.yml     # DLT pipeline (Bronze → Silver → Gold)
│   ├── jobs.yml         # Orchestration jobs (setup + monthly refresh)
│   └── app.yml          # Databricks App resource
├── demos/               # Workshop notebooks (Modules 1–7)
├── app/
│   ├── app.py           # Streamlit application
│   ├── app.yaml         # App config (env vars substituted from bundle)
│   └── requirements.txt
└── README.md
```

## Quick Start — Deploy to Your Workspace

### 1. Configure variables

Edit `databricks.yml` — set the `catalog` default (or add a named target):

```yaml
variables:
  catalog:
    default: your_catalog   # ← change this
  schema:
    default: actuarial_workshop
  # ... (endpoint_name, warehouse_id, pg_host, pg_database — see file)
```

Or add a named target under `targets:` with workspace-specific overrides.

### 2. Validate

```bash
databricks bundle validate
```

### 3. Deploy

```bash
databricks bundle deploy
```

This uploads the notebooks, creates the DLT pipeline, orchestration jobs, and
updates the Databricks App.

### 4. Run the setup job

```bash
databricks bundle run actuarial_workshop_setup
```

This runs all modules in sequence:
1. Generate synthetic policy CDC data
2. Run the DLT pipeline (Bronze → Silver → Gold)
3. Build rolling features (Module 2)
4. Register the Feature Store + Online Table (Module 3)
5. Fit SARIMA / GARCH / Monte Carlo models (Module 4)
6. Register model to UC Registry + create Model Serving endpoint (Module 5)

### 5. Start the app

Start the Databricks App from the Apps UI or:
```bash
databricks apps start actuarial-workshop
```

---

## Targets

| Target | Workspace | Notes |
|--------|-----------|-------|
| `dev` (default) | Your configured profile | Adds `[dev <user>]` prefix to resource names |
| _(add your own)_ | your-workspace | See `databricks.local.yml.example` |

Example for a custom workspace:
```bash
databricks bundle deploy --target dev \
  --var catalog=my_catalog \
  --var schema=actuarial_workshop \
  --var warehouse_id=abc123
```

---

## Configuration Reference

All configurable values live in `databricks.yml` under `variables:`.

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | UC catalog name (must exist) | `my_catalog` |
| `schema` | UC schema (created if missing) | `actuarial_workshop` |
| `endpoint_name` | Model Serving endpoint name | `actuarial-workshop-sarima-forecaster` |
| `warehouse_id` | SQL Warehouse ID for the app | _(empty)_ |
| `pg_host` | Lakebase hostname (empty = disable annotations) | _(empty)_ |
| `pg_database` | Lakebase database name | `actuarial_workshop_db` |
| `notification_email` | Email for job failure alerts | _(empty)_ |

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

After running the Full Setup job, the following resources will be created in your workspace:

| Resource | Name |
|----------|------|
| DLT Pipeline | `actuarial-workshop-medallion` |
| Setup Job | `Actuarial Workshop — Full Setup` |
| Monthly Refresh Job | `Actuarial Workshop — Monthly Model Refresh` |
| Model Serving Endpoint | value of `endpoint_name` variable (default: `actuarial-workshop-sarima-forecaster`) |
| Databricks App | `actuarial-workshop` |
| Feature Table | `{catalog}.{schema}.segment_monthly_features` |
| UC Model | `{catalog}.{schema}.sarima_claims_forecaster` |
