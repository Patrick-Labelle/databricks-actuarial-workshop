# Databricks Actuarial Workshop

A complete 1-day actuarial workshop — demo notebooks, DLT pipeline, statistical
models, model serving, and a Streamlit dashboard — packaged as a single
**Databricks Asset Bundle** for one-command deployment to any workspace.

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
│   └── lakebase_setup.py     # Local Lakebase setup script (runs from deploy.sh using CLI OAuth JWT)
├── src/
│   ├── ops/
│   │   ├── app_setup.py      # UC grants + model serving CAN_QUERY (runs as job task)
│   │   └── cleanup.py        # Post-workshop teardown notebook
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

### 2. Install prerequisites

`deploy.sh` requires **Databricks CLI >= 0.287.0** (Lakebase resource support) and
**`psycopg2-binary`** (local Lakebase database setup):

```bash
pip install psycopg2-binary
```

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
> 4. Runs the full setup job, which seeds the data, trains the SARIMA model, and grants the
>    app's service principal permissions on all UC catalog/schema/tables, Lakebase, and the model-serving endpoint.
> 5. Deploys the app source code only after all permissions are in place, so the app starts without permission errors.

Between steps 2 and 4, `deploy.sh` runs `scripts/lakebase_setup.py` locally to
provision the PostgreSQL database, create the `scenario_annotations` table, and
grant the app service principal access. This runs locally (not on a Databricks
cluster) because Lakebase Autoscaling's `databricks_auth` extension only accepts
standard OAuth JWTs issued by the workspace OIDC endpoint — internal cluster tokens
are not accepted (see [Lakebase authentication](#lakebase-authentication) below).

The setup job runs the following modules in sequence:
1. Generate synthetic policy CDC data
2. Run the DLT pipeline (Bronze → Silver → Gold)
3. Build rolling features (Module 2)
4. Register the Feature Store + Online Table (Module 3)
5. Fit SARIMA / GARCH / Monte Carlo models (Module 4)
6. Register model to UC Registry + create Model Serving endpoint (Module 5)
7. **App setup** — grant UC `USE CATALOG`, `USE SCHEMA`, `SELECT` on all tables, and `CAN_QUERY` on the serving endpoint to the app service principal

---

## Ray-Enabled Deployment (Module 4 on GPU ML cluster)

Module 4 (`04_classical_stats_at_scale.py`) includes Ray-on-Spark code for GPU-accelerated
Monte Carlo simulation. By default, the setup job skips the Ray section (`run_ray: "skip"`)
so the full pipeline runs on serverless. To demonstrate Ray + GPU, use the Ray-enabled variant:

```bash
./deploy-ray.sh        # deploys to e2-demo-ray target (~20 min, includes ~5-10 min cluster spin-up)
./destroy-ray.sh       # full teardown for the e2-demo-ray deployment
```

`deploy-ray.sh` and `destroy-ray.sh` are thin wrappers around `deploy.sh` / `destroy.sh` that pass
`--target e2-demo-ray`. The `e2-demo-ray` target (defined in `databricks.local.yml`) inherits all
workspace settings from `e2-demo` and adds one override: Task 5 (`fit_statistical_models`) is moved
from serverless to a **DBR 17.3-gpu-ml** job cluster (1 × `g4dn.xlarge`, NVIDIA Tesla T4),
and `run_ray: "auto"` + `job_mode: "false"` are passed to the notebook.

### GPU Monte Carlo (t-Copula, SARIMA-driven VaR evolution + stress scenarios)

Module 4 uses Ray-on-Spark with PyTorch for GPU-accelerated Monte Carlo — all 64 tasks
dispatched simultaneously in a single Ray batch (~640M total paths):

1. **Baseline** (static means): 4 tasks × 10M = **40M paths** → `monte_carlo_results` (used by Module 5/app)
2. **12-month VaR evolution** (SARIMA-driven means): 12 months × 4 tasks × 10M = **480M paths**
   → `portfolio_risk_timeline` showing how capital requirements evolve along the SARIMA forecast path
3. **Stress scenarios** (3 × 4 tasks × 10M = **120M paths**) → `stress_test_scenarios`:
   - `cat_event`: 1-in-250yr catastrophe — Property 3.5×, Auto 1.8×, Liability 1.4×, stressed ρ,
     and a **Poisson(λ=0.05) jump process** for discrete large-loss events
   - `stress_corr`: systemic/contagion risk — correlations spike to 0.65–0.75
   - `inflation_shock`: +30% loss-cost inflation across all lines, +15% CV uncertainty

- **t-Copula (df=4)** captures tail dependence between Property, Auto, and Liability lines
- **100% GPU path:** `torch.distributions.StudentT.cdf()` (CUDA-native kernel, PyTorch 2.7+)
  replaces `scipy.special.betainc` — all computation stays on GPU with zero CPU-GPU transfers
- Ray workers use `num_gpus=0.25` + `num_cpus=0.5` fractional allocation: **4 concurrent tasks
  per T4 = full GPU utilization**; 64 total tasks complete in ~90 seconds for 640M total paths
- Cluster: **CPU-only driver** (`m5.xlarge`) + 1 × `g4dn.xlarge` GPU worker — Ray compute runs
  entirely on the worker; the driver only coordinates task dispatch
- Regional claims breakdown written to `regional_claims_forecast` (SARIMA aggregated by region × month)

### Ray CPU Reservation

`setup_ray_cluster` is called with `num_cpus_worker_node=2` on the 4-vCPU `g4dn.xlarge` worker,
leaving 2 vCPUs free for Spark. `num_cpus=0.5` per Ray task allows 4 tasks to run concurrently
(4 × 0.5 = 2.0 CPUs ≤ 2 available), keeping the T4 fully utilized. After Monte Carlo completes
(all 64 tasks collected), `shutdown_ray_cluster()` is called before any Spark write to release
all resources immediately.

> **Note:** Classic/GPU ML clusters are not available on serverless-only workspaces (e.g. FEVM).
> The Ray variant targets the `e2-demo-field-eng` workspace.

---

## Targets

| Target | Workspace | Notes |
|--------|-----------|-------|
| `dev` (default) | Your configured profile | Adds `[dev <user>]` prefix to resource names |
| _(your target)_ | Defined in `databricks.local.yml` | gitignored; auto-merged by bundle |
| `e2-demo-ray` | e2-demo-field-eng | Same as e2-demo + Task 5 on DBR 17.3-gpu-ml cluster (1×g4dn.xlarge, Tesla T4, Ray+GPU) |

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

`deploy.sh` runs `databricks bundle validate --output json` to resolve all variables,
writes `app/_bundle_config.py` with the literal values, runs `databricks bundle deploy`,
starts app compute if needed, waits for the setup job to complete (which grants all SP
permissions), then deploys the app source code last so it starts with full permissions.
`app/_bundle_config.py` is gitignored but force-included in the bundle sync via `databricks.yml`.

---

## Workshop Modules

| Module | Notebook | Key Concepts |
|--------|----------|-------------|
| 1 | `01_dlt_pipeline_and_jobs.py` | DLT, Medallion, SCD Type 2, Jobs API |
| 2 | `02_spark_vs_ray.py` | Pandas API on Spark, applyInPandas, Ray |
| 3 | `03_feature_store.py` | UC Feature Store, point-in-time joins, Online Tables |
| 4 | `04_classical_stats_at_scale.py` | SARIMA/GARCH, t-Copula Monte Carlo (SARIMA-driven VaR evolution), Ray+GPU, MLflow |
| 5 | `05_mlflow_uc_serving.py` | PyFunc, UC Model Registry, Model Serving |
| 6 | `06_dabs_cicd.py` | DABs CI/CD, Azure DevOps |
| Bonus | `07_databricks_apps.py` | Databricks Apps, Lakebase |

All notebooks accept `catalog`, `schema`, and `endpoint_name` as widget
parameters (passed automatically by the bundle jobs).

---

## Running Modules Interactively

Notebooks can also be run interactively in the workspace. Clone or upload the
`src/` directory, then run each notebook in order. The widget defaults at
the top of each notebook allow standalone execution without the bundle.

---

## Deployed Resources

After running `./deploy.sh`, the following resources will be created:

| Resource | Name |
|----------|------|
| Lakebase instance | `actuarial-workshop-lakebase` (provisioned by bundle deploy) |
| Lakebase database | `actuarial_workshop_db` (created by `scripts/lakebase_setup.py` in `deploy.sh`) |
| DLT Pipeline | `actuarial-workshop-medallion` |
| Setup Job | `Actuarial Workshop — Full Setup` |
| Monthly Refresh Job | `Actuarial Workshop — Monthly Model Refresh` |
| Model Serving Endpoint | value of `endpoint_name` variable |
| Databricks App | `actuarial-workshop` |
| Feature Table | `{catalog}.{schema}.segment_monthly_features` |
| UC Model | `{catalog}.{schema}.sarima_claims_forecaster` |
| Monte Carlo results | `{catalog}.{schema}.monte_carlo_results` (40M baseline paths) |
| VaR timeline | `{catalog}.{schema}.portfolio_risk_timeline` (12-month SARIMA-driven) |
| Stress scenarios | `{catalog}.{schema}.stress_test_scenarios` (CAT, systemic risk, inflation) |
| Regional forecast | `{catalog}.{schema}.regional_claims_forecast` (SARIMA aggregated by region × month) |
| Monte Carlo endpoint | `actuarial-workshop-monte-carlo` (AI Gateway enabled: usage tracking, inference table, rate limits) |

---

## Lakebase Authentication

Lakebase Autoscaling endpoints authenticate via the `databricks_auth` PostgreSQL
extension. This extension validates **standard OAuth JWTs** (RFC 7519 tokens issued
by the workspace OIDC endpoint), which it receives as the PostgreSQL password. It
validates the token's signature against the workspace OIDC public keys and maps the
`sub` claim to the PostgreSQL username.

**What works:** OAuth JWTs from `databricks auth token` (eyJ... prefix, ~850 chars,
includes `sub`, `iss`, `aud`, `scope` claims).

**What does NOT work:**
- Internal Databricks cluster tokens (`apiToken()`, ~36 chars, opaque, no JWT claims)
- PATs (`dapi...`, also opaque tokens rejected by JWT validation)
- Serverless `DATABRICKS_TOKEN` (also an opaque internal token)

This is why `scripts/lakebase_setup.py` runs **locally from `deploy.sh`** using the
CLI's OAuth JWT, rather than from a Databricks job task. The app itself uses
`generate_database_credential` (which does produce a valid OAuth credential for
service principals) to authenticate at runtime.

---

## Teardown

```bash
./destroy.sh --target my-workspace
```

`destroy.sh` is a single-command full teardown. It removes all workshop assets in the correct order:

| Asset | How |
|---|---|
| UC schema + all tables | Statement Execution API (`DROP SCHEMA … CASCADE`) — auto-starts warehouse |
| Online Table | REST API |
| Model Serving endpoint | REST API |
| UC registered model + versions | REST API |
| MLflow experiments | REST API |
| Databricks App | Bundle destroy |
| Lakebase instance (+ all databases) | Bundle destroy (async delete) |
| Setup + Monthly Refresh jobs | Bundle destroy |
| DLT pipeline | Bundle destroy |
| Workspace bundle folder | `databricks workspace delete --recursive` |

> **Note:** `destroy.sh` only removes resources deployed in the current target.
> Jobs or pipelines left over from previous deployments (different targets or re-deploys)
> must be deleted manually from the workspace UI or via the REST API.

> **Lakebase re-deploy:** Lakebase instance deletion is asynchronous. If you run
> `./deploy.sh` immediately after `./destroy.sh` you may see `Instance name is not unique`.
> Wait a few minutes and retry — the deletion propagates in the background.
