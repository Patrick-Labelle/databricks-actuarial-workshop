# Job Pipeline Source — `src/`

These notebooks are executed by the **Full Setup** and **Monthly Refresh** jobs defined in `resources/jobs.yml`. They contain job-only logic — no `display()` calls or interactive exploration.

For interactive versions with visualizations and learning notes, see [`interactive_workshop/`](../interactive_workshop/).

---

## Job Pipeline Modules

| Module | File | Job Task | Description |
|--------|------|----------|-------------|
| 1 | `01_data_pipeline.py` | `generate_source_data` | Synthetic data generation + SDP pipeline definitions (Bronze → Silver → Gold) |
| 2 | `02_feature_store.py` | `build_feature_store` | UC Feature Store registration with point-in-time joins |
| 3 | `03_classical_stats_at_scale.py` | `fit_statistical_models` | SARIMAX + GARCH(1,1) per segment, Chain Ladder, Bootstrap Chain Ladder (Ray), model registration to UC, MLflow logging |
| 4 | `04_model_serving.py` | `prepare_app_infrastructure` | Serving endpoints + AI Gateway, Online Table, Lakebase setup, Genie Space |

### Operations (`ops/`)

| File | Job Task | Description |
|------|----------|-------------|
| `app_setup.py` | `setup_app_dependencies` | UC grants on all 3 schemas + CAN_QUERY on serving endpoints for app SP |
| `set_table_metadata.py` | `set_table_metadata` | Add UC table and column descriptions for Genie / lineage |
| `cleanup.py` | *(manual)* | Post-workshop asset cleanup (drops all 3 schemas) |

### Interactive-Only Modules

These modules are not part of the job pipeline and live only in [`interactive_workshop/`](../interactive_workshop/):

| Module | Description |
|--------|-------------|
| 2 — Performance at Scale | Four ETL approaches timed, run-many-models, for-loop anti-patterns |
| 6 — CI/CD with DABs | Asset Bundles, `databricks.yml`, Azure DevOps pipeline |
| 7 — Databricks Apps + Lakebase | Streamlit on serverless, Lakebase architecture walkthrough |

---

## Schema Organization

Tables are split across three UC schemas:

- **`data_schema`** (`actuarial_data`) — Medallion pipeline tables (raw → bronze → silver → gold + features)
- **`models_schema`** (`actuarial_models`) — Model outputs (predictions_*), UC registered models, Online Table, AI Gateway inference tables
- **`app_schema`** (`actuarial_app`) — Synced tables for low-latency app reads

Notebooks that only use data tables (`01`, `02`) receive a single `schema` widget mapped to `data_schema`.
Notebooks that cross schemas (`03`, `04`, `ops/set_table_metadata.py`) receive `data_schema` and `models_schema` widgets.
`ops/app_setup.py` and `ops/cleanup.py` receive all three schema widgets.

---

## Deployed Resources

### Data Schema (`{catalog}.{data_schema}`)

| Table | Source | Description |
|-------|--------|-------------|
| `raw_reserve_development` | Module 1 | Synthetic reserve CDC source |
| `raw_claims_events` | Module 1 | ~42M individual claim incidents |
| `raw_macro_indicators` | fetch_macro_data | Statistics Canada macro data |
| `bronze_reserve_cdc` | SDP | Raw reserve development CDC, append-only |
| `bronze_claims` | SDP | Raw claim events, append-only |
| `bronze_macro_indicators` | SDP | Raw macro indicator ingestion |
| `silver_reserves` | SDP | SCD Type 2 reserve history (Apply Changes) |
| `silver_macro_indicators` | SDP | SCD Type 2 macro data |
| `silver_rolling_features` | SDP | Rolling means, volatility features per segment |
| `gold_reserve_triangle` | SDP | Loss development triangle (accident month × dev lag) |
| `gold_claims_monthly` | SDP | Segment × month claims aggregate (40 segments × 84 months) |
| `gold_macro_features` | SDP | Pivoted macro features by region × month |
| `features_segment_monthly` | Module 2 | UC Feature Table (SARIMAX exog vars) |

### Models Schema (`{catalog}.{models_schema}`)

| Table | Source | Description |
|-------|--------|-------------|
| `predictions_frequency_forecast` | Module 3 | SARIMAX+GARCH forecasts + conditional volatility |
| `predictions_bootstrap_reserves` | Module 3 | Bootstrap Chain Ladder IBNR distribution (VaR, CVaR) |
| `predictions_reserve_evolution` | Module 3 | 12-month reserve adequacy outlook |
| `predictions_reserve_scenarios` | Module 3 | Reserve scenarios (adverse dev, judicial inflation, pandemic, superimposed) |
| `predictions_runoff_projection` | Module 3 | Multi-period surplus trajectory |
| `predictions_ldf_volatility` | Module 3 | Development factor volatility per product line |
| `predictions_regime_parameters` | Module 3 | Regime-switching parameters (Normal/Crisis) |
| `predictions_reserve_validation` | Module 3 | Reserve adequacy validation |

### Models & Endpoints

| Resource | Name |
|----------|------|
| Declarative Pipeline | `actuarial-workshop-medallion` |
| UC Model | `{catalog}.{models_schema}.frequency_forecaster` (`@Champion`) |
| UC Model | `{catalog}.{models_schema}.bootstrap_reserve_simulator` (`@Champion`) |
| Frequency Forecaster Endpoint | `actuarial-workshop-frequency-forecaster` |
| Bootstrap Reserve Endpoint | `actuarial-workshop-bootstrap-reserves` |
| Feature Table | `{catalog}.{data_schema}.features_segment_monthly` |
| Online Table | `{catalog}.{models_schema}.segment_features_online` |

---

## MLflow Experiments

| Experiment | Contents |
|---|---|
| `actuarial_workshop_frequency_forecast` | Bulk SARIMAX+GARCH (40 segments) + Frequency Forecaster PyFunc registration |
| `actuarial_workshop_bootstrap_reserves` | Bootstrap Chain Ladder + scenarios + evolution + Bootstrap Reserve Simulator PyFunc registration |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'dlt'`
Expected when running Module 1 outside a declarative pipeline. The `IN_PIPELINE` guard handles this — pipeline cells are skipped, standalone data generation runs normally.

### Ray `setup_ray_cluster` fails
The job cluster uses ML Runtime 16.4 with Ray pre-installed. If running interactively, ensure you're on a Ray-capable cluster.

### Model Serving endpoint times out
Endpoints can take 5–10 minutes to become ready. The notebook polls for readiness.
