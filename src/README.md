# Job Pipeline Source — `src/`

These notebooks are executed by the **Full Setup** and **Monthly Refresh** jobs defined in `resources/jobs.yml`. They contain job-only logic — no `display()` calls or interactive exploration.

For interactive versions with visualizations and learning notes, see [`interactive_workshop/`](../interactive_workshop/).

---

## Job Pipeline Modules

| Module | File | Job Task | Description |
|--------|------|----------|-------------|
| 1 | `01_data_pipeline.py` | `generate_source_data` | Synthetic data generation + SDP pipeline definitions (Bronze → Silver → Gold) |
| 2 | `02_feature_store.py` | `build_feature_store` | UC Feature Store registration with point-in-time joins |
| 3 | `03_classical_stats_at_scale.py` | `fit_statistical_models` | SARIMAX + GARCH(1,1) per segment, Chain Ladder, Bootstrap Chain Ladder (Ray), MLflow logging |
| 4 | `04_model_serving.py` | `prepare_app_infrastructure` | Serving endpoints + AI Gateway, Online Table, Lakebase setup |

### Operations (`ops/`)

| File | Job Task | Description |
|------|----------|-------------|
| `app_setup.py` | `setup_app_dependencies` | UC grants + CAN_QUERY on serving endpoints for app SP |
| `set_table_metadata.py` | `set_table_metadata` | Add UC table and column descriptions for Genie / lineage |
| `cleanup.py` | *(manual)* | Post-workshop asset cleanup |

### Interactive-Only Modules

These modules are not part of the job pipeline and live only in [`interactive_workshop/`](../interactive_workshop/):

| Module | Description |
|--------|-------------|
| 2 — Performance at Scale | Four ETL approaches timed, run-many-models, for-loop anti-patterns |
| 6 — CI/CD with DABs | Asset Bundles, `databricks.yml`, Azure DevOps pipeline |
| 7 — Databricks Apps + Lakebase | Streamlit on serverless, Lakebase architecture walkthrough |

---

## Deployed Resources

### Unity Catalog Tables (`{catalog}.{schema}`)

| Table | Source | Description |
|-------|--------|-------------|
| `raw_reserve_development` | Module 1 | Synthetic reserve CDC source |
| `bronze_reserve_cdc` | SDP | Raw reserve development CDC, append-only |
| `silver_reserves` | SDP | SCD Type 2 reserve history (Apply Changes) |
| `gold_reserve_triangle` | SDP | Loss development triangle (accident month × dev lag) |
| `bronze_claims` | SDP | Raw claim events, append-only |
| `gold_claims_monthly` | SDP | Segment × month claims aggregate (40 segments × 84 months) |
| `silver_rolling_features` | SDP | Rolling means, volatility features per segment |
| `features_segment_monthly` | Module 2 | UC Feature Table (SARIMAX exog vars) |
| `predictions_frequency_forecast` | Module 3 | SARIMAX+GARCH forecasts + conditional volatility |
| `predictions_bootstrap_reserves` | Module 3 | Bootstrap Chain Ladder IBNR distribution (VaR, CVaR) |
| `predictions_reserve_evolution` | Module 3 | 12-month reserve adequacy outlook |
| `predictions_reserve_scenarios` | Module 3 | Reserve scenarios (adverse dev, judicial inflation, pandemic, superimposed) |
| `predictions_runoff_projection` | Module 3 | Multi-period surplus trajectory |
| `predictions_ldf_volatility` | Module 3 | Development factor volatility per product line |

### Models & Endpoints

| Resource | Name |
|----------|------|
| Declarative Pipeline | `actuarial-workshop-medallion` |
| UC Model | `{catalog}.{schema}.frequency_forecaster` (`@Champion`) |
| UC Model | `{catalog}.{schema}.bootstrap_reserve_simulator` (`@Champion`) |
| Frequency Forecaster Endpoint | `actuarial-workshop-frequency-forecaster` |
| Bootstrap Reserve Endpoint | `actuarial-workshop-bootstrap-reserves` |
| Feature Table | `{catalog}.{schema}.features_segment_monthly` |
| Online Table | `{catalog}.{schema}.segment_features_online` |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'dlt'`
Expected when running Module 1 outside a declarative pipeline. The `IN_PIPELINE` guard handles this — pipeline cells are skipped, standalone data generation runs normally.

### Ray `setup_ray_cluster` fails
The job cluster uses ML Runtime 16.4 with Ray pre-installed. If running interactively, ensure you're on a Ray-capable cluster.

### Model Serving endpoint times out
Endpoints can take 5–10 minutes to become ready. The notebook polls for readiness.
