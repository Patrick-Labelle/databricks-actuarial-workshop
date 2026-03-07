# Interactive Workshop Notebooks

These notebooks are designed for **hands-on learning** during the workshop.
Run them interactively, one at a time, attached to a Databricks cluster.

## Prerequisites

- A Databricks workspace with classic compute (not serverless-only)
- A Unity Catalog catalog where you have CREATE SCHEMA permission
- A classic ML cluster (DBR 16.4+ ML runtime) for Modules 3-5

## Schema Organization

Tables are organized across three UC schemas:
- **`actuarial_data`** — Data pipeline tables (raw, bronze, silver, gold, features)
- **`actuarial_models`** — Model outputs (predictions_*) and registered models
- **`actuarial_app`** — Synced tables for low-latency app reads

Modules 1-3 write to `actuarial_data`. Module 4 writes to both `actuarial_data` and `actuarial_models`. Module 5 reads from `actuarial_models` for serving endpoints.

## Notebook Order

| # | Notebook | Description | Cluster |
|---|----------|-------------|---------|
| 1 | `01_data_pipeline.py` | Generate synthetic data + explore the declarative pipeline | Any cluster or SDP pipeline |
| 2 | `02_performance_at_scale.py` | Spark scaling patterns (ETL and modeling) | Any cluster |
| 3 | `03_feature_store.py` | UC Feature Store with point-in-time joins | Classic ML cluster |
| 4 | `04_classical_stats_at_scale.py` | SARIMAX/GARCH + Bootstrap Chain Ladder + model registration | Classic ML cluster (4+ workers) |
| 5 | `05_model_serving.py` | Endpoints, Online Table, Lakebase, Genie | Classic ML cluster |
| 6 | `06_dabs_cicd.py` | Asset Bundles + CI/CD concepts | Any cluster |
| 7 | `07_databricks_apps.py` | Streamlit app architecture + Lakebase | Any cluster |

## Relationship to `src/`

The `src/` folder contains the **job-optimized** versions of Modules 1, 3, 4, and 5
using compressed numbering (01–04), while `interactive_workshop/` uses the original
numbering (01–07). These run automatically via the setup job (`resources/jobs.yml`)
and are stripped of interactive display calls and educational content. The
`interactive_workshop/` versions include all display() calls, demos, and explanations.
