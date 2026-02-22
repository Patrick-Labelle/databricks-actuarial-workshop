# Databricks notebook source
# MAGIC %md
# MAGIC # Module 6: Production CI/CD
# MAGIC ## Databricks Asset Bundles + Azure DevOps
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC
# MAGIC ---
# MAGIC ### What Are Databricks Asset Bundles (DABs)?
# MAGIC
# MAGIC DABs let you treat **Databricks resources as code**:
# MAGIC - Jobs, DLT Pipelines, Model Serving endpoints, notebooks — all in YAML
# MAGIC - Version-controlled, peer-reviewed, testable
# MAGIC - Deploy to Dev / Staging / Prod with a single CLI command
# MAGIC - The same CI/CD practices actuarial/finance teams use for application code
# MAGIC
# MAGIC **Why this matters for actuaries:**
# MAGIC - Reproducibility: every model promotion is a Git commit
# MAGIC - Auditability: change history, approvals, who deployed what and when
# MAGIC - Safety: staging environment validates before production
# MAGIC - Consistency: eliminate "works on my laptop" problems

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Project Structure
# MAGIC
# MAGIC A DABs project for this workshop looks like:
# MAGIC
# MAGIC ```
# MAGIC actuarial-workshop/
# MAGIC ├── databricks.yml          ← Top-level bundle config (name, workspace host)
# MAGIC ├── resources/
# MAGIC │   ├── jobs.yml            ← Job definitions
# MAGIC │   ├── serving.yml         ← Model Serving endpoint definitions
# MAGIC │   └── pipelines.yml       ← DLT pipeline definitions (Module 1)
# MAGIC ├── notebooks/
# MAGIC │   ├── 04_classical_stats_at_scale.py
# MAGIC │   ├── 05_mlflow_uc_serving.py
# MAGIC │   └── 06_dabs_cicd.py
# MAGIC ├── src/
# MAGIC │   └── monte_carlo.py      ← Reusable Python modules
# MAGIC ├── tests/
# MAGIC │   └── test_monte_carlo.py ← Unit tests (run in CI)
# MAGIC └── azure-pipelines.yml     ← Azure DevOps pipeline
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `databricks.yml` — The Root Bundle Config
# MAGIC
# MAGIC ```yaml
# MAGIC # databricks.yml
# MAGIC bundle:
# MAGIC   name: actuarial-workshop
# MAGIC
# MAGIC workspace:
# MAGIC   host: https://e2-demo-field-eng.cloud.databricks.com
# MAGIC
# MAGIC include:
# MAGIC   - resources/*.yml     # Pull in job, serving, pipeline definitions
# MAGIC
# MAGIC # Environment-specific overrides
# MAGIC targets:
# MAGIC   development:
# MAGIC     mode: development        # Prepends [dev username] to all resource names
# MAGIC     default: true
# MAGIC     workspace:
# MAGIC       root_path: /Users/${workspace.current_user.userName}/.bundle/${bundle.name}/dev
# MAGIC
# MAGIC   staging:
# MAGIC     workspace:
# MAGIC       host: https://e2-demo-field-eng.cloud.databricks.com
# MAGIC       root_path: /Shared/.bundle/${bundle.name}/staging
# MAGIC     variables:
# MAGIC       catalog:  staging_catalog
# MAGIC       schema:   actuarial_workshop
# MAGIC
# MAGIC   production:
# MAGIC     workspace:
# MAGIC       host: https://e2-demo-field-eng.cloud.databricks.com
# MAGIC       root_path: /Shared/.bundle/${bundle.name}/prod
# MAGIC     variables:
# MAGIC       catalog:  patrick_labelle      # or your production catalog
# MAGIC       schema:   actuarial_workshop
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `resources/jobs.yml` — Job Definitions
# MAGIC
# MAGIC ```yaml
# MAGIC # resources/jobs.yml
# MAGIC variables:
# MAGIC   catalog:
# MAGIC     default: patrick_labelle
# MAGIC   schema:
# MAGIC     default: actuarial_workshop
# MAGIC
# MAGIC resources:
# MAGIC   jobs:
# MAGIC     actuarial_model_pipeline:
# MAGIC       name: "[${bundle.target}] Actuarial Model Pipeline"
# MAGIC       description: "End-to-end: data prep → SARIMA/GARCH → MLflow → serving refresh"
# MAGIC
# MAGIC       schedule:
# MAGIC         quartz_cron_expression: "0 0 2 1 * ?"   # Monthly, 2am on the 1st
# MAGIC         timezone_id: America/Toronto
# MAGIC         pause_status: UNPAUSED
# MAGIC
# MAGIC       # Email on failure
# MAGIC       email_notifications:
# MAGIC         on_failure:
# MAGIC           - patrick.labelle@databricks.com
# MAGIC
# MAGIC       tasks:
# MAGIC         - task_key: generate_silver_data
# MAGIC           description: "Load/refresh claims time series to Silver"
# MAGIC           notebook_task:
# MAGIC             notebook_path: ${workspace.file_path}/notebooks/04_classical_stats_at_scale
# MAGIC             base_parameters:
# MAGIC               catalog:  ${var.catalog}
# MAGIC               schema:   ${var.schema}
# MAGIC           environment_key: ml_env
# MAGIC
# MAGIC         - task_key: fit_sarima_garch
# MAGIC           description: "Fit SARIMA/GARCH per segment; log to MLflow"
# MAGIC           depends_on:
# MAGIC             - task_key: generate_silver_data
# MAGIC           notebook_task:
# MAGIC             notebook_path: ${workspace.file_path}/notebooks/04_classical_stats_at_scale
# MAGIC             base_parameters:
# MAGIC               step: fit_models
# MAGIC           environment_key: ml_env
# MAGIC
# MAGIC         - task_key: register_champion_model
# MAGIC           description: "Promote best model to @Champion in UC Registry"
# MAGIC           depends_on:
# MAGIC             - task_key: fit_sarima_garch
# MAGIC           notebook_task:
# MAGIC             notebook_path: ${workspace.file_path}/notebooks/05_mlflow_uc_serving
# MAGIC             base_parameters:
# MAGIC               step: register
# MAGIC           environment_key: ml_env
# MAGIC
# MAGIC         - task_key: refresh_serving_endpoint
# MAGIC           description: "Update Model Serving endpoint to new Champion version"
# MAGIC           depends_on:
# MAGIC             - task_key: register_champion_model
# MAGIC           notebook_task:
# MAGIC             notebook_path: ${workspace.file_path}/notebooks/05_mlflow_uc_serving
# MAGIC             base_parameters:
# MAGIC               step: deploy_endpoint
# MAGIC           environment_key: ml_env
# MAGIC
# MAGIC       environments:
# MAGIC         - environment_key: ml_env
# MAGIC           spec:
# MAGIC             client: "4"
# MAGIC             dependencies:
# MAGIC               - "statsmodels>=0.14"
# MAGIC               - "arch>=7.0"
# MAGIC               - "mlflow>=2.14"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `resources/serving.yml` — Model Serving Endpoint
# MAGIC
# MAGIC ```yaml
# MAGIC # resources/serving.yml
# MAGIC resources:
# MAGIC   model_serving_endpoints:
# MAGIC     sarima_forecaster_endpoint:
# MAGIC       name: "[${bundle.target}] actuarial-sarima-forecaster"
# MAGIC       config:
# MAGIC         served_models:
# MAGIC           - model_name: "patrick_labelle.actuarial_workshop.sarima_claims_forecaster"
# MAGIC             model_version: "1"               # Override per environment
# MAGIC             workload_size: "Small"
# MAGIC             scale_to_zero_enabled: true
# MAGIC         auto_capture_config:
# MAGIC           catalog_name:  ${var.catalog}
# MAGIC           schema_name:   ${var.schema}
# MAGIC           table_name_prefix: "sarima_endpoint"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. CLI Workflow — Validate / Deploy / Run
# MAGIC
# MAGIC ### Local Development
# MAGIC ```bash
# MAGIC # Authenticate (done once)
# MAGIC databricks auth login https://e2-demo-field-eng.cloud.databricks.com --profile e2-demo-west
# MAGIC
# MAGIC # Validate the bundle (checks YAML syntax, resource references, permissions)
# MAGIC databricks bundle validate --profile e2-demo-west
# MAGIC
# MAGIC # Deploy to your personal development workspace path
# MAGIC databricks bundle deploy --target development --profile e2-demo-west
# MAGIC
# MAGIC # Run the pipeline manually
# MAGIC databricks bundle run actuarial_model_pipeline --target development --profile e2-demo-west
# MAGIC
# MAGIC # Watch run status
# MAGIC databricks bundle run actuarial_model_pipeline --restart-on-failure --no-wait
# MAGIC ```
# MAGIC
# MAGIC ### Promotion to Staging
# MAGIC ```bash
# MAGIC # Deploy to staging (uses staging workspace and catalog)
# MAGIC databricks bundle deploy --target staging
# MAGIC
# MAGIC # Run integration tests against staging
# MAGIC databricks bundle run actuarial_model_pipeline --target staging
# MAGIC ```
# MAGIC
# MAGIC ### Production Deployment
# MAGIC ```bash
# MAGIC # Deploy to production (typically done by CI/CD, not manually)
# MAGIC databricks bundle deploy --target production
# MAGIC ```
# MAGIC
# MAGIC **What `bundle validate` checks:**
# MAGIC - YAML syntax and schema validation
# MAGIC - Resource references (do the notebook paths exist?)
# MAGIC - Permission validation (does the service principal have access?)
# MAGIC - Environment variable completeness

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Azure DevOps Pipeline
# MAGIC
# MAGIC ```yaml
# MAGIC # azure-pipelines.yml
# MAGIC trigger:
# MAGIC   branches:
# MAGIC     include:
# MAGIC       - main
# MAGIC   tags:
# MAGIC     include:
# MAGIC       - "release/*"
# MAGIC
# MAGIC variables:
# MAGIC   DATABRICKS_HOST: https://e2-demo-field-eng.cloud.databricks.com
# MAGIC
# MAGIC stages:
# MAGIC
# MAGIC # ── Stage 1: PR Validation ─────────────────────────────────────────────────
# MAGIC   - stage: Validate
# MAGIC     displayName: "Validate Bundle + Unit Tests"
# MAGIC     condition: eq(variables['Build.Reason'], 'PullRequest')
# MAGIC     jobs:
# MAGIC       - job: validate
# MAGIC         pool:
# MAGIC           vmImage: ubuntu-latest
# MAGIC         steps:
# MAGIC           - task: UsePythonVersion@0
# MAGIC             inputs:
# MAGIC               versionSpec: "3.12"
# MAGIC
# MAGIC           - script: |
# MAGIC               pip install databricks-cli uv
# MAGIC               uv sync
# MAGIC             displayName: "Install dependencies"
# MAGIC
# MAGIC           # Authenticate via Workload Identity Federation (no stored secrets)
# MAGIC           - task: AzureCLI@2
# MAGIC             displayName: "Authenticate Databricks (OIDC)"
# MAGIC             inputs:
# MAGIC               azureSubscription: "databricks-sp"
# MAGIC               scriptType: bash
# MAGIC               scriptLocation: inlineScript
# MAGIC               inlineScript: |
# MAGIC                 export DATABRICKS_HOST=$(DATABRICKS_HOST)
# MAGIC                 export DATABRICKS_TOKEN=$(az account get-access-token \
# MAGIC                   --resource "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d" \
# MAGIC                   --query accessToken -o tsv)
# MAGIC                 databricks bundle validate
# MAGIC
# MAGIC           - script: |
# MAGIC               uv run pytest tests/ -v --tb=short
# MAGIC             displayName: "Run unit tests"
# MAGIC
# MAGIC # ── Stage 2: Staging Deploy (merge to main) ────────────────────────────────
# MAGIC   - stage: DeployStaging
# MAGIC     displayName: "Deploy to Staging"
# MAGIC     condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
# MAGIC     dependsOn: []
# MAGIC     jobs:
# MAGIC       - deployment: deploy_staging
# MAGIC         environment: staging
# MAGIC         pool:
# MAGIC           vmImage: ubuntu-latest
# MAGIC         strategy:
# MAGIC           runOnce:
# MAGIC             deploy:
# MAGIC               steps:
# MAGIC                 - task: AzureCLI@2
# MAGIC                   displayName: "Bundle Deploy → Staging"
# MAGIC                   inputs:
# MAGIC                     azureSubscription: "databricks-sp"
# MAGIC                     scriptType: bash
# MAGIC                     scriptLocation: inlineScript
# MAGIC                     inlineScript: |
# MAGIC                       export DATABRICKS_HOST=$(DATABRICKS_HOST)
# MAGIC                       export DATABRICKS_TOKEN=$(az account get-access-token \
# MAGIC                         --resource "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d" \
# MAGIC                         --query accessToken -o tsv)
# MAGIC                       pip install databricks-cli
# MAGIC                       databricks bundle deploy --target staging
# MAGIC                       databricks bundle run actuarial_model_pipeline --target staging --no-wait
# MAGIC
# MAGIC # ── Stage 3: Production Deploy (release tag) ───────────────────────────────
# MAGIC   - stage: DeployProduction
# MAGIC     displayName: "Deploy to Production"
# MAGIC     condition: startsWith(variables['Build.SourceBranch'], 'refs/tags/release/')
# MAGIC     dependsOn: DeployStaging
# MAGIC     jobs:
# MAGIC       - deployment: deploy_prod
# MAGIC         environment: production   # Azure DevOps environment with approval gate
# MAGIC         pool:
# MAGIC           vmImage: ubuntu-latest
# MAGIC         strategy:
# MAGIC           runOnce:
# MAGIC             deploy:
# MAGIC               steps:
# MAGIC                 - task: AzureCLI@2
# MAGIC                   displayName: "Bundle Deploy → Production"
# MAGIC                   inputs:
# MAGIC                     azureSubscription: "databricks-sp"
# MAGIC                     scriptType: bash
# MAGIC                     scriptLocation: inlineScript
# MAGIC                     inlineScript: |
# MAGIC                       export DATABRICKS_HOST=$(DATABRICKS_HOST)
# MAGIC                       export DATABRICKS_TOKEN=$(az account get-access-token \
# MAGIC                         --resource "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d" \
# MAGIC                         --query accessToken -o tsv)
# MAGIC                       pip install databricks-cli
# MAGIC                       databricks bundle deploy --target production
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Authentication: Workload Identity Federation
# MAGIC
# MAGIC **Never store PATs or service account secrets in Azure DevOps variables.**
# MAGIC
# MAGIC Use **Workload Identity Federation** (OIDC):
# MAGIC
# MAGIC ```
# MAGIC Azure DevOps Pipeline
# MAGIC        ↓ (OIDC JWT)
# MAGIC Azure Active Directory
# MAGIC        ↓ (federated credential trust)
# MAGIC Databricks Service Principal
# MAGIC        ↓ (UC permissions)
# MAGIC Databricks Workspace
# MAGIC ```
# MAGIC
# MAGIC Setup steps:
# MAGIC 1. Create an Azure AD Service Principal (`databricks-deploy-sp`)
# MAGIC 2. Add it as a Databricks workspace user with appropriate permissions
# MAGIC 3. Configure Federated Credential in AAD pointing to your ADO org/project
# MAGIC 4. In ADO: create `Service Connection` using Workload Identity Federation
# MAGIC 5. No secrets stored anywhere — token is issued at pipeline runtime

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices Checklist
# MAGIC
# MAGIC ### Bundle Development
# MAGIC - [ ] One bundle per repository (or per major domain)
# MAGIC - [ ] Use `mode: development` in dev target — adds `[dev username]` prefix, prevents collisions
# MAGIC - [ ] Parameterize catalog/schema via `variables` — never hardcode in resource definitions
# MAGIC - [ ] Always run `bundle validate` before `bundle deploy`
# MAGIC - [ ] Pin environment dependencies with exact versions (reproducibility)
# MAGIC
# MAGIC ### CI/CD Pipeline
# MAGIC - [ ] Unit tests run on every PR (fast, no cluster needed)
# MAGIC - [ ] Integration tests run in staging (real cluster, real data)
# MAGIC - [ ] Manual approval gate before production deployment
# MAGIC - [ ] Workload Identity Federation — no stored PATs
# MAGIC - [ ] Separate service principals for staging vs production
# MAGIC - [ ] Pin the Databricks CLI version in your pipeline
# MAGIC
# MAGIC ### Model Promotion
# MAGIC - [ ] Never manually edit production bundle YAML — always via PR
# MAGIC - [ ] Champion/Challenger alias pattern: update alias, not endpoint config
# MAGIC - [ ] MLflow model versions tagged with Git commit hash
# MAGIC - [ ] Rollback = point `@Champion` alias to previous version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Quick Demo: Validate and Deploy This Bundle
# MAGIC
# MAGIC The cells below show what the output of `bundle validate` and `bundle deploy` look like.
# MAGIC You can run these from a terminal with the Databricks CLI installed.

# COMMAND ----------

import subprocess, json

def run_cmd(cmd: str) -> str:
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

# Show current bundle project (if it existed on this cluster)
print("Typical `databricks bundle validate` output:")
print("""
{
  "bundle": {
    "name": "actuarial-workshop",
    "target": "development"
  },
  "workspace": {
    "host": "https://e2-demo-field-eng.cloud.databricks.com",
    "current_user": {
      "user_name": "patrick.labelle@databricks.com"
    },
    "root_path": "/Users/patrick.labelle@databricks.com/.bundle/actuarial-workshop/dev",
    "file_path": "/Users/patrick.labelle@databricks.com/.bundle/actuarial-workshop/dev/files"
  },
  "resources": {
    "jobs": {
      "actuarial_model_pipeline": {
        "id": "123456789",
        "name": "[dev patrick.labelle] Actuarial Model Pipeline"
      }
    },
    "model_serving_endpoints": {
      "sarima_forecaster_endpoint": {
        "name": "[dev patrick.labelle] actuarial-sarima-forecaster"
      }
    }
  }
}
""")

print("Typical `databricks bundle deploy` output:")
print("""
Uploading bundle files to /Users/patrick.labelle@databricks.com/.bundle/actuarial-workshop/dev/files...
  ./notebooks/04_classical_stats_at_scale.py (5.2 KB)
  ./notebooks/05_mlflow_uc_serving.py (4.8 KB)
  ./notebooks/06_dabs_cicd.py (3.1 KB)
  ./src/monte_carlo.py (1.2 KB)

Deploying resources...
  Updating job [dev patrick.labelle] Actuarial Model Pipeline...
  Updating model serving endpoint [dev patrick.labelle] actuarial-sarima-forecaster...

Deployment complete!
  Job URL: https://e2-demo-field-eng.cloud.databricks.com/jobs/123456789
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Concept | Key Takeaway |
# MAGIC |---|---|
# MAGIC | `databricks.yml` | Declares all Databricks resources as YAML — version controlled |
# MAGIC | `targets` | Dev / Staging / Prod environments with per-environment overrides |
# MAGIC | `bundle validate` | Catches errors before deployment — run on every PR |
# MAGIC | `bundle deploy` | Syncs files and creates/updates all declared resources |
# MAGIC | Azure DevOps | 3-stage pipeline: PR validation → Staging deploy → Production deploy |
# MAGIC | Auth | Workload Identity Federation — zero stored secrets |
# MAGIC | Model promotion | Champion alias + bundle version tag = full audit trail |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Workshop Complete 🎉
# MAGIC
# MAGIC You've covered the full production stack:
# MAGIC
# MAGIC ```
# MAGIC Bronze/Silver/Gold (DLT)
# MAGIC         ↓
# MAGIC Feature Store (UC, point-in-time)
# MAGIC         ↓
# MAGIC Classical Stats at Scale (SARIMA/GARCH, applyInPandas)
# MAGIC         ↓
# MAGIC Monte Carlo Simulation (Ray task-parallel)
# MAGIC         ↓
# MAGIC MLflow + UC Model Registry (Champion alias, lineage)
# MAGIC         ↓
# MAGIC Model Serving (REST endpoint, scale-to-zero)
# MAGIC         ↓
# MAGIC Databricks Asset Bundles + Azure DevOps CI/CD
# MAGIC ```
# MAGIC
# MAGIC Everything is governed, reproducible, and production-ready — meeting actuarial and regulatory standards.