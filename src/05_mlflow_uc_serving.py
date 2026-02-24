# Databricks notebook source
# MAGIC %md
# MAGIC # Module 5: Governance & Operations
# MAGIC ## MLflow + Unity Catalog Model Registry + Model Serving
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC
# MAGIC ---
# MAGIC ### What We'll Cover
# MAGIC 1. **Wrap a Classical Model** — Package SARIMAX as `mlflow.pyfunc.PythonModel`
# MAGIC 2. **Log to MLflow** — Signature, parameters, metrics, artifacts
# MAGIC 3. **Register to UC Model Registry** — Governed lineage, `Champion` alias
# MAGIC 4. **Deploy Model Serving Endpoint** — REST endpoint for real-time forecasting
# MAGIC 5. **Call the Endpoint** — Demonstrate the request/response contract
# MAGIC
# MAGIC ---
# MAGIC ### The Governance Story
# MAGIC
# MAGIC For actuaries, **audit trail and reproducibility** are non-negotiable. MLflow + Unity Catalog provides:
# MAGIC
# MAGIC | Requirement | How Databricks Addresses It |
# MAGIC |---|---|
# MAGIC | Reproduce any forecast | MLflow run captures data version, code, params, environment |
# MAGIC | Who approved production model? | UC aliases (`Champion`) with owner/timestamp |
# MAGIC | Which model scored last quarter's renewals? | Model version linked to serving endpoint logs |
# MAGIC | Feature leakage prevention | Feature Store point-in-time joins (Module 3) |
# MAGIC | Access control | UC permissions on catalog/schema/model |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
# Passed as base_parameters by the bundle job; defaults used when interactive.
dbutils.widgets.text("catalog",       "my_catalog",                           "UC Catalog")
dbutils.widgets.text("schema",        "actuarial_workshop",                   "UC Schema")
dbutils.widgets.text("endpoint_name", "actuarial-workshop-sarima-forecaster", "Serving Endpoint")
CATALOG        = dbutils.widgets.get("catalog")
SCHEMA         = dbutils.widgets.get("schema")
ENDPOINT_NAME  = dbutils.widgets.get("endpoint_name")
MODEL_NAME     = f"{CATALOG}.{SCHEMA}.sarima_claims_forecaster"

# MLflow 3 default: Unity Catalog as registry backend
mlflow.set_registry_uri("databricks-uc")

print(f"MLflow version:   {mlflow.__version__}")
print(f"Registry URI:     {mlflow.get_registry_uri()}")
print(f"Model name:       {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Train the "Champion" Model
# MAGIC
# MAGIC We train on the `Personal_Auto__Ontario` segment — highest volume, used as the primary validation segment.
# MAGIC In practice you'd select the best segment model from Module 4's experiment runs.

# COMMAND ----------

# Load from Delta (written in Module 4)
try:
    claims_pdf = (
        spark.table(f"{CATALOG}.{SCHEMA}.claims_time_series")
        .filter("segment_id = 'Personal_Auto__Ontario'")
        .orderBy("month")
        .toPandas()
    )
    print(f"Loaded {len(claims_pdf)} months from Delta table")
except Exception:
    # Regenerate if Module 4 hasn't been run
    print("Generating sample data (run Module 4 first for full dataset)")
    np.random.seed(42)
    months = pd.date_range("2019-01-01", periods=60, freq="MS")
    SEASONALITY = {1: 1.25, 2: 1.20, 3: 1.10, 4: 0.95, 5: 0.90, 6: 0.88,
                   7: 0.85, 8: 0.87, 9: 0.92, 10: 1.00, 11: 1.10, 12: 1.20}
    base = 450 * 1.4  # Personal Auto × Ontario
    y = [max(0, base * (1+0.003*i) * SEASONALITY[m.month] * (1+np.random.normal(0, 0.08)))
         for i, m in enumerate(months)]
    claims_pdf = pd.DataFrame({"month": months.date, "claims_count": [int(round(v)) for v in y]})

y_train = claims_pdf["claims_count"].astype(float).values

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define MLflow PyFunc Wrapper
# MAGIC
# MAGIC `mlflow.pyfunc.PythonModel` is the bridge between classical statsmodels and the MLflow serving infrastructure.
# MAGIC We wrap the fitted SARIMAX object so it can:
# MAGIC - Accept a JSON request (number of forecast months)
# MAGIC - Return structured forecast output (mean + confidence intervals)

# COMMAND ----------

class SARIMAXForecaster(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for a fitted statsmodels SARIMAX model.

    Input:  pandas DataFrame with column `horizon` (int) — months to forecast
    Output: pandas DataFrame with columns: month_offset, forecast_mean, lo95, hi95

    This pattern works for any classical Python model: statsmodels, lifelines,
    scikit-survival, arch — wrap it, log it, serve it.
    """

    def load_context(self, context):
        """Load the pickled SARIMAX model from the MLflow artifact store."""
        import pickle
        import os
        model_path = os.path.join(context.artifacts["sarimax_model"], "model.pkl")
        with open(model_path, "rb") as f:
            self.model_fit = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecast.

        Args:
            model_input: DataFrame with column `horizon` (int, 1–24)

        Returns:
            DataFrame with forecast mean and 95% confidence intervals.
        """
        horizon = int(model_input["horizon"].iloc[0])
        horizon = max(1, min(horizon, 24))  # clamp to [1, 24]

        forecast = self.model_fit.get_forecast(steps=horizon)
        mean_fcst = forecast.predicted_mean
        ci        = np.asarray(forecast.conf_int(alpha=0.05))  # numpy or DataFrame depending on statsmodels version

        return pd.DataFrame({
            "month_offset":   list(range(1, horizon + 1)),
            "forecast_mean":  list(mean_fcst.round(1)),
            "forecast_lo95":  list(np.round(ci[:, 0], 1)),
            "forecast_hi95":  list(np.round(ci[:, 1], 1)),
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train, Log, and Register
# MAGIC
# MAGIC We run a single MLflow run that:
# MAGIC 1. Fits SARIMAX
# MAGIC 2. Logs parameters, metrics, and plots
# MAGIC 3. Saves the model artifact
# MAGIC 4. Registers to Unity Catalog Model Registry

# COMMAND ----------

import os, pickle, tempfile, cloudpickle, scipy, statsmodels as _statsmodels
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_current_user = spark.sql("SELECT current_user()").collect()[0][0]
# Use a flat path under /Users/<email>/ — avoid nested subdirectories which
# require the parent to pre-exist (fails on fresh workspaces).
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_sarima_claims_forecaster")

with mlflow.start_run(run_name="sarima_personal_auto_ontario_champion") as run:

    # ── Fit model ────────────────────────────────────────────────────────────
    model = SARIMAX(
        y_train,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)

    # ── Compute metrics ───────────────────────────────────────────────────────
    fitted_vals = fit.fittedvalues[12:]
    actual_vals = y_train[12:]
    mape = float(np.mean(np.abs((actual_vals - fitted_vals) / np.clip(actual_vals, 1, None))) * 100)
    rmse = float(np.sqrt(np.mean((actual_vals - fitted_vals)**2)))

    # ── Log parameters ────────────────────────────────────────────────────────
    mlflow.set_tags({
        "segment_id":      "Personal_Auto__Ontario",
        "workshop_module": "5",
        "model_class":     "SARIMAX",
        "audience":        "actuarial-workshop",
    })
    mlflow.log_params({
        "order_p":      1, "order_d": 0, "order_q": 1,
        "seasonal_P":   1, "seasonal_D": 1, "seasonal_Q": 0,
        "seasonal_m":   12,
        "training_months": len(y_train),
        "segment":      "Personal_Auto__Ontario",
    })
    mlflow.log_metrics({
        "mape_pct": round(mape, 2),
        "rmse":     round(rmse, 1),
        "aic":      round(fit.aic, 2),
        "bic":      round(fit.bic, 2),
    })

    # ── Save forecast plot as artifact ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(y_train)), y_train, label="Actual", lw=1.5)
    ax.plot(range(12, len(fit.fittedvalues)), fit.fittedvalues[12:],
            label="Fitted", lw=1.5, ls="--")

    fc = fit.get_forecast(steps=12)
    fc_mean = fc.predicted_mean
    fc_ci   = np.asarray(fc.conf_int())  # numpy array; statsmodels may return DataFrame or ndarray
    t_fc = range(len(y_train), len(y_train) + 12)
    ax.plot(t_fc, fc_mean, label="Forecast (12m)", color="orange", lw=2)
    ci_lo, ci_hi = fc_ci[:, 0], fc_ci[:, 1]
    ax.fill_between(t_fc, ci_lo, ci_hi, alpha=0.2, color="orange")
    ax.set_title("SARIMA(1,0,1)(1,1,0,12) — Personal Auto Ontario")
    ax.set_xlabel("Month offset")
    ax.set_ylabel("Monthly Claims Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(tmpdir, "forecast_plot.png")
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(plot_path, artifact_path="plots")
    plt.close()

    # ── Save pickled model for PyFunc ─────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        model_pkl_path = os.path.join(tmpdir, "model.pkl")
        with open(model_pkl_path, "wb") as f:
            pickle.dump(fit, f)

        # Define MLflow input/output signature
        input_schema  = mlflow.types.Schema([mlflow.types.ColSpec("integer", "horizon")])
        output_schema = mlflow.types.Schema([
            mlflow.types.ColSpec("integer", "month_offset"),
            mlflow.types.ColSpec("double",  "forecast_mean"),
            mlflow.types.ColSpec("double",  "forecast_lo95"),
            mlflow.types.ColSpec("double",  "forecast_hi95"),
        ])
        signature = mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)

        # Log PyFunc model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="sarima_forecaster",
            python_model=SARIMAXForecaster(),
            artifacts={"sarimax_model": tmpdir},
            signature=signature,
            registered_model_name=MODEL_NAME,   # Auto-registers to UC
            pip_requirements=[
                # Pin exact training-time versions so the serving env is built from
                # the package cache without running pip's dependency resolver.
                f"statsmodels=={_statsmodels.__version__}",
                f"numpy=={np.__version__}",
                f"scipy=={scipy.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
            ],
        )

    RUN_ID = run.info.run_id
    print(f"\nModel logged and registered to: {MODEL_NAME}")
    print(f"Run ID:  {RUN_ID}")
    print(f"MAPE:    {mape:.1f}%  |  RMSE: {rmse:.0f}  |  AIC: {fit.aic:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Set `Champion` Alias
# MAGIC
# MAGIC Aliases let serving endpoints reference models by **role** rather than version number.
# MAGIC `@Champion` always points to the approved production model.
# MAGIC When you promote a new Challenger, you update the alias — not the endpoint config.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest version just registered
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(int(v.version) for v in versions)

# Set Champion alias
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=latest_version,
)

print(f"Set @Champion → version {latest_version}")
print(f"\nFull model path: models:/{MODEL_NAME}@Champion")

# Add descriptive tag
client.set_model_version_tag(
    name=MODEL_NAME,
    version=str(latest_version),
    key="approved_by",
    value="actuarial-workshop-demo",
)
client.set_model_version_tag(
    name=MODEL_NAME,
    version=str(latest_version),
    key="segment",
    value="Personal_Auto__Ontario",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy a Model Serving Endpoint
# MAGIC
# MAGIC We use the Databricks Serving REST API to create a real-time endpoint.
# MAGIC The endpoint will:
# MAGIC - Auto-scale from 0 → N (scale-to-zero when idle)
# MAGIC - Load the model from the `@Champion` alias
# MAGIC - Expose a POST endpoint that accepts our `horizon` JSON input

# COMMAND ----------

import requests, json

# Get workspace URL and token
WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .apiToken().get()
)

endpoint_config = {
    "name": ENDPOINT_NAME,
    "config": {
        "served_models": [{
            "name":                   "sarima-champion",
            "model_name":             MODEL_NAME,
            "model_version":          str(latest_version),
            "workload_size":          "Small",
            "scale_to_zero_enabled":  True,
        }]
    },
}

# Check if endpoint already exists
resp = requests.get(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
    headers={"Authorization": f"Bearer {TOKEN}"},
)

if resp.status_code == 200:
    # Update existing endpoint
    resp = requests.put(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/config",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json=endpoint_config["config"],
    )
    print(f"Endpoint updated: {resp.status_code}")
else:
    # Create new endpoint
    resp = requests.post(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json=endpoint_config,
    )
    print(f"Endpoint created: {resp.status_code}")

if resp.status_code in (200, 201):
    endpoint_url = f"https://{WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    print(f"\nEndpoint URL:\n  {endpoint_url}")
    print(f"\nNote: Endpoint takes ~5 minutes to reach 'Ready' state.")
else:
    print(f"Error: {resp.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Call the Endpoint
# MAGIC
# MAGIC Once the endpoint is `Ready`, you can call it from anywhere with a token:
# MAGIC - External Python scripts, actuarial desktop tools
# MAGIC - Azure API Management gateway
# MAGIC - React/Streamlit dashboards (via Databricks Apps)

# COMMAND ----------

import time

def wait_for_endpoint(endpoint_name: str, timeout_seconds: int = 600) -> bool:
    """Poll until endpoint reaches 'READY' state."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        resp = requests.get(
            f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{endpoint_name}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        if resp.status_code == 200:
            state = resp.json().get("state", {}).get("ready", "")
            print(f"  Endpoint state: {state}")
            if state == "READY":
                return True
        time.sleep(30)
    return False

print("Waiting for endpoint to become ready (check Serving UI for live status)...")
print("Skipping wait for demo — calling endpoint assumes it's already ready.\n")

# ── Sample call ───────────────────────────────────────────────────────────────
def call_forecasting_endpoint(horizon: int) -> dict:
    """Call the SARIMA forecasting endpoint.

    Returns an error dict (instead of raising) on timeout or HTTP errors so the
    notebook cell doesn't fail when the endpoint is still warming up after creation.
    """
    payload = {
        "dataframe_records": [{"horizon": horizon}]
    }
    try:
        resp = requests.post(
            f"https://{WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations",
            headers={
                "Authorization":  f"Bearer {TOKEN}",
                "Content-Type":   "application/json",
            },
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": resp.text, "status_code": resp.status_code}
    except requests.exceptions.Timeout:
        return {"note": "Endpoint is still warming up — it takes ~5 min to reach READY state. "
                        "Re-run this cell once the endpoint is READY."}
    except Exception as exc:
        return {"error": str(exc)}

# Request a 6-month forecast
print("Requesting 6-month forecast from Model Serving endpoint...\n")
result = call_forecasting_endpoint(horizon=6)
print(json.dumps(result, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expected Response Format
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "predictions": [
# MAGIC     {"month_offset": 1, "forecast_mean": 642.3, "forecast_lo95": 580.1, "forecast_hi95": 704.5},
# MAGIC     {"month_offset": 2, "forecast_mean": 635.1, "forecast_lo95": 568.2, "forecast_hi95": 702.0},
# MAGIC     ...
# MAGIC   ]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC The response is JSON — consumable by any actuarial workflow, dashboard, or downstream system.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Inspect Lineage in Unity Catalog
# MAGIC
# MAGIC Every step is captured in UC lineage automatically:
# MAGIC
# MAGIC ```
# MAGIC claims_time_series (Delta table)
# MAGIC        ↓ (training data)
# MAGIC MLflow Run [sarima_personal_auto_ontario_champion]
# MAGIC        ↓ (registered model)
# MAGIC {catalog}.{schema}.sarima_claims_forecaster @ Champion
# MAGIC        ↓ (serving)
# MAGIC actuarial-workshop-sarima-forecaster (endpoint)
# MAGIC        ↓ (request logs)
# MAGIC system.serving.served_entities_request_logs (audit trail)
# MAGIC ```

# COMMAND ----------

# Query endpoint request logs (available after first call)
# system.serving.served_entities_request_logs may not be enabled in all workspaces
try:
    spark.sql(f"""
        SELECT
            timestamp_ms,
            model_name,
            model_version,
            status_code,
            execution_time_ms,
            request_id
        FROM system.serving.served_entities_request_logs
        WHERE endpoint_name = '{ENDPOINT_NAME}'
        ORDER BY timestamp_ms DESC
        LIMIT 10
    """).display()
except Exception as e:
    print(f"Note: system.serving.served_entities_request_logs not available in this workspace: {e}")
    print("This is a workspace-level system table that may need to be enabled separately.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | What Happened |
# MAGIC |---|---|
# MAGIC | Log model | SARIMAX wrapped as PyFunc with signature → logged to MLflow experiment |
# MAGIC | Register | Model version created in `{catalog}.{schema}.sarima_claims_forecaster` |
# MAGIC | Alias | `@Champion` alias points to approved version — decouples serving from version numbers |
# MAGIC | Serve | Real-time REST endpoint with scale-to-zero, UC permissions, request logging |
# MAGIC | Lineage | Full chain: training data → experiment → model version → endpoint → logs |
# MAGIC
# MAGIC **Next:** Module 6 — Package everything as a Databricks Asset Bundle and wire into Azure DevOps CI/CD.