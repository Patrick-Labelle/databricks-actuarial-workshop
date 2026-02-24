# Databricks notebook source
# MAGIC %md
# MAGIC # Module 6: Monte Carlo Portfolio Risk — Model Serving Endpoint
# MAGIC ## t-Copula Simulation as a REST API via MLflow PyFunc
# MAGIC
# MAGIC **Workshop: Statistical Modeling at Scale on Databricks**
# MAGIC
# MAGIC ---
# MAGIC ### What We'll Cover
# MAGIC 1. **Wrap the Simulation** — Package t-Copula Monte Carlo as `mlflow.pyfunc.PythonModel`
# MAGIC 2. **Parameterise the Endpoint** — Accept scenario assumptions at request time (means, CVs, correlations)
# MAGIC 3. **Register to UC** — Governed model with `Champion` alias
# MAGIC 4. **Deploy CPU Endpoint** — REST endpoint for on-demand scenario analysis
# MAGIC 5. **Call the Endpoint** — Demonstrate a stressed-scenario capital calculation
# MAGIC
# MAGIC ---
# MAGIC ### Design Rationale
# MAGIC
# MAGIC The SARIMA endpoint (Module 5) was **parameter-free** — the model was fitted once and the
# MAGIC endpoint simply called `get_forecast(steps=N)`. The Monte Carlo endpoint is fundamentally
# MAGIC different: the simulation is **parameterised at inference time**. Users can send:
# MAGIC
# MAGIC - **Stressed means** — simulate a hard market with 20% higher loss costs
# MAGIC - **Increased CVs** — model parameter uncertainty for ORSA sensitivity tests
# MAGIC - **Modified correlations** — stress-test a cat scenario with elevated inter-line dependency
# MAGIC
# MAGIC The endpoint runs a fresh Monte Carlo on every call. This requires no stored model state —
# MAGIC just the simulation code. CPU compute is sufficient (scipy t-CDF is not GPU-accelerated).
# MAGIC
# MAGIC | Aspect | SARIMA endpoint (Module 5) | Monte Carlo endpoint (Module 6) |
# MAGIC |---|---|---|
# MAGIC | Model state | Fitted ARIMA parameters (pickle) | None — simulation is stateless |
# MAGIC | Input | `horizon` (int) | 11 scenario parameters |
# MAGIC | Output | Monthly forecast table | Portfolio risk metrics (VaR, CVaR) |
# MAGIC | Compute | CPU, ~5 ms | CPU, ~500 ms – 5 s (scales with n_scenarios) |
# MAGIC | Primary use | Renewal pricing, reserving | Capital modelling, ORSA, stress testing |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import scipy
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
dbutils.widgets.text("catalog",        "my_catalog",                          "UC Catalog")
dbutils.widgets.text("schema",         "actuarial_workshop",                  "UC Schema")
dbutils.widgets.text("mc_endpoint_name", "actuarial-workshop-monte-carlo",    "MC Endpoint Name")

CATALOG          = dbutils.widgets.get("catalog")
SCHEMA           = dbutils.widgets.get("schema")
MC_ENDPOINT_NAME = dbutils.widgets.get("mc_endpoint_name")
MODEL_NAME       = f"{CATALOG}.{SCHEMA}.monte_carlo_portfolio"

mlflow.set_registry_uri("databricks-uc")

print(f"MLflow version:   {mlflow.__version__}")
print(f"Model name:       {MODEL_NAME}")
print(f"Endpoint:         {MC_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define the PyFunc Model
# MAGIC
# MAGIC The Monte Carlo simulation has no stored parameters — all assumptions are passed in the request.
# MAGIC The `predict()` method runs the full t-Copula simulation on each call.
# MAGIC
# MAGIC **Input schema** (one row per scenario request):
# MAGIC
# MAGIC | Parameter | Type | Default | Description |
# MAGIC |---|---|---|---|
# MAGIC | `mean_property_M` | double | 12.5 | Expected annual loss for Commercial Property ($M) |
# MAGIC | `mean_auto_M` | double | 8.3 | Expected annual loss for Commercial Auto ($M) |
# MAGIC | `mean_liability_M` | double | 5.7 | Expected annual loss for Liability ($M) |
# MAGIC | `cv_property` | double | 0.35 | Coefficient of variation for Property |
# MAGIC | `cv_auto` | double | 0.28 | CV for Auto |
# MAGIC | `cv_liability` | double | 0.42 | CV for Liability |
# MAGIC | `corr_prop_auto` | double | 0.40 | Correlation: Property ↔ Auto |
# MAGIC | `corr_prop_liab` | double | 0.20 | Correlation: Property ↔ Liability |
# MAGIC | `corr_auto_liab` | double | 0.30 | Correlation: Auto ↔ Liability |
# MAGIC | `n_scenarios` | long | 10000 | Number of Monte Carlo paths |
# MAGIC | `copula_df` | long | 4 | t-Copula degrees of freedom |

# COMMAND ----------

class MonteCarloPyFunc(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for t-Copula + Lognormal Marginals Monte Carlo.

    Parameterised simulation: all assumptions arrive in the request, so analysts
    can run stressed scenarios (hard market, cat event, parameter uncertainty)
    without retraining.

    Actuarial design:
      - t-Copula (df=4): captures tail dependence / common shocks between lines
      - Lognormal marginals: consistent with the collective risk model (Panjer,
        Klugman) for skewed, non-negative insurance losses
      - Cholesky decomposition: enforces positive semi-definite correlation structure

    Input:  DataFrame with one row of scenario parameters (see schema above)
    Output: DataFrame with one row of portfolio risk metrics
    """

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        from scipy.stats import t as tdist, norm as scipy_norm

        row = model_input.iloc[0]

        # ── Read scenario parameters (with defaults) ──────────────────────────
        means = np.array([
            float(row.get("mean_property_M",  12.5)),
            float(row.get("mean_auto_M",       8.3)),
            float(row.get("mean_liability_M",  5.7)),
        ])
        cv = np.array([
            float(row.get("cv_property",  0.35)),
            float(row.get("cv_auto",      0.28)),
            float(row.get("cv_liability", 0.42)),
        ])
        corr_prop_auto = float(row.get("corr_prop_auto",  0.40))
        corr_prop_liab = float(row.get("corr_prop_liab",  0.20))
        corr_auto_liab = float(row.get("corr_auto_liab",  0.30))
        n_scenarios    = int(row.get("n_scenarios", 10_000))
        copula_df      = int(row.get("copula_df",   4))

        # Safety bounds: guard against very large or degenerate inputs
        n_scenarios = max(1_000, min(n_scenarios, 100_000))
        copula_df   = max(2,     min(copula_df,   30))
        means       = np.clip(means, 0.01, 1_000.0)
        cv          = np.clip(cv,    0.01, 5.0)
        corr_prop_auto = np.clip(corr_prop_auto, -0.99, 0.99)
        corr_prop_liab = np.clip(corr_prop_liab, -0.99, 0.99)
        corr_auto_liab = np.clip(corr_auto_liab, -0.99, 0.99)

        # ── Lognormal parameters ──────────────────────────────────────────────
        # E[X] = means, CV = cv  →  sigma2 = log(1 + CV^2),  mu = log(E[X]) - sigma2/2
        sigma2   = np.log(1 + cv**2)
        mu_ln    = np.log(means) - sigma2 / 2
        sigma_ln = np.sqrt(sigma2)

        # ── Correlation matrix (Cholesky decomposition) ───────────────────────
        corr = np.array([
            [1.0,            corr_prop_auto, corr_prop_liab],
            [corr_prop_auto, 1.0,            corr_auto_liab],
            [corr_prop_liab, corr_auto_liab, 1.0           ],
        ])
        # Nearest PSD fix: clip off-diagonals if needed
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            # Nearest SPD via eigenvalue floor
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = np.maximum(eigvals, 1e-8)
            corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
            np.fill_diagonal(corr, 1.0)
            chol = np.linalg.cholesky(corr)

        # ── t-Copula simulation (Sklar's theorem) ────────────────────────────
        # 1. Draw correlated normals  z ~ N(0, I)
        # 2. Apply Cholesky to correlate:  x_cor = z @ chol.T
        # 3. Mix with chi2:  t_cor = x_cor / sqrt(chi2 / df)  →  multivariate t
        # 4. t-CDF  →  uniform marginals  (Sklar)
        # 5. Inverse lognormal CDF  →  loss sample
        rng   = np.random.default_rng(42)
        z     = rng.standard_normal((n_scenarios, 3))
        chi2  = rng.chisquare(copula_df, n_scenarios)
        x_cor = z @ chol.T
        t_cor = x_cor / np.sqrt(chi2[:, None] / copula_df)
        u     = tdist.cdf(t_cor, df=copula_df)
        q     = scipy_norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        losses = np.exp(mu_ln + sigma_ln * q)
        total  = losses.sum(axis=1)

        # ── Risk metrics ──────────────────────────────────────────────────────
        var_99_threshold = np.percentile(total, 99)

        return pd.DataFrame([{
            "expected_loss_M":  round(float(total.mean()), 3),
            "var_95_M":         round(float(np.percentile(total, 95)),  3),
            "var_99_M":         round(float(np.percentile(total, 99)),  3),
            "var_995_M":        round(float(np.percentile(total, 99.5)), 3),
            "cvar_99_M":        round(float(total[total >= var_99_threshold].mean()), 3),
            "max_loss_M":       round(float(total.max()), 3),
            "n_scenarios_used": n_scenarios,
            "copula":           f"t-copula(df={copula_df})",
        }])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validate the Model Locally
# MAGIC
# MAGIC Run a quick sanity check with baseline parameters before registering.

# COMMAND ----------

# Baseline scenario (matches Module 4 calibration)
_baseline_input = pd.DataFrame([{
    "mean_property_M":  12.5,
    "mean_auto_M":       8.3,
    "mean_liability_M":  5.7,
    "cv_property":       0.35,
    "cv_auto":           0.28,
    "cv_liability":      0.42,
    "corr_prop_auto":    0.40,
    "corr_prop_liab":    0.20,
    "corr_auto_liab":    0.30,
    "n_scenarios":    10_000,
    "copula_df":           4,
}])

_model = MonteCarloPyFunc()
_baseline_result = _model.predict(None, _baseline_input)
print("Baseline simulation (10,000 scenarios):")
print(_baseline_result.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log, Register, and Set Champion Alias

# COMMAND ----------

import os, tempfile

_current_user = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{_current_user}/actuarial_workshop_monte_carlo_portfolio")

# Dataset reference for UC lineage
_claims_dataset = mlflow.data.load_delta(
    table_name=f"{CATALOG}.{SCHEMA}.claims_time_series",
    name="claims_time_series",
)

# ── Define MLflow signature ───────────────────────────────────────────────────
input_schema = mlflow.types.Schema([
    mlflow.types.ColSpec("double",  "mean_property_M"),
    mlflow.types.ColSpec("double",  "mean_auto_M"),
    mlflow.types.ColSpec("double",  "mean_liability_M"),
    mlflow.types.ColSpec("double",  "cv_property"),
    mlflow.types.ColSpec("double",  "cv_auto"),
    mlflow.types.ColSpec("double",  "cv_liability"),
    mlflow.types.ColSpec("double",  "corr_prop_auto"),
    mlflow.types.ColSpec("double",  "corr_prop_liab"),
    mlflow.types.ColSpec("double",  "corr_auto_liab"),
    mlflow.types.ColSpec("long",    "n_scenarios"),
    mlflow.types.ColSpec("long",    "copula_df"),
])
output_schema = mlflow.types.Schema([
    mlflow.types.ColSpec("double", "expected_loss_M"),
    mlflow.types.ColSpec("double", "var_95_M"),
    mlflow.types.ColSpec("double", "var_99_M"),
    mlflow.types.ColSpec("double", "var_995_M"),
    mlflow.types.ColSpec("double", "cvar_99_M"),
    mlflow.types.ColSpec("double", "max_loss_M"),
    mlflow.types.ColSpec("long",   "n_scenarios_used"),
    mlflow.types.ColSpec("string", "copula"),
])
signature = mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)

# ── Train run ─────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="monte_carlo_portfolio_champion") as run:
    mlflow.log_input(_claims_dataset, context="training")

    mlflow.set_tags({
        "model_class":     "MonteCarloPyFunc",
        "copula":          "t-copula",
        "marginals":       "lognormal",
        "workshop_module": "6",
        "audience":        "actuarial-workshop",
    })
    mlflow.log_params({
        "copula_df":              4,
        "n_lines":                3,
        "default_n_scenarios":    10_000,
        "mean_property_M_base":   12.5,
        "mean_auto_M_base":        8.3,
        "mean_liability_M_base":   5.7,
        "cv_property_base":       0.35,
        "cv_auto_base":           0.28,
        "cv_liability_base":      0.42,
        "corr_prop_auto_base":    0.40,
        "corr_prop_liab_base":    0.20,
        "corr_auto_liab_base":    0.30,
    })

    # Log baseline metrics for experiment tracking
    _r = _baseline_result.iloc[0]
    mlflow.log_metrics({
        "baseline_expected_loss_M":  float(_r["expected_loss_M"]),
        "baseline_var_95_M":         float(_r["var_95_M"]),
        "baseline_var_99_M":         float(_r["var_99_M"]),
        "baseline_var_995_M":        float(_r["var_995_M"]),
        "baseline_cvar_99_M":        float(_r["cvar_99_M"]),
    })

    # Log the PyFunc model (no pickle — simulation is stateless)
    model_info = mlflow.pyfunc.log_model(
        artifact_path="monte_carlo_pyfunc",
        python_model=MonteCarloPyFunc(),
        signature=signature,
        registered_model_name=MODEL_NAME,
        pip_requirements=[
            f"scipy=={scipy.__version__}",
            f"numpy=={np.__version__}",
        ],
    )

    RUN_ID = run.info.run_id
    print(f"\nModel registered to: {MODEL_NAME}")
    print(f"Run ID:  {RUN_ID}")
    print(f"Baseline VaR(99%): ${_r['var_99_M']:.1f}M  |  CVaR(99%): ${_r['cvar_99_M']:.1f}M")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Set `Champion` Alias

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(int(v.version) for v in versions)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=latest_version,
)
print(f"Set @Champion → version {latest_version}")

client.set_model_version_tag(
    name=MODEL_NAME, version=str(latest_version),
    key="approved_by", value="actuarial-workshop-demo",
)
client.set_model_version_tag(
    name=MODEL_NAME, version=str(latest_version),
    key="simulation_type", value="t-copula-lognormal",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy CPU Model Serving Endpoint
# MAGIC
# MAGIC The Monte Carlo simulation runs entirely on NumPy and SciPy — no GPU acceleration
# MAGIC is needed for t-CDF evaluation. A CPU endpoint is sufficient and more cost-effective.
# MAGIC
# MAGIC **Scale-to-zero** is enabled: the endpoint idles when not in use and wakes on first request.
# MAGIC Cold-start latency is ~30–60 seconds; subsequent calls return in ~1–5 seconds depending on
# MAGIC the number of scenarios requested.

# COMMAND ----------

import requests as _req, json as _json

WORKSPACE_URL = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .apiToken().get()
)

_headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

endpoint_config = {
    "name": MC_ENDPOINT_NAME,
    "config": {
        "served_models": [{
            "name":                  "monte-carlo-champion",
            "model_name":            MODEL_NAME,
            "model_version":         str(latest_version),
            "workload_size":         "Small",
            "scale_to_zero_enabled": True,
            # CPU compute — scipy.stats.t.cdf is not GPU-accelerated
        }]
    },
}

resp = _req.get(
    f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{MC_ENDPOINT_NAME}",
    headers=_headers,
)

if resp.status_code == 200:
    resp = _req.put(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints/{MC_ENDPOINT_NAME}/config",
        headers=_headers,
        json=endpoint_config["config"],
    )
    print(f"Endpoint updated: {resp.status_code}")
else:
    resp = _req.post(
        f"https://{WORKSPACE_URL}/api/2.0/serving-endpoints",
        headers=_headers,
        json=endpoint_config,
    )
    print(f"Endpoint created: {resp.status_code}")

if resp.status_code in (200, 201):
    print(f"\nEndpoint URL:\n  https://{WORKSPACE_URL}/serving-endpoints/{MC_ENDPOINT_NAME}/invocations")
    print("Note: Endpoint takes ~5 minutes to reach 'Ready' state.")
else:
    print(f"Error: {resp.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Demo: Baseline vs Stressed Scenario
# MAGIC
# MAGIC Call the endpoint with two scenarios to demonstrate the capital impact of a stressed assumption:
# MAGIC - **Baseline:** Calibrated parameters from Module 4
# MAGIC - **Stressed:** +20% mean losses (hard market), elevated inter-line correlation (cat scenario)

# COMMAND ----------

import time as _time

def call_mc_endpoint(scenario_params: dict) -> dict:
    """Call the Monte Carlo endpoint. Returns result dict or error."""
    payload = {"dataframe_records": [scenario_params]}
    try:
        resp = _req.post(
            f"https://{WORKSPACE_URL}/serving-endpoints/{MC_ENDPOINT_NAME}/invocations",
            headers=_headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text, "status_code": resp.status_code}
    except _req.exceptions.Timeout:
        return {"note": "Endpoint still warming up (~5 min after creation). Re-run once READY."}
    except Exception as exc:
        return {"error": str(exc)}


_baseline_params = {
    "mean_property_M": 12.5, "mean_auto_M": 8.3, "mean_liability_M": 5.7,
    "cv_property": 0.35, "cv_auto": 0.28, "cv_liability": 0.42,
    "corr_prop_auto": 0.40, "corr_prop_liab": 0.20, "corr_auto_liab": 0.30,
    "n_scenarios": 10000, "copula_df": 4,
}

_stressed_params = {
    "mean_property_M": 15.0,  # +20% — hard market
    "mean_auto_M": 9.96,      # +20%
    "mean_liability_M": 6.84, # +20%
    "cv_property": 0.40,      # elevated uncertainty
    "cv_auto": 0.33,
    "cv_liability": 0.50,
    "corr_prop_auto": 0.55,   # elevated cat correlation
    "corr_prop_liab": 0.35,
    "corr_auto_liab": 0.45,
    "n_scenarios": 10000, "copula_df": 4,
}

print("Calling endpoint with baseline scenario...")
_b = call_mc_endpoint(_baseline_params)
print("Calling endpoint with stressed scenario (+20% loss costs, elevated correlations)...")
_s = call_mc_endpoint(_stressed_params)

for label, result in [("Baseline", _b), ("Stressed (+20% / cat correlations)", _s)]:
    print(f"\n{label}:")
    if "predictions" in result:
        pred = result["predictions"][0] if isinstance(result["predictions"], list) else result["predictions"]
        print(f"  E[Loss]:    ${pred.get('expected_loss_M', 'N/A'):.1f}M")
        print(f"  VaR(99%):   ${pred.get('var_99_M', 'N/A'):.1f}M")
        print(f"  VaR(99.5%): ${pred.get('var_995_M', 'N/A'):.1f}M")
        print(f"  CVaR(99%):  ${pred.get('cvar_99_M', 'N/A'):.1f}M")
    else:
        print(f"  {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | What Happened |
# MAGIC |---|---|
# MAGIC | PyFunc | t-Copula + Lognormal simulation wrapped as a stateless `predict()` function |
# MAGIC | Lineage | `claims_time_series` → MLflow run → `monte_carlo_portfolio@Champion` |
# MAGIC | Endpoint | CPU REST endpoint — `actuarial-workshop-monte-carlo` |
# MAGIC | Scenario API | 11 parameters: means, CVs, correlations, n_scenarios, copula_df |
# MAGIC | Use cases | Hard market analysis, cat scenario stress testing, ORSA capital sensitivity |
# MAGIC
# MAGIC **Integration:** The Databricks App (Module 7) calls this endpoint from the
# MAGIC "Scenario Analysis" tab — analysts can run custom stress scenarios interactively
# MAGIC without cluster access.
