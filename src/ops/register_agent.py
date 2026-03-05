# Databricks notebook source
# MAGIC %md
# MAGIC # Register Actuarial Chatbot as Databricks Agent
# MAGIC
# MAGIC Registers the chatbot as a proper Databricks Agent via MLflow so it appears
# MAGIC in the AI Gateway agents tab with automatic request/response logging,
# MAGIC tool call tracing, and usage metrics.

# COMMAND ----------

dbutils.widgets.text("catalog", "my_catalog", "UC Catalog")
dbutils.widgets.text("schema", "actuarial_workshop", "UC Schema")
dbutils.widgets.text("endpoint_name", "actuarial-workshop-sarima-forecaster", "SARIMA endpoint")
dbutils.widgets.text("mc_endpoint_name", "actuarial-workshop-monte-carlo", "MC endpoint")
dbutils.widgets.text("warehouse_id", "", "SQL Warehouse ID")
dbutils.widgets.text("llm_endpoint_name", "databricks-meta-llama-3-3-70b-instruct", "LLM endpoint")
dbutils.widgets.text("genie_space_id", "", "Genie Space ID")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
MC_ENDPOINT_NAME = dbutils.widgets.get("mc_endpoint_name")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")
LLM_ENDPOINT_NAME = dbutils.widgets.get("llm_endpoint_name")
GENIE_SPACE_ID = dbutils.widgets.get("genie_space_id")

AGENT_MODEL_NAME = f"{CATALOG}.{SCHEMA}.actuarial_chatbot_agent"
AGENT_ENDPOINT_NAME = "actuarial-workshop-chatbot-agent"

print(f"Catalog/Schema:     {CATALOG}.{SCHEMA}")
print(f"Agent model:        {AGENT_MODEL_NAME}")
print(f"Agent endpoint:     {AGENT_ENDPOINT_NAME}")

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

CURRENT_USER = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{CURRENT_USER}/actuarial_workshop_chatbot_agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Agent Model
# MAGIC
# MAGIC Uses code-based logging — the `responses_agent.py` module defines the
# MAGIC `ActuarialChatbotAgent` ChatModel class and calls `mlflow.models.set_model()`.

# COMMAND ----------

import os

# The agent code lives in app/chatbot/responses_agent.py relative to the repo root.
# For code-based logging, we point to the Python file and specify its dependencies.
_agent_code_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "app", "chatbot", "responses_agent.py",
)

# Resources the agent needs — passed as environment variables to the serving endpoint
_resources = [
    mlflow.models.resources.DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    mlflow.models.resources.DatabricksServingEndpoint(endpoint_name=ENDPOINT_NAME),
    mlflow.models.resources.DatabricksServingEndpoint(endpoint_name=MC_ENDPOINT_NAME),
    mlflow.models.resources.DatabricksSQLWarehouse(warehouse_id=WAREHOUSE_ID),
]
if GENIE_SPACE_ID:
    _resources.append(
        mlflow.models.resources.DatabricksGenieSpace(genie_space_id=GENIE_SPACE_ID),
    )

with mlflow.start_run(run_name="chatbot_agent_registration") as _run:
    mlflow.set_tags({
        "workshop_module": "ops",
        "model_class": "ActuarialChatbotAgent",
        "audience": "actuarial-workshop",
    })

    model_info = mlflow.pyfunc.log_model(
        artifact_path="actuarial_chatbot",
        python_model=_agent_code_path,
        registered_model_name=AGENT_MODEL_NAME,
        resources=_resources,
        pip_requirements=[
            "mlflow>=3.0",
            "databricks-openai>=0.11.0",
            "databricks-sdk>=0.81",
            "pandas>=2.0",
            "psycopg2-binary>=2.9",
        ],
        code_paths=[
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app"),
        ],
    )

    print(f"Agent model logged: {model_info.model_uri}")
    print(f"Registered to: {AGENT_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set @Champion Alias

# COMMAND ----------

_client = MlflowClient()
_versions = _client.search_model_versions(f"name='{AGENT_MODEL_NAME}'")
_latest_ver = max(int(v.version) for v in _versions)
_client.set_registered_model_alias(
    name=AGENT_MODEL_NAME, alias="Champion", version=_latest_ver,
)
print(f"Set @Champion → version {_latest_ver}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Agent Endpoint
# MAGIC
# MAGIC Creates (or updates) a serving endpoint for the registered agent.
# MAGIC The endpoint appears in the AI Gateway agents tab automatically.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

try:
    w.serving_endpoints.create_and_wait(
        name=AGENT_ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=AGENT_MODEL_NAME,
                    entity_version=str(_latest_ver),
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                )
            ],
        ),
    )
    print(f"Agent endpoint created: {AGENT_ENDPOINT_NAME}")
except Exception as e:
    if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
        # Update existing endpoint
        w.serving_endpoints.update_config_and_wait(
            name=AGENT_ENDPOINT_NAME,
            served_entities=[
                ServedEntityInput(
                    entity_name=AGENT_MODEL_NAME,
                    entity_version=str(_latest_ver),
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                )
            ],
        )
        print(f"Agent endpoint updated: {AGENT_ENDPOINT_NAME}")
    else:
        print(f"Agent endpoint creation failed: {e}")
        raise

print(f"\nAgent registration complete.")
print(f"  Model:    {AGENT_MODEL_NAME} @Champion (v{_latest_ver})")
print(f"  Endpoint: {AGENT_ENDPOINT_NAME}")
print(f"  The agent is now visible in AI Gateway → Agents tab.")
