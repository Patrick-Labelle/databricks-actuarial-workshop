# Databricks notebook source
# MAGIC %md
# MAGIC # Register Actuarial Chatbot as Databricks Agent
# MAGIC
# MAGIC Registers the chatbot as a proper Databricks Agent via MLflow so it appears
# MAGIC in the AI Gateway agents tab with automatic request/response logging,
# MAGIC tool call tracing, and usage metrics.

# COMMAND ----------

%pip install mlflow>=3.1.3 databricks-agents>=1.1.0 --quiet
dbutils.library.restartPython()

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
if not GENIE_SPACE_ID:
    try:
        GENIE_SPACE_ID = dbutils.jobs.taskValues.get(
            taskKey="prepare_app_infrastructure", key="genie_space_id"
        )
        print(f"  Resolved genie_space_id from upstream task: {GENIE_SPACE_ID}")
    except Exception:
        pass

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
# MAGIC Defines the ChatModel inline and logs it as a pyfunc model.
# MAGIC The agent wraps the existing tool-calling loop from `app/chatbot/agent.py`.

# COMMAND ----------

from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatCompletionResponse


class ActuarialChatbotAgent(ChatModel):
    """MLflow ChatModel wrapper for the actuarial risk assistant."""

    def predict(self, context, messages, params=None):
        # Guard import — chatbot module is only available at serving time
        # (bundled via code_paths), not during log_model signature inference.
        try:
            from chatbot.agent import chat
        except (ImportError, ModuleNotFoundError):
            return ChatCompletionResponse.from_dict({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Agent not initialized"},
                }]
            })

        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        output_parts = []
        for chunk in chat(msg_dicts):
            output_parts.append(chunk)

        return ChatCompletionResponse.from_dict({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(output_parts)},
            }]
        })


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

# Resolve the app directory from workspace paths for code_paths
_nb_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_nb_path = _nb_ctx.notebookPath().get()
# _nb_path: .../files/src/ops/register_agent → repo root: .../files
import os
_repo_root = "/Workspace" + os.path.dirname(os.path.dirname(os.path.dirname(_nb_path)))
_app_dir = os.path.join(_repo_root, "app")
print(f"Notebook path: {_nb_path}")
print(f"App dir (code_paths): {_app_dir}")

with mlflow.start_run(run_name="chatbot_agent_registration") as _run:
    mlflow.set_tags({
        "workshop_module": "ops",
        "model_class": "ActuarialChatbotAgent",
        "audience": "actuarial-workshop",
    })

    model_info = mlflow.pyfunc.log_model(
        artifact_path="actuarial_chatbot",
        python_model=ActuarialChatbotAgent(),
        registered_model_name=AGENT_MODEL_NAME,
        resources=_resources,
        pip_requirements=[
            "mlflow>=3.1.3",
            "databricks-agents>=1.1.0",
            "databricks-openai>=0.11.0",
            "databricks-sdk>=0.81",
            "pandas>=2.0",
            "psycopg2-binary>=2.9",
        ],
        code_paths=[_app_dir],
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
# MAGIC Uses `databricks.agents.deploy()` which automatically sets up:
# MAGIC - Serving endpoint (appears in AI Gateway agents tab)
# MAGIC - Review App for stakeholder feedback
# MAGIC - Inference tables for request/response logging
# MAGIC - Real-time MLflow tracing

# COMMAND ----------

from databricks import agents

deployment = agents.deploy(
    AGENT_MODEL_NAME,
    _latest_ver,
    endpoint_name=AGENT_ENDPOINT_NAME,
    scale_to_zero=True,
)

print(f"\nAgent deployment complete.")
print(f"  Model:      {AGENT_MODEL_NAME} @Champion (v{_latest_ver})")
print(f"  Endpoint:   {deployment.endpoint_name}")
print(f"  Review App: {deployment.review_app_url}")
