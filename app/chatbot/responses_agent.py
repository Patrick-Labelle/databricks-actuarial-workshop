"""MLflow ChatModel wrapper for the actuarial chatbot.

Registers the chatbot as a Databricks Agent so it appears in the
AI Gateway agents tab with automatic request/response logging,
tool call tracing, and usage metrics.
"""

import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatCompletionResponse


class ActuarialChatbotAgent(ChatModel):
    """MLflow ChatModel wrapper for the actuarial risk assistant."""

    def predict(self, context, messages, params=None):
        """Run the agent loop and return the final response."""
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


# Required for code-based logging
mlflow.models.set_model(ActuarialChatbotAgent())
