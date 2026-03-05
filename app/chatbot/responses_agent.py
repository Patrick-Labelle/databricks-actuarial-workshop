"""MLflow ResponsesAgent wrapper for the actuarial chatbot.

Registers the chatbot as a Databricks Agent so it appears in the
AI Gateway agents tab with automatic request/response logging,
tool call tracing, and usage metrics.

Usage:
    # In a registration notebook:
    import mlflow
    mlflow.pyfunc.log_model(
        artifact_path="actuarial_chatbot",
        python_model="app/chatbot/responses_agent.py",
        ...
    )
"""

import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatResponse, ChatMessage, ChatChoice


class ActuarialChatbotAgent(ChatModel):
    """MLflow ChatModel wrapper for the actuarial risk assistant.

    Wraps the existing tool-calling agent loop (DatabricksOpenAI + function tools)
    as a registered Databricks Agent for AI Gateway visibility.
    """

    def predict(self, context, messages, params=None):
        """Run the agent loop and return the final response.

        Args:
            context: MLflow model context
            messages: List of ChatMessage objects (conversation history)
            params: Optional ChatParams

        Returns:
            ChatResponse with the agent's final answer
        """
        from chatbot.agent import chat

        # Convert MLflow ChatMessage objects to dicts for the existing chat() API
        msg_dicts = []
        for m in messages:
            msg_dicts.append({"role": m.role, "content": m.content})

        # Collect all streamed output from the agent loop
        output_parts = []
        for chunk in chat(msg_dicts):
            output_parts.append(chunk)

        final_text = "".join(output_parts)

        return ChatResponse(
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=final_text),
                )
            ]
        )


# Required for code-based logging
mlflow.models.set_model(ActuarialChatbotAgent())
