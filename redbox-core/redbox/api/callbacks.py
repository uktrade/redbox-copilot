from logging import Logger
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs.llm_result import LLMResult


class LoggerCallbackHandler(BaseCallbackHandler):
    def __init__(self, logger: Logger):
        self.logger: Logger = logger

    def redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive information such as API keys."""
        repr_str = data['repr']

        start_token = "openai_api_key='"
        end_token = "',"

        start_index = repr_str.find(start_token) + len(start_token)
        end_index = repr_str.find(end_token, start_index)

        current_openai_api_key = repr_str[start_index:end_index]

        new_repr_str = repr_str.replace(f"openai_api_key='{current_openai_api_key}'", "openai_api_key='<REDACTED>'")

        data['repr'] = new_repr_str

        return data

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:  # noqa:ARG002
        """Run when LLM starts running."""
        redacted_serialized = self.redact_secrets(serialized)
        self.logger.info("LLM start: %s, prompts: %s", redacted_serialized, prompts)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:  # noqa:ARG002
        """Run when LLM ends running."""
        self.logger.info("LLM end: %s", response)

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:  # noqa:ARG002
        """Run when chain starts running."""
        self.logger.info("Chain start: %s, inputs: %s", serialized, inputs)

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:  # noqa:ARG002
        """Run when chain ends running."""
        self.logger.info("Chain end: %s", outputs)

    def on_chain_error(self, error: BaseException, **kwargs: Any):  # noqa:ARG002
        """Run when chain errors."""
        self.logger.error("Chain error: %s", error)

    def on_text(self, text: str, **kwargs: Any) -> None:  # noqa:ARG002
        """Run on arbitrary text."""
        self.logger.info("Text: %s", text)
