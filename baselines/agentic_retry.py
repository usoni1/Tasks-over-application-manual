from __future__ import annotations

from typing import Any


DEFAULT_MODEL_RETRY_ATTEMPTS = 6


def build_retryable_chat_model(
    model_name: str,
    stop_after_attempt: int = DEFAULT_MODEL_RETRY_ATTEMPTS,
    temperature: float = 0.0,
) -> Any:
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model_name,
        temperature=temperature,
        max_retries=max(int(stop_after_attempt), 1),
    )


__all__ = [
    "DEFAULT_MODEL_RETRY_ATTEMPTS",
    "build_retryable_chat_model",
]