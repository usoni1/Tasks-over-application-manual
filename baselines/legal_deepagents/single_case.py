from __future__ import annotations

from typing import Any

from baselines.legal_react_v2.config import LegalReactV2Config
from baselines.legal_react_v2.single_case import (
    build_score_payload,
    build_single_case_prompt,
    fetch_case_record,
    message_content_to_text,
    normalize_prediction,
    score_prediction,
    serialize_agent_messages,
)

from .agent import OffenseLevelPrediction, build_legal_deep_agent


DEFAULT_MAX_AGENT_STEPS = 60


def _structured_prediction(result: dict[str, Any]) -> dict[str, Any]:
    structured = result.get("structured_response")
    if isinstance(structured, OffenseLevelPrediction):
        return structured.model_dump()
    if isinstance(structured, dict):
        return structured
    if structured is not None and hasattr(structured, "model_dump"):
        return structured.model_dump()

    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Legal Deep Agents run returned neither structured output nor messages.")
    raise RuntimeError(f"Legal Deep Agents run did not return structured output: {message_content_to_text(messages[-1].content)!r}")


def run_single_case_prediction(
    config: LegalReactV2Config,
    summary_text: str,
    model_name: str,
    year: int | None = None,
    case_record: dict[str, Any] | None = None,
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
    model_provider: str | None = None,
) -> dict[str, Any]:
    agent = build_legal_deep_agent(model=model_name, config=config, model_provider=model_provider)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": build_single_case_prompt(summary_text=summary_text, year=year, case_record=case_record),
                }
            ]
        },
        config={"recursion_limit": max_agent_steps},
    )
    prediction = normalize_prediction(_structured_prediction(result))
    return {
        "prediction": prediction,
        "score_payload": build_score_payload(prediction),
        "messages": serialize_agent_messages(result.get("messages", [])),
        "structured_response": prediction,
    }


__all__ = ["DEFAULT_MAX_AGENT_STEPS", "fetch_case_record", "run_single_case_prediction", "score_prediction"]
