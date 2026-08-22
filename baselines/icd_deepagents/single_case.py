from __future__ import annotations

from typing import Any

from baselines.icd_react.config import ICDReactConfig
from baselines.icd_react.single_case import (
    coerce_icd_code_list,
    extract_json_object,
    fetch_case_record,
    message_content_to_text,
    normalize_prediction,
    score_prediction,
    serialize_agent_messages,
)
from baselines.icd_react_v2.single_case import build_single_case_prompt, resolve_prompt_suffix

from .agent import ICDPrediction, build_icd_deep_agent


DEFAULT_MAX_AGENT_STEPS = 60


def _structured_prediction(result: dict[str, Any]) -> dict[str, Any]:
    structured = result.get("structured_response")
    if isinstance(structured, ICDPrediction):
        return structured.model_dump()
    if isinstance(structured, dict):
        return structured
    if structured is not None and hasattr(structured, "model_dump"):
        return structured.model_dump()

    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("ICD Deep Agents run returned neither structured output nor messages.")
    return extract_json_object(message_content_to_text(messages[-1].content))


def run_single_case_prediction(
    config: ICDReactConfig,
    summary_text: str,
    model_name: str,
    prompt_variant: str = "active_first_no_repeat",
    prompt_suffix: str | None = None,
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
    model_provider: str | None = None,
) -> dict[str, Any]:
    agent = build_icd_deep_agent(model=model_name, config=config, model_provider=model_provider)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": build_single_case_prompt(
                        summary_text=summary_text,
                        prompt_suffix=resolve_prompt_suffix(prompt_variant=prompt_variant, prompt_suffix=prompt_suffix),
                    ),
                }
            ]
        },
        config={"recursion_limit": max_agent_steps},
    )
    prediction = normalize_prediction(_structured_prediction(result))
    return {
        "prediction": prediction,
        "messages": serialize_agent_messages(result.get("messages", [])),
        "structured_response": prediction,
    }


__all__ = [
    "DEFAULT_MAX_AGENT_STEPS",
    "coerce_icd_code_list",
    "fetch_case_record",
    "run_single_case_prediction",
    "score_prediction",
]
