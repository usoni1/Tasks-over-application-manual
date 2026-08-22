from __future__ import annotations

import json
import re
from typing import Any

from baselines.legal_rag.single_case import fetch_case_record, normalize_text_field, score_prediction as score_offense_level_prediction

from .agent import build_legal_agent
from .config import LegalReactV2Config


DEFAULT_MAX_AGENT_STEPS = 60
PROMPT_SAFE_CASE_FIELDS = {
    "acceptance_of_responsibility",
}


def normalize_case_prompt_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
        return stripped
    return value


def build_case_record_prompt_context(case_record: dict[str, Any] | None) -> str:
    if not case_record:
        return ""

    visible_fields: dict[str, Any] = {}
    for key, value in case_record.items():
        if key not in PROMPT_SAFE_CASE_FIELDS:
            continue
        normalized_value = normalize_case_prompt_value(value)
        if normalized_value in (None, "", [], {}):
            continue
        visible_fields[key] = normalized_value

    if not visible_fields:
        return ""

    return json.dumps(visible_fields, ensure_ascii=False, indent=2, sort_keys=True)


def build_single_case_prompt(summary_text: str, year: int | None, case_record: dict[str, Any] | None = None) -> str:
    year_instruction = f"Use the {int(year)} edition of the legal manuals.\n" if year is not None else ""
    structured_case_context = build_case_record_prompt_context(case_record)
    structured_case_section = (
        "\n\nStructured case record fields (non-target):\n"
        f"{structured_case_context}"
        if structured_case_context
        else ""
    )
    return (
        "You are reviewing one federal sentencing case.\n\n"
        "Task:\n"
        f"- {year_instruction}".replace("- ", "", 1)
        + "- Compute the final total offense level supported by the case facts and the legal manuals.\n"
        + "- Follow the required sentencing workflow in the system instructions.\n"
        + "- Return only valid JSON matching this schema exactly:\n"
        + "{\n"
        + '  "offense_level": 0,\n'
        + '  "justifications": ["short manual-grounded justification 1", "short manual-grounded justification 2"]\n'
        + "}\n"
        + "- If the final total offense level cannot be determined from the available facts and inspected manual text, return null for offense_level.\n\n"
        + "Case summary:\n"
        + f"{summary_text.strip()}"
        + structured_case_section
    )


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    parts.append(str(text_value))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Agent output did not contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Agent output JSON must decode to an object.")
    return payload


def normalize_offense_level(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    match = re.fullmatch(r"[-+]?\d+", text)
    if not match:
        return None
    return int(text)


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(prediction)
    normalized["offense_level"] = normalize_offense_level(prediction.get("offense_level"))
    justifications = prediction.get("justifications")
    if isinstance(justifications, list):
        normalized["justifications"] = [str(item).strip() for item in justifications if str(item).strip()]
    else:
        normalized["justifications"] = []
    return normalized


def serialize_agent_messages(messages: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        serialized.append(
            {
                "type": message.__class__.__name__,
                "content": message_content_to_text(getattr(message, "content", "")),
                "tool_calls": getattr(message, "tool_calls", None),
                "name": getattr(message, "name", None),
            }
        )
    return serialized


def build_score_payload(prediction: dict[str, Any]) -> dict[str, Any]:
    offense_level = prediction.get("offense_level")
    return {
        "predicted_offense_level_total": None if offense_level is None else str(offense_level),
    }


def run_single_case_prediction(
    config: LegalReactV2Config,
    summary_text: str,
    model_name: str,
    year: int | None = None,
    case_record: dict[str, Any] | None = None,
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
) -> dict[str, Any]:
    agent = build_legal_agent(model=model_name, config=config)
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
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Legal ReAct v2 agent returned no messages.")
    final_message = messages[-1]
    final_message_text = message_content_to_text(final_message.content)
    if final_message_text.strip() == "Sorry, need more steps to process this request.":
        raise RuntimeError(
            f"Legal ReAct v2 agent exhausted max_agent_steps={max_agent_steps} before producing a final answer."
        )
    prediction = normalize_prediction(extract_json_object(final_message_text))
    return {
        "prediction": prediction,
        "score_payload": build_score_payload(prediction),
        "messages": serialize_agent_messages(messages),
        "final_message_text": final_message_text,
    }


def score_prediction(prediction: dict[str, Any], case_record: dict[str, Any]) -> dict[str, float | int]:
    return score_offense_level_prediction(build_score_payload(prediction), case_record)


__all__ = [
    "DEFAULT_MAX_AGENT_STEPS",
    "build_single_case_prompt",
    "fetch_case_record",
    "run_single_case_prediction",
    "score_prediction",
]