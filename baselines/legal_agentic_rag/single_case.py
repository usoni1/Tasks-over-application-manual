from __future__ import annotations

import json
import re
from typing import Any

from baselines.legal_rag.single_case import fetch_case_record, normalize_text_field, score_prediction
from langgraph.errors import GraphRecursionError

from .agent import build_legal_agentic_rag_agent
from .config import LegalAgenticRAGConfig


DEFAULT_MAX_AGENT_STEPS = 30
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


def build_single_case_prompt(
    summary_text: str,
    year: int | None,
    case_record: dict[str, Any] | None = None,
) -> str:
    year_instruction = f"Use the {int(year)} edition of the legal materials.\n" if year is not None else ""
    structured_case_context = build_case_record_prompt_context(case_record)
    structured_case_section = (
        "\n\nStructured case record fields (non-target):\n"
        f"{structured_case_context}"
        if structured_case_context
        else ""
    )
    return (
        "You are analyzing a federal sentencing case against the U.S. Sentencing Guidelines and Title 18 materials.\n\n"
        "Task:\n"
        f"- {year_instruction}".replace("- ", "", 1)
        + "- Read the case summary.\n"
        + "- Use only the available legal retrieval tools.\n"
        + "- You may use search_legal_manuals_full_case_tool once for a broad first-pass retrieval over the entire case summary.\n"
        + "- After the broad pass, use search_legal_manuals_tool for focused follow-up searches.\n"
        + "- Predict the most defensible total offense level.\n"
        + "- If the retrieved context clearly supports it, also predict the criminal history category and guideline range.\n"
        + "- Return only valid JSON matching the schema exactly.\n\n"
        + "Rules:\n"
        + "- Use only information present in the case summary and the retrieved legal context.\n"
        + "- Once you have enough support, stop searching and return the final JSON immediately.\n"
        + "- If repeated searches are not producing useful new evidence, return the best conservative answer supported by the evidence you already have.\n"
        + "- Prefer omission over speculation.\n"
        + "- If you cannot support a field, return null for that field.\n\n"
        + "Output schema:\n"
        + "{\n"
        + '  "predicted_offense_level_total": "string or null",\n'
        + '  "predicted_criminal_history_category": "string or null",\n'
        + '  "predicted_guidelines_low_months": "string or null",\n'
        + '  "predicted_guidelines_high_months": "string or null",\n'
        + '  "confidence": 0.0,\n'
        + '  "rationale": "short explanation",\n'
        + '  "supporting_evidence": ["short evidence item 1", "short evidence item 2"]\n'
        + "}\n\n"
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


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(prediction)
    normalized["predicted_offense_level_total"] = normalize_text_field(prediction.get("predicted_offense_level_total"))
    normalized["predicted_criminal_history_category"] = normalize_text_field(
        prediction.get("predicted_criminal_history_category")
    )
    normalized["predicted_guidelines_low_months"] = normalize_text_field(prediction.get("predicted_guidelines_low_months"))
    normalized["predicted_guidelines_high_months"] = normalize_text_field(
        prediction.get("predicted_guidelines_high_months")
    )
    supporting = prediction.get("supporting_evidence")
    if isinstance(supporting, list):
        normalized["supporting_evidence"] = [str(item).strip() for item in supporting if str(item).strip()]
    else:
        normalized["supporting_evidence"] = []
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


def count_tool_calls(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            total += len(tool_calls)
    return total


def build_budget_exhausted_prediction(max_agent_steps: int) -> dict[str, Any]:
    return {
        "predicted_offense_level_total": None,
        "predicted_criminal_history_category": None,
        "predicted_guidelines_low_months": None,
        "predicted_guidelines_high_months": None,
        "confidence": 0.0,
        "rationale": (
            f"The agent exhausted its reasoning budget of {int(max_agent_steps)} steps before returning a final answer. "
            "This case is recorded as an unsupported prediction."
        ),
        "supporting_evidence": [
            f"Agent reached max_agent_steps={int(max_agent_steps)} without producing final JSON."
        ],
    }


def run_single_case_prediction(
    config: LegalAgenticRAGConfig,
    summary_text: str,
    model_name: str,
    top_k: int = 8,
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
    source_year: int | None = None,
    case_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    agent = build_legal_agentic_rag_agent(
        model=model_name,
        config=config,
        default_top_k=top_k,
        source_year=source_year,
    )
    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": build_single_case_prompt(
                            summary_text=summary_text,
                            year=source_year,
                            case_record=case_record,
                        ),
                    }
                ]
            },
            config={"recursion_limit": max_agent_steps},
        )
    except GraphRecursionError:
        return {
            "prediction": build_budget_exhausted_prediction(max_agent_steps=max_agent_steps),
            "messages": [],
            "final_message_text": "",
            "tool_call_count": 0,
        }
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Legal agentic RAG agent returned no messages.")
    final_message = messages[-1]
    final_message_text = message_content_to_text(final_message.content)
    if final_message_text.strip() == "Sorry, need more steps to process this request.":
        return {
            "prediction": build_budget_exhausted_prediction(max_agent_steps=max_agent_steps),
            "messages": serialize_agent_messages(messages),
            "final_message_text": final_message_text,
            "tool_call_count": count_tool_calls(messages),
        }
    prediction = normalize_prediction(extract_json_object(final_message_text))
    return {
        "prediction": prediction,
        "messages": serialize_agent_messages(messages),
        "final_message_text": final_message_text,
        "tool_call_count": count_tool_calls(messages),
    }


__all__ = [
    "DEFAULT_MAX_AGENT_STEPS",
    "build_single_case_prompt",
    "fetch_case_record",
    "run_single_case_prediction",
    "score_prediction",
]