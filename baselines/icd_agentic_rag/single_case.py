from __future__ import annotations

import json
import re
from typing import Any

from baselines.icd_rag.single_case import coerce_icd_code_list, fetch_case_record, score_prediction
from langgraph.errors import GraphRecursionError

from .agent import build_icd_agentic_rag_agent
from .config import ICDAgenticRAGConfig


DEFAULT_MAX_AGENT_STEPS = 30


def build_single_case_prompt(summary_text: str) -> str:
    return (
        "You are assigning ICD-10-CM diagnosis codes for a single clinical case summary.\n\n"
        "Task:\n"
        "- Read the case summary.\n"
        "- Use only the available ICD retrieval tools.\n"
        "- You may use search_icd_manuals_full_case_tool once for a broad first-pass retrieval over the entire case summary.\n"
        "- After the broad pass, use search_icd_manuals_tool for focused follow-up searches.\n"
        "- Predict the diagnosis ICD-10-CM codes explicitly supported by the summary.\n"
        "- Predict all supported diagnosis codes, not just the primary diagnosis.\n\n"
        "Rules:\n"
        "- Use only information present in the case summary and the retrieved ICD manual snippets.\n"
        "- Do not invent conditions that are not stated or strongly implied.\n"
        "- Prefer being conservative over guessing.\n"
        "- If the summary does not support a code clearly, leave it out.\n"
        "- Return only valid JSON matching this schema exactly.\n\n"
        "Output schema:\n"
        "{\n"
        '  "predicted_icd_codes": ["CODE1", "CODE2"],\n'
        '  "confidence": 0.0,\n'
        '  "rationale": "short explanation",\n'
        '  "supporting_evidence": ["short evidence item 1", "short evidence item 2"]\n'
        "}\n\n"
        "Clinical case summary:\n"
        f"{summary_text.strip()}"
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
    normalized["predicted_icd_codes"] = coerce_icd_code_list(prediction.get("predicted_icd_codes"))
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
        "predicted_icd_codes": [],
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
    config: ICDAgenticRAGConfig,
    summary_text: str,
    model_name: str,
    top_k: int = 8,
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
) -> dict[str, Any]:
    agent = build_icd_agentic_rag_agent(model=model_name, config=config, default_top_k=top_k)
    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": build_single_case_prompt(summary_text=summary_text),
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
        raise RuntimeError("ICD agentic RAG agent returned no messages.")
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