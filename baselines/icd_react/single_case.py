from __future__ import annotations

from collections.abc import Iterable
import json
import re
from typing import Any

from .agent import build_icd_agent
from .config import ICDReactConfig


DEFAULT_MAX_AGENT_STEPS = 50


PROMPT_VARIANTS: dict[str, str] = {
    "baseline": "",
    "same_turn": (
        "Efficiency guidance:\n"
        "- If you have several candidate ICD codes to verify, issue the needed tool calls in the same turn.\n"
        "- Keep the final rationale concise and keep supporting_evidence to the shortest useful list."
    ),
    "no_repeat_tabular_guideline": (
        "Efficiency guidance:\n"
        "- If you have several candidate ICD codes to verify, issue the needed tool calls in the same turn.\n"
        "- If you have already inspected a tabular entry for a candidate code, do not reopen that same code unless conflicting evidence appears.\n"
        "- Do not reopen the same guideline unless new evidence makes it necessary.\n"
        "- Keep the final rationale concise and keep supporting_evidence to the shortest useful list."
    ),
    "active_first_no_repeat": (
        "Efficiency guidance:\n"
        "- If you have several candidate ICD codes to verify, issue the needed tool calls in the same turn.\n"
        "- Prioritize diagnoses that were evaluated, treated, monitored, or clearly affected this admission before revisiting lower-yield background history.\n"
        "- If you have already inspected a tabular entry for a candidate code, do not reopen that same code unless conflicting evidence appears.\n"
        "- Do not reopen the same guideline unless new evidence makes it necessary.\n"
        "- Keep the final rationale concise and keep supporting_evidence to the shortest useful list."
    ),
}


def normalize_icd_code(value: Any) -> str:
    """Normalize an ICD code candidate into the repo's compact alphanumeric form."""
    cleaned = str(value or "").strip().upper()
    return re.sub(r"[^A-Z0-9]", "", cleaned)


def coerce_icd_code_list(raw_value: Any) -> list[str]:
    """Coerce JSON-like, comma-delimited, or iterable code payloads into a clean list."""
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        codes = [normalize_icd_code(item) for item in raw_value]
        return [code for code in codes if code]

    if isinstance(raw_value, tuple):
        codes = [normalize_icd_code(item) for item in raw_value]
        return [code for code in codes if code]

    if isinstance(raw_value, Iterable) and not isinstance(raw_value, (str, bytes, dict)):
        codes = [normalize_icd_code(item) for item in raw_value]
        return [code for code in codes if code]

    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [code for code in (normalize_icd_code(item) for item in parsed) if code]
        separators = [",", "\n", ";", "|"]
        pieces = [stripped]
        for separator in separators:
            if separator in stripped:
                pieces = stripped.replace("\r", "\n").replace(";", ",").replace("|", ",").replace("\n", ",").split(",")
                break
        return [code for code in (normalize_icd_code(item) for item in pieces) if code]

    normalized = normalize_icd_code(raw_value)
    return [normalized] if normalized else []


def build_single_case_prompt(summary_text: str, prompt_suffix: str | None = None) -> str:
    """Build the user prompt for one ICD manual-navigation case.

    The prompt keeps the task narrow: inspect the clinical summary, browse the ICD
    manuals with tools, and return only a JSON object containing the final code set
    plus a short rationale and supporting evidence.
    """
    prompt = (
        "You are reviewing one clinical case for ICD-10-CM diagnosis coding.\n\n"
        "Task:\n"
        "- Use the ICD manual tools to navigate the Index, Tabular List, and Guidelines as needed.\n"
        "- Do not finalize a code until you inspect the Tabular entry.\n"
        "- Return the final ICD-10-CM diagnosis codes you would actually assign, not just high-level category headings.\n"
        "- Do not invent extra child-code specificity that is not supported by the case summary and the manual text you inspected.\n"
        "- Be conservative and avoid unsupported codes.\n"
        "- Return only valid JSON matching this schema exactly.\n\n"
        "Schema:\n"
        "{\n"
        '  "predicted_icd_codes": ["CODE1", "CODE2"],\n'
        '  "confidence": 0.0,\n'
        '  "rationale": "short explanation",\n'
        '  "supporting_evidence": ["short evidence item 1", "short evidence item 2"]\n'
        "}\n\n"
        "Clinical case summary:\n"
        f"{summary_text.strip()}"
    )
    suffix = str(prompt_suffix or "").strip()
    if suffix:
        prompt = f"{prompt}\n\n{suffix}"
    return prompt


def resolve_prompt_suffix(prompt_variant: str = "baseline", prompt_suffix: str | None = None) -> str:
    """Resolve a named prompt variant plus any extra caller-supplied suffix text."""
    variant_name = str(prompt_variant or "baseline").strip() or "baseline"
    if variant_name not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown ICD prompt variant: {variant_name}. Available variants: {sorted(PROMPT_VARIANTS)}")

    parts = [PROMPT_VARIANTS[variant_name].strip(), str(prompt_suffix or "").strip()]
    return "\n".join(part for part in parts if part)


def message_content_to_text(content: Any) -> str:
    """Flatten a LangChain message content payload into a single text string."""
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
    """Extract the first JSON object from agent output text."""
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
    """Normalize the ICD ReAct prediction into the repo's common ICD output shape."""
    normalized = dict(prediction)
    normalized["predicted_icd_codes"] = coerce_icd_code_list(prediction.get("predicted_icd_codes"))
    supporting = prediction.get("supporting_evidence")
    if isinstance(supporting, list):
        normalized["supporting_evidence"] = [str(item).strip() for item in supporting if str(item).strip()]
    else:
        normalized["supporting_evidence"] = []
    return normalized


def serialize_agent_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Serialize LangChain messages into MLflow-safe dictionaries."""
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


def fetch_case_record(
    spark: Any,
    table_name: str,
    hadm_id: int | None = None,
    subject_id: int | None = None,
    note_id: str | None = None,
) -> dict[str, Any]:
    """Fetch one ICD strict-table case record for manual-agent scoring."""
    df = spark.table(table_name)
    available_columns = set(df.columns)
    required_columns = {"hadm_id", "subject_id", "note_id", "input_text", "output_icd_codes"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Table {table_name} is missing required column(s): {missing_columns}")

    selected_columns = [
        column_name
        for column_name in [
            "hadm_id",
            "subject_id",
            "note_id",
            "input_text",
            "output_icd_codes",
            "real_discharge_year_min",
            "real_discharge_year_max",
        ]
        if column_name in available_columns
    ]
    case_df = df.select(*selected_columns)

    if hadm_id is not None:
        case_df = case_df.filter(case_df["hadm_id"] == hadm_id)
    if subject_id is not None:
        case_df = case_df.filter(case_df["subject_id"] == subject_id)
    if note_id is not None:
        case_df = case_df.filter(case_df["note_id"] == note_id)

    row = case_df.orderBy("hadm_id", "note_id").limit(1).collect()
    if not row:
        raise RuntimeError("No case matched the provided selection from the strict ICD table.")

    payload = row[0].asDict(recursive=True)
    payload["expected_icd_codes"] = coerce_icd_code_list(payload.get("output_icd_codes"))
    payload["case_summary"] = str(payload.get("input_text") or "").strip()
    return payload


def score_prediction(predicted_codes: list[str], expected_codes: list[str]) -> dict[str, float | int]:
    """Score predicted ICD codes with the same metrics used by the RAG baseline."""
    predicted_set = set(predicted_codes)
    expected_set = set(expected_codes)
    true_positive = len(predicted_set & expected_set)
    precision = (true_positive / len(predicted_set)) if predicted_set else 0.0
    recall = (true_positive / len(expected_set)) if expected_set else 0.0
    predicted_primary_code = next((code for code in predicted_codes if normalize_icd_code(code)), None)
    expected_primary_code = next((code for code in expected_codes if normalize_icd_code(code)), None)
    return {
        "predicted_count": len(predicted_set),
        "expected_count": len(expected_set),
        "true_positive_count": true_positive,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "predicted_primary_code": predicted_primary_code,
        "expected_primary_code": expected_primary_code,
        "primary_diagnosis_accuracy": round(
            float(expected_primary_code is not None and predicted_primary_code == expected_primary_code),
            6,
        ),
    }


def run_single_case_prediction(
    config: ICDReactConfig,
    summary_text: str,
    model_name: str,
    prompt_variant: str = "baseline",
    prompt_suffix: str | None = None,
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
) -> dict[str, Any]:
    """Run one ICD case through the manual-navigation agent and return normalized output."""
    agent = build_icd_agent(model=model_name, config=config)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": build_single_case_prompt(
                        summary_text,
                        prompt_suffix=resolve_prompt_suffix(prompt_variant=prompt_variant, prompt_suffix=prompt_suffix),
                    ),
                }
            ]
        },
        config={"recursion_limit": max_agent_steps},
    )
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("ICD ReAct agent returned no messages.")
    final_message = messages[-1]
    final_message_text = message_content_to_text(final_message.content)
    if final_message_text.strip() == "Sorry, need more steps to process this request.":
        raise RuntimeError(
            f"ICD ReAct agent exhausted max_agent_steps={max_agent_steps} before producing a final answer."
        )
    prediction = normalize_prediction(extract_json_object(final_message_text))
    return {
        "prediction": prediction,
        "messages": serialize_agent_messages(messages),
        "final_message_text": final_message_text,
    }