from __future__ import annotations

import importlib.metadata
import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

from .prompts import render_system_prompt


DEFAULT_AGENT_NAME = "federal_sentencing_review_agent_v1"
DEFAULT_MAX_BATCH_SIZE = 8
DEFAULT_REVIEW_ARTIFACT_SCHEMA = {
    "docket_id": "string",
    "docket_support_status": "supported | unsupported",
    "queue_label": "ready_for_review | needs_manual_followup | insufficient_evidence",
    "selected_documents": [
        {
            "document_id": "string",
            "source_file_name": "string",
            "document_role": "government_sentencing_memo | defense_sentencing_memo | plea_agreement | stipulation | presentence_report | other | unknown",
            "why_selected": "string",
        }
    ],
    "case_facts": [
        {
            "fact": "string",
            "source_document": "string",
            "evidence_excerpt": "string",
            "support_strength": "strong | medium | weak",
        }
    ],
    "offense_level_steps": [
        {
            "step_name": "string",
            "claim": "string",
            "guideline_reference": "string | null",
            "proposed_adjustment": "integer | null",
            "included_in_total_offense_level": "boolean",
            "sentencing_evidence": [
                {
                    "source_document": "string",
                    "evidence_excerpt": "string",
                    "why_it_matters": "string",
                    "support_strength": "strong | medium | weak",
                }
            ],
            "guideline_support": [
                {
                    "guideline_reference": "string",
                    "guideline_text_excerpt": "string",
                    "why_it_matters": "string",
                }
            ],
            "justification": "string",
            "support_strength": "strong | medium | weak",
        }
    ],
    "final_total_offense_level": "integer | null",
}
DEFAULT_CASE_FACT_EXTRACTION_SCHEMA = {
    "case_facts": [
        {
            "fact": "string",
            "source_document": "string",
            "evidence_excerpt": "string",
            "support_strength": "strong | medium | weak",
        }
    ]
}


@dataclass(slots=True)
class DocketReviewInput:
    docket_id: str
    case_summary: str | None = None
    selected_documents: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    case_facts: Mapping[str, Any] | None = None
    guideline_context: Mapping[str, Any] | None = None
    reviewer_context: Mapping[str, Any] | None = None


def build_review_agent(
    model: str,
    *,
    tools: Sequence[Any] | None = None,
    system_prompt: str | None = None,
    response_format: Any | None = None,
    name: str = DEFAULT_AGENT_NAME,
    **create_agent_kwargs: Any,
) -> Any:
    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        installed_version = importlib.metadata.version("langchain")
        raise RuntimeError(
            "The installed LangChain runtime does not expose langchain.agents.create_agent. "
            f"Installed version: {installed_version}. Update the active environment to a current "
            "LangChain v1 runtime before using the review agent builder."
        ) from exc

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": list(tools or []),
        "system_prompt": system_prompt or render_system_prompt(),
        "name": name,
        **create_agent_kwargs,
    }
    if response_format is not None:
        agent_kwargs["response_format"] = response_format
    return create_agent(**agent_kwargs)


def normalize_review_input(review_input: DocketReviewInput | Mapping[str, Any]) -> DocketReviewInput:
    if isinstance(review_input, DocketReviewInput):
        return review_input
    if is_dataclass(review_input):
        raw_value = asdict(review_input)
    elif isinstance(review_input, Mapping):
        raw_value = dict(review_input)
    else:
        raise TypeError("review_input must be a DocketReviewInput or mapping.")

    docket_id = str(raw_value.get("docket_id") or "").strip()
    if not docket_id:
        raise ValueError("review_input must include a non-empty docket_id.")

    selected_documents = raw_value.get("selected_documents") or []
    if not isinstance(selected_documents, Sequence) or isinstance(selected_documents, (str, bytes, bytearray)):
        raise TypeError("selected_documents must be a sequence of mappings.")

    return DocketReviewInput(
        docket_id=docket_id,
        case_summary=_normalize_optional_text(raw_value.get("case_summary")),
        selected_documents=tuple(_normalize_document(document) for document in selected_documents),
        case_facts=_normalize_optional_mapping(raw_value.get("case_facts")),
        guideline_context=_normalize_optional_mapping(raw_value.get("guideline_context")),
        reviewer_context=_normalize_optional_mapping(raw_value.get("reviewer_context")),
    )


def build_review_prompt(review_input: DocketReviewInput | Mapping[str, Any], *, prompt_suffix: str | None = None) -> str:
    normalized_input = normalize_review_input(review_input)
    prompt_payload = _drop_empty(
        {
            "docket_id": normalized_input.docket_id,
            "case_summary": normalized_input.case_summary,
            "selected_documents": list(normalized_input.selected_documents),
            "case_facts": normalized_input.case_facts,
            "guideline_context": normalized_input.guideline_context,
            "reviewer_context": normalized_input.reviewer_context,
        }
    )
    prompt = (
        "You are reviewing one federal sentencing docket.\n\n"
        "Return only valid JSON matching this schema exactly:\n"
        f"{json.dumps(DEFAULT_REVIEW_ARTIFACT_SCHEMA, indent=2, ensure_ascii=False)}\n\n"
        "Review bundle:\n"
        f"{json.dumps(prompt_payload, indent=2, ensure_ascii=False)}"
    )
    suffix = _normalize_optional_text(prompt_suffix)
    if suffix:
        prompt = f"{prompt}\n\nAdditional instructions:\n{suffix}"
    return prompt


def build_agent_invoke_input(review_input: DocketReviewInput | Mapping[str, Any], *, prompt_suffix: str | None = None) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": build_review_prompt(review_input, prompt_suffix=prompt_suffix),
            }
        ]
    }


def run_docket_review(
    agent: Any,
    review_input: DocketReviewInput | Mapping[str, Any],
    *,
    prompt_suffix: str | None = None,
    config: Any | None = None,
) -> Any:
    invoke_input = build_agent_invoke_input(review_input, prompt_suffix=prompt_suffix)
    if config is None:
        return agent.invoke(invoke_input)
    return agent.invoke(invoke_input, config=config)


def run_docket_review_batch(
    agent: Any,
    review_inputs: Sequence[DocketReviewInput | Mapping[str, Any]],
    *,
    prompt_suffix: str | None = None,
    batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    config: Any | None = None,
    show_progress: bool = False,
    tqdm_desc: str = "Review dockets",
) -> list[Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    invoke_inputs = [build_agent_invoke_input(review_input, prompt_suffix=prompt_suffix) for review_input in review_inputs]
    if not invoke_inputs:
        return []

    batch_starts: Any = range(0, len(invoke_inputs), batch_size)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            batch_starts = tqdm(
                batch_starts,
                total=(len(invoke_inputs) + batch_size - 1) // batch_size,
                desc=tqdm_desc,
            )
        except ImportError:
            pass

    results: list[Any] = []
    for start in batch_starts:
        stop = start + batch_size
        current_batch = invoke_inputs[start:stop]
        batch_result = agent.batch(current_batch, config=config, return_exceptions=True)
        results.extend(batch_result)
    return results


def build_case_facts_extraction_prompt(
    review_input: DocketReviewInput | Mapping[str, Any],
    review_artifact: Mapping[str, Any],
) -> str:
    normalized_input = normalize_review_input(review_input)
    prompt_payload = _drop_empty(
        {
            "docket_id": normalized_input.docket_id,
            "selected_documents": list(normalized_input.selected_documents),
            "reviewer_context": normalized_input.reviewer_context,
        }
    )
    artifact_payload = _drop_empty(
        {
            "docket_id": review_artifact.get("docket_id"),
            "docket_support_status": review_artifact.get("docket_support_status"),
            "queue_label": review_artifact.get("queue_label"),
            "offense_level_steps": review_artifact.get("offense_level_steps"),
            "final_total_offense_level": review_artifact.get("final_total_offense_level"),
        }
    )
    return (
        "You are extracting exhaustive factual statements from one federal sentencing docket.\n\n"
        "Return only valid JSON matching this schema exactly:\n"
        f"{json.dumps(DEFAULT_CASE_FACT_EXTRACTION_SCHEMA, indent=2, ensure_ascii=False)}\n\n"
        "Extraction rules:\n"
        "- Extract all grounded case facts you can find in the docket bundle, not just facts used in the offense-level computation.\n"
        "- Prefer high recall over shortness. The case_facts list should be expansive and realistic, not a brief summary.\n"
        "- Keep each fact atomic and concrete. Split large compound facts into separate entries when practical.\n"
        "- Include procedural facts, charge facts, plea facts, admitted conduct, loss figures, date ranges, entities, relevant conduct, tax amounts, sentencing positions, and other concrete facts grounded in the text.\n"
        "- Do not include guideline rules, guideline references, offense-level calculations, offense-level totals, Guidelines ranges, enhancements, reductions, specific offense characteristics, acceptance-of-responsibility adjustments, or other sentencing arithmetic as case_facts.\n"
        "- Even when you exclude the sentencing calculation itself, keep the underlying factual predicates and legal-background facts needed for a downstream system to reconstruct the calculation by consulting the Guidelines/manual and the review artifact.\n"
        "- Preserve facts such as counts of conviction, statutes of conviction, per-count statutory maximums, loss amounts, tax amounts, victim and entity relationships, role facts, date ranges, and other grounded facts that a downstream system would need to reason its way to the correct guideline step.\n"
        "- Do not include sentencing recommendations, sentencing advocacy, requested months, requested concurrency, requested variances, time-served arguments, custody-credit calculations, pandemic mitigation arguments, fine recommendations, or other party positions about what sentence the Court should impose.\n"
        "- Do not include appeal-waiver boilerplate, collateral-attack waiver boilerplate, immigration advisals, special-assessment payment instructions, debtor-form instructions, generic probation or supervised-release boilerplate, or similar plea-consequence language unless it is needed to understand the offense conduct or statutory exposure.\n"
        "- If a source sentence mixes factual material with sentencing calculation language, rewrite it so that only the underlying factual material remains in case_facts.\n"
        "- The existing grounded review artifact is context only. Do not copy or restate offense_level_steps, final_total_offense_level, guideline citations, or calculation summaries into case_facts.\n"
        "- Do not include personally identifying information in any case_fact. Remove or anonymize names, initials, street addresses, phone numbers, email addresses, dates of birth, social security numbers, account numbers, and similar identifiers. Use generic placeholders such as Defendant A, Victim A, Witness A, Agent A, or Person A when needed.\n"
        "- Every fact must include source_document and a short evidence_excerpt.\n"
        "- Deduplicate only near-identical repeats.\n\n"
        "Existing grounded review artifact:\n"
        f"{json.dumps(artifact_payload, indent=2, ensure_ascii=False)}\n\n"
        "Docket bundle:\n"
        f"{json.dumps(prompt_payload, indent=2, ensure_ascii=False)}"
    )


def build_case_facts_invoke_input(
    review_input: DocketReviewInput | Mapping[str, Any],
    review_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": build_case_facts_extraction_prompt(review_input, review_artifact),
            }
        ]
    }


def run_case_facts_extraction(
    agent: Any,
    review_input: DocketReviewInput | Mapping[str, Any],
    review_artifact: Mapping[str, Any],
    *,
    config: Any | None = None,
) -> Any:
    invoke_input = build_case_facts_invoke_input(review_input, review_artifact)
    if config is None:
        return agent.invoke(invoke_input)
    return agent.invoke(invoke_input, config=config)


def run_case_facts_extraction_batch(
    agent: Any,
    review_inputs: Sequence[DocketReviewInput | Mapping[str, Any]],
    review_artifacts: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    config: Any | None = None,
    show_progress: bool = False,
    tqdm_desc: str = "Extract case facts",
) -> list[Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if len(review_inputs) != len(review_artifacts):
        raise ValueError("review_inputs and review_artifacts must have the same length.")

    invoke_inputs = [
        build_case_facts_invoke_input(review_input, review_artifact)
        for review_input, review_artifact in zip(review_inputs, review_artifacts)
    ]
    if not invoke_inputs:
        return []

    batch_starts: Any = range(0, len(invoke_inputs), batch_size)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            batch_starts = tqdm(
                batch_starts,
                total=(len(invoke_inputs) + batch_size - 1) // batch_size,
                desc=tqdm_desc,
            )
        except ImportError:
            pass

    results: list[Any] = []
    for start in batch_starts:
        stop = start + batch_size
        current_batch = invoke_inputs[start:stop]
        batch_result = agent.batch(current_batch, config=config, return_exceptions=True)
        results.extend(batch_result)
    return results


def merge_case_facts_into_artifact(
    review_artifact: Mapping[str, Any],
    case_facts_payload: Mapping[str, Any],
) -> dict[str, Any]:
    merged_artifact = dict(review_artifact)
    merged_artifact["case_facts"] = list(case_facts_payload.get("case_facts") or [])
    return merged_artifact


def extract_review_artifact(result: Mapping[str, Any]) -> dict[str, Any]:
    messages = result.get("messages")
    if not isinstance(messages, Sequence) or not messages:
        raise ValueError("Agent result did not include any messages.")

    final_message = messages[-1]
    if isinstance(final_message, Mapping):
        content = final_message.get("content")
    else:
        content = getattr(final_message, "content", None)
    return extract_json_object(message_content_to_text(content))


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, Mapping):
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


def serialize_agent_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, Mapping):
            content = message.get("content")
            serialized.append(
                {
                    "type": str(message.get("type") or message.get("role") or "message"),
                    "content": message_content_to_text(content),
                    "tool_calls": message.get("tool_calls"),
                    "name": message.get("name"),
                }
            )
            continue

        serialized.append(
            {
                "type": message.__class__.__name__,
                "content": message_content_to_text(getattr(message, "content", "")),
                "tool_calls": getattr(message, "tool_calls", None),
                "name": getattr(message, "name", None),
            }
        )
    return serialized


def _normalize_document(document: Any) -> Mapping[str, Any]:
    if is_dataclass(document):
        raw_document = asdict(document)
    elif isinstance(document, Mapping):
        raw_document = dict(document)
    else:
        raise TypeError("Each selected document must be a mapping or dataclass.")
    normalized_document = _drop_empty(raw_document)
    if not isinstance(normalized_document, Mapping):
        raise TypeError("Each selected document must normalize to a mapping.")
    return normalized_document


def _normalize_optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping value.")
    normalized_value = _drop_empty(dict(value))
    if normalized_value is None:
        return None
    if not isinstance(normalized_value, Mapping):
        raise TypeError("Expected a mapping value.")
    return normalized_value


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _drop_empty(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for key, item in value.items():
            normalized_item = _drop_empty(item)
            if normalized_item in (None, {}, [], ()):
                continue
            normalized_mapping[str(key)] = normalized_item
        return normalized_mapping or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized_items = []
        for item in value:
            normalized_item = _drop_empty(item)
            if normalized_item in (None, {}, [], ()):
                continue
            normalized_items.append(normalized_item)
        return normalized_items or None
    return value


__all__ = [
    "DEFAULT_AGENT_NAME",
    "DEFAULT_CASE_FACT_EXTRACTION_SCHEMA",
    "DEFAULT_MAX_BATCH_SIZE",
    "DEFAULT_REVIEW_ARTIFACT_SCHEMA",
    "DocketReviewInput",
    "build_agent_invoke_input",
    "build_case_facts_extraction_prompt",
    "build_case_facts_invoke_input",
    "build_review_agent",
    "build_review_prompt",
    "extract_json_object",
    "extract_review_artifact",
    "merge_case_facts_into_artifact",
    "message_content_to_text",
    "normalize_review_input",
    "run_case_facts_extraction",
    "run_case_facts_extraction_batch",
    "run_docket_review",
    "run_docket_review_batch",
    "serialize_agent_messages",
]