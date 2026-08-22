from __future__ import annotations

import re
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

from .agent import build_icd_agent


DEFAULT_MAX_AGENT_STEPS = 60


DISCHARGE_DIAGNOSIS_SECTION_PATTERNS = [
    re.compile(r"^\s*(discharge diagnoses?|final diagnoses?|hospital diagnoses?)\s*:?\s*$", re.IGNORECASE),
]
STOP_SECTION_PATTERNS = [
    re.compile(
        r"^\s*(discharge medications?|medications on discharge|brief hospital course|hospital course|history of present illness|past medical history|physical exam|discharge condition|disposition|follow[- ]?up)\s*:?\s*$",
        re.IGNORECASE,
    ),
]


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
    "active_first_recall_audit": (
        "Efficiency guidance:\n"
        "- If you have several candidate ICD codes to verify, issue the needed tool calls in the same turn.\n"
        "- Prioritize diagnoses that were evaluated, treated, monitored, or clearly affected this admission before revisiting lower-yield background history.\n"
        "- If you have already inspected a tabular entry for a candidate code, do not reopen that same code unless conflicting evidence appears.\n"
        "- Do not reopen the same guideline unless new evidence makes it necessary.\n"
        "- Keep the final rationale concise and keep supporting_evidence to the shortest useful list.\n"
        "Recall audit before finalizing:\n"
        "- Do one brief final omission check over the summary and any explicit discharge-diagnosis section for clearly active conditions you have not yet verified.\n"
        "- Scan explicit discharge diagnoses line by line; for each clearly active diagnosis, either keep a verified code or omit it only for a concrete manual reason.\n"
        "- If a condition appears clinically active or is explicitly carried as a discharge diagnosis, prefer a supported parent code over omitting it entirely when child-level specificity is unclear.\n"
        "- Pay special attention to active acute conditions, injuries or external-cause details, obstetric delivery or outcome details, neoplasm site details, acute organ dysfunctions, electrolyte abnormalities, arrhythmias, and other encounter-specific diagnoses that may be easy to miss.\n"
        "- Still exclude conditions that are only background history and did not affect this encounter."
    ),
}


def extract_discharge_diagnosis_lines(summary_text: str, max_lines: int = 12) -> list[str]:
    lines = [line.strip(" -\t") for line in str(summary_text or "").splitlines()]
    extracted: list[str] = []
    in_section = False

    for raw_line in lines:
        line = str(raw_line or "").strip()
        if not line:
            if in_section and extracted:
                break
            continue

        if any(pattern.match(line) for pattern in DISCHARGE_DIAGNOSIS_SECTION_PATTERNS):
            in_section = True
            continue

        if in_section and any(pattern.match(line) for pattern in STOP_SECTION_PATTERNS):
            break

        if not in_section:
            continue

        normalized = re.sub(r"\s+", " ", line).strip(" -:;")
        if not normalized:
            continue
        if normalized not in extracted:
            extracted.append(normalized)
        if len(extracted) >= max_lines:
            break

    return extracted


def build_single_case_prompt(summary_text: str, prompt_suffix: str | None = None) -> str:
    """Build the v2 single-case ICD prompt around the manual workflow in method.md."""
    discharge_diagnosis_lines = extract_discharge_diagnosis_lines(summary_text)
    discharge_diagnosis_block = ""
    if discharge_diagnosis_lines:
        discharge_diagnosis_block = "\n\nExplicit discharge-diagnosis lines detected:\n" + "\n".join(
            f"- {line}" for line in discharge_diagnosis_lines
        )

    prompt = (
        "You are reviewing one clinical case for ICD-10-CM diagnosis coding.\n\n"
        "Task:\n"
        "- Use only the ICD manual tools.\n"
        "- Focus first on diagnoses that were evaluated, treated, monitored, medication-managed, or otherwise clearly active for this encounter.\n"
        "- If the summary contains an explicit Discharge Diagnosis, Final Diagnoses, or Hospital Diagnoses section, treat those diagnosis lines as strong candidate evidence for this encounter.\n"
        "- Treat past medical history sections, discharge medication lists, and boilerplate problem lists as weak evidence by themselves.\n"
        "- Do not include personal history codes, long-term medication-use codes, tobacco history or status codes, or stable chronic comorbidities unless the summary shows they affected care during this encounter or were explicitly documented as active discharge diagnoses.\n"
        "- If a condition appears only as background history and did not affect this admission, leave it out.\n"
        "- Before answering, do one final completeness sweep for active secondary diagnoses and separately codeable findings that clearly affected this encounter.\n"
        "- In that final sweep, scan any explicit discharge-diagnosis section line by line and make sure each clearly active diagnosis is either represented by a verified code or intentionally omitted for a manual reason such as redundancy, integral symptom coding, or insufficient specificity.\n"
        "- If a status/history/BMI/long-term-use item is explicitly documented as an assessed discharge diagnosis or encounter-relevant factor, you may keep it after verifying the exact code.\n"
        "- In that final sweep, check for commonly missed but still codeable active findings: additional acute sites of the same process, abnormal imaging findings that were assessed, nutritional or BMI-related diagnoses, electrolyte or metabolic abnormalities, GI bleeding or blood-loss diagnoses, and noncompliance that materially affected care.\n"
        "- Do not finalize a diagnosis code until you inspect the exact Tabular entry.\n"
        "- Use the Official Guidelines only when a concrete rule is likely to change the answer; in many cases zero or one guideline section is enough.\n"
        "- Once you have enough manual evidence for the final supported code set, stop browsing and answer.\n"
        "- Be conservative and avoid unsupported codes or invented specificity.\n"
        "- Return only valid JSON matching this schema exactly.\n\n"
        "Schema:\n"
        "{\n"
        '  "predicted_icd_codes": ["CODE1", "CODE2"],\n'
        '  "confidence": 0.0,\n'
        '  "rationale": "short explanation",\n'
        '  "supporting_evidence": ["short manual-grounded evidence item 1", "short manual-grounded evidence item 2"]\n'
        "}\n\n"
        "Clinical case summary:\n"
        f"{summary_text.strip()}"
        f"{discharge_diagnosis_block}"
    )
    suffix = str(prompt_suffix or "").strip()
    if suffix:
        prompt = f"{prompt}\n\n{suffix}"
    return prompt


def resolve_prompt_suffix(prompt_variant: str = "baseline", prompt_suffix: str | None = None) -> str:
    variant_name = str(prompt_variant or "baseline").strip() or "baseline"
    if variant_name not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown ICD v2 prompt variant: {variant_name}. Available variants: {sorted(PROMPT_VARIANTS)}")

    parts = [PROMPT_VARIANTS[variant_name].strip(), str(prompt_suffix or "").strip()]
    return "\n".join(part for part in parts if part)


def run_single_case_prediction(
    config: ICDReactConfig,
    summary_text: str,
    model_name: str,
    prompt_variant: str = "baseline",
    prompt_suffix: str | None = None,
    max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
) -> dict[str, Any]:
    """Run one ICD case through the v2 manual-navigation agent and return normalized output."""
    agent = build_icd_agent(model=model_name, config=config)
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
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("ICD ReAct v2 agent returned no messages.")
    final_message = messages[-1]
    final_message_text = message_content_to_text(final_message.content)
    if final_message_text.strip() == "Sorry, need more steps to process this request.":
        raise RuntimeError(
            f"ICD ReAct v2 agent exhausted max_agent_steps={max_agent_steps} before producing a final answer."
        )
    prediction = normalize_prediction(extract_json_object(final_message_text))
    return {
        "prediction": prediction,
        "messages": serialize_agent_messages(messages),
        "final_message_text": final_message_text,
    }


__all__ = [
    "DEFAULT_MAX_AGENT_STEPS",
    "PROMPT_VARIANTS",
    "build_single_case_prompt",
    "coerce_icd_code_list",
    "extract_discharge_diagnosis_lines",
    "fetch_case_record",
    "resolve_prompt_suffix",
    "run_single_case_prediction",
    "score_prediction",
]