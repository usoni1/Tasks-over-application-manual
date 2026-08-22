from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .azure_search import VECTOR_FIELD_NAME, build_embeddings_client, create_clients
from .config import ICDRAGConfig


DEFAULT_TEMPLATE_NAME = "single_case_coding.j2"
DEFAULT_LLM_MAX_ATTEMPTS = 4


def normalize_icd_code(value: Any) -> str:
    cleaned = str(value or "").strip().upper()
    return re.sub(r"[^A-Z0-9]", "", cleaned)


def coerce_icd_code_list(raw_value: Any) -> list[str]:
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

    return [normalize_icd_code(raw_value)] if normalize_icd_code(raw_value) else []


def load_prompt_template(prompts_dir: Path, template_name: str = DEFAULT_TEMPLATE_NAME) -> str:
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def build_single_case_chain(model_name: str, prompts_dir: Path, llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS):
    prompt = ChatPromptTemplate.from_template(
        load_prompt_template(prompts_dir=prompts_dir),
        template_format="jinja2",
    )
    model = init_chat_model(model_name, temperature=0)
    parser = JsonOutputParser()
    return (prompt | model | parser).with_retry(stop_after_attempt=llm_max_attempts)


def fetch_case_record(
    spark: Any,
    table_name: str,
    hadm_id: int | None = None,
    subject_id: int | None = None,
    note_id: str | None = None,
) -> dict[str, Any]:
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


def retrieve_candidate_chunks(config: ICDRAGConfig, summary_text: str, top_k: int) -> list[dict[str, Any]]:
    _, search_client = create_clients(config)
    embeddings = build_embeddings_client(config)
    query_vector = embeddings.embed_query(summary_text)
    results = list(
        search_client.search(
            search_text=summary_text,
            vector_queries=[
                {
                    "kind": "vector",
                    "vector": query_vector,
                    "fields": VECTOR_FIELD_NAME,
                    "k": top_k,
                }
            ],
            top=top_k,
            select=[
                "chunk_id",
                "source_type",
                "document_title",
                "chunk_title",
                "semantic_path",
                "code",
                "source_path",
                "text",
            ],
        )
    )

    payload: list[dict[str, Any]] = []
    for result in results:
        payload.append(
            {
                "chunk_id": result.get("chunk_id"),
                "source_type": result.get("source_type"),
                "document_title": result.get("document_title"),
                "chunk_title": result.get("chunk_title"),
                "semantic_path": result.get("semantic_path"),
                "code": result.get("code"),
                "source_path": result.get("source_path"),
                "score": result.get("@search.score"),
                "text": result.get("text") or "",
            }
        )
    return payload


def format_retrieved_chunks(chunks: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        header_parts = [
            f"source={chunk.get('source_type') or 'unknown'}",
            f"score={chunk.get('score')}",
        ]
        if chunk.get("document_title"):
            header_parts.append(f"document={chunk['document_title']}")
        if chunk.get("chunk_title"):
            header_parts.append(f"chunk={chunk['chunk_title']}")
        if chunk.get("semantic_path"):
            header_parts.append(f"path={chunk['semantic_path']}")
        if chunk.get("code"):
            header_parts.append(f"code={chunk['code']}")
        header = " | ".join(header_parts)
        sections.append(f"[{idx}] {header}\n{chunk.get('text') or ''}".strip())
    return "\n\n".join(sections)


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(prediction)
    normalized["predicted_icd_codes"] = coerce_icd_code_list(prediction.get("predicted_icd_codes"))
    supporting = prediction.get("supporting_evidence")
    if isinstance(supporting, list):
        normalized["supporting_evidence"] = [str(item).strip() for item in supporting if str(item).strip()]
    return normalized


def _first_icd_code(codes: list[str]) -> str | None:
    for code in codes:
        normalized = normalize_icd_code(code)
        if normalized:
            return normalized
    return None


def score_prediction(predicted_codes: list[str], expected_codes: list[str]) -> dict[str, float | int]:
    predicted_set = set(predicted_codes)
    expected_set = set(expected_codes)
    true_positive = len(predicted_set & expected_set)
    precision = (true_positive / len(predicted_set)) if predicted_set else 0.0
    recall = (true_positive / len(expected_set)) if expected_set else 0.0
    predicted_primary_code = _first_icd_code(predicted_codes)
    expected_primary_code = _first_icd_code(expected_codes)
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
    config: ICDRAGConfig,
    summary_text: str,
    model_name: str,
    prompts_dir: Path,
    top_k: int = 8,
    llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
) -> dict[str, Any]:
    retrieved_chunks = retrieve_candidate_chunks(config=config, summary_text=summary_text, top_k=top_k)
    chain = build_single_case_chain(model_name=model_name, prompts_dir=prompts_dir, llm_max_attempts=llm_max_attempts)
    prediction = chain.invoke(
        {
            "case_summary": summary_text,
            "retrieved_context": format_retrieved_chunks(retrieved_chunks),
            "top_k": top_k,
        }
    )
    return {
        "retrieved_chunks": retrieved_chunks,
        "prediction": normalize_prediction(prediction),
    }