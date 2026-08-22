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
from .config import LegalRAGConfig


DEFAULT_TEMPLATE_NAME = "single_case_sentencing.j2"
DEFAULT_LLM_MAX_ATTEMPTS = 4


def normalize_text_field(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if text else None


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


def parse_jsonish_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, tuple):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, Iterable) and not isinstance(raw_value, (str, bytes, dict)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [piece.strip() for piece in re.split(r"[\n,;|]", stripped) if piece.strip()]
    return [str(raw_value).strip()] if str(raw_value).strip() else []


def fetch_case_record(
    spark: Any,
    table_name: str,
    docket_id: int | None = None,
    year: int | None = None,
    government_sm_doc_id: int | None = None,
) -> dict[str, Any]:
    df = spark.table(table_name)
    available_columns = set(df.columns)
    required_columns = {"docket_id", "year", "case_facts_summary"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Table {table_name} is missing required column(s): {missing_columns}")

    selected_columns = list(df.columns)
    case_df = df.select(*selected_columns)

    if docket_id is not None:
        case_df = case_df.filter(case_df["docket_id"] == docket_id)
    if year is not None:
        case_df = case_df.filter(case_df["year"] == year)
    if government_sm_doc_id is not None and "government_sm_doc_id" in available_columns:
        case_df = case_df.filter(case_df["government_sm_doc_id"] == government_sm_doc_id)

    rows = case_df.orderBy("year", "docket_id").limit(1).collect()
    if not rows:
        raise RuntimeError("No federal sentencing case matched the provided selection.")

    payload = rows[0].asDict(recursive=True)
    summary_text = normalize_text_field(payload.get("case_facts_summary")) or ""
    charges = parse_jsonish_list(payload.get("charges_or_offense_json"))
    if charges:
        summary_text = f"{summary_text}\n\nCharges or offense: {'; '.join(charges)}".strip()

    payload["case_summary"] = summary_text
    payload["expected_offense_level_total"] = normalize_text_field(payload.get("offense_level_total"))
    payload["expected_criminal_history_category"] = normalize_text_field(payload.get("criminal_history_category"))
    payload["expected_guidelines_low_months"] = normalize_text_field(payload.get("guidelines_low_months"))
    payload["expected_guidelines_high_months"] = normalize_text_field(payload.get("guidelines_high_months"))
    payload["charges_or_offense"] = charges
    return payload


def retrieve_candidate_chunks(
    config: LegalRAGConfig,
    summary_text: str,
    top_k: int,
    source_year: int | None = None,
) -> list[dict[str, Any]]:
    _, search_client = create_clients(config)
    embeddings = build_embeddings_client(config)
    query_vector = embeddings.embed_query(summary_text)
    search_filter = f"source_year eq {int(source_year)}" if source_year is not None else None
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
            filter=search_filter,
            select=[
                "chunk_id",
                "source_type",
                "source_year",
                "document_title",
                "chunk_title",
                "semantic_path",
                "citation",
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
                "source_year": result.get("source_year"),
                "document_title": result.get("document_title"),
                "chunk_title": result.get("chunk_title"),
                "semantic_path": result.get("semantic_path"),
                "citation": result.get("citation"),
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
            f"year={chunk.get('source_year')}",
            f"score={chunk.get('score')}",
        ]
        if chunk.get("document_title"):
            header_parts.append(f"document={chunk['document_title']}")
        if chunk.get("chunk_title"):
            header_parts.append(f"chunk={chunk['chunk_title']}")
        if chunk.get("citation"):
            header_parts.append(f"citation={chunk['citation']}")
        if chunk.get("semantic_path"):
            header_parts.append(f"path={chunk['semantic_path']}")
        header = " | ".join(header_parts)
        sections.append(f"[{idx}] {header}\n{chunk.get('text') or ''}".strip())
    return "\n\n".join(sections)


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(prediction)
    normalized["predicted_offense_level_total"] = normalize_text_field(prediction.get("predicted_offense_level_total"))
    normalized["predicted_criminal_history_category"] = normalize_text_field(prediction.get("predicted_criminal_history_category"))
    normalized["predicted_guidelines_low_months"] = normalize_text_field(prediction.get("predicted_guidelines_low_months"))
    normalized["predicted_guidelines_high_months"] = normalize_text_field(prediction.get("predicted_guidelines_high_months"))
    supporting = prediction.get("supporting_evidence")
    if isinstance(supporting, list):
        normalized["supporting_evidence"] = [str(item).strip() for item in supporting if str(item).strip()]
    else:
        normalized["supporting_evidence"] = []
    return normalized


def score_prediction(prediction: dict[str, Any], case_record: dict[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    expected_text = normalize_text_field(case_record.get("expected_offense_level_total"))
    predicted_text = normalize_text_field(prediction.get("predicted_offense_level_total"))
    evaluated = int(expected_text is not None)
    exact_matches = int(evaluated and predicted_text == expected_text)
    if evaluated:
        metrics["offense_level_total_exact_match"] = exact_matches
    metrics["evaluated_target_count"] = evaluated
    metrics["exact_match_count"] = exact_matches
    metrics["exact_match_rate"] = round((exact_matches / evaluated), 6) if evaluated else 0.0
    return metrics


def run_single_case_prediction(
    config: LegalRAGConfig,
    summary_text: str,
    model_name: str,
    prompts_dir: Path,
    top_k: int = 8,
    llm_max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
    source_year: int | None = None,
) -> dict[str, Any]:
    retrieved_chunks = retrieve_candidate_chunks(config=config, summary_text=summary_text, top_k=top_k, source_year=source_year)
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