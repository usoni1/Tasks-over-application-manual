from baselines.legal_rag import (
    resolve_execution_env,
    resolve_legal_prompts_dir,
    resolve_model_name,
    resolve_spark_session,
    resolve_strict_table,
    run_balanced_chunk_and_upload,
)

from .agent import DEFAULT_AGENT_NAME, DEFAULT_SYSTEM_PROMPT, build_legal_agentic_rag_agent
from .config import LegalAgenticRAGConfig, load_config
from .single_case import fetch_case_record, run_single_case_prediction, score_prediction


__all__ = [
    "DEFAULT_AGENT_NAME",
    "DEFAULT_SYSTEM_PROMPT",
    "LegalAgenticRAGConfig",
    "build_legal_agentic_rag_agent",
    "fetch_case_record",
    "load_config",
    "resolve_execution_env",
    "resolve_legal_prompts_dir",
    "resolve_model_name",
    "resolve_spark_session",
    "resolve_strict_table",
    "run_balanced_chunk_and_upload",
    "run_single_case_prediction",
    "score_prediction",
]