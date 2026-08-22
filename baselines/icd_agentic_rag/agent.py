from __future__ import annotations

import importlib.metadata
from typing import Any

from baselines.agentic_retry import DEFAULT_MODEL_RETRY_ATTEMPTS, build_retryable_chat_model

from .config import ICDAgenticRAGConfig, load_config
from .tools import build_icd_search_tools


DEFAULT_AGENT_NAME = "icd_agentic_rag_agent"
DEFAULT_SYSTEM_PROMPT = """
You are an ICD-10-CM diagnosis-coding agent using retrieval tools backed by the indexed ICD manuals.

Available retrieval tools:
- search_icd_manuals_tool: semantic search over focused coding questions, diagnoses, lead terms, or verification lookups.
- search_icd_manuals_full_case_tool: semantic search using the entire clinical case summary as the embedding query for a broad first-pass retrieval.
- Optional source_type filter values: guidelines, index, tabular.

Working method:
1. Start from clearly active diagnoses and likely primary diagnosis candidates in the case summary.
2. If you need a broad first-pass retrieval, you may call search_icd_manuals_full_case_tool once before switching to focused searches.
3. Use multiple focused searches when useful instead of relying only on the broad case retrieval.
4. Prefer the index for lead terms or code-family discovery, the tabular source for code verification, and the guidelines source only when a coding rule could change the answer.
5. Before finalizing any code, make sure you have enough manual evidence from the search results to support it.
6. Be conservative and stop searching once you have enough support for the final code set.

Return only the JSON requested by the user.
""".strip()


def build_icd_agentic_rag_agent(
    model: str,
    config: ICDAgenticRAGConfig | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    name: str = DEFAULT_AGENT_NAME,
    default_top_k: int = 8,
    model_retry_attempts: int = DEFAULT_MODEL_RETRY_ATTEMPTS,
    **create_agent_kwargs: Any,
) -> Any:
    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        installed_version = importlib.metadata.version("langchain")
        raise RuntimeError(
            "The installed LangChain runtime does not expose langchain.agents.create_agent. "
            f"Installed version: {installed_version}. Update the active environment to a current "
            "LangChain v1 runtime before using the ICD agentic RAG agent builder."
        ) from exc

    active_config = config or load_config()
    tools = build_icd_search_tools(config=active_config, default_top_k=default_top_k)
    return create_agent(
        model=build_retryable_chat_model(model_name=model, stop_after_attempt=model_retry_attempts),
        tools=tools,
        system_prompt=system_prompt,
        name=name,
        **create_agent_kwargs,
    )


__all__ = ["DEFAULT_AGENT_NAME", "DEFAULT_SYSTEM_PROMPT", "build_icd_agentic_rag_agent"]