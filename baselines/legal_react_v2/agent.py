from __future__ import annotations

import importlib.metadata
from typing import Any

from .config import LegalReactV2Config, load_config
from .prompts import render_system_prompt
from .tools import build_legal_manual_tools


DEFAULT_AGENT_NAME = "legal_manual_agent_v2"


def build_legal_agent(
    model: str,
    config: LegalReactV2Config | None = None,
    system_prompt: str | None = None,
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
            "LangChain v1 runtime before using the legal ReAct v2 agent builder."
        ) from exc

    active_config = config or load_config()
    tools = build_legal_manual_tools(active_config)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt or render_system_prompt(),
        name=name,
        **create_agent_kwargs,
    )


__all__ = ["DEFAULT_AGENT_NAME", "build_legal_agent"]