from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from pydantic import BaseModel, Field

from baselines.icd_deepagents.agent import _resolve_model, resolve_model_provider
from baselines.legal_react_v2.config import LegalReactV2Config, load_config
from baselines.legal_react_v2.tools import build_legal_manual_tools


DEFAULT_AGENT_NAME = "legal_deepagents_skills"
SKILLS_DIR = Path(__file__).with_name("skills")
SKILLS_SOURCE = "/skills"


class OffenseLevelPrediction(BaseModel):
    offense_level: int | None = Field(default=None, description="Final total offense level, or null when unsupported.")
    justifications: list[str] = Field(default_factory=list, description="Concise manual-grounded justifications.")


COORDINATOR_PROMPT = """
You are a federal sentencing manual navigation coordinator.

Your job is to compute the strictly supported final total offense level by following the sentencing workflow exactly as laid out in the project method. Read /skills/coordinate-sentencing/SKILL.md before taking action.

Core stance:
- Work like a careful legal user of the manuals, not like a shortcut predictor.
- Use the provided legal manual tools through the appropriate specialist to inspect controlling text before making any offense-level claim.
- Stay grounded in the case year when choosing which manual edition to inspect.
- Prefer explicit manual text over intuition.
- If the record does not support a conclusion, return null instead of inventing certainty.
- Do not back-solve from an apparent plea outcome, sentence outcome, or likely negotiated resolution.

Required workflow for computing offense level:
1. Identify the statutory count of conviction: delegate to statute-identifier. It must identify the specific count-of-conviction statute from the case record and confirm statute text when needed.
2. Map the statute through Appendix A: delegate to guideline-mapper. It must use Appendix A, inspect competing candidate sections when necessary, and use USSG structural browsing when no exact Appendix A match exists.
3. Determine the Chapter Two offense level: delegate to chapter-two-analyst. It must identify the base offense level and apply only supported characteristics, cross references, special instructions, commentary, and application notes.
4. Apply Chapter Three adjustments: delegate to chapter-three-analyst after the Chapter Two result. It must consider only supported victim, role, obstruction, multiple-count, and acceptance adjustments.
5. Audit the total: delegate all reports to offense-level-auditor for the final structured result.

Call each named specialist exactly once in this order. Pass the case facts and concise prior reports forward. Do not re-delegate a completed specialist task.

Decision rules:
- Do not guess missing facts.
- Do not retain an enhancement or reduction unless inspected manual text supports it.
- When the mapping report identifies multiple guideline paths, require the relevant inspected comparison before the final decision.
- Do not select a Chapter Two guideline from memory or general legal knowledge when Appendix A has no exact hit.
- Return null when a material adjustment, cross reference, or arithmetic step lacks support in the inspected manual text and case facts.
- Do not infer the final total simply because an adjustment is common, likely, or usually applied in similar cases.
- Stop browsing once the specialist reports contain enough manual evidence for the answer.

The offense-level-auditor's structured response is the final answer.
""".strip()


def _specialist_prompt(skill_name: str, instruction: str) -> str:
    return (
        f"Before taking any other action, use read_file to read {SKILLS_SOURCE}/{skill_name}/SKILL.md. "
        f"Then follow its instructions. {instruction}"
    )


def _tool_groups(config: LegalReactV2Config) -> dict[str, list[Any]]:
    tools_by_name = {getattr(tool, "name", getattr(tool, "__name__", "")): tool for tool in build_legal_manual_tools(config)}
    groups = {
        "title18": ["list_title18_chapters_tool", "open_title18_chapter_tool", "open_title18_section_tool"],
        "appendix_mapping": ["list_appendix_a_entries_tool", "list_ussg_chapters_tool", "open_ussg_subheading_tool", "open_ussg_section_tool"],
        "ussg_section": ["open_ussg_section_tool"],
    }
    missing = {name: [tool_name for tool_name in names if tool_name not in tools_by_name] for name, names in groups.items()}
    missing = {name: names for name, names in missing.items() if names}
    if missing:
        raise RuntimeError(f"Legal manual tool factory is missing expected tool(s): {missing}")
    return {name: [tools_by_name[tool_name] for tool_name in names] for name, names in groups.items()}


def build_legal_deep_agent(
    model: str,
    config: LegalReactV2Config | None = None,
    name: str = DEFAULT_AGENT_NAME,
    model_provider: str | None = None,
    **create_agent_kwargs: Any,
) -> Any:
    active_config = config or load_config()
    resolved_model = _resolve_model(model, resolve_model_provider(model_provider))
    groups = _tool_groups(active_config)
    backend = FilesystemBackend(root_dir=str(SKILLS_DIR.parent), virtual_mode=True)

    subagents = [
        {
            "name": "statute_identifier",
            "description": "Identifies and verifies the statutory count of conviction using Title 18.",
            "system_prompt": _specialist_prompt("statute-identifier", "Return a concise statute-identification report for Appendix A mapping."),
            "tools": groups["title18"],
            "skills": [SKILLS_SOURCE],
        },
        {
            "name": "guideline_mapper",
            "description": "Maps a verified statute to the controlling Chapter Two guideline through Appendix A or structural USSG browsing.",
            "system_prompt": _specialist_prompt("guideline-mapper", "Return concise mapping evidence and the selected or unresolved guideline section."),
            "tools": groups["appendix_mapping"],
            "skills": [SKILLS_SOURCE],
        },
        {
            "name": "chapter_two_analyst",
            "description": "Computes the Chapter Two offense level using the controlling guideline and supported facts.",
            "system_prompt": _specialist_prompt("chapter-two-analyst", "Return the Chapter Two subtotal, calculation, evidence, and unresolved facts."),
            "tools": groups["ussg_section"],
            "skills": [SKILLS_SOURCE],
        },
        {
            "name": "chapter_three_analyst",
            "description": "Applies supported Chapter Three adjustments after the Chapter Two calculation.",
            "system_prompt": _specialist_prompt("chapter-three-analyst", "Return the Chapter Three adjustment calculation and resulting subtotal."),
            "tools": groups["ussg_section"],
            "skills": [SKILLS_SOURCE],
        },
        {
            "name": "offense_level_auditor",
            "description": "Audits all sentencing reports and returns the strictly supported final total offense level.",
            "system_prompt": _specialist_prompt("offense-level-auditor", "Return only the required structured offense-level prediction."),
            "tools": groups["ussg_section"],
            "skills": [SKILLS_SOURCE],
            "response_format": OffenseLevelPrediction,
        },
    ]
    return create_deep_agent(
        model=resolved_model,
        system_prompt=COORDINATOR_PROMPT,
        tools=[],
        subagents=subagents,
        skills=[SKILLS_SOURCE],
        backend=backend,
        response_format=OffenseLevelPrediction,
        name=name,
        **create_agent_kwargs,
    )


__all__ = ["DEFAULT_AGENT_NAME", "OffenseLevelPrediction", "build_legal_deep_agent"]
