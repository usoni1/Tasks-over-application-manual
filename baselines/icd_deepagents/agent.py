from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from azure.identity import ClientSecretCredential, get_bearer_token_provider
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
import httpx
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, Field

from baselines.icd_react.config import ICDReactConfig, load_config
from baselines.icd_react_v2.tools import build_icd_manual_tools


DEFAULT_AGENT_NAME = "icd_deepagents_skills"
SKILLS_DIR = Path(__file__).with_name("skills")
SKILLS_SOURCE = "/skills"


class ICDPrediction(BaseModel):
    predicted_icd_codes: list[str] = Field(default_factory=list, description="Final ICD-10-CM diagnosis codes.")
    confidence: float | None = Field(default=None, description="Confidence from 0 to 1, or null when unavailable.")
    rationale: str = Field(default="", description="Brief manual-grounded explanation.")
    supporting_evidence: list[str] = Field(default_factory=list, description="Brief manual-grounded evidence items.")


COORDINATOR_PROMPT = """
You are an ICD-10-CM coding coordinator. Work like a careful manual coder. You must follow the coordinate-icd-coding skill, which can be read via read_file at /skills/coordinate-icd-coding/SKILL.md.

The Alphabetic Index finds lead terms and candidate code families. The Tabular List is the authority for code validity, specificity, hierarchy, notes, and final confirmation. Official Guidelines control general and chapter-specific rules, including diabetes, sepsis, sequencing, and mutually exclusive coding choices.

Delegate the full case to the named specialists in the required order, passing each later specialist the case summary and concise earlier reports. Do not make an ICD coding claim from unverified intuition.

Required workflow:
1. Identify and Search: Have index_researcher identify diagnoses, symptoms, and chronic conditions clearly active for this encounter and retrieve candidate families from the Alphabetic Index.
2. Verify and Specify: Have tabular_verifier confirm every potential final code in the Tabular List. Prefer the most specific supported code, but step back to the most specific supported parent when the record does not justify child-level specificity.
3. Reference Rules: Call guideline_reviewer at most once if a concrete rule could change inclusion, exclusion, specificity, sequencing, combination coding, or additional-code requirements. Do not invoke it merely to restate an explicit Tabular note.
4. Aggregate and Filter: Have coding_auditor produce the final compliant ordered code set and final structured response.

Call index_researcher exactly once, then tabular_verifier exactly once, then coding_auditor exactly once. Do not re-delegate a completed specialist task: pass its report forward and resolve remaining uncertainty in the coding_auditor's final audit.

The auditor's structured response is the final answer.
""".strip()


def _specialist_prompt(skill_name: str, instruction: str) -> str:
    skill_path = f"{SKILLS_SOURCE}/{skill_name}/SKILL.md"
    return (
        f"Before taking any other action, use read_file to read {skill_path}. "
        f"Then follow its instructions. {instruction}"
    )


def _tool_groups(config: ICDReactConfig) -> dict[str, list[Any]]:
    tools_by_name = {getattr(tool, "name", getattr(tool, "__name__", "")): tool for tool in build_icd_manual_tools(config)}
    groups = {
        "index": ["list_index_letter_headings_tool", "open_index_heading_hierarchy_tool"],
        "tabular": [
            "list_tabular_chapters_tool",
            "open_tabular_chapter_tool",
            "open_tabular_block_tool",
            "open_tabular_code_tool",
        ],
        "guidelines": ["list_guideline_toc_tool", "open_guideline_section_tool"],
        "tabular_exact": ["open_tabular_code_tool"],
    }
    missing = {name: [tool_name for tool_name in names if tool_name not in tools_by_name] for name, names in groups.items()}
    missing = {name: names for name, names in missing.items() if names}
    if missing:
        raise RuntimeError(f"ICD manual tool factory is missing expected tool(s): {missing}")
    return {name: [tools_by_name[tool_name] for tool_name in names] for name, names in groups.items()}


def _uses_azure_ad_credentials() -> bool:
    return all(
        os.environ.get(name, "").strip()
        for name in ("AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID", "AZURE_ENDPOINT")
    )


def resolve_model_provider(value: str | None = None) -> str:
    provider = (value or os.environ.get("ICD_DEEPAGENTS_MODEL_PROVIDER") or "").strip().lower()
    if not provider:
        return "azure_ad" if _uses_azure_ad_credentials() else "default"
    if provider not in {"default", "azure_ad", "azure_key", "gateway"}:
        raise ValueError("model_provider must be one of: default, azure_ad, azure_key, gateway")
    return provider


def _azure_deployment_name(model: str) -> str:
    configured_deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
    if configured_deployment:
        return configured_deployment
    return model.partition(":")[2] or model


def _azure_api_key_deployment_name(model: str) -> str:
    configured_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if configured_deployment:
        return configured_deployment
    return model.partition(":")[2] or model


def _gateway_http_client() -> httpx.Client | None:
    ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE", "").strip()
    if not ca_bundle:
        return None
    ca_bundle_path = Path(ca_bundle).expanduser()
    if not ca_bundle_path.is_file():
        raise RuntimeError(f"REQUESTS_CA_BUNDLE does not point to a file: {ca_bundle_path}")
    return httpx.Client(verify=str(ca_bundle_path))


def _gateway_base_url(value: str) -> str:
    return value.rstrip("/") if value.rstrip("/").endswith("/v1") else f"{value.rstrip('/')}/v1"


def _resolve_model(model: str, model_provider: str | None = None) -> Any:
    provider = resolve_model_provider(model_provider)
    if provider == "default":
        return model

    if provider == "gateway":
        api_key = os.environ.get("AI_RESEARCH_GATEWAY_API_KEY", "").strip()
        base_url = os.environ.get("AI_RESEARCH_GATEWAY_BASE_URL", "").strip()
        if not api_key or not base_url:
            raise RuntimeError(
                "Gateway mode requires AI_RESEARCH_GATEWAY_API_KEY and AI_RESEARCH_GATEWAY_BASE_URL."
            )
        return ChatOpenAI(
            model=model.partition(":")[2] or model,
            api_key=api_key,
            base_url=_gateway_base_url(base_url),
            temperature=0,
            http_client=_gateway_http_client(),
        )

    if provider == "azure_key":
        required = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_API_KEY")
        if not all(os.environ.get(name, "").strip() for name in required):
            raise RuntimeError("Azure API-key mode requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_API_KEY.")
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
            azure_deployment=_azure_api_key_deployment_name(model),
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )

    if not _uses_azure_ad_credentials():
        raise RuntimeError("Azure AD mode requires AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID, and AZURE_ENDPOINT.")

    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_ENDPOINT"].rstrip("/"),
        azure_deployment=_azure_deployment_name(model),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview")),
        azure_ad_token_provider=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default"),
    )


def build_icd_deep_agent(
    model: str,
    config: ICDReactConfig | None = None,
    name: str = DEFAULT_AGENT_NAME,
    model_provider: str | None = None,
    **create_agent_kwargs: Any,
) -> Any:
    active_config = config or load_config()
    provider = resolve_model_provider(model_provider)
    resolved_model = _resolve_model(model, provider)
    groups = _tool_groups(active_config)
    backend = FilesystemBackend(root_dir=str(SKILLS_DIR.parent), virtual_mode=True)

    subagents = [
        {
            "name": "index_researcher",
            "description": "Identifies encounter-active diagnoses and retrieves Alphabetic Index candidate code families.",
            "system_prompt": _specialist_prompt(
                "index-researcher", "Return a concise candidate report for the next specialist."
            ),
            "tools": groups["index"],
            "skills": [SKILLS_SOURCE],
        },
        {
            "name": "tabular_verifier",
            "description": "Verifies candidate ICD-10-CM codes in the Tabular List and identifies rule-sensitive issues.",
            "system_prompt": _specialist_prompt(
                "tabular-verifier", "Return verified candidates, exclusions, and unresolved guideline questions."
            ),
            "tools": groups["tabular"],
            "skills": [SKILLS_SOURCE],
        },
        {
            "name": "guideline_reviewer",
            "description": "Reviews a specific ICD Official Guidelines question that could change the code set.",
            "system_prompt": _specialist_prompt(
                "guideline-reviewer", "Return a concise manual-grounded recommendation for the specific issue provided."
            ),
            "tools": groups["guidelines"],
            "skills": [SKILLS_SOURCE],
        },
        {
            "name": "coding_auditor",
            "description": "Audits verified evidence and returns the final ordered ICD-10-CM diagnosis code prediction.",
            "system_prompt": _specialist_prompt(
                "coding-auditor", "Return only the required structured ICD prediction."
            ),
            "tools": groups["tabular_exact"],
            "skills": [SKILLS_SOURCE],
            "response_format": ICDPrediction,
        },
    ]
    return create_deep_agent(
        model=resolved_model,
        system_prompt=COORDINATOR_PROMPT,
        tools=[],
        subagents=subagents,
        skills=[SKILLS_SOURCE],
        backend=backend,
        response_format=ICDPrediction,
        name=name,
        **create_agent_kwargs,
    )


__all__ = ["DEFAULT_AGENT_NAME", "ICDPrediction", "build_icd_deep_agent", "resolve_model_provider"]
