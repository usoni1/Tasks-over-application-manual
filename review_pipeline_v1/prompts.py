from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


PROMPTS_DIR = Path(__file__).with_name("prompts")
SYSTEM_PROMPT_TEMPLATE_NAME = "system_prompt.jinja2"


def render_system_prompt(*, extra_instructions: str | None = None) -> str:
    environment = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template(SYSTEM_PROMPT_TEMPLATE_NAME)
    return template.render(extra_instructions=(extra_instructions or "").strip()).strip()


__all__ = ["PROMPTS_DIR", "SYSTEM_PROMPT_TEMPLATE_NAME", "render_system_prompt"]