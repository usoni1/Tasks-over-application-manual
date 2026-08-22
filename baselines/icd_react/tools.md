# ICD ReAct Tools

This baseline starts from a simple rule: expose the ICD manuals as browseable manual objects, not as coder-specific workflow helpers.

The current v0 tool layer is intentionally narrow. It lets an agent inspect the same manual surfaces a human coder would use without encoding the coding procedure inside the tool itself.

## Manual Access

- `EXECUTION_ENV=databricks` defaults to the catalog-backed manual root at `/Volumes/usdo_aa_catalog/research_tam_datasets/mimic/ICD_manual`.
- `EXECUTION_ENV=local` defaults to `data/ICD-2019-manual` inside the repo.
- You can override either mode with `ICD_MANUALS_ROOT` or `ICD_MANUALS_ROOT_LOCAL` / `ICD_MANUALS_ROOT_DATABRICKS`.

## Design Principle

Each tool may expose manual structure, but it may not apply coding workflow on the agent's behalf.

Good:

- list available index terms
- open one index subtree
- browse tabular chapters and sections
- open one tabular code node
- list guideline sections
- open one guideline section

Out of scope for this layer:

- suggest the correct lead term
- validate the final diagnosis code set
- rank candidate codes semantically
- apply sequencing rules automatically

## Implemented Tools

### `list_index_main_terms`

Purpose:

- browse high-level Alphabetic Index entries by letter or prefix

Inputs:

- `letter: str | None`
- `prefix: str | None`
- `limit: int = 50`

Returns:

- one row per `mainTerm`
- `entry_id`, `letter`, `title`, `code`, `see`, `see_also`, `child_count`

Grounding:

- backed by `icd10cm_index_2019.xml`
- corresponds directly to `letter -> mainTerm`

### `open_index_term`

Purpose:

- open a single high-level Index entry and inspect its subtree

Inputs:

- `entry_id: str`

Returns:

- the main term title and metadata
- nested child terms
- `code`, `see`, `see_also`

Grounding:

- backed by the same `mainTerm -> term -> term ...` XML tree

### `open_tabular_entry`

Purpose:

- inspect one Tabular code node as the final authority surface

Inputs:

- `code: str`

Returns:

- code and description
- chapter and section descriptions
- ancestor codes
- child codes
- chapter, section, and local note groups

Grounding:

- backed by `icd10cm_tabular_2019.xml`
- note groups include `includes`, `inclusionTerm`, `excludes1`, `excludes2`, `codeFirst`, `useAdditionalCode`, `codeAlso`, and `sevenChrNote`

### `list_tabular_chapters`

Purpose:

- browse the top-level Tabular chapter headings and code ranges

Inputs:

- `code_prefix: str | None`
- `limit: int = 50`

Returns:

- `chapter_id`, `description`, `code_range`, `section_count`

Grounding:

- backed by `chapter -> sectionIndex -> sectionRef`
- corresponds to headings like `Certain infectious and parasitic diseases (A00-B99)`

### `open_tabular_chapter`

Purpose:

- open one chapter and inspect the next level down

Inputs:

- `chapter_id: str`

Returns:

- chapter description
- chapter note groups
- section list with `section_id`, `first_code`, `last_code`, and description

Grounding:

- backed by `sectionRef` rows such as `A00-A09 Intestinal infectious diseases`

### `open_tabular_section`

Purpose:

- open one section and inspect the top-level codes directly below it

Inputs:

- `section_id: str`

Returns:

- section description and note groups
- parent chapter metadata
- top-level codes that can be opened with `open_tabular_entry`

Grounding:

- backed by `section id="..." -> diag`

### `list_guideline_toc`

Purpose:

- expose a lightweight guideline table of contents

Inputs:

- `section_prefix: str | None`
- `limit: int = 50`

Returns:

- `section_id`, heading path, page bounds, preview

Grounding:

- backed by `2019-icd10-coding-guidelines-.pdf`
- section extraction is currently heuristic and derived from heading patterns in the PDF text

### `open_guideline_section`

Purpose:

- fetch one guideline section by `section_id`

Inputs:

- `section_id: str`

Returns:

- heading path
- page bounds
- full extracted section text

## LangChain Integration

`build_icd_manual_tools(config)` returns plain Python callables with descriptive docstrings.

This matches the current LangChain documentation pattern where `create_agent(...)` can infer tool metadata from function signatures and docstrings.

`build_icd_agent(model, ...)` targets that newer `create_agent` entry point.

The wrappers are intentionally thin. All reasoning is expected to stay in the agent loop, not inside the tool code.

## Current Limitations

- guideline section extraction is heuristic and should be replaced with a stable section catalog if we make this the main benchmark path
- `open_index_term` currently opens a high-level `mainTerm`; nested child terms are returned inline instead of requiring separate open calls
- this v0 layer is ICD-only and does not yet include Neoplasm Table, Drug Table, or External Cause browsing
- the currently active local Python environment may still expose the older `create_react_agent`-era API; if so, `build_icd_agent(...)` will fail with a version-mismatch error until that environment is updated