## Tool 1: Alphabetic Index Keyword Explorer

Status: implemented.

Agreed purpose:

- expose the top-level ICD Alphabetic Index headings for one starting letter such as A
- let the agent inspect the exact hierarchy under one selected heading
- keep both steps structural only so the agent still has to choose the correct clinical term

Agreed design:

- first tool lists all main headings for one letter in one call
- second tool opens one exact heading returned by the first tool
- no fuzzy fallback behavior inside the tool logic
- reuse the existing FY2019 ICD Index XML parser and config resolution

Implemented behavior:

- `list_index_letter_headings(letter, prefix=None, limit=500, start_index=0)` returns top-level headings for one letter
- each heading row includes `entry_id`, `title`, direct `code`, cross-references, and `child_count`
- the browse tool supports pagination-style traversal with `start_index` plus `limit`
- `open_index_heading_hierarchy(entry_id)` returns the exact heading and its full nested child hierarchy
- letter input is validated so the first tool does not silently accept malformed requests

Files:

- baselines/icd_react_v2/tools.py
- baselines/icd_react_v2/tool_playground.ipynb

Immediate validation target:

- run the notebook cell that calls `list_index_letter_headings(letter="A", limit=25, start_index=0)`
- confirm the result contains A headings only and exposes stable `entry_id` values
- rerun with a non-zero `start_index` and confirm the returned slice shifts forward
- run the notebook cell that calls `open_index_heading_hierarchy(...)`
- confirm the returned payload includes nested child terms under the selected heading

## Tool 2: Tabular Chapter Browser

Status: implemented.

Agreed purpose:

- expose the ICD Tabular List chapter table of contents
- keep this step minimal so the agent has to choose one chapter before seeing any block-level detail

Agreed design:

- return all matching chapters in one call
- do not include chapter notes or block rows yet
- allow optional code-family filtering through `code_prefix`
- reuse the existing FY2019 ICD Tabular XML parser

Implemented behavior:

- `list_tabular_chapters(code_prefix=None)` returns the Tabular chapter list
- each chapter row includes `chapter_id`, `chapter_heading`, `description`, and `code_range`

Immediate validation target:

- run the notebook cell that calls `list_tabular_chapters()`
- confirm the first chapter is `Certain infectious and parasitic diseases (A00-B99)`
- confirm the returned payload is chapter-only and does not expand blocks yet

## Tool 3: Tabular Chapter Opener

Status: implemented.

Agreed purpose:

- open one exact Tabular chapter such as `1`
- expose the chapter note groups and its blocks only
- keep this step structural so the agent still has to choose which exact block to inspect next

Agreed design:

- take an exact `chapter_id` returned by Tool 2
- return chapter metadata, chapter note groups, and block rows only
- do not jump directly to one exact ICD code entry

Implemented behavior:

- `open_tabular_chapter(chapter_id)` returns `chapter_id`, `chapter_heading`, `note_groups`, `block_count`, and `blocks`
- each block row includes `section_id`, code range fields, and `description`

Immediate validation target:

- run the notebook cell that opens Chapter `1`
- confirm the first block description is `Intestinal infectious diseases (A00-A09)`
- confirm the returned chapter payload stops at block rows and does not expand direct codes yet

## Tool 4: Tabular Block Opener

Status: implemented.

Agreed purpose:

- open one exact Tabular block such as `A00-A09`
- expose the direct codes beneath that block such as `A00`, `A01`, and `A02`
- keep this as a narrower structural follow-up before one exact code open

Agreed design:

- take an exact `section_id` returned by Tool 3
- return block metadata, local note groups, and direct child codes
- do not jump directly to one exact ICD code entry

Implemented behavior:

- `open_tabular_block(section_id)` returns `section_id`, chapter context, `note_groups`, `code_count`, and `codes`
- each code row includes the direct code, description, child count, and any local note groups attached at that top level

Immediate validation target:

- run the notebook cell that opens the first block from Chapter `1`
- confirm the block payload includes direct codes under `A00-A09`
- confirm the returned direct codes begin with `A00 Cholera`

## Tool 5: Tabular Code Opener

Status: implemented.

Agreed purpose:

- open one exact ICD Tabular code such as `A00`
- expose the authoritative code-level hierarchy and note groups
- keep all coding reasoning in the agent after the text structure is exposed

Agreed design:

- take one exact code string
- return chapter context, section context, ancestor codes, child codes, and note groups
- reuse the existing exact-code opener instead of duplicating Tabular traversal logic

Implemented behavior:

- `open_tabular_code(code)` returns the exact code payload from the Tabular List
- the response includes `ancestor_codes`, `child_codes`, `chapter_notes`, `section_notes`, and `entry_notes`
- nearby suggestions are preserved on not-found results because that behavior already exists in the underlying opener

Immediate validation target:

- run the notebook cell that opens `A00`
- confirm the payload includes child codes `A00.0`, `A00.1`, and `A00.9`

## Tool 6: Guidelines TOC Browser

Status: implemented.

Agreed purpose:

- expose the full FY2019 ICD coding-guidelines table of contents
- let the agent see the available rule sections before opening one exact section
- keep this step structural so the agent still has to choose which rule section to inspect next

Agreed design:

- return the full extracted TOC in one call by default
- preserve stable `section_id` values for exact follow-up opens
- allow optional title-prefix filtering through `section_prefix`
- parse the TOC from the exported DocIntel JSON instead of the old PDF heuristic

Implemented behavior:

- `list_guideline_toc(section_prefix=None, limit=None)` returns the matching guideline TOC rows
- each TOC row includes `section_id`, `title`, and `level`
- `limit=None` returns the full matching TOC rather than a truncated slice

Immediate validation target:

- run the notebook cell that calls `list_guideline_toc(limit=None)`
- confirm the payload contains simple title rows with stable `section_id` values

## Tool 7: Guidelines Section Opener

Status: implemented.

Agreed purpose:

- open one exact guideline section using the id returned by the TOC browser
- expose the extracted guideline text together with its title and page bounds
- keep the reasoning burden in the agent after the rule text is exposed

Agreed design:

- take an exact `section_id` returned by Tool 6
- return title, page bounds, and extracted text
- locate the section in the exported DocIntel body text and slice it by TOC order

Implemented behavior:

- `open_guideline_section(section_id)` returns the selected guideline section payload
- the response includes `title`, `level`, `page_start`, `page_end`, and `text`

Immediate validation target:

- run the notebook cell that opens the first `section_id` returned by Tool 6
- confirm the payload contains extracted guideline text for that exact TOC row
