## Tool 1: Title 18 Chapter Browser

Status: implemented.

Agreed purpose:

- expose the Title 18 chapter table of contents for one edition
- give the agent the same first orientation step a lawyer would take
- keep the tool structural only and avoid helping the agent choose for free

Agreed design:

- return all chapters in one call
- do not return sections yet
- do not return full statute text yet
- no fallback behavior inside the tool logic
- use the existing Title 18 source path and keep USSG export usage separate

Corrected source split:

- USSG uses the DocIntel export tree under data/legal_sources/_docintel_text/ussg
- USC Title 18 is sourced separately from data/legal_sources_unpacked/usc_title18/<year>/title18.html
- Title 18 does not need a DocIntel export for this tool

Implemented behavior:

- return all chapter headings for one Title 18 edition in one call
- preserve part headings when they exist
- do not return sections yet
- do not return statute body text yet
- keep the tool structural so the agent still has to choose what to inspect next

Files:

- baselines/legal_react_v2/config.py
- baselines/legal_react_v2/tools.py
- baselines/legal_react_v2/playground.py

Immediate validation target:

- run the playground cell that calls list_title18_chapters(2024)
- confirm the returned chapter count and sample headings look sane

## Tool 2: Title 18 Chapter Opener

Status: implemented.

Agreed purpose:

- open one chapter returned by Tool 1
- expose the section headings directly under that chapter
- keep the tool structural so the agent still has to choose which statute section to inspect next

Agreed design:

- take an exact `chapter_id` returned by Tool 1 for the same year
- return chapter metadata plus immediate section rows
- do not return full statute body text yet
- no fuzzy fallback behavior inside the tool logic

Implemented behavior:

- returns `chapter_id`, `part_heading`, `chapter_heading`, `section_count`, and `sections`
- each section row includes `entry_id`, `citation`, and `section_heading`
- returns a `not_found` payload on an exact-id miss instead of guessing

Immediate validation target:

- run the notebook cell that calls `open_title18_chapter(...)`
- confirm the selected chapter returns section headings only and no statute body text

## Tool 3: Title 18 Section Opener

Status: implemented.

Agreed purpose:

- open one exact section returned by Tool 2
- expose the full statute text for that one section
- keep the reasoning burden in the agent after the text is exposed

Agreed design:

- take an exact `entry_id` returned by Tool 2 for the same year
- return section metadata plus extracted body text
- no fuzzy fallback behavior inside the tool logic

Implemented behavior:

- returns `entry_id`, `citation`, `part_heading`, `chapter_heading`, `section_heading`, and `text`
- uses the same cached Title 18 parse that powers Tools 1 and 2
- returns a `not_found` payload on an exact-id miss instead of guessing

Immediate validation target:

- run the notebook cell that calls `open_title18_section(...)`
- confirm the result includes full statute text for one selected section

## Tool 4: Appendix A Browser

Status: implemented.

Agreed purpose:

- expose the USSG Statutory Index mapping from statute to candidate guideline sections
- provide the bridge from the count of conviction to Chapter Two without choosing among candidates for free

Agreed design:

- read Appendix A from the existing USSG DocIntel export for one year
- allow browsing by statute citation prefix
- return candidate guideline sections exactly as listed in Appendix A
- no fuzzy guessing beyond straightforward citation normalization

Implemented behavior:

- returns `entry_id`, `statute_citation`, and `guideline_sections`
- reads Appendix A from `GLMFull.docintel.json` under the configured USSG export root
- supports targeted lookup such as `18 U.S.C. § 1546`

Immediate validation target:

- run the notebook cell that calls `list_appendix_a_entries(2024, statute_prefix="18 U.S.C. § 1546")`
- confirm the result includes the expected candidate guideline sections

## Tool 5: USSG Section Opener

Status: implemented.

Agreed purpose:

- open one exact guideline section after the Appendix A join
- expose the guideline text together with commentary and application-note blocks
- keep the reasoning burden in the agent after the section text is exposed

Agreed design:

- take a guideline citation such as `§2L2.2` or `2L2.2`
- read from the existing USSG DocIntel export for the selected year
- return the parsed section blocks under that guideline heading

Implemented behavior:

- returns `citation`, `section_heading`, `chapter_heading`, `part_heading`, `blocks`, and combined `text`
- accepts normalized citations from Appendix A output
- returns a `not_found` payload on an exact miss instead of guessing

Immediate validation target:

- run the notebook cell that calls `open_ussg_section(2024, "§2L2.2")`
- confirm the result includes the guideline text and commentary/application-note blocks
