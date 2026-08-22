# Review Pipeline V1 Plan

## Goal
Build a new isolated review pipeline under `review_pipeline_v1` for federal sentencing review. The pipeline should:
- export case PDFs to text with Azure Document Intelligence
- assemble per-docket source bundles from exported text and metadata
- generate structured review artifacts for offense-level review
- emit flat CSV outputs that are easy to refresh behind a simple interface

## Current Status
Completed or in place:
- Created an isolated `review_pipeline_v1` workspace.
- Built a Databricks-friendly export script at `review_pipeline_v1/scripts/export_federal_sentencing_docintel_text.py`.
- Built a standalone notebook version for interactive runs and debugging.
- Updated the export flow to stream PDFs instead of waiting to prebuild the full file list.
- Added print-based progress reporting with processed/exported/skipped/failed counts, elapsed time, rate, and optional ETA.
- Confirmed the script only needs standard library modules plus `azure-ai-formrecognizer`, which is already present in `requirements.txt`.

Current boundary:
- Databricks is used for the initial case-PDF to text export.
- Downstream iteration should use exported text and flat artifacts rather than the legacy sentencing pipeline.

## Phase Plan

### Phase 1: Export Layer
Objective:
Create a reliable export layer from docket-organized PDFs to text artifacts.

Deliverables:
- one JSON output per source PDF
- optional manifest JSONL and CSV files
- progress logging suitable for long Databricks job runs

Expected fields per export:
- `docket_id`
- `source_pdf_path`
- `source_file_name`
- `docintel_model`
- `page_count`
- `content_length`
- `full_text`
- `pages`

Notes:
- This layer does not infer government memo vs defense memo.
- The docket id is inferred from the first folder name under the input root.

### Phase 2: Docket Selector
Objective:
Build a selector that reconstructs the relevant document bundle for one docket from exported artifacts.

Deliverables:
- script or notebook that groups exports by docket id
- source preference rules for memo selection
- support for plea and stipulation materials where appropriate

Rules to preserve:
- prefer government sentencing material when available
- otherwise fall back to the best available sentencing-related memo
- keep acceptance-of-responsibility separately modeled because it can rely on plea or stipulation support
- do not assume memo role can be inferred from filenames alone

### Phase 3: Review Artifact Schema
Objective:
Define the output contract for one reviewable docket.

Recommended top-level fields:
- docket metadata
- selected source documents
- case facts
- offense-level calculation steps
- guideline explanation per step
- evidence snippet per step
- evidence-to-guideline linkage explanation
- support strength
- queue label
- reviewer decisions
- PII review surface

Output strategy:
- structured JSON artifact per docket for detailed review
- flattened CSV summary for queueing and interface refresh

### Phase 4: Iteration Notebook
Objective:
Create a notebook that inspects one docket at a time and validates the artifact shape before scale-out.

Deliverables:
- single-docket inspection notebook
- prompt and schema debugging loop
- previews of evidence, explanations, and output rows

### Phase 5: Overnight Generation Job
Objective:
Run the full docket set once the single-docket path is stable.

Deliverables:
- job that walks eligible dockets
- per-docket artifact outputs
- refreshed CSV dump for downstream interface use

Gate before running this:
- single-docket artifact is correct
- CSV shape is stable
- source selection rules are explicit enough to avoid manual cleanup at scale

## Verification Checklist
- The export script can run as a Databricks job against the docket-folder PDF volume.
- Export artifacts are written without relying on the legacy review pipeline.
- A selector can reconstruct one docket bundle from exported text and manifest data alone.
- One docket can be turned into a structured review artifact from text-only inputs.
- The CSV output is flat enough for queueing and interface refresh.

## Decisions So Far
- `review_pipeline_v1` is intentionally isolated from the older sentencing extraction stack.
- The first productionizable component is the Doc Intelligence export job, not the agent.
- The operational artifact contract should be simple enough that refreshed files can be dropped into the interface with minimal work.
- Government-vs-defense memo identity should not be assumed from filename conventions.

## Immediate Next Steps
1. Confirm the export outputs and manifest landed where expected after the completed run.
2. Define the exact docket-level selector logic using the new export artifacts.
3. Define the structured review schema for one docket.
4. Build a one-docket notebook or script that produces the first end-to-end review artifact.
5. Only then move to the overnight generation job and CSV refresh flow.
