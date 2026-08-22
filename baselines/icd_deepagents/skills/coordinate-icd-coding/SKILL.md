---
name: coordinate-icd-coding
description: Coordinate the deterministic ICD-10-CM coding workflow through least-privilege specialists.
---

You coordinate one ICD-10-CM coding case. The case summary is evidence, not a source of unverified coding rules.

1. Delegate candidate-family research to `index_researcher` for diagnoses, symptoms, and chronic conditions clearly active during the encounter.
2. Delegate every candidate retained for the answer to `tabular_verifier`. A code is not final until its exact Tabular entry has been inspected.
3. Delegate to `guideline_reviewer` only when a concrete rule could change inclusion, exclusion, specificity, sequencing, combination coding, or an additional-code requirement. Do not invoke it merely to restate an explicit Tabular note.
4. Delegate the complete evidence packet to `coding_auditor` for final ordering, filtering, and structured output.

Prioritize high-yield active problems. Do not reopen the same Tabular entry or guideline section unless conflicting evidence appears. Stop once enough manual evidence supports the answer.
