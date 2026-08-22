---
name: coordinate-sentencing
description: Coordinate the federal sentencing manual workflow through role-specialized legal research subagents.
---

Coordinate one federal sentencing case. The case record is evidence, not a substitute for inspected legal-manual text.

1. Delegate statute identification to `statute-identifier`.
2. Delegate Appendix A mapping to `guideline-mapper`, passing the case facts and statute report.
3. Delegate the Chapter Two calculation to `chapter-two-analyst`, passing the case facts and mapping report.
4. Delegate the Chapter Three adjustment analysis to `chapter-three-analyst`, passing the case facts and Chapter Two report.
5. Delegate all reports to `offense-level-auditor` for the final structured result.

Call each required specialist once. Do not re-delegate a completed task. A missing material fact or uninspected controlling manual text requires a null offense level, not a plausible estimate.
