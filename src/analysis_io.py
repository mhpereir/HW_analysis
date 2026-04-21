"""I/O for internal analysis products in the heatwave analysis pipeline.

Pipeline role:
- Workflow layer: Stage 1/Stage 2 handoff for saved analysis-ready artifacts.

Responsibilities:
- Save the harmonized analysis-ready regional dataset produced by Stage 1.
- Reopen saved analysis-ready datasets for Stage 2 workflows.
- Validate expected metadata and dataset conventions on read.
- Manage stable internal filenames or paths for reusable pipeline products.

Out of scope:
- Raw source data loading.
- Source-specific filename handling.
- Scientific preprocessing or harmonization logic.
- Event logic, composites, or plotting.
"""
