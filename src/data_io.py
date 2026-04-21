"""Raw data access for the heatwave analysis pipeline.

Pipeline role:
- Workflow layer: raw data access within top-level Stage 1.

Responsibilities:
- Open raw source datasets.
- Handle source-specific filename conventions.
- Standardize variable names where practical.
- Preserve xarray-native lazy loading.

Out of scope:
- Major preprocessing or transformations.
- Event logic.
- Composite generation.
- Plotting.
"""
