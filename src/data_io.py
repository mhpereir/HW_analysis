"""Raw data access for the heatwave analysis pipeline.

Pipeline role:
- Stage 1: open source datasets while keeping them close to source form.

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
