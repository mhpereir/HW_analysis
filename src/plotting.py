"""Plotting functions for the heatwave analysis pipeline.

Pipeline role:
- Workflow layer: plotting and export within top-level Stage 2.

Responsibilities:
- Accept prepared datasets, composites, or tables.
- Render composite time series panels.
- Render top-event individual traces.
- Support future map-based composite visualizations.

Out of scope:
- Raw data loading.
- Mask generation.
- Event definition.
- Core analysis logic embedded inside plotting functions.
"""
