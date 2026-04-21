"""Composite generation for the heatwave analysis pipeline.

Pipeline role:
- Stage 4: derive event-centered composite products from harmonized datasets.

Responsibilities:
- Build peak-aligned composites.
- Extract event-centered windows.
- Compute ensemble member composites.
- Compute event-mean and ensemble-mean reductions.
- Compute percentile envelopes across events or members.
- Extract top-event traces or subsets.

Out of scope:
- Raw data loading.
- Event selection logic.
- Plot rendering.
"""
