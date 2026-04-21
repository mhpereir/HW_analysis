"""Harmonization of heterogeneous source data into a common analysis dataset.

Pipeline role:
- Stage 2: transform raw variables into a stable, analysis-ready regional dataset.

Responsibilities:
- Map variables from different sources to shared internal names.
- Align time coordinates.
- Align source-specific spatial domains.
- Apply regional averaging where required.
- Harmonize variables to a target analysis timestep.
- Assemble the common analysis-ready dataset.

Out of scope:
- Event definition.
- Composite generation.
- Plotting.
"""
