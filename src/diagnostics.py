"""Domain-specific derived diagnostics for the heatwave analysis pipeline.

Pipeline role:
- Compute scientific diagnostics that sit above generic preprocessing.

Responsibilities:
- Compute combined radiative metrics.
- Perform residual checks.
- Compute transformed or normalized diagnostics.
- Derive optional event metrics for ranking or labeling.

Out of scope:
- Generic preprocessing utilities.
- Raw data loading.
- Plotting.
"""
