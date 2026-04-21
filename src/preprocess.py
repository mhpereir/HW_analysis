"""Low-level preprocessing utilities for the heatwave analysis pipeline.

Pipeline role:
- Provide reusable preprocessing operations used before or during harmonization.

Responsibilities:
- Floor or otherwise standardize time coordinates.
- Drop leap days.
- Convert units.
- Standardize coordinates.
- Compute area-weighted regional means.
- Compute day-of-year climatologies and anomalies.
- Provide resampling and interpolation utilities.

Out of scope:
- Event definition.
- Composite logic.
- Plotting code.
- Higher-level scientific diagnostics.
"""
