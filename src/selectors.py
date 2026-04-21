"""Reusable selection logic for the heatwave analysis pipeline.

Pipeline role:
- Stage 3: build masks or filtered subsets used to define events.

Responsibilities:
- Compute heatwave threshold exceedance masks.
- Compute LWA threshold exceedance masks.
- Compute compound selection masks.
- Apply seasonal filtering so selected events stay within the target season.
- Support quantile-based selection modes.
- Support top-N event selection modes.

Out of scope:
- Converting masks into event objects or event IDs.
- Composite computation.
- Plotting.
"""
