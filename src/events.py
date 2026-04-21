"""Event construction utilities for the heatwave analysis pipeline.

Pipeline role:
- Convert selection masks into first-class event definitions.

Responsibilities:
- Convert boolean masks into contiguous event IDs.
- Filter events by duration.
- Identify event peaks.
- Assign event ranks.
- Extract event-level summary metadata.

Out of scope:
- Building selection masks.
- Composite computation.
- Plotting.
"""
