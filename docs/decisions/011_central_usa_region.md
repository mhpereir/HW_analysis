# Decision 011: Central USA Regional Geometry

## Status

Accepted.

## Decision

Register `central_usa` in the shared region configuration with these bounds:

```text
latitude:   36 to 46 degrees north
longitude: -105 to -95 degrees east
```

Use the same regional reduction, threshold selection, product builders,
seasonal defaults, and plotting workflows as the other configured regions.
The shared plotting metadata supplies the display name `Central USA` and a
distinct region color. Existing command-line and scheduler region arguments
use the canonical identifier `central_usa`.

## Rationale

These bounds match the prepared Central USA Eulerian heat-budget campaign and
the existing TAS and LWA threshold products. A shared definition keeps their
regional geometry consistent throughout the analysis.

## Compatibility

The change adds regional configuration, plotting metadata, documentation, and
the geometry regression case. Existing master schedulers, scientific
calculations, product schemas, seasonal-selection rules, and other regional
definitions remain authoritative.

Production uses explicit matching source paths and fresh output namespaces
under the shared artifact root. Registering the region does not accept or
regenerate historical products. The prepared campaign uses q90 thresholds and
1940-2024 for consistency with the available TAS, LWA, and threshold inputs;
the shared scripts retain their existing configurable analysis arguments.

## Validation

- Check the configured geometry against the EHB and threshold domain.
- Exercise regional reduction and the existing all-region plotting checks.
- Run the full local test suite and lint checks against current master.
- Before production acceptance, validate the exact deployed revision using
  the same Stage-1 and Stage-2 contracts as other regions, including source
  coverage, event membership, finiteness, advection closure, and independent
  numerical checks. Preserve existing failed attempts and their provenance.
