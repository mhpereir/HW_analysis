# Decision 010: Central USA Regional Geometry

## Status

Accepted.

## Decision

Register `central_usa` as an active regional analysis domain with these bounds:

```text
latitude:   36 to 46 degrees north
longitude: -105 to -95 degrees east
```

Stage-1 products use the same cosine-latitude-weighted regional reduction and
the same TAS-selected and LWA_A-selected event logic as the established
regions. Production uses q90 thresholds and the 1940-2024 analysis period.

## Rationale

These bounds match the Central USA geometry used by the prepared Eulerian heat
budget campaign and by the existing TAS and LWA threshold products. Keeping one
shared geometry prevents an apparently region-matched product from combining
inconsistent spatial averages.

## Compatibility

Adding the region does not alter any existing regional definition or product.
Central USA outputs must use a fresh run namespace and explicit input paths.
Existing seven-region products and figures remain valid for their recorded
inventories.

The EHB campaign contains annual May-October data through 2025, while the
current Stage-1 analysis remains 1940-2024 to match the daily TAS, LWA, and
threshold coverage used by the established regional products.

## Validation

- The configured bounds must equal the EHB and threshold geometry.
- Both TAS q90 and LWA_A q90 Stage-1 builds must satisfy contract version 2.
- Required fields must be finite and the signed face tendencies must reconstruct
  total advection within numerical tolerance.
- The two threshold selections must produce independent event-summary
  populations while retaining the same harmonized hourly physical fields.
