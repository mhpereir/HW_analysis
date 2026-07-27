# Decision 002: PCA Feature Matrix

## Status

Accepted current default, with supported alternatives documented.

## Decision

The default PCA matrix emphasizes event pathway structure and uses the current
default feature tuple from `scripts/event_features/build_stage3_event_feature_pca.py`:

```text
I_dTdt_pre
f_adiabatic_pre
f_advection_pre
sqrt_I_lwa_a_pre_peak
T_anom_mean_ant
cos_days_from_solstice
duration
```

Supported derived inputs include:

```text
f_adiabatic_pre
f_diabatic_pre
f_advection_pre
sqrt_I_lwa_a_pre_peak
cos_days_from_solstice
log10_tas_excess_integral
```

`f_diabatic_pre` is supported and may be requested explicitly, but it is not
part of the current default matrix.

## Rationale

The default matrix avoids making event severity the primary driver of the first
PCA pass. Severity variables such as `tas_anom_peak`,
`tas_excess_integral`, and `log10_tas_excess_integral` are better treated as
diagnostic or outcome variables unless a run explicitly requests them as PCA
inputs.

## Consequences

- PCA must be fit on scaled features, not raw mixed-unit variables.
- `feature_center` and `feature_scale` are part of the Stage-3 product contract.
- Events with invalid derived values are dropped and recorded in
  `valid_event_mask_original`.
- Sensitivity tests should compare matrices with and without additional
  heat-budget fractions when interpretation depends on compositional features.
