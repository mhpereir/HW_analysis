# HW Analysis Script

This document outlines the code for the HW analysis project. This project brings together data products for other projects, namely LWA calculation, HW thresholds, Eulerian Heat Budget calculations, and PBL height (in pressure units). It will also make use of some variables that are only avaliable on ARCO.

## Variables needed

- locally available
  - temperature at surface
  - HW threshold
  - LWA (anticyclonic, cyclonic)
  - LWA threshold
  - PBL height (pressure units)

- ARCO sourced
  - cloud cover fraction
  - net solar/longwave radiation at the surface

## Variable processing

Spatial scale missmatch: Some of the variables will need to be spatially averaged (weighted average on spherical coordinates), namely the ARCO and PBL variables.

Time-step missmatch: Right now the LWA variables were calculated on daily timescales. This is not compatible with the hourly timestep of the other data. Maybe we should re-run the LWA calculation in hour timesteps? Or make the code handle both timesteps, and the plotting of LWA is coarsers.

Time-step selection:

- select based on HW ID (from HW threshold);
- select based on LWA ID (from LWA threshold)
- allow for selection of different threshold quantiles during different runs (don't hardcode q)

## Plots to make

time series composite:

- multiple row panel, on share-x axis
- first panel: $\langle T \rangle$ (left y-axis), V (right y-axis)
- second panel: $\frac{d\langle T \rangle}{dt}$
- third panel: Net advection, Adiabatic, Diabatic (all on same y-axis)
- net solar/longwave radiation at surface (left y-axis), Cloud cover fraction (right y-axis)
- LWA (anticyclonic, cyclonic) (left y-axis), PBL height (right y-axis)

also show the time series for a few individual heatwaves, probably the top three heatwaves that reach the maximum value of selector (either LWA (anticyclonic) or temperature; based on how days were selected HW threshold vs LWA threshold).
