# Exploratory matching of heatwaves by `I_dyn_pre` sign

## Status and question

This is the configuration-driven exploration for command-center task A2.8. It
does not create or modify a Stage-2 product, and the candidate matching
specifications have not yet been promoted to one final scientific default.

The question is whether positive and negative integrated dynamical-heating
heatwaves can be compared after matching their event severity. The initial
severity variable is `tas_anom_peak`, the maximum temperature anomaly within
each detected heatwave event.

For this exploration:

```text
I_dyn_pre = I_adiabatic_pre + I_advection_pre
```

Both terms are Stage-2 sums over the inclusive 96-hour pre-peak window
`(-96, 0)`. The Stage-2 builder stores the result as `I_dyn_pre`. Positive
`I_dyn_pre` events are the candidate population and negative `I_dyn_pre` events
are the reference population.

## Workflow architecture

The matching workflow has three tracked inputs:

1. the Stage-2 event-feature table containing `I_dyn_pre`;
2. [`matching_settings.json`](matching_settings.json), which defines the
   grouping metric, matching method, variable families, and SD calipers; and
3. the reusable implementation in [`src/selectors.py`](../../src/selectors.py).

The plotting script reads `I_dyn_pre`, calls
`selectors.match_events_by_metric_sign()` for every configured specification,
and uses the returned indices directly. It writes figures and a
`matching_summary.json`, not a matched-event data product. The summary reports
the Stage-2 input and matching-settings paths and SHA-256 checksums for
provenance.

The separate matched face-advection consumer is
`scripts/plot_advection_direction_exploration_matched_clim_anom.py`. It applies
a named specification from the same settings file, validates the selected
Stage-2 event IDs and peak times against Stage 1, and then builds separate
positive and negative climatological-anomaly composites. Its new figure uses
solid lines for positive `I_dyn_pre` and dashed lines for negative `I_dyn_pre`;
the pre-existing all-event face-advection figure remains unchanged.

This preserves the complete Stage-2 event universe and allows different plots
to reproduce the same selection from one settings file. The accepted boundary
and algorithm are recorded in
[decision 007](../../docs/decisions/007_idyn_sign_matching.md).

## Mathematical description of SD clipping

The implementation uses an SD caliper to remove inadmissible *pairs*, followed
by optimal one-to-one assignment. It does not truncate, winsorize, or otherwise
alter any event value. Let the grouping value for event $e$ be

$$
g_e = I_{\mathrm{dyn,pre},e}.
$$

The two nonzero-sign populations are

$$
\mathcal{N} = \{e : g_e < 0\}, \qquad
\mathcal{P} = \{e : g_e > 0\}.
$$

Events with $g_e=0$ are excluded. The script also rejects the input if
`I_dyn_pre` is non-finite or if either sign population is empty. Under the
tracked settings, $\mathcal{N}$ is the reference population and
$\mathcal{P}$ is the candidate population.

Suppose a specification contains $K$ matching variables, with event value
$x_{e,k}$ for variable $k$. A `timedelta64` matching variable, such as
duration, is first expressed as a floating-point number of days. For each
variable, the selector calculates the sample variance separately over the
complete negative and positive populations:

$$
s_{-,k}^2 = \frac{1}{n_- - 1}
\sum_{e \in \mathcal{N}}(x_{e,k}-\bar{x}_{-,k})^2,
\qquad
s_{+,k}^2 = \frac{1}{n_+ - 1}
\sum_{e \in \mathcal{P}}(x_{e,k}-\bar{x}_{+,k})^2.
$$

Their pooled within-group sample standard deviation is

$$
s_k = \sqrt{
\frac{(n_- - 1)s_{-,k}^2 + (n_+ - 1)s_{+,k}^2}
     {n_- + n_+ - 2}
}.
$$

This scale is computed once, before matching, and is not recomputed as events
are retained or excluded. Matching stops with an error if a variable has no
finite positive pooled scale. For reference event $i$ and candidate event
$j$, the standardized absolute difference on variable $k$ is

$$
\delta_{ijk} = \frac{|x_{i,k}-x_{j,k}|}{s_k}.
$$

If the configured caliper for variable $k$ is $c_k$, pair $(i,j)$ is
admissible exactly when

$$
\delta_{ijk} \le c_k \quad \text{for every } k=1,\ldots,K.
$$

The boundary is inclusive. A single scalar `caliper_sd` is applied separately
to every variable in the specification. The reusable selector can also accept
a mapping with a different positive caliper for each variable. Thus a
multivariable 0.20-SD specification does not apply one joint 0.20-SD radius:
it requires every individual standardized difference to be at most 0.20.

For every admissible pair, the assignment distance is the root-mean-square of
the standardized differences:

$$
d_{ij} = \sqrt{\frac{1}{K}\sum_{k=1}^{K}\delta_{ijk}^2}.
$$

Every matching variable therefore has equal weight after SD standardization.
For a one-variable specification, such as the primary `tas_anom_peak` match,
$d_{ij}=\delta_{ij1}$.

Let $m_{ij}\in\{0,1\}$ indicate whether an admissible reference-candidate
pair is selected. Without-replacement matching imposes

$$
\sum_j m_{ij} \le 1, \qquad
\sum_i m_{ij} \le 1, \qquad
m_{ij}=0 \text{ for inadmissible pairs}.
$$

The optimization is lexicographic. It first finds the largest feasible number
of pairs,

$$
M^* = \max_m \sum_{i,j}m_{ij},
$$

and, among assignments with $M^*$ pairs, minimizes total distance:

$$
m^* = \underset{m:\,\sum m_{ij}=M^*}{\arg\min}
\sum_{i,j}m_{ij}d_{ij}.
$$

The code realizes those two priorities in one call to SciPy's linear-sum
assignment solver. With $n_R$ reference events and
$c_{\max}=\max_k c_k$, it sets

$$
\Lambda=(n_R+1)(c_{\max}+1),
$$

assigns cost $d_{ij}$ to an admissible real pair, cost $2\Lambda$ to an
inadmissible real pair, and appends $n_R$ dummy candidate columns with cost
$\Lambda$. A reference event assigned to a dummy is unmatched. Because every
admissible distance is at most $c_{\max}$, the dummy penalty makes one extra
admissible real match preferable to any possible reduction in the summed real
pair distances. The inadmissible-pair penalty and the availability of one dummy
per reference event prevent inadmissible pairs from being returned.

Before constructing this cost matrix, both populations are sorted by unique
`event_id`. This makes the result independent of the Stage-2 row order. The
returned indices still address the original Stage-2 table, and each event can
appear in at most one returned pair.

The SMD values reported for balance are diagnostics rather than assignment
costs. For any audited variable $y$, the script reports

$$
\operatorname{SMD}(y) =
\frac{\bar{y}_{+}-\bar{y}_{-}}{s_{\mathrm{pooled}}(y)}.
$$

The before-match SMD uses all negative and positive events. The after-match
SMD recomputes the pooled sample SD from the two retained groups. Consequently,
the after-match SMD denominator is not necessarily the full-population scale
used to form the matching calipers and pair distances.

## 96-hour data snapshot and provenance

The numerical results and figures below were regenerated from the canonical
96-hour `I_dyn_pre` product by PBS job `2589929.venus` at source commit
`9225f2ce3d96c0df7864f274b940bfb2a167b79e`.

The source is the canonical PNW Bartusek surface-to-700 hPa, tas-q90,
1940-2024 Stage-1 product on Venus:

```text
/home/mhpereir/HW_analysis/results/stage1/a2_7_climatology_20260806/
harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_1940_2024.nc
```

The compact Stage-2 input used for the production figures was:

```text
/home/mhpereir/HW_analysis/results/stage2_event_features/
hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc
sha256: 1a38fa88040bb3597a643e26dfd2882ecb4ac2e859285a01d524f127ef197780
```

The tracked matching settings were:

```text
/home/mhpereir/HW_analysis/scripts/idyn_matching_exploration/
matching_settings.json
sha256: 6b65aaf712ef494a16e5794a19c24b2f2516e5565251a8e922a4311bec434244
```

The Stage-2 universe requires complete June-August events. It contains 258
events, all with finite and nonzero `I_dyn_pre`:

| `I_dyn_pre` sign | Events | Mean `tas_anom_peak` |
| --- | ---: | ---: |
| Negative | 117 | 3.212 K |
| Positive | 141 | 3.747 K |

Across all events, the Pearson correlation between `I_dyn_pre` and
`tas_anom_peak` is 0.403. The unmatched standardized mean difference (SMD) in
peak anomaly is 0.645, where SMD is positive-group mean minus negative-group
mean divided by the pooled within-group standard deviation.

![Unmatched I_dyn_pre populations and peak-anomaly distributions](../../results/Idyn_matching_exploration/idyn_population_overview.png)

## Primary exploratory match

The primary specification uses deterministic, one-to-one optimal matching
without replacement:

- reference population: negative `I_dyn_pre` events;
- candidate population: positive `I_dyn_pre` events;
- matching variable: `tas_anom_peak`;
- distance: absolute difference divided by the pooled within-group SD;
- caliper: 0.20 pooled SD; and
- objective: maximize pair count first, then minimize total distance.

This produces 97 pairs. It retains 97 of 117 negative events and 97 of 141
positive events. The excluded 20 negative and 44 positive events lie outside
the selected comparison set.

| Pair diagnostic | Value |
| --- | ---: |
| Mean absolute anomaly difference | 0.037 K |
| Maximum absolute anomaly difference | 0.165 K |
| Peak-anomaly SMD before matching | 0.645 |
| Peak-anomaly SMD after matching | 0.041 |

![Before and after peak-anomaly matching](../../results/Idyn_matching_exploration/tas_anom_matching_diagnostics.png)

The matched comparison therefore describes positive events that resemble the
negative-event population in peak anomaly. It does not describe the full
positive-event population, especially its extreme warm tail.

## What else becomes balanced

The table below audits variables that were not used by the primary match.
Duration and days from June 21 are reported in days. Only `tas_anom_peak` was
constrained by the matching algorithm.

| Variable | Negative mean before | Positive mean before | SMD before | Negative mean after | Positive mean after | SMD after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peak temperature anomaly [K] | 3.212 | 3.747 | 0.645 | 3.322 | 3.349 | 0.041 |
| Peak temperature [K] | 291.245 | 291.928 | 0.394 | 291.443 | 291.752 | 0.194 |
| Peak threshold excess [K] | 0.745 | 1.285 | 0.677 | 0.848 | 0.906 | 0.092 |
| Integrated threshold excess [K day] | 1.579 | 3.454 | 0.561 | 1.849 | 2.036 | 0.083 |
| Duration [day] | 2.385 | 3.305 | 0.466 | 2.639 | 2.701 | 0.034 |
| Days from June 21 [day] | 19.137 | 27.979 | 0.351 | 19.691 | 31.773 | 0.498 |
| Antecedent mean anomaly [K] | 1.170 | 0.389 | -0.646 | 1.243 | 0.128 | -0.936 |

Peak threshold excess, integrated threshold excess, and duration become well
balanced as a consequence of matching peak anomaly. Absolute peak temperature
improves but retains a modest difference. Season timing becomes less balanced,
and the large antecedent-anomaly contrast becomes larger. Five of the seven
audited variables improve in absolute SMD.

Those remaining contrasts are not automatically matching failures. They may
be part of the mechanism that distinguishes the dynamical-sign populations.
The scientific design must decide which variables define comparability and
which variables remain outcomes or explanatory diagnostics.

![Balance audit and multi-variable retention sensitivity](../../results/Idyn_matching_exploration/covariate_balance_and_sensitivity.png)

## Follow-up: integrated warming plus antecedent temperature

The enlarged antecedent-temperature contrast after peak-anomaly matching is
consistent with a compensation pattern: within events reaching comparable
peak anomalies, the positive-`I_dyn_pre` group begins from a cooler antecedent
state. In the matched sample, the positive-group antecedent mean anomaly is
0.128 K versus 1.243 K in the negative group. Matching alone does not establish
that those events *need* a cooler start, but it makes that mechanism a useful
hypothesis for the composites.

Matching on `I_dTdt_pre` and `T_anom_mean_ant` directly was tested against the
same seven-variable balance audit. It is not a better replacement under the
criterion that every shown absolute SMD should improve without excessive
sample loss.

| Specification | Caliper [pooled SD] | Pairs | Variables improved | Mean absolute SMD | Worst absolute SMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before matching | - | - | - | 0.534 | 0.677 |
| Peak anomaly | 0.20 | 97 | 5 of 7 | 0.268 | 0.936 |
| `I_dTdt_pre` and antecedent anomaly | 0.10 | 33 | 5 of 7 | 0.424 | 1.122 |
| `I_dTdt_pre` and antecedent anomaly | 0.20 | 57 | 4 of 7 | 0.476 | 1.306 |
| Peak anomaly, season timing, and antecedent anomaly | 0.75 | 91 | 7 of 7 | 0.293 | 0.420 |
| Peak anomaly, season timing, and antecedent anomaly | 0.50 | 70 | 7 of 7 | 0.194 | 0.322 |

The strict 0.10-SD integrated-warming match retains 33 pairs and improves five
of the seven audit variables, but its worst absolute SMD is 1.122. Relaxing the
caliper to 0.20 retains 57 pairs but improves only four variables and increases
the worst absolute SMD to 1.306. Neither specification improves the full
balance audit.

The weak overlap is scientifically meaningful. The unmatched `I_dTdt_pre` SMD
is 1.382, and `I_dTdt_pre` is also strongly correlated with `I_dyn_pre` across
events (`r = 0.718`). It is therefore not a neutral nuisance covariate.
Conditioning on it asks a narrower question about how events with the same
realized pre-peak warming and antecedent state partition their dynamical and
other contributions.

![Balance and retention tradeoffs among candidate specifications](../../results/Idyn_matching_exploration/matching_specification_tradeoff.png)

For the next comparison, the most defensible high-retention candidate in this
small search is peak anomaly plus season timing plus antecedent anomaly. A
0.75-SD per-variable caliper retains 91 pairs, or 78% of the 117-event negative
reference population, and improves all seven SMDs. Its mean and worst absolute
SMDs are 0.293 and 0.420. The 0.50-SD version is a stronger-balance sensitivity
that retains 70 pairs, with mean and worst absolute SMDs of 0.194 and 0.322.
Neither achieves uniformly negligible imbalance, so these should remain two
explicitly labeled estimands rather than being selected after looking at
downstream composite differences.

## Retention sensitivity

Using the same 0.20 pooled-SD caliper separately for every requested variable:

| Matching variables | Matched pairs |
| --- | ---: |
| Peak anomaly | 97 |
| Peak anomaly and days from June 21 | 73 |
| Peak anomaly and duration | 74 |
| Peak anomaly, days from June 21, and duration | 38 |

Adding season timing or duration improves balance on those variables but
changes the retained negative-event population and therefore changes the
comparison being made. The three-variable specification retains fewer than
one third of the 117 negative events.

## Questions for the next design step

1. Should severity mean the existing event maximum `tas_anom_peak`, or the
   anomaly sampled specifically at the Stage-2 anchor `peak_time`?
2. Should season timing be a required matching variable, a stratification
   variable, or only a post-match diagnostic?
3. Is duration part of event comparability, or is it a possible consequence of
   the different dynamical evolution?
4. Should the primary comparison preserve antecedent temperature as a
   mechanism, or should the 91-pair three-variable match define the comparable
   event population?
5. Should `I_dTdt_pre` be reserved for a mechanism-conditioned sensitivity
   because of its strong association with `I_dyn_pre`?
6. Which existing temporal and spatial composites should first consume the
   matched event IDs as a sensitivity analysis?

## Reproduce the exploration

The script consumes only the compact Stage-2 table. Generated figures remain
under the ignored `results/` tree.

```bash
mamba activate dev_env
python scripts/idyn_matching_exploration/explore_idyn_matching.py \
  --input-path /path/to/hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc \
  --settings-path scripts/idyn_matching_exploration/matching_settings.json \
  --output-dir results/Idyn_matching_exploration
```

The tracked Venus production entrypoint is:

```text
schedulers/schedule_explore_idyn_matching.sh
```

It stages all five outputs, validates that they are nonempty, and only then
replaces the four figures and `matching_summary.json` in the final directory.

Edit a copied settings file and pass it with `--settings-path` when evaluating
different matching variables or SD calipers. Use `--overwrite` only when
intentionally replacing prior exploratory figures.
