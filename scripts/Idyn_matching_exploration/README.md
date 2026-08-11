# Exploratory matching of heatwaves by I_dyn sign

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

## Historical 72-hour data snapshot and provenance

The numerical results and figures below preserve the preliminary exploration
generated with the former inclusive `(-72, 0)` window and a runtime component
sum. They are not validation results for the current 96-hour `I_dyn_pre`
contract. Regenerate them from the rebuilt suffix-free `tas` Stage-2 product
before using the pair counts or SMD values as current results.

The source is the canonical PNW Bartusek surface-to-700 hPa, tas-q90,
1940-2024 Stage-1 product on Venus:

```text
/home/mhpereir/HW_analysis/results/stage1/a2_7_climatology_20260806/
harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_1940_2024.nc
```

The existing compact Stage-2 table was compared with an in-memory Stage-2
rebuild from that exact Stage-1 file. Event IDs and every variable used here
were identical, including `I_advection_pre`, `I_adiabatic_pre`, peak severity,
duration, season timing, and antecedent anomaly. The compact input used for the
local figures was:

```text
/home/mhpereir/HW_analysis/results/stage2_event_features/
hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc
sha256: 5df97ddaaffb7be26fca0fdfd4979ee728faa27506b8003d101f6ffc59455252
```

The historical Stage-2 universe requires complete June-August events. It
contains 258 events, all with finite and nonzero `I_dyn`:

| I_dyn sign | Events | Mean `tas_anom_peak` |
| --- | ---: | ---: |
| Negative | 90 | 3.179 K |
| Positive | 168 | 3.679 K |

Across all events, the Pearson correlation between `I_dyn` and
`tas_anom_peak` is 0.339. The unmatched standardized mean difference (SMD) in
peak anomaly is 0.598, where SMD is positive-group mean minus negative-group
mean divided by the pooled within-group standard deviation.

![Unmatched I_dyn populations and peak-anomaly distributions](../../results/Idyn_matching_exploration/idyn_population_overview.png)

## Primary exploratory match

The primary specification uses deterministic, one-to-one optimal matching
without replacement:

- reference population: negative `I_dyn` events;
- candidate population: positive `I_dyn` events;
- matching variable: `tas_anom_peak`;
- distance: absolute difference divided by the pooled within-group SD;
- caliper: 0.20 pooled SD; and
- objective: maximize pair count first, then minimize total distance.

This produces 90 pairs. Every negative event is retained, while 90 of 168
positive events are retained. The excluded 78 positive events lie outside the
selected comparison set.

| Pair diagnostic | Value |
| --- | ---: |
| Mean absolute anomaly difference | 0.023 K |
| Maximum absolute anomaly difference | 0.128 K |
| Peak-anomaly SMD before matching | 0.598 |
| Peak-anomaly SMD after matching | 0.011 |

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
| Peak temperature anomaly [K] | 3.179 | 3.679 | 0.598 | 3.179 | 3.185 | 0.011 |
| Peak temperature [K] | 291.190 | 291.848 | 0.378 | 291.190 | 291.662 | 0.281 |
| Peak threshold excess [K] | 0.708 | 1.218 | 0.632 | 0.708 | 0.746 | 0.068 |
| Integrated threshold excess [K day] | 1.496 | 3.197 | 0.504 | 1.496 | 1.443 | -0.032 |
| Duration [day] | 2.444 | 3.125 | 0.340 | 2.444 | 2.267 | -0.111 |
| Days from June 21 [day] | 21.278 | 25.411 | 0.162 | 21.278 | 28.622 | 0.301 |
| Antecedent mean anomaly [K] | 1.237 | 0.479 | -0.621 | 1.237 | 0.104 | -1.018 |

Peak threshold excess and integrated threshold excess become well balanced as
a consequence of matching peak anomaly. Absolute peak temperature retains a
moderate difference. Season timing becomes less balanced, and the large
antecedent-anomaly contrast becomes larger.

Those remaining contrasts are not automatically matching failures. They may
be part of the mechanism that distinguishes the dynamical-sign populations.
The scientific design must decide which variables define comparability and
which variables remain outcomes or explanatory diagnostics.

![Balance audit and multi-variable retention sensitivity](../../results/Idyn_matching_exploration/covariate_balance_and_sensitivity.png)

## Follow-up: integrated warming plus antecedent temperature

The enlarged antecedent-temperature contrast after peak-anomaly matching is
consistent with a compensation pattern: within events reaching comparable
peak anomalies, the positive-`I_dyn` group begins from a cooler antecedent
state. Matching alone does not establish that those events *need* a cooler
start, but it makes that mechanism a useful hypothesis for the composites.

Matching on `I_dTdt_pre` and `T_anom_mean_ant` directly was tested against the
same seven-variable balance audit. It is not a better replacement under the
criterion that every shown absolute SMD should improve without excessive
sample loss.

| Specification | Caliper [pooled SD] | Pairs | Variables improved | Mean absolute SMD | Worst absolute SMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before matching | - | - | - | 0.462 | 0.632 |
| Peak anomaly | 0.20 | 90 | 5 of 7 | 0.260 | 1.018 |
| `I_dTdt_pre` and antecedent anomaly | 0.10 | 22 | 6 of 7 | 0.203 | 0.544 |
| `I_dTdt_pre` and antecedent anomaly | 0.20 | 43 | 4 of 7 | 0.422 | 1.048 |
| Peak anomaly, season timing, and antecedent anomaly | 0.75 | 83 | 7 of 7 | 0.212 | 0.302 |
| Peak anomaly, season timing, and antecedent anomaly | 0.50 | 69 | 7 of 7 | 0.097 | 0.165 |

The strict 0.10-SD integrated-warming match balances both requested matching
variables: the `I_dTdt_pre` SMD changes from 1.630 to -0.015 and the antecedent
anomaly SMD changes from -0.621 to -0.041. However, it retains only 22 of the
90 negative events, and the absolute season-timing SMD increases from 0.162 to
0.544. Relaxing the caliper to 0.20 retains 43 pairs but improves only four of
the seven audit variables.

The weak overlap is scientifically meaningful. Before matching, mean
`I_dTdt_pre` is 1.787 K in negative-`I_dyn` events and 4.878 K in positive
events, giving an SMD of 1.630. `I_dTdt_pre` is also strongly correlated with
`I_dyn` across events (`r = 0.779`). It is therefore not a neutral nuisance
covariate. Conditioning on it asks a narrower question about how events with
the same realized pre-peak warming and antecedent state partition their
dynamical and other contributions.

![Balance and retention tradeoffs among candidate specifications](../../results/Idyn_matching_exploration/matching_specification_tradeoff.png)

For the next comparison, the most defensible high-retention candidate in this
small search is peak anomaly plus season timing plus antecedent anomaly. A
0.75-SD per-variable caliper retains 83 pairs, or 92% of the negative-event
reference population, and improves all seven SMDs. Its 0.50-SD version is a
stronger-balance sensitivity that retains 69 pairs and reduces the worst
absolute SMD to 0.165. These should remain two explicitly labeled estimands,
not be selected after looking at downstream composite differences.

## Retention sensitivity

Using the same 0.20 pooled-SD caliper separately for every requested variable:

| Matching variables | Matched pairs |
| --- | ---: |
| Peak anomaly | 90 |
| Peak anomaly and days from June 21 | 73 |
| Peak anomaly and duration | 79 |
| Peak anomaly, days from June 21, and duration | 40 |

Adding season timing or duration improves balance on those variables but
changes the retained negative-event population and therefore changes the
comparison being made. The three-variable specification retains fewer than
half of the 90 negative events.

## Questions for the next design step

1. Should severity mean the existing event maximum `tas_anom_peak`, or the
   anomaly sampled specifically at the Stage-2 anchor `peak_time`?
2. Should season timing be a required matching variable, a stratification
   variable, or only a post-match diagnostic?
3. Is duration part of event comparability, or is it a possible consequence of
   the different dynamical evolution?
4. Should the primary comparison preserve antecedent temperature as a
   mechanism, or should the 83-pair three-variable match define the comparable
   event population?
5. Should `I_dTdt_pre` be reserved for a mechanism-conditioned sensitivity
   because of its strong association with `I_dyn`?
6. Which existing temporal and spatial composites should first consume the
   matched event IDs as a sensitivity analysis?

## Reproduce the exploration

The script consumes only the compact Stage-2 table. Generated figures remain
under the ignored `results/` tree.

```bash
mamba activate dev_env
python scripts/Idyn_matching_exploration/explore_idyn_matching.py \
  --input-path /path/to/hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc \
  --settings-path scripts/Idyn_matching_exploration/matching_settings.json \
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
