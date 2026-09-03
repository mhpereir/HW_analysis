"""Validated static settings for the A2.8 I_dyn matching workflow."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

DEFAULT_SETTINGS_PATH = Path(__file__).with_name("matching_settings.json")
SETTINGS_SCHEMA_VERSION = 2

SUPPORTED_METHOD = "maximum_cardinality_minimum_distance"
SUPPORTED_STANDARDIZATION = "pooled_group_standard_deviation"
SUPPORTED_DISTANCE = "root_mean_square_standardized_difference"
SUPPORTED_CALIPER_RULE = "all_variables_within_threshold"


@dataclass(frozen=True)
class MatchingFamily:
    """One named set of event-level matching variables."""

    identifier: str
    label: str
    variables: tuple[str, ...]


@dataclass(frozen=True)
class MatchingSpecification:
    """One named matching family and SD-caliper combination."""

    identifier: str
    family: str
    caliper_sd: float


@dataclass(frozen=True)
class MatchingSettings:
    """Complete validated settings consumed by matching-aware plots."""

    source_path: Path
    sha256: str
    schema_version: int
    group_variable: str
    reference_sign: str
    method: str
    standardization: str
    distance: str
    replacement: bool
    caliper_rule: str
    balance_variables: tuple[str, ...]
    families: Mapping[str, MatchingFamily]
    specifications: Mapping[str, MatchingSpecification]
    primary_specification: str
    retention_sensitivity: tuple[str, ...]
    balance_comparison: tuple[str, ...]
    summary_specifications: tuple[str, ...]
    frontier_families: tuple[str, ...]
    frontier_calipers_sd: tuple[float, ...]

    def family(self, identifier: str) -> MatchingFamily:
        """Return a configured matching family by identifier."""
        try:
            return self.families[identifier]
        except KeyError as exc:
            raise KeyError(f"Unknown matching family: {identifier}") from exc

    def specification(self, identifier: str) -> MatchingSpecification:
        """Return a configured matching specification by identifier."""
        try:
            return self.specifications[identifier]
        except KeyError as exc:
            raise KeyError(f"Unknown matching specification: {identifier}") from exc


def load_matching_settings(
    path: str | Path = DEFAULT_SETTINGS_PATH,
) -> MatchingSettings:
    """Load and strictly validate one JSON matching-settings file."""
    source_path = Path(path).expanduser().resolve()
    raw_bytes = source_path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid matching-settings JSON: {source_path}") from exc
    if not isinstance(raw, dict):
        raise TypeError("Matching settings must contain a top-level JSON object.")

    _require_exact_keys(
        raw,
        {
            "schema_version",
            "group",
            "matching",
            "balance_variables",
            "families",
            "specifications",
            "plots",
        },
        context="matching settings",
    )
    schema_version = _integer(raw["schema_version"], context="schema_version")
    if schema_version != SETTINGS_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported matching-settings schema_version: "
            f"{schema_version}; expected {SETTINGS_SCHEMA_VERSION}."
        )

    group = _mapping(raw["group"], context="group")
    _require_exact_keys(
        group,
        {"variable", "reference_sign"},
        context="group",
    )
    group_variable = _nonempty_string(
        group["variable"],
        context="group.variable",
    )
    reference_sign = _nonempty_string(
        group["reference_sign"],
        context="group.reference_sign",
    )
    if reference_sign not in {"negative", "positive"}:
        raise ValueError("group.reference_sign must be 'negative' or 'positive'.")

    matching = _mapping(raw["matching"], context="matching")
    _require_exact_keys(
        matching,
        {"method", "standardization", "distance", "replacement", "caliper_rule"},
        context="matching",
    )
    method = _supported_string(
        matching["method"],
        expected=SUPPORTED_METHOD,
        context="matching.method",
    )
    standardization = _supported_string(
        matching["standardization"],
        expected=SUPPORTED_STANDARDIZATION,
        context="matching.standardization",
    )
    distance = _supported_string(
        matching["distance"],
        expected=SUPPORTED_DISTANCE,
        context="matching.distance",
    )
    replacement = matching["replacement"]
    if replacement is not False:
        raise ValueError("matching.replacement must be false.")
    caliper_rule = _supported_string(
        matching["caliper_rule"],
        expected=SUPPORTED_CALIPER_RULE,
        context="matching.caliper_rule",
    )

    balance_variables = _unique_strings(
        raw["balance_variables"],
        context="balance_variables",
    )
    families = _load_families(raw["families"])
    specifications = _load_specifications(raw["specifications"], families=families)

    plots = _mapping(raw["plots"], context="plots")
    _require_exact_keys(
        plots,
        {
            "primary_specification",
            "retention_sensitivity",
            "balance_comparison",
            "summary_specifications",
            "frontier_families",
            "frontier_calipers_sd",
        },
        context="plots",
    )
    primary_specification = _nonempty_string(
        plots["primary_specification"],
        context="plots.primary_specification",
    )
    retention_sensitivity = _unique_strings(
        plots["retention_sensitivity"],
        context="plots.retention_sensitivity",
    )
    balance_comparison = _unique_strings(
        plots["balance_comparison"],
        context="plots.balance_comparison",
    )
    summary_specifications = _unique_strings(
        plots["summary_specifications"],
        context="plots.summary_specifications",
    )
    frontier_families = _unique_strings(
        plots["frontier_families"],
        context="plots.frontier_families",
    )
    frontier_calipers_sd = _positive_floats(
        plots["frontier_calipers_sd"],
        context="plots.frontier_calipers_sd",
    )

    _require_references(
        (
            primary_specification,
            *retention_sensitivity,
            *balance_comparison,
            *summary_specifications,
        ),
        available=specifications,
        context="plots specification",
    )
    _require_references(
        frontier_families,
        available=families,
        context="plots frontier family",
    )

    return MatchingSettings(
        source_path=source_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        schema_version=schema_version,
        group_variable=group_variable,
        reference_sign=reference_sign,
        method=method,
        standardization=standardization,
        distance=distance,
        replacement=bool(replacement),
        caliper_rule=caliper_rule,
        balance_variables=balance_variables,
        families=MappingProxyType(families),
        specifications=MappingProxyType(specifications),
        primary_specification=primary_specification,
        retention_sensitivity=retention_sensitivity,
        balance_comparison=balance_comparison,
        summary_specifications=summary_specifications,
        frontier_families=frontier_families,
        frontier_calipers_sd=frontier_calipers_sd,
    )


def _load_families(raw: object) -> dict[str, MatchingFamily]:
    families_raw = _mapping(raw, context="families")
    if not families_raw:
        raise ValueError("families must define at least one matching family.")
    families: dict[str, MatchingFamily] = {}
    for identifier, value in families_raw.items():
        identifier = _nonempty_string(identifier, context="family identifier")
        family = _mapping(value, context=f"family {identifier!r}")
        _require_exact_keys(
            family,
            {"label", "variables"},
            context=f"family {identifier!r}",
        )
        families[identifier] = MatchingFamily(
            identifier=identifier,
            label=_nonempty_string(
                family["label"],
                context=f"family {identifier!r}.label",
            ),
            variables=_unique_strings(
                family["variables"],
                context=f"family {identifier!r}.variables",
            ),
        )
    return families


def _load_specifications(
    raw: object,
    *,
    families: Mapping[str, MatchingFamily],
) -> dict[str, MatchingSpecification]:
    specifications_raw = _mapping(raw, context="specifications")
    if not specifications_raw:
        raise ValueError("specifications must define at least one specification.")
    specifications: dict[str, MatchingSpecification] = {}
    for identifier, value in specifications_raw.items():
        identifier = _nonempty_string(identifier, context="specification identifier")
        specification = _mapping(value, context=f"specification {identifier!r}")
        _require_exact_keys(
            specification,
            {"family", "caliper_sd"},
            context=f"specification {identifier!r}",
        )
        family = _nonempty_string(
            specification["family"],
            context=f"specification {identifier!r}.family",
        )
        if family not in families:
            raise ValueError(
                f"Specification {identifier!r} references unknown family {family!r}."
            )
        specifications[identifier] = MatchingSpecification(
            identifier=identifier,
            family=family,
            caliper_sd=_positive_float(
                specification["caliper_sd"],
                context=f"specification {identifier!r}.caliper_sd",
            ),
        )
    return specifications


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object.")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError(f"Invalid keys for {context}: {'; '.join(details)}.")


def _nonempty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string.")
    return value


def _unique_strings(value: object, *, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a nonempty array of strings.")
    out = tuple(
        _nonempty_string(item, context=f"{context} item")
        for item in value
    )
    if not out:
        raise ValueError(f"{context} must be a nonempty array of strings.")
    if len(set(out)) != len(out):
        raise ValueError(f"{context} must not contain duplicates.")
    return out


def _integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer.")
    return value


def _positive_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite positive number.")
    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{context} must be a finite positive number.")
    return out


def _positive_floats(value: object, *, context: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a nonempty array of numbers.")
    out = tuple(
        _positive_float(item, context=f"{context} item")
        for item in value
    )
    if not out:
        raise ValueError(f"{context} must be a nonempty array of numbers.")
    if len(set(out)) != len(out):
        raise ValueError(f"{context} must not contain duplicates.")
    return out


def _supported_string(value: object, *, expected: str, context: str) -> str:
    out = _nonempty_string(value, context=context)
    if out != expected:
        raise ValueError(f"{context} must be {expected!r}; got {out!r}.")
    return out


def _require_references(
    values: Sequence[str],
    *,
    available: Mapping[str, object],
    context: str,
) -> None:
    missing = sorted(set(values).difference(available))
    if missing:
        raise ValueError(
            f"{context} references undefined identifier(s): {', '.join(missing)}."
        )
