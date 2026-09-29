"""Shared seasonal populations and Stage-2 consumer contracts (decision 010)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import xarray as xr

from . import selectors

JJA_MONTHS = (6, 7, 8)
SEASON_ATTRS = (
    "all_seasons",
    "season_months",
    "require_full_event",
    "season_selection_rule",
    "season_anchor",
)


def resolve_months(
    season_months: Sequence[int] | None = None, *, all_seasons: bool = False
) -> tuple[int, ...] | None:
    """Resolve the default JJA population or an explicit all-season request."""
    if all_seasons:
        if season_months is not None:
            raise ValueError("Pass season_months or all_seasons=True, not both.")
        return None
    months = JJA_MONTHS if season_months is None else tuple(season_months)
    if not months or any(
        isinstance(month, (bool, np.bool_))
        or not isinstance(month, (int, np.integer))
        or not 1 <= month <= 12
        for month in months
    ):
        raise ValueError(
            "--season-months must contain calendar months between 1 and 12."
        )
    return tuple(sorted(set(months)))


def add_season_arguments(
    parser: argparse.ArgumentParser, *, default_full_event: bool | None = True
) -> None:
    """Add shared CLI switches; None omits the event-only membership switch."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--season-months",
        type=int,
        nargs="+",
        metavar="MONTH",
        help="Target calendar months (default: 6 7 8, JJA). Does not clip source history.",
    )
    group.add_argument(
        "--all-seasons",
        action="store_true",
        help="Explicitly use the full saved population instead of seasonal selection.",
    )
    if default_full_event is not None:
        parser.add_argument(
            "--require-full-event",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Require the entire detected event interval within the season "
                f"(default: {default_full_event}); otherwise use peak_time."
            ),
        )
    parser.set_defaults(_default_full_event=default_full_event)


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse and resolve shared defaults once, before downstream validation."""
    args = parser.parse_args()
    try:
        months = resolve_months(args.season_months, all_seasons=args.all_seasons)
        args.season_months = None if months is None else list(months)
        if args._default_full_event is not None:
            if args.all_seasons and args.require_full_event is True:
                raise ValueError(
                    "--require-full-event cannot be combined with --all-seasons."
                )
            if args.require_full_event is None:
                args.require_full_event = (
                    args._default_full_event and not args.all_seasons
                )
        del args._default_full_event
    except ValueError as exc:
        parser.error(str(exc))
    return args


def season_kwargs(args: argparse.Namespace) -> dict:
    """Return the resolved population options accepted by product consumers."""
    result = {"season_months": args.season_months, "all_seasons": args.all_seasons}
    if hasattr(args, "require_full_event"):
        result["require_full_event"] = args.require_full_event
    return result


def select_event_population(
    ds: xr.Dataset,
    *,
    season_months: Sequence[int] | None = None,
    all_seasons: bool = False,
    require_full_event: bool | None = None,
) -> xr.Dataset:
    """Select event rows, preserving every source timestamp for window extraction."""
    months = resolve_months(season_months, all_seasons=all_seasons)
    if months is None:
        if require_full_event:
            raise ValueError("Full-event season selection requires seasonal months.")
        return ds
    return selectors.select_events_by_season(
        ds,
        months,
        require_full_event=True if require_full_event is None else require_full_event,
    )


def selection_attrs(
    months: Sequence[int] | None, *, anchor: str, require_full_event: bool = False
) -> dict[str, str]:
    """Describe membership independently of feature-window geometry."""
    rule = (
        "all_seasons"
        if months is None
        else ("full_event" if require_full_event else "anchor_month")
    )
    return {
        "season_months": "" if months is None else ",".join(map(str, months)),
        "season_selection_rule": rule,
        "season_anchor": anchor,
    }


def validate_stage2_season(
    ds: xr.Dataset,
    *,
    season_months: Sequence[int] | None = None,
    all_seasons: bool = False,
    require_full_event: bool | None = False,
) -> None:
    """Reject incompatible provenance or dates before consuming a Stage-2 table.

    ``require_full_event=None`` accepts either documented event rule, for a
    consumer that will explicitly select complete events before matching.
    Baseline membership is always anchor-only.
    """
    months = resolve_months(season_months, all_seasons=all_seasons)
    stage = ds.attrs.get("pipeline_stage")
    if stage == "stage_2_event_features":
        dim, anchor = "event", "peak_time"
        actual_full = _binary_attr(ds, "require_full_event")
        if require_full_event is not None and actual_full != int(require_full_event):
            option = (
                "--require-full-event" if actual_full else "--no-require-full-event"
            )
            raise ValueError(
                f"Stage-2 event membership differs from the requested rule. Use {option} "
                "to consume this population, or rebuild Stage 2 with the requested rule."
            )
    elif stage == "stage_2_baseline_features":
        dim, anchor, actual_full = "baseline_day", "reference_time", 0
    else:
        raise ValueError(
            f"Expected a Stage-2 feature product; got pipeline_stage={stage!r}."
        )

    actual_all = _binary_attr(ds, "all_seasons")
    if actual_all != int(all_seasons):
        raise ValueError(
            "Stage-2 season mismatch: all-season and seasonal populations cannot be "
            "interchanged. Use matching --all-seasons/--season-months options or rebuild Stage 2."
        )
    raw_months = ds.attrs.get("season_months", "")
    try:
        declared = tuple(
            int(value) for value in str(raw_months).split(",") if value.strip()
        )
        actual_months = resolve_months(declared) if declared else None
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid Stage-2 season_months metadata.") from exc
    if actual_months != months or (actual_all and actual_full):
        raise ValueError(
            f"Stage-2 season_months={raw_months!r} is incompatible with requested "
            f"months={months!r}. Rebuild or request the matching population explicitly."
        )
    expected_attrs = selection_attrs(
        months, anchor=anchor, require_full_event=bool(actual_full)
    )
    for name in ("season_selection_rule", "season_anchor"):
        if name in ds.attrs and ds.attrs[name] != expected_attrs[name]:
            raise ValueError(
                f"Contradictory Stage-2 {name} metadata: {ds.attrs[name]!r}."
            )
    if anchor not in ds or ds[anchor].dims != (dim,):
        raise ValueError(f"Stage-2 season validation requires {anchor}({dim}).")
    dates = ds[anchor]
    if not np.issubdtype(dates.dtype, np.datetime64) or bool(dates.isnull().any()):
        raise ValueError(f"Stage-2 {anchor} must contain finite decoded timestamps.")
    if months is not None:
        if not bool(dates.dt.month.isin(months).all()):
            raise ValueError(
                f"Stage-2 {anchor} contains dates outside declared season {months}."
            )
        if actual_full:
            selected = selectors.select_events_by_season(
                ds, months, require_full_event=True
            )
            if selected.sizes[dim] != ds.sizes[dim]:
                raise ValueError(
                    "Stage-2 full-event membership contains intervals outside its season."
                )


def _binary_attr(ds: xr.Dataset, name: str) -> int:
    value = ds.attrs.get(name)
    if value not in (0, 1):
        raise ValueError(f"Stage-2 season validation requires {name}=0 or 1 metadata.")
    return int(value)


def validate_inherited_event_season(ds: xr.Dataset, **season_options) -> None:
    """Validate the retained event audit rows of a derived spatial product.

    Its original Stage-2 membership metadata and timestamps must be preserved
    by the builder; averaged fields cannot be filtered back to a new season.
    """
    event_view = ds.copy(deep=False)
    event_view.attrs = {**ds.attrs, "pipeline_stage": "stage_2_event_features"}
    validate_stage2_season(event_view, **season_options)
