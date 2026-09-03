"""Plot the regional domains represented in a Stage 1 run directory."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5netcdf
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, config, plot_style

DEFAULT_RUN_ID = "bf232281_20260819"
DEFAULT_RUN_DIR = REPO_ROOT / "results/stage1/runs" / DEFAULT_RUN_ID
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results/region_vis"
    / f"stage1_regional_domains_{DEFAULT_RUN_ID}.png"
)
DEFAULT_EXPECTED_REGION_COUNT = 7
STAGE1_PRODUCT_GLOB = "harmonized_regional_timeseries_*.nc"
MINIMUM_MAP_LATITUDE = 15.0


@dataclass(frozen=True)
class RegionDomain:
    """One distinct regional domain and the Stage 1 files that represent it."""

    name: str
    west: float
    east: float
    south: float
    north: float
    product_paths: tuple[Path, ...]

    @property
    def label(self) -> str:
        """Return the shared display name for the region."""
        try:
            return plot_style.REGION_NAME_MAPPING[self.name]
        except KeyError as exc:
            raise ValueError(
                f"Region {self.name!r} has no shared display name in plot_style."
            ) from exc

    @property
    def color(self) -> str:
        """Return the shared boundary color for the region."""
        try:
            return plot_style.REGION_COLORS[self.name]
        except KeyError as exc:
            raise ValueError(
                f"Region {self.name!r} has no shared color in plot_style."
            ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the distinct configured regions represented by Stage 1 "
            "NetCDF products in one run directory."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Stage 1 run directory (default: {DEFAULT_RUN_DIR}).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output figure path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--expected-region-count",
        type=int,
        default=DEFAULT_EXPECTED_REGION_COUNT,
        help=(
            "Fail unless this many distinct regions are discovered "
            f"(default: {DEFAULT_EXPECTED_REGION_COUNT})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_region_count <= 0:
        raise ValueError("--expected-region-count must be a positive integer.")

    run_dir = args.run_dir.expanduser().resolve()
    domains = discover_stage1_regions(run_dir)
    if len(domains) != args.expected_region_count:
        raise ValueError(
            f"Expected {args.expected_region_count} distinct regions in {run_dir}; "
            f"found {len(domains)}: {', '.join(domain.name for domain in domains)}."
        )

    fig = plot_region_domains(domains, run_label=run_dir.name)
    try:
        written = write_figure(fig, args.output_path)
    finally:
        plt.close(fig)

    product_count = sum(len(domain.product_paths) for domain in domains)
    print(
        f"Wrote {len(domains)} regional domains discovered from "
        f"{product_count} Stage 1 products: {written}"
    )
    return 0


def discover_stage1_regions(run_dir: str | Path) -> tuple[RegionDomain, ...]:
    """Return unique, validated regional domains represented in a Stage 1 run."""
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if not resolved_run_dir.is_dir():
        raise FileNotFoundError(
            f"Stage 1 run directory does not exist: {resolved_run_dir}"
        )

    product_paths = tuple(sorted(resolved_run_dir.glob(STAGE1_PRODUCT_GLOB)))
    if not product_paths:
        raise FileNotFoundError(
            f"No Stage 1 products matching {STAGE1_PRODUCT_GLOB!r} were found "
            f"in {resolved_run_dir}."
        )

    inventory: dict[str, tuple[tuple[float, float, float, float], list[Path]]] = {}
    for product_path in product_paths:
        region, bounds = read_stage1_region_metadata(product_path)
        configured_bounds = configured_region_bounds(region)
        if not np.allclose(bounds, configured_bounds, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Stage 1 bounds in {product_path} do not match src/config.py for "
                f"{region!r}: stored={bounds}, configured={configured_bounds}."
            )

        existing = inventory.get(region)
        if existing is None:
            inventory[region] = (bounds, [product_path])
            continue
        existing_bounds, existing_paths = existing
        if not np.allclose(bounds, existing_bounds, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Stage 1 products for {region!r} contain inconsistent bounds: "
                f"{existing_bounds} and {bounds}."
            )
        existing_paths.append(product_path)

    return tuple(
        RegionDomain(
            name=region,
            west=bounds[0],
            east=bounds[1],
            south=bounds[2],
            north=bounds[3],
            product_paths=tuple(paths),
        )
        for region, (bounds, paths) in sorted(inventory.items())
    )


def read_stage1_region_metadata(
    product_path: str | Path,
) -> tuple[str, tuple[float, float, float, float]]:
    """Read the Stage 1 marker, region, and stored averaging bounds."""
    resolved_path = Path(product_path).expanduser().resolve()
    try:
        with h5netcdf.File(resolved_path, "r") as dataset:
            pipeline_stage = _text_attribute(
                dataset.attrs.get("pipeline_stage"),
                attribute="pipeline_stage",
                path=resolved_path,
            )
            if pipeline_stage != analysis_io.EXPECTED_PIPELINE_STAGE:
                raise ValueError(
                    f"Expected pipeline_stage={analysis_io.EXPECTED_PIPELINE_STAGE!r} "
                    f"in {resolved_path}; found {pipeline_stage!r}."
                )
            region = _text_attribute(
                dataset.attrs.get("region"),
                attribute="region",
                path=resolved_path,
            )
            try:
                regional_temperature = dataset.variables["tas_region"]
            except KeyError as exc:
                raise ValueError(
                    f"Stage 1 product is missing required variable 'tas_region': "
                    f"{resolved_path}"
                ) from exc
            south, north = _bounds_attribute(
                regional_temperature.attrs.get("lat_bounds"),
                attribute="tas_region:lat_bounds",
                path=resolved_path,
            )
            west, east = _bounds_attribute(
                regional_temperature.attrs.get("lon_bounds"),
                attribute="tas_region:lon_bounds",
                path=resolved_path,
            )
    except OSError as exc:
        raise ValueError(f"Could not open Stage 1 product {resolved_path}: {exc}") from exc

    bounds = (west, east, south, north)
    if west >= east or south >= north:
        raise ValueError(
            f"Stage 1 product contains unordered regional bounds in "
            f"{resolved_path}: {bounds}."
        )
    return region, bounds


def configured_region_bounds(region: str) -> tuple[float, float, float, float]:
    """Return west, east, south, north from the canonical region registry."""
    try:
        latitude, longitude = config.REGIONS[region]
    except KeyError as exc:
        available = ", ".join(sorted(config.REGIONS))
        raise ValueError(
            f"Stage 1 product uses unknown region {region!r}; configured regions: "
            f"{available}."
        ) from exc
    return (
        float(longitude.start),
        float(longitude.stop),
        float(latitude.start),
        float(latitude.stop),
    )


def plot_region_domains(
    domains: Sequence[RegionDomain],
    *,
    run_label: str,
) -> plt.Figure:  # type: ignore[type-arg]
    """Render regional boundary wireframes on a Northern Hemisphere map."""
    if not domains:
        raise ValueError("At least one regional domain is required.")
    names = [domain.name for domain in domains]
    if len(names) != len(set(names)):
        raise ValueError("Regional domains must have unique names.")
    for domain in domains:
        _ = domain.label, domain.color

    plot_style.apply_theme()
    projection = ccrs.NorthPolarStereo(central_longitude=-100.0)
    data_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=plot_style.publication_figsize("full", aspect=0.58))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    fig.subplots_adjust(left=0.02, right=0.76, bottom=0.04, top=0.88)

    decorate_northern_hemisphere(ax, data_crs)

    legend_handles = []
    for domain in domains:
        ax.add_patch(
            Rectangle(
                (domain.west, domain.south),
                domain.east - domain.west,
                domain.north - domain.south,
                fill=False,
                edgecolor=domain.color,
                linewidth=2.4,
                transform=data_crs,
                zorder=6,
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=domain.color,
                linewidth=2.4,
                label=domain.label,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(0.985, 0.50),
        title="Region",
        **plot_style.legend_kwargs(handlelength=2.8),
    )
    ax.set_title(f"Stage 1 Regional Domains\nRun {run_label}", pad=16)
    return fig


def decorate_northern_hemisphere(ax, data_crs: ccrs.PlateCarree) -> None:
    """Apply a restrained circular Northern Hemisphere base map."""
    ax.set_extent(
        [-180.0, 180.0, MINIMUM_MAP_LATITUDE, 90.0],
        crs=data_crs,
    )
    ax.add_feature(
        cfeature.OCEAN.with_scale("110m"),
        facecolor="#F7FAFC",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="#ECECEC",
        edgecolor="none",
        zorder=1,
    )
    ax.coastlines(resolution="110m", color="#555555", linewidth=0.65, zorder=3)
    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        edgecolor="#888888",
        linewidth=0.35,
        zorder=3,
    )
    gridlines = ax.gridlines(
        crs=data_crs,
        draw_labels=False,
        color="#9A9A9A",
        linewidth=0.4,
        alpha=0.65,
        linestyle=":",
        zorder=2,
    )
    gridlines.xlocator = FixedLocator(np.arange(-180, 181, 30))
    gridlines.ylocator = FixedLocator(np.arange(15, 91, 15))

    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    circle = mpath.Path(
        np.column_stack([np.sin(angles), np.cos(angles)]) * 0.5 + 0.5
    )
    ax.set_boundary(circle, transform=ax.transAxes)


def write_figure(fig: plt.Figure, output_path: str | Path) -> Path:  # type: ignore[type-arg]
    """Write a map using the shared output resolution."""
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        resolved_path,
        dpi=plot_style.DPI,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    return resolved_path


def _text_attribute(value, *, attribute: str, path: Path) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Stage 1 product has invalid {attribute!r} metadata in {path}: "
            f"{value!r}."
        )
    return value.strip()


def _bounds_attribute(
    value,
    *,
    attribute: str,
    path: Path,
) -> tuple[float, float]:
    try:
        bounds = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Stage 1 product has invalid {attribute!r} metadata in {path}: "
            f"{value!r}."
        ) from exc
    if bounds.shape != (2,) or not np.isfinite(bounds).all():
        raise ValueError(
            f"Stage 1 product has invalid {attribute!r} metadata in {path}: "
            f"{value!r}."
        )
    return float(bounds[0]), float(bounds[1])


if __name__ == "__main__":
    raise SystemExit(main())
