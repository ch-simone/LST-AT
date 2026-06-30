from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


FILENAME_RE = re.compile(
    r"^(MODIS_LST|ERA5Land_Tair)_(Monthly|Daily)_(.+)_(\d{4})_(\d{2})(?:_(\d{2}))?_day-night_COG\.tif$"
)


@dataclass(frozen=True)
class PairedRaster:
    city: str
    year: int
    month: int
    day: int | None
    temporal_resolution: str
    modis_path: Path
    era5_path: Path


@dataclass(frozen=True)
class SingleMapExample:
    city: str
    year: int
    month: int
    day: int | None
    temporal_resolution: str
    phase: str
    modis_path: Path
    era5_path: Path
    band_index: int


def build_pairs(data_root: str | Path) -> list[PairedRaster]:
    data_root = Path(data_root)
    records: dict[tuple[str, str, int, int, int | None], dict[str, Path]] = {}

    for path in data_root.rglob("*.tif"):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        prefix, temporal_resolution, city, year, month, day = match.groups()
        if temporal_resolution == "Daily" and day is None:
            continue
        if temporal_resolution == "Monthly" and day is not None:
            continue
        product = "modis" if prefix == "MODIS_LST" else "era5"
        key = (
            temporal_resolution.lower(),
            city,
            int(year),
            int(month),
            int(day) if day is not None else None,
        )
        records.setdefault(key, {})[product] = path

    pairs = []
    for (temporal_resolution, city, year, month, day), paths in sorted(records.items()):
        if "modis" in paths and "era5" in paths:
            pairs.append(
                PairedRaster(
                    city=city,
                    year=year,
                    month=month,
                    day=day,
                    temporal_resolution=temporal_resolution,
                    modis_path=paths["modis"],
                    era5_path=paths["era5"],
                )
            )
    return pairs


def build_examples(
    data_root: str | Path,
    years: set[int] | list[int] | None = None,
    phases: tuple[str, ...] = ("day", "night"),
) -> list[SingleMapExample]:
    allowed_years = set(years) if years is not None else None
    examples = []
    for pair in build_pairs(data_root):
        if allowed_years is not None and pair.year not in allowed_years:
            continue
        for phase in phases:
            if phase not in ("day", "night"):
                raise ValueError(f"Unknown phase: {phase}")
            examples.append(
                SingleMapExample(
                    city=pair.city,
                    year=pair.year,
                    month=pair.month,
                    day=pair.day,
                    temporal_resolution=pair.temporal_resolution,
                    phase=phase,
                    modis_path=pair.modis_path,
                    era5_path=pair.era5_path,
                    band_index=0 if phase == "day" else 1,
                )
            )
    return examples
