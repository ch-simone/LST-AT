from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import numpy as np

from .geotiff import read_geotiff
from .index import SingleMapExample


@dataclass(frozen=True)
class Normalization:
    lst_mean: float = 25.0
    lst_std: float = 15.0
    tair_mean: float = 20.0
    tair_std: float = 12.0
    apply_modis_correction: bool = True


class LstatDataset:
    def __init__(
        self,
        examples: list[SingleMapExample],
        normalization: Normalization,
        include_mask_channel: bool = True,
        include_time_channels: bool = True,
    ):
        self.examples = examples
        self.normalization = normalization
        self.include_mask_channel = include_mask_channel
        self.include_time_channels = include_time_channels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        modis = read_geotiff(example.modis_path)
        era5 = read_geotiff(example.era5_path)

        x_raw = modis.array[:, :, example.band_index].astype("float32")
        y_raw = era5.array[:, :, example.band_index].astype("float32")
        x_valid = modis.valid_mask[:, :, example.band_index]
        y_valid = era5.valid_mask[:, :, example.band_index]
        valid = x_valid & y_valid

        if self.normalization.apply_modis_correction:
            x_raw = (x_raw + 273.15) / 0.02 - 273.15

        x = (x_raw - self.normalization.lst_mean) / self.normalization.lst_std
        y = (y_raw - self.normalization.tair_mean) / self.normalization.tair_std
        x = np.where(valid, x, 0.0).astype("float32")
        y = np.where(valid, y, 0.0).astype("float32")

        channels = [x]
        if self.include_mask_channel:
            channels.append(valid.astype("float32"))
        if self.include_time_channels:
            channels.extend(
                _time_channels(
                    year=example.year,
                    month=example.month,
                    day=example.day,
                    phase=example.phase,
                    shape=x.shape,
                )
            )

        return {
            "x": np.stack(channels).astype("float32"),
            "y": y[None, :, :].astype("float32"),
            "mask": valid[None, :, :].astype("float32"),
            "city": example.city,
            "year": example.year,
            "month": example.month,
            "day": -1 if example.day is None else example.day,
            "temporal_resolution": example.temporal_resolution,
            "phase": example.phase,
        }


def input_channel_count(include_mask_channel: bool, include_time_channels: bool) -> int:
    count = 1
    if include_mask_channel:
        count += 1
    if include_time_channels:
        count += 3
    return count


def _time_channels(
    year: int,
    month: int,
    day: int | None,
    phase: str,
    shape: tuple[int, int],
) -> list[np.ndarray]:
    h, w = shape
    if day is None:
        # Monthly samples get a mid-month annual position.
        annual_position = (month - 0.5) / 12.0
    else:
        day_of_year = date(year, month, day).timetuple().tm_yday
        days_in_year = 366 if _is_leap_year(year) else 365
        annual_position = (day_of_year - 1) / days_in_year
    angle = 2.0 * math.pi * annual_position
    day_flag = 1.0 if phase == "day" else 0.0
    return [
        np.full((h, w), day_flag, dtype="float32"),
        np.full((h, w), math.sin(angle), dtype="float32"),
        np.full((h, w), math.cos(angle), dtype="float32"),
    ]


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
