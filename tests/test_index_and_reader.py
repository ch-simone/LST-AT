from pathlib import Path

from lstat.collate import pad_collate
from lstat.dataset import LstatDataset, Normalization, _time_channels
from lstat.geotiff import read_geotiff
from lstat.index import build_examples, build_pairs


DATA_ROOT = Path("/Users/simonechierichini/Documents/Codex/LST-AT/data/Monthly")


def test_build_pairs_and_examples():
    pairs = build_pairs(DATA_ROOT)
    examples = build_examples(DATA_ROOT)
    assert len(pairs) == 21769
    assert len(examples) == 43538
    assert {examples[0].phase, examples[1].phase} == {"day", "night"}


def test_read_sample_pair_shapes_match():
    modis = read_geotiff(
        DATA_ROOT
        / "2024"
        / "10"
        / "MODIS"
        / "MODIS_LST_Monthly_Rome_2024_10_day-night_COG.tif"
    )
    era5 = read_geotiff(
        DATA_ROOT
        / "2024"
        / "10"
        / "ERA5-Land"
        / "ERA5Land_Tair_Monthly_Rome_2024_10_day-night_COG.tif"
    )
    assert modis.array.shape == (12, 20, 2)
    assert era5.array.shape == (12, 20, 2)
    assert modis.valid_mask[:, :, 0].sum() > 0
    assert era5.valid_mask[:, :, 0].sum() > 0


def test_collate_keeps_original_shapes():
    examples = [
        example
        for example in build_examples(DATA_ROOT, years=[2024])
        if example.city == "Rome" and example.month == 10
    ][:2]
    dataset = LstatDataset(examples, Normalization())
    batch = pad_collate([dataset[0], dataset[1]], min_size=32, multiple=8)
    assert tuple(batch["x"].shape) == (2, 5, 32, 32)
    assert batch["shape"].tolist() == [[12, 20], [12, 20]]


def test_build_daily_pairs_from_filename(tmp_path):
    daily_root = tmp_path / "Daily"
    modis = (
        daily_root
        / "2024"
        / "10"
        / "15"
        / "MODIS"
        / "MODIS_LST_Daily_Rome_2024_10_15_day-night_COG.tif"
    )
    era5 = (
        daily_root
        / "2024"
        / "10"
        / "15"
        / "ERA5-Land"
        / "ERA5Land_Tair_Daily_Rome_2024_10_15_day-night_COG.tif"
    )
    modis.parent.mkdir(parents=True)
    era5.parent.mkdir(parents=True)
    modis.touch()
    era5.touch()

    pairs = build_pairs(daily_root)
    examples = build_examples(daily_root)

    assert len(pairs) == 1
    assert pairs[0].temporal_resolution == "daily"
    assert pairs[0].day == 15
    assert len(examples) == 2
    assert {example.phase for example in examples} == {"day", "night"}


def test_daily_time_channels_use_day_of_year():
    jan = _time_channels(year=2024, month=1, day=1, phase="day", shape=(2, 2))
    jul = _time_channels(year=2024, month=7, day=1, phase="day", shape=(2, 2))
    assert len(jan) == 3
    assert jan[1][0, 0] != jul[1][0, 0]
