from pathlib import Path

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
