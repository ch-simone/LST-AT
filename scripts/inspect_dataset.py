from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lstat.config import load_config
from lstat.geotiff import read_geotiff
from lstat.index import build_examples, build_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--sample-city", default="Rome")
    parser.add_argument("--sample-year", type=int, default=2024)
    parser.add_argument("--sample-month", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = Path(config["data_root"])
    pairs = build_pairs(data_root)
    examples = build_examples(data_root)
    years = sorted({pair.year for pair in pairs})
    cities = sorted({pair.city for pair in pairs})
    by_year_month = Counter((pair.year, pair.month) for pair in pairs)
    by_city = Counter(pair.city for pair in pairs)

    print(f"data_root: {data_root}")
    print(f"paired rasters: {len(pairs)}")
    print(f"single-map examples: {len(examples)}")
    print(f"cities: {len(cities)}")
    print(f"years: {years[0]}-{years[-1]} ({len(years)})")
    print(f"year-months: {len(by_year_month)}")
    print(f"city pair counts: min={min(by_city.values())} max={max(by_city.values())}")

    sample = next(
        pair
        for pair in pairs
        if pair.city == args.sample_city
        and pair.year == args.sample_year
        and pair.month == args.sample_month
    )
    modis = read_geotiff(sample.modis_path)
    era5 = read_geotiff(sample.era5_path)
    print("")
    print(f"sample: {sample.city} {sample.year}-{sample.month:02d}")
    print(f"MODIS shape: {modis.array.shape}, nodata={modis.nodata}")
    print(f"ERA5 shape:  {era5.array.shape}, nodata={era5.nodata}")
    print(f"paired grid match: {modis.array.shape[:2] == era5.array.shape[:2]}")


if __name__ == "__main__":
    main()
