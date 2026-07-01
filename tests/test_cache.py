import json

import numpy as np

from lstat.cache import CachedLstatDataset
from lstat.collate import pad_collate


def test_cached_dataset_reads_original_shape(tmp_path):
    cache_dir = tmp_path / "train"
    cache_dir.mkdir()
    x = np.lib.format.open_memmap(
        cache_dir / "x.npy",
        mode="w+",
        dtype="float16",
        shape=(1, 5, 4, 4),
    )
    y = np.lib.format.open_memmap(
        cache_dir / "y.npy",
        mode="w+",
        dtype="float16",
        shape=(1, 1, 4, 4),
    )
    mask = np.lib.format.open_memmap(
        cache_dir / "mask.npy",
        mode="w+",
        dtype="uint8",
        shape=(1, 1, 4, 4),
    )
    x[:] = 0
    y[:] = 0
    mask[:] = 0
    x[0, :, :2, :3] = 1
    y[0, :, :2, :3] = 2
    mask[0, :, :2, :3] = 1
    x.flush()
    y.flush()
    mask.flush()
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "split": "train",
                "count": 1,
                "dtype": "float16",
                "channels": 5,
                "height": 4,
                "width": 4,
                "records": [
                    {
                        "city": "Rome",
                        "year": 2024,
                        "month": 10,
                        "day": 7,
                        "temporal_resolution": "daily",
                        "phase": "day",
                        "height": 2,
                        "width": 3,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = CachedLstatDataset(cache_dir)
    item = dataset[0]
    assert item["x"].shape == (5, 2, 3)
    assert item["x"].dtype == np.float32
    assert item["mask"].sum() == 6

    batch = pad_collate([item], min_size=4, multiple=4)
    assert tuple(batch["x"].shape) == (1, 5, 4, 4)
    assert batch["phase"] == ["day"]


def test_sample_cache_reads_original_shape(tmp_path):
    cache_dir = tmp_path / "train"
    sample_dir = cache_dir / "samples"
    sample_dir.mkdir(parents=True)
    np.savez(
        sample_dir / "00000000.npz",
        x=np.ones((5, 2, 3), dtype="float16"),
        y=np.full((1, 2, 3), 2, dtype="float16"),
        mask=np.ones((1, 2, 3), dtype="uint8"),
    )
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "storage": "samples",
                "split": "train",
                "count": 1,
                "dtype": "float16",
                "channels": 5,
                "height": 2,
                "width": 3,
                "records": [
                    {
                        "city": "Rome",
                        "year": 2024,
                        "month": 10,
                        "day": 7,
                        "temporal_resolution": "daily",
                        "phase": "night",
                        "height": 2,
                        "width": 3,
                        "file": "samples/00000000.npz",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = CachedLstatDataset(cache_dir)
    item = dataset[0]
    assert item["x"].shape == (5, 2, 3)
    assert item["x"].dtype == np.float32
    assert item["mask"].sum() == 6
    assert item["phase"] == "night"
