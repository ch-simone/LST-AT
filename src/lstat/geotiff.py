from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import struct
import zlib

import numpy as np


TIFF_TYPE_SIZE = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
}

TIFF_TYPE_FORMAT = {
    1: "B",
    2: "c",
    3: "H",
    4: "I",
    5: "II",
    6: "b",
    7: "B",
    8: "h",
    9: "i",
    10: "ii",
    11: "f",
    12: "d",
}

TAG_NAMES = {
    256: "width",
    257: "height",
    258: "bits_per_sample",
    259: "compression",
    273: "strip_offsets",
    277: "samples_per_pixel",
    278: "rows_per_strip",
    279: "strip_byte_counts",
    284: "planar_configuration",
    317: "predictor",
    322: "tile_width",
    323: "tile_length",
    324: "tile_offsets",
    325: "tile_byte_counts",
    339: "sample_format",
    33550: "model_pixel_scale",
    33922: "model_tiepoint",
    42112: "gdal_metadata",
    42113: "nodata",
}


@dataclass(frozen=True)
class Raster:
    array: np.ndarray
    nodata: float | None
    metadata: dict

    @property
    def valid_mask(self) -> np.ndarray:
        valid = np.isfinite(self.array)
        if self.nodata is not None:
            valid &= ~np.isclose(self.array, self.nodata)
        return valid


def read_geotiff(path: str | Path) -> Raster:
    path = Path(path)
    stat = path.stat()
    backend = os.environ.get("LSTAT_TIFF_BACKEND", "auto").lower()
    return _read_geotiff_cached(str(path), stat.st_mtime_ns, stat.st_size, backend)


@lru_cache(maxsize=int(os.environ.get("LSTAT_RASTER_CACHE_SIZE", "512")))
def _read_geotiff_cached(
    path: str,
    mtime_ns: int,
    size: int,
    backend: str,
) -> Raster:
    """Read the small tiled float GeoTIFFs used by this project.

    This is intentionally narrow: it supports little/big endian tiled or
    stripped, deflate-compressed, chunky float rasters. That avoids making
    rasterio/GDAL mandatory for the first project scaffold.
    """
    path = Path(path)
    if backend not in {"auto", "builtin", "tifffile"}:
        raise ValueError(
            "LSTAT_TIFF_BACKEND must be one of: auto, builtin, tifffile"
        )
    if backend in {"auto", "tifffile"}:
        try:
            return _read_with_tifffile(path)
        except ImportError:
            if backend == "tifffile":
                raise

    data = path.read_bytes()
    endian = _read_endian(data)
    tags = _read_first_ifd(data, endian)

    compression = tags.get("compression", 1)
    if compression not in (1, 8):
        raise ValueError(f"Unsupported TIFF compression {compression} in {path}")

    samples = int(tags.get("samples_per_pixel", 1))
    height = int(tags["height"])
    width = int(tags["width"])
    dtype = _numpy_dtype(endian, tags["sample_format"], tags["bits_per_sample"])

    if "tile_offsets" in tags:
        array = _read_tiled(data, tags, compression, dtype, height, width, samples)
    else:
        array = _read_stripped(data, tags, compression, dtype, height, width, samples)

    nodata = float(tags["nodata"]) if "nodata" in tags else None
    return Raster(array=array, nodata=nodata, metadata=tags)


def _read_with_tifffile(path: Path) -> Raster:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = {
            TAG_NAMES.get(tag.code, str(tag.code)): tag.value
            for tag in page.tags.values()
            if TAG_NAMES.get(tag.code, str(tag.code)) is not None
        }
        array = page.asarray()

    width = int(tags.get("width", array.shape[-1]))
    height = int(tags.get("height", array.shape[-2]))
    samples = int(tags.get("samples_per_pixel", 1))
    array = _normalize_tifffile_array(array, height=height, width=width, samples=samples)
    nodata = float(tags["nodata"]) if "nodata" in tags else None
    return Raster(array=array.astype("float32", copy=False), nodata=nodata, metadata=tags)


def _normalize_tifffile_array(
    array: np.ndarray,
    height: int,
    width: int,
    samples: int,
) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D TIFF array, got shape {array.shape}")
    if array.shape == (height, width, samples):
        return array
    if array.shape == (samples, height, width):
        return np.moveaxis(array, 0, -1)
    if array.shape[-1] == samples:
        return array
    if array.shape[0] == samples:
        return np.moveaxis(array, 0, -1)
    raise ValueError(f"Cannot normalize TIFF array shape {array.shape}")


def _read_endian(data: bytes) -> str:
    if data[:2] == b"II":
        return "<"
    if data[:2] == b"MM":
        return ">"
    raise ValueError("Not a TIFF file")


def _read_first_ifd(data: bytes, endian: str) -> dict:
    magic = struct.unpack(endian + "H", data[2:4])[0]
    if magic != 42:
        raise ValueError(f"Unsupported TIFF magic number {magic}")

    ifd_offset = struct.unpack(endian + "I", data[4:8])[0]
    entry_count = struct.unpack(endian + "H", data[ifd_offset : ifd_offset + 2])[0]
    tags = {}

    for i in range(entry_count):
        offset = ifd_offset + 2 + i * 12
        tag, typ, count, value_offset = struct.unpack(endian + "HHII", data[offset : offset + 12])
        name = TAG_NAMES.get(tag)
        if name is None:
            continue
        tags[name] = _read_tag_value(
            data=data,
            endian=endian,
            typ=typ,
            count=count,
            value_offset=value_offset,
            inline=data[offset + 8 : offset + 12],
        )

    return tags


def _read_tag_value(
    data: bytes,
    endian: str,
    typ: int,
    count: int,
    value_offset: int,
    inline: bytes,
):
    size = TIFF_TYPE_SIZE[typ] * count
    raw = inline if size <= 4 else data[value_offset : value_offset + size]

    if typ == 2:
        return raw.split(b"\0")[0].decode("utf-8", "replace")

    values = []
    if typ in (5, 10):
        for i in range(count):
            a, b = struct.unpack(endian + TIFF_TYPE_FORMAT[typ], raw[i * 8 : i * 8 + 8])
            values.append(a / b if b else None)
    else:
        item_size = TIFF_TYPE_SIZE[typ]
        fmt = endian + TIFF_TYPE_FORMAT[typ]
        for i in range(count):
            values.append(struct.unpack(fmt, raw[i * item_size : i * item_size + item_size])[0])

    return values[0] if count == 1 else tuple(values)


def _numpy_dtype(endian: str, sample_format, bits_per_sample) -> np.dtype:
    sample_format = _first(sample_format)
    bits_per_sample = _first(bits_per_sample)
    if sample_format == 3 and bits_per_sample == 32:
        return np.dtype(endian + "f4")
    if sample_format == 3 and bits_per_sample == 64:
        return np.dtype(endian + "f8")
    raise ValueError(
        f"Unsupported sample format/bits: {sample_format}/{bits_per_sample}"
    )


def _read_tiled(
    data: bytes,
    tags: dict,
    compression: int,
    dtype: np.dtype,
    height: int,
    width: int,
    samples: int,
) -> np.ndarray:
    tile_width = int(tags["tile_width"])
    tile_height = int(tags["tile_length"])
    offsets = _as_tuple(tags["tile_offsets"])
    byte_counts = _as_tuple(tags["tile_byte_counts"])
    tiles_x = (width + tile_width - 1) // tile_width
    result = np.zeros((height, width, samples), dtype=dtype.newbyteorder("="))

    for tile_index, (offset, byte_count) in enumerate(zip(offsets, byte_counts)):
        raw = data[int(offset) : int(offset) + int(byte_count)]
        if compression == 8:
            raw = zlib.decompress(raw)
        tile = np.frombuffer(raw, dtype=dtype).reshape(tile_height, tile_width, samples)
        tile_y = tile_index // tiles_x
        tile_x = tile_index % tiles_x
        y0 = tile_y * tile_height
        x0 = tile_x * tile_width
        y1 = min(y0 + tile_height, height)
        x1 = min(x0 + tile_width, width)
        result[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]

    return result


def _read_stripped(
    data: bytes,
    tags: dict,
    compression: int,
    dtype: np.dtype,
    height: int,
    width: int,
    samples: int,
) -> np.ndarray:
    chunks = []
    for offset, byte_count in zip(
        _as_tuple(tags["strip_offsets"]),
        _as_tuple(tags["strip_byte_counts"]),
    ):
        raw = data[int(offset) : int(offset) + int(byte_count)]
        chunks.append(zlib.decompress(raw) if compression == 8 else raw)
    raw_data = b"".join(chunks)
    return np.frombuffer(raw_data, dtype=dtype).reshape(height, width, samples)


def _as_tuple(value) -> tuple:
    return value if isinstance(value, tuple) else (value,)


def _first(value):
    return value[0] if isinstance(value, tuple) else value
