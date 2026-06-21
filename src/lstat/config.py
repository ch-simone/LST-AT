from __future__ import annotations

from pathlib import Path
import tomllib


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def resolve_output_dir(config: dict, config_path: str | Path) -> Path:
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(config_path).resolve().parent.parent / output_dir
    return output_dir
