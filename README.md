# LST to AT

Deep learning experiments for predicting near-surface air temperature (AT) maps
from MODIS land surface temperature (LST) maps.

The first training setup treats day and night as separate single-map examples:

```text
MODIS LST_Day   -> ERA5-Land Tair_Day
MODIS LST_Night -> ERA5-Land Tair_Night
```

The default model is a compact residual U-Net with a masked regression loss.
Inputs are:

- corrected LST map
- valid-pixel mask
- day/night flag
- month sine/cosine channels

## Dataset

The expected dataset root is:

```text
/Users/simonechierichini/Documents/Codex/LST-AT/data/Monthly
```

with files arranged as:

```text
Monthly/{year}/{month}/MODIS/MODIS_LST_Monthly_{city}_{year}_{month}_day-night_COG.tif
Monthly/{year}/{month}/ERA5-Land/ERA5Land_Tair_Monthly_{city}_{year}_{month}_day-night_COG.tif
```

## Setup

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

For Weights & Biases training monitoring:

```bash
pip install -e ".[tracking]"
wandb login
```

For PyTorch, install the build appropriate for your machine from
https://pytorch.org/get-started/locally/ if the generic dependency is not ideal.

## Inspect Data

```bash
python scripts/inspect_dataset.py --config configs/default.toml
```

## Train

Quick smoke run:

```bash
python -m lstat.train --config configs/smoke.toml
```

Small CPU learning run:

```bash
python -m lstat.train --config configs/local_cpu.toml
```

Full configured run:

```bash
python -m lstat.train --config configs/default.toml
```

Artifacts are written to `runs/resunet_monthly` by default.
Training metrics are logged to Weights & Biases when `[wandb].enabled = true`.
Validation residual images are logged at epoch 1, every
`[evaluation].log_image_interval` epochs, and the final epoch. The residual is:

```text
target AT - predicted AT
```

## Notes

The MODIS values in this dataset appear to have been stored after a double
temperature transform. By default the loader applies:

```python
lst_celsius = (stored_value + 273.15) / 0.02 - 273.15
```

This matches plausible sampled LST values, but it should be verified against
the dataset provenance before using the model for formal analysis.
