# External Data and Model Assets

This repository intentionally does not store large climate datasets, generated samples, pretrained model checkpoints, or binary artifacts. They are deposited on Zenodo and staged under the checkout as described below.

## Archive DOIs

- Datasets (ERA5-Land, toy, smoke-test subset): [`10.5281/zenodo.21635446`](https://doi.org/10.5281/zenodo.21635446)
- Pretrained models and generated samples: [`10.5281/zenodo.21635468`](https://doi.org/10.5281/zenodo.21635468)
- Code archive, all versions: [`10.5281/zenodo.21636106`](https://doi.org/10.5281/zenodo.21636106)
- Code archive, `v1.0.0`: [`10.5281/zenodo.21636107`](https://doi.org/10.5281/zenodo.21636107)

Checksums are published as `SHA256SUMS-data.txt` in the dataset record and `SHA256SUMS-models-samples.txt` inside `eta-models-samples-v1.0.0.zip` in the model record. Verify with `shasum -a 256 -c`.

Both records are released under CC-BY-4.0. The ERA5-Land files are derived products of ERA5-Land (Copernicus Climate Change Service / ECMWF) and remain subject to the Licence to Use Copernicus Products.

## Environment Variables

The default path setup points both code and artifacts at the repository root:

```sh
export ETA_CODE_DIR=/path/to/eta
export ETA_WORK_DIR=/path/to/eta
```

You may copy `.env.example` to `.env` for local shell tooling, but notebooks and scripts default to the repository root when `ETA_WORK_DIR` is unset. Ordinary runs should leave `ETA_WORK_DIR` unset; use it only for an advanced local storage override.

## In-Repository Data Layout

```text
/path/to/eta/data/
  toy.pth
  era5land_USA_SouthEast_1999-2023_dailymax.nc
  README.md
```

Both files are published in the data archive above and stage directly under `data/`. The ERA5-Land file is a derived product: daily-maximum total precipitation for the US Southeast over 1999-2023 on an 80x160 grid. Original ERA5-Land data remains available from the Copernicus Climate Data Store under its own licence terms.

## In-Repository Checkpoint Layout

```text
/path/to/eta/models/
  precip-srcnn/
    # srcnn-mse-*.pth, srcnn-eta-*.pth, GEVD, and misspecified-tail checkpoints.
  fm/
    # LR and HR Flow Matching checkpoints.
/path/to/eta/samples/
  # Generated sample arrays for plot-only workflows.
```

The complete local layout used by notebooks and command-line wrappers is:

```text
data/toy.pth
data/era5land_USA_SouthEast_1999-2023_dailymax.nc
models/precip-srcnn/
models/fm/
samples/
```

The public archive should mirror this structure under the repository root, or provide a documented mapping.

The command-line experiment wrappers expect this notebook-compatible structure by default. If an archive separates data and checkpoints into different roots, either stage symlinks under the checkout or pass explicit checkpoint/sample paths where the relevant script supports them.

## Missing Asset Checks

Use the non-downloading checker to verify local roots before running data-dependent notebooks:

```sh
python scripts/check_external_assets.py
```

The checker reports missing roots and expected subdirectories, then points to the DOI placeholders. It does not download, upload, or modify any external assets.
