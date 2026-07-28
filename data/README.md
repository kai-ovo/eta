# Data Placeholder

Large data files are intentionally not stored in this GitHub repository. They are published on Zenodo:

- Datasets (ERA5-Land, toy, smoke-test subset): [`10.5281/zenodo.21635446`](https://doi.org/10.5281/zenodo.21635446)
- Checksums: `SHA256SUMS-data.txt` in the same record; verify with `shasum -a 256 -c`
- License: CC-BY-4.0. The ERA5-Land files are derived products of ERA5-Land (Copernicus Climate Change Service / ECMWF) and remain subject to the Licence to Use Copernicus Products.

Stage downloaded or generated data under the repository checkout:

```text
/path/to/eta/data/
  toy.pth
  era5land_USA_SouthEast_1999-2023_dailymax.nc
```

`toy.pth` holds the toy dataset tensors (`X_all`, `Y_all`, `in_grid`, `out_grid`, `x_train`, `y_train`). `era5land_USA_SouthEast_1999-2023_dailymax.nc` holds daily-maximum total precipitation for the US Southeast over 1999-2023 on an 80x160 grid, derived from ERA5-Land. Both are published in the data archive above.

Do not commit `.pt`, `.pth`, `.nc`, `.npy`, `.npz`, `.h5`, `.hdf5`, `.zarr`, or other large binary data artifacts here.
