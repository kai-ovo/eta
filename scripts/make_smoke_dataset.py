"""Build the small ERA5-Land smoke-test subset from the full daily-max file.

The subset keeps only the `tp` variable and the full 80x160 spatial grid, which
the pretrained SRCNN checkpoints require at ds_fact=10. It subsamples time on a
uniform stride so the marginal distribution of the scalar observable
`g(u)=max(u)` and its tail fraction stay close to the full record.

Requires xarray and netCDF4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = "data/era5land_USA_SouthEast_1999-2023_dailymax.nc"
TRIM_TAIL_THRESH = 240.0
W1_TAIL_THRESH = 150.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ndays", type=int, default=1000, help="Target number of retained days.")
    parser.add_argument("--complevel", type=int, default=4, help="zlib compression level, 0 disables.")
    return parser.parse_args()


def main() -> int:
    import xarray as xr

    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Missing source dataset: {args.source}")

    source = xr.open_dataset(args.source, engine="netcdf4")
    n_full = source.sizes["time"]
    if args.ndays >= n_full:
        raise ValueError(f"--ndays={args.ndays} must be smaller than the source record ({n_full} days)")

    stride = n_full / args.ndays
    keep = np.unique((np.arange(args.ndays) * stride).astype(int))
    subset = source[["tp"]].isel(time=keep)

    subset.attrs = dict(source.attrs)
    subset.attrs.update(
        {
            "title": "ERA5-Land daily-maximum total precipitation, US Southeast, smoke-test subset",
            "summary": (
                f"Uniform time subsample ({len(keep)} of {n_full} days) of the daily-maximum "
                "total precipitation field used in the eta-learning experiments. Retains the full "
                "80x160 spatial grid and the tp variable only. Intended for pipeline verification, "
                "not for reproducing published numbers."
            ),
            "source_days": n_full,
            "subset_days": len(keep),
            "subset_method": "uniform time stride",
        }
    )

    encoding = {"tp": {"zlib": args.complevel > 0, "complevel": args.complevel, "dtype": "float32"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_netcdf(args.output, engine="netcdf4", encoding=encoding)

    def tail_report(values: np.ndarray, label: str) -> None:
        maxima = (values * 1000).max(axis=(1, 2))
        kept = maxima[maxima <= TRIM_TAIL_THRESH]
        n_tail = int((kept > W1_TAIL_THRESH).sum())
        print(
            f"{label}: {len(maxima)} days, {len(kept)} after trim(>{TRIM_TAIL_THRESH:g}), "
            f"{n_tail} tail days (>{W1_TAIL_THRESH:g}) = {100 * n_tail / max(len(kept), 1):.2f}%"
        )

    tail_report(source["tp"].values, "full  ")
    tail_report(subset["tp"].values, "subset")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
