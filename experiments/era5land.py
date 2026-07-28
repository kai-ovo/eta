"""Command-line version of ERA5Land.ipynb."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.common import (
    add_common_path_args,
    add_precip_args,
    build_srcnn,
    evaluate_precip_model,
    load_state_flexible,
    prepare_precip_context,
    precip_eta_path,
    precip_mse_path,
    precip_tail_objects,
    save_npz,
    save_table_csv,
    train_precip_eta_cli,
    train_srcnn_mse_cli,
)
from revision_utils import precipitation_metrics_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    add_precip_args(parser)
    parser.add_argument("--w1-tail-thresh", type=float, default=150.0)
    parser.add_argument("--omega", type=int, default=30)
    parser.add_argument("--lambda", dest="lambd", type=float, default=1.0)
    parser.add_argument("--mse-epochs", type=int, default=500)
    parser.add_argument("--eta-epochs", type=int, default=150)
    parser.add_argument("--train-mse", action="store_true")
    parser.add_argument("--train-eta", action="store_true")
    parser.add_argument("--skip-mse-eval", action="store_true")
    parser.add_argument("--skip-eta-eval", action="store_true")
    parser.add_argument("--no-varying-days", action="store_true", help="Use fixed true tail days instead of IICT dynamic days.")
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ctx = prepare_precip_context(args)
    _, w1_truedays, w1_truemax = precip_tail_objects(ctx.era5, args.w1_tail_thresh)

    model_mse = build_srcnn(args.ds_fact, args.data_parallel).to(ctx.device)
    mse_path = precip_mse_path(ctx.work_dir, args.num_years, args.ds_fact)
    if args.train_mse:
        print(f"Training MSE baseline and saving to {mse_path}")
        train_srcnn_mse_cli(model_mse, ctx.tensors, args.mse_epochs, args.lr, ctx.device)
        mse_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_mse.state_dict(), mse_path)
    elif mse_path.exists():
        print(f"Loading MSE baseline from {mse_path}")
        model_mse = load_state_flexible(model_mse, mse_path, ctx.device)
    else:
        raise FileNotFoundError(f"Missing MSE checkpoint: {mse_path}. Pass --train-mse to train it.")

    predictions = {}
    metrics_rows = []
    if not args.skip_mse_eval:
        full_output_mse, mse_test_mse, mse_w1 = evaluate_precip_model(model_mse, ctx.tensors.test_loader, w1_truemax, ctx.device)
        predictions["MSE downscaler"] = full_output_mse
        metrics_rows.append({"model": "MSE downscaler", "test_mse": mse_test_mse, "tail_w1": mse_w1})
        print(f"MSE baseline: test_mse={mse_test_mse:.6f}, tail_w1={mse_w1:.6f}")

    eta_path = precip_eta_path(ctx.work_dir, args.num_years, args.ds_fact, int(args.w1_tail_thresh), args.omega)
    model_eta = build_srcnn(args.ds_fact, args.data_parallel).to(ctx.device)
    if args.train_eta:
        print(f"Training eta downscaler and saving best W1 checkpoint to {eta_path}")
        model_eta = deepcopy(model_mse).to(ctx.device)
        model_eta, _ = train_precip_eta_cli(
            model_eta,
            ctx.tensors,
            ctx.era5,
            args.seed,
            args.eta_epochs,
            w1_truemax,
            w1_truedays,
            args.lambd,
            args.omega,
            varying_days=not args.no_varying_days,
            lr=args.lr,
            save_path=eta_path,
            device=ctx.device,
        )
    elif eta_path.exists():
        print(f"Loading eta checkpoint from {eta_path}")
        model_eta = load_state_flexible(model_eta, eta_path, ctx.device)
    elif not args.skip_eta_eval:
        raise FileNotFoundError(f"Missing eta checkpoint: {eta_path}. Pass --train-eta to train it.")

    if not args.skip_eta_eval:
        full_output_eta, eta_test_mse, eta_w1 = evaluate_precip_model(model_eta, ctx.tensors.test_loader, w1_truemax, ctx.device)
        predictions["eta downscaler"] = full_output_eta
        metrics_rows.append({"model": "eta downscaler", "test_mse": eta_test_mse, "tail_w1": eta_w1})
        print(f"eta downscaler: test_mse={eta_test_mse:.6f}, tail_w1={eta_w1:.6f}")

    if predictions:
        data_range = float(np.nanmax(ctx.era5.tp_trim_numpy) - np.nanmin(ctx.era5.tp_trim_numpy))
        spatial_df = precipitation_metrics_table(
            ctx.era5.tp_trim_numpy,
            predictions,
            lower_threshold_quantiles=(0.7, 0.8, 0.9),
            upper_threshold_quantiles=(0.95, 0.975, 0.99),
            data_range=data_range,
            mse_method="MSE downscaler",
            eta_method="eta downscaler",
        )
        print(spatial_df.to_string(index=False))
        save_table_csv(spatial_df, args.metrics_csv)

    save_npz(
        args.save_predictions,
        true_fields=ctx.era5.tp_trim_numpy,
        mse_predictions=predictions.get("MSE downscaler", np.empty((0,))),
        eta_predictions=predictions.get("eta downscaler", np.empty((0,))),
    )
    if metrics_rows:
        print(pd.DataFrame(metrics_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
