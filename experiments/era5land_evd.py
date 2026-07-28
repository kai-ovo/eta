"""Command-line version of ERA5Land-EVD.ipynb."""

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
    alpha_tag,
    build_srcnn,
    empirical_quantiles,
    evaluate_precip_model,
    gevd_reference,
    load_required_mse_model,
    load_state_flexible,
    prepare_precip_context,
    precip_gevd_eta_path,
    precip_misspec_eta_path,
    save_npz,
    save_table_csv,
    train_precip_eta_cli,
)
from revision_utils import precip_spatial_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    add_precip_args(parser)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--num-w1-days", type=int, default=350)
    parser.add_argument("--omega", type=int, default=1)
    parser.add_argument("--lambda", dest="lambd", type=float, default=1.0)
    parser.add_argument("--eta-epochs", type=int, default=150)
    parser.add_argument("--train-eta", action="store_true")
    parser.add_argument("--misspec-alpha", type=float, nargs="*", default=[])
    parser.add_argument("--train-missing-misspec", action="store_true")
    parser.add_argument("--no-varying-days", action="store_true")
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ctx = prepare_precip_context(args)
    ref = gevd_reference(ctx.era5, tau=args.tau, num_w1_days=args.num_w1_days)
    w1_tail_thresh = ref["y_tau"]
    w1_truedays = ref["w1_truedays"]
    w1_truemax = torch.tensor(ref["Q_gevd"], dtype=torch.float32)

    print(
        "GEVD reference: "
        f"tau={args.tau}, y_tau={w1_tail_thresh:.6f}, "
        f"shape={ref['fit']['shape']:.6f}, loc={ref['fit']['location']:.6f}, scale={ref['fit']['scale']:.6f}, "
        f"cutoff={ref['fit']['cutoff']:.6f}"
    )

    model_mse = load_required_mse_model(args, ctx.work_dir, ctx.device)

    eta_path = precip_gevd_eta_path(ctx.work_dir, args.num_years, args.ds_fact, w1_tail_thresh, args.omega)
    model_eta = build_srcnn(args.ds_fact, args.data_parallel).to(ctx.device)
    if args.train_eta:
        print(f"Training GEVD eta downscaler and saving best W1 checkpoint to {eta_path}")
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
        print(f"Loading GEVD eta checkpoint from {eta_path}")
        model_eta = load_state_flexible(model_eta, eta_path, ctx.device)
    else:
        raise FileNotFoundError(f"Missing GEVD eta checkpoint: {eta_path}. Pass --train-eta to train it.")

    output_eta, eta_mse, eta_w1 = evaluate_precip_model(model_eta, ctx.tensors.test_loader, w1_truemax, ctx.device)
    data_range = float(np.nanmax(ctx.era5.tp_trim_numpy) - np.nanmin(ctx.era5.tp_trim_numpy))
    rows = []
    spatial = precip_spatial_metrics(output_eta, ctx.era5.tp_trim_numpy, data_range=data_range)
    rows.append({"model": "GEVD eta", "alpha": np.nan, "test_mse": eta_mse, "tail_w1": eta_w1, **spatial})
    print(f"GEVD eta: test_mse={eta_mse:.6f}, tail_w1={eta_w1:.6f}")

    q_np = ref["quantiles"].detach().cpu().numpy()
    Q_true = empirical_quantiles(ctx.era5.max_values, q_np)
    misspec_outputs = {}
    for alpha in args.misspec_alpha:
        Q_alpha = Q_true + alpha * (ref["Q_gevd"] - Q_true)
        model_path = precip_misspec_eta_path(ctx.work_dir, alpha, args.num_years, args.ds_fact, w1_tail_thresh, args.omega)
        model_alpha = build_srcnn(args.ds_fact, args.data_parallel).to(ctx.device)
        if not model_path.exists():
            if not args.train_missing_misspec:
                rows.append({"model": "misspecified eta", "alpha": alpha, "status": "missing_not_trained"})
                print(f"Missing alpha={alpha:g} checkpoint: {model_path}")
                continue
            print(f"Training misspecified-tail eta alpha={alpha:g} -> {model_path}")
            model_alpha = deepcopy(model_mse).to(ctx.device)
            model_alpha, _ = train_precip_eta_cli(
                model_alpha,
                ctx.tensors,
                ctx.era5,
                args.seed,
                args.eta_epochs,
                torch.tensor(Q_alpha, dtype=torch.float32),
                w1_truedays,
                args.lambd,
                args.omega,
                varying_days=not args.no_varying_days,
                lr=args.lr,
                save_path=model_path,
                device=ctx.device,
            )
        else:
            print(f"Loading misspecified-tail eta alpha={alpha:g}: {model_path.name}")
            model_alpha = load_state_flexible(model_alpha, model_path, ctx.device)

        output_alpha, alpha_mse, alpha_w1 = evaluate_precip_model(
            model_alpha, ctx.tensors.test_loader, torch.tensor(Q_alpha, dtype=torch.float32), ctx.device
        )
        misspec_outputs[f"alpha_{alpha_tag(alpha)}"] = output_alpha
        spatial = precip_spatial_metrics(output_alpha, ctx.era5.tp_trim_numpy, data_range=data_range)
        rows.append(
            {
                "model": "misspecified eta",
                "alpha": alpha,
                "alpha_tag": alpha_tag(alpha),
                "test_mse": alpha_mse,
                "tail_w1": alpha_w1,
                "status": "evaluated",
                **spatial,
            }
        )

    metrics_df = pd.DataFrame(rows)
    print(metrics_df.to_string(index=False))
    save_table_csv(metrics_df, args.metrics_csv)
    save_npz(args.save_predictions, true_fields=ctx.era5.tp_trim_numpy, gevd_eta_predictions=output_eta, **misspec_outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
