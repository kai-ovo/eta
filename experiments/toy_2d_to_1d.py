"""Command-line version of toy--2D->1D.ipynb."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from experiments.common import (
    add_common_path_args,
    add_toy_args,
    build_fcnn,
    load_toy_scalar_data,
    pretrain_toy_scalar,
    print_metric_rows,
    save_npz,
    setup_cli,
    toy_scalar_quantiles,
)
from kde import get_data_pdf
from revision_utils import tail_w1_quantile
from test_utils import eval_result
from train_utils import train_eta_xy, train_nn
from utils import global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    add_toy_args(parser, default_eta_seeds=[25], include_lr=True)
    parser.add_argument("--lambda-sweep", action="store_true")
    parser.add_argument("--lambda-list", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
    return parser.parse_args()


def train_mse(data, device: torch.device, epochs: int, lr: float):
    global_seed(1)
    model = build_fcnn(outdim=1, init="xavier normal", positive_output=True)
    return train_nn(
        model,
        data.x_train,
        data.y_train,
        nn.MSELoss,
        torch.optim.Adam,
        epochs,
        data.x_train.shape[0],
        device,
        lr=lr,
    )


def train_eta(data, seed: int, pretrain_epochs: int, eta_epochs: int, lambd: float, omega: int, device: torch.device):
    pretrain_res = pretrain_toy_scalar(seed, pretrain_epochs, device)
    return train_eta_xy(
        pretrain_res.model,
        data.x_train,
        data.y_train,
        data.X,
        data.Y,
        toy_scalar_quantiles(),
        torch.optim.Adam,
        eta_epochs,
        data.x_train.shape[0],
        device,
        _lamb=lambd,
        omega=omega,
    )


def main() -> int:
    args = parse_args()
    ctx = setup_cli(args)
    data = load_toy_scalar_data(ctx.work_dir, toy_data=args.toy_data)

    outputs = {"truth_grid": data.Y_GRID.detach().cpu().numpy(), "truth_samples": data.Y.detach().cpu().numpy()}
    q = toy_scalar_quantiles()
    rows = []

    if not args.skip_mse:
        mse_res = train_mse(data, ctx.device, args.mse_epochs, args.lr)
        mse_eval = eval_result(mse_res, data.GRIDS, data.X, best_model=False, device=ctx.device)
        y_mse, py_mse, _ = get_data_pdf(mse_eval.y_samples, data.Y, None)
        rows.append(
            {
                "run": "mse",
                "seed": 1,
                "lambda": np.nan,
                "final_train_mse": float(mse_res.loss[-1]),
                "tail_w1": tail_w1_quantile(mse_eval.y_samples, data.Y, q),
                "pdf_support_points": int(len(y_mse)),
            }
        )
        outputs["mse_grid"] = mse_eval.y_grid.detach().cpu().numpy()
        outputs["mse_samples"] = mse_eval.y_samples.detach().cpu().numpy()
        outputs["mse_pdf_y"] = y_mse
        outputs["mse_pdf_py"] = py_mse

    if not args.skip_eta:
        eta_grids = []
        eta_samples = []
        for seed in args.eta_seeds:
            eta_res = train_eta(data, seed, args.pretrain_epochs, args.eta_epochs, args.lambd, args.omega, ctx.device)
            eta_eval = eval_result(eta_res, data.GRIDS, data.X, best_model=True, device=ctx.device)
            y_eta, py_eta, _ = get_data_pdf(eta_eval.y_samples, data.Y, None)
            rows.append(
                {
                    "run": "eta",
                    "seed": seed,
                    "lambda": args.lambd,
                    "final_train_mse": float(eta_res.mse[-1]),
                    "tail_w1": tail_w1_quantile(eta_eval.y_samples, data.Y, q),
                    "pdf_support_points": int(len(y_eta)),
                }
            )
            eta_grids.append(eta_eval.y_grid.detach().cpu().numpy())
            eta_samples.append(eta_eval.y_samples.detach().cpu().numpy())
        outputs["eta_grids"] = np.stack(eta_grids) if eta_grids else np.empty((0,))
        outputs["eta_samples"] = np.stack(eta_samples) if eta_samples else np.empty((0,))

    if args.lambda_sweep:
        sweep_tail_w1 = []
        for lambd in args.lambda_list:
            eta_res = train_eta(data, args.eta_seeds[0], args.pretrain_epochs, args.eta_epochs, lambd, args.omega, ctx.device)
            eta_eval = eval_result(eta_res, data.GRIDS, data.X, best_model=True, device=ctx.device)
            sweep_tail_w1.append(tail_w1_quantile(eta_eval.y_samples, data.Y, q))
            rows.append(
                {
                    "run": "lambda_sweep",
                    "seed": args.eta_seeds[0],
                    "lambda": lambd,
                    "final_train_mse": float(eta_res.mse[-1]),
                    "tail_w1": float(sweep_tail_w1[-1]),
                    "pdf_support_points": np.nan,
                }
            )
        outputs["lambda_list"] = np.asarray(args.lambda_list, dtype=float)
        outputs["lambda_tail_w1"] = np.asarray(sweep_tail_w1, dtype=float)

    print_metric_rows(rows, include_lambda=True)
    save_npz(args.save_npz, **outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
