"""Command-line version of toy--2D->2D.ipynb."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from experiments.common import (
    add_common_path_args,
    add_toy_args,
    build_fcnn,
    load_toy_state_data,
    pretrain_toy_state,
    print_metric_rows,
    save_npz,
    setup_cli,
    state_observable,
    toy_state_quantiles,
)
from kde import get_data_pdf
from revision_utils import tail_w1_quantile
from test_utils import eval_result_2d
from train_utils import train_eta_xuy, train_nn
from utils import global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    add_toy_args(parser, default_eta_seeds=[28])
    return parser.parse_args()


def train_mse(data, device: torch.device, epochs: int):
    global_seed(1)
    model = build_fcnn(outdim=2, init="xavier normal", positive_output=False)
    return train_nn(
        model,
        data.x_train,
        data.u_train,
        nn.MSELoss,
        torch.optim.Adam,
        epochs,
        data.x_train.shape[0],
        device,
    )


def train_eta(data, seed: int, pretrain_epochs: int, eta_epochs: int, lambd: float, omega: int, device: torch.device):
    pretrain_res = pretrain_toy_state(data, seed, pretrain_epochs, device)
    return train_eta_xuy(
        pretrain_res.model,
        data.x_train,
        data.u_train,
        data.X,
        data.Y,
        state_observable,
        toy_state_quantiles(),
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
    data = load_toy_state_data(ctx.work_dir, toy_data=args.toy_data)

    outputs = {
        "truth_u1_grid": data.U1_GRID.detach().cpu().numpy(),
        "truth_u2_grid": data.U2_GRID.detach().cpu().numpy(),
        "truth_observable_samples": data.Y.detach().cpu().numpy(),
    }
    q = toy_state_quantiles()
    rows = []

    if not args.skip_mse:
        mse_res = train_mse(data, ctx.device, args.mse_epochs)
        mse_eval = eval_result_2d(mse_res, data.GRIDS, data.X, state_observable, best_model=False, device=ctx.device)
        y_mse, py_mse, _ = get_data_pdf(mse_eval.y_samples, data.Y, None)
        rows.append(
            {
                "run": "mse",
                "seed": 1,
                "final_train_mse": float(mse_res.loss[-1]),
                "tail_w1": tail_w1_quantile(mse_eval.y_samples, data.Y, q),
                "pdf_support_points": int(len(y_mse)),
            }
        )
        outputs["mse_u1_grid"] = mse_eval.u1_grid
        outputs["mse_u2_grid"] = mse_eval.u2_grid
        outputs["mse_observable_samples"] = mse_eval.y_samples.detach().cpu().numpy()
        outputs["mse_pdf_y"] = y_mse
        outputs["mse_pdf_py"] = py_mse

    if not args.skip_eta:
        eta_u1 = []
        eta_u2 = []
        eta_observable = []
        for seed in args.eta_seeds:
            eta_res = train_eta(data, seed, args.pretrain_epochs, args.eta_epochs, args.lambd, args.omega, ctx.device)
            eta_eval = eval_result_2d(eta_res, data.GRIDS, data.X, state_observable, best_model=True, device=ctx.device)
            rows.append(
                {
                    "run": "eta",
                    "seed": seed,
                    "final_train_mse": float(eta_res.mse[-1]),
                    "tail_w1": tail_w1_quantile(eta_eval.y_samples, data.Y, q),
                    "pdf_support_points": np.nan,
                }
            )
            eta_u1.append(eta_eval.u1_grid)
            eta_u2.append(eta_eval.u2_grid)
            eta_observable.append(eta_eval.y_samples.detach().cpu().numpy())
        outputs["eta_u1_grids"] = np.stack(eta_u1) if eta_u1 else np.empty((0,))
        outputs["eta_u2_grids"] = np.stack(eta_u2) if eta_u2 else np.empty((0,))
        outputs["eta_observable_samples"] = np.stack(eta_observable) if eta_observable else np.empty((0,))

    print_metric_rows(rows)
    save_npz(args.save_npz, **outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
