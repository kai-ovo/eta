"""Tiny synthetic smoke test for MSE and quantile-W1 wiring.

This does not reproduce a paper experiment. It only verifies that the core
Python helpers import and that the supervised plus observable-quantile loss can
be evaluated on CPU without external data or checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for import_path in (REPO_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from metric import wloss


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(16, 2)
    y = torch.sin(x[:, 0]) + 0.25 * x[:, 1] ** 2

    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    mse = nn.MSELoss()

    quantiles = torch.tensor([0.5, 0.75, 0.9])
    reference_quantiles = torch.quantile(y, quantiles)

    for _ in range(2):
        optimizer.zero_grad()
        pred = model(x).squeeze()
        pred_quantiles = torch.quantile(pred, quantiles)
        supervised = mse(pred, y)
        observable_w1 = wloss(pred_quantiles, reference_quantiles, quantiles, p=1)
        loss = supervised + observable_w1
        loss.backward()
        optimizer.step()

    print(f"smoke_supervised_mse={float(supervised.detach()):.6f}")
    print(f"smoke_quantile_w1={float(observable_w1.detach()):.6f}")
    print("smoke_test_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
