#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from world_force.model import MissingStateTCN


class Normalized(nn.Module):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model((history - self.mean) / self.std)


def state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    velocity = nn.functional.smooth_l1_loss(pred[:, :3], target[:, :3])
    height = nn.functional.smooth_l1_loss(pred[:, 3:4], target[:, 3:4])
    return 4.0 * velocity + 16.0 * height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--channels", type=int, default=192)
    args = parser.parse_args()
    data = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = np.load(data / "train.npz")
    valid = np.load(data / "validation.npz")
    tx = torch.from_numpy(train["sensor_history"])
    ty = torch.from_numpy(train["privileged_targets"])
    vx = torch.from_numpy(valid["sensor_history"])
    vy = torch.from_numpy(valid["privileged_targets"])
    mean = tx.mean((0, 1))
    std = tx.std((0, 1)).clamp_min(1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MissingStateTCN(input_dim=tx.shape[-1], channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=512, shuffle=True)
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x = ((x - mean) / std).to(device)
            y = y.to(device)
            pred, _ = model(x)
            loss = state_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            pred, _ = model(((vx - mean) / std).to(device))
            validation = state_loss(pred, vy.to(device)).item()
        history.append({"epoch": epoch + 1, "train_loss": total / len(tx), "validation_loss": validation})
        if validation < best:
            best = validation
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "input_dim": tx.shape[-1], "channels": args.channels}, output / "state.pt")
    checkpoint = torch.load(output / "state.pt", map_location="cpu")
    model = MissingStateTCN(input_dim=checkpoint["input_dim"], channels=checkpoint["channels"]).eval()
    model.load_state_dict(checkpoint["model"])
    wrapper = Normalized(model, checkpoint["mean"], checkpoint["std"]).eval()
    example = torch.zeros(1, 25, checkpoint["input_dim"])
    torch.jit.trace(wrapper, example).save(str(output / "estimator.ts"))
    (output / "training_report.json").write_text(json.dumps({"best_validation_loss": best, "epochs": history}, indent=2) + "\n")


if __name__ == "__main__":
    main()
