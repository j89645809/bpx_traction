#!/usr/bin/env python3
"""Train/export the direct BPX 45-D force estimator."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from force_estimator.bpx_model import BpxForceTCN


def load_split(path: Path):
    z = np.load(path)
    target = z["force_body"] if "force_body" in z else z["force_world"]
    return torch.from_numpy(z["sensor_history"]).float(), torch.from_numpy(target).float()


def loss_fn(pred: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    force = nn.functional.smooth_l1_loss(pred / 80.0, target / 80.0)
    active = target.norm(dim=1) > 2.0
    if active.any():
        direction = 1.0 - nn.functional.cosine_similarity(pred[active], target[active], dim=1).mean()
    else:
        direction = pred.sum() * 0.0
    err = (pred - target).norm(dim=1).detach()
    target_conf = torch.exp(-err / 15.0).clamp(0.0, 1.0).unsqueeze(1)
    calibration = nn.functional.mse_loss(confidence, target_conf)
    return force + 0.2 * direction + 0.2 * calibration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = Path(args.dataset); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_split(data / "train.npz")
    val_x, val_y = load_split(data / "validation.npz")
    mean = train_x.mean((0, 1)); std = train_x.std((0, 1)).clamp_min(1e-4)
    model = BpxForceTCN().cuda(); opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=False)
    best = float("inf"); history = []
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for x, y in loader:
            x = ((x - mean) / std).cuda(); y = y.cuda()
            pred, conf = model(x); loss = loss_fn(pred, y, conf)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            pred, conf = model(((val_x - mean) / std).cuda()); val_loss = loss_fn(pred, val_y.cuda(), conf).item()
        row = {"epoch": epoch + 1, "train_loss": total / len(train_x), "validation_loss": val_loss}
        history.append(row)
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "input_dim": 45, "history_length": 25, "seed": args.seed}, out / "bpx_force_best.pt")
    ckpt = torch.load(out / "bpx_force_best.pt", map_location="cpu", weights_only=False)
    model = BpxForceTCN().cpu().eval(); model.load_state_dict(ckpt["model"])
    class Normalized(nn.Module):
        def __init__(self, m, mu, sigma):
            super().__init__(); self.m = m; self.register_buffer("mean", mu); self.register_buffer("std", sigma)
        def forward(self, x): return self.m((x - self.mean) / self.std)
    wrapper = Normalized(model, ckpt["mean"], ckpt["std"]).eval(); example = torch.zeros(1, 25, 45)
    torch.onnx.export(wrapper, example, out / "bpx_force.onnx", input_names=["policy_history"], output_names=["force_body", "force_confidence"], dynamic_axes={"policy_history": {0: "batch"}, "force_world": {0: "batch"}, "force_confidence": {0: "batch"}}, opset_version=17, dynamo=False)
    torch.jit.trace(wrapper, example).save(str(out / "bpx_force.ts"))
    np.savez(out / "normalization.npz", mean=ckpt["mean"].numpy(), std=ckpt["std"].numpy())
    (out / "training_report.json").write_text(json.dumps({"best_validation_loss": best, "epochs": history, "input_shape": [25, 45], "target_frame": "body", "outputs": {"force_body": [3], "force_confidence": [1]}}, indent=2) + "\n")

if __name__ == "__main__": main()
