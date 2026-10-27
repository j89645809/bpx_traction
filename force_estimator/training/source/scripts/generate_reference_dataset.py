#!/usr/bin/env python3
"""Generate a deterministic sensor-like development dataset.

This is a physics-inspired pipeline/CI dataset, not an Oracle simulator result.
Its manifest makes that provenance explicit so it cannot be mistaken for Isaac
or MuJoCo truth. The same NPZ schema is used by the simulator collectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def make_split(rng: np.random.Generator, n: int, ood: bool = False):
    steps, dim = 25, 64
    t = np.arange(steps, dtype=np.float32) * 0.02
    x = np.zeros((n, steps, dim), np.float32)
    y = np.zeros((n, 15), np.float32)
    active = rng.random(n) > 0.20
    direction = rng.normal(size=(n, 3)).astype(np.float32)
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    magnitude = rng.uniform(5, 40 if not ood else 55, size=(n, 1)).astype(np.float32) * active[:, None]
    force = direction * magnitude
    yaw = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    q_wb = np.stack((np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)), axis=1).astype(np.float32)
    response = np.stack((0.012 * force[:, 0], 0.010 * force[:, 1], -0.002 * force[:, 2]), axis=1)
    height = (0.30 - 0.0008 * force[:, 2] + rng.normal(0, 0.006 if not ood else 0.012, n)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, (n, 1)).astype(np.float32)
    contact = np.zeros((n, 4), np.float32)
    contact[:, 0] = contact[:, 3] = (np.sin(phase[:, 0]) > -0.15)
    contact[:, 1] = contact[:, 2] = (np.sin(phase[:, 0]) < 0.15)
    loads = contact * (0.45 + 0.08 * force[:, 2:3] / 40.0)
    delay = rng.integers(0, 4 if ood else 3, size=n)
    for k in range(steps):
        ramp = np.clip((k - delay) / 8.0, 0, 1).astype(np.float32)[:, None]
        a = force * 0.025 * ramp + rng.normal(0, 0.08 if not ood else 0.16, (n, 3))
        gyro = np.stack((0.05 * np.sin(phase[:, 0] + t[k]), 0.04 * np.cos(phase[:, 0] + t[k]), 0.08 * response[:, 1]), 1)
        gravity = np.stack((0.03 * force[:, 0] / 40, 0.03 * force[:, 1] / 40, -np.ones(n)), 1)
        joint_phase = phase + t[k] * 2 * np.pi / 0.6 + np.arange(12)[None, :] * 0.23
        q = 0.35 * np.sin(joint_phase) + 0.002 * force[:, :1]
        dq = 0.35 * (2 * np.pi / 0.6) * np.cos(joint_phase)
        tau = 0.22 * force[:, :1] * np.tile(np.array([[1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]], np.float32), (n,1)) + rng.normal(0, 0.5, (n,12))
        action = 0.7 * q + rng.normal(0, 0.02, (n,12))
        cmd = np.stack((response[:, 0], response[:, 1], np.zeros(n)), 1)
        x[:, k] = np.concatenate((a, gyro, gravity, q_wb, q, dq, tau, action, cmd), axis=1)
    y[:, 0:3] = response
    y[:, 3] = height
    y[:, 4:8] = contact
    y[:, 8:12] = loads
    y[:, 12:15] = force
    # Independent drift, dropped samples and timestamp-like misalignment.
    x += rng.normal(0, 0.015 if not ood else 0.035, x.shape).astype(np.float32)
    drop = rng.random((n, steps)) < (0.01 if not ood else 0.04)
    for k in range(1, steps):
        x[drop[:, k], k] = x[drop[:, k], k - 1]
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=12000)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(410818)
    sizes = {"train": int(args.samples * .7), "validation": int(args.samples * .15), "test": args.samples - int(args.samples * .85), "ood": max(1000, args.samples // 6)}
    for name, size in sizes.items():
        x, y = make_split(rng, size, ood=name == "ood")
        np.savez_compressed(out / f"{name}.npz", sensor_history=x, privileged_targets=y)
    manifest = {
        "schema_version": "gate41-estimator-dataset-v1",
        "provenance": "physics_inspired_reference_only",
        "oracle_claim": False,
        "split_policy": "independent scenario seeds; no adjacent windows across splits",
        "sensor_history_shape": [25, 64], "privileged_target_dimension": 15,
        "leakage_check": "passed_by_construction", "sizes": sizes, "seed": 410818,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
