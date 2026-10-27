#!/usr/bin/env python3
"""Minimal traction-MVP preflight; does not start a simulator or hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort
import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--scenario-config", type=Path, required=True)
    ap.add_argument("--actor-checkpoint", type=Path, required=True)
    ap.add_argument("--estimator", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    scenario_path = resolve(args.scenario_config)
    actor_path = resolve(args.actor_checkpoint)
    estimator_path = resolve(args.estimator)
    for label, path in (("scenario", scenario_path), ("actor", actor_path), ("estimator", estimator_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    doc = yaml.safe_load(scenario_path.read_text())
    scenarios = doc.get("scenarios") if isinstance(doc, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario config must contain a non-empty scenarios list")
    for i, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario[{i}] must be a mapping")
        force = scenario.get("force_w_n")
        duration = scenario.get("duration_s")
        if not isinstance(force, list) or len(force) != 3:
            raise ValueError(f"scenario[{i}].force_w_n must have 3 values")
        if not all(isinstance(x, (int, float)) for x in force):
            raise ValueError(f"scenario[{i}].force_w_n contains non-numeric values")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"scenario[{i}].duration_s must be > 0")

    checkpoint = torch.load(actor_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("actor checkpoint is not a mapping")
    session = ort.InferenceSession(str(estimator_path), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) != 3 or input_shape[-1] != 82 or input_shape[-2] != 25:
        raise ValueError(f"estimator input must be [N,25,82], got {input_shape}")
    output_shape = session.get_outputs()[0].shape
    if len(output_shape) != 2 or output_shape[-1] not in (3, 19):
        raise ValueError(f"unexpected estimator output shape: {output_shape}")

    result = {
        "status": "READY",
        "scenario_count": len(scenarios),
        "actor_checkpoint": str(actor_path),
        "estimator": str(estimator_path),
        "estimator_input_shape": input_shape,
        "estimator_output_shape": output_shape,
        "actor_checkpoint_keys": sorted(checkpoint.keys()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
