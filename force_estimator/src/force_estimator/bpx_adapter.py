"""Direct BPX adapter for the 45-D BPX force ONNX model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .contracts import BPX_INPUT_DIM, FORCE_VALUE_SLICE, HISTORY_LENGTH, POLICY_FORCE_SCALE_N
from .preprocessing import load_normalization


def _rotation_matrix_wxyz(q_wb: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wb, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError("q_wb must have final dimension 4 in wxyz order")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("q_wb contains a zero-norm quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z + y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def world_to_body(force_world: np.ndarray, q_wb: np.ndarray) -> np.ndarray:
    """Convert body-to-world quaternion ``q_wb`` and world force to body force."""
    return np.einsum("...ji,...j->...i", _rotation_matrix_wxyz(q_wb), force_world)


class BpxForceEstimator:
    """Run the 64-D estimator and return BPX-ready force/confidence tensors.

    The model must expose one input ``[B, 25, 45]`` and two outputs:
    ``values [B, 15]`` and ``confidence [B, 4]``. The last three values are
    world-frame force in newtons and confidence column 3 is force confidence.
    """

    def __init__(self, model_path: str | Path, normalization_path: str | Path, session: Any = None):
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - optional deployment dependency
                raise RuntimeError("install onnxruntime to use BpxForceEstimator") from exc
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.session = session
        self.mean, self.std = load_normalization(normalization_path, expected_dim=BPX_INPUT_DIM)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 2:
            raise ValueError("BPX estimator must expose one input and values/confidence outputs")
        self.input_name = inputs[0].name

    def predict(
        self,
        history: np.ndarray,
        q_wb: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        history = np.asarray(history, dtype=np.float32)
        q_wb = np.asarray(q_wb, dtype=np.float32)
        if history.ndim != 3 or history.shape[1:] != (HISTORY_LENGTH, BPX_INPUT_DIM):
            raise ValueError(f"expected history [B, {HISTORY_LENGTH}, {BPX_INPUT_DIM}]")
        if q_wb.shape != (history.shape[0], 4):
            raise ValueError("q_wb must have shape [B, 4] in wxyz order")
        if valid_mask is None:
            valid = np.ones(history.shape[0], dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != (history.shape[0],):
                raise ValueError("valid_mask must have shape [B]")
        finite_input = np.isfinite(history).all(axis=(1, 2)) & np.isfinite(q_wb).all(axis=1)
        model_input = (history - self.mean) / self.std
        values, confidence = self.session.run(None, {self.input_name: model_input.astype(np.float32)})
        values = np.asarray(values, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        if values.shape == (history.shape[0], 3) and confidence.shape == (history.shape[0], 1):
            force_world, force_confidence = values, confidence
        elif values.shape == (history.shape[0], 15) and confidence.shape == (history.shape[0], 4):
            force_world, force_confidence = values[:, FORCE_VALUE_SLICE], confidence[:, 3:4]
        else:
            raise ValueError(f"unexpected model outputs: values={values.shape}, confidence={confidence.shape}")
        force_body = world_to_body(force_world, q_wb)
        force_policy = np.clip(force_body / POLICY_FORCE_SCALE_N, -1.0, 1.0)
        force_confidence = np.clip(force_confidence, 0.0, 1.0)
        valid &= finite_input & np.isfinite(values).all(axis=1) & np.isfinite(confidence).all(axis=1)
        force_policy[~valid] = 0.0
        force_confidence[~valid] = 0.0
        return force_policy.astype(np.float32), force_confidence.astype(np.float32)
