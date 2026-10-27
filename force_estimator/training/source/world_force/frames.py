"""Frame conversions for the Gate 4.1 force adapter.

Quaternion order is always ``wxyz`` and represents body-to-world rotation.
"""

from __future__ import annotations

import numpy as np


def normalize_quaternion(q_wb: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wb, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("zero-norm quaternion")
    return q / norm


def quaternion_matrix(q_wb: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q_wb)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def body_to_world(force_b: np.ndarray, q_wb: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", quaternion_matrix(q_wb), np.asarray(force_b))


def world_to_body(force_w: np.ndarray, q_wb: np.ndarray) -> np.ndarray:
    return np.einsum("...ji,...j->...i", quaternion_matrix(q_wb), np.asarray(force_w))


def startup_frame_id(sequence: int, timestamp_ns: int) -> str:
    return f"W0-{timestamp_ns}-{sequence:04d}"
