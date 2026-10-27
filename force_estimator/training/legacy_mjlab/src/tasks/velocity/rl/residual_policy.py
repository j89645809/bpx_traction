from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
import os
from collections.abc import Iterator
from functools import lru_cache

import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories


@dataclass(frozen=True)
class ResidualGateConfig:
    force_start_n: float = 45.0
    force_full_n: float = 50.0
    recovery_seconds: float = 1.5
    settled_speed_mps: float = 0.05
    settled_seconds: float = 0.2
    step_dt: float = 0.02

    def __post_init__(self) -> None:
        if not self.force_start_n < self.force_full_n:
            raise ValueError("force_start_n must be smaller than force_full_n")
        if self.recovery_seconds <= 0.0 or self.settled_seconds <= 0.0:
            raise ValueError("recovery and settled durations must be positive")
        if self.step_dt <= 0.0:
            raise ValueError("step_dt must be positive")


@dataclass
class ResidualGateState:
    value: torch.Tensor
    settled_steps: torch.Tensor


PHASE_INITIAL = 0
PHASE_PULL = 1
PHASE_RELEASE_ARMED = 2
PHASE_REVERSAL = 3
PHASE_RECOVERY = 4

# High-force emergency action stages.  These values are control metadata, not
# observations: the learned emergency head still receives only the allowed
# pose/angular-rate/contact/direction history.
EMERGENCY_STAGE_PULL_STABILIZE = 0
EMERGENCY_STAGE_RELEASE_BRAKE = 1
EMERGENCY_STAGE_REVERSE_CONFIRM = 2
EMERGENCY_STAGE_RECOVERY_SETTLE = 3


def emergency_stage_scales(
    stage: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scaffold scale, learned residual scale) per environment.

    Pull is deliberately scaffold-only; learned authority is unlocked only
    after release/reversal confirmation.  Unknown stages fail closed to the
    pull setting.
    """
    stage = stage.to(device=device, dtype=torch.long).reshape(-1)
    # The scaffold is deliberately stronger than the learned branch during
    # the first high-force frame: a 0.18 multiplier made the lateral brace
    # only a few percent of an action and could not arrest a 60 N upset.
    scaffold = torch.full_like(stage, 0.60, dtype=dtype)
    learned = torch.zeros_like(stage, dtype=dtype)
    # Pull is normally scaffold-only.  A high-force emergency controller can
    # explicitly unlock a small, bounded learned authority during the first
    # >50 N frames.  This is kept behind an environment switch so ordinary
    # inference remains fail-closed and the head is still zero-initialized.
    pull_learned_scale = float(
        os.environ.get("FINETUNE_EMERGENCY_PULL_LEARNED_SCALE", "0.0")
    )
    if not 0.0 <= pull_learned_scale <= 1.0:
        raise ValueError("FINETUNE_EMERGENCY_PULL_LEARNED_SCALE must be in [0, 1]")
    learned = torch.where(
        stage == EMERGENCY_STAGE_PULL_STABILIZE,
        torch.full_like(learned, pull_learned_scale),
        learned,
    )
    scaffold = torch.where(stage == EMERGENCY_STAGE_RELEASE_BRAKE,
                           torch.full_like(scaffold, 0.75), scaffold)
    scaffold = torch.where(stage == EMERGENCY_STAGE_REVERSE_CONFIRM,
                           torch.full_like(scaffold, 0.90), scaffold)
    scaffold = torch.where(stage == EMERGENCY_STAGE_RECOVERY_SETTLE,
                           torch.full_like(scaffold, 0.72), scaffold)
    reverse_learned_scale = float(
        os.environ.get("FINETUNE_EMERGENCY_REVERSE_LEARNED_SCALE", "0.35")
    )
    if not 0.0 <= reverse_learned_scale <= 1.0:
        raise ValueError("FINETUNE_EMERGENCY_REVERSE_LEARNED_SCALE must be in [0, 1]")
    learned = torch.where(
        stage == EMERGENCY_STAGE_REVERSE_CONFIRM,
        torch.full_like(learned, reverse_learned_scale),
        learned,
    )
    learned = torch.where(stage == EMERGENCY_STAGE_RECOVERY_SETTLE,
                          torch.ones_like(learned), learned)
    return scaffold, learned


def emergency_scaffold_action(
    features: torch.Tensor,
    stage: torch.Tensor,
    *,
    limit: float,
) -> torch.Tensor:
    """Build a bounded, symmetric posture/brace action by stage.

    ``features`` is the latest 12-frame slice (roll/pitch, angular rates,
    contacts, finite horizontal direction).  No force magnitude or command is
    consumed.  The scaffold is deterministic and is always added as a small
    bounded correction; the learned head cannot replace it.
    """
    if features.ndim != 2 or features.shape[-1] not in (12, 60):
        raise ValueError("features must have shape (N, 12) or (N, 60)")
    latest = features[:, -12:]
    if stage.shape[0] != latest.shape[0]:
        raise ValueError("stage must contain one value per environment")
    scaffold_scale, _ = emergency_stage_scales(
        stage, dtype=latest.dtype, device=latest.device
    )
    reverse_boost = float(os.environ.get("FINETUNE_EMERGENCY_REVERSE_BOOST", "1.0"))
    if reverse_boost <= 0.0:
        raise ValueError("FINETUNE_EMERGENCY_REVERSE_BOOST must be positive")
    scaffold_scale = torch.where(
        stage == EMERGENCY_STAGE_REVERSE_CONFIRM,
        scaffold_scale * reverse_boost,
        scaffold_scale,
    )
    # Keep the structure explicitly stage separated: release/reverse use a
    # small transition brake, while confirmed recovery gets an independent
    # angular-rate damping term and a symmetric support template.  This keeps
    # the emergency controller from turning a direction cue into a raw
    # velocity-following command.
    recovery = stage == EMERGENCY_STAGE_RECOVERY_SETTLE
    reverse = stage == EMERGENCY_STAGE_REVERSE_CONFIRM
    orient_recovery = float(os.environ.get("FINETUNE_EMERGENCY_RECOVERY_ORIENT_GAIN", "0.22"))
    rate_recovery = float(os.environ.get("FINETUNE_EMERGENCY_RECOVERY_RATE_GAIN", "0.34"))
    orient_gain = torch.where(recovery, torch.full_like(stage, orient_recovery, dtype=latest.dtype),
                              torch.full_like(stage, 0.30, dtype=latest.dtype))
    rate_gain = torch.where(recovery, torch.full_like(stage, rate_recovery, dtype=latest.dtype),
                            torch.where(reverse, torch.full_like(stage, 0.16, dtype=latest.dtype),
                                        torch.full_like(stage, 0.10, dtype=latest.dtype)))
    gain_override = os.environ.get("FINETUNE_EMERGENCY_PD_GAIN_SCALE")
    if gain_override is not None:
        gain_scale = float(gain_override)
        if gain_scale < 0.0:
            raise ValueError("FINETUNE_EMERGENCY_PD_GAIN_SCALE must be non-negative")
        orient_gain = orient_gain * gain_scale
        rate_gain = rate_gain * gain_scale
    roll = -(orient_gain * latest[:, 1] + rate_gain * latest[:, 4])
    pitch = -(orient_gain * latest[:, 0] + rate_gain * latest[:, 3])
    # Immediate dynamics-risk brace.  This is deliberately separate from the
    # learned emergency heads: it activates in any hard-force stage when
    # angular-rate or support loss indicates that the base policy is already
    # leaving its recoverable envelope.  It consumes only pose/rate/contact
    # features, never force magnitude or velocity commands.
    risk_rate_gain = float(os.environ.get("FINETUNE_EMERGENCY_RISK_RATE_GAIN", "0.0"))
    risk_orient_gain = float(os.environ.get("FINETUNE_EMERGENCY_RISK_ORIENT_GAIN", "0.0"))
    risk_lateral_gain = float(os.environ.get("FINETUNE_EMERGENCY_RISK_LATERAL_GAIN", "0.0"))
    risk_contact_gain = float(os.environ.get("FINETUNE_EMERGENCY_RISK_CONTACT_GAIN", "0.0"))
    risk_sign = float(os.environ.get("FINETUNE_EMERGENCY_RISK_SIGN", "1.0"))
    if risk_sign not in (-1.0, 1.0):
        raise ValueError("FINETUNE_EMERGENCY_RISK_SIGN must be -1 or 1")
    if min(risk_rate_gain, risk_orient_gain, risk_lateral_gain, risk_contact_gain) < 0.0:
        raise ValueError("risk brace gains must be non-negative")
    if max(risk_rate_gain, risk_orient_gain, risk_lateral_gain, risk_contact_gain) > 0.0:
        risk_rate_threshold = float(os.environ.get("FINETUNE_EMERGENCY_RISK_RATE_THRESHOLD", "1.5"))
        risk_rate_width = max(float(os.environ.get("FINETUNE_EMERGENCY_RISK_RATE_WIDTH", "1.5")), 1.0e-6)
        rate_norm = torch.linalg.vector_norm(latest[:, 3:6], dim=1)
        rate_risk = ((rate_norm - risk_rate_threshold) / risk_rate_width).clamp(0.0, 1.0)
        contact_loss = (latest[:, 6:10] < 0.5).to(latest.dtype).mean(dim=1)
        risk = torch.maximum(rate_risk, contact_loss)
        risk = risk * (stage != EMERGENCY_STAGE_RELEASE_BRAKE).to(latest.dtype)
        roll = roll - risk_sign * risk * (risk_orient_gain * latest[:, 1] + risk_rate_gain * latest[:, 4])
        pitch = pitch - risk_sign * risk * (risk_orient_gain * latest[:, 0] + risk_rate_gain * latest[:, 3])
        lateral = risk_sign * risk * (risk_lateral_gain * latest[:, 11])
    else:
        risk = torch.zeros_like(latest[:, 0])
        lateral = torch.zeros_like(latest[:, 0])
    scaffold_sign = float(os.environ.get("FINETUNE_EMERGENCY_SCAFFOLD_SIGN", "1.0"))
    if scaffold_sign not in (-1.0, 1.0):
        raise ValueError("FINETUNE_EMERGENCY_SCAFFOLD_SIGN must be -1 or 1")
    roll = scaffold_sign * roll
    pitch = scaffold_sign * pitch
    # Direction-aware hip brace is the structural correction for the pure
    # lateral (±Y) failures.  It is bounded, symmetric, and stage-scaled.
    lateral_gain = float(os.environ.get("FINETUNE_EMERGENCY_LATERAL_GAIN", "0.25"))
    if lateral_gain < 0.0:
        raise ValueError("FINETUNE_EMERGENCY_LATERAL_GAIN must be non-negative")
    lateral_sign = float(os.environ.get("FINETUNE_EMERGENCY_LATERAL_SIGN", "-1.0"))
    if lateral_sign not in (-1.0, 1.0):
        raise ValueError("FINETUNE_EMERGENCY_LATERAL_SIGN must be -1 or 1")
    reverse_lateral_sign = float(
        os.environ.get("FINETUNE_EMERGENCY_REVERSE_LATERAL_SIGN", str(lateral_sign))
    )
    if reverse_lateral_sign not in (-1.0, 1.0):
        raise ValueError("FINETUNE_EMERGENCY_REVERSE_LATERAL_SIGN must be -1 or 1")
    # Allow the two lateral directions to use independent brace signs/gains.
    # Defaults reproduce the legacy global-sign behavior exactly.
    y_pos_sign = float(os.environ.get("FINETUNE_EMERGENCY_Y_POS_LATERAL_SIGN", "1.0"))
    y_neg_sign = float(os.environ.get("FINETUNE_EMERGENCY_Y_NEG_LATERAL_SIGN", "1.0"))
    if y_pos_sign not in (-1.0, 1.0) or y_neg_sign not in (-1.0, 1.0):
        raise ValueError("direction-specific lateral signs must be -1 or 1")
    y_pos_gain = float(os.environ.get("FINETUNE_EMERGENCY_Y_POS_LATERAL_GAIN", str(lateral_gain)))
    y_neg_gain = float(os.environ.get("FINETUNE_EMERGENCY_Y_NEG_LATERAL_GAIN", str(lateral_gain)))
    if y_pos_gain < 0.0 or y_neg_gain < 0.0:
        raise ValueError("direction-specific lateral gains must be non-negative")
    sign = torch.where(
        (stage == EMERGENCY_STAGE_REVERSE_CONFIRM)
        | (stage == EMERGENCY_STAGE_RECOVERY_SETTLE),
        torch.full_like(latest[:, 11], reverse_lateral_sign),
        torch.full_like(latest[:, 11], lateral_sign),
    )
    y_dir = latest[:, 11]
    directional_sign = torch.where(
        y_dir >= 0.0,
        torch.full_like(y_dir, y_pos_sign),
        torch.full_like(y_dir, y_neg_sign),
    )
    directional_gain = torch.where(
        y_dir >= 0.0,
        torch.full_like(y_dir, y_pos_gain),
        torch.full_like(y_dir, y_neg_gain),
    )
    pull_gain = float(os.environ.get("FINETUNE_EMERGENCY_PULL_LATERAL_GAIN", str(lateral_gain)))
    reverse_gain = float(os.environ.get("FINETUNE_EMERGENCY_REVERSE_LATERAL_GAIN", str(lateral_gain)))
    recovery_gain = float(os.environ.get("FINETUNE_EMERGENCY_RECOVERY_LATERAL_GAIN", str(reverse_gain)))
    if min(pull_gain, reverse_gain, recovery_gain) < 0.0:
        raise ValueError("stage-specific lateral gains must be non-negative")
    stage_gain = torch.where(
        stage == EMERGENCY_STAGE_RECOVERY_SETTLE,
        torch.full_like(y_dir, recovery_gain),
        torch.where(
            stage == EMERGENCY_STAGE_REVERSE_CONFIRM,
            torch.full_like(y_dir, reverse_gain),
            torch.full_like(y_dir, pull_gain),
        ),
    )
    lateral = lateral + sign * directional_sign * directional_gain * (stage_gain / max(lateral_gain, 1.0e-6)) * y_dir
    contact_gain = float(os.environ.get("FINETUNE_EMERGENCY_CONTACT_GAIN", "0.0"))
    if contact_gain < 0.0:
        raise ValueError("FINETUNE_EMERGENCY_CONTACT_GAIN must be non-negative")
    # The emergency feature layout carries four bounded foot-contact values
    # at indices 6:10.  A small deficit-compensation term keeps a lightly
    # loaded leg from being driven deeper into the body during a lateral
    # upset; it is independent of force magnitude and command velocity.
    contact_deficit = (0.5 - latest[:, 6:10]).clamp(-0.5, 0.5)
    out = torch.zeros((latest.shape[0], 12), device=latest.device, dtype=latest.dtype)
    out[:, [0, 6]] = roll.unsqueeze(1)
    out[:, [3, 9]] = -roll.unsqueeze(1)
    out[:, [1, 2, 4, 5]] = (pitch + lateral).unsqueeze(1)
    out[:, [7, 8, 10, 11]] = -(pitch + lateral).unsqueeze(1)
    pattern = os.environ.get("FINETUNE_EMERGENCY_LATERAL_PATTERN", "mirrored")
    if pattern == "mirrored":
        out[:, [0, 6]] += lateral.unsqueeze(1)
        out[:, [3, 9]] -= lateral.unsqueeze(1)
    elif pattern == "uniform":
        out[:, [0, 3, 6, 9]] += lateral.unsqueeze(1)
    elif pattern == "front_rear":
        out[:, [0, 3]] += lateral.unsqueeze(1)
        out[:, [6, 9]] -= lateral.unsqueeze(1)
    elif pattern == "diagonal":
        out[:, [0, 9]] += lateral.unsqueeze(1)
        out[:, [3, 6]] -= lateral.unsqueeze(1)
    elif pattern == "side_support":
        # Transfer support laterally through the thigh/calf chains rather
        # than relying on hip abduction alone.  This is the emergency
        # recovery template for the ±Y and diagonal cases.
        out[:, [1, 2, 4, 5]] = pitch.unsqueeze(1)
        out[:, [7, 8, 10, 11]] = -pitch.unsqueeze(1)
        out[:, [1, 2, 7, 8]] += lateral.unsqueeze(1)
        out[:, [4, 5, 10, 11]] -= lateral.unsqueeze(1)
    else:
        raise ValueError(
            "FINETUNE_EMERGENCY_LATERAL_PATTERN must be mirrored, uniform, front_rear, diagonal, or side_support"
        )
    # Recovery support is deliberately symmetric front/rear; the additional
    # damping is supplied by the rate term above rather than by a learned
    # following command.
    if contact_gain > 0.0:
        out[:, [1, 2]] += contact_gain * contact_deficit[:, 0:1]
        out[:, [4, 5]] += contact_gain * contact_deficit[:, 1:2]
        out[:, [7, 8]] += contact_gain * contact_deficit[:, 2:3]
        out[:, [10, 11]] += contact_gain * contact_deficit[:, 3:4]
    if risk_contact_gain > 0.0:
        risk_col = risk.unsqueeze(1)
        out[:, [1, 2]] += risk_col * risk_contact_gain * contact_deficit[:, 0:1]
        out[:, [4, 5]] += risk_col * risk_contact_gain * contact_deficit[:, 1:2]
        out[:, [7, 8]] += risk_col * risk_contact_gain * contact_deficit[:, 2:3]
        out[:, [10, 11]] += risk_col * risk_contact_gain * contact_deficit[:, 3:4]
    return (scaffold_scale.unsqueeze(1) * out).clamp(-limit, limit)


def emergency_dynamics_brake_action(
    features: torch.Tensor,
    stage: torch.Tensor,
    *,
    limit: float,
) -> torch.Tensor:
    """Contact-agnostic local angular-rate brake from identified ±Y models.

    The matrices are ridge-identified from the 60 N action traces.  They are
    deliberately used only as a small bounded feedback term during confirmed
    reverse/recovery; pull remains base-equivalent.  The learned emergency
    heads can still refine the response around this stabilizing prior.
    """
    gain = float(os.environ.get("FINETUNE_EMERGENCY_DYNAMICS_BRAKE_GAIN", "0.0"))
    if gain < 0.0:
        raise ValueError("FINETUNE_EMERGENCY_DYNAMICS_BRAKE_GAIN must be non-negative")
    out = torch.zeros((features.shape[0], 12), device=features.device, dtype=features.dtype)
    if gain == 0.0:
        return out
    latest = features[:, -12:]
    # u = -K omega, with K obtained from the measured local action-to-rate
    # Jacobians.  Rows are the 12 joint actions; columns are wx, wy, wz.
    k_pos = torch.tensor([
        [-0.014,0.011,-0.006],[-0.016,-0.004,0.008],[-0.002,-0.008,-0.022],
        [-0.025,-0.002,-0.037],[-0.006,-0.010,0.087],[-0.009,-0.016,0.039],
        [-0.009,0.000,0.051],[0.007,-0.007,-0.034],[0.009,0.008,0.012],
        [-0.020,0.004,0.100],[-0.014,-0.013,-0.003],[-0.005,0.020,-0.038],
    ], device=features.device, dtype=features.dtype)
    k_neg = torch.tensor([
        [-0.011,0.005,-0.047],[-0.027,0.008,0.059],[-0.008,-0.007,-0.002],
        [-0.021,-0.008,-0.070],[0.010,-0.003,0.031],[-0.012,-0.008,0.024],
        [-0.008,0.006,0.058],[0.012,-0.010,-0.041],[0.005,0.014,0.032],
        [-0.029,0.013,0.105],[-0.025,-0.007,0.017],[-0.005,0.015,-0.021],
    ], device=features.device, dtype=features.dtype)
    direction_y = latest[:, 11]
    K = torch.where(direction_y[:, None, None] >= 0.0, k_pos[None], k_neg[None])
    json_path = os.environ.get("FINETUNE_EMERGENCY_CONTACT_K_JSON", "")
    if json_path:
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                table = json.load(handle)
            masks = (latest[:, 6:10] >= 0.5).to(torch.int64)
            rows = []
            for i in range(features.shape[0]):
                direction = "y_pos" if float(direction_y[i]) >= 0.0 else "y_neg"
                key = "".join(str(int(v)) for v in masks[i].tolist())
                payload = table.get(direction, {}).get(key)
                if payload is None:
                    payload = table.get(direction, {}).get("1111")
                rows.append(torch.tensor(payload["K"] if payload else (k_pos if direction == "y_pos" else k_neg).tolist(), device=features.device, dtype=features.dtype))
            K = torch.stack(rows, dim=0)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # Diagnostic prior is fail-open to the embedded pooled matrices;
            # it is never required for loading or deployment.
            pass
    # The identified matrices use u=-K*omega.
    raw = -torch.bmm(K, latest[:, 3:6].unsqueeze(-1)).squeeze(-1)
    contact_gain = float(os.environ.get("FINETUNE_EMERGENCY_DYNAMICS_CONTACT_WEIGHT", "0.0"))
    if contact_gain < 0.0:
        raise ValueError("FINETUNE_EMERGENCY_DYNAMICS_CONTACT_WEIGHT must be non-negative")
    if contact_gain > 0.0:
        contact = latest[:, 6:10].clamp(0.0, 1.0)
        # Each leg contributes three consecutive action channels.  Loaded
        # legs get a larger share of the stabilizing action; a lost-contact
        # leg is never used as the primary brace.
        leg_weight = 1.0 + contact_gain * (contact - 0.5)
        raw = raw * leg_weight.repeat_interleave(3, dim=1)
    active = ((stage == EMERGENCY_STAGE_REVERSE_CONFIRM)
              | (stage == EMERGENCY_STAGE_RECOVERY_SETTLE))
    if os.environ.get("FINETUNE_EMERGENCY_DYNAMICS_BRAKE_PULL", "0") == "1":
        # Above the hard-force threshold the brake may also act during the
        # initial pull brace, but only as a rate-damping residual; the normal
        # <=50 N pull path remains untouched by this opt-in.
        active = active | (stage == EMERGENCY_STAGE_PULL_STABILIZE)
    return (gain * active.to(features.dtype)[:, None] * raw).clamp(-limit, limit)


def emergency_contact_transition_action(
    features: torch.Tensor,
    stage: torch.Tensor,
    *,
    limit: float,
) -> torch.Tensor:
    """One-shot support reallocation when a stance leg is lost."""
    gain = float(os.environ.get("FINETUNE_EMERGENCY_CONTACT_TRANSITION_GAIN", "0.0"))
    if gain < 0.0:
        raise ValueError("FINETUNE_EMERGENCY_CONTACT_TRANSITION_GAIN must be non-negative")
    out = torch.zeros((features.shape[0], 12), device=features.device, dtype=features.dtype)
    if gain == 0.0 or features.shape[-1] < 24:
        return out
    hist = features.reshape(features.shape[0], -1, 12)
    prev = hist[:, -2, 6:10]
    curr = hist[:, -1, 6:10]
    lost = ((prev >= 0.5) & (curr < 0.5)).to(features.dtype)
    lost_any = (lost.sum(dim=1) > 0.0).to(features.dtype)
    if os.environ.get("FINETUNE_EMERGENCY_CONTACT_TRANSITION_PERSIST", "0") == "1":
        # Keep the bounded brace active while a stance leg remains absent;
        # a one-step pulse can end before the body has reloaded a support leg.
        lost_any = (curr.sum(dim=1) < 3.5).to(features.dtype)
    active = ((stage == EMERGENCY_STAGE_REVERSE_CONFIRM)
              | (stage == EMERGENCY_STAGE_RECOVERY_SETTLE))
    if os.environ.get("FINETUNE_EMERGENCY_CONTACT_TRANSITION_PULL", "0") == "1":
        # A lost stance during a >50 N pull is already an emergency event;
        # permit the same bounded support reallocation used after reversal.
        # This opt-in never opens the ordinary force-following residual gate.
        active = active | (stage == EMERGENCY_STAGE_PULL_STABILIZE)
    direction = torch.where(hist[:, -1, 11] >= 0.0,
                            torch.ones_like(lost[:, 0]),
                            -torch.ones_like(lost[:, 0]))
    transition_sign = float(os.environ.get("FINETUNE_EMERGENCY_CONTACT_TRANSITION_SIGN", "1.0"))
    if transition_sign not in (-1.0, 1.0):
        raise ValueError("FINETUNE_EMERGENCY_CONTACT_TRANSITION_SIGN must be -1 or 1")
    direction = transition_sign * direction
    # Allocate the transient brace to legs that remain loaded, centered so
    # that no net following command is introduced.
    support = curr * (1.0 - lost)
    support = support / support.sum(dim=1, keepdim=True).clamp_min(1.0)
    support = support - support.mean(dim=1, keepdim=True)
    strength = gain * lost_any * active.to(features.dtype) * direction
    for leg in range(4):
        q = strength * support[:, leg]
        out[:, 3 * leg] = q
        out[:, 3 * leg + 1] = -0.5 * q
        out[:, 3 * leg + 2] = -0.5 * q
    return out.clamp(-limit, limit)


@lru_cache(maxsize=4)
def _load_state_feedback_table(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def emergency_state_feedback_action(
    features: torch.Tensor,
    stage: torch.Tensor,
    *,
    limit: float,
) -> torch.Tensor:
    """Apply a fitted contact/direction-conditioned high-force residual."""
    gain = float(os.environ.get("FINETUNE_EMERGENCY_STATE_FB_GAIN", "0.0"))
    if gain < 0.0:
        raise ValueError("FINETUNE_EMERGENCY_STATE_FB_GAIN must be non-negative")
    out = torch.zeros((features.shape[0], 12), device=features.device, dtype=features.dtype)
    if gain == 0.0:
        return out
    path = os.environ.get("FINETUNE_EMERGENCY_STATE_FB_JSON", "")
    if not path:
        return out
    table = _load_state_feedback_table(path).get("groups", {})
    latest = features[:, -12:]
    direction = latest[:, 10:12]
    dx, dy = direction[:, 0], direction[:, 1]
    diagonal = (dx.abs() >= 0.35) & (dy.abs() >= 0.35)
    x_dominant = dx.abs() >= dy.abs()
    names = []
    for i in range(features.shape[0]):
        if bool(diagonal[i]):
            names.append("diag_" + ("p" if float(dx[i]) >= 0 else "n")
                         + ("p" if float(dy[i]) >= 0 else "n"))
        elif bool(x_dominant[i]):
            names.append("x_pos" if float(dx[i]) >= 0 else "x_neg")
        else:
            names.append("y_pos" if float(dy[i]) >= 0 else "y_neg")
    active = ((stage == EMERGENCY_STAGE_REVERSE_CONFIRM)
              | (stage == EMERGENCY_STAGE_RECOVERY_SETTLE))
    if os.environ.get("FINETUNE_EMERGENCY_STATE_FB_PULL", "0") == "1":
        active = active | (stage == EMERGENCY_STAGE_PULL_STABILIZE)
    damping = float(os.environ.get("FINETUNE_EMERGENCY_STATE_FB_DAMPING", "0.05"))
    if damping < 0.0:
        raise ValueError("FINETUNE_EMERGENCY_STATE_FB_DAMPING must be non-negative")
    for i, name in enumerate(names):
        group = table.get(name, {})
        mask = "".join(str(int(v)) for v in (latest[i, 6:10] >= 0.5).tolist())
        item = group.get(mask) or group.get("1111")
        if not isinstance(item, dict):
            continue
        try:
            A = torch.as_tensor(item["A"], device=features.device, dtype=features.dtype)
            K = torch.as_tensor(item["K"], device=features.device, dtype=features.dtype)
            state = torch.cat((latest[i, :2], latest[i, 3:6], latest[i, 6:10]))
            bias = torch.as_tensor(item.get("bias", [0.0, 0.0, 0.0]), device=features.device, dtype=features.dtype)
            desired = -(state @ A + bias) - damping * latest[i, 3:6]
            out[i] = desired @ K.T
        except (KeyError, TypeError, ValueError, RuntimeError):
            continue
    return (gain * active.to(features.dtype)[:, None] * out).clamp(-limit, limit)


@dataclass(frozen=True)
class PhaseConditionedResidualGateConfig:
    force_start_n: float = 45.0
    force_full_n: float = 50.0
    pull_dwell_seconds: float = 0.20
    release_dwell_seconds: float = 0.10
    release_arm_force_n: float = 43.0
    release_cancel_force_n: float = 46.0
    reversal_dwell_seconds: float = 0.10
    reversal_dot_threshold: float = -0.5
    recovery_seconds: float = 1.5
    settled_speed_mps: float = 0.05
    settled_seconds: float = 0.20
    step_dt: float = 0.02

    def __post_init__(self) -> None:
        if not self.force_start_n < self.force_full_n:
            raise ValueError("force_start_n must be smaller than force_full_n")
        if self.release_dwell_seconds <= 0.0:
            raise ValueError("release dwell must be positive")
        if self.pull_dwell_seconds <= 0.0:
            raise ValueError("pull dwell must be positive")
        if self.reversal_dwell_seconds <= 0.0:
            raise ValueError("reversal dwell must be positive")
        if not 0.0 < self.release_arm_force_n < self.release_cancel_force_n:
            raise ValueError("release arm/cancel thresholds must be increasing and positive")
        if not -1.0 <= self.reversal_dot_threshold <= 1.0:
            raise ValueError("reversal dot threshold must be in [-1, 1]")
        if self.recovery_seconds <= 0.0 or self.settled_seconds <= 0.0:
            raise ValueError("recovery and settled durations must be positive")
        if self.settled_speed_mps < 0.0:
            raise ValueError("settled speed must be non-negative")
        if self.step_dt <= 0.0:
            raise ValueError("step_dt must be positive")


@dataclass
class PhaseConditionedResidualGateState:
    phase: torch.Tensor
    initial_direction_xy: torch.Tensor
    release_steps: torch.Tensor
    recovery_value: torch.Tensor
    settled_steps: torch.Tensor
    reversal_steps: torch.Tensor
    pull_steps: torch.Tensor


def initial_phase_conditioned_gate_state(
    num_envs: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> PhaseConditionedResidualGateState:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    return PhaseConditionedResidualGateState(
        phase=torch.full(
            (num_envs, 1), PHASE_INITIAL, device=device, dtype=torch.int64
        ),
        initial_direction_xy=torch.zeros((num_envs, 2), device=device, dtype=dtype),
        release_steps=torch.zeros((num_envs, 1), device=device, dtype=torch.int64),
        recovery_value=torch.zeros((num_envs, 1), device=device, dtype=dtype),
        settled_steps=torch.zeros((num_envs, 1), device=device, dtype=torch.int64),
        reversal_steps=torch.zeros((num_envs, 1), device=device, dtype=torch.int64),
        pull_steps=torch.zeros((num_envs, 1), device=device, dtype=torch.int64),
    )


def update_phase_conditioned_residual_gate(
    force_xy_n: torch.Tensor,
    horizontal_speed_mps: torch.Tensor,
    state: PhaseConditionedResidualGateState,
    config: PhaseConditionedResidualGateConfig,
    *,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, PhaseConditionedResidualGateState]:
    """Update a base-only-pull, release-armed reversal gate without mutation."""
    if force_xy_n.ndim != 2 or force_xy_n.shape[1] != 2:
        raise ValueError("force_xy_n must have shape (N, 2)")
    batch = force_xy_n.shape[0]
    scalar_shape = (batch, 1)
    if horizontal_speed_mps.shape != scalar_shape:
        raise ValueError("horizontal_speed_mps must have shape (N, 1)")
    if (
        state.phase.shape != scalar_shape
        or state.initial_direction_xy.shape != (batch, 2)
        or state.release_steps.shape != scalar_shape
        or state.recovery_value.shape != scalar_shape
        or state.settled_steps.shape != scalar_shape
        or state.reversal_steps.shape != scalar_shape
        or state.pull_steps.shape != scalar_shape
    ):
        raise ValueError("force, speed, and phase state must have matching batches")

    phase = state.phase.clone()
    initial_direction = state.initial_direction_xy.clone()
    release_steps = state.release_steps.clone()
    recovery_value = state.recovery_value.clone()
    settled_steps = state.settled_steps.clone()
    reversal_steps = state.reversal_steps.clone()
    pull_steps = state.pull_steps.clone()
    if reset_mask is not None:
        reset = reset_mask.to(device=phase.device, dtype=torch.bool).reshape(-1, 1)
        if reset.shape != scalar_shape:
            raise ValueError("reset_mask must contain one value per environment")
        phase = torch.where(reset, torch.full_like(phase, PHASE_INITIAL), phase)
        initial_direction = torch.where(
            reset.expand(-1, 2), torch.zeros_like(initial_direction), initial_direction
        )
        release_steps = torch.where(reset, torch.zeros_like(release_steps), release_steps)
        recovery_value = torch.where(
            reset, torch.zeros_like(recovery_value), recovery_value
        )
        settled_steps = torch.where(reset, torch.zeros_like(settled_steps), settled_steps)
        reversal_steps = torch.where(reset, torch.zeros_like(reversal_steps), reversal_steps)
        pull_steps = torch.where(reset, torch.zeros_like(pull_steps), pull_steps)

    phase_before = phase.clone()
    force_norm = torch.linalg.vector_norm(force_xy_n, dim=-1, keepdim=True)
    # Hysteresis: a release candidate must be below the lower threshold, while
    # an armed/reversal phase remains active until force exceeds the upper one.
    force_active = force_norm > config.release_cancel_force_n
    force_gate = (
        (force_norm - config.force_start_n)
        / (config.force_full_n - config.force_start_n)
    ).clamp(0.0, 1.0)
    direction = force_xy_n / force_norm.clamp_min(1.0e-6)
    gate = torch.zeros_like(recovery_value)

    initial_active = (phase_before == PHASE_INITIAL) & force_active
    phase = torch.where(initial_active, torch.full_like(phase, PHASE_PULL), phase)
    initial_direction = torch.where(
        initial_active.expand(-1, 2), direction, initial_direction
    )

    pull = phase_before == PHASE_PULL
    pull_active = pull & force_active
    pull_steps = torch.where(pull_active, pull_steps + 1, torch.zeros_like(pull_steps))
    pull_required = max(1, math.ceil(config.pull_dwell_seconds / config.step_dt))
    pull_ready = pull_active & (pull_steps >= pull_required)
    gate = torch.where(pull_ready, force_gate, gate)
    release_candidate = pull & (force_norm < config.release_arm_force_n)
    release_steps = torch.where(
        pull & ~release_candidate, torch.zeros_like(release_steps), release_steps
    )
    release_steps = torch.where(
        release_candidate, release_steps + 1, release_steps
    )
    release_required = max(
        1, math.ceil(config.release_dwell_seconds / config.step_dt)
    )
    release_ready = release_candidate & (release_steps >= release_required)
    phase = torch.where(
        release_ready, torch.full_like(phase, PHASE_RELEASE_ARMED), phase
    )

    armed = phase_before == PHASE_RELEASE_ARMED
    # A confirmed reversal must clear the full-force threshold.  The lower
    # hysteresis threshold only keeps the release arm alive; allowing it to
    # open the residual would let estimator noise leak into the pull segment.
    armed_active = armed & (force_norm >= config.force_full_n)
    direction_dot = torch.sum(direction * initial_direction, dim=1, keepdim=True)
    valid_reversal = armed_active & (
        direction_dot <= config.reversal_dot_threshold
    )
    reversal_steps = torch.where(
        armed_active & ~valid_reversal,
        torch.zeros_like(reversal_steps),
        reversal_steps,
    )
    reversal_steps = torch.where(
        valid_reversal,
        reversal_steps + 1,
        reversal_steps,
    )
    reversal_required = max(
        1, math.ceil(config.reversal_dwell_seconds / config.step_dt)
    )
    reversal_ready = valid_reversal & (reversal_steps >= reversal_required)
    base_repull = armed_active & ~valid_reversal
    phase = torch.where(
        reversal_ready, torch.full_like(phase, PHASE_REVERSAL), phase
    )
    gate = torch.where(reversal_ready, force_gate, gate)
    recovery_value = torch.where(reversal_ready, force_gate, recovery_value)
    phase = torch.where(base_repull, torch.full_like(phase, PHASE_PULL), phase)
    initial_direction = torch.where(
        base_repull.expand(-1, 2), direction, initial_direction
    )
    release_steps = torch.where(
        base_repull, torch.zeros_like(release_steps), release_steps
    )
    reversal_steps = torch.where(
        base_repull, torch.zeros_like(reversal_steps), reversal_steps
    )
    pull_steps = torch.where(base_repull, torch.zeros_like(pull_steps), pull_steps)

    reversal = phase_before == PHASE_REVERSAL
    reversal_active = reversal & force_active
    reversal_released = reversal & ~force_active
    gate = torch.where(reversal_active, force_gate, gate)
    recovery_value = torch.where(reversal_active, force_gate, recovery_value)
    phase = torch.where(
        reversal_released, torch.full_like(phase, PHASE_RECOVERY), phase
    )

    recovery = (phase_before == PHASE_RECOVERY) | reversal_released
    recovery_active = recovery & force_active
    recovery_released = recovery & ~force_active
    recovery_steps = max(1, math.ceil(config.recovery_seconds / config.step_dt))
    decay_per_step = 1.0 / recovery_steps
    decayed = (recovery_value - decay_per_step).clamp_min(0.0)
    decayed = torch.where(
        decayed <= decay_per_step * 1.0e-4, torch.zeros_like(decayed), decayed
    )
    recovery_value = torch.where(recovery_released, decayed, recovery_value)
    gate = torch.where(recovery_released, recovery_value, gate)
    settled = recovery_released & (
        horizontal_speed_mps < config.settled_speed_mps
    )
    settled_steps = torch.where(
        settled, settled_steps + 1, torch.zeros_like(settled_steps)
    )
    clear_after = max(1, math.ceil(config.settled_seconds / config.step_dt))
    recovery_done = recovery_released & (
        (recovery_value <= 0.0) | (settled_steps >= clear_after)
    )
    gate = torch.where(recovery_done, torch.zeros_like(gate), gate)
    recovery_value = torch.where(
        recovery_done, torch.zeros_like(recovery_value), recovery_value
    )
    settled_steps = torch.where(
        recovery_done, torch.zeros_like(settled_steps), settled_steps
    )
    release_steps = torch.where(
        recovery_done, torch.zeros_like(release_steps), release_steps
    )
    reversal_steps = torch.where(
        recovery_done, torch.zeros_like(reversal_steps), reversal_steps
    )
    pull_steps = torch.where(recovery_done, torch.zeros_like(pull_steps), pull_steps)
    initial_direction = torch.where(
        recovery_done.expand(-1, 2), torch.zeros_like(initial_direction), initial_direction
    )
    phase = torch.where(
        recovery_done, torch.full_like(phase, PHASE_INITIAL), phase
    )

    phase = torch.where(
        recovery_active, torch.full_like(phase, PHASE_PULL), phase
    )
    initial_direction = torch.where(
        recovery_active.expand(-1, 2), direction, initial_direction
    )
    release_steps = torch.where(
        recovery_active, torch.zeros_like(release_steps), release_steps
    )
    reversal_steps = torch.where(
        recovery_active, torch.zeros_like(reversal_steps), reversal_steps
    )
    pull_steps = torch.where(recovery_active, torch.zeros_like(pull_steps), pull_steps)
    recovery_value = torch.where(
        recovery_active, torch.zeros_like(recovery_value), recovery_value
    )
    settled_steps = torch.where(
        recovery_active, torch.zeros_like(settled_steps), settled_steps
    )
    gate = torch.where(recovery_active, torch.zeros_like(gate), gate)

    return gate, PhaseConditionedResidualGateState(
        phase=phase,
        initial_direction_xy=initial_direction,
        release_steps=release_steps,
        recovery_value=recovery_value,
        settled_steps=settled_steps,
        reversal_steps=reversal_steps,
        pull_steps=pull_steps,
    )


def initial_gate_state(
    num_envs: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> ResidualGateState:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    return ResidualGateState(
        value=torch.zeros((num_envs, 1), device=device, dtype=dtype),
        settled_steps=torch.zeros((num_envs, 1), device=device, dtype=torch.int64),
    )


def update_residual_gate(
    force_xy_n: torch.Tensor,
    horizontal_speed_mps: torch.Tensor,
    state: ResidualGateState,
    config: ResidualGateConfig,
    *,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ResidualGateState]:
    """Update the high-force/recovery gate without mutating ``state``."""
    if force_xy_n.ndim != 2 or force_xy_n.shape[1] != 2:
        raise ValueError("force_xy_n must have shape (N, 2)")
    if horizontal_speed_mps.shape != state.value.shape or force_xy_n.shape[0] != state.value.shape[0]:
        raise ValueError("force, speed, and state tensors must have matching batch sizes")

    value = state.value.clone()
    settled_steps = state.settled_steps.clone()
    if reset_mask is not None:
        reset_mask = reset_mask.to(device=value.device, dtype=torch.bool).reshape(-1, 1)
        if reset_mask.shape != value.shape:
            raise ValueError("reset_mask must contain one value per environment")
        value = torch.where(reset_mask, torch.zeros_like(value), value)
        settled_steps = torch.where(reset_mask, torch.zeros_like(settled_steps), settled_steps)

    force_norm_n = torch.linalg.vector_norm(force_xy_n, dim=-1, keepdim=True)
    force_gate = ((force_norm_n - config.force_start_n) / (
        config.force_full_n - config.force_start_n
    )).clamp(0.0, 1.0)
    force_active = force_norm_n > config.force_start_n

    recovery_steps = max(1, math.ceil(config.recovery_seconds / config.step_dt))
    decay_per_step = 1.0 / recovery_steps
    decayed = value - decay_per_step
    decayed = torch.where(decayed <= decay_per_step * 1e-4, torch.zeros_like(decayed), decayed)
    value = torch.where(force_active, torch.maximum(value, force_gate), decayed)

    settled = (~force_active) & (horizontal_speed_mps < config.settled_speed_mps)
    settled_steps = torch.where(settled, settled_steps + 1, torch.zeros_like(settled_steps))
    clear_after_steps = max(1, math.ceil(config.settled_seconds / config.step_dt))
    value = torch.where(settled_steps >= clear_after_steps, torch.zeros_like(value), value)

    gate = torch.maximum(force_gate, value)
    return gate, ResidualGateState(value=value, settled_steps=settled_steps)


class ResidualMLPModel(MLPModel):
    """Frozen base actor plus bounded residual and independent emergency means."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        residual_hidden_dims: tuple[int, int] = (128, 64),
        residual_limit: float = 0.10,
        emergency_hidden_dims: tuple[int, int] = (128, 64),
        emergency_limit: float = 0.15,
    ) -> None:
        if "residual_gate" not in obs:
            raise ValueError("ResidualMLPModel requires a residual_gate observation")
        base_obs_groups = {
            name: [
                group
                for group in groups
                if group not in {"residual_gate", "emergency_features", "emergency_gate"}
            ]
            for name, groups in obs_groups.items()
        }
        super().__init__(
            obs,
            base_obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        # Preserve the conservative default while allowing controlled
        # high-force training runs to request more residual authority.
        if "FINETUNE_RESIDUAL_LIMIT" in os.environ:
            residual_limit = float(os.environ["FINETUNE_RESIDUAL_LIMIT"])
        self.residual_limit = float(residual_limit)
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.obs_dim, residual_hidden_dims[0]),
            nn.ELU(),
            nn.Linear(residual_hidden_dims[0], residual_hidden_dims[1]),
            nn.ELU(),
            nn.Linear(residual_hidden_dims[1], output_dim),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        # Explicit command-conditioned gait head.  It starts at zero so old
        # checkpoints retain their behavior; when enabled it learns a small
        # bounded propulsion correction from the frozen actor latent, which
        # already contains the commanded velocity and phase.
        self.velocity_head_limit = float(os.environ.get("FINETUNE_VELOCITY_HEAD_LIMIT", "0.06"))
        self.velocity_head_mlp = nn.Sequential(
            nn.Linear(self.obs_dim, residual_hidden_dims[0]), nn.ELU(),
            nn.Linear(residual_hidden_dims[0], residual_hidden_dims[1]), nn.ELU(),
            nn.Linear(residual_hidden_dims[1], output_dim),
        )
        nn.init.zeros_(self.velocity_head_mlp[-1].weight)
        nn.init.zeros_(self.velocity_head_mlp[-1].bias)
        self.lateral_velocity_head_limit = float(os.environ.get("FINETUNE_LATERAL_HEAD_LIMIT", "0.04"))
        self.lateral_velocity_head_mlp = nn.Sequential(
            nn.Linear(self.obs_dim, residual_hidden_dims[0]), nn.ELU(),
            nn.Linear(residual_hidden_dims[0], residual_hidden_dims[1]), nn.ELU(),
            nn.Linear(residual_hidden_dims[1], output_dim),
        )
        nn.init.zeros_(self.lateral_velocity_head_mlp[-1].weight)
        nn.init.zeros_(self.lateral_velocity_head_mlp[-1].bias)
        if emergency_limit <= 0.0:
            raise ValueError("emergency_limit must be positive")
        if "FINETUNE_EMERGENCY_LIMIT" in os.environ:
            emergency_limit = float(os.environ["FINETUNE_EMERGENCY_LIMIT"])
        self.emergency_limit = float(emergency_limit)
        self.emergency_history_enabled = os.environ.get("FINETUNE_EMERGENCY_HISTORY", "0") == "1"
        self.emergency_history_frames = 5 if self.emergency_history_enabled else 1
        self.emergency_input_dim = 12 * self.emergency_history_frames
        self.emergency_mlp = nn.Sequential(
            nn.Linear(self.emergency_input_dim, emergency_hidden_dims[0]),
            nn.ELU(),
            nn.Linear(emergency_hidden_dims[0], emergency_hidden_dims[1]),
            nn.ELU(),
            nn.Linear(emergency_hidden_dims[1], output_dim),
        )
        nn.init.zeros_(self.emergency_mlp[-1].weight)
        nn.init.zeros_(self.emergency_mlp[-1].bias)
        # Independent stability expert: direction history is masked out at
        # inference, so this head learns only pose/rate/contact support
        # actions and cannot become a force-following policy.
        self.emergency_stability_mlp = nn.Sequential(
            nn.Linear(self.emergency_input_dim, emergency_hidden_dims[0]),
            nn.ELU(),
            nn.Linear(emergency_hidden_dims[0], emergency_hidden_dims[1]),
            nn.ELU(),
            nn.Linear(emergency_hidden_dims[1], output_dim),
        )
        nn.init.zeros_(self.emergency_stability_mlp[-1].weight)
        nn.init.zeros_(self.emergency_stability_mlp[-1].bias)
        # Low-dimensional virtual dynamics head.  It predicts roll brake,
        # pitch brake, and lateral support rather than unconstrained joint
        # actions; the fixed contact/phase mapping below makes the learned
        # output interpretable and prevents contact-mode averaging.
        def _make_virtual_head() -> nn.Sequential:
            head = nn.Sequential(
                nn.Linear(self.emergency_input_dim, emergency_hidden_dims[0]), nn.ELU(),
                nn.Linear(emergency_hidden_dims[0], emergency_hidden_dims[1]), nn.ELU(),
                nn.Linear(emergency_hidden_dims[1], 3),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            return head
        self.emergency_virtual_mlp = _make_virtual_head()
        self.emergency_virtual_reverse_mlp = _make_virtual_head()
        self.emergency_virtual_recovery_mlp = _make_virtual_head()
        # Stage-specific heads prevent one shared emergency mapping from
        # averaging incompatible reverse-confirm and recovery-settle actions.
        # They receive the same pose/rate/contact/direction history and are
        # selected only by internal FSM metadata.
        def _make_stage_head() -> nn.Sequential:
            head = nn.Sequential(
                nn.Linear(self.emergency_input_dim, emergency_hidden_dims[0]),
                nn.ELU(),
                nn.Linear(emergency_hidden_dims[0], emergency_hidden_dims[1]),
                nn.ELU(),
                nn.Linear(emergency_hidden_dims[1], output_dim),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            return head

        self.emergency_reverse_mlp = _make_stage_head()
        self.emergency_recovery_mlp = _make_stage_head()
        # Direction-conditioned delta heads refine the existing stage heads
        # without changing the legacy action at initialization.  The delta
        # is selected from the confirmed finite Y direction and therefore
        # cannot silently average +Y and -Y recovery dynamics.
        for _stage_name in ("pull", "reverse", "recovery"):
            for _direction_name in ("y_pos", "y_neg"):
                _delta = _make_stage_head()
                setattr(self, f"emergency_{_stage_name}_{_direction_name}_delta_mlp", _delta)
        # Dedicated lateral recovery sub-head.  It outputs one bounded
        # scalar which is expanded into a mirrored hip-brace pattern; keeping
        # it separate prevents a generic 12-D head from spending capacity on
        # unrelated joints during the ±Y failure mode.
        self.emergency_lateral_mlp = nn.Sequential(
            nn.Linear(self.emergency_input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.emergency_lateral_mlp[-1].weight)
        nn.init.zeros_(self.emergency_lateral_mlp[-1].bias)
        # Independent per-joint support.  A four-channel hip-only brace cannot
        # represent asymmetric contact loss during diagonal reversal: the
        # stabilizing response must be able to coordinate the hip, thigh and
        # calf channels for each leg.  The head still receives only the
        # restricted pose/rate/contact/direction history and remains exactly
        # zero-initialized.
        self.emergency_support_mlp = nn.Sequential(
            nn.Linear(self.emergency_input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 12),
        )
        nn.init.zeros_(self.emergency_support_mlp[-1].weight)
        nn.init.zeros_(self.emergency_support_mlp[-1].bias)
        # Direction-specific support heads prevent a single shared brace from
        # averaging the asymmetric +Y and -Y contact-loss dynamics.
        self.emergency_support_y_pos_mlp = nn.Sequential(
            nn.Linear(self.emergency_input_dim, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 12)
        )
        self.emergency_support_y_neg_mlp = nn.Sequential(
            nn.Linear(self.emergency_input_dim, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 12)
        )
        for _head in (self.emergency_support_y_pos_mlp, self.emergency_support_y_neg_mlp):
            nn.init.zeros_(_head[-1].weight)
            nn.init.zeros_(_head[-1].bias)
        # Contact-phase joint adapter: a separate high-capacity path for the
        # confirmed reverse/recovery contact transitions.  It is zero
        # initialized and never receives force magnitude or velocity command.
        self.emergency_contact_adapter_mlp = nn.Sequential(
            nn.Linear(self.emergency_input_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, output_dim)
        )
        nn.init.zeros_(self.emergency_contact_adapter_mlp[-1].weight)
        nn.init.zeros_(self.emergency_contact_adapter_mlp[-1].bias)
        # Joint-state adapter for contact transitions.  These inputs are all
        # robot-observable (joint position, joint velocity, previous action,
        # and filtered foot-contact force summary); the final layer is zero at
        # initialization so loading the existing safe checkpoint preserves
        # its behavior until this branch is explicitly enabled/trained.
        self.emergency_joint_adapter_mlp = nn.Sequential(
            nn.Linear(40, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, output_dim)
        )
        nn.init.zeros_(self.emergency_joint_adapter_mlp[-1].weight)
        nn.init.zeros_(self.emergency_joint_adapter_mlp[-1].bias)
        # Separate adapter for the robot-observable actuator torque/current
        # estimate. Keeping it as a new module preserves compatibility with
        # checkpoints whose 40-D joint adapter predates this input.
        self.emergency_joint_torque_adapter_mlp = nn.Sequential(
            nn.Linear(52, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, output_dim)
        )
        nn.init.zeros_(self.emergency_joint_torque_adapter_mlp[-1].weight)
        nn.init.zeros_(self.emergency_joint_torque_adapter_mlp[-1].bias)
        # Explicit contact-mode expert. The one-hot contact mask prevents
        # averaging incompatible support topologies during diagonal recovery;
        # the final layer is zero so existing checkpoints remain unchanged.
        self.emergency_contact_mode_mlp = nn.Sequential(
            nn.Linear(self.emergency_input_dim + 16 + 4, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, output_dim)
        )
        nn.init.zeros_(self.emergency_contact_mode_mlp[-1].weight)
        nn.init.zeros_(self.emergency_contact_mode_mlp[-1].bias)

    def _emergency_mean(self, obs: TensorDict) -> torch.Tensor:
        features = obs.get("emergency_features")
        gate = obs.get("emergency_gate")
        if features is None or gate is None:
            return torch.zeros(
                (*obs["actor"].shape[:-1], self.mlp[-1].out_features),
                device=obs["actor"].device,
                dtype=obs["actor"].dtype,
            )
        if features.shape[-1] != self.emergency_input_dim:
            raise ValueError(
                f"emergency_features must have last dimension {self.emergency_input_dim}"
            )
        if gate.shape[-1] != 1:
            raise ValueError("emergency_gate must have last dimension 1")
        stage = obs.get("emergency_stage")
        stage_present = stage is not None
        if stage is None:
            stage = torch.zeros(features.shape[0], device=features.device, dtype=torch.long)
        learned_action = self.emergency_limit * torch.tanh(self.emergency_mlp(features))
        stability_features = features.reshape(features.shape[0], -1, 12).clone()
        stability_features[:, :, 10:12] = 0.0
        stability_features = stability_features.reshape(features.shape[0], -1)
        stability_action = self.emergency_limit * torch.tanh(
            self.emergency_stability_mlp(stability_features)
        )
        virtual_values = self.emergency_virtual_mlp(features)
        virtual_reverse = self.emergency_virtual_reverse_mlp(features)
        virtual_recovery = self.emergency_virtual_recovery_mlp(features)
        virtual_values = torch.where(
            (stage == EMERGENCY_STAGE_REVERSE_CONFIRM).unsqueeze(1),
            virtual_reverse, virtual_values
        )
        virtual_values = torch.where(
            (stage == EMERGENCY_STAGE_RECOVERY_SETTLE).unsqueeze(1),
            virtual_recovery, virtual_values
        )
        virtual_action = self.emergency_limit * torch.tanh(virtual_values)
        virtual_gain = float(os.environ.get("FINETUNE_EMERGENCY_VIRTUAL_GAIN", "0.0"))
        if virtual_gain < 0.0:
            raise ValueError("FINETUNE_EMERGENCY_VIRTUAL_GAIN must be non-negative")
        # Fixed virtual-to-joint map: hip roll pair, pitch pair, then
        # lateral support through the loaded/unloaded diagonal.  The map is
        # intentionally phase-agnostic; phase only selects the learned head.
        virtual_joint = torch.zeros_like(learned_action)
        vroll, vpitch, vlast = virtual_action.unbind(dim=1)
        virtual_joint[:, [0, 6]] = vroll.unsqueeze(1)
        virtual_joint[:, [3, 9]] = -vroll.unsqueeze(1)
        virtual_joint[:, [1, 2, 4, 5]] = vpitch.unsqueeze(1)
        virtual_joint[:, [7, 8, 10, 11]] = -vpitch.unsqueeze(1)
        virtual_joint[:, [0, 9]] += vlast.unsqueeze(1)
        virtual_joint[:, [3, 6]] -= vlast.unsqueeze(1)
        reverse_action = self.emergency_limit * torch.tanh(
            self.emergency_reverse_mlp(features)
        )
        recovery_action = self.emergency_limit * torch.tanh(
            self.emergency_recovery_mlp(features)
        )
        lateral_action = self.emergency_limit * torch.tanh(
            self.emergency_lateral_mlp(features)
        ).squeeze(-1)
        shared_support_action = self.emergency_limit * torch.tanh(
            self.emergency_support_mlp(features)
        )
        support_pos = self.emergency_limit * torch.tanh(
            self.emergency_support_y_pos_mlp(features)
        )
        support_neg = self.emergency_limit * torch.tanh(
            self.emergency_support_y_neg_mlp(features)
        )
        scaffold = emergency_scaffold_action(
            features, stage, limit=self.emergency_limit
        )
        _, learned_scale = emergency_stage_scales(
            stage, dtype=features.dtype, device=features.device
        )
        learned_override = os.environ.get("FINETUNE_EMERGENCY_LEARNED_SCALE_OVERRIDE")
        if learned_override is not None:
            learned_scale = learned_scale * float(learned_override)
        if not stage_present:
            # Preserve legacy zero-initialization semantics for callers that
            # do not provide the new stage metadata; the runner always does.
            scaffold = torch.zeros_like(scaffold)
            learned_scale = torch.zeros_like(learned_scale)
        learned_action = torch.where(
            (stage == EMERGENCY_STAGE_REVERSE_CONFIRM).unsqueeze(1),
            reverse_action,
            torch.where(
                (stage == EMERGENCY_STAGE_RECOVERY_SETTLE).unsqueeze(1),
                recovery_action,
                learned_action,
            ),
        )
        direction_y = features[:, -1] if self.emergency_history_enabled else features[:, 11]
        direction_valid = direction_y.abs() > 0.2
        direction_positive = direction_y >= 0.0
        pull_pos = self.emergency_pull_y_pos_delta_mlp(features)
        pull_neg = self.emergency_pull_y_neg_delta_mlp(features)
        reverse_pos = self.emergency_reverse_y_pos_delta_mlp(features)
        reverse_neg = self.emergency_reverse_y_neg_delta_mlp(features)
        recovery_pos = self.emergency_recovery_y_pos_delta_mlp(features)
        recovery_neg = self.emergency_recovery_y_neg_delta_mlp(features)
        direction_delta = torch.where(
            direction_positive.unsqueeze(1),
            pull_pos,
            pull_neg,
        )
        direction_delta = torch.where(
            (stage == EMERGENCY_STAGE_REVERSE_CONFIRM).unsqueeze(1),
            torch.where(direction_positive.unsqueeze(1), reverse_pos, reverse_neg),
            direction_delta,
        )
        direction_delta = torch.where(
            (stage == EMERGENCY_STAGE_RECOVERY_SETTLE).unsqueeze(1),
            torch.where(direction_positive.unsqueeze(1), recovery_pos, recovery_neg),
            direction_delta,
        )
        direction_delta = self.emergency_limit * torch.tanh(direction_delta)
        learned_action = learned_action + direction_valid.unsqueeze(1) * direction_delta
        support_action = torch.where(
            (direction_y >= 0.0).unsqueeze(1), support_pos, support_neg
        )
        # Keep both directional heads in the autograd graph for optimizer
        # bookkeeping even when a minibatch contains only one sign.
        support_action = support_action + 0.0 * (support_pos + support_neg + shared_support_action)
        contact_adapter = self.emergency_limit * torch.tanh(
            self.emergency_contact_adapter_mlp(features)
        )
        joint_features = obs.get("emergency_joint_features")
        if joint_features is None:
            joint_features = torch.zeros(
                (features.shape[0], 40), device=features.device, dtype=features.dtype
            )
        if joint_features.shape[-1] != 40:
            raise ValueError("emergency_joint_features must have last dimension 40")
        joint_adapter = self.emergency_limit * torch.tanh(
            self.emergency_joint_adapter_mlp(joint_features)
        )
        torque_features = obs.get("emergency_torque_features")
        if torque_features is None:
            torque_features = torch.zeros(
                (features.shape[0], 52), device=features.device, dtype=features.dtype
            )
        if torque_features.shape[-1] != 52:
            raise ValueError("emergency_torque_features must have last dimension 52")
        torque_adapter = self.emergency_limit * torch.tanh(
            self.emergency_joint_torque_adapter_mlp(torque_features)
        )
        latest = features[:, -12:]
        contact_bits = (latest[:, 6:10] >= 0.5).to(torch.long)
        contact_code = (contact_bits * torch.tensor(
            [1, 2, 4, 8], device=features.device, dtype=torch.long
        )).sum(dim=1)
        contact_mode = torch.nn.functional.one_hot(
            contact_code.clamp(0, 15), num_classes=16
        ).to(features.dtype)
        stage_mode = torch.nn.functional.one_hot(
            stage.clamp(0, 3), num_classes=4
        ).to(features.dtype)
        mode_features = torch.cat((features, contact_mode, stage_mode), dim=1)
        contact_mode_action = self.emergency_limit * torch.tanh(
            self.emergency_contact_mode_mlp(mode_features)
        )
        # Keep a deterministic posture/ang-rate recovery component in the
        # emergency path.  The learned head can refine it, but cannot erase
        # the stabilizing response when the >50 N gate is active.
        target_features = features[:, -12:] if self.emergency_history_enabled else features
        sign = float(os.environ.get("FINETUNE_EMERGENCY_TARGET_SIGN", "1.0"))
        if sign not in (-1.0, 1.0):
            raise ValueError("FINETUNE_EMERGENCY_TARGET_SIGN must be -1 or 1")
        roll = sign * (-(0.30 * target_features[:, 1] + 0.10 * target_features[:, 4]))
        pitch = sign * (-(0.30 * target_features[:, 0] + 0.10 * target_features[:, 3]))
        direction_gain = float(os.environ.get("FINETUNE_EMERGENCY_DIRECTION_GAIN", "0.0"))
        roll = roll - direction_gain * target_features[:, 11]
        pitch = pitch - direction_gain * target_features[:, 10]
        target = torch.zeros_like(learned_action)
        target[:, [0, 6]] = roll.unsqueeze(1)
        target[:, [3, 9]] = -roll.unsqueeze(1)
        target[:, [1, 2, 4, 5]] = pitch.unsqueeze(1)
        target[:, [7, 8, 10, 11]] = -pitch.unsqueeze(1)
        target_scale = float(os.environ.get("FINETUNE_EMERGENCY_TARGET_SCALE", "1.0"))
        if target_scale <= 0.0:
            raise ValueError("FINETUNE_EMERGENCY_TARGET_SCALE must be positive")
        target = (target_scale * target).clamp(-self.emergency_limit, self.emergency_limit)
        blend = float(os.environ.get("FINETUNE_EMERGENCY_TARGET_BLEND", "0.0"))
        if not 0.0 <= blend <= 1.0:
            raise ValueError("FINETUNE_EMERGENCY_TARGET_BLEND must be in [0, 1]")
        learned_action = (1.0 - blend) * learned_action + blend * target
        # Structured action: deterministic brace is always present; the
        # learned branch is unlocked only after reversal/recovery stage.
        action = scaffold + learned_scale.unsqueeze(1) * learned_action
        stability_gain = float(os.environ.get("FINETUNE_EMERGENCY_STABILITY_GAIN", "0.0"))
        if stability_gain < 0.0:
            raise ValueError("FINETUNE_EMERGENCY_STABILITY_GAIN must be non-negative")
        action = action + stability_gain * stability_action
        action = action + virtual_gain * virtual_joint
        action = action + emergency_dynamics_brake_action(
            features, stage, limit=self.emergency_limit
        )
        action = action + emergency_state_feedback_action(
            features, stage, limit=self.emergency_limit
        )
        action = action + emergency_contact_transition_action(
            features, stage, limit=self.emergency_limit
        )
        lateral = torch.zeros_like(action)
        pattern = os.environ.get("FINETUNE_EMERGENCY_LATERAL_PATTERN", "mirrored")
        if pattern == "mirrored":
            lateral[:, [0, 6]] = lateral_action.unsqueeze(1)
            lateral[:, [3, 9]] = -lateral_action.unsqueeze(1)
        elif pattern == "uniform":
            lateral[:, [0, 3, 6, 9]] = lateral_action.unsqueeze(1)
        elif pattern == "front_rear":
            lateral[:, [0, 3]] = lateral_action.unsqueeze(1)
            lateral[:, [6, 9]] = -lateral_action.unsqueeze(1)
        elif pattern == "diagonal":
            lateral[:, [0, 9]] = lateral_action.unsqueeze(1)
            lateral[:, [3, 6]] = -lateral_action.unsqueeze(1)
        elif pattern == "side_support":
            lateral[:, [1, 2, 7, 8]] = lateral_action.unsqueeze(1)
            lateral[:, [4, 5, 10, 11]] = -lateral_action.unsqueeze(1)
        else:
            raise ValueError(
                "FINETUNE_EMERGENCY_LATERAL_PATTERN must be mirrored, uniform, front_rear, diagonal, or side_support"
            )
        action = action + learned_scale.unsqueeze(1) * lateral
        # Learned support is a bounded per-joint correction.  It is unlocked
        # only in confirmed reverse/recovery stages through learned_scale;
        # pull therefore remains scaffold-only and base-equivalent.
        support_gain = float(os.environ.get("FINETUNE_EMERGENCY_SUPPORT_GAIN", "1.0"))
        if support_gain < 0.0:
            raise ValueError("FINETUNE_EMERGENCY_SUPPORT_GAIN must be non-negative")
        action = action + learned_scale.unsqueeze(1) * support_gain * support_action
        adapter_gain = float(os.environ.get("FINETUNE_EMERGENCY_CONTACT_ADAPTER_GAIN", "0.0"))
        if adapter_gain < 0.0:
            raise ValueError("FINETUNE_EMERGENCY_CONTACT_ADAPTER_GAIN must be non-negative")
        adapter_stage = (
            (stage == EMERGENCY_STAGE_REVERSE_CONFIRM)
            | (stage == EMERGENCY_STAGE_RECOVERY_SETTLE)
        ).unsqueeze(1)
        action = action + adapter_stage.to(action.dtype) * adapter_gain * contact_adapter
        joint_gain = float(os.environ.get("FINETUNE_EMERGENCY_JOINT_ADAPTER_GAIN", "0.0"))
        if joint_gain < 0.0:
            raise ValueError("FINETUNE_EMERGENCY_JOINT_ADAPTER_GAIN must be non-negative")
        action = action + adapter_stage.to(action.dtype) * joint_gain * joint_adapter
        torque_gain = float(os.environ.get("FINETUNE_EMERGENCY_TORQUE_ADAPTER_GAIN", "0.0"))
        if torque_gain < 0.0:
            raise ValueError("FINETUNE_EMERGENCY_TORQUE_ADAPTER_GAIN must be non-negative")
        action = action + adapter_stage.to(action.dtype) * torque_gain * torque_adapter
        mode_gain = float(os.environ.get("FINETUNE_EMERGENCY_CONTACT_MODE_GAIN", "0.0"))
        if mode_gain < 0.0:
            raise ValueError("FINETUNE_EMERGENCY_CONTACT_MODE_GAIN must be non-negative")
        action = action + mode_gain * contact_mode_action
        return gate * action

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        base_mean = self.mlp(latent)
        residual_gate = obs["residual_gate"]
        residual_mean = self.residual_limit * torch.tanh(self.residual_mlp(latent))
        emergency_mean = self._emergency_mean(obs)
        emergency_gate = obs.get("emergency_gate")
        if emergency_gate is not None:
            base_scale = float(os.environ.get("FINETUNE_EMERGENCY_BASE_SCALE", "1.0"))
            if not 0.0 <= base_scale <= 1.0:
                raise ValueError("FINETUNE_EMERGENCY_BASE_SCALE must be in [0, 1]")
            active = emergency_gate > 1.0e-6
            base_mean = torch.where(active, base_scale * base_mean, base_mean)
        velocity_action = self.velocity_head_limit * torch.tanh(self.velocity_head_mlp(latent))
        lateral_action = self.lateral_velocity_head_limit * torch.tanh(self.lateral_velocity_head_mlp(latent))
        if os.environ.get("FINETUNE_VELOCITY_HEAD_MASK", "") == "propulsion":
            lateral_action = lateral_action * lateral_action.new_tensor(
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
            )
        if os.environ.get("FINETUNE_VELOCITY_HEAD_MASK", "") == "propulsion":
            # Keep the learned command correction out of the distal calf
            # joints; those joints primarily preserve contact/clearance and
            # were the source of rare illegal contacts in the unrestricted
            # command-head smoke.  Policy order is [hip, thigh, calf] per leg.
            mask = velocity_action.new_tensor(
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
            )
            velocity_action = velocity_action * mask
        diag_neg_scale = float(os.environ.get("FINETUNE_VELOCITY_HEAD_DIAG_NEG_SCALE", "1.0"))
        if not 0.0 < diag_neg_scale <= 1.0:
            raise ValueError("FINETUNE_VELOCITY_HEAD_DIAG_NEG_SCALE must be in (0, 1]")
        command = obs["actor"][:, 13:15]
        diag_neg = (command[:, 0] < -0.04) & (command[:, 1] < -0.04)
        velocity_action = torch.where(
            diag_neg.unsqueeze(1), velocity_action * diag_neg_scale, velocity_action
        )
        lateral_scale = float(os.environ.get("FINETUNE_VELOCITY_HEAD_LATERAL_SCALE", "1.0"))
        if lateral_scale < 1.0 or lateral_scale > 2.0:
            raise ValueError("FINETUNE_VELOCITY_HEAD_LATERAL_SCALE must be in [1, 2]")
        lateral_dominant = torch.abs(command[:, 1]) > torch.abs(command[:, 0])
        velocity_action = torch.where(
            lateral_dominant.unsqueeze(1), velocity_action * lateral_scale, velocity_action
        )
        if os.environ.get("FINETUNE_VELOCITY_HEAD_ENABLE", "0") != "1":
            velocity_action = torch.zeros_like(velocity_action)
        if os.environ.get("FINETUNE_LATERAL_HEAD_ENABLE", "0") != "1":
            lateral_action = torch.zeros_like(lateral_action)
        # The propulsion head is a command-channel correction, not a generic
        # residual.  Keep it off for standstill/base-only frames, but do not
        # tie it to the phase residual gate: the latter is intentionally
        # closed during pull dwell and is meant only to protect the residual
        # branch.  Above the hard-force threshold the command is zeroed by
        # the runner, so this gate also fails closed for emergency protection.
        velocity_gate_mode = os.environ.get("FINETUNE_VELOCITY_HEAD_GATE", "residual")
        if velocity_gate_mode == "command":
            command = obs["actor"][:, 13:15]
            velocity_gate = (torch.linalg.vector_norm(command, dim=1, keepdim=True) > 0.04).to(base_mean.dtype)
        elif velocity_gate_mode == "residual":
            velocity_gate = residual_gate
        else:
            raise ValueError("FINETUNE_VELOCITY_HEAD_GATE must be 'residual' or 'command'")
        lateral_gate = velocity_gate * (torch.abs(command[:, 1]) > torch.abs(command[:, 0])).unsqueeze(1).to(base_mean.dtype)
        combined_mean = base_mean + residual_gate * residual_mean + velocity_gate * velocity_action + lateral_gate * lateral_action + emergency_mean
        if emergency_gate is not None and os.environ.get("FINETUNE_EMERGENCY_OVERRIDE", "0") == "1":
            active = emergency_gate > 1.0e-6
            # Above the hard-force threshold the dedicated recovery path is
            # authoritative: do not let the nominal locomotion policy fight
            # the posture/contacts-only emergency controller.
            combined_mean = torch.where(active, emergency_mean, combined_mean)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(combined_mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(combined_mean)
        return combined_mean

    def deterministic_action_components(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return base action, gated action, and raw residual action."""
        latent = self.get_latent(obs, None, None)
        base_mean = self.mlp(latent)
        residual_action = self.residual_limit * torch.tanh(self.residual_mlp(latent))
        emergency_action = self._emergency_mean(obs)
        emergency_gate = obs.get("emergency_gate")
        if emergency_gate is not None:
            base_scale = float(os.environ.get("FINETUNE_EMERGENCY_BASE_SCALE", "1.0"))
            if not 0.0 <= base_scale <= 1.0:
                raise ValueError("FINETUNE_EMERGENCY_BASE_SCALE must be in [0, 1]")
            active = emergency_gate > 1.0e-6
            base_mean = torch.where(active, base_scale * base_mean, base_mean)
        velocity_action = self.velocity_head_limit * torch.tanh(self.velocity_head_mlp(latent))
        lateral_action = self.lateral_velocity_head_limit * torch.tanh(self.lateral_velocity_head_mlp(latent))
        if os.environ.get("FINETUNE_VELOCITY_HEAD_MASK", "") == "propulsion":
            lateral_action = lateral_action * lateral_action.new_tensor(
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
            )
        if os.environ.get("FINETUNE_VELOCITY_HEAD_MASK", "") == "propulsion":
            mask = velocity_action.new_tensor(
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
            )
            velocity_action = velocity_action * mask
        diag_neg_scale = float(os.environ.get("FINETUNE_VELOCITY_HEAD_DIAG_NEG_SCALE", "1.0"))
        if not 0.0 < diag_neg_scale <= 1.0:
            raise ValueError("FINETUNE_VELOCITY_HEAD_DIAG_NEG_SCALE must be in (0, 1]")
        command = obs["actor"][:, 13:15]
        diag_neg = (command[:, 0] < -0.04) & (command[:, 1] < -0.04)
        velocity_action = torch.where(
            diag_neg.unsqueeze(1), velocity_action * diag_neg_scale, velocity_action
        )
        lateral_scale = float(os.environ.get("FINETUNE_VELOCITY_HEAD_LATERAL_SCALE", "1.0"))
        if lateral_scale < 1.0 or lateral_scale > 2.0:
            raise ValueError("FINETUNE_VELOCITY_HEAD_LATERAL_SCALE must be in [1, 2]")
        lateral_dominant = torch.abs(command[:, 1]) > torch.abs(command[:, 0])
        velocity_action = torch.where(
            lateral_dominant.unsqueeze(1), velocity_action * lateral_scale, velocity_action
        )
        if os.environ.get("FINETUNE_VELOCITY_HEAD_ENABLE", "0") != "1":
            velocity_action = torch.zeros_like(velocity_action)
        if os.environ.get("FINETUNE_LATERAL_HEAD_ENABLE", "0") != "1":
            lateral_action = torch.zeros_like(lateral_action)
        velocity_gate_mode = os.environ.get("FINETUNE_VELOCITY_HEAD_GATE", "residual")
        if velocity_gate_mode == "command":
            command = obs["actor"][:, 13:15]
            velocity_gate = (torch.linalg.vector_norm(command, dim=1, keepdim=True) > 0.04).to(base_mean.dtype)
        elif velocity_gate_mode == "residual":
            velocity_gate = obs["residual_gate"]
        else:
            raise ValueError("FINETUNE_VELOCITY_HEAD_GATE must be 'residual' or 'command'")
        lateral_gate = velocity_gate * (torch.abs(command[:, 1]) > torch.abs(command[:, 0])).unsqueeze(1).to(base_mean.dtype)
        combined_mean = base_mean + obs["residual_gate"] * residual_action + velocity_gate * velocity_action + lateral_gate * lateral_action + emergency_action
        if emergency_gate is not None and os.environ.get("FINETUNE_EMERGENCY_OVERRIDE", "0") == "1":
            combined_mean = torch.where(
                emergency_gate > 1.0e-6, emergency_action, combined_mean
            )
        if self.distribution is None:
            return base_mean, combined_mean, residual_action
        return (
            self.distribution.deterministic_output(base_mean),
            self.distribution.deterministic_output(combined_mean),
            residual_action,
        )

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(("residual_mlp.", "velocity_head_mlp.", "lateral_velocity_head_mlp.", "emergency_mlp.", "emergency_stability_mlp.", "emergency_virtual_mlp.", "emergency_virtual_reverse_mlp.", "emergency_virtual_recovery_mlp.", "emergency_reverse_mlp.", "emergency_recovery_mlp.", "emergency_lateral_mlp.", "emergency_support_mlp.", "emergency_support_y_pos_mlp.", "emergency_support_y_neg_mlp.", "emergency_contact_adapter_mlp.", "emergency_joint_adapter_mlp.", "emergency_joint_torque_adapter_mlp.", "emergency_contact_mode_mlp.", "emergency_pull_y_", "emergency_reverse_y_", "emergency_recovery_y_"))
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "base checkpoint is incompatible: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )

    def freeze_base(self) -> None:
        update_base = os.environ.get("FINETUNE_UPDATE_BASE_ACTOR", "0") == "1"
        for module in (self.obs_normalizer, self.distribution):
            if module is not None:
                module.requires_grad_(False)
        if self.mlp is not None:
            self.mlp.requires_grad_(update_base)
        self.residual_mlp.requires_grad_(True)
        self.velocity_head_mlp.requires_grad_(True)
        self.lateral_velocity_head_mlp.requires_grad_(True)
        self.emergency_mlp.requires_grad_(True)
        self.emergency_stability_mlp.requires_grad_(True)
        self.emergency_virtual_mlp.requires_grad_(True)
        self.emergency_virtual_reverse_mlp.requires_grad_(True)
        self.emergency_virtual_recovery_mlp.requires_grad_(True)
        self.emergency_reverse_mlp.requires_grad_(True)
        self.emergency_recovery_mlp.requires_grad_(True)
        self.emergency_support_mlp.requires_grad_(True)
        self.emergency_support_y_pos_mlp.requires_grad_(True)
        self.emergency_support_y_neg_mlp.requires_grad_(True)
        self.emergency_contact_adapter_mlp.requires_grad_(True)
        self.emergency_joint_adapter_mlp.requires_grad_(True)
        self.emergency_contact_mode_mlp.requires_grad_(True)
        self.emergency_joint_torque_adapter_mlp.requires_grad_(True)
        for _name, _module in self.named_children():
            if _name.startswith("emergency_") and _name.endswith("_delta_mlp"):
                _module.requires_grad_(True)

    def residual_parameters(self) -> Iterator[nn.Parameter]:
        return iter(
            tuple(self.residual_mlp.parameters())
            + tuple(self.velocity_head_mlp.parameters())
            + tuple(self.lateral_velocity_head_mlp.parameters())
            + tuple(self.emergency_mlp.parameters())
            + tuple(self.emergency_stability_mlp.parameters())
            + tuple(self.emergency_virtual_mlp.parameters())
            + tuple(self.emergency_virtual_reverse_mlp.parameters())
            + tuple(self.emergency_virtual_recovery_mlp.parameters())
            + tuple(self.emergency_reverse_mlp.parameters())
            + tuple(self.emergency_recovery_mlp.parameters())
            + tuple(self.emergency_lateral_mlp.parameters())
            + tuple(self.emergency_support_mlp.parameters())
            + tuple(self.emergency_support_y_pos_mlp.parameters())
            + tuple(self.emergency_support_y_neg_mlp.parameters())
            + tuple(self.emergency_contact_adapter_mlp.parameters())
            + tuple(self.emergency_joint_adapter_mlp.parameters())
            + tuple(self.emergency_contact_mode_mlp.parameters())
            + tuple(self.emergency_joint_torque_adapter_mlp.parameters())
            + tuple(
                parameter
                for name, module in self.named_children()
                if name.startswith("emergency_") and name.endswith("_delta_mlp")
                for parameter in module.parameters()
            )
        )

    def update_normalization(self, obs: TensorDict) -> None:
        return None

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxResidualMLPModel(self, verbose)


class _OnnxResidualMLPModel(nn.Module):
    is_recurrent: bool = False

    def __init__(self, model: ResidualMLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.residual_mlp = copy.deepcopy(model.residual_mlp)
        self.residual_limit = model.residual_limit
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )
        self.input_size = model.obs_dim

    def forward(self, obs: torch.Tensor, residual_gate: torch.Tensor) -> torch.Tensor:
        latent = self.obs_normalizer(obs)
        output = self.mlp(latent) + residual_gate * self.residual_limit * torch.tanh(
            self.residual_mlp(latent)
        )
        return self.deterministic_output(output)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, self.input_size), torch.zeros(1, 1)

    @property
    def input_names(self) -> list[str]:
        return ["obs", "residual_gate"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
