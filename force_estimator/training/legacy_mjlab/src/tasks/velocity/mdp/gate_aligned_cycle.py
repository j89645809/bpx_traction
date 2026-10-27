from __future__ import annotations

import math

import torch


PHASE_MULTIPLIERS = {
    "pull": 1.0,
    "release": 0.0,
    "reverse_dwell": 0.0,
    "reverse": -1.0,
    "final_release": 0.0,
}


def cycle_phase(
    time_s: float,
    hold_s: float,
    release_s: float,
    reverse_dwell_s: float = 0.0,
    phase_offset_s: float = 0.0,
) -> tuple[str, float]:
    if hold_s <= 0.0 or release_s <= 0.0:
        raise ValueError("hold_s and release_s must be positive")
    if reverse_dwell_s < 0.0:
        raise ValueError("reverse_dwell_s must be non-negative")
    if not math.isfinite(phase_offset_s):
        raise ValueError("phase_offset_s must be finite")
    cycle_s = 2.0 * (hold_s + release_s) + reverse_dwell_s
    offset = (float(time_s) + float(phase_offset_s)) % cycle_s
    if offset < hold_s:
        return "pull", offset
    if offset < hold_s + release_s:
        return "release", offset
    reverse_start = hold_s + release_s + reverse_dwell_s
    if offset < hold_s + release_s + reverse_dwell_s:
        return "reverse_dwell", offset
    if offset < reverse_start + hold_s:
        return "reverse", offset
    return "final_release", offset


def phase_multiplier(phase: str) -> float:
    try:
        return PHASE_MULTIPLIERS[phase]
    except KeyError as exc:
        raise ValueError(f"unknown gate-aligned phase: {phase}") from exc


def _validate_band(
    force_min: float, force_max: float, label: str, *, allow_zero: bool = False
) -> None:
    lower_invalid = force_min < 0.0 if allow_zero else force_min <= 0.0
    if lower_invalid or force_max <= force_min:
        raise ValueError(f"invalid {label} force band: {force_min}..{force_max}")


def sample_planar_force(
    count: int,
    force_min: float,
    force_max: float,
    replay: tuple[tuple[float, float, float], ...],
    hard_direction_fraction: float,
    generator: torch.Generator | None,
    device: torch.device,
    hard_direction_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    _validate_band(force_min, force_max, "primary")
    if not 0.0 <= hard_direction_fraction <= 1.0:
        raise ValueError("hard_direction_fraction must be in [0, 1]")

    replay_total = 0.0
    for replay_min, replay_max, fraction in replay:
        _validate_band(replay_min, replay_max, "replay", allow_zero=True)
        if fraction < 0.0:
            raise ValueError("replay fractions must be non-negative")
        replay_total += fraction
    if replay_total > 1.0 + 1.0e-12:
        raise ValueError("replay fractions must sum to at most one")

    random_kwargs = {"device": device, "generator": generator}
    magnitude = force_min + (force_max - force_min) * torch.rand(count, **random_kwargs)
    cohort = torch.rand(count, **random_kwargs)
    offset = 0.0
    for replay_min, replay_max, fraction in replay:
        mask = (cohort >= offset) & (cohort < offset + fraction)
        replay_magnitude = replay_min + (replay_max - replay_min) * torch.rand(
            count, **random_kwargs
        )
        magnitude = torch.where(mask, replay_magnitude, magnitude)
        offset += fraction

    angle = -math.pi + 2.0 * math.pi * torch.rand(count, **random_kwargs)
    hard_mask = torch.rand(count, **random_kwargs) < hard_direction_fraction
    hard_angles = torch.tensor(
        (
            0.0,
            math.pi,
            0.5 * math.pi,
            -0.5 * math.pi,
            0.25 * math.pi,
            -0.25 * math.pi,
            0.75 * math.pi,
            -0.75 * math.pi,
        ),
        device=device,
        dtype=angle.dtype,
    )
    if hard_direction_weights is None:
        hard_index = torch.randint(len(hard_angles), (count,), **random_kwargs)
    else:
        if len(hard_direction_weights) != len(hard_angles):
            raise ValueError("hard_direction_weights must contain eight values")
        weights = torch.tensor(
            hard_direction_weights, device=device, dtype=torch.float32
        )
        if torch.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("hard_direction_weights must be non-negative with positive sum")
        hard_index = torch.multinomial(
            weights, count, replacement=True, generator=generator
        )
    angle = torch.where(hard_mask, hard_angles[hard_index], angle)

    force = torch.zeros((count, 3), device=device, dtype=torch.float32)
    force[:, 0] = magnitude * torch.cos(angle)
    force[:, 1] = magnitude * torch.sin(angle)
    return force


def sample_recovery_state(
    count: int,
    speed_min: float,
    speed_max: float,
    tilt_max_rad: float,
    angular_velocity_max: float,
    generator: torch.Generator | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample body-frame velocity, angular velocity, roll and pitch recovery states."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if speed_min < 0.0 or speed_max < speed_min:
        raise ValueError("invalid recovery speed range")
    if not 0.0 <= tilt_max_rad <= math.radians(10.0):
        raise ValueError("recovery tilt must be between zero and ten degrees")
    if angular_velocity_max < 0.0:
        raise ValueError("recovery angular velocity must be non-negative")
    kwargs = {"device": device, "generator": generator}
    direction = -math.pi + 2.0 * math.pi * torch.rand(count, **kwargs)
    speed = speed_min + (speed_max - speed_min) * torch.rand(count, **kwargs)
    linear_velocity_b = torch.zeros((count, 3), device=device)
    linear_velocity_b[:, 0] = speed * torch.cos(direction)
    linear_velocity_b[:, 1] = speed * torch.sin(direction)
    tilt_direction = -math.pi + 2.0 * math.pi * torch.rand(count, **kwargs)
    tilt = tilt_max_rad * torch.rand(count, **kwargs)
    roll = tilt * torch.cos(tilt_direction)
    pitch = tilt * torch.sin(tilt_direction)
    angular_velocity_b = torch.zeros((count, 3), device=device)
    angular_velocity_b[:, :2] = (
        2.0 * angular_velocity_max * torch.rand((count, 2), **kwargs)
        - angular_velocity_max
    )
    return linear_velocity_b, angular_velocity_b, roll, pitch


def sample_recovery_phase_offsets(
    count: int,
    gait_cycle_s: float,
    generator: torch.Generator | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample four quarter-cycle offsets and their contact-pattern IDs."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if not math.isfinite(gait_cycle_s) or gait_cycle_s <= 0.0:
        raise ValueError("gait_cycle_s must be finite and positive")
    pattern = torch.randint(4, (count,), device=device, generator=generator)
    offsets = pattern.to(dtype=torch.float32) * (gait_cycle_s / 4.0)
    return offsets, pattern


def align_recovery_velocity_with_pull(
    linear_velocity_b: torch.Tensor,
    pull_force_b: torch.Tensor,
    fraction: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Align selected recovery velocities with the preceding pull direction.

    The speed magnitude is preserved.  This creates the physically relevant
    reverse-transition state: the base is still moving with the old pull when
    the force and commanded velocity change sign.
    """
    if linear_velocity_b.ndim != 2 or linear_velocity_b.shape[1] != 3:
        raise ValueError("linear_velocity_b must have shape (N, 3)")
    if pull_force_b.shape != linear_velocity_b.shape:
        raise ValueError("pull_force_b must match linear_velocity_b")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("alignment fraction must be in [0, 1]")

    aligned = linear_velocity_b.clone()
    if fraction == 0.0 or len(aligned) == 0:
        return aligned
    planar_force = pull_force_b[:, :2]
    force_norm = torch.linalg.vector_norm(planar_force, dim=1, keepdim=True)
    valid_force = force_norm.squeeze(1) > 1.0e-6
    selected = torch.rand(
        len(aligned), device=aligned.device, generator=generator
    ) < fraction
    selected &= valid_force
    speed = torch.linalg.vector_norm(aligned[:, :2], dim=1, keepdim=True)
    aligned_xy = planar_force / force_norm.clamp_min(1.0e-6) * speed
    aligned[:, :2] = torch.where(selected.unsqueeze(1), aligned_xy, aligned[:, :2])
    return aligned
