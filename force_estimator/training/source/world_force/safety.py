"""Independent safety supervisor used before policy output is accepted."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class SafetyState(str, Enum):
    STAND = "STAND"
    SENSOR_CHECK = "SENSOR_CHECK"
    ESTIMATOR_WARMUP = "ESTIMATOR_WARMUP"
    FOLLOW_READY = "FOLLOW_READY"
    TUG_FOLLOW = "TUG_FOLLOW"
    DECELERATE = "DECELERATE"
    SAFE_STOP = "SAFE_STOP"


@dataclass(frozen=True)
class SafetyLimits:
    max_roll_rad: float = 0.55
    max_pitch_rad: float = 0.55
    min_height_m: float = 0.18
    max_height_m: float = 0.55
    min_confidence: float = 0.55
    low_confidence_grace_s: float = 0.20
    communication_timeout_s: float = 0.10
    max_command_delta_per_s: float = 1.5


class SafetySupervisor:
    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()
        self.state = SafetyState.STAND
        self._low_confidence_s = 0.0
        self._frame_id: str | None = None

    def reset_frame(self, frame_id: str) -> None:
        if self._frame_id is not None and frame_id != self._frame_id:
            self.state = SafetyState.ESTIMATOR_WARMUP
        self._frame_id = frame_id
        self._low_confidence_s = 0.0

    def evaluate(self, *, roll: float, pitch: float, height: float,
                 confidence: float, communication_age_s: float, dt: float,
                 finite: bool = True) -> SafetyState:
        hard_fault = (
            not finite
            or not all(math.isfinite(x) for x in (roll, pitch, height, confidence))
            or abs(roll) > self.limits.max_roll_rad
            or abs(pitch) > self.limits.max_pitch_rad
            or not self.limits.min_height_m <= height <= self.limits.max_height_m
            or communication_age_s > self.limits.communication_timeout_s
        )
        if hard_fault:
            self.state = SafetyState.SAFE_STOP
            return self.state
        self._low_confidence_s = self._low_confidence_s + dt if confidence < self.limits.min_confidence else 0.0
        if self._low_confidence_s > self.limits.low_confidence_grace_s:
            self.state = SafetyState.DECELERATE
        elif self.state not in (SafetyState.ESTIMATOR_WARMUP, SafetyState.STAND):
            self.state = SafetyState.TUG_FOLLOW
        return self.state
