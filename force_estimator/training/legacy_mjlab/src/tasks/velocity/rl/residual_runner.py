from __future__ import annotations

import hashlib
import json
import os
import torch
from tensordict import TensorDict

from .residual_policy import (
    PHASE_INITIAL,
    PHASE_PULL,
    PHASE_RECOVERY,
    PHASE_RELEASE_ARMED,
    PHASE_REVERSAL,
    EMERGENCY_STAGE_PULL_STABILIZE,
    EMERGENCY_STAGE_RELEASE_BRAKE,
    EMERGENCY_STAGE_REVERSE_CONFIRM,
    EMERGENCY_STAGE_RECOVERY_SETTLE,
    PhaseConditionedResidualGateConfig,
    ResidualGateConfig,
    ResidualMLPModel,
    initial_phase_conditioned_gate_state,
    initial_gate_state,
    update_phase_conditioned_residual_gate,
    update_residual_gate,
    emergency_scaffold_action,
)
from .runner import GuideDogOnPolicyRunner


BASE_CHECKPOINT_SHA256 = "4f5b24931f04701c9bcb20a697be9895ce2deddf10d743533a65375800662a9f"
RESIDUAL_POLICY_VERSION = 1
LEGACY_GATE_MODE = "legacy_force_recovery"
PHASE_GATE_MODE = "phase_conditioned_v1"


class EmergencyHistoryBuffer:
    """Causal per-environment history for the high-force emergency head."""

    def __init__(self, batch_size: int, feature_dim: int = 12, frames: int = 5, device="cpu") -> None:
        if batch_size < 1 or feature_dim < 1 or frames < 1:
            raise ValueError("history dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.frames = int(frames)
        self._history = torch.zeros(batch_size, self.frames, self.feature_dim, device=device)

    def _resize(self, batch_size: int, device, dtype) -> None:
        if self._history.shape[0] == batch_size and self._history.device == device and self._history.dtype == dtype:
            return
        self._history = torch.zeros(batch_size, self.frames, self.feature_dim, device=device, dtype=dtype)

    def reset(self, mask: torch.Tensor, current_features: torch.Tensor) -> None:
        if current_features.ndim != 2 or current_features.shape[1] != self.feature_dim:
            raise ValueError("current_features has the wrong shape")
        self._resize(current_features.shape[0], current_features.device, current_features.dtype)
        reset_mask = mask.to(device=current_features.device, dtype=torch.bool).reshape(-1)
        if reset_mask.shape[0] != current_features.shape[0]:
            raise ValueError("reset mask has the wrong shape")
        self._history[reset_mask] = current_features[reset_mask].unsqueeze(1)

    def push(self, current_features: torch.Tensor) -> None:
        if current_features.ndim != 2 or current_features.shape[1] != self.feature_dim:
            raise ValueError("current_features has the wrong shape")
        self._resize(current_features.shape[0], current_features.device, current_features.dtype)
        self._history = torch.cat((self._history[:, 1:], current_features.detach().unsqueeze(1)), dim=1)

    def features(self) -> torch.Tensor:
        return self._history.reshape(self._history.shape[0], -1).clone()


def strict_emergency_force_gate(true_force_xy: torch.Tensor, threshold_n: float = 50.0) -> torch.Tensor:
    """Return a hard, true-force-only emergency gate (strictly above threshold)."""
    if true_force_xy.ndim != 2 or true_force_xy.shape[-1] != 2:
        raise ValueError("true_force_xy must have shape (N, 2)")
    if threshold_n <= 0.0:
        raise ValueError("threshold_n must be positive")
    return (torch.linalg.vector_norm(true_force_xy, dim=-1, keepdim=True) > threshold_n).to(true_force_xy.dtype)


def compute_static_support_action(
    hold_action: torch.Tensor,
    emergency_features: torch.Tensor,
    contact_force: torch.Tensor,
    *,
    max_delta: float = 0.12,
    gravity_gain: float = 0.10,
    angular_gain: float = 0.04,
) -> torch.Tensor:
    """Apply bounded contact-conditioned posture damping in static protection.

    ``emergency_features`` contains [projected_gravity(3), angular_velocity(3),
    contact_bits(4), force_direction(2)].  The force direction is intentionally
    ignored here: static protection stabilizes the body and never follows it.
    The correction is distributed over currently loaded legs, so a lost foot
    cannot receive the same target as a supporting foot.
    """
    if hold_action.ndim != 2 or hold_action.shape[-1] != 12:
        raise ValueError("hold_action must have shape (N, 12)")
    if emergency_features.ndim != 2 or emergency_features.shape[-1] != 12:
        raise ValueError("emergency_features must have shape (N, 12)")
    if contact_force.ndim != 2 or contact_force.shape[-1] != 4:
        raise ValueError("contact_force must have shape (N, 4)")
    if hold_action.shape[0] != emergency_features.shape[0] or hold_action.shape[0] != contact_force.shape[0]:
        raise ValueError("batch dimensions must match")
    if max_delta < 0.0 or gravity_gain < 0.0 or angular_gain < 0.0:
        raise ValueError("controller gains must be non-negative")
    loaded = contact_force.clamp_min(0.0)
    loaded = loaded / loaded.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    coverage = (contact_force > 1.0e-3).float().mean(dim=1, keepdim=True)
    gravity = emergency_features[:, :3]
    angular = emergency_features[:, 3:6]
    roll = (-gravity[:, 1:2] * gravity_gain - angular[:, 0:1] * angular_gain) * coverage
    pitch = (gravity[:, 0:1] * gravity_gain - angular[:, 1:2] * angular_gain) * coverage
    correction = torch.zeros_like(hold_action)
    for leg in range(4):
        weight = loaded[:, leg:leg + 1]
        correction[:, 3 * leg:3 * leg + 1] = roll * weight
        correction[:, 3 * leg + 1:3 * leg + 2] = pitch * weight
        correction[:, 3 * leg + 2:3 * leg + 3] = -0.5 * (roll + pitch) * weight
    return hold_action + correction.clamp(-max_delta, max_delta)


class EmergencyHistoryBuffer:
    """Causal per-environment history for the high-force emergency head."""

    def __init__(self, batch_size: int, feature_dim: int = 12, frames: int = 5, device="cpu") -> None:
        if batch_size < 1 or feature_dim < 1 or frames < 1:
            raise ValueError("history dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.frames = int(frames)
        self._history = torch.zeros(batch_size, self.frames, self.feature_dim, device=device)

    def _resize(self, batch_size: int, device, dtype) -> None:
        if self._history.shape[0] == batch_size and self._history.device == device and self._history.dtype == dtype:
            return
        self._history = torch.zeros(batch_size, self.frames, self.feature_dim, device=device, dtype=dtype)

    def reset(self, mask: torch.Tensor, current_features: torch.Tensor) -> None:
        if current_features.ndim != 2 or current_features.shape[1] != self.feature_dim:
            raise ValueError("current_features has the wrong shape")
        self._resize(current_features.shape[0], current_features.device, current_features.dtype)
        reset_mask = mask.to(device=current_features.device, dtype=torch.bool).reshape(-1)
        if reset_mask.shape[0] != current_features.shape[0]:
            raise ValueError("reset mask has the wrong shape")
        self._history[reset_mask] = current_features[reset_mask].unsqueeze(1)

    def push(self, current_features: torch.Tensor) -> None:
        if current_features.ndim != 2 or current_features.shape[1] != self.feature_dim:
            raise ValueError("current_features has the wrong shape")
        self._resize(current_features.shape[0], current_features.device, current_features.dtype)
        self._history = torch.cat((self._history[:, 1:], current_features.detach().unsqueeze(1)), dim=1)

    def features(self) -> torch.Tensor:
        return self._history.reshape(self._history.shape[0], -1).clone()


def expected_base_checkpoint_sha256() -> str:
    """Resolve the parent hash explicitly, retaining the historical default."""
    return os.environ.get("FINETUNE_BASE_CHECKPOINT_SHA256", BASE_CHECKPOINT_SHA256)


def residual_gate_checkpoint_fields(
    phase_enabled: bool,
    phase_config: PhaseConditionedResidualGateConfig | None,
) -> dict:
    if phase_enabled:
        if phase_config is None:
            raise ValueError("phase gate mode requires an exact configuration")
        return {
            "residual_gate_mode": PHASE_GATE_MODE,
            "phase_conditioned_gate_config": vars(phase_config),
        }
    return {"residual_gate_mode": LEGACY_GATE_MODE}


def validate_residual_gate_checkpoint(
    checkpoint: dict,
    *,
    phase_enabled: bool,
    phase_config: PhaseConditionedResidualGateConfig | None,
    allow_migration: bool,
    reset_optimizer: bool,
) -> None:
    saved_mode = checkpoint.get("residual_gate_mode", LEGACY_GATE_MODE)
    desired_mode = PHASE_GATE_MODE if phase_enabled else LEGACY_GATE_MODE
    if saved_mode == LEGACY_GATE_MODE and desired_mode == PHASE_GATE_MODE:
        if not allow_migration:
            raise RuntimeError("legacy-to-phase gate migration requires explicit opt-in")
        if not reset_optimizer:
            raise RuntimeError("legacy-to-phase gate migration requires optimizer reset")
        if phase_config is None:
            raise RuntimeError("phase gate migration requires an exact configuration")
        return
    if saved_mode != desired_mode:
        raise RuntimeError(
            f"residual checkpoint gate mode mismatch: {saved_mode!r} != {desired_mode!r}"
        )
    if desired_mode == PHASE_GATE_MODE:
        if phase_config is None:
            raise RuntimeError("phase gate mode requires an exact configuration")
        if checkpoint.get("phase_conditioned_gate_config") != vars(phase_config):
            config_migration = (
                os.environ.get("FINETUNE_ALLOW_GATE_CONFIG_MIGRATION", "0") == "1"
            )
            if not (config_migration and reset_optimizer):
                raise RuntimeError("phase checkpoint gate configuration mismatch")


def module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def configure_residual_optimizer(
    algorithm,
    residual_learning_rate: float = 5e-5,
    critic_learning_rate: float = 5e-5,
) -> None:
    actor = algorithm.actor
    if not isinstance(actor, ResidualMLPModel):
        raise TypeError("residual optimizer requires ResidualMLPModel")
    if residual_learning_rate <= 0.0 or critic_learning_rate <= 0.0:
        raise ValueError("residual and critic learning rates must be positive")
    emergency_lr_raw = os.environ.get("FINETUNE_EMERGENCY_HEAD_LR")
    if emergency_lr_raw is None:
        param_groups = [
            {
                "name": "residual",
                "params": tuple(actor.residual_parameters()),
                "lr": residual_learning_rate,
            }
        ]
    else:
        emergency_lr = float(emergency_lr_raw)
        if emergency_lr <= 0.0:
            raise ValueError("FINETUNE_EMERGENCY_HEAD_LR must be positive")
        param_groups = [
            {
                "name": "residual",
                "params": (
                    tuple(actor.residual_mlp.parameters())
                    + tuple(actor.velocity_head_mlp.parameters())
                    + tuple(actor.lateral_velocity_head_mlp.parameters())
                ),
                "lr": residual_learning_rate,
            },
            {
                "name": "emergency",
                "params": (
                    tuple(actor.emergency_mlp.parameters())
                    + tuple(actor.emergency_stability_mlp.parameters())
                    + tuple(actor.emergency_virtual_mlp.parameters())
                    + tuple(actor.emergency_virtual_reverse_mlp.parameters())
                    + tuple(actor.emergency_virtual_recovery_mlp.parameters())
                    + tuple(actor.emergency_reverse_mlp.parameters())
                    + tuple(actor.emergency_recovery_mlp.parameters())
                    + tuple(actor.emergency_lateral_mlp.parameters())
                    + tuple(actor.emergency_support_mlp.parameters())
                    + tuple(actor.emergency_support_y_pos_mlp.parameters())
                    + tuple(actor.emergency_support_y_neg_mlp.parameters())
                    + tuple(actor.emergency_contact_adapter_mlp.parameters())
                    + tuple(actor.emergency_joint_adapter_mlp.parameters())
                ),
                "lr": emergency_lr,
            },
        ]
    if os.environ.get("FINETUNE_UPDATE_BASE_ACTOR", "0") == "1":
        base_lr = float(os.environ.get("FINETUNE_BASE_ACTOR_LR", "1e-5"))
        if base_lr <= 0.0:
            raise ValueError("FINETUNE_BASE_ACTOR_LR must be positive")
        base_params = tuple(
            parameter for parameter in actor.mlp.parameters() if parameter.requires_grad
        )
        if base_params:
            param_groups.append({"name": "base", "params": base_params, "lr": base_lr})
    param_groups.append(
        {
            "name": "critic",
            "params": tuple(algorithm.critic.parameters()),
            "lr": critic_learning_rate,
        }
    )
    algorithm.optimizer = torch.optim.Adam(param_groups)
    algorithm.learning_rate = residual_learning_rate
    algorithm.schedule = "fixed"


def residual_learning_rates() -> tuple[float, float]:
    return (
        float(os.environ.get("FINETUNE_RESIDUAL_HEAD_LR", "5e-5")),
        float(os.environ.get("FINETUNE_LR", "5e-5")),
    )


def emergency_stabilizer_target(features: torch.Tensor) -> torch.Tensor:
    """Build a bounded posture/ang-rate recovery target for the emergency head.

    ``features`` is [gravity xyz, angular velocity xyz, four contacts,
    horizontal direction xy].  The target intentionally ignores direction,
    command, phase, and force magnitude; it is a conservative symmetric
    posture correction used only while the >50 N emergency gate is active.
    """
    if features.ndim != 2 or features.shape[-1] not in (12, 60):
        raise ValueError("emergency features must have shape (N, 12) or (N, 60)")
    current = features[:, -12:]
    roll = -(0.30 * current[:, 1] + 0.10 * current[:, 4])
    pitch = -(0.30 * current[:, 0] + 0.10 * current[:, 3])
    target = torch.zeros(
        (features.shape[0], 12), device=features.device, dtype=features.dtype
    )
    # Leg order is [FL, FR, RL, RR], each with [hip, thigh, calf].
    left = torch.tensor([0, 6], device=features.device)
    right = torch.tensor([3, 9], device=features.device)
    front = torch.tensor([1, 2, 4, 5], device=features.device)
    rear = torch.tensor([7, 8, 10, 11], device=features.device)
    target[:, left] = roll.unsqueeze(1)
    target[:, right] = -roll.unsqueeze(1)
    target[:, front] = pitch.unsqueeze(1)
    target[:, rear] = -pitch.unsqueeze(1)
    return target.clamp(-0.25, 0.25)


class GuideDogResidualRunner(GuideDogOnPolicyRunner):
    """GuideDog runner that trains only a bounded residual and the critic."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        actor = self.alg.get_policy()
        if not isinstance(actor, ResidualMLPModel):
            raise TypeError("GuideDogResidualRunner requires ResidualMLPModel")
        actor.freeze_base()
        self.force_estimator.requires_grad_(False).eval()
        # Acceptance and live tether control both expose the privileged
        # tether event on the environment.  Bind it once so the hard-force
        # emergency gate can react to the true manual force on the same
        # control frame instead of waiting for estimator latency.
        self._tether_event = None
        try:
            self._tether_event = self.env.unwrapped.event_manager.get_term_cfg(
                "tether_pull"
            ).func
        except (AttributeError, KeyError):
            # Fail closed: estimator-only gating remains available when a
            # deployment environment intentionally omits the tether event.
            self._tether_event = None
        self.residual_gate_config = ResidualGateConfig(
            step_dt=float(self.env.unwrapped.step_dt)
        )
        self.residual_gate_state = initial_gate_state(
            self.env.num_envs, device=self.device
        )
        self._phase_gate_enabled = (
            os.environ.get("FINETUNE_PHASE_CONDITIONED_RESIDUAL_GATE", "0") == "1"
        )
        # Keep the pull/reversal gate fail-closed by default, but allow a
        # stage-specific force band to confirm reversal during lower-force
        # recovery training.  The thresholds are recorded in the checkpoint
        # configuration, so a changed band cannot silently resume training.
        self.phase_gate_config = PhaseConditionedResidualGateConfig(
            force_start_n=float(os.environ.get("FINETUNE_PHASE_GATE_FORCE_START_N", "45.0")),
            force_full_n=float(os.environ.get("FINETUNE_PHASE_GATE_FORCE_FULL_N", "50.0")),
            pull_dwell_seconds=float(os.environ.get("FINETUNE_PHASE_GATE_PULL_DWELL_S", "0.20")),
            release_dwell_seconds=float(os.environ.get("FINETUNE_PHASE_GATE_RELEASE_DWELL_S", "0.15")),
            release_arm_force_n=float(os.environ.get("FINETUNE_PHASE_GATE_RELEASE_ARM_N", "43.0")),
            release_cancel_force_n=float(os.environ.get("FINETUNE_PHASE_GATE_RELEASE_CANCEL_N", "46.0")),
            reversal_dwell_seconds=float(os.environ.get("FINETUNE_PHASE_GATE_REVERSAL_DWELL_S", "0.20")),
            step_dt=float(self.env.unwrapped.step_dt),
        )
        self.phase_gate_state = initial_phase_conditioned_gate_state(
            self.env.num_envs, device=self.device
        )
        # Privileged tether force is used only as a fail-closed phase lock:
        # it may suppress a stale residual during the original pull/release,
        # but never activates the residual branch.  Keeping the previous
        # force lets us distinguish the first release from final recovery.
        self._last_true_force_xy = torch.zeros(
            (self.env.num_envs, 2), device=self.device
        )
        self._true_pull_direction_xy = torch.zeros(
            (self.env.num_envs, 2), device=self.device
        )
        self._phase_alignment_trace: list[dict] = []
        self._hard_force_mask = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.bool
        )
        self._emergency_stage = torch.full(
            (self.env.num_envs,), EMERGENCY_STAGE_PULL_STABILIZE,
            device=self.device, dtype=torch.long
        )
        self._emergency_prev_hard = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.bool
        )
        self._emergency_prev_direction = torch.zeros(
            (self.env.num_envs, 2), device=self.device
        )
        self._emergency_pull_direction = torch.zeros(
            (self.env.num_envs, 2), device=self.device
        )
        self._emergency_reverse_steps = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.long
        )
        self._last_safe_actions: torch.Tensor | None = None
        self._emergency_confirmed_direction = torch.zeros(
            (self.env.num_envs, 2), device=self.device
        )
        self._emergency_pending_direction = torch.zeros(
            (self.env.num_envs, 2), device=self.device
        )
        self._emergency_pending_count = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.long
        )
        self._emergency_history_enabled = os.environ.get("FINETUNE_EMERGENCY_HISTORY", "0") == "1"
        self._emergency_history = (
            EmergencyHistoryBuffer(self.env.num_envs, feature_dim=12, frames=5, device=self.device)
            if self._emergency_history_enabled
            else None
        )
        self._emergency_history_initialized = False
        self._residual_constraint_batches: list[tuple[torch.Tensor, ...]] = []
        self._behavior_clone_weight = float(os.environ.get("FINETUNE_ACTION_BC_WEIGHT", "0.05"))
        self._residual_l2_weight = float(os.environ.get("FINETUNE_RESIDUAL_L2_WEIGHT", "0.01"))
        residual_lr, critic_lr = residual_learning_rates()
        configure_residual_optimizer(
            self.alg,
            residual_learning_rate=residual_lr,
            critic_learning_rate=critic_lr,
        )
        self._frozen_force_estimator_sha256 = module_sha256(self.force_estimator)

    def _update_emergency_stage(
        self,
        true_force_xy: torch.Tensor,
        hard_force: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update a true-force-driven emergency action FSM.

        The FSM is internal control metadata.  It does not enter the learned
        feature vector, but prevents a stale/low force estimate from keeping
        the controller in pull-only mode during a physical reversal.
        """
        batch = true_force_xy.shape[0]
        device, dtype = true_force_xy.device, true_force_xy.dtype
        if not hasattr(self, "_emergency_prev_hard") or self._emergency_prev_hard.shape[0] != batch:
            self._emergency_prev_hard = torch.zeros(batch, device=device, dtype=torch.bool)
            self._emergency_prev_direction = torch.zeros((batch, 2), device=device, dtype=dtype)
            self._emergency_pull_direction = torch.zeros((batch, 2), device=device, dtype=dtype)
            self._emergency_reverse_steps = torch.zeros(batch, device=device, dtype=torch.long)
            self._emergency_stage = torch.full((batch,), EMERGENCY_STAGE_PULL_STABILIZE, device=device, dtype=torch.long)
        hard = hard_force.reshape(-1).to(device=device, dtype=torch.bool)
        prev_hard = self._emergency_prev_hard[:batch]
        norm = torch.linalg.vector_norm(true_force_xy, dim=-1)
        direction = true_force_xy / norm.unsqueeze(1).clamp_min(1.0e-6)
        prev_direction = self._emergency_prev_direction[:batch]
        pull_direction = self._emergency_pull_direction[:batch]
        stage = self._emergency_stage[:batch]
        reverse_steps = self._emergency_reverse_steps[:batch]
        if reset_mask is not None:
            reset = reset_mask.reshape(-1)[:batch].to(device=device, dtype=torch.bool)
            prev_hard = torch.where(reset, torch.zeros_like(prev_hard), prev_hard)
            direction = torch.where(reset.unsqueeze(1), torch.zeros_like(direction), direction)
            pull_direction = torch.where(reset.unsqueeze(1), torch.zeros_like(pull_direction), pull_direction)
            stage = torch.where(reset, torch.full_like(stage, EMERGENCY_STAGE_PULL_STABILIZE), stage)
            reverse_steps = torch.where(reset, torch.zeros_like(reverse_steps), reverse_steps)
        new_pull = hard & ~prev_hard
        has_pull = torch.linalg.vector_norm(pull_direction, dim=-1) > 1.0e-6
        same_as_pull = torch.sum(direction * pull_direction, dim=-1) > 0.2
        pull_direction = torch.where(
            (new_pull & (~has_pull | same_as_pull)).unsqueeze(1), direction, pull_direction
        )
        opposite_prev = (
            hard & prev_hard
            & (torch.linalg.vector_norm(direction, dim=-1) > 1.0e-6)
            & (torch.sum(direction * prev_direction, dim=-1) < -0.2)
        )
        opposite_pull = (
            hard & has_pull & (torch.sum(direction * pull_direction, dim=-1) < -0.2)
        )
        reverse_candidate = opposite_prev | opposite_pull
        dwell = max(1, int(os.environ.get("FINETUNE_EMERGENCY_STAGE_DWELL_STEPS", "10")))
        reverse_steps = torch.where(reverse_candidate, reverse_steps + 1, torch.where(hard, torch.zeros_like(reverse_steps), reverse_steps))
        # A direction reversal can happen while the force remains above the
        # hard threshold.  In that case ``new_pull`` is false, so gating the
        # reverse-confirm state on it would skip the confirmation phase and
        # jump directly toward recovery.  Enter reverse-confirm on the first
        # reversing frame, then let the dwell transition promote it to settle.
        stage = torch.where(
            reverse_candidate & (reverse_steps < dwell),
            torch.full_like(stage, EMERGENCY_STAGE_REVERSE_CONFIRM),
            stage,
        )
        stage = torch.where(new_pull & ~opposite_pull, torch.full_like(stage, EMERGENCY_STAGE_PULL_STABILIZE), stage)
        stage = torch.where(~hard & prev_hard, torch.full_like(stage, EMERGENCY_STAGE_RELEASE_BRAKE), stage)
        stage = torch.where(
            hard & (reverse_steps >= dwell),
            torch.full_like(stage, EMERGENCY_STAGE_RECOVERY_SETTLE),
            stage,
        )
        # A non-reversing hard-force continuation remains in pull stabilize;
        # the learned branch cannot open merely because the gate is high.
        stage = torch.where(hard & ~reverse_candidate & (stage == EMERGENCY_STAGE_RELEASE_BRAKE), torch.full_like(stage, EMERGENCY_STAGE_PULL_STABILIZE), stage)
        self._emergency_prev_hard[:batch] = hard.detach()
        self._emergency_prev_direction[:batch] = direction.detach()
        self._emergency_pull_direction[:batch] = pull_direction.detach()
        self._emergency_reverse_steps[:batch] = reverse_steps.detach()
        self._emergency_stage[:batch] = stage.detach()
        return stage

    def _confirm_emergency_direction(
        self,
        direction: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hold a confirmed horizontal direction until a reversal dwells.

        The emergency head receives only the confirmed finite direction.  A
        transient sign flip therefore cannot immediately reverse its joint
        action; the new direction must remain consistent for a configurable
        number of control steps.
        """
        if direction.ndim != 2 or direction.shape[-1] != 2:
            raise ValueError("emergency direction must have shape (N, 2)")
        batch = direction.shape[0]
        device, dtype = direction.device, direction.dtype
        if (
            not hasattr(self, "_emergency_confirmed_direction")
            or self._emergency_confirmed_direction.shape[0] != batch
        ):
            self._emergency_confirmed_direction = torch.zeros((batch, 2), device=device, dtype=dtype)
            self._emergency_pending_direction = torch.zeros((batch, 2), device=device, dtype=dtype)
            self._emergency_pending_count = torch.zeros(batch, device=device, dtype=torch.long)
        confirmed = self._emergency_confirmed_direction[:batch].to(device=device, dtype=dtype)
        pending = self._emergency_pending_direction[:batch].to(device=device, dtype=dtype)
        count = self._emergency_pending_count[:batch].to(device=device)
        if reset_mask is not None:
            reset = reset_mask.reshape(-1)[:batch].to(device=device, dtype=torch.bool)
            confirmed = torch.where(reset.unsqueeze(1), torch.zeros_like(confirmed), confirmed)
            pending = torch.where(reset.unsqueeze(1), torch.zeros_like(pending), pending)
            count = torch.where(reset, torch.zeros_like(count), count)
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        valid = norm.reshape(-1) > 1.0e-6
        candidate = direction / norm.clamp_min(1.0e-6)
        has_confirmed = torch.linalg.vector_norm(confirmed, dim=-1) > 1.0e-6
        same_pending = (
            torch.linalg.vector_norm(pending, dim=-1) > 1.0e-6
        ) & (torch.sum(candidate * pending, dim=-1) > 0.98)
        count = torch.where(valid & same_pending, count + 1, torch.where(valid, torch.ones_like(count), count)
        )
        pending = torch.where((valid & same_pending).unsqueeze(1), pending, candidate)
        confirmed = torch.where((valid & ~has_confirmed).unsqueeze(1), candidate, confirmed)
        dwell = max(1, int(os.environ.get("FINETUNE_EMERGENCY_DIRECTION_DWELL_STEPS", "1")))
        same_confirmed = (
            torch.linalg.vector_norm(confirmed, dim=-1) > 1.0e-6
        ) & (torch.sum(candidate * confirmed, dim=-1) > 0.98)
        switch = valid & has_confirmed & ~same_confirmed & (count >= dwell)
        confirmed = torch.where(switch.unsqueeze(1), candidate, confirmed)
        count = torch.where(switch, torch.zeros_like(count), count)
        self._emergency_confirmed_direction[:batch] = confirmed.detach()
        self._emergency_pending_direction[:batch] = pending.detach()
        self._emergency_pending_count[:batch] = count.detach()
        return confirmed

    def _collect_residual_constraint_examples(self, obs: TensorDict) -> None:
        if not self._phase_gate_enabled or "residual_gate" not in obs:
            return
        self._residual_constraint_batches.append(
            (
                obs["actor"].detach().clone(),
                obs["residual_gate"].detach().clone(),
                obs["emergency_features"].detach().clone(),
                obs.get("emergency_joint_features", torch.zeros(
                    (obs["actor"].shape[0], 40),
                    device=obs["actor"].device,
                    dtype=obs["actor"].dtype,
                )).detach().clone(),
                obs.get("emergency_torque_features", torch.zeros(
                    (obs["actor"].shape[0], 52),
                    device=obs["actor"].device,
                    dtype=obs["actor"].dtype,
                )).detach().clone(),
                obs["emergency_gate"].detach().clone(),
                obs.get("emergency_stage", torch.zeros(
                    obs["actor"].shape[0], device=obs["actor"].device, dtype=torch.long
                )).detach().clone(),
            )
        )

    def _update_residual_constraints(self) -> dict[str, float]:
        if not self._residual_constraint_batches:
            return {
                "residual_constraint_loss": 0.0,
                "residual_bc_loss": 0.0,
                "emergency_bc_loss": 0.0,
                "emergency_active_fraction": 0.0,
                "emergency_lateral_bc_loss": 0.0,
            }
        actor = self.alg.get_policy()
        observations = torch.cat([item[0] for item in self._residual_constraint_batches])
        gates = torch.cat([item[1] for item in self._residual_constraint_batches])
        emergency_features = torch.cat(
            [item[2] for item in self._residual_constraint_batches]
        )
        emergency_joint_features = torch.cat(
            [item[3] for item in self._residual_constraint_batches]
        )
        emergency_torque_features = torch.cat(
            [item[4] for item in self._residual_constraint_batches]
        )
        emergency_gates = torch.cat(
            [item[5] for item in self._residual_constraint_batches]
        )
        emergency_stages = torch.cat(
            [item[6] for item in self._residual_constraint_batches]
        )
        self._residual_constraint_batches.clear()
        batch = TensorDict(
            {
                "actor": observations,
                "residual_gate": gates,
                "emergency_features": emergency_features,
                "emergency_joint_features": emergency_joint_features,
                "emergency_torque_features": emergency_torque_features,
                "emergency_gate": emergency_gates,
                "emergency_stage": emergency_stages,
            },
            batch_size=[observations.shape[0]],
        )
        base_action, combined_action, residual_action = actor.deterministic_action_components(batch)
        active = gates.reshape(-1) > 1.0e-6
        losses = []
        bc_loss = torch.zeros((), device=observations.device)
        residual_loss = torch.zeros((), device=observations.device)
        emergency_bc_loss = torch.zeros((), device=observations.device)
        lateral_bc_loss = torch.zeros((), device=observations.device)
        if torch.any(active):
            action_delta = combined_action[active] - base_action[active]
            bc_loss = torch.mean(torch.square(action_delta))
            residual_loss = torch.mean(torch.square(residual_action[active]))
            losses.extend(
                [self._behavior_clone_weight * bc_loss, self._residual_l2_weight * residual_loss]
            )
        emergency_active = emergency_gates.reshape(-1) > 1.0e-6
        emergency_active_fraction = float(emergency_active.float().mean().detach())
        if torch.any(emergency_active):
            emergency_action = (
                combined_action - base_action - gates * residual_action
            )
            target = emergency_scaffold_action(
                emergency_features,
                emergency_stages,
                limit=actor.emergency_limit,
            )
            emergency_bc_loss = torch.mean(
                torch.square(emergency_action[emergency_active] - target[emergency_active])
            )
            emergency_weight = float(
                os.environ.get("FINETUNE_EMERGENCY_BC_WEIGHT", "1.0")
            )
            losses.append(emergency_weight * emergency_bc_loss)
            # Train the dedicated lateral head with a bounded direction-only
            # target. This is a training constraint, not an extra runtime
            # input: the head still consumes only the allowed history.
            latest = emergency_features[:, -12:]
            lateral_pred = actor.emergency_limit * torch.tanh(
                actor.emergency_lateral_mlp(emergency_features)
            ).squeeze(-1)
            lateral_sign = float(os.environ.get("FINETUNE_EMERGENCY_LATERAL_SIGN", "-1.0"))
            lateral_gain = float(os.environ.get("FINETUNE_EMERGENCY_LATERAL_BC_GAIN", "0.50"))
            lateral_target = (
                lateral_sign * lateral_gain * actor.emergency_limit * latest[:, 11]
            ).clamp(-actor.emergency_limit, actor.emergency_limit)
            lateral_bc_loss = torch.mean(
                torch.square(lateral_pred[emergency_active] - lateral_target[emergency_active])
            )
            lateral_weight = float(
                os.environ.get("FINETUNE_EMERGENCY_LATERAL_BC_WEIGHT", "2.0")
            )
            losses.append(lateral_weight * lateral_bc_loss)
        if not losses:
            return {
                "residual_constraint_loss": 0.0,
                "residual_bc_loss": 0.0,
                "emergency_bc_loss": 0.0,
                "emergency_active_fraction": emergency_active_fraction,
                "emergency_lateral_bc_loss": float(lateral_bc_loss.detach()),
            }
        loss = sum(losses)
        self.alg.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuple(actor.residual_parameters()), 1.0)
        self.alg.optimizer.step()
        return {
            "residual_constraint_loss": float(loss.detach()),
            "residual_bc_loss": float(bc_loss.detach()),
            "emergency_bc_loss": float(emergency_bc_loss.detach()),
            "emergency_active_fraction": emergency_active_fraction,
            "emergency_lateral_bc_loss": float(lateral_bc_loss.detach()),
        }

    def _augment_policy_observation(
        self,
        obs: TensorDict,
        estimated_force: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> TensorDict:
        batch_size = obs["actor"].shape[0]
        # The emergency head is deliberately decoupled from force-following
        # context.  It sees only orientation, angular velocity, foot contact,
        # and a bounded horizontal direction estimated by the force estimator.
        # No command, phase, force magnitude, or actor velocity command enters
        # this feature vector.
        actor_for_emergency = obs["actor"]
        force_direction = estimated_force[:, :2]
        force_direction = force_direction / torch.linalg.vector_norm(
            force_direction, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        emergency_features = torch.cat(
            (
                actor_for_emergency[:, 9:12],
                actor_for_emergency[:, 6:9],
                # Actor layout is: ... joint_vel[30:42], last_action[42:54],
                # foot_contact[54:58], foot_contact_forces[58:62],
                # estimated_force[62:65].  Contact must come from the
                # privileged/filtered contact fields, not last_action: the
                # recovery controller uses this four-bit support mask to
                # reallocate its brace when a stance leg is lost.
                actor_for_emergency[:, 54:58],
                force_direction,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)
        obs["emergency_features"] = emergency_features
        self._last_static_support_features = emergency_features.detach()
        self._last_static_support_contact_force = actor_for_emergency[:, 58:62].detach()
        # Joint-level recovery inputs are all robot-observable.  Keep this
        # branch separate from the legacy 60-D history so old checkpoints can
        # load unchanged and the new adapter can remain zero-initialized.
        obs["emergency_joint_features"] = torch.cat(
            (
                actor_for_emergency[:, 18:30],
                actor_for_emergency[:, 30:42],
                actor_for_emergency[:, 42:54],
                actor_for_emergency[:, 58:62],
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)
        env = getattr(self, "env", None)
        robot = None
        try:
            robot = env.unwrapped.scene["robot"] if env is not None else None
        except (AttributeError, KeyError, TypeError):
            robot = None
        robot_data = getattr(robot, "data", None)
        actuator_force = getattr(robot_data, "actuator_force", None)
        if actuator_force is None:
            torque = torch.zeros(
                (actor_for_emergency.shape[0], 12),
                device=actor_for_emergency.device,
                dtype=actor_for_emergency.dtype,
            )
        else:
            torque = actuator_force.reshape(actuator_force.shape[0], -1).to(actor_for_emergency)
        if torque.shape[-1] == 12:
            torque = torque[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
        obs["emergency_torque_features"] = torch.cat(
            (
                actor_for_emergency[:, 18:30],
                actor_for_emergency[:, 30:42],
                actor_for_emergency[:, 42:54],
                actor_for_emergency[:, 58:62],
                torque,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)
        obs["emergency_gate"] = torch.zeros(
            (batch_size, 1), device=obs["actor"].device, dtype=obs["actor"].dtype
        )
        obs["emergency_stage"] = torch.full(
            (batch_size,), EMERGENCY_STAGE_PULL_STABILIZE,
            device=obs["actor"].device, dtype=torch.long
        )
        horizontal_speed = torch.linalg.vector_norm(
            obs["actor"][:, 3:5], dim=-1, keepdim=True
        )
        if getattr(self, "_phase_gate_enabled", False):
            if self.phase_gate_state.phase.shape[0] != batch_size:
                self.phase_gate_state = initial_phase_conditioned_gate_state(
                    batch_size,
                    device=obs["actor"].device,
                    dtype=obs["actor"].dtype,
                )
            if (
                not hasattr(self, "_last_true_force_xy")
                or self._last_true_force_xy.shape[0] != batch_size
            ):
                self._last_true_force_xy = torch.zeros(
                    (batch_size, 2),
                    device=obs["actor"].device,
                    dtype=obs["actor"].dtype,
                )
            if (
                not hasattr(self, "_true_pull_direction_xy")
                or self._true_pull_direction_xy.shape[0] != batch_size
            ):
                self._true_pull_direction_xy = torch.zeros(
                    (batch_size, 2),
                    device=obs["actor"].device,
                    dtype=obs["actor"].dtype,
                )
            # The estimator is intentionally the only signal allowed to open
            # the gate.  A stale estimator can nevertheless carry a reversal
            # state across the first release; use the tether's privileged
            # force only to reset that state back to base-only.  Same-direction
            # pull while in reversal/recovery, or the release immediately
            # following that pull, is always forced back to pull/initial.
            true_force_xy = None
            tether_event = getattr(self, "_tether_event", None)
            manual_tether = bool(getattr(tether_event, "manual_enabled", False))
            if tether_event is not None:
                # Acceptance drives a manual force immediately before policy
                # evaluation; use that pending value for the emergency cut so
                # the first high-force frame cannot slip through one cycle.
                if getattr(tether_event, "manual_enabled", False):
                    candidate = getattr(tether_event, "manual_force_b", None)
                else:
                    candidate = getattr(tether_event, "last_force_b", None)
                if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
                    true_force_xy = candidate[:batch_size, :2].to(
                        device=obs["actor"].device, dtype=obs["actor"].dtype
                    )
            if true_force_xy is None or true_force_xy.shape != (batch_size, 2):
                true_force_xy = torch.zeros(
                    (batch_size, 2), device=obs["actor"].device, dtype=obs["actor"].dtype
                )
            if not manual_tether:
                # `last_force_b` is simulator-only privileged state.  In live
                # operation, phase/recovery state must be driven by the same
                # calibrated estimator available to the robot, never by the
                # simulator's true force channel.
                true_force_xy = estimated_force[:, :2].detach()
            # The tether event has two historical producers: the simulator
            # publishes normalized force (N/80), while the manual acceptance
            # adapter may expose newtons.  The phase FSM thresholds below are
            # explicitly in newtons.  Normalize this boundary once, using a
            # conservative magnitude test; keeping the rest of the FSM in N
            # prevents a 60 N event from being interpreted as 0.75 N and
            # stuck in INITIAL/PULL forever.
            raw_true_norm = torch.linalg.vector_norm(true_force_xy, dim=-1, keepdim=True)
            normalized_input = raw_true_norm <= 2.0
            true_force_xy = torch.where(
                normalized_input.expand(-1, 2),
                true_force_xy * 80.0,
                true_force_xy,
            )
            previous_true_force_xy = self._last_true_force_xy[:batch_size].to(
                device=obs["actor"].device, dtype=obs["actor"].dtype
            )
            true_norm = torch.linalg.vector_norm(true_force_xy, dim=-1, keepdim=True)
            previous_norm = torch.linalg.vector_norm(
                previous_true_force_xy, dim=-1, keepdim=True
            )
            current_direction = true_force_xy / true_norm.clamp_min(1.0e-6)
            previous_direction = previous_true_force_xy / previous_norm.clamp_min(1.0e-6)
            reset_bool = (
                reset_mask.to(device=obs["actor"].device, dtype=torch.bool).reshape(-1)
                if reset_mask is not None
                else torch.zeros(
                    batch_size, device=obs["actor"].device, dtype=torch.bool
                )
            )
            true_pull_direction = self._true_pull_direction_xy[:batch_size].to(
                device=obs["actor"].device, dtype=obs["actor"].dtype
            )
            true_pull_direction = torch.where(
                reset_bool.unsqueeze(1),
                torch.zeros_like(true_pull_direction),
                true_pull_direction,
            )
            capture_pull = (
                (self.phase_gate_state.phase.reshape(-1) == PHASE_INITIAL)
                & (true_norm.reshape(-1) >= self.phase_gate_config.force_start_n)
            )
            true_pull_direction = torch.where(
                capture_pull.unsqueeze(1), current_direction, true_pull_direction
            )
            current_same_pull = (
                (true_norm >= self.phase_gate_config.force_start_n)
                & (torch.sum(current_direction * true_pull_direction, dim=1, keepdim=True) > 0.2)
            )
            # Keep reversal confirmation anchored to the physical pull
            # direction.  This only strengthens the base-only pull guard; it
            # cannot create a residual gate.
            self.phase_gate_state.initial_direction_xy = torch.where(
                current_same_pull.expand(-1, 2),
                true_pull_direction,
                self.phase_gate_state.initial_direction_xy,
            )
            first_release_after_pull = (
                (true_norm < self.phase_gate_config.release_arm_force_n)
                & (previous_norm >= self.phase_gate_config.force_start_n)
                & (torch.sum(previous_direction * true_pull_direction, dim=1, keepdim=True) > 0.2)
            )
            final_release_after_reverse = (
                (true_norm < self.phase_gate_config.release_arm_force_n)
                & (previous_norm >= self.phase_gate_config.force_start_n)
                & (torch.sum(previous_direction * true_pull_direction, dim=1, keepdim=True) < -0.2)
                & (self.phase_gate_state.phase == PHASE_REVERSAL)
            )
            stale_pull_lock = (
                (self.phase_gate_state.phase == PHASE_REVERSAL)
                | (self.phase_gate_state.phase == PHASE_RECOVERY)
            ) & (current_same_pull | first_release_after_pull)
            safe_reset = stale_pull_lock.reshape(-1)
            safe_reset = safe_reset | reset_bool
            gate_force_xy = estimated_force[:, :2] * 80.0
            # On a confirmed reverse, a true-force release is a fail-closed
            # transition into recovery.  It may close a stale estimator state,
            # but it cannot open the residual branch.
            gate_force_xy = torch.where(
                final_release_after_reverse.expand(-1, 2),
                torch.zeros_like(gate_force_xy),
                gate_force_xy,
            )
            phase_before_update = self.phase_gate_state.phase.clone()
            gate, self.phase_gate_state = update_phase_conditioned_residual_gate(
                gate_force_xy,
                horizontal_speed,
                self.phase_gate_state,
                self.phase_gate_config,
                reset_mask=safe_reset,
            )
            # Keep the transition frame base-only.  The residual opens only
            # on the following frame after reversal dwell confirmation.
            pre_reversal_phase = phase_before_update == PHASE_RELEASE_ARMED
            gate = torch.where(pre_reversal_phase, torch.zeros_like(gate), gate)
            # The physical tether phase is authoritative for the hard
            # pull/release isolation contract.  Estimator hysteresis may
            # retain a reversal state across a new cycle; it must not open
            # the residual branch during pull, release, or reverse dwell.
            physical_phase = getattr(tether_event, "_gate_phase", None)
            # Acceptance and live tether FSMs use both labels: the manual
            # trace exposes the physical phase as ``reverse`` while the
            # dwell state uses ``reverse_dwell``.  Treat both as the
            # transition-protected interval so residual authority cannot
            # open before zero-crossing confirmation.
            if physical_phase in {"release", "reverse", "reverse_dwell"}:
                gate = torch.zeros_like(gate)
            # Optional speed-calibration branch: expose the residual/velocity
            # head during the confirmed 15--45 N pull phase so it can learn
            # velocity tracking without changing the frozen base gait.  It is
            # strictly opt-in and hard-clamped below the 50 N protection
            # threshold; release/reverse remain base-only.
            if os.environ.get("FINETUNE_PULL_RESIDUAL_ENABLE", "0") == "1":
                pull_min_n = float(os.environ.get("FINETUNE_PULL_RESIDUAL_MIN_N", "15"))
                pull_max_n = float(os.environ.get("FINETUNE_PULL_RESIDUAL_MAX_N", "45"))
                pull_gate_scale = float(os.environ.get("FINETUNE_PULL_RESIDUAL_GATE_SCALE", "1.0"))
                if not (0.0 <= pull_gate_scale <= 1.0):
                    raise ValueError("FINETUNE_PULL_RESIDUAL_GATE_SCALE must be in [0, 1]")
                pull_force_n = true_norm.reshape(-1)
                pull_phase = (physical_phase is None) or (physical_phase == "pull")
                pull_mask = (
                    pull_phase
                    & (pull_force_n >= pull_min_n)
                    & (pull_force_n <= pull_max_n)
                )
                pull_gate = torch.full_like(gate, pull_gate_scale)
                gate = torch.where(pull_mask.unsqueeze(1), pull_gate, gate)
            # During a confirmed reverse transition, hold the locomotion
            # policy at zero command while the body crosses through zero.
            # This is still within the <=50 N allowance (pull resumes after
            # the transition), but prevents the base policy from chasing a
            # reversed velocity command while support legs are changing.
            if physical_phase in {"reverse", "reverse_dwell"}:
                obs["actor"][:, 13:16] = 0.0
                obs["actor"][:, 16:18] = 0.0
                self._actor_command[:batch_size] = 0.0
                if isinstance(getattr(self, "last_actor_command", None), torch.Tensor):
                    self.last_actor_command[:batch_size] = 0.0
            # Residual runner must enforce the same hard high-force contract
            # as the base runner.  In manual acceptance, the tether force is
            # authoritative: exactly 50 N remains in the 45--50 N follow band
            # and only true force >50 N trips hard protection.  Estimator-only
            # trips are used for live/non-manual operation to cover latency.
            hard_force_n = float(os.environ.get("GUIDE_DOG_HARD_FORCE_N", "50.0"))
            hard_estimate_n = float(
                os.environ.get("GUIDE_DOG_HARD_FORCE_ESTIMATE_N", str(hard_force_n))
            )
            manual_tether = bool(getattr(tether_event, "manual_enabled", False))
            if getattr(self, "_emergency_history_enabled", False):
                # Manual acceptance may use privileged true force for exact
                # threshold validation.  Live/deployment execution must use
                # only the calibrated estimator because true_force is not a
                # robot-available signal.
                hard_force = strict_emergency_force_gate(
                    true_force_xy, hard_force_n
                ).reshape(-1).to(torch.bool) if manual_tether else (
                    torch.linalg.vector_norm(estimated_force[:, :2] * 80.0, dim=1)
                    > hard_estimate_n
                )
                # Targeted training may expose a known high-force band even
                # when the event wrapper does not publish the privileged
                # label to the runner.  This switch is deliberately training
                # only; acceptance/deployment never sets it.
                train_force_min = float(os.environ.get("FINETUNE_FORCE_MIN", "0.0"))
                if (
                    os.environ.get("FINETUNE_EMERGENCY_TRAIN_FORCE_ALWAYS_ON", "0") == "1"
                    and train_force_min >= 50.0
                ):
                    hard_force = torch.ones_like(hard_force)
            elif manual_tether:
                hard_force = true_norm.reshape(-1) > hard_force_n
            else:
                hard_force = torch.linalg.vector_norm(
                    estimated_force[:, :2] * 80.0, dim=1
                ) > hard_estimate_n
            self._hard_force_mask = hard_force.detach()
            # Robot-observable direction for optional static-protection posture
            # shaping.  This is only a direction unit vector; it never feeds
            # a velocity/follow command and is disabled unless explicitly
            # requested by the static-protection smoke.
            force_planar = estimated_force[:, :2]
            self._static_hold_force_unit = force_planar / torch.linalg.vector_norm(
                force_planar, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            stage = self._update_emergency_stage(
                true_force_xy, hard_force, reset_mask=reset_mask
            )
            # Optional low-force reverse recovery probe.  This is disabled by
            # default and only used for the targeted <=50 N reverse smoke:
            # pull-following remains unchanged, while the constrained
            # recovery head can brace during the zero-crossing window.
            low_force_reverse_recovery = (
                os.environ.get("FINETUNE_REVERSE_EMERGENCY_GATE", "0") == "1"
                and physical_phase in {"reverse", "reverse_dwell"}
            )
            if low_force_reverse_recovery:
                reverse_min_n = float(
                    os.environ.get("FINETUNE_REVERSE_EMERGENCY_MIN_N", "45.0")
                )
                low_force_reverse_recovery = (
                    (true_norm.reshape(-1) >= reverse_min_n) & ~hard_force
                )
                stage = torch.where(
                    low_force_reverse_recovery,
                    torch.full_like(stage, EMERGENCY_STAGE_REVERSE_CONFIRM),
                    stage,
                )
            # Targeted high-force training can explicitly balance the stage
            # distribution.  The simulator's default tether event usually
            # holds one direction, so reverse/recovery states otherwise have
            # almost no samples even when the emergency gate is forced on.
            # This override is training-only and never used by acceptance or
            # deployment; unknown values fail closed to the FSM stage.
            train_stage = os.environ.get("FINETUNE_EMERGENCY_TRAIN_STAGE", "")
            if (
                os.environ.get("FINETUNE_EMERGENCY_TRAIN_FORCE_ALWAYS_ON", "0") == "1"
                and train_stage in {"pull", "reverse", "recovery"}
            ):
                forced_stage = {
                    "pull": EMERGENCY_STAGE_PULL_STABILIZE,
                    "reverse": EMERGENCY_STAGE_REVERSE_CONFIRM,
                    "recovery": EMERGENCY_STAGE_RECOVERY_SETTLE,
                }[train_stage]
                stage = torch.where(
                    hard_force,
                    torch.full_like(stage, forced_stage),
                    stage,
                )
            # In manual/privileged acceptance, use the same-frame tether
            # direction for the emergency head.  This is still only a finite
            # horizontal direction feature (no magnitude/command/phase), and
            # removes estimator-direction latency exactly at a force step.
            direction_candidate = force_direction
            if manual_tether:
                true_direction = true_force_xy / true_norm.clamp_min(1.0e-6)
                direction_candidate = torch.where(
                    (true_norm > 1.0e-6), true_direction, emergency_features[:, 10:12]
                ).clamp(-1.0, 1.0)
            confirmed_direction = self._confirm_emergency_direction(
                direction_candidate, reset_mask=reset_mask
            )
            emergency_features[:, 10:12] = confirmed_direction.clamp(-1.0, 1.0)
            obs["emergency_features"] = emergency_features
            if getattr(self, "_emergency_history_enabled", False):
                if self._emergency_history is None:
                    self._emergency_history = EmergencyHistoryBuffer(
                        batch_size, feature_dim=12, frames=5, device=emergency_features.device
                    )
                reset_bool = (
                    reset_mask.to(device=emergency_features.device, dtype=torch.bool).reshape(-1)
                    if reset_mask is not None
                    else torch.zeros(batch_size, device=emergency_features.device, dtype=torch.bool)
                )
                if not self._emergency_history_initialized:
                    reset_bool = torch.ones_like(reset_bool)
                    self._emergency_history_initialized = True
                self._emergency_history.reset(reset_bool, emergency_features)
                if torch.any(~reset_bool):
                    self._emergency_history.push(emergency_features)
                obs["emergency_features"] = self._emergency_history.features()
            # Targeted high-force training has a known force band even when
            # the simulator event does not expose privileged force labels to
            # the runner.  Opt in explicitly and only for a >=55 N run; never
            # leak this shortcut into mixed-force or deployment evaluation.
            force_min = float(os.environ.get("FINETUNE_FORCE_MIN", "0.0"))
            force_max = float(os.environ.get("FINETUNE_FORCE_MAX", "0.0"))
            if (
                not getattr(self, "_emergency_history_enabled", False)
                and os.environ.get("FINETUNE_EMERGENCY_FORCE_ALWAYS_ON", "0") == "1"
                and force_min >= 55.0
                and force_max >= force_min
            ):
                hard_force = torch.ones_like(hard_force)
                self._hard_force_mask = hard_force.detach()
                obs["emergency_gate"] = torch.ones(
                    (batch_size, 1), device=obs["actor"].device, dtype=obs["actor"].dtype
                )
            # High-force protection remains base-only through pull, release,
            # and reversal dwell.  A separately opt-in recovery residual is
            # allowed only after the phase machine has confirmed recovery;
            # this gives the frozen base policy a learnable stabilizing branch
            # without reopening force-following during the dangerous phases.
            recovery_residual_enabled = (
                os.environ.get("FINETUNE_HARD_FORCE_RECOVERY_RESIDUAL", "0") == "1"
            )
            recovery_residual = hard_force & (
                self.phase_gate_state.phase.reshape(-1) == PHASE_RECOVERY
            )
            if os.environ.get("FINETUNE_HARD_FORCE_RECOVERY_RESIDUAL_ALL_PHASES", "0") == "1":
                # Emergency stabilization may be enabled from the hard-force
                # trip itself, but it still receives zero command/phase/force
                # observations and therefore cannot become a follow policy.
                recovery_residual = hard_force.clone()
            if not recovery_residual_enabled:
                recovery_residual = torch.zeros_like(recovery_residual)
            recovery_gate_scale = float(
                os.environ.get("FINETUNE_HARD_FORCE_RECOVERY_RESIDUAL_SCALE", "1.0")
            )
            if not 0.0 <= recovery_gate_scale <= 1.0:
                raise ValueError("FINETUNE_HARD_FORCE_RECOVERY_RESIDUAL_SCALE must be in [0, 1]")
            recovery_gate = (
                recovery_residual.unsqueeze(1).to(gate.dtype) * recovery_gate_scale
            )
            gate = torch.where(hard_force.unsqueeze(1), recovery_gate, gate)
            # Action structure follows the physical phase while the learned
            # emergency head remains force-gated.  The stage is metadata used
            # to shape/limit the action; it is never concatenated to head
            # inputs, preserving the pose/rate/contact/direction contract.
            self._emergency_stage[:batch_size] = stage.detach()
            obs["emergency_stage"] = stage
            # The independent emergency head is the only learned action branch
            # allowed to run above the physical 50 N threshold.  Its gate is
            # based on true force in manual acceptance and estimator/true force
            # in live operation, while its features remain direction/pose/
            # contact-only as constructed above.
            emergency_head_enabled = os.environ.get(
                "FINETUNE_EMERGENCY_HEAD", "0"
            ) == "1"
            emergency_active = hard_force | low_force_reverse_recovery
            obs["emergency_gate"] = (
                emergency_active.unsqueeze(1).to(gate.dtype)
                if emergency_head_enabled
                else torch.zeros_like(gate)
            )
            if torch.any(hard_force):
                safe_actor = obs["actor"].clone()
                if os.environ.get("GUIDE_DOG_HARD_FORCE_KEEP_COMMAND", "0") != "1":
                    safe_actor[hard_force, 13:16] = 0.0
                if os.environ.get("GUIDE_DOG_HARD_FORCE_KEEP_PHASE", "0") != "1":
                    safe_actor[hard_force, 16:18] = 0.0
                # True base-only protection: remove the high-force estimator
                # channel as well as command/phase, so the frozen base policy
                # cannot react to a >50N event through a stale force feature.
                safe_actor[hard_force, -3:] = 0.0
                obs["actor"] = safe_actor
                # Keep diagnostics and deployment-side command consumers
                # synchronized with the protected actor observation.  The
                # parent runner records last_actor_command before this
                # residual gate runs; leaving it unchanged made the trace
                # report a stale follow command for one frame even though
                # the actor and environment command were already zeroed.
                if isinstance(getattr(self, "last_actor_command", None), torch.Tensor):
                    self.last_actor_command[hard_force] = 0.0
                safe_critic = obs["critic"].clone()
                safe_critic[hard_force, : self.actor_obs_dim] = safe_actor[hard_force]
                obs["critic"] = safe_critic
                self._actor_command[hard_force] = 0.0
            self._last_true_force_xy[:batch_size] = true_force_xy.detach()
            self._true_pull_direction_xy[:batch_size] = true_pull_direction.detach()
            self._accumulate_phase_gate_metrics(gate)
            obs["residual_gate"] = gate
            self._last_emergency_features = obs["emergency_features"].detach().clone()
            self._last_emergency_stage = obs["emergency_stage"].detach().clone()
            record_alignment = getattr(self, "_record_phase_alignment", None)
            if record_alignment is not None:
                record_alignment(
                    estimated_force * 80.0,
                    true_force_xy,
                    gate,
                    obs["emergency_gate"],
                )
            return obs
        if self.residual_gate_state.value.shape[0] != batch_size:
            self.residual_gate_state = initial_gate_state(
                batch_size,
                device=obs["actor"].device,
                dtype=obs["actor"].dtype,
            )
        gate, self.residual_gate_state = update_residual_gate(
            estimated_force[:, :2] * 80.0,
            horizontal_speed,
            self.residual_gate_state,
            self.residual_gate_config,
            reset_mask=reset_mask,
        )
        obs["residual_gate"] = gate
        self._last_emergency_features = obs["emergency_features"].detach().clone()
        self._last_emergency_stage = obs.get(
            "emergency_stage",
            torch.zeros(batch_size, device=obs["actor"].device, dtype=torch.long),
        ).detach().clone()
        record_alignment = getattr(self, "_record_phase_alignment", None)
        if record_alignment is not None:
            record_alignment(
                estimated_force * 80.0,
                torch.zeros_like(estimated_force),
                gate,
                obs["emergency_gate"],
            )
        return obs

    def _sanitize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Hold the last pre-trip joint action during a hard-force event.

        Base-only observation isolation prevents the residual branch from
        following the tether, but replacing the action with the newly sampled
        zero-force policy output can itself create a one-step posture jump.
        Holding the previous action is a bounded emergency stabilization mode;
        it does not use force direction or command-following information.
        """
        if self._last_safe_actions is None or self._last_safe_actions.shape != actions.shape:
            self._last_safe_actions = actions.detach().clone()
            return actions
        mask = getattr(self, "_hard_force_mask", None)
        if isinstance(mask, torch.Tensor) and mask.numel() == actions.shape[0]:
            mask = mask.to(device=actions.device, dtype=torch.bool)
            if os.environ.get("FINETUNE_EMERGENCY_HEAD", "0") == "1":
                # The independent emergency head must receive its own action
                # during high-force training; do not replace it with the
                # previous base action as the legacy hold-last safeguard did.
                blend = float(os.environ.get("FINETUNE_EMERGENCY_ACTION_BLEND", "1.0"))
                if not 0.0 <= blend <= 1.0:
                    raise ValueError("FINETUNE_EMERGENCY_ACTION_BLEND must be in [0, 1]")
                previous = self._last_safe_actions
                safe_actions = blend * actions + (1.0 - blend) * previous
            else:
                safe_actions = torch.where(
                    mask.unsqueeze(1), self._last_safe_actions, actions
                )
        else:
            safe_actions = actions
        static_hold_spec = os.environ.get("FINETUNE_STATIC_HOLD_ACTION", "").strip()
        if static_hold_spec and isinstance(mask, torch.Tensor) and mask.numel() == actions.shape[0]:
            try:
                values = [float(item.strip()) for item in static_hold_spec.split(",")]
            except ValueError as exc:
                raise ValueError("FINETUNE_STATIC_HOLD_ACTION must be 12 comma-separated floats") from exc
            if len(values) != actions.shape[-1]:
                raise ValueError("FINETUNE_STATIC_HOLD_ACTION must contain 12 values")
            hold = torch.tensor(values, device=actions.device, dtype=actions.dtype).unsqueeze(0)
            lean_gain = float(os.environ.get("FINETUNE_STATIC_HOLD_FORCE_LEAN", "0.0"))
            force_unit = getattr(self, "_static_hold_force_unit", None)
            if lean_gain != 0.0 and isinstance(force_unit, torch.Tensor):
                # Hip-roll and thigh groups are ordered in the policy as
                # [hip, thigh, calf] per leg.  The correction is bounded and
                # direction-only; it is not a follow command.
                hold = hold.expand(actions.shape[0], -1).clone()
                y = force_unit[:, 1].to(device=actions.device, dtype=actions.dtype)
                x = force_unit[:, 0].to(device=actions.device, dtype=actions.dtype)
                for index in (0, 3, 6, 9):
                    hold[:, index] = (hold[:, index] - lean_gain * y).clamp(-1.0, 1.0)
                for index in (1, 4, 7, 10):
                    hold[:, index] = (hold[:, index] - 0.5 * lean_gain * x).clamp(-1.0, 1.0)
            else:
                hold = hold.expand(actions.shape[0], -1)
            if os.environ.get("FINETUNE_STATIC_SUPPORT_CONTROLLER", "0") == "1":
                features = getattr(self, "_last_static_support_features", None)
                contact_force = getattr(self, "_last_static_support_contact_force", None)
                if isinstance(features, torch.Tensor) and isinstance(contact_force, torch.Tensor):
                    support_action = compute_static_support_action(
                        hold,
                        features.to(device=actions.device, dtype=actions.dtype),
                        contact_force.to(device=actions.device, dtype=actions.dtype),
                        max_delta=float(os.environ.get("FINETUNE_STATIC_SUPPORT_MAX_DELTA", "0.12")),
                        gravity_gain=float(os.environ.get("FINETUNE_STATIC_SUPPORT_GRAVITY_GAIN", "0.10")),
                        angular_gain=float(os.environ.get("FINETUNE_STATIC_SUPPORT_ANGULAR_GAIN", "0.04")),
                    )
                    hold = torch.where(mask.unsqueeze(1), support_action, hold)
            safe_actions = torch.where(mask.unsqueeze(1), hold, safe_actions)
        # Limit one-step emergency joint-position jumps during hard-force
        # reversal/recovery while leaving normal and base-only actions intact.
        if isinstance(mask, torch.Tensor) and mask.numel() == actions.shape[0]:
            slew = float(os.environ.get("FINETUNE_EMERGENCY_ACTION_SLEW", "0.0"))
            if slew < 0.0:
                raise ValueError("FINETUNE_EMERGENCY_ACTION_SLEW must be non-negative")
            if slew > 0.0 and (
                os.environ.get("FINETUNE_EMERGENCY_HEAD", "0") == "1"
                or bool(static_hold_spec)
            ):
                previous = self._last_safe_actions
                limited = previous + (safe_actions - previous).clamp(-slew, slew)
                safe_actions = torch.where(mask.unsqueeze(1), limited, safe_actions)
        self._last_safe_actions = safe_actions.detach().clone()
        return safe_actions

    def _accumulate_phase_gate_metrics(self, gate: torch.Tensor) -> None:
        counts = torch.bincount(
            self.phase_gate_state.phase.reshape(-1), minlength=5
        ).detach().cpu()
        if not hasattr(self, "_phase_metric_counts"):
            self._phase_metric_counts = torch.zeros(5, dtype=torch.int64)
            self._phase_metric_gate_sum = 0.0
            self._phase_metric_gate_max = 0.0
            self._phase_metric_samples = 0
        self._phase_metric_counts += counts
        self._phase_metric_gate_sum += float(gate.detach().sum().cpu())
        self._phase_metric_gate_max = max(
            self._phase_metric_gate_max, float(gate.detach().max().cpu())
        )
        self._phase_metric_samples += gate.numel()

    def _record_phase_alignment(
        self,
        estimated_force_n: torch.Tensor,
    true_force_n: torch.Tensor,
    gate: torch.Tensor,
    emergency_gate: torch.Tensor | None = None,
    ) -> None:
        """Keep deployment-estimate vs privileged-force gate transitions auditable."""
        # Lightweight unit-test/deployment stubs may call this audit hook
        # before phase-gate state is allocated.  Logging must never change
        # control behavior or make those callers fail.
        if hasattr(self, "phase_gate_state") and hasattr(self.phase_gate_state, "phase"):
            phase = self.phase_gate_state.phase.reshape(-1).detach().cpu().tolist()
        else:
            phase = [0] * int(gate.numel())
        estimated = estimated_force_n.detach().cpu().tolist()
        true = true_force_n.detach().cpu().tolist()
        gates = gate.detach().reshape(-1).cpu().tolist()
        if not hasattr(self, "_phase_alignment_trace"):
            self._phase_alignment_trace = []
        self._phase_alignment_trace.extend(
            {
                "estimated_force_n": estimated[index],
                "true_force_n": true[index],
                "phase": int(phase[index]),
                "gate": float(gates[index]),
                "emergency_gate": float(
                    emergency_gate.reshape(-1)[index].detach().cpu()
                ) if emergency_gate is not None else 0.0,
            }
            for index in range(len(gates))
        )
        if len(self._phase_alignment_trace) > 20000:
            del self._phase_alignment_trace[:-20000]

    def _extra_training_metrics(self) -> dict[str, float]:
        phase_names = {
            PHASE_INITIAL: "initial",
            PHASE_PULL: "pull",
            PHASE_RELEASE_ARMED: "release_armed",
            PHASE_REVERSAL: "reversal",
            PHASE_RECOVERY: "recovery",
        }
        counts = getattr(self, "_phase_metric_counts", torch.zeros(5, dtype=torch.int64))
        samples = int(getattr(self, "_phase_metric_samples", 0))
        denominator = float(samples) if samples else 1.0
        metrics = {
            f"phase_gate/{name}_fraction": float(counts[phase_id]) / denominator
            for phase_id, name in phase_names.items()
        }
        metrics["phase_gate/effective_mean"] = (
            float(getattr(self, "_phase_metric_gate_sum", 0.0)) / denominator
        )
        metrics["phase_gate/effective_max"] = float(
            getattr(self, "_phase_metric_gate_max", 0.0)
        )
        self._phase_metric_counts = torch.zeros(5, dtype=torch.int64)
        self._phase_metric_gate_sum = 0.0
        self._phase_metric_gate_max = 0.0
        self._phase_metric_samples = 0
        return metrics

    def _update_force_estimator(self) -> float:
        if os.environ.get("FINETUNE_TRAIN_FORCE_ESTIMATOR", "0") == "1":
            return super()._update_force_estimator()
        self._force_inputs.clear()
        self._force_labels.clear()
        return 0.0

    def _assert_frozen_estimator(self) -> None:
        if os.environ.get("FINETUNE_TRAIN_FORCE_ESTIMATOR", "0") == "1":
            return
        current = module_sha256(self.force_estimator)
        if current != self._frozen_force_estimator_sha256:
            raise RuntimeError("frozen force estimator hash drifted")

    def save(self, path: str, infos=None) -> None:
        self._assert_frozen_estimator()
        checkpoint = self.alg.save()
        estimator_hash = (
            module_sha256(self.force_estimator)
            if os.environ.get("FINETUNE_TRAIN_FORCE_ESTIMATOR", "0") == "1"
            else self._frozen_force_estimator_sha256
        )
        checkpoint.update(
            {
                "iter": self.current_learning_iteration,
                "infos": infos,
                "residual_policy_version": RESIDUAL_POLICY_VERSION,
                "base_checkpoint_sha256": expected_base_checkpoint_sha256(),
                "force_estimator_sha256": estimator_hash,
                "force_estimator_state_dict": self.force_estimator.state_dict(),
                "force_estimator_input_dim": self.force_input_dim,
                "residual_gate_config": vars(self.residual_gate_config),
                "high_force_speed_scale": float(
                    os.environ.get(
                        "FINETUNE_HIGH_FORCE_SPEED_SCALE",
                        os.environ.get("GUIDE_DOG_HIGH_FORCE_SPEED_SCALE", "1.0"),
                    )
                ),
                "frozen_parameter_names": [
                    name
                    for name, parameter in self.alg.actor.named_parameters()
                    if not parameter.requires_grad
                ],
                **residual_gate_checkpoint_fields(
                    self._phase_gate_enabled,
                    self.phase_gate_config if self._phase_gate_enabled else None,
                ),
            }
        )
        torch.save(checkpoint, path)
        self.logger.save_model(path, self.current_learning_iteration)
        trace_path = os.path.join(os.path.dirname(path), "phase_alignment_trace.jsonl")
        with open(trace_path, "w", encoding="utf-8") as trace_file:
            for row in getattr(self, "_phase_alignment_trace", []):
                trace_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        try:
            import wandb
            if wandb.run:
                wandb.save(trace_path, base_path=os.path.dirname(trace_path))
        except (ImportError, AttributeError):
            pass

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        checkpoint = torch.load(
            path, weights_only=False, map_location=map_location or self.device
        )
        actor = self.alg.actor
        is_residual = checkpoint.get("residual_policy_version") == RESIDUAL_POLICY_VERSION
        reset_optimizer = os.environ.get("FINETUNE_RESET_ACTOR_OPTIMIZER", "0") == "1"
        allow_gate_migration = (
            os.environ.get("FINETUNE_ALLOW_GATE_MIGRATION", "0") == "1"
        )
        if is_residual:
            if checkpoint.get("base_checkpoint_sha256") != expected_base_checkpoint_sha256():
                raise RuntimeError("residual checkpoint base SHA-256 mismatch")
            validate_residual_gate_checkpoint(
                checkpoint,
                phase_enabled=self._phase_gate_enabled,
                phase_config=(
                    self.phase_gate_config if self._phase_gate_enabled else None
                ),
                allow_migration=allow_gate_migration,
                reset_optimizer=reset_optimizer,
            )
            if (
                checkpoint.get("residual_gate_mode", LEGACY_GATE_MODE)
                == LEGACY_GATE_MODE
                and checkpoint.get("residual_gate_config")
                != vars(self.residual_gate_config)
            ):
                raise RuntimeError("residual checkpoint gate configuration mismatch")
            if checkpoint.get("high_force_speed_scale") != 0.4:
                diagnostic_migration = (
                    os.environ.get("FINETUNE_ACCEPTANCE_CONTRACT") == "diagnostic"
                    and os.environ.get("FINETUNE_ALLOW_SPEED_SCALE_MIGRATION", "0") == "1"
                )
                if not diagnostic_migration:
                    raise RuntimeError("residual checkpoint speed scale mismatch")
            incompatible = actor.load_state_dict(
                checkpoint["actor_state_dict"], strict=False
            )
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(("residual_mlp.", "velocity_head_mlp.", "lateral_velocity_head_mlp.", "emergency_mlp.", "emergency_stability_mlp.", "emergency_virtual_mlp.", "emergency_virtual_reverse_mlp.", "emergency_virtual_recovery_mlp.", "emergency_reverse_mlp.", "emergency_recovery_mlp.", "emergency_lateral_mlp.", "emergency_support_mlp.", "emergency_support_y_pos_mlp.", "emergency_support_y_neg_mlp.", "emergency_contact_adapter_mlp.", "emergency_joint_adapter_mlp.", "emergency_pull_y_", "emergency_reverse_y_", "emergency_recovery_y_"))
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "residual checkpoint actor is incompatible: "
                    f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
                )
        else:
            actor.load_base_state_dict(checkpoint["actor_state_dict"])
        self.alg.critic.load_state_dict(checkpoint["critic_state_dict"], strict=strict)

        use_frozen_estimator = os.environ.get("GUIDE_DOG_USE_FROZEN_ESTIMATOR", "0") == "1"
        estimator_state = checkpoint.get("force_estimator_state_dict")
        if use_frozen_estimator:
            self._load_frozen_estimator()
            estimator_state = self.force_estimator.state_dict()
            self._checkpoint_estimator_loaded = True
            self.force_bias.zero_()
        else:
            if estimator_state is None:
                raise RuntimeError("checkpoint has no force estimator state")
            self.force_estimator.load_state_dict(estimator_state, strict=True)
        train_force_estimator = os.environ.get(
            "FINETUNE_TRAIN_FORCE_ESTIMATOR", "0"
        ) == "1"
        if train_force_estimator:
            self.force_estimator.requires_grad_(True).train()
        else:
            self.force_estimator.requires_grad_(False).eval()
        self._frozen_force_estimator_sha256 = module_sha256(self.force_estimator)
        expected_estimator_hash = None if use_frozen_estimator else checkpoint.get("force_estimator_sha256")
        if expected_estimator_hash and expected_estimator_hash != self._frozen_force_estimator_sha256:
            raise RuntimeError("checkpoint force estimator SHA-256 mismatch")

        actor.freeze_base()
        residual_lr, critic_lr = residual_learning_rates()
        configure_residual_optimizer(
            self.alg,
            residual_learning_rate=residual_lr,
            critic_learning_rate=critic_lr,
        )
        if (
            is_residual
            and not reset_optimizer
            and checkpoint.get("optimizer_state_dict") is not None
            and not (isinstance(load_cfg, dict) and load_cfg.get("actor"))
        ):
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Checkpoint optimizer state predates the independent emergency head in
        # older runs.  Re-apply the requested emergency learning rate after any
        # optimizer restore so the new head cannot silently fall back to the
        # legacy residual rate.
        emergency_lr_raw = os.environ.get("FINETUNE_EMERGENCY_HEAD_LR")
        if emergency_lr_raw is not None:
            emergency_lr = float(emergency_lr_raw)
            for group in self.alg.optimizer.param_groups:
                if group.get("name") == "emergency":
                    group["lr"] = emergency_lr
        self.current_learning_iteration = int(checkpoint.get("iter", 0))
        self._checkpoint_estimator_loaded = True
        infos = checkpoint.get("infos")
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"][
                "common_step_counter"
            ]
        return infos
