import os
import time
from pathlib import Path

import torch
import wandb
from rsl_rl.utils import check_nan
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.utils.lab_api.math import quat_apply_inverse

from .force_estimator import ForceEstimator


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = wandb.run.name if wandb.run else "local"  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if wandb.run:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class GuideDogOnPolicyRunner(VelocityOnPolicyRunner):
  """PPO runner with a jointly trained GuideDog-style force estimator."""

  stabilize_start_deg_default = 2.0
  stabilize_full_deg_default = 8.0

  history_length = 25
  estimator_batch_size = 512
  # Stage 1 learns active locomotion without a physical pull. Stage 2 adds
  # force with a full teacher, then Stage 3 fades to deployment-only Fhat.
  gait_warmup_iterations = 1500
  force_teacher_iterations = 1500
  teacher_fade_iterations = 2000

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    raw_obs = self.env.get_observations()
    self.actor_obs_dim = raw_obs["actor"].shape[-1]
    # The actor remains 73-D.  The deployable estimator sees the 70-D
    # proprioception plus the twelve motor torque estimates.
    self.force_input_dim = self.actor_obs_dim - 3 + 12
    self.force_estimator = ForceEstimator(
      self.force_input_dim, self.history_length
    ).to(self.device)
    self.force_optimizer = torch.optim.Adam(self.force_estimator.parameters(), lr=1.0e-3)
    self.force_bias = torch.zeros(self.env.num_envs, 3, device=self.device)
    self._policy_force_smooth = torch.zeros(self.env.num_envs, 3, device=self.device)
    self._checkpoint_estimator_loaded = False
    self.force_history = torch.zeros(
      self.env.num_envs,
      self.history_length,
      self.force_input_dim,
      device=self.device,
    )
    self._force_inputs: list[torch.Tensor] = []
    self._force_labels: list[torch.Tensor] = []
    self._actor_command = torch.zeros(self.env.num_envs, 2, device=self.device)
    self._hard_force_mask = torch.zeros(
      self.env.num_envs, device=self.device, dtype=torch.bool
    )
    self._last_safe_actions: torch.Tensor | None = None
    # Play-only switch for isolating locomotion from force following.
    self._inference_command_source = "estimated_force"
    self._inference_release_gate = False
    self._inference_teacher_force = torch.zeros(
      self.env.num_envs, 3, device=self.device
    )
    try:
      self._tether_event = self.env.unwrapped.event_manager.get_term_cfg(
        "tether_pull"
      ).func
    except (KeyError, AttributeError):
      self._tether_event = None

  def _teacher_command_weight(self) -> float:
    """Weight of the true-force teacher after force is introduced."""
    override = os.environ.get("FINETUNE_TEACHER_COMMAND_WEIGHT")
    if override is not None:
      value = float(override)
      if not 0.0 <= value <= 1.0:
        raise ValueError("FINETUNE_TEACHER_COMMAND_WEIGHT must be in [0, 1]")
      return value
    progress = self.current_learning_iteration
    full_teacher_until = self.gait_warmup_iterations + self.force_teacher_iterations
    if progress < full_teacher_until:
      return 1.0
    fade_progress = (progress - full_teacher_until) / self.teacher_fade_iterations
    return float(max(0.0, 1.0 - fade_progress))

  def _phase_from_command(self, command: torch.Tensor) -> torch.Tensor:
    """Generate gait phase from the exact command shown to the actor."""
    env = self.env.unwrapped
    phase_offset_s = 0.0
    if self._tether_event is not None:
      phase_offset_s = getattr(self._tether_event, "recovery_phase_offset_s", 0.0)
    phase_offset_s += self._command_phase_offset(command) * 0.6
    phase_rate = self._command_phase_rate(command)
    global_phase = (
      ((env.episode_length_buf * env.step_dt) * (1.0 + phase_rate)) + phase_offset_s
    ) % 0.6 / 0.6
    phase = torch.stack(
      (
        torch.sin(global_phase * torch.pi * 2.0),
        torch.cos(global_phase * torch.pi * 2.0),
      ),
      dim=-1,
    )
    stand = torch.linalg.norm(command, dim=1) < 0.05
    return torch.where(stand.unsqueeze(1), torch.zeros_like(phase), phase)

  @staticmethod
  def _command_phase_rate(command: torch.Tensor) -> torch.Tensor:
    """Return an optional additive gait-rate term from the shown command."""
    max_rate = float(os.environ.get("FINETUNE_COMMAND_PHASE_RATE", "0.0"))
    if max_rate <= 0.0:
      return torch.zeros(command.shape[0], device=command.device, dtype=command.dtype)
    speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    return max_rate * torch.clamp((speed - 0.05) / 0.25, 0.0, 1.0)

  @staticmethod
  def _command_phase_offset(command: torch.Tensor) -> torch.Tensor:
    """Optional lateral direction phase offset, expressed as gait cycles."""
    magnitude = float(os.environ.get("FINETUNE_LATERAL_PHASE_OFFSET", "0.0"))
    if magnitude == 0.0:
      return torch.zeros(command.shape[0], device=command.device, dtype=command.dtype)
    x = command[:, 0]
    y = command[:, 1]
    lateral = torch.abs(y) > torch.abs(x)
    return torch.where(lateral, magnitude * torch.sign(y), torch.zeros_like(y))

  @staticmethod
  def _policy_command(command: torch.Tensor) -> torch.Tensor:
    """Reparameterize only the policy input; preserve external target command."""
    scale = float(os.environ.get("FINETUNE_POLICY_LATERAL_SCALE", "1.0"))
    if scale <= 0.0:
      raise ValueError("FINETUNE_POLICY_LATERAL_SCALE must be positive")
    result = command.clone()
    result[:, 1] = result[:, 1] * scale
    return result

  @staticmethod
  def _follow_command(estimated_force: torch.Tensor) -> torch.Tensor:
    """Map normalized Fhat to the same velocity command in train and play."""
    # Keep the certified mapping as the default, but allow a bounded
    # deployment/training calibration sweep.  The live diagnostics showed
    # command/velocity ratios of only 0.15--0.30; exposing the slope and caps
    # as environment parameters lets us test a faster command contract without
    # changing the force estimator or the >50 N hard-protection path.
    slope = float(os.environ.get("GUIDE_DOG_FOLLOW_MAP_SLOPE", "0.025"))
    x_cap = float(os.environ.get("GUIDE_DOG_FOLLOW_X_CAP", "0.385"))
    y_cap = float(os.environ.get("GUIDE_DOG_FOLLOW_Y_CAP", "0.224"))
    if slope <= 0.0 or x_cap <= 0.0 or y_cap <= 0.0:
      raise ValueError("follow mapping slope and caps must be positive")
    follow = torch.sign(estimated_force[:, :2]) * torch.clamp(
      (torch.abs(estimated_force[:, :2] * 80.0) - 6.0) * slope, min=0.0
    )
    follow[:, 0] = torch.clamp(follow[:, 0], -x_cap, x_cap)
    follow[:, 1] = torch.clamp(follow[:, 1], -y_cap, y_cap)
    # Use separate speed limits for the 30--50 N protection band and the
    # >50 N emergency band.  The former still follows, but needs a slower
    # command to avoid diagonal reverse failures; the latter is controlled by
    # the independent emergency recovery path.  The legacy variable remains
    # the fallback for both bands so existing runs keep their contract.
    force_n = torch.linalg.norm(estimated_force[:, :2] * 80.0, dim=1, keepdim=True)
    legacy_scale = float(os.environ.get("GUIDE_DOG_HIGH_FORCE_SPEED_SCALE", "1.0"))
    mid_scale = float(
      os.environ.get("GUIDE_DOG_MID_FORCE_SPEED_SCALE", str(legacy_scale))
    )
    over_limit_scale = float(
      os.environ.get("GUIDE_DOG_OVER_LIMIT_SPEED_SCALE", str(legacy_scale))
    )
    # The estimator is intentionally conservative and can report a value
    # above the physical force near the 45--50 N boundary.  Use the same
    # calibrated live hard-force estimate threshold for command scaling as
    # the protection gate; otherwise a 44 N event estimated at 60 N is
    # incorrectly forced to zero before the hard gate actually trips.
    hard_estimate_threshold = float(
      os.environ.get("GUIDE_DOG_HARD_FORCE_ESTIMATE_N", "50.0")
    )
    if hard_estimate_threshold <= 0.0:
      raise ValueError("GUIDE_DOG_HARD_FORCE_ESTIMATE_N must be positive")
    if min(mid_scale, over_limit_scale) < 0.0:
      raise ValueError("force speed scales must be non-negative")
    scale = torch.where(
      force_n >= hard_estimate_threshold,
      torch.full_like(force_n, over_limit_scale),
      torch.where(
        force_n >= 30.0,
        torch.full_like(force_n, mid_scale),
        torch.ones_like(force_n),
      ),
    )
    command_gain = float(os.environ.get("GUIDE_DOG_FOLLOW_COMMAND_GAIN", "1.0"))
    if command_gain <= 0.0 or command_gain > 1.5:
      raise ValueError("GUIDE_DOG_FOLLOW_COMMAND_GAIN must be in (0, 1.5]")
    follow = follow * scale * command_gain
    vertical_scale = 1.0 - 0.35 * torch.clamp(
      (torch.abs(estimated_force[:, 2] * 80.0) - 8.0) / 20.0, 0.0, 1.0
    )
    return follow * vertical_scale.unsqueeze(-1)

  @staticmethod
  def command_effective_force(estimated_force: torch.Tensor) -> torch.Tensor:
    """Apply a noise-safe startup floor, then preserve the estimator magnitude."""
    effective = estimated_force.clone()
    fx_n = effective[:, 0] * 80.0
    abs_fx_n = torch.abs(fx_n)
    activated = abs_fx_n >= 3.0
    effective_abs_n = torch.where(
      activated,
      torch.maximum(abs_fx_n, torch.full_like(abs_fx_n, 12.0)),
      abs_fx_n,
    )
    effective[:, 0] = torch.sign(fx_n) * effective_abs_n / 80.0
    return effective

  def _policy_observation(
    self, raw_obs: TensorDict, dones: torch.Tensor | None = None
  ) -> TensorDict:
    """Build the actor input through the deployment Fhat -> command chain.

    True tether force remains privileged: it labels the estimator and defines
    the tracking reward, but it never supplies the actor velocity command.
    """
    if dones is not None and dones.any():
      self.force_history[dones.bool()] = 0.0
      self.force_bias[dones.bool()] = 0.0
      self._policy_force_smooth[dones.bool()] = 0.0
    raw_actor = raw_obs["actor"]
    proprio = raw_actor[:, :-3]
    estimator_proprio = proprio.clone()
    # Never let the true-force-derived command leak into force estimation.
    estimator_proprio[:, 13:16] = 0.0
    raw_tau = self.env.unwrapped.scene["robot"].data.actuator_force
    tau = raw_tau.reshape(raw_tau.shape[0], 12).to(estimator_proprio)
    # MuJoCo/SDK actuator order is FR,FL,RR,RL; policy order is FL,FR,RL,RR.
    tau = tau[:, [3,4,5,0,1,2,9,10,11,6,7,8]]
    estimator_frame = torch.cat((estimator_proprio, tau), dim=1)
    self.force_history = torch.roll(self.force_history, shifts=1, dims=1)
    self.force_history[:, 0] = estimator_frame
    with torch.no_grad():
      raw_estimated_force = self._estimate_force(self.force_history)
    # Calibrate the estimator zero point only on genuinely force-free samples.
    # The critic's true-force channel is privileged simulation data.  It may
    # be used for the explicit manual/teacher acceptance harness, but must
    # not affect the deployment estimator or actor path.
    manual_tether = bool(
      getattr(getattr(self, "_tether_event", None), "manual_enabled", False)
    )
    true_force = raw_obs["critic"][:, -3:] if manual_tether else torch.zeros_like(raw_estimated_force)
    zero_force = manual_tether & (
      torch.linalg.vector_norm(true_force, dim=1) < (1.0 / 80.0)
    )
    self.force_bias = torch.where(
      zero_force.unsqueeze(1),
      self.force_bias * 0.98 + raw_estimated_force.detach() * 0.02,
      self.force_bias,
    )
    estimated_force = (raw_estimated_force - self.force_bias).clamp(-1.0, 1.0)
    # Emergency base-only contract: above the physical force limit, remove
    # force-derived command and force channels before the actor is evaluated.
    # The privileged true-force check is only a simulation/training guard;
    # deployment still has the calibrated estimator threshold.
    hard_force_n = float(os.environ.get("GUIDE_DOG_HARD_FORCE_N", "50.0"))
    hard_estimate_n = float(
      os.environ.get("GUIDE_DOG_HARD_FORCE_ESTIMATE_N", str(hard_force_n))
    )
    hard_force = torch.linalg.vector_norm(
      estimated_force[:, :2] * 80.0, dim=1
    ) > hard_estimate_n
    if manual_tether:
      hard_force = hard_force | (
        torch.linalg.vector_norm(true_force[:, :2] * 80.0, dim=1) > hard_force_n
      )
    self._hard_force_mask = hard_force.detach()
    estimated_force = torch.where(
      hard_force.unsqueeze(1), torch.zeros_like(estimated_force), estimated_force
    )
    # During reverse confirmation, avoid a one-step sign jump in the actor
    # observation/command. The tether itself is already zero during dwell;
    # this smooths the estimated force over the same 200 ms safety window.
    physical_phase = getattr(self._tether_event, "_gate_phase", None)
    ramp_s = float(os.environ.get("FINETUNE_REVERSE_RAMP_S", "0"))
    if physical_phase in {"reverse_dwell", "reverse"} and ramp_s > 0.0:
      alpha = min(1.0, float(self.env.unwrapped.step_dt) / ramp_s)
      estimated_force = self._policy_force_smooth + alpha * (estimated_force - self._policy_force_smooth)
    self._policy_force_smooth = estimated_force.detach()

    estimated_command = self._follow_command(
      self.command_effective_force(estimated_force)
    )
    # The final three critic entries are normalized privileged F_true labels.
    teacher_command = self._follow_command(raw_obs["critic"][:, -3:])
    if self.current_learning_iteration < self.gait_warmup_iterations:
      # Force-free locomotion foundation: the policy must actively walk.
      actor_command = proprio[:, 13:15].clone()
    else:
      teacher_weight = self._teacher_command_weight()
      actor_command = (
        teacher_weight * teacher_command
        + (1.0 - teacher_weight) * estimated_command
      )
      if self._tether_event is not None:
        retention_mask = self._tether_event.retention_mask
        actor_command = torch.where(
          retention_mask.unsqueeze(1),
          self._tether_event.retention_command,
          actor_command,
        )
    physical_phase = getattr(self._tether_event, "_gate_phase", None)
    if physical_phase in {"release", "reverse_dwell", "final_release"}:
      actor_command = torch.zeros_like(actor_command)
    actor_command = torch.where(
      hard_force.unsqueeze(1), torch.zeros_like(actor_command), actor_command
    )
    # Optional closed-loop speed compensation.  This is intentionally
    # separate from the force-to-speed map: it uses the measured body speed
    # to correct under-tracking, while bounding the command increase so the
    # gait cannot be pushed into the contact-instability region by a single
    # large force sample.  It is opt-in for diagnostics/deployment trials.
    feedback_gain = float(os.environ.get("GUIDE_DOG_SPEED_FEEDBACK_GAIN", "0"))
    if feedback_gain < 0.0 or feedback_gain > 1.0:
      raise ValueError("GUIDE_DOG_SPEED_FEEDBACK_GAIN must be in [0, 1]")
    if feedback_gain > 0.0:
      measured = self.env.unwrapped.scene["robot"].data.root_link_lin_vel_b[:, :2]
      target = actor_command[:, :2]
      target_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True)
      corrected = target + feedback_gain * (target - measured)
      max_ratio = float(os.environ.get("GUIDE_DOG_SPEED_FEEDBACK_MAX_RATIO", "1.10"))
      if max_ratio < 1.0 or max_ratio > 1.25:
        raise ValueError("GUIDE_DOG_SPEED_FEEDBACK_MAX_RATIO must be in [1, 1.25]")
      max_norm = target_norm * max_ratio
      corrected_norm = torch.linalg.vector_norm(corrected, dim=1, keepdim=True)
      corrected = torch.where(
        (corrected_norm > max_norm) & (corrected_norm > 1e-6),
        corrected * (max_norm / corrected_norm.clamp_min(1e-6)),
        corrected,
      )
      actor_command[:, :2] = torch.where(
        (target_norm > 0.04) & (~hard_force).unsqueeze(1), corrected, target
      )
    self._reward_command = actor_command.detach().clone()
    actor_gain = float(os.environ.get("FINETUNE_ACTOR_COMMAND_OBS_GAIN", "1.0"))
    if actor_gain <= 0.0 or actor_gain > 3.0:
      raise ValueError("FINETUNE_ACTOR_COMMAND_OBS_GAIN must be in (0, 3]")
    actor_command = torch.clamp(actor_command * actor_gain, -1.0, 1.0)
    self._actor_command = actor_command.detach()
    actor_proprio = proprio.clone()
    policy_command = self._policy_command(actor_command)
    actor_proprio[:, 13:15] = policy_command
    actor_proprio[:, 15] = 0.0
    actor_proprio[:, 16:18] = self._phase_from_command(policy_command)
    obs = raw_obs.clone()
    actor = torch.cat((actor_proprio, estimated_force), dim=-1)
    obs["actor"] = actor
    # Critic starts with the actor terms, then has privileged terms including
    # the true external-force label at its final three entries.
    critic = raw_obs["critic"].clone()
    critic[:, : self.actor_obs_dim] = actor
    obs["critic"] = critic
    augmented = self._augment_policy_observation(obs, estimated_force, dones)
    record_alignment = getattr(self, "_record_phase_alignment", None)
    if record_alignment is not None and "residual_gate" in augmented:
      record_alignment(
        estimated_force * 80.0,
        true_force * 80.0,
        augmented["residual_gate"],
        augmented.get("emergency_gate"),
      )
    return augmented

  def _augment_policy_observation(
    self,
    obs: TensorDict,
    estimated_force: torch.Tensor,
    reset_mask: torch.Tensor | None = None,
  ) -> TensorDict:
    """Optional policy-observation extension for specialized runners."""
    return obs

  def _extra_training_metrics(self) -> dict[str, float]:
    """Optional rollout metrics for specialized runners."""
    return {}

  def _estimate_force(self, history: torch.Tensor) -> torch.Tensor:
    # Frozen deployment estimators are trained on normalized 82-D history.
    # The base SelfPull task uses this runner directly, so normalization must
    # live here rather than only in specialized joint-robust subclasses.
    mean = getattr(self, "_force_mean", None)
    std = getattr(self, "_force_std", None)
    if mean is not None and std is not None:
      history = (history - mean) / std
    return self.force_estimator(history).detach()

  def _apply_actor_command_to_env(self) -> None:
    """Make the reward target exactly match the command in the actor input."""
    cmd = self.env.unwrapped.command_manager.get_term("twist").vel_command_b
    cmd[:, :2] = getattr(self, "_reward_command", self._actor_command)
    cmd[:, 2] = 0.0

  def _sanitize_actions(self, actions: torch.Tensor) -> torch.Tensor:
    """Optional safety post-processing before actions reach the environment."""
    mask = getattr(self, "_hard_force_mask", None)
    if self._last_safe_actions is None or self._last_safe_actions.shape != actions.shape:
      self._last_safe_actions = actions.detach().clone()
      return actions
    if not isinstance(mask, torch.Tensor) or mask.numel() != actions.shape[0]:
      self._last_safe_actions = actions.detach().clone()
      return actions
    mask = mask.to(device=actions.device, dtype=torch.bool).unsqueeze(1)
    if os.environ.get("GUIDE_DOG_HARD_FORCE_ZERO_ACTION", "0") == "1":
      safe = torch.where(mask, torch.zeros_like(actions), actions)
    elif os.environ.get("GUIDE_DOG_HARD_FORCE_HOLD_ACTION", "0") == "1":
      safe = torch.where(mask, self._last_safe_actions, actions)
    else:
      safe = actions
    self._last_safe_actions = safe.detach().clone()
    return safe

  def _collect_force_examples(self, raw_obs: TensorDict) -> None:
    labels = raw_obs["critic"][:, -3:]
    count = min(self.estimator_batch_size, self.env.num_envs)
    ids = torch.randperm(self.env.num_envs, device=self.device)[:count]
    self._force_inputs.append(self.force_history[ids].detach().clone())
    self._force_labels.append(labels[ids].detach().clone())

  def _update_force_estimator(self) -> float:
    if not self._force_inputs:
      return 0.0
    inputs = torch.cat(self._force_inputs)
    labels = torch.cat(self._force_labels)
    self._force_inputs.clear()
    self._force_labels.clear()
    prediction = self.force_estimator(inputs)
    loss = torch.nn.functional.mse_loss(prediction, labels)
    self.force_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.force_estimator.parameters(), 1.0)
    self.force_optimizer.step()
    return float(loss.detach())

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    raw_obs = self.env.get_observations().to(self.device)
    obs = self._policy_observation(raw_obs)
    self.alg.train_mode()
    self.force_estimator.train()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      # The generic tether event is intentionally disabled by default so play
      # and deployment never receive an unsolicited random pull.  Training
      # must enable it explicitly after the force-free gait warmup; otherwise
      # the estimator only sees zero labels and the teacher/fade curriculum is
      # silently ineffective.
      if self._tether_event is not None:
        # Gate-aligned residual training must keep the physical tether event
        # autonomous for the full rollout.  Otherwise a resumed run can
        # briefly re-enter the force-free warmup path and leave last_force_b
        # at zero, making the phase gate appear stuck.  The override is
        # opt-in and does not affect play/deployment or ordinary training.
        force_event_always_on = (
          os.environ.get("FINETUNE_FORCE_EVENT_ALWAYS_ON", "0") == "1"
        )
        self._tether_event.set_autonomous_enabled(
          force_event_always_on or it >= self.gait_warmup_iterations
        )
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          actions = self._sanitize_actions(actions)
          collect_constraint = getattr(self, "_collect_residual_constraint_examples", None)
          if collect_constraint is not None:
            collect_constraint(obs)
          # The actor action, velocity-tracking reward, and physical step share
          # the same teacher/Fhat command for this control period.
          self._apply_actor_command_to_env()
          raw_next, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(raw_next, rewards, dones)
          raw_next = raw_next.to(self.device)
          rewards, dones = rewards.to(self.device), dones.to(self.device)
          next_obs = self._policy_observation(raw_next, dones)
          self._collect_force_examples(raw_next)
          self.alg.process_env_step(next_obs, rewards, dones, extras)
          # Older mjlab configs omit rnd_cfg when RND is disabled.
          intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
          raw_obs, obs = raw_next, next_obs
        collect_time = time.time() - start
        self.alg.compute_returns(obs)

      update_start = time.time()
      loss_dict = self.alg.update()
      update_constraint = getattr(self, "_update_residual_constraints", None)
      if update_constraint is not None:
        loss_dict.update(update_constraint())
      loss_dict["force_estimator_loss"] = self._update_force_estimator()
      loss_dict["teacher_command_weight"] = self._teacher_command_weight()
      loss_dict.update(self._extra_training_metrics())
      learn_time = time.time() - update_start
      self.current_learning_iteration = it
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
      )
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None):
    super().save(path, infos)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["force_estimator_state_dict"] = self.force_estimator.state_dict()
    checkpoint["force_optimizer_state_dict"] = self.force_optimizer.state_dict()
    checkpoint["force_estimator_input_dim"] = self.force_input_dim
    # Persist the exact normalization used by the deployment estimator.  The
    # supervised episode estimator is trained on this standardized history;
    # dropping these tensors makes a later actor checkpoint silently fall back
    # to the legacy frozen-estimator normalization.
    if hasattr(self, "_force_mean") and hasattr(self, "_force_std"):
      checkpoint["force_estimator_input_mean"] = self._force_mean.detach().cpu()
      checkpoint["force_estimator_input_std"] = self._force_std.detach().cpu()
    torch.save(checkpoint, path)

    export_dir = os.path.dirname(path)
    estimator_path = os.path.join(export_dir, "force_estimator.onnx")
    estimator = self.force_estimator.to("cpu").eval()
    dummy = torch.zeros(1, self.history_length, self.force_input_dim)
    mean = checkpoint.get("force_estimator_input_mean")
    std = checkpoint.get("force_estimator_input_std")
    if mean is not None and std is not None:
      class _NormalizedEstimator(torch.nn.Module):
        def __init__(self, model, input_mean, input_std):
          super().__init__()
          self.model = model
          self.register_buffer("input_mean", input_mean)
          self.register_buffer("input_std", input_std.clamp_min(1.0e-4))
        def forward(self, history):
          return self.model((history - self.input_mean) / self.input_std)
      export_estimator = _NormalizedEstimator(estimator, mean, std).eval()
    else:
      export_estimator = estimator
    torch.onnx.export(
      export_estimator,
      dummy,
      estimator_path,
      input_names=["proprio_history"],
      output_names=["estimated_force"],
      opset_version=18,
      dynamo=False,
    )
    self.force_estimator.to(self.device).train()

  def load(self, path: str, *args, **kwargs):
    infos = super().load(path, *args, **kwargs)
    checkpoint = torch.load(path, weights_only=False, map_location=self.device)
    # The policy checkpoint may contain an older estimator backbone (the
    # 32-wide v20 policy estimator).  When the deployment contract explicitly
    # requests the frozen estimator artifact, do not silently replace it with
    # that embedded state: the actor and the gate must consume the same frozen
    # estimator used at deployment.  This is opt-in while existing checkpoints
    # are being audited for low-force direction parity.
    if os.environ.get("GUIDE_DOG_USE_FROZEN_ESTIMATOR", "0") == "1":
      frozen = os.environ.get(
        "GUIDE_DOG_FROZEN_ESTIMATOR",
        "artifacts/v20_tau82_static_ramp_body_map_estimator/estimator.pt",
      )
      frozen_path = Path(frozen)
      if frozen_path.is_dir():
        frozen_path = frozen_path / "estimator.pt"
      frozen_ckpt = torch.load(
        str(frozen_path), weights_only=False, map_location=self.device
      )
      self.force_estimator = ForceEstimator(
        int(frozen_ckpt.get("input_dim", self.force_input_dim)),
        int(frozen_ckpt.get("history_length", self.history_length)),
        hidden_dim=int(frozen_ckpt.get("hidden_dim", 96)),
      ).to(self.device)
      self.force_estimator.load_state_dict(frozen_ckpt["model"])
      self.force_estimator.eval()
      for parameter in self.force_estimator.parameters():
        parameter.requires_grad_(False)
      self._force_mean = frozen_ckpt["mean"].to(self.device)
      self._force_std = frozen_ckpt["std"].to(self.device).clamp_min(1.0e-4)
      self._checkpoint_estimator_loaded = True
      self.force_bias.zero_()
      print(f"[INFO] using frozen deployment estimator: {frozen_path}")
      return infos
    state = checkpoint.get("force_estimator_state_dict")
    if state is None:
      # A checkpoint can be left by a process that failed between the base
      # runner save and the estimator augmentation. Keep the initialized
      # deployment estimator and resume the policy safely from that checkpoint.
      print(f"[WARN] {path} has no force estimator state; keeping initialized estimator")
      self.force_estimator.eval()
      return infos
    saved_input_dim = int(state["encoder.0.weight"].shape[1])
    if saved_input_dim != self.force_input_dim:
      print(
        f"[WARN] {path} estimator input {saved_input_dim} does not match "
        f"current {self.force_input_dim}; keeping the current estimator"
      )
      self.force_estimator.eval()
      return infos
    # Older checkpoints may use a wider estimator backbone (e.g. 96 vs 32).
    # Rebuild the module to match the serialized architecture before loading.
    saved_hidden = int(state["encoder.0.weight"].shape[0])
    current_hidden = int(self.force_estimator.encoder[0].out_features)
    if saved_hidden != current_hidden:
      self.force_estimator = ForceEstimator(
        self.force_input_dim, self.history_length, hidden_dim=saved_hidden
      ).to(self.device)
      self.force_optimizer = torch.optim.Adam(
        self.force_estimator.parameters(), lr=1.0e-3
      )
    self.force_estimator.load_state_dict(state, strict=True)
    self._checkpoint_estimator_loaded = True
    self.force_estimator.eval()
    return infos

  def set_inference_command_source(self, source: str) -> None:
    if source not in {"estimated_force", "direct_twist", "true_force_teacher"}:
      raise ValueError(f"Unsupported inference command source: {source}")
    self._inference_command_source = source

  def set_inference_release_gate(self, released: bool) -> None:
    """Zero estimated-force commands while an external release is explicit."""
    self._inference_release_gate = bool(released)

  def set_inference_teacher_force(self, force_n: torch.Tensor) -> None:
    """Set the body-frame force used only by the Gate 2 teacher play mode."""
    force_n = force_n.to(device=self.device, dtype=self._inference_teacher_force.dtype)
    if force_n.ndim == 1:
      force_n = force_n.unsqueeze(0).expand(self.env.num_envs, -1)
    self._inference_teacher_force.copy_(force_n / 80.0)

  def _shape_high_force_actor_observation(self, force: torch.Tensor) -> torch.Tensor:
    """Optionally attenuate only the actor's high-force context channel.

    The command still uses the full calibrated force; this gate is useful for
    the high-force curriculum where a 30 N label otherwise drives the learned
    actor outside its stable lateral gait region. It is disabled by default.
    """
    scale = float(os.environ.get("GUIDE_DOG_HIGH_FORCE_OBS_SCALE", "1.0"))
    if scale >= 0.999:
      return force
    norm = torch.linalg.norm(force[:, :2], dim=1, keepdim=True)
    return torch.where(norm >= 20.0, force * scale, force)

  def _prepare_inference_force(self, estimated_force: torch.Tensor) -> torch.Tensor:
    """Return the force channel supplied to the actor during play."""
    return estimated_force

  def _inference_command_from_force(
    self, estimated_force: torch.Tensor
  ) -> torch.Tensor:
    """Map the play-time force channel to the actor command."""
    return self._follow_command(self.command_effective_force(estimated_force))

  def _stabilize_actions(
    self, actions: torch.Tensor, early: bool = False
  ) -> torch.Tensor:
    """Blend actions toward the learned standing pose as tilt becomes unsafe.

    This shared runtime safety layer is used by Viser and the force-matrix
    evaluator. It stays inactive during PPO collection, preserving the policy
    action contract while preventing a falling robot from receiving full joint
    commands at deployment time.
    """
    try:
      quat = self.env.unwrapped.scene["robot"].data.root_link_quat_w
    except (AttributeError, KeyError):
      return actions
    tilt_deg = torch.rad2deg(
      torch.acos(torch.clamp(
        1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1.0, 1.0
      ))
    )
    if early:
      start_deg, full_deg = 0.0, float(os.environ.get("GUIDE_DOG_STABILIZE_EARLY_FULL_DEG", "4.0"))
    else:
      start_deg = float(
        os.environ.get(
          "GUIDE_DOG_STABILIZE_START_DEG", str(self.stabilize_start_deg_default)
        )
      )
      full_deg = float(
        os.environ.get(
          "GUIDE_DOG_STABILIZE_FULL_DEG", str(self.stabilize_full_deg_default)
        )
      )
    keep = torch.clamp((full_deg - tilt_deg) / (full_deg - start_deg), 0.0, 1.0).unsqueeze(1)
    return actions * keep

  def get_inference_policy(self, device: str | None = None):
    """Return the deployment-equivalent ForceEstimator -> policy chain."""
    self.alg.eval_mode()
    self.force_estimator.eval()
    policy = self.alg.get_policy().to(device or self.device)
    history: torch.Tensor | None = None
    self.last_estimated_force: torch.Tensor | None = None
    self.last_actor_command: torch.Tensor | None = None

    @torch.inference_mode()
    def inference(raw_obs: TensorDict) -> torch.Tensor:
      nonlocal history
      raw_actor = raw_obs["actor"]
      proprio = raw_actor[:, :-3]
      estimator_proprio = proprio.clone()
      estimator_proprio[:, 13:16] = 0.0
      raw_tau = self.env.unwrapped.scene["robot"].data.actuator_force
      tau = raw_tau.reshape(raw_tau.shape[0], 12).to(estimator_proprio)
      tau = tau[:, [3,4,5,0,1,2,9,10,11,6,7,8]]
      estimator_frame = torch.cat((estimator_proprio, tau), dim=1)
      if history is None or history.shape[0] != proprio.shape[0]:
        history = torch.zeros(
          proprio.shape[0], self.history_length, self.force_input_dim,
          device=proprio.device, dtype=proprio.dtype,
        )
      history = torch.roll(history, shifts=1, dims=1)
      history[:, 0] = estimator_frame
      inference._history_steps = getattr(inference, "_history_steps", 0) + 1
      dump_path = os.environ.get("GUIDE_DOG_DUMP_HISTORY")
      if dump_path and inference._history_steps == 300:
        values = history[0].detach().cpu().reshape(-1).tolist()
        Path(dump_path).write_text(" ".join(map(str, values)) + "\n")
      # Keep play/Viser on the same estimator path as training. Subclasses
      # use this hook to apply the saved input normalization before inference.
      estimated_force = self._estimate_force(history)
      estimated_force = self._prepare_inference_force(estimated_force)
      if dump_path and inference._history_steps == 300:
        Path(dump_path + ".force").write_text(
          " ".join(map(str, estimated_force[0].detach().cpu().tolist())) + "\n"
        )
      recalibrate_startup = os.environ.get("GUIDE_DOG_FORCE_RECALIBRATE_STARTUP", "0") == "1"
      if inference._history_steps <= 20 and (
        not self._checkpoint_estimator_loaded or recalibrate_startup
      ):
        # Frozen-estimator checkpoints use the legacy startup baseline. A
        # supervised checkpoint already predicts the calibrated force label.
        self.force_bias[:proprio.shape[0]] = (
          self.force_bias[:proprio.shape[0]] * 0.90
          + estimated_force.detach() * 0.10
        )
      if self._inference_command_source == "true_force_teacher":
        teacher_zero = (
          torch.linalg.vector_norm(self._inference_teacher_force[:proprio.shape[0]], dim=1)
          < (1.0 / 80.0)
        )
      else:
        teacher_zero = torch.zeros(
          proprio.shape[0], device=proprio.device, dtype=torch.bool
        )
      bias = self.force_bias[:proprio.shape[0]]
      bias = torch.where(
        teacher_zero.unsqueeze(1),
        bias * 0.98 + estimated_force.detach() * 0.02,
        bias,
      )
      if self._checkpoint_estimator_loaded and not recalibrate_startup:
        bias = torch.zeros_like(bias)
      self.force_bias[:proprio.shape[0]] = bias
      estimated_force = (estimated_force - bias).clamp(-1.0, 1.0)
      if self._inference_release_gate and self._inference_command_source == "estimated_force":
        estimated_force = torch.zeros_like(estimated_force)
      self.last_estimated_force = estimated_force.detach()
      actor_proprio = proprio.clone()
      if self._inference_command_source == "estimated_force":
        # Deployment path: own-state history -> Fhat -> follow command.
        actor_proprio[:, 13:15] = self._inference_command_from_force(
          estimated_force
        )
      elif self._inference_command_source == "true_force_teacher":
        # Gate 2 diagnostic: isolate locomotion under pull from estimator error.
        actor_proprio[:, 13:15] = self._follow_command(
          self._inference_teacher_force[:actor_proprio.shape[0]]
        )
      # The play/Viser path does not pass through the PPO rollout command
      # builder above, so apply the same optional bounded speed feedback here.
      # This keeps the diagnostic and deployment paths identical while leaving
      # the default (gain=0) behavior unchanged.
      feedback_gain = float(os.environ.get("GUIDE_DOG_SPEED_FEEDBACK_GAIN", "0"))
      if feedback_gain < 0.0 or feedback_gain > 1.0:
        raise ValueError("GUIDE_DOG_SPEED_FEEDBACK_GAIN must be in [0, 1]")
      if feedback_gain > 0.0 and self._inference_command_source != "direct_twist":
        measured = self.env.unwrapped.scene["robot"].data.root_link_lin_vel_b[:, :2]
        target = actor_proprio[:, 13:15]
        target_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True)
        if (
          not hasattr(self, "_speed_feedback_integral")
          or self._speed_feedback_integral.shape[0] != target.shape[0]
        ):
          self._speed_feedback_integral = torch.zeros_like(target)
        error = target - measured
        integral_gain = float(os.environ.get("GUIDE_DOG_SPEED_FEEDBACK_KI", "0"))
        if integral_gain < 0.0 or integral_gain > 0.5:
          raise ValueError("GUIDE_DOG_SPEED_FEEDBACK_KI must be in [0, 0.5]")
        active = target_norm > 0.04
        self._speed_feedback_integral = torch.where(
          active,
          self._speed_feedback_integral + integral_gain * error,
          torch.zeros_like(self._speed_feedback_integral),
        )
        integral_limit_ratio = float(
          os.environ.get("GUIDE_DOG_SPEED_FEEDBACK_INTEGRAL_MAX_RATIO", "0.75")
        )
        if integral_limit_ratio < 0.0 or integral_limit_ratio > 1.0:
          raise ValueError(
            "GUIDE_DOG_SPEED_FEEDBACK_INTEGRAL_MAX_RATIO must be in [0, 1]"
          )
        integral_limit = target_norm * integral_limit_ratio
        integral_norm = torch.linalg.vector_norm(
          self._speed_feedback_integral, dim=1, keepdim=True
        )
        self._speed_feedback_integral = torch.where(
          (integral_norm > integral_limit) & (integral_norm > 1e-6),
          self._speed_feedback_integral
          * (integral_limit / integral_norm.clamp_min(1e-6)),
          self._speed_feedback_integral,
        )
        corrected = target + feedback_gain * error + self._speed_feedback_integral
        max_ratio = float(os.environ.get("GUIDE_DOG_SPEED_FEEDBACK_MAX_RATIO", "1.10"))
        if max_ratio < 1.0 or max_ratio > 1.75:
          raise ValueError("GUIDE_DOG_SPEED_FEEDBACK_MAX_RATIO must be in [1, 1.75]")
        max_norm = target_norm * max_ratio
        corrected_norm = torch.linalg.vector_norm(corrected, dim=1, keepdim=True)
        corrected = torch.where(
          (corrected_norm > max_norm) & (corrected_norm > 1e-6),
          corrected * (max_norm / corrected_norm.clamp_min(1e-6)),
          corrected,
        )
        actor_proprio[:, 13:15] = torch.where(
          target_norm > 0.04, corrected, target
        )
      # direct_twist retains the Viser Twist command in indices 13:15.
      if os.environ.get("GUIDE_DOG_FORCE_YAW_ALIGN", "0") == "1":
        # Rotate the body toward the measured pull direction before asking the
        # locomotion policy to track it. This is an optional high-force
        # robustness path; the default deployment contract remains yaw=0.
        yaw = torch.atan2(estimated_force[:, 1], estimated_force[:, 0])
        actor_proprio[:, 15] = torch.clamp(0.8 * yaw, -0.8, 0.8)
      else:
        actor_proprio[:, 15] = 0.0
      self.last_actor_command = actor_proprio[:, 13:15].detach().clone()
      # Keep the built-in purple Twist arrow synchronized with the command
      # actually supplied to the actor during estimated/teacher validation.
      if self._inference_command_source != "direct_twist":
        env_command = self.env.unwrapped.command_manager.get_term(
          "twist"
        ).vel_command_b
        count = actor_proprio.shape[0]
        env_command[:count, :2] = actor_proprio[:, 13:15]
        env_command[:count, 2] = 0.0
      actor_proprio[:, 16:18] = self._phase_from_command(actor_proprio[:, 13:15])
      actor_force = self._shape_high_force_actor_observation(estimated_force)
      actor = torch.cat((actor_proprio, actor_force), dim=-1)
      if os.environ.get("GUIDE_DOG_TRACE_OBS") == "1" and not getattr(
        inference, "_trace_written", False
      ):
        trace = actor[0].detach().cpu().tolist()
        print("PYTHON_ACTOR73=" + repr(trace), flush=True)
        inference._trace_written = True
      obs = raw_obs.clone()
      obs["actor"] = actor
      reset_mask = None
      tether_term = getattr(self, "_tether_event", None)
      try:
        reset_mask = self.env.episode_length_buf[:actor.shape[0]] == 0
      except AttributeError:
        pass
      obs = self._augment_policy_observation(obs, estimated_force, reset_mask)
      # Apply the same hard-force contract during deployment/evaluation as in
      # PPO rollout collection.  Previously `_sanitize_actions()` was only
      # called by the training loop, so acceptance could still send learned
      # pull actions after the >50 N emergency threshold was crossed.
      try:
        if tether_term is None:
          tether_term = self.env.unwrapped.event_manager.get_term_cfg(
            "tether_pull"
          ).func
        # Acceptance injects a force immediately before policy evaluation,
        # while last_force_b is only updated by the subsequent physics step.
        # Use the injected value in that mode to avoid carrying the previous
        # scenario's high-force state into the next scenario.
        if bool(getattr(tether_term, "manual_enabled", False)):
          true_force = tether_term.manual_force_b[: actor.shape[0], :2]
          true_force_n = torch.linalg.vector_norm(true_force, dim=1)
        else:
          # `last_force_b` is simulator-only privileged state.  It must not
          # participate in the deploy/evaluation hard gate; use the calibrated
          # estimator path below instead.
          true_force_n = torch.zeros(
            actor.shape[0], device=actor.device, dtype=actor.dtype
          )
      except (AttributeError, KeyError):
        true_force_n = torch.zeros(
          actor.shape[0], device=actor.device, dtype=actor.dtype
        )
      estimate_force_n = torch.linalg.vector_norm(
        estimated_force[:, :2], dim=1
      ) * 80.0
      hard_force_n = float(os.environ.get("GUIDE_DOG_HARD_FORCE_N", "50"))
      hard_estimate_n = float(
        os.environ.get("GUIDE_DOG_HARD_FORCE_ESTIMATE_N", str(hard_force_n))
      )
      if bool(getattr(locals().get("tether_term", None), "manual_enabled", False)):
        # In the deterministic acceptance matrix the manual force is the
        # ground-truth safety signal; the estimator is intentionally evaluated
        # separately and must not create false >50 N trips at 45 N.
        self._hard_force_mask = true_force_n > hard_force_n
      else:
        self._hard_force_mask = (true_force_n > hard_force_n) | (
          estimate_force_n > hard_estimate_n
        )
      learned_actions = policy(obs)
      recovery_active = getattr(
        self, "_hard_recovery_active", torch.zeros_like(self._hard_force_mask)
      )
      emergency_mask = self._hard_force_mask | recovery_active
      if torch.any(emergency_mask):
        # Emergency recovery is base-only, not a zero-torque command.  Re-run
        # the mature policy with force-derived command/phase/context removed;
        # this preserves the standing gait and avoids the fall caused by
        # abruptly zeroing all joint targets above 50 N.
        safe_obs = obs.clone()
        safe_actor = safe_obs["actor"].clone()
        hard = emergency_mask
        if os.environ.get("GUIDE_DOG_HARD_FORCE_KEEP_COMMAND", "0") != "1":
          safe_actor[hard, 13:16] = 0.0
        if os.environ.get("GUIDE_DOG_HARD_FORCE_KEEP_PHASE", "0") != "1":
          safe_actor[hard, 16:18] = 0.0
        # Keep the measured direction only when explicitly testing the
        # emergency recovery controller.  The command and phase remain zero,
        # so this cannot reopen normal pull following or the residual branch.
        if os.environ.get("GUIDE_DOG_HARD_FORCE_KEEP_FORCE") == "1":
          force_scale = float(
            os.environ.get("GUIDE_DOG_HARD_FORCE_FORCE_SCALE", "1.0")
          )
          if os.environ.get("GUIDE_DOG_HARD_FORCE_DIRECTION_ONLY") == "1":
            horizontal = safe_actor[hard, -3:-1]
            direction = horizontal / torch.linalg.vector_norm(
              horizontal, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            recovery_force_n = float(
              os.environ.get("GUIDE_DOG_HARD_FORCE_DIRECTION_N", "20.0")
            )
            recovery_force = torch.zeros_like(safe_actor[hard, -3:])
            recovery_force[:, :2] = direction * (recovery_force_n / 80.0)
            safe_actor[hard, -3:] = recovery_force * force_scale
          else:
            safe_actor[hard, -3:] = safe_actor[hard, -3:] * force_scale
        else:
          safe_actor[hard, -3:] = 0.0
        safe_obs["actor"] = safe_actor
        if "residual_gate" in safe_obs.keys():
          safe_gate = safe_obs["residual_gate"].clone()
          if os.environ.get("FINETUNE_HARD_FORCE_RECOVERY_RESIDUAL", "0") == "1":
            # residual_runner has already restricted this value to the
            # confirmed recovery phase; preserve it while keeping command,
            # phase, and force observations base-only.
            safe_gate[hard] = safe_obs["residual_gate"][hard]
          else:
            safe_gate[hard] = 0.0
          safe_obs["residual_gate"] = safe_gate
        base_only_actions = policy(safe_obs)
        base_scale = float(
          os.environ.get("GUIDE_DOG_HARD_FORCE_BASE_ACTION_SCALE", "1.0")
        )
        if base_scale < 1.0:
          base_only_actions = base_only_actions * base_scale
        learned_actions = self._smooth_hard_force_recovery(
          learned_actions, base_only_actions, self._hard_force_mask, true_force_n
        )
      elif hasattr(self, "_hard_recovery_active"):
        # Count down the release dwell even after the force first falls below
        # the emergency threshold; normal policy output is restored only once
        # the recovery state has settled.
        learned_actions = self._smooth_hard_force_recovery(
          learned_actions, learned_actions, self._hard_force_mask, true_force_n
        )
      return self._stabilize_actions(learned_actions)

    def reset() -> None:
      # The viewer calls policy.reset() with every environment reset.  Clear the
      # temporal estimator state so a previous pull cannot generate a stale command.
      nonlocal history
      history = None
      reset_command_state = getattr(self, "reset_follow_command_state", None)
      if reset_command_state is not None:
        reset_command_state()
      self.last_estimated_force = None
      self.last_actor_command = None

    inference.reset = reset
    return inference


  def _smooth_hard_force_recovery(
    self,
    learned_actions: torch.Tensor,
    safe_actions: torch.Tensor,
    hard_force: torch.Tensor,
    force_n: torch.Tensor,
  ) -> torch.Tensor:
    """Blend into base-only recovery and hold it through release dwell."""
    batch = learned_actions.shape[0]
    device = learned_actions.device
    if not hasattr(self, "_hard_recovery_active") or self._hard_recovery_active.shape[0] != batch:
      self._hard_recovery_active = torch.zeros(batch, dtype=torch.bool, device=device)
      self._hard_recovery_steps = torch.zeros(batch, dtype=torch.int32, device=device)
      self._hard_release_steps = torch.zeros(batch, dtype=torch.int32, device=device)
      self._hard_recovery_prev = learned_actions.detach().clone()
    active = self._hard_recovery_active | hard_force
    below_release = force_n <= 45.0
    self._hard_release_steps = torch.where(
      active & below_release,
      self._hard_release_steps + 1,
      torch.zeros_like(self._hard_release_steps),
    )
    settled = active & (self._hard_release_steps >= 10)
    active = active & ~settled
    blend_steps = max(1, int(os.environ.get("GUIDE_DOG_HARD_FORCE_BLEND_STEPS", "8")))
    self._hard_recovery_steps = torch.where(
      active,
      torch.clamp(self._hard_recovery_steps + 1, max=blend_steps),
      torch.zeros_like(self._hard_recovery_steps),
    )
    alpha = (self._hard_recovery_steps.to(learned_actions.dtype) / float(blend_steps)).clamp(0.0, 1.0)
    blended = safe_actions + (self._hard_recovery_prev - safe_actions) * (1.0 - alpha).unsqueeze(1)
    output = torch.where(active.unsqueeze(1), blended, learned_actions)
    self._hard_recovery_active = active
    self._hard_recovery_prev = output.detach().clone()
    return output


class GuideDogTeacherOnPolicyRunner(GuideDogOnPolicyRunner):
  """Gate 2/2.1 runner that keeps the true-force teacher fully enabled."""

  def _teacher_command_weight(self) -> float:
    return 1.0


class GuideDogFadeOnPolicyRunner(GuideDogOnPolicyRunner):
  """Gate 4 runner with an explicit, checkpoint-aligned teacher fade."""

  teacher_fade_start_iteration = 3498
  teacher_fade_iterations = 2000

  def _teacher_command_weight(self) -> float:
    progress = max(0, self.current_learning_iteration - self.teacher_fade_start_iteration)
    return float(max(0.0, 1.0 - progress / self.teacher_fade_iterations))


class GuideDogCalibratedTeacherOnPolicyRunner(GuideDogTeacherOnPolicyRunner):
  """Gate 3.1 runner that balances force and no-force estimator examples."""

  active_force_weight = 5.0
  active_force_threshold = 2.0 / 80.0

  def _update_force_estimator(self) -> float:
    if not self._force_inputs:
      return 0.0
    inputs = torch.cat(self._force_inputs)
    labels = torch.cat(self._force_labels)
    self._force_inputs.clear()
    self._force_labels.clear()

    prediction = self.force_estimator(inputs)
    per_example_loss = torch.nn.functional.mse_loss(
      prediction, labels, reduction="none"
    ).mean(dim=-1)
    active = torch.linalg.vector_norm(labels[:, :2], dim=-1) > self.active_force_threshold
    weights = torch.where(
      active,
      torch.full_like(per_example_loss, self.active_force_weight),
      torch.ones_like(per_example_loss),
    )
    loss = torch.sum(weights * per_example_loss) / torch.sum(weights)
    self.force_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.force_estimator.parameters(), 1.0)
    self.force_optimizer.step()
    return float(loss.detach())


class GuideDogCalibratedFadeOnPolicyRunner(GuideDogCalibratedTeacherOnPolicyRunner):
  """Gate 4A: retain balanced estimator learning while fading the teacher."""

  teacher_fade_start_iteration = 4996
  teacher_fade_iterations = 2000

  def _teacher_command_weight(self) -> float:
    progress = max(0, self.current_learning_iteration - self.teacher_fade_start_iteration)
    return float(max(0.0, 1.0 - progress / self.teacher_fade_iterations))


class _GuideDogExpansionRunner(GuideDogCalibratedTeacherOnPolicyRunner):
  """Reintroduce a short teacher window for one new force-distribution gate."""

  stage_start_iteration = 0
  full_teacher_iterations = 500
  teacher_fade_iterations = 1000

  def _teacher_command_weight(self) -> float:
    relative = max(0, self.current_learning_iteration - self.stage_start_iteration)
    if relative < self.full_teacher_iterations:
      return 1.0
    fade = (relative - self.full_teacher_iterations) / self.teacher_fade_iterations
    return float(max(0.0, 1.0 - fade))


class GuideDogGate2FyRunner(_GuideDogExpansionRunner):
  stage_start_iteration = 5994


class GuideDogGate21FyActiveRunner(_GuideDogExpansionRunner):
  stage_start_iteration = 7993


class GuideDogGate3PlanarRunner(_GuideDogExpansionRunner):
  stage_start_iteration = 9492


class GuideDogGate31PlanarBalancedRunner(_GuideDogExpansionRunner):
  stage_start_iteration = 11491


class GuideDogGate43DRunner(_GuideDogExpansionRunner):
  stage_start_iteration = 13490


class GuideDogGate41SlowdownRunner(_GuideDogExpansionRunner):
  """Gate 4.1 command map with explicit braking under vertical load."""

  stage_start_iteration = 15489

  @staticmethod
  def _follow_command(estimated_force: torch.Tensor) -> torch.Tensor:
    follow = torch.sign(estimated_force[:, :2]) * torch.clamp(
      (torch.abs(estimated_force[:, :2] * 80.0) - 6.0) * 0.025, min=0.0
    )
    follow[:, 0] = torch.clamp(follow[:, 0], -0.35, 0.385)
    follow[:, 1] = torch.clamp(follow[:, 1], -0.224, 0.224)
    lateral_calibration = float(os.environ.get("FINETUNE_EXTERNAL_LATERAL_SCALE", "1.0"))
    if lateral_calibration <= 0.0 or lateral_calibration > 1.0:
      raise ValueError("FINETUNE_EXTERNAL_LATERAL_SCALE must be in (0, 1]")
    follow[:, 1] = follow[:, 1] * lateral_calibration
    forward_calibration = float(os.environ.get("FINETUNE_EXTERNAL_FORWARD_SCALE", "1.0"))
    if forward_calibration <= 0.0 or forward_calibration > 1.0:
      raise ValueError("FINETUNE_EXTERNAL_FORWARD_SCALE must be in (0, 1]")
    follow[:, 0] = follow[:, 0] * forward_calibration
    # High-force lateral pulls need a conservative speed target while the
    # stabilizing gait is being learned. Keep this deployment/training gate
    # opt-in so existing checkpoints retain their original command contract.
    legacy_scale = float(os.environ.get("GUIDE_DOG_HIGH_FORCE_SPEED_SCALE", "1.0"))
    mid_scale = float(os.environ.get("GUIDE_DOG_MID_FORCE_SPEED_SCALE", str(legacy_scale)))
    over_limit_scale = float(os.environ.get("GUIDE_DOG_OVER_LIMIT_SPEED_SCALE", str(legacy_scale)))
    force_n = torch.linalg.norm(estimated_force[:, :2] * 80.0, dim=1, keepdim=True)
    scale = torch.where(
      force_n >= 50.0,
      torch.full_like(force_n, over_limit_scale),
      torch.where(force_n >= 30.0, torch.full_like(force_n, mid_scale), torch.ones_like(force_n)),
    )
    command_gain = float(os.environ.get("GUIDE_DOG_FOLLOW_COMMAND_GAIN", "1.0"))
    if command_gain <= 0.0 or command_gain > 1.5:
      raise ValueError("GUIDE_DOG_FOLLOW_COMMAND_GAIN must be in (0, 1.5]")
    follow = follow * scale * command_gain
    vertical_scale = 1.0 - 0.35 * torch.clamp(
      (torch.abs(estimated_force[:, 2] * 80.0) - 8.0) / 20.0,
      0.0,
      1.0,
    )
    return follow * vertical_scale.unsqueeze(-1)


class GuideDogDirectionFinetuneRunner(GuideDogGate41SlowdownRunner):
  """Fine-tune the actor with the exact v16 deployment estimator in the loop."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    estimator_path = os.environ.get("DIRECTION_ESTIMATOR_TS", "")
    if not estimator_path:
      raise RuntimeError(
        "DIRECTION_ESTIMATOR_TS must point to the certified v16 estimator.ts"
      )
    self.direction_estimator = torch.jit.load(
      estimator_path, map_location=self.device
    ).eval()
    self.direction_history = torch.zeros(
      self.env.num_envs, 25, 64, device=self.device
    )
    self.direction_history_initialized = torch.zeros(
      self.env.num_envs, dtype=torch.bool, device=self.device
    )
    self.direction_previous_force_b = torch.zeros(
      self.env.num_envs, 3, device=self.device
    )

  def _direction_sensor(self, raw_actor: torch.Tensor) -> torch.Tensor:
    robot = self.env.unwrapped.scene["robot"].data
    return torch.cat(
      (
        raw_actor[:, 0:3],
        raw_actor[:, 6:9],
        raw_actor[:, 9:12],
        robot.root_link_quat_w,
        raw_actor[:, 18:30],
        raw_actor[:, 30:42],
        robot.actuator_force,
        raw_actor[:, 42:54],
        raw_actor[:, 13:16],
      ),
      dim=1,
    )

  def _policy_observation(
    self, raw_obs: TensorDict, dones: torch.Tensor | None = None
  ) -> TensorDict:
    """Reproduce the v16 certification observation path during PPO updates."""
    raw_actor = raw_obs["actor"]
    sensor = self._direction_sensor(raw_actor)
    reset = ~self.direction_history_initialized
    if dones is not None:
      reset |= dones.bool()
      self.direction_previous_force_b[dones.bool()] = 0.0
    if reset.any():
      self.direction_history[reset] = sensor[reset].unsqueeze(1)
      self.direction_history_initialized[reset] = True
    self.direction_history = torch.roll(self.direction_history, shifts=1, dims=1)
    self.direction_history[:, 0] = sensor

    with torch.no_grad():
      values, confidence = self.direction_estimator(self.direction_history)
    robot = self.env.unwrapped.scene["robot"].data
    root_quat_w = robot.root_link_quat_w
    velocity_b = quat_apply_inverse(root_quat_w, values[:, :3])
    # The frozen estimator is trained against tether.last_force_b, i.e. the
    # body-frame wrench.  Do not rotate it a second time here; only the
    # velocity output above is world-frame and needs world->body conversion.
    force_b_raw = values[:, 12:15]
    force_b_raw = torch.where(
      confidence[:, 3:4] >= 0.55,
      force_b_raw,
      torch.zeros_like(force_b_raw),
    )
    delta = torch.linalg.norm(
      force_b_raw - self.direction_previous_force_b, dim=1, keepdim=True
    )
    previous_active = torch.linalg.norm(
      self.direction_previous_force_b, dim=1, keepdim=True
    ) > 1.0
    alpha = torch.where(
      (delta > 15.0) & previous_active,
      torch.full_like(delta, 0.35),
      torch.ones_like(delta),
    )
    force_b = self.direction_previous_force_b + alpha * (
      force_b_raw - self.direction_previous_force_b
    )
    self.direction_previous_force_b = force_b.detach()

    actor_proprio = raw_actor[:, :-3].clone()
    actor_proprio[:, 3:6] = velocity_b
    actor_proprio[:, 12:13] = values[:, 3:4]
    actor_command = self._follow_command(force_b / 80.0)
    self._actor_command = actor_command.detach()
    actor_proprio[:, 13:15] = actor_command
    actor_proprio[:, 15] = 0.0
    actor_proprio[:, 16:18] = self._phase_from_command(actor_command)
    actor_proprio[:, 54:58] = values[:, 4:8]
    estimated_loads = torch.zeros(
      self.env.num_envs, 4, 3, device=self.device, dtype=actor_proprio.dtype
    )
    estimated_loads[:, :, 2] = -torch.log1p(values[:, 8:12] * 80.0)
    actor_proprio[:, 58:70] = estimated_loads.flatten(1)
    actor = torch.cat((actor_proprio, force_b / 80.0), dim=-1)

    obs = raw_obs.clone()
    obs["actor"] = actor
    critic = raw_obs["critic"].clone()
    critic[:, : self.actor_obs_dim] = actor
    obs["critic"] = critic
    return obs

  def _collect_force_examples(self, raw_obs: TensorDict) -> None:
    del raw_obs

  def _update_force_estimator(self) -> float:
    # v16 is frozen: this fine-tuning stage changes only actor/critic PPO state.
    return 0.0

  def load(self, path: str, *args, **kwargs):
    infos = super().load(path, *args, **kwargs)
    # Loading the original checkpoint also restores its optimizer learning
    # rate. Reapply the explicitly requested fine-tuning rate afterwards.
    learning_rate = float(os.environ.get("DIRECTION_FINETUNE_LR", "1e-4"))
    self.alg.learning_rate = learning_rate
    for param_group in self.alg.optimizer.param_groups:
      param_group["lr"] = learning_rate
    return infos


class GuideDogGate5WrenchRunner(_GuideDogExpansionRunner):
  stage_start_iteration = 17488



class GuideDogJointRobustActorRunner(GuideDogOnPolicyRunner):
  """Fine-tune the actor against one frozen deployment estimator."""

  shape_high_force_scale_default = 1.0
  frozen_default = "artifacts/v20_tau82_static_ramp_body_map_estimator/estimator.pt"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._load_frozen_estimator()
    self._robust_force_prev = torch.zeros(self.env.num_envs, 3, device=self.device)
    self._robust_force_bias = torch.zeros(self.env.num_envs, 3, device=self.device)
    # The old body-map estimator required a hand-tuned startup offset.  The
    # current-task estimator is trained against calibrated body-frame labels,
    # so applying that legacy offset creates a false 8 N pull before the gate
    # even sees the estimate.  Keep the override explicit for legacy runs,
    # but default to no bias for a deployable estimator checkpoint.
    bias_text = os.environ.get("GUIDE_DOG_STATIC_FORCE_BIAS_N", "0,0,0")
    try:
      bias_n = tuple(float(v.strip()) for v in bias_text.split(","))
      if len(bias_n) != 3:
        raise ValueError
    except ValueError:
      raise ValueError(
        "GUIDE_DOG_STATIC_FORCE_BIAS_N must be three comma-separated Newton values"
      ) from None
    self._static_force_bias = torch.tensor(bias_n, device=self.device).view(1, 3) / 80.0
    # Match the strict robust gate distribution: half of training envs see
    # deployment-like force perturbations, with teacher/nominal anchors kept
    # at one quarter each to limit forgetting.
    cohort = torch.arange(self.env.num_envs, device=self.device) % 4
    self._teacher_cohort = cohort == 0
    self._nominal_cohort = cohort == 1
    self._robust_cohort = cohort >= 2
    # Stateful command contract shared by training and play/sim2sim.  Ten
    # control steps are 200 ms at 50 Hz; fifteen steps are 300 ms.  The
    # hysteresis prevents estimator noise from starting gait, while the slew
    # limiter forces every reversal through zero instead of an instantaneous
    # positive-to-negative command jump.
    self._follow_active = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
    self._follow_on_steps = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)
    self._follow_off_steps = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)
    self._follow_command_state = torch.zeros(self.env.num_envs, 2, device=self.device)
    self._speed_feedback_integral = torch.zeros(self.env.num_envs, 2, device=self.device)
    # Smooth estimator spikes before hysteresis.  This is part of the
    # traction command contract so brief robust-mode outliers cannot create a
    # false gait request or prevent release from stopping.
    self._follow_force_state = torch.zeros(self.env.num_envs, 3, device=self.device)
    self._zero_bias_state = torch.zeros(self.env.num_envs, 3, device=self.device)
    self._zero_calibration_steps = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)
    self._zero_quiet_streak = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)

  def reset_follow_command_state(self, env_ids: torch.Tensor | None = None) -> None:
    # Viewer resets can be called outside the policy's inference context while
    # these state buffers may have been materialized as inference tensors.
    # Keep reset mutation inside InferenceMode so reconnect/reset cannot crash
    # the Viser process. These buffers are runtime state, never gradients.
    with torch.inference_mode():
      ids = torch.arange(self.env.num_envs, device=self.device) if env_ids is None else env_ids
      self._follow_active[ids] = False
      self._follow_on_steps[ids] = 0
      self._follow_off_steps[ids] = 0
      self._follow_command_state[ids] = 0.0
      self._speed_feedback_integral[ids] = 0.0
      self._follow_force_state[ids] = 0.0
      self._zero_bias_state[ids] = 0.0
      self._zero_calibration_steps[ids] = 0
      self._zero_quiet_streak[ids] = 0
      if hasattr(self, "_hard_recovery_active"):
        self._hard_recovery_active[ids] = False
        self._hard_recovery_steps[ids] = 0
        self._hard_release_steps[ids] = 0
        self._hard_recovery_prev[ids] = 0.0

  def _smooth_hard_force_recovery(
    self,
    learned_actions: torch.Tensor,
    safe_actions: torch.Tensor,
    hard_force: torch.Tensor,
    force_n: torch.Tensor,
  ) -> torch.Tensor:
    """Blend into base-only recovery and hold it through release dwell."""
    batch = learned_actions.shape[0]
    device = learned_actions.device
    if not hasattr(self, "_hard_recovery_active") or self._hard_recovery_active.shape[0] != batch:
      self._hard_recovery_active = torch.zeros(batch, dtype=torch.bool, device=device)
      self._hard_recovery_steps = torch.zeros(batch, dtype=torch.int32, device=device)
      self._hard_release_steps = torch.zeros(batch, dtype=torch.int32, device=device)
      self._hard_recovery_prev = learned_actions.detach().clone()
    active = self._hard_recovery_active | hard_force
    below_release = force_n <= 45.0
    self._hard_release_steps = torch.where(
      active & below_release,
      self._hard_release_steps + 1,
      torch.zeros_like(self._hard_release_steps),
    )
    settled = active & (self._hard_release_steps >= 10)
    active = active & ~settled
    blend_steps = max(1, int(os.environ.get("GUIDE_DOG_HARD_FORCE_BLEND_STEPS", "8")))
    self._hard_recovery_steps = torch.where(
      active,
      torch.clamp(self._hard_recovery_steps + 1, max=blend_steps),
      torch.zeros_like(self._hard_recovery_steps),
    )
    alpha = (self._hard_recovery_steps.to(learned_actions.dtype) / float(blend_steps)).clamp(0.0, 1.0)
    blended = safe_actions + (self._hard_recovery_prev - safe_actions) * (1.0 - alpha).unsqueeze(1)
    output = torch.where(active.unsqueeze(1), blended, learned_actions)
    self._hard_recovery_active = active
    self._hard_recovery_prev = output.detach().clone()
    return output

  def _shape_follow_command(self, force_norm: torch.Tensor, dones: torch.Tensor | None = None) -> torch.Tensor:
    """Apply the single v20 force-to-command contract in body frame."""
    if dones is not None and dones.any():
      self.reset_follow_command_state(dones.nonzero(as_tuple=False).flatten())
    alpha = 0.25
    self._follow_force_state = self._follow_force_state + alpha * (force_norm[:, :3] - self._follow_force_state)
    raw_force_n = self._follow_force_state * 80.0
    raw_horizontal_mag = torch.linalg.norm(raw_force_n[:, :2], dim=1)
    # State-conditioned zero-force calibration.  Only a quiet reset window
    # can establish the stance bias; a 15 N pull present at reset is never
    # absorbed into the zero point.
    quiet = raw_horizontal_mag < 8.0
    self._zero_quiet_streak = torch.where(
      quiet,
      self._zero_quiet_streak + 1,
      torch.zeros_like(self._zero_quiet_streak),
    )
    calibrating = (self._zero_calibration_steps < 80) & (self._zero_quiet_streak >= 5)
    self._zero_bias_state = torch.where(
      calibrating.unsqueeze(1),
      self._zero_bias_state * 0.90 + self._follow_force_state * 0.10,
      self._zero_bias_state,
    )
    self._zero_calibration_steps = torch.where(
      self._zero_calibration_steps < 80,
      self._zero_calibration_steps + 1,
      self._zero_calibration_steps,
    )
    force_n = (self._follow_force_state - self._zero_bias_state) * 80.0
    horizontal_mag = torch.linalg.norm(force_n[:, :2], dim=1)
    # A high-force event is an emergency safety condition, not another
    # command-shaping regime.  Use both the current input and filtered
    # estimate so the base-only cut cannot be delayed by the smoothing filter.
    hard_force_threshold_n = float(os.environ.get("GUIDE_DOG_HARD_FORCE_N", "50.0"))
    input_horizontal_mag = torch.linalg.norm(force_norm[:, :2] * 80.0, dim=1)
    hard_force = (input_horizontal_mag >= hard_force_threshold_n) | (
      horizontal_mag >= hard_force_threshold_n
    )
    # During a confirmed quiet reset window, finish zero-point calibration
    # before allowing gait to open. A real pull is not quiet and is unaffected.
    zero_startup = (self._zero_calibration_steps < 80) & quiet
    on = (horizontal_mag >= 6.0) & ~zero_startup
    off = horizontal_mag <= 4.0
    self._follow_on_steps = torch.where(on, self._follow_on_steps + 1, torch.zeros_like(self._follow_on_steps))
    self._follow_off_steps = torch.where(off, self._follow_off_steps + 1, torch.zeros_like(self._follow_off_steps))
    fast_open_threshold_n = float(os.environ.get("GUIDE_DOG_FAST_OPEN_THRESHOLD_N", "inf"))
    fast_open_steps = int(os.environ.get("GUIDE_DOG_FAST_OPEN_STEPS", "2"))
    open_steps = torch.where(
      horizontal_mag >= fast_open_threshold_n,
      torch.full_like(self._follow_on_steps, fast_open_steps),
      torch.full_like(self._follow_on_steps, 10),
    )
    next_follow_active = torch.where(
      self._follow_active,
      self._follow_off_steps < 15,
      self._follow_on_steps >= open_steps,
    )
    self._follow_active = torch.where(
      hard_force,
      torch.zeros_like(next_follow_active),
      next_follow_active,
    )

    follow_gain = float(os.environ.get("GUIDE_DOG_FOLLOW_GAIN", "0.025"))
    target = torch.sign(force_n[:, :2]) * torch.clamp((torch.abs(force_n[:, :2]) - 6.0) * follow_gain, min=0.0)
    # A small deployment safety scale for the historically unstable -Fy
    # quadrant. It preserves the sign and estimator path while reducing the
    # lateral command that can amplify roll on a randomized base state.
    neg_y_scale = float(os.environ.get("GUIDE_DOG_NEG_Y_SCALE", "1.0"))
    target[:, 1] = torch.where(force_n[:, 1] < 0.0, target[:, 1] * neg_y_scale, target[:, 1])
    # Once follow is active, retain a small directional command while the
    # estimate briefly falls into the 4--6 N hysteresis band.  The 6 N
    # opening and 4 N closing thresholds remain unchanged; this prevents
    # low-force magnitude jitter from repeatedly erasing the walking intent.
    low_band = self._follow_active & (horizontal_mag >= 4.0) & (horizontal_mag < 6.0)
    direction = force_n[:, :2] / horizontal_mag.unsqueeze(1).clamp_min(1.0e-6)
    target = torch.where(
      low_band.unsqueeze(1), direction * 0.03, target
    )
    target[:, 0] = torch.clamp(target[:, 0], -0.35, 0.385)
    y_cap = float(os.environ.get("GUIDE_DOG_Y_CMD_CAP", "0.224"))
    target[:, 1] = torch.clamp(target[:, 1], -y_cap, y_cap)
    vertical_scale = 1.0 - 0.35 * torch.clamp((torch.abs(force_n[:, 2]) - 8.0) / 20.0, 0.0, 1.0)
    target = target * vertical_scale.unsqueeze(-1)
    high_force_scale = float(
      os.environ.get(
        "GUIDE_DOG_SHAPE_HIGH_FORCE_SCALE",
        str(self.shape_high_force_scale_default),
      )
    )
    high_force = horizontal_mag >= 20.0
    target = torch.where(high_force.unsqueeze(1), target * high_force_scale, target)
    target = torch.where(self._follow_active.unsqueeze(1), target, torch.zeros_like(target))
    # Brake first for >=60-degree changes, covering +/-90 and 180 degrees.
    state_norm = torch.linalg.norm(self._follow_command_state, dim=1)
    target_norm = torch.linalg.norm(target, dim=1)
    direction_dot = (self._follow_command_state * target).sum(dim=1)
    # A direction change must first reach an actual zero command.  The old
    # 0.06 threshold allowed a residual 0.05 command to turn immediately.
    reversing = (state_norm > 1.0e-4) & (target_norm > 1.0e-6) & (direction_dot < 0.5 * state_norm * target_norm)
    target = torch.where(reversing.unsqueeze(1), torch.zeros_like(target), target)
    # 0.06 command units per control step gives a bounded, reversible slew.
    delta = torch.clamp(target - self._follow_command_state, -0.06, 0.06)
    next_command_state = self._follow_command_state + delta
    # Emergency base-only protection must not leave a residual command alive
    # for another slew step after the hard threshold is crossed.
    self._follow_command_state = torch.where(
      hard_force.unsqueeze(1),
      torch.zeros_like(next_command_state),
      next_command_state,
    )
    return self._follow_command_state.clone()

  def _load_frozen_estimator(self) -> None:
    frozen = os.environ.get("GUIDE_DOG_FROZEN_ESTIMATOR", self.frozen_default)
    frozen_path = Path(frozen)
    if frozen_path.is_dir():
      frozen_path = frozen_path / "estimator.pt"
    ckpt = torch.load(str(frozen_path), map_location=self.device, weights_only=False)
    self.force_estimator = ForceEstimator(
      int(ckpt.get("input_dim", 82)), int(ckpt.get("history_length", 25)),
      hidden_dim=int(ckpt.get("hidden_dim", 96)),
    ).to(self.device)
    self.force_estimator.load_state_dict(ckpt["model"])
    self.force_estimator.eval()
    for parameter in self.force_estimator.parameters():
      parameter.requires_grad_(False)
    self._force_mean = ckpt["mean"].to(self.device)
    self._force_std = ckpt["std"].to(self.device).clamp_min(1.0e-4)

  def _estimate_force(self, history: torch.Tensor) -> torch.Tensor:
    return self.force_estimator((history - self._force_mean) / self._force_std).detach()

  def _shape_high_force_actor_observation(self, force: torch.Tensor) -> torch.Tensor:
    """Optionally attenuate only the actor's high-force context channel.

    The command still uses the full calibrated force; this gate is useful for
    the high-force curriculum where a 30 N label otherwise drives the learned
    actor outside its stable lateral gait region. It is disabled by default.
    """
    scale = float(os.environ.get("GUIDE_DOG_HIGH_FORCE_OBS_SCALE", "1.0"))
    if scale >= 0.999:
      return force
    norm = torch.linalg.norm(force[:, :2], dim=1, keepdim=True)
    return torch.where(norm >= 20.0, force * scale, force)

  def _prepare_inference_force(self, estimated_force: torch.Tensor) -> torch.Tensor:
    """Apply the same calibrated force channel used by v20 actor training."""
    selected_force = estimated_force - self._static_force_bias
    horizontal = torch.linalg.norm(selected_force[:, :2], dim=1, keepdim=True)
    selected_force = selected_force * torch.clamp(
      (45.0 / 80.0) / horizontal.clamp_min(1.0e-6), max=1.0
    )
    low = (horizontal >= (6.0 / 80.0)) & (horizontal < (15.0 / 80.0))
    selected_force = selected_force.clone()
    selected_force[:, :2] = torch.where(
      low, selected_force[:, :2] * 1.5, selected_force[:, :2]
    )
    return selected_force

  def _inference_command_from_force(
    self, estimated_force: torch.Tensor
  ) -> torch.Tensor:
    """Use the v20 low-pass, hysteresis, slew, and reversal contract in play."""
    return self._shape_follow_command(estimated_force)

  def _perturb_estimated_force(self, nominal: torch.Tensor) -> torch.Tensor:
    """Deployment-like Fhat perturbation for the fixed robust cohort."""
    n = nominal.shape[0]
    g = torch.Generator(device=nominal.device)
    control_step = int(self.env.unwrapped.common_step_counter)
    g.manual_seed(int(self.current_learning_iteration * 1000003 + control_step * 1009 + 17))
    scale = torch.empty(n, 1, device=nominal.device).uniform_(0.80, 1.20, generator=g)
    noise = torch.randn(n, 3, device=nominal.device, generator=g) * 0.025
    self._robust_force_bias.mul_(0.995).add_(
      torch.randn(n, 3, device=nominal.device, generator=g) * 0.0015
    ).clamp_(-0.035, 0.035)
    angle = torch.empty(n, device=nominal.device).uniform_(-0.20, 0.20, generator=g)
    c, si = torch.cos(angle), torch.sin(angle)
    rotated = nominal.clone()
    rotated[:, 0] = c * nominal[:, 0] - si * nominal[:, 1]
    rotated[:, 1] = si * nominal[:, 0] + c * nominal[:, 1]
    perturbed = rotated * scale + self._robust_force_bias + noise
    # Sensor uncertainty must not turn a truly released/no-pull episode into
    # a gait request. Keep the horizontal perturbation below the 4 N closing
    # threshold whenever the nominal estimate is already in the no-force band.
    no_force = torch.linalg.norm(nominal[:, :2], dim=1) < (4.0 / 80.0)
    perturbed = torch.where(no_force.unsqueeze(1), perturbed * 0.0, perturbed)
    delayed = torch.rand(n, device=nominal.device, generator=g) < 0.15
    perturbed = torch.where(delayed.unsqueeze(1), self._robust_force_prev, perturbed)
    # Match the rollout robust contract: a delayed frame replays the already
    # perturbed force, not the clean nominal estimate.  Otherwise training
    # and certification see different direction/magnitude jumps.
    self._robust_force_prev = perturbed.detach()
    return perturbed.clamp(-1.5, 1.5)

  def _policy_observation(
    self, raw_obs: TensorDict, dones: torch.Tensor | None = None
  ) -> TensorDict:
    """Mix true, frozen-estimator and perturbed-estimator actor cohorts.

    The parent builds the deployment chain first.  We then replace one third
    of the environments with a true body-frame teacher.  The other two thirds
    retain the frozen estimator, whose output already contains both clean and
    perturbed cohorts from ``_estimate_force``.  This keeps the reward command,
    actor command, gait phase and force channel mutually consistent.
    """
    obs = super()._policy_observation(raw_obs, dones)
    if self.current_learning_iteration < self.gait_warmup_iterations:
      return obs
    actor = obs["actor"].clone()
    estimated_force = actor[:, -3:].clone()
    # Do not leak a previous episode's force estimate into a new robust
    # episode.  The delayed perturbation path can otherwise replay the last
    # non-zero force during no-pull_stand, creating a false gait command.
    estimated_force = estimated_force - self._static_force_bias
    if dones is not None and dones.any():
      ids = dones.nonzero(as_tuple=False).flatten()
      self._robust_force_prev[ids] = 0.0
      self._robust_force_bias[ids] = 0.0
    teacher_force = raw_obs["critic"][:, -3:].to(actor)
    perturbed_force = self._perturb_estimated_force(estimated_force)
    # Cohorts are fixed per environment. A robot never changes command source
    # in the middle of a force transition, which would create an artificial
    # command jump unrelated to the physical tether.
    selected_force = estimated_force.clone()
    selected_force[self._teacher_cohort] = teacher_force[self._teacher_cohort]
    selected_force[self._robust_cohort] = perturbed_force[self._robust_cohort]
    # The force channel must match the conditioned command source exactly.
    h = torch.linalg.norm(selected_force[:, :2], dim=1, keepdim=True)
    selected_force = selected_force * torch.clamp((45.0 / 80.0) / h.clamp_min(1.0e-6), max=1.0)
    low = (h >= (6.0 / 80.0)) & (h < (15.0 / 80.0))
    selected_force = selected_force.clone()
    selected_force[:, :2] = torch.where(low, selected_force[:, :2] * 1.5, selected_force[:, :2])
    # Shape the selected source, not each source independently. This keeps the
    # force channel, command, and gait phase exactly paired per environment.
    actor_command = self._shape_follow_command(selected_force, dones)
    if self._tether_event is not None:
      actor_command = torch.where(
        self._tether_event.retention_mask.unsqueeze(1),
        self._tether_event.retention_command,
        actor_command,
      )
    actor[:, 13:15] = actor_command
    actor[:, 15] = 0.0
    actor[:, 16:18] = self._phase_from_command(actor_command)
    actor[:, -3:] = self._shape_high_force_actor_observation(selected_force)
    obs["actor"] = actor
    critic = obs["critic"].clone()
    critic[:, : self.actor_obs_dim] = actor
    obs["critic"] = critic
    self._actor_command = actor_command.detach()
    return obs

  @staticmethod
  def _follow_command(estimated_force: torch.Tensor) -> torch.Tensor:
    """Robust body-force map used by the joint actor fine-tune.

    The old map left 8 N of dead zone and allowed high-force commands to reach
    the lateral speed limit too quickly.  This version activates at 6 N,
    gives a usable command around 10 N, and saturates earlier so vertical or
    diagonal pulls do not turn into an over-aggressive gait request.
    """
    force_n = estimated_force * 80.0
    follow = torch.sign(force_n[:, :2]) * torch.clamp(
      (torch.abs(force_n[:, :2]) - 6.0) * 0.025, min=0.0
    )
    follow[:, 0] = torch.clamp(follow[:, 0], -0.35, 0.385)
    follow[:, 1] = torch.clamp(follow[:, 1], -0.224, 0.224)
    # High-force lateral pulls need a conservative speed target while the
    # stabilizing gait is being learned. Keep this deployment/training gate
    # opt-in so existing checkpoints retain their original command contract.
    legacy_scale = float(os.environ.get("GUIDE_DOG_HIGH_FORCE_SPEED_SCALE", "1.0"))
    mid_scale = float(os.environ.get("GUIDE_DOG_MID_FORCE_SPEED_SCALE", str(legacy_scale)))
    over_limit_scale = float(os.environ.get("GUIDE_DOG_OVER_LIMIT_SPEED_SCALE", str(legacy_scale)))
    force_n = torch.linalg.norm(estimated_force[:, :2] * 80.0, dim=1, keepdim=True)
    scale = torch.where(
      force_n >= 50.0,
      torch.full_like(force_n, over_limit_scale),
      torch.where(force_n >= 30.0, torch.full_like(force_n, mid_scale), torch.ones_like(force_n)),
    )
    command_gain = float(os.environ.get("GUIDE_DOG_FOLLOW_COMMAND_GAIN", "1.0"))
    if command_gain <= 0.0 or command_gain > 1.5:
      raise ValueError("GUIDE_DOG_FOLLOW_COMMAND_GAIN must be in (0, 1.5]")
    follow = follow * scale * command_gain
    vertical_scale = 1.0 - 0.35 * torch.clamp(
      (torch.abs(force_n[:, 2]) - 8.0) / 20.0, 0.0, 1.0
    )
    return follow * vertical_scale.unsqueeze(-1)

  def _update_force_estimator(self) -> float:
    self._force_inputs.clear()
    self._force_labels.clear()
    return 0.0

  def load(self, path: str, *args, **kwargs):
    infos = super().load(path, *args, **kwargs)
    # Prefer an estimator serialized with the checkpoint.  Older v152
    # checkpoints may not contain one; only those fall back to the frozen
    # deployment estimator.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    estimator_state = checkpoint.get("force_estimator_state_dict")
    estimator_compatible = (
      estimator_state is not None
      and int(estimator_state["encoder.0.weight"].shape[1]) == self.force_input_dim
    )
    # An acceptance/training run can explicitly select a deployment estimator
    # via GUIDE_DOG_FROZEN_ESTIMATOR.  In that mode the actor checkpoint's
    # legacy estimator must not silently override the selected artifact.
    use_frozen = os.environ.get("GUIDE_DOG_USE_FROZEN_ESTIMATOR", "0") == "1"
    if use_frozen or not estimator_compatible:
      self._load_frozen_estimator()
      self._checkpoint_estimator_loaded = False
      self.force_bias.zero_()
    else:
      if checkpoint.get("force_estimator_input_mean") is not None:
        self._force_mean = checkpoint["force_estimator_input_mean"].to(self.device)
        self._force_std = checkpoint["force_estimator_input_std"].to(self.device).clamp_min(1.0e-4)
      # Supervised estimator checkpoints include their own label calibration;
      # do not apply the legacy frozen-estimator static bias or startup bias.
      self._checkpoint_estimator_loaded = True
      if hasattr(self, "_static_force_bias"):
        self._static_force_bias.zero_()
      self.force_bias.zero_()
      self.force_estimator.eval()
    return infos


class GuideDogV152Runner(GuideDogJointRobustActorRunner):
  """Low-rate v152 continuation with stage-controlled estimator robustness."""

  # v152 was trained with conservative commands for >=20 N. Keep the same
  # contract in play/Viser even when no launcher environment is present.
  shape_high_force_scale_default = 0.25
  # Preserve full corrective actions through normal traction tilt. The policy
  # is trained without the inference-only action attenuation.
  stabilize_start_deg_default = 15.0
  stabilize_full_deg_default = 30.0
  robust_training = False
  learning_rate = 5.0e-7
  # v152 starts from iteration 1000 of an already trained traction actor.
  # Re-entering the inherited force-free warmup until iteration 1500 changes
  # the command source seen by the policy and caused the attempt-1 regression.
  gait_warmup_iterations = 0

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # v152 is an autonomous force curriculum. Certification enables manual
    # force injection, while training must explicitly enable the event's
    # autonomous path; otherwise every curriculum force is silently zero.
    if self._tether_event is not None:
      self._tether_event.set_autonomous_enabled(True)
    # Preserve the certified locomotion representation by default. A targeted
    # continuation may opt into one additional hidden layer through the
    # environment variable without changing the v152 baseline behavior.
    layer_spec = os.environ.get("GUIDE_DOG_V152_ACTOR_LAYERS", "mlp.6.")
    trainable_prefixes = tuple(x.strip() for x in layer_spec.split(",") if x.strip())
    for name, parameter in self.alg.actor.named_parameters():
      parameter.requires_grad_(name.startswith(trainable_prefixes))
    if not self.robust_training:
      # Retain a 25% teacher anchor while training the other environments on
      # the frozen deployment estimator. Robust perturbations start in stage 3.
      cohort = torch.arange(self.env.num_envs, device=self.device) % 4
      self._teacher_cohort = cohort == 0
      self._nominal_cohort = cohort != 0
      self._robust_cohort = torch.zeros_like(self._teacher_cohort)
      if os.environ.get("GUIDE_DOG_V152_TEACHER_ONLY") == "1":
        # First learn a real high-force trot with an exact force signal.  A
        # later continuation restores nominal/robust estimator cohorts; this
        # avoids asking PPO to solve force estimation error and gait breakout
        # in the same update distribution.
        self._teacher_cohort[:] = True
        self._nominal_cohort[:] = False

  def _perturb_estimated_force(self, nominal: torch.Tensor) -> torch.Tensor:
    if not self.robust_training:
      return nominal
    previous = self._robust_force_prev.clone()
    perturbed = super()._perturb_estimated_force(nominal)
    # Explicit output dropout complements the scale, direction, bias, noise,
    # and one-frame delay family in the inherited perturbation contract.
    g = torch.Generator(device=nominal.device)
    control_step = int(self.env.unwrapped.common_step_counter)
    g.manual_seed(int(self.current_learning_iteration * 1000003 + control_step * 1009 + 152))
    dropped = torch.rand(nominal.shape[0], device=nominal.device, generator=g) < 0.08
    return torch.where(dropped.unsqueeze(1), previous, perturbed)

  def load(self, path: str, *args, **kwargs):
    infos = super().load(path, *args, **kwargs)
    # The first branch from v151 keeps model weights but discards its Adam
    # momentum. Subsequent v152 chunks retain optimizer state.
    if "go2_traction_mjlab_v151" in str(path):
      self.alg.optimizer.state.clear()
    self.alg.learning_rate = float(
      os.environ.get("GUIDE_DOG_V152_LR", str(self.learning_rate))
    )
    for param_group in self.alg.optimizer.param_groups:
      param_group["lr"] = self.learning_rate
    return infos


class GuideDogV152Stage1Runner(GuideDogV152Runner):
  pass


class GuideDogV152Stage2Runner(GuideDogV152Runner):
  pass


class GuideDogV152Stage3Runner(GuideDogV152Runner):
  robust_training = True
