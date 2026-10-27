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
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class GuideDogOnPolicyRunner(VelocityOnPolicyRunner):
  """PPO runner with a jointly trained GuideDog-style force estimator."""

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
    self.force_history = torch.zeros(
      self.env.num_envs,
      self.history_length,
      self.force_input_dim,
      device=self.device,
    )
    self._force_inputs: list[torch.Tensor] = []
    self._force_labels: list[torch.Tensor] = []
    self._actor_command = torch.zeros(self.env.num_envs, 2, device=self.device)
    # Play-only switch for isolating locomotion from force following.
    self._inference_command_source = "estimated_force"
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
    progress = self.current_learning_iteration
    full_teacher_until = self.gait_warmup_iterations + self.force_teacher_iterations
    if progress < full_teacher_until:
      return 1.0
    fade_progress = (progress - full_teacher_until) / self.teacher_fade_iterations
    return float(max(0.0, 1.0 - fade_progress))

  def _phase_from_command(self, command: torch.Tensor) -> torch.Tensor:
    """Generate gait phase from the exact command shown to the actor."""
    env = self.env.unwrapped
    global_phase = (env.episode_length_buf * env.step_dt) % 0.6 / 0.6
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
  def _follow_command(estimated_force: torch.Tensor) -> torch.Tensor:
    """Map normalized Fhat to the same velocity command in train and play."""
    follow = torch.sign(estimated_force[:, :2]) * torch.clamp(
      (torch.abs(estimated_force[:, :2] * 80.0) - 8.0) * 0.02, min=0.0
    )
    follow[:, 0] = torch.clamp(follow[:, 0], -0.6, 0.8)
    follow[:, 1] = torch.clamp(follow[:, 1], -0.4, 0.4)
    vertical_scale = 1.0 - 0.6 * torch.clamp(
      (torch.abs(estimated_force[:, 2] * 80.0) - 10.0) / 25.0, 0.0, 1.0
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
      estimated_force = self._estimate_force(self.force_history)

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
    self._actor_command = actor_command.detach()
    actor_proprio = proprio.clone()
    actor_proprio[:, 13:15] = actor_command
    actor_proprio[:, 15] = 0.0
    actor_proprio[:, 16:18] = self._phase_from_command(actor_command)
    obs = raw_obs.clone()
    actor = torch.cat((actor_proprio, estimated_force), dim=-1)
    obs["actor"] = actor
    # Critic starts with the actor terms, then has privileged terms including
    # the true external-force label at its final three entries.
    critic = raw_obs["critic"].clone()
    critic[:, : self.actor_obs_dim] = actor
    obs["critic"] = critic
    return obs

  def _estimate_force(self, history: torch.Tensor) -> torch.Tensor:
    return self.force_estimator(history).detach()

  def _estimate_force(self, history: torch.Tensor) -> torch.Tensor:
    return self.force_estimator(history).detach()

  def _estimate_force(self, history: torch.Tensor) -> torch.Tensor:
    return self.force_estimator(history).detach()

  def _apply_actor_command_to_env(self) -> None:
    """Make the reward target exactly match the command in the actor input."""
    cmd = self.env.unwrapped.command_manager.get_term("twist").vel_command_b
    cmd[:, :2] = self._actor_command
    cmd[:, 2] = 0.0

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
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
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
          intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
          raw_obs, obs = raw_next, next_obs
        collect_time = time.time() - start
        self.alg.compute_returns(obs)

      update_start = time.time()
      loss_dict = self.alg.update()
      loss_dict["force_estimator_loss"] = self._update_force_estimator()
      loss_dict["teacher_command_weight"] = self._teacher_command_weight()
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
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
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
    torch.save(checkpoint, path)

    export_dir = os.path.dirname(path)
    estimator_path = os.path.join(export_dir, "force_estimator.onnx")
    estimator = self.force_estimator.to("cpu").eval()
    dummy = torch.zeros(1, self.history_length, self.force_input_dim)
    torch.onnx.export(
      estimator,
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
    if ("force_estimator_state_dict" in checkpoint
        and checkpoint.get("force_estimator_input_dim", self.force_input_dim) == self.force_input_dim):
      self.force_estimator.load_state_dict(checkpoint["force_estimator_state_dict"])
      if "force_optimizer_state_dict" in checkpoint:
        self.force_optimizer.load_state_dict(checkpoint["force_optimizer_state_dict"])
    return infos

  def set_inference_command_source(self, source: str) -> None:
    if source not in {"estimated_force", "direct_twist", "true_force_teacher"}:
      raise ValueError(f"Unsupported inference command source: {source}")
    self._inference_command_source = source

  def set_inference_teacher_force(self, force_n: torch.Tensor) -> None:
    """Set the body-frame force used only by the Gate 2 teacher play mode."""
    force_n = force_n.to(device=self.device, dtype=self._inference_teacher_force.dtype)
    if force_n.ndim == 1:
      force_n = force_n.unsqueeze(0).expand(self.env.num_envs, -1)
    self._inference_teacher_force.copy_(force_n / 80.0)

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
      estimated_force = self.force_estimator(history)
      self.last_estimated_force = estimated_force.detach()
      actor_proprio = proprio.clone()
      if self._inference_command_source == "estimated_force":
        # Deployment path: own-state history -> Fhat -> follow command.
        actor_proprio[:, 13:15] = self._follow_command(
          self.command_effective_force(estimated_force)
        )
      elif self._inference_command_source == "true_force_teacher":
        # Gate 2 diagnostic: isolate locomotion under pull from estimator error.
        actor_proprio[:, 13:15] = self._follow_command(
          self._inference_teacher_force[:actor_proprio.shape[0]]
        )
      # direct_twist retains the Viser Twist command in indices 13:15.
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
      actor = torch.cat((actor_proprio, estimated_force), dim=-1)
      if os.environ.get("GUIDE_DOG_TRACE_OBS") == "1" and not getattr(
        inference, "_trace_written", False
      ):
        trace = actor[0].detach().cpu().tolist()
        print("PYTHON_ACTOR73=" + repr(trace), flush=True)
        inference._trace_written = True
      obs = raw_obs.clone()
      obs["actor"] = actor
      return policy(obs)

    def reset() -> None:
      # The viewer calls policy.reset() with every environment reset.  Clear the
      # temporal estimator state so a previous pull cannot generate a stale command.
      nonlocal history
      history = None
      self.last_estimated_force = None
      self.last_actor_command = None

    inference.reset = reset
    return inference


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
      (torch.abs(estimated_force[:, :2] * 80.0) - 8.0) * 0.02, min=0.0
    )
    follow[:, 0] = torch.clamp(follow[:, 0], -0.6, 0.8)
    follow[:, 1] = torch.clamp(follow[:, 1], -0.4, 0.4)
    # No slowdown below 5 N; smoothly reach 55% speed at |Fz|=20 N.
    vertical_scale = 1.0 - 0.45 * torch.clamp(
      (torch.abs(estimated_force[:, 2] * 80.0) - 5.0) / 15.0,
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
    force_b_raw = quat_apply_inverse(root_quat_w, values[:, 12:15])
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

  frozen_default = "artifacts/v20_tau82_static_ramp_body_map_estimator/estimator.pt"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._load_frozen_estimator()
    self._robust_force_prev = torch.zeros(self.env.num_envs, 3, device=self.device)
    self._robust_force_bias = torch.zeros(self.env.num_envs, 3, device=self.device)
    # Stateful command contract shared by training and play/sim2sim.  Ten
    # control steps are 200 ms at 50 Hz; fifteen steps are 300 ms.  The
    # hysteresis prevents estimator noise from starting gait, while the slew
    # limiter forces every reversal through zero instead of an instantaneous
    # positive-to-negative command jump.
    self._follow_active = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
    self._follow_on_steps = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)
    self._follow_off_steps = torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.device)
    self._follow_command_state = torch.zeros(self.env.num_envs, 2, device=self.device)

  def reset_follow_command_state(self, env_ids: torch.Tensor | None = None) -> None:
    ids = torch.arange(self.env.num_envs, device=self.device) if env_ids is None else env_ids
    self._follow_active[ids] = False
    self._follow_on_steps[ids] = 0
    self._follow_off_steps[ids] = 0
    self._follow_command_state[ids] = 0.0

  def _shape_follow_command(self, force_norm: torch.Tensor, dones: torch.Tensor | None = None) -> torch.Tensor:
    """Apply the single v20 force-to-command contract in body frame."""
    if dones is not None and dones.any():
      self.reset_follow_command_state(dones.nonzero(as_tuple=False).flatten())
    force_n = force_norm[:, :3] * 80.0
    horizontal_mag = torch.linalg.norm(force_n[:, :2], dim=1)
    on = horizontal_mag >= 6.0
    off = horizontal_mag <= 4.0
    self._follow_on_steps = torch.where(on, self._follow_on_steps + 1, torch.zeros_like(self._follow_on_steps))
    self._follow_off_steps = torch.where(off, self._follow_off_steps + 1, torch.zeros_like(self._follow_off_steps))
    self._follow_active = torch.where(self._follow_active, self._follow_off_steps < 15, self._follow_on_steps >= 10)

    target = torch.sign(force_n[:, :2]) * torch.clamp((torch.abs(force_n[:, :2]) - 6.0) * 0.025, min=0.0)
    target[:, 0] = torch.clamp(target[:, 0], -0.50, 0.55)
    target[:, 1] = torch.clamp(target[:, 1], -0.32, 0.32)
    vertical_scale = 1.0 - 0.35 * torch.clamp((torch.abs(force_n[:, 2]) - 8.0) / 20.0, 0.0, 1.0)
    target = target * vertical_scale.unsqueeze(-1)
    target = torch.where(self._follow_active.unsqueeze(1), target, torch.zeros_like(target))
    # 0.06 command units per control step gives a bounded, reversible slew.
    delta = torch.clamp(target - self._follow_command_state, -0.06, 0.06)
    self._follow_command_state = self._follow_command_state + delta
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
    nominal = self.force_estimator((history - self._force_mean) / self._force_std).detach()
    n = nominal.shape[0]
    g = torch.Generator(device=nominal.device)
    control_step = int(self.env.unwrapped.common_step_counter)
    g.manual_seed(int(self.current_learning_iteration * 1000003 + control_step * 1009 + 17))
    # Across every rollout batch retain a clean frozen-estimator cohort and
    # expose the rest to deployment-like scale, direction, bias, noise and
    # one-control-step delay. The estimator itself remains frozen.
    keep = torch.rand(n, device=nominal.device, generator=g) < 0.33
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
    delayed = torch.rand(n, device=nominal.device, generator=g) < 0.15
    perturbed = torch.where(delayed.unsqueeze(1), self._robust_force_prev, perturbed)
    self._robust_force_prev = nominal.detach()
    return torch.where(keep.unsqueeze(1), nominal, perturbed).clamp(-1.5, 1.5)

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
    teacher_force = raw_obs["critic"][:, -3:].to(actor)
    n = actor.shape[0]
    g = torch.Generator(device=actor.device)
    control_step = int(self.env.unwrapped.common_step_counter)
    g.manual_seed(int(self.current_learning_iteration * 2000003 + control_step * 2017 + 29))
    teacher_mask = torch.rand(n, device=actor.device, generator=g) < (1.0 / 3.0)
    selected_force = torch.where(
      teacher_mask.unsqueeze(1), teacher_force, estimated_force
    )
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
    actor[:, -3:] = selected_force
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
    follow[:, 0] = torch.clamp(follow[:, 0], -0.50, 0.55)
    follow[:, 1] = torch.clamp(follow[:, 1], -0.32, 0.32)
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
    self._load_frozen_estimator()
    return infos
