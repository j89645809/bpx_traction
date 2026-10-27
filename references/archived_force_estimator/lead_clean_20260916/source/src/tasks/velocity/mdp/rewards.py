from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_direction_balanced_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  lateral_weight: float = 3.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Velocity tracking reward with explicit lateral/diagonal coverage.

  The ordinary isotropic reward under-weights Y-dominant commands because the
  certified X command is larger.  Weighting the lateral error only when the
  requested direction has a lateral component gives the policy a learning
  signal for Y and diagonal pulls without changing the command map or the
  safety gate.
  """
  if std <= 0.0 or lateral_weight < 1.0:
    raise ValueError("std must be positive and lateral_weight must be >= 1")
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]
  lateral_requested = torch.abs(command[:, 1]) > 0.04
  weights = torch.ones_like(error)
  weights[:, 1] = torch.where(
    lateral_requested,
    torch.full_like(weights[:, 1], lateral_weight),
    torch.ones_like(weights[:, 1]),
  )
  xy_error = torch.sum(weights * torch.square(error), dim=1)
  z_error = torch.square(asset.data.root_link_lin_vel_b[:, 2])
  return torch.exp(-(xy_error + 2.0 * z_error) / std**2)


def track_velocity_ratio(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  lateral_weight: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward relative command tracking so small lateral commands matter.

  Absolute velocity error overweights the larger X command.  Normalizing by
  requested command magnitude makes the reward target the actual/theoretical
  ratio directly while remaining bounded near standstill.
  """
  if std <= 0.0 or lateral_weight < 1.0:
    raise ValueError("std must be positive and lateral_weight must be >= 1")
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  target = command[:, :2]
  actual = asset.data.root_link_lin_vel_b[:, :2]
  norm = torch.linalg.vector_norm(target, dim=1, keepdim=True).clamp_min(0.06)
  relative_error = (actual - target) / norm
  weights = torch.ones_like(relative_error)
  weights[:, 1] = torch.where(
    torch.abs(target[:, 1]) > 0.04,
    torch.full_like(weights[:, 1], lateral_weight),
    torch.ones_like(weights[:, 1]),
  )
  error = torch.sum(weights * torch.square(relative_error), dim=1)
  return torch.exp(-error / std**2)


def track_low_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  command_min: float = 0.06,
  command_max: float = 0.16,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Give low-force follow commands a useful locomotion learning signal.

  The generic velocity reward is intentionally broad and differs only weakly
  between standing still and tracking a 0.1 command.  This masked, narrower
  term targets the 10 N startup/turn cases without increasing the incentive
  to run faster under high pulls.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  command_norm = torch.linalg.norm(command[:, :2], dim=1)
  active = (command_norm >= command_min) & (command_norm <= command_max)
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-error / std**2) * active.float()


def align_low_force_direction(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_min: float = 0.06,
  command_max: float = 0.16,
  scale: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward signed body-velocity projection for low-force commands.

  This term is bounded and direction-only.  It rewards the correct sign for
  10 N turns without increasing the incentive to run faster in high-force or
  three-dimensional jump cases.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_xy = command[:, :2]
  command_norm = torch.linalg.norm(cmd_xy, dim=1)
  active = (command_norm >= command_min) & (command_norm <= command_max)
  direction = cmd_xy / command_norm.clamp_min(1.0e-6).unsqueeze(1)
  projection = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * direction, dim=1)
  return torch.tanh(projection / scale) * active.float()


def traction_force_direction_cosine(
  env: ManagerBasedRlEnv,
  min_force_n: float = 15.0,
  max_force_n: float = 30.0,
  release_margin_n: float = 4.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward body velocity aligned with the applied body-frame traction force."""
  tether = env.event_manager.get_term_cfg("tether_pull").func
  force = tether.last_force_b[:, :2]
  velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]
  force_norm = torch.linalg.norm(force, dim=1)
  vel_norm = torch.linalg.norm(velocity, dim=1)
  active = (force_norm >= min_force_n) & (force_norm <= max_force_n) & (force_norm > release_margin_n)
  cosine = torch.sum(force * velocity, dim=1) / (force_norm * vel_norm).clamp_min(1.0e-4)
  return cosine.clamp(-1.0, 1.0) * active.float()


def traction_force_velocity_projection(
  env: ManagerBasedRlEnv,
  target_speed: float = 0.10,
  scale: float = 0.08,
  min_force_n: float = 15.0,
  max_force_n: float = 30.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward positive velocity projection in the applied traction direction."""
  tether = env.event_manager.get_term_cfg("tether_pull").func
  force = tether.last_force_b[:, :2]
  velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]
  force_norm = torch.linalg.norm(force, dim=1)
  direction = force / force_norm.clamp_min(1.0e-4).unsqueeze(1)
  projection = torch.sum(velocity * direction, dim=1)
  active = (force_norm >= min_force_n) & (force_norm <= max_force_n)
  return torch.exp(-torch.square(projection - target_speed) / (scale * scale)) * active.float()


def traction_velocity_progress(
  env: ManagerBasedRlEnv,
  target_speed: float = 0.14,
  min_force_n: float = 10.0,
  max_force_n: float = 32.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense signed progress reward that cannot be maximized by bracing.

  The Gaussian tracking term is nearly flat when a policy starts close to
  zero velocity.  This linear, clipped term supplies a useful improvement
  signal from the first centimetre per second while still bounding reward.
  """
  tether = env.event_manager.get_term_cfg("tether_pull").func
  force = tether.last_force_b[:, :2]
  force_norm = torch.linalg.vector_norm(force, dim=1)
  direction = force / force_norm.clamp_min(1.0e-6).unsqueeze(1)
  velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]
  projection = torch.sum(velocity * direction, dim=1)
  active = (force_norm >= min_force_n) & (force_norm <= max_force_n)
  return torch.clamp(projection / target_speed, -1.0, 1.0) * active.float()


def traction_trot_contact_reward(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  stance_fraction: float,
  sensor_name: str,
  min_force_n: float = 10.0,
  max_force_n: float = 32.0,
) -> torch.Tensor:
  """Reward the complete diagonal contact mask during an active pull.

  The generic per-foot match reward gives an all-stance policy 0.5 forever.
  Subtracting that bracing baseline makes four-foot support worth zero while
  the intended diagonal pair reaches one.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None
  is_contact = sensor.data.current_contact_time > 0.0
  phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=phase.dtype).view(1, -1)
  desired_contact = ((phase + offsets) % 1.0) < stance_fraction
  match = (desired_contact == is_contact).float().mean(dim=1)
  tether = env.event_manager.get_term_cfg("tether_pull").func
  magnitude = torch.linalg.vector_norm(tether.last_force_b[:, :2], dim=1)
  active = (magnitude >= min_force_n) & (magnitude <= max_force_n)
  return torch.clamp((match - 0.5) * 2.0, min=0.0) * active.float()


def traction_trot_swing_reward(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  stance_fraction: float,
  target_height: float,
  sensor_name: str,
  min_force_n: float = 10.0,
  max_force_n: float = 32.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense reward for lifting the scheduled diagonal swing pair."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_air = sensor.data.found == 0
  if in_air.ndim == 3:
    in_air = in_air.squeeze(-1)
  phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=phase.dtype).view(1, -1)
  desired_swing = ((phase + offsets) % 1.0) >= stance_fraction
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  lift = torch.clamp(foot_z / target_height, 0.0, 1.0)
  swing_score = (lift * in_air.float() * desired_swing.float()).sum(dim=1)
  swing_score = swing_score / desired_swing.float().sum(dim=1).clamp_min(1.0)
  tether = env.event_manager.get_term_cfg("tether_pull").func
  magnitude = torch.linalg.vector_norm(tether.last_force_b[:, :2], dim=1)
  active = (magnitude >= min_force_n) & (magnitude <= max_force_n)
  return swing_score * active.float()


def traction_trot_min_swing_reward(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  stance_fraction: float,
  target_height: float,
  sensor_name: str,
  min_force_n: float = 10.0,
  max_force_n: float = 18.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the weaker foot of each scheduled diagonal swing pair.

  This prevents one high-clearance leg from compensating for a dragging mate
  in the pair, which the average swing reward permits.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_air = sensor.data.found == 0
  if in_air.ndim == 3:
    in_air = in_air.squeeze(-1)
  phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=phase.dtype).view(1, -1)
  desired_swing = ((phase + offsets) % 1.0) >= stance_fraction
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  score = torch.clamp(foot_z / target_height, 0.0, 1.0) * in_air.float()
  score = torch.where(desired_swing, score, torch.ones_like(score))
  weaker_swing = score.min(dim=1).values
  tether = env.event_manager.get_term_cfg("tether_pull").func
  magnitude = torch.linalg.vector_norm(tether.last_force_b[:, :2], dim=1)
  active = (magnitude >= min_force_n) & (magnitude <= max_force_n)
  return weaker_swing * active.float()


def traction_all_stance_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_force_n: float = 10.0,
  max_force_n: float = 32.0,
) -> torch.Tensor:
  """Penalize the four-leg bracing solution during commanded traction."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contact = sensor.data.found > 0
  if contact.ndim == 3:
    contact = contact.squeeze(-1)
  tether = env.event_manager.get_term_cfg("tether_pull").func
  magnitude = torch.linalg.vector_norm(tether.last_force_b[:, :2], dim=1)
  active = (magnitude >= min_force_n) & (magnitude <= max_force_n)
  return contact.all(dim=1).float() * active.float()


def traction_effort_saturation_l2(
  env: ManagerBasedRlEnv,
  effort_limits: tuple[float, ...],
  threshold: float = 0.85,
  min_force_n: float = 10.0,
  max_force_n: float = 32.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize sustained joint efforts approaching actuator limits under pull."""
  asset: Entity = env.scene[asset_cfg.name]
  effort = torch.abs(asset.data.actuator_force.reshape(env.num_envs, -1))
  limits = torch.as_tensor(
    effort_limits, device=effort.device, dtype=effort.dtype
  ).view(1, -1)
  excess = torch.clamp(effort / limits - threshold, min=0.0)
  tether = env.event_manager.get_term_cfg("tether_pull").func
  magnitude = torch.linalg.vector_norm(tether.last_force_b[:, :2], dim=1)
  active = (magnitude >= min_force_n) & (magnitude <= max_force_n)
  return torch.sum(torch.square(excess), dim=1) * active.float()


def traction_release_gate_l2(
  env: ManagerBasedRlEnv,
  release_force_n: float = 4.0,
  command_scale: float = 0.5,
  velocity_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize stale command and drift after the applied traction is released."""
  tether = env.event_manager.get_term_cfg("tether_pull").func
  force_norm = torch.linalg.norm(tether.last_force_b[:, :2], dim=1)
  command = env.command_manager.get_term("twist").vel_command_b[:, :2]
  velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]
  released = force_norm <= release_force_n
  return (command_scale * torch.sum(command.square(), dim=1) + velocity_scale * torch.sum(velocity.square(), dim=1)) * released.float()

def high_force_hold_stability_l2(
  env: ManagerBasedRlEnv,
  min_force_n: float = 28.0,
  max_force_n: float = 35.0,
  angular_scale: float = 0.35,
  action_rate_scale: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense upright/action-rate penalty for high-force horizontal holds."""
  tether = env.event_manager.get_term_cfg("tether_pull").func
  magnitude = torch.linalg.vector_norm(tether.last_force_w[:, :2], dim=1)
  active = (magnitude >= min_force_n) & (magnitude <= max_force_n)
  asset: Entity = env.scene[asset_cfg.name]
  orientation = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  angular = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
  return (orientation + angular_scale * angular + action_rate_scale * action_rate) * active.float()


def positive_axis_force_mask(
  force_b: torch.Tensor,
  min_force_n: float,
  max_force_n: float,
  direction_cosine_min: float,
) -> torch.Tensor:
  """Select high-force +X/+Y axis cohorts without including diagonals."""
  if force_b.ndim != 2 or force_b.shape[1] != 3:
    raise ValueError("force_b must have shape (N, 3)")
  if min_force_n < 0.0 or max_force_n <= min_force_n:
    raise ValueError("invalid directional-slip force range")
  if not 0.0 < direction_cosine_min <= 1.0:
    raise ValueError("direction_cosine_min must be in (0, 1]")
  planar = force_b[:, :2]
  magnitude = torch.linalg.vector_norm(planar, dim=1)
  unit = planar / magnitude.unsqueeze(1).clamp_min(1.0e-6)
  positive_axis_cosine = torch.maximum(unit[:, 0], unit[:, 1])
  return (
    (magnitude >= min_force_n)
    & (magnitude <= max_force_n)
    & (positive_axis_cosine >= direction_cosine_min)
  )


def gate_directional_foot_slip_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_force_n: float = 45.0,
  max_force_n: float = 52.0,
  direction_cosine_min: float = 0.85,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense slip precursor for the measured +X/+Y pull and reverse failures."""
  tether = env.event_manager.get_term_cfg("tether_pull").func
  phase = getattr(tether, "_gate_previous_phase", None)
  active = positive_axis_force_mask(
    tether._gate_base_force,
    min_force_n,
    max_force_n,
    direction_cosine_min,
  )
  if phase not in {"pull", "reverse"}:
    active = torch.zeros_like(active)
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  slip_l2 = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact, dim=1)
  env.extras["log"]["Metrics/r1d_directional_slip_active_fraction"] = active.float().mean()
  return slip_l2 * active.float()


def gate_reverse_recovery_l2(
  env: ManagerBasedRlEnv,
  min_force_n: float = 45.0,
  max_force_n: float = 52.0,
  speed_scale: float = 0.20,
  angular_scale: float = 0.50,
  action_rate_scale: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Transition-only penalty for residual motion when reverse is confirmed.

  This is deliberately inactive during pull and release.  It targets the
  short reverse/recovery window where the old pull velocity, body tilt and
  abrupt action changes combine to trigger illegal contact.
  """
  tether = env.event_manager.get_term_cfg("tether_pull").func
  phase = getattr(tether, "_gate_previous_phase", None)
  force = tether._gate_base_force
  magnitude = torch.linalg.vector_norm(force[:, :2], dim=1)
  active = (
    (phase == "reverse")
    & (magnitude >= min_force_n)
    & (magnitude <= max_force_n)
  )
  asset: Entity = env.scene[asset_cfg.name]
  orientation = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  angular = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  speed = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  action_rate = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )
  return (
    orientation
    + speed_scale * speed
    + angular_scale * angular
    + action_rate_scale * action_rate
  ) * active.float()


def lateral_high_force_hold_stability_l2(
  env: ManagerBasedRlEnv,
  min_force_n: float = 28.0,
  max_force_n: float = 35.0,
  direction_cosine_min: float = 0.85,
  angular_scale: float = 0.35,
  action_rate_scale: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Directional stability penalty for sustained +/-Y high-force holds."""
  tether = env.event_manager.get_term_cfg("tether_pull").func
  force = tether.last_force_w[:, :2]
  magnitude = torch.linalg.vector_norm(force, dim=1)
  cosine_y = force[:, 1] / magnitude.clamp_min(1.0e-6)
  active = (
    (magnitude >= min_force_n)
    & (magnitude <= max_force_n)
    & (torch.abs(cosine_y) >= direction_cosine_min)
    & (~tether.v152_release_variant)
  )
  asset: Entity = env.scene[asset_cfg.name]
  orientation = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  angular = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
  return (orientation + angular_scale * angular + action_rate_scale * action_rate) * active.float()


def negative_lateral_high_force_hold_stability_l2(
  env: ManagerBasedRlEnv,
  min_force_n: float = 28.0,
  max_force_n: float = 35.0,
  direction_cosine_min: float = 0.85,
  angular_scale: float = 0.45,
  action_rate_scale: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Extra stability shaping for the historically fragile -Y hold."""
  tether = env.event_manager.get_term_cfg("tether_pull").func
  force = tether.last_force_w[:, :2]
  magnitude = torch.linalg.vector_norm(force, dim=1)
  cosine_y = force[:, 1] / magnitude.clamp_min(1.0e-6)
  active = (
    (magnitude >= min_force_n)
    & (magnitude <= max_force_n)
    & (cosine_y <= -direction_cosine_min)
    & (~tether.v152_release_variant)
  )
  asset: Entity = env.scene[asset_cfg.name]
  orientation = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  angular = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
  return (orientation + angular_scale * angular + action_rate_scale * action_rate) * active.float()


def negative_lateral_high_force_hold_stability_l2(
  env: ManagerBasedRlEnv,
  min_force_n: float = 28.0,
  max_force_n: float = 35.0,
  direction_cosine_min: float = 0.85,
  angular_scale: float = 0.45,
  action_rate_scale: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Extra stability shaping for the historically fragile -Y hold."""
  tether = env.event_manager.get_term_cfg("tether_pull").func
  force = tether.last_force_w[:, :2]
  magnitude = torch.linalg.vector_norm(force, dim=1)
  cosine_y = force[:, 1] / magnitude.clamp_min(1.0e-6)
  active = ((magnitude >= min_force_n) & (magnitude <= max_force_n)
    & (cosine_y <= -direction_cosine_min) & (~tether.v152_release_variant))
  asset: Entity = env.scene[asset_cfg.name]
  orientation = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  angular = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
  return (orientation + angular_scale * angular + action_rate_scale * action_rate) * active.float()


def zero_command_velocity_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.04,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base drift only while the shaped follow command is zero."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  stopped = torch.linalg.norm(command[:, :2], dim=1) <= command_threshold
  horizontal_speed_sq = torch.sum(
    torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return horizontal_speed_sq * stopped.float()


def velocity_overshoot_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  margin: float = 0.10,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize horizontal speed that exceeds the active follow command.

  This supplies a dense braking signal before a high-reversal transition can
  drive a hip into the ground; speeds within ``margin`` of the command are not
  penalized.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  speed = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
  target = torch.linalg.norm(command[:, :2], dim=1)
  return torch.square(torch.relu(speed - target - margin))


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def n45_hold_stability_l2(
  env: ManagerBasedRlEnv,
  min_force_n: float = 12.0,
  max_force_n: float = 18.0,
  direction_cosine: float = 0.985,
  target_direction_xy: tuple[float, float] = (0.70710678, -0.70710678),
  angular_scale: float = 0.15,
  action_rate_scale: float = 0.03,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize instability only during sustained 15 N, -45-degree pulls.

  The force and hold/release identity are privileged training signals and are
  not added to policy observations. Release scenarios remain untouched.
  """
  tether = env.event_manager.get_term_cfg("tether_pull").func
  force_w = tether.last_force_w
  horizontal = force_w[:, :2]
  magnitude = torch.linalg.norm(horizontal, dim=1)
  target = horizontal.new_tensor(target_direction_xy)
  target = target / torch.linalg.norm(target).clamp_min(1.0e-6)
  cosine = torch.sum(horizontal * target, dim=1) / magnitude.clamp_min(1.0e-6)
  hold = ~tether.v152_release_variant
  active = (
    hold
    & (magnitude >= min_force_n)
    & (magnitude <= max_force_n)
    & (cosine >= direction_cosine)
  )
  env.extras["log"]["Metrics/n45_hold_active_fraction"] = active.float().mean()
  env.extras["log"]["Metrics/tether_force_mean_n"] = magnitude.mean()
  asset: Entity = env.scene[asset_cfg.name]
  orientation = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  angular = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  action_rate = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )
  return (orientation + angular_scale * angular + action_rate_scale * action_rate) * active.float()


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def base_to_feet_clearance_l2(
  env: ManagerBasedRlEnv,
  minimum_clearance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base collapse before a non-foot ground contact occurs.

  The height is measured relative to the mean foot-site height, so the term
  remains meaningful on randomized terrain and does not require privileged
  terrain height in the deployed policy observation.
  """
  asset: Entity = env.scene[asset_cfg.name]
  base_z = asset.data.root_link_pos_w[:, 2]
  feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].mean(dim=1)
  clearance = base_z - feet_z
  shortfall = torch.clamp(minimum_clearance - clearance, min=0.0)
  env.extras["log"]["Metrics/base_to_feet_clearance_mean"] = clearance.mean()
  env.extras["log"]["Metrics/base_to_feet_clearance_min"] = clearance.min()
  return torch.square(shortfall)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward
