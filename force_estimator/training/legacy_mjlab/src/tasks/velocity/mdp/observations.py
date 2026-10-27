from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def estimated_force_placeholder(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reserve three actor inputs for the learned force estimate."""
  return torch.zeros(env.num_envs, 3, device=env.device)


def external_force_label(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  scale: float = 1.0 / 80.0,
) -> torch.Tensor:
  """Privileged base-frame wrench label used only to supervise the estimator."""
  asset: Entity = env.scene[asset_cfg.name]
  force_w = asset.data.body_external_wrench[..., :3].sum(dim=1)
  force_b = quat_apply_inverse(asset.data.root_link_quat_w, force_w)
  return force_b * scale


def external_force_label_world(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  scale: float = 1.0 / 80.0,
) -> torch.Tensor:
  """Privileged world-frame force label; labels/metrics only."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_external_wrench[..., :3].sum(dim=1) * scale


def external_force_label_body(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  scale: float = 1.0 / 80.0,
) -> torch.Tensor:
  """Explicit body-frame alias retained for frozen-policy supervision."""
  return external_force_label(env, asset_cfg=asset_cfg, scale=scale)


def base_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Return base height for vertical disturbance inference."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def _deployment_scalar_foot_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scale_range: tuple[float, float] = (0.75, 1.25),
  bias_range: tuple[float, float] = (-2.0, 2.0),
) -> torch.Tensor:
  """Emulate the four scalar normal-force channels available on a stock Go2."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  raw_normal = torch.abs(sensor_data.force[..., 2])
  step = env.episode_length_buf
  count = env.num_envs
  device = raw_normal.device
  if not hasattr(env, "_v20_foot_force_scale"):
    env._v20_foot_force_scale = torch.ones(count, 4, device=device)
    env._v20_foot_force_bias = torch.zeros(count, 4, device=device)
    env._v20_foot_force_last_step = torch.full_like(step, -1)
  reset = (step == 0) & (step != env._v20_foot_force_last_step)
  if reset.any():
    n = int(reset.sum())
    env._v20_foot_force_scale[reset] = torch.empty(n, 4, device=device).uniform_(
      *scale_range
    )
    env._v20_foot_force_bias[reset] = torch.empty(n, 4, device=device).uniform_(
      *bias_range
    )
  env._v20_foot_force_last_step.copy_(step)
  return torch.clamp(
    raw_normal * env._v20_foot_force_scale + env._v20_foot_force_bias,
    min=0.0,
  )


def deployment_foot_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_n: float = 10.0,
  scale_range: tuple[float, float] = (0.75, 1.25),
  bias_range: tuple[float, float] = (-2.0, 2.0),
) -> torch.Tensor:
  """Contact flags derived from the same scalar channels used in deployment."""
  scalar = _deployment_scalar_foot_forces(
    env, sensor_name, scale_range=scale_range, bias_range=bias_range
  )
  return (scalar > threshold_n).float()


def deployment_scalar_foot_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scale_range: tuple[float, float] = (0.75, 1.25),
  bias_range: tuple[float, float] = (-2.0, 2.0),
) -> torch.Tensor:
  """Encode four scalar loads as zero tangential and negative-Z features."""
  scalar = _deployment_scalar_foot_forces(
    env, sensor_name, scale_range=scale_range, bias_range=bias_range
  )
  loads = torch.zeros(
    env.num_envs, 4, 3, device=scalar.device, dtype=scalar.dtype
  )
  loads[:, :, 2] = -torch.log1p(scalar)
  return loads.flatten(start_dim=1)


def deployment_base_lin_vel(
  env: ManagerBasedRlEnv,
  accel_sensor_name: str,
  contact_sensor_name: str,
  contact_threshold_n: float = 10.0,
  accel_bias_range: tuple[float, float] = (-0.04, 0.04),
  velocity_limit_mps: float = 3.0,
) -> torch.Tensor:
  """Contact-aided IMU velocity estimate using only deployable measurements."""
  acceleration = env.scene[accel_sensor_name].data
  asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  gravity_w = torch.tensor(
    (0.0, 0.0, -1.0), device=acceleration.device, dtype=acceleration.dtype
  ).expand(env.num_envs, -1)
  projected_gravity = quat_apply_inverse(asset.data.root_link_quat_w, gravity_w)
  step = env.episode_length_buf
  if not hasattr(env, "_v20_base_lin_vel"):
    env._v20_base_lin_vel = torch.zeros_like(acceleration)
    env._v20_accel_bias = torch.zeros_like(acceleration)
    env._v20_velocity_last_step = torch.full_like(step, -1)
  changed = step != env._v20_velocity_last_step
  reset = changed & (step == 0)
  if reset.any():
    env._v20_base_lin_vel[reset] = 0.0
    env._v20_accel_bias[reset] = torch.empty(
      int(reset.sum()), 3, device=acceleration.device
    ).uniform_(*accel_bias_range)
  update = changed & ~reset
  compensated = (
    acceleration + env._v20_accel_bias + 9.81 * projected_gravity
  )
  if update.any():
    env._v20_base_lin_vel[update] += compensated[update] * env.step_dt
    scalar_force = _deployment_scalar_foot_forces(
      env, contact_sensor_name
    )
    stable = (
      (scalar_force > contact_threshold_n).sum(dim=1) >= 3
    ) & (torch.linalg.vector_norm(compensated, dim=1) < 0.3)
    damp = update & stable
    env._v20_base_lin_vel[damp] *= 0.90
    env._v20_base_lin_vel.clamp_(-velocity_limit_mps, velocity_limit_mps)
  env._v20_velocity_last_step.copy_(step)
  return env._v20_base_lin_vel.clone()


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
