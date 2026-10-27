from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    self.vel_command_b[env_ids, :] *= (torch.norm(self.vel_command_b[env_ids, :], dim=1) > 0.1).unsqueeze(1)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )


class SelfStatePullCommand(UniformVelocityCommand):
  """Generate a compliant velocity command from the robot's own body velocity.

  No force value is observed.  Horizontal motion induced by an external pull is
  detected through the base IMU velocity and converted into a smooth command for
  the velocity policy to maintain.
  """

  cfg: SelfStatePullCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # A reset starts with no requested motion.  Subsequent commands come only
    # from the measured body state in _update_command().
    self.vel_command_b[env_ids, :] = 0.0
    self.is_heading_env[env_ids] = False
    self.is_standing_env[env_ids] = False

    # Calibrate the IMU and support baseline at reset.  These are state
    # references, not external-force measurements.
    self._acc_bias[env_ids] = self._imu_linear_acc()[env_ids]
    self._support_reference[env_ids] = self._support_magnitude()[env_ids]

  def __init__(self, cfg: SelfStatePullCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._acc_bias = torch.zeros(self.num_envs, 3, device=self.device)
    self._support_reference = torch.zeros(self.num_envs, device=self.device)
    self.vertical_disturbance = torch.zeros(self.num_envs, device=self.device)

  def _imu_linear_acc(self) -> torch.Tensor:
    return self._env.scene["robot/imu_lin_acc"].data

  def _support_magnitude(self) -> torch.Tensor:
    contact_force = self._env.scene["feet_ground_contact"].data.force
    assert contact_force is not None
    return torch.linalg.vector_norm(contact_force, dim=-1).flatten(start_dim=1).sum(dim=1)

  def _update_command(self) -> None:
    measured_vel_b = self.robot.data.root_link_lin_vel_b
    measured_xy = measured_vel_b[:, :2]
    acc_delta = self._imu_linear_acc() - self._acc_bias

    # At rest, slowly track the stationary IMU/support reference.  Strong
    # acceleration is excluded so a newly tightened rope is not absorbed into
    # the baseline.
    still = (
      (torch.linalg.vector_norm(measured_xy, dim=-1) < 0.04)
      & (torch.abs(measured_vel_b[:, 2]) < 0.04)
      & (torch.linalg.vector_norm(acc_delta[:, :2], dim=-1) < self.cfg.acceleration_dead_zone)
    )
    if still.any():
      self._acc_bias[still] = (
        self.cfg.reference_smoothing * self._acc_bias[still]
        + (1.0 - self.cfg.reference_smoothing) * self._imu_linear_acc()[still]
      )
      self._support_reference[still] = (
        self.cfg.reference_smoothing * self._support_reference[still]
        + (1.0 - self.cfg.reference_smoothing) * self._support_magnitude()[still]
      )

    acc_xy = torch.where(
      torch.abs(acc_delta[:, :2]) < self.cfg.acceleration_dead_zone,
      torch.zeros_like(acc_delta[:, :2]),
      acc_delta[:, :2],
    )
    target_xy = (
      measured_xy * self.cfg.velocity_gain
      + acc_xy * self.cfg.acceleration_gain
    )

    # Reject stationary IMU noise, then keep the command inside the policy's
    # low-speed training range.
    target_xy = torch.where(
      torch.abs(target_xy) < self.cfg.dead_zone,
      torch.zeros_like(target_xy),
      target_xy,
    )
    target_xy[:, 0].clamp_(*self.cfg.ranges.lin_vel_x)
    target_xy[:, 1].clamp_(*self.cfg.ranges.lin_vel_y)

    support_change = torch.abs(
      self._support_magnitude() - self._support_reference
    ) / torch.clamp(self._support_reference, min=1.0)
    vertical_signal = (
      torch.abs(acc_delta[:, 2])
      + self.cfg.vertical_velocity_gain * torch.abs(measured_vel_b[:, 2])
      + self.cfg.support_change_gain * support_change
    )
    self.vertical_disturbance = torch.clamp(
      (vertical_signal - self.cfg.vertical_disturbance_start)
      / (self.cfg.vertical_disturbance_full - self.cfg.vertical_disturbance_start),
      0.0,
      1.0,
    )
    horizontal_scale = 1.0 - self.cfg.max_horizontal_slowdown * self.vertical_disturbance

    self.vel_command_b[:, :2] = horizontal_scale.unsqueeze(1) * (
      self.cfg.smoothing * self.vel_command_b[:, :2]
      + (1.0 - self.cfg.smoothing) * target_xy
    )
    self.vel_command_b[:, 2] = 0.0


@dataclass(kw_only=True)
class SelfStatePullCommandCfg(UniformVelocityCommandCfg):
  """Configuration for state-inferred, sensor-free pull following."""

  dead_zone: float = 0.08
  velocity_gain: float = 1.0
  acceleration_gain: float = 0.12
  acceleration_dead_zone: float = 0.35
  smoothing: float = 0.85
  reference_smoothing: float = 0.995
  vertical_velocity_gain: float = 0.7
  support_change_gain: float = 3.0
  vertical_disturbance_start: float = 0.8
  vertical_disturbance_full: float = 4.0
  max_horizontal_slowdown: float = 0.6

  def build(self, env: ManagerBasedRlEnv) -> SelfStatePullCommand:
    return SelfStatePullCommand(self, env)
