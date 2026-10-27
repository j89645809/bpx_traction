"""Unitree Go2 velocity environment configurations."""

from dataclasses import replace
import os
from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.mdp.tether_events import apply_tether_pull
import src.tasks.velocity.mdp as self_pull_mdp

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    #cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.6, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.4, 0.4)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_go2_self_pull_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """GuideDog-style force-estimation and force-responsive velocity task."""
  cfg = unitree_go2_flat_env_cfg(play=play)

  cfg.commands["twist"] = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.05,
    heading_command=False,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
    ),
  )

  # A sustained tether wrench at a random link and local point.  Its force is
  # deliberately hidden from the observation.
  cfg.events.pop("push_robot", None)
  cfg.events["tether_pull"] = EventTermCfg(
    func=apply_tether_pull,
    mode="step",
    params={
      "vertical_force_range": (-35.0, 35.0),
      # Curriculum is keyed to common simulation steps (24 steps/iteration).
      # The runner enables autonomous pulls at iteration 1500; after that,
      # widen the force envelope gradually instead of exposing 20-80 N at once.
      "horizontal_force_range": (12.0, 80.0),
      "horizontal_force_curriculum_steps": (
        (72000, 20.0),    # 1500-2999: 12-20 N teacher
        (120000, 30.0),   # 3000-4999: 12-30 N fade
        (168000, 40.0),   # 5000-6999: 12-40 N estimator-led
        (240000, 60.0),   # 7000-9999: 12-60 N directional robustness
        (10**12, 80.0),  # 10000+: 12-80 N only after lower-force training
      ),
      "vertical_force_curriculum_steps": (
        (120000, (0.0, 0.0)),
        (168000, (-10.0, 10.0)),
        (240000, (-20.0, 20.0)),
        (10**12, (-35.0, 35.0)),
      ),
      "duration_s": (0.8, 2.0),
      "cooldown_s": (0.8, 2.0),
      # Explicitly retain the frozen Gate 4.1 semantics. World-force
      # experiments override this with ``world_fixed``.
      "force_frame": "legacy_body",
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("base_link", "Head_lower"),
      ),
      # Random application point in the selected link's local frame.
      "body_point_offset_ranges": (
        (-0.08, 0.08),
        (-0.06, 0.06),
        (-0.06, 0.06),
      ),
    },
  )

  return cfg


def unitree_go2_self_pull_fx_only_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Minimal forward tether-following task: +20 N Fx for 1 s, then zero for 1 s."""
  cfg = unitree_go2_self_pull_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params.update({
    "fixed_force_b": (20.0, 0.0, 0.0),
    "fixed_cycle_s": 1.0,
    "horizontal_force_range": (20.0, 20.0),
    "vertical_force_range": (0.0, 0.0),
    "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    "body_point_offset_ranges": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
  })
  return cfg



def unitree_go2_self_pull_fx_bilateral_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Fx-only bilateral pull-following with a 20/40/60/80 N curriculum."""
  cfg = unitree_go2_self_pull_fx_only_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params.pop("fixed_force_b", None)
  tether.params["bilateral_fx_curriculum"] = True
  tether.params["force_free_warmup_steps"] = 36000
  # Stage 1 trains active forward/backward walking with no physical pull.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-0.6, 0.8)
  if play:
    # Keep Viser sliders valid; Fx-only training itself remains one-dimensional.
    twist.ranges.lin_vel_y = (-0.4, 0.4)
    twist.ranges.ang_vel_z = (-0.5, 0.5)
  else:
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 0.15
  # The runner writes the exact actor command so PPO reward and actor input agree.
  tether.params["write_follow_command"] = False
  return cfg

def unitree_go2_self_pull_fx_teacher20_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 2: active Fx locomotion under a centered 12-20 N teacher pull."""
  cfg = unitree_go2_self_pull_fx_bilateral_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params["force_free_warmup_steps"] = 0
  tether.params["bilateral_force_magnitude_range"] = (12.0, 20.0)
  tether.params["retention_env_fraction"] = 0.5
  tether.params["retention_command_range"] = (0.2, 0.5)
  # Low-force commands start at 0.08 m/s, so they must activate the gait
  # phase and contact rewards instead of being classified as standing.
  cfg.rewards["foot_gait"].weight = 1.0
  cfg.rewards["foot_gait"].params["command_threshold"] = 0.05
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.05
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.05
  cfg.rewards["pose"].params["walking_threshold"] = 0.05
  # Keep this gate one-dimensional and prevent the base command curriculum
  # from reintroducing vy/wz during training.
  cfg.curriculum.pop("command_vel", None)
  return cfg


def unitree_go2_self_pull_fx_teacher20_polish_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 2.1: reduce yaw drift and action jitter without changing the task."""
  cfg = unitree_go2_self_pull_fx_teacher20_env_cfg(play=play)
  cfg.rewards["track_angular_velocity"].weight = 1.5
  cfg.rewards["track_angular_velocity"].params["std"] = 0.3
  cfg.rewards["body_ang_vel"].weight = -0.1
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["joint_acc_l2"].weight = -5.0e-7
  cfg.rewards["foot_gait"].weight = 0.75
  return cfg


def _unitree_go2_self_pull_expansion_env_cfg(
  play: bool,
  *,
  direction_mode: str,
  vertical_force_range: tuple[float, float],
  random_application_point: bool,
) -> ManagerBasedRlEnvCfg:
  """Shared post-Fx curriculum with deployment-equivalent observations."""
  cfg = unitree_go2_self_pull_fx_teacher20_polish_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params.pop("fixed_cycle_s", None)
  tether.params.pop("bilateral_fx_curriculum", None)
  tether.params.pop("bilateral_force_magnitude_range", None)
  tether.params.pop("force_free_warmup_steps", None)
  tether.params.pop("retention_env_fraction", None)
  tether.params.pop("retention_command_range", None)
  tether.params.update({
    "horizontal_force_range": (12.0, 20.0),
    "vertical_force_range": vertical_force_range,
    "duration_s": (1.0, 2.0),
    "cooldown_s": (1.0, 2.0),
    "force_direction_mode": direction_mode,
    "fx_retention_fraction": 0.0,
    "write_follow_command": False,
  })
  if random_application_point:
    tether.params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=("base_link", "Head_lower")
    )
    tether.params["body_point_offset_ranges"] = (
      (-0.08, 0.08),
      (-0.06, 0.06),
      (-0.06, 0.06),
    )
  else:
    tether.params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names=("base_link",)
    )
    tether.params["body_point_offset_ranges"] = (
      (0.0, 0.0),
      (0.0, 0.0),
      (0.0, 0.0),
    )
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-0.6, 0.8)
  twist.ranges.lin_vel_y = (-0.4, 0.4)
  twist.ranges.ang_vel_z = (-0.5, 0.5) if play else (0.0, 0.0)
  cfg.curriculum.pop("command_vel", None)
  return cfg


def unitree_go2_self_pull_gate2_fy_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 2: centered bilateral Fy, retaining 25% centered Fx samples."""
  cfg = _unitree_go2_self_pull_expansion_env_cfg(
    play,
    direction_mode="fy_bilateral",
    vertical_force_range=(0.0, 0.0),
    random_application_point=False,
  )
  cfg.events["tether_pull"].params["fx_retention_fraction"] = 0.25
  return cfg


def unitree_go2_self_pull_gate21_fy_active_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 2.1: force active lateral stepping with a no-pull vy cohort."""
  cfg = unitree_go2_self_pull_gate2_fy_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params["direct_lateral_retention_fraction"] = 0.5
  tether.params["direct_lateral_command_range"] = (0.2, 0.4)
  cfg.rewards["track_linear_velocity"].weight = 1.5
  cfg.rewards["foot_gait"].weight = 1.0
  return cfg


def unitree_go2_self_pull_gate3_planar_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 3: centered horizontal pulls sampled over the full circle."""
  return _unitree_go2_self_pull_expansion_env_cfg(
    play,
    direction_mode="random_horizontal",
    vertical_force_range=(0.0, 0.0),
    random_application_point=False,
  )


def unitree_go2_self_pull_gate31_planar_balanced_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 3.1: balanced axial/diagonal pulls with lateral-gait retention."""
  cfg = _unitree_go2_self_pull_expansion_env_cfg(
    play,
    direction_mode="planar_balanced",
    vertical_force_range=(0.0, 0.0),
    random_application_point=False,
  )
  tether = cfg.events["tether_pull"]
  tether.params["direct_lateral_retention_fraction"] = 0.3
  tether.params["direct_lateral_command_range"] = (0.2, 0.4)
  cfg.rewards["track_linear_velocity"].weight = 1.5
  cfg.rewards["foot_gait"].weight = 1.0
  return cfg


def unitree_go2_self_pull_gate4_3d_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 4: centered horizontal pulls with moderate vertical load."""
  cfg = _unitree_go2_self_pull_expansion_env_cfg(
    play,
    direction_mode="random_horizontal",
    vertical_force_range=(-20.0, 20.0),
    random_application_point=False,
  )
  cfg.events["tether_pull"].params["horizontal_force_range"] = (12.0, 40.0)
  return cfg


def unitree_go2_self_pull_gate41_slowdown_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 4.1: make vertical load produce measurable physical slowdown."""
  cfg = unitree_go2_self_pull_gate4_3d_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  # Half of the training pulls use exactly +/-20 N so the policy repeatedly
  # practices braking at the validation boundary.
  tether.params["vertical_endpoint_fraction"] = 0.5
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.25
  return cfg


def unitree_go2_self_pull_gate41_world_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 4.1 frozen-actor task with canonical world-fixed tether forces."""
  cfg = unitree_go2_self_pull_gate41_slowdown_env_cfg(play=play)
  cfg.events["tether_pull"].params["force_frame"] = "world_fixed"
  return cfg


def unitree_go2_self_pull_direction_finetune_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Actor fine-tuning task that exactly spans the certification force grid."""
  cfg = unitree_go2_self_pull_gate41_world_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params.update({
    "force_direction_mode": "certification_matrix_hard40",
    # Each active interval teaches sustained following; every following
    # cooldown explicitly teaches release and return-to-stand behavior.
    "duration_s": (2.0, 4.0),
    "cooldown_s": (1.0, 2.0),
    "horizontal_force_range": (5.0, 40.0),
    "vertical_force_range": (-20.0, 20.0),
    "vertical_endpoint_fraction": 0.0,
    "direct_lateral_retention_fraction": 0.0,
  })
  return cfg


def unitree_go2_self_pull_continuous_jump_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v19: continuous world directions with direct force-vector jumps."""
  cfg = unitree_go2_self_pull_gate41_world_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params.update({
    "force_direction_mode": "continuous_world_jump",
    "force_frame": "world_fixed",
    # Zero cooldown makes the expired vector and newly sampled vector replace
    # one another in the same control step: no artificial zero-force gap.
    "duration_s": (0.5, 2.0),
    "cooldown_s": (0.0, 0.0),
    "horizontal_force_range": (5.0, 40.0),
    "vertical_force_range": (-20.0, 20.0),
    "vertical_endpoint_fraction": 0.20,
    "direct_lateral_retention_fraction": 0.0,
    "retention_env_fraction": 0.0,
  })
  return cfg


def unitree_go2_self_pull_deployment_observable_v20_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """v20: v19 force coverage with only Go2-deployable actor observations."""
  cfg = unitree_go2_self_pull_continuous_jump_env_cfg(play=play)
  actor = cfg.observations["actor"].terms
  sensor_scale = (1.0, 1.0) if play else (0.75, 1.25)
  sensor_bias = (0.0, 0.0) if play else (-2.0, 2.0)
  accel_bias = (0.0, 0.0) if play else (-0.04, 0.04)
  actor["base_lin_vel"] = replace(
    actor["base_lin_vel"],
    func=self_pull_mdp.deployment_base_lin_vel,
    params={
      "accel_sensor_name": "robot/imu_lin_acc",
      "contact_sensor_name": "feet_ground_contact",
      "contact_threshold_n": 10.0,
      "accel_bias_range": accel_bias,
    },
  )
  actor["foot_contact"] = replace(
    actor["foot_contact"],
    func=self_pull_mdp.deployment_foot_contact,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_n": 10.0,
      "scale_range": sensor_scale,
      "bias_range": sensor_bias,
    },
  )
  actor["foot_contact_forces"] = replace(
    actor["foot_contact_forces"],
    func=self_pull_mdp.deployment_scalar_foot_contact_forces,
    params={
      "sensor_name": "feet_ground_contact",
      "scale_range": sensor_scale,
      "bias_range": sensor_bias,
    },
  )
  return cfg


def unitree_go2_self_pull_joint_robust_matrix_v20_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Deployable observations plus the complete hold/release force matrix."""
  cfg = unitree_go2_self_pull_deployment_observable_v20_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params.update({
    "force_direction_mode": "certification_matrix_hard40",
    "force_frame": "world_fixed",
    "duration_s": (2.0, 4.0),
    "cooldown_s": (1.0, 2.0),
    "horizontal_force_range": (5.0, 40.0),
    "vertical_force_range": (-20.0, 20.0),
    "vertical_endpoint_fraction": 0.0,
    "direct_lateral_retention_fraction": 0.0,
    "retention_env_fraction": 0.0,
    "static_ramp_release": False,
    "jump_training": False,
  })
  return cfg


def unitree_go2_self_pull_joint_robust_ramp_v20_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Zero-force stance followed by slow ramp, hold and release."""
  cfg = unitree_go2_self_pull_deployment_observable_v20_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params.update({
    "force_frame": "world_fixed",
    "write_follow_command": False,
    "static_ramp_release": True,
    "static_cycle_s": 6.0,
    "static_target_force_range": (5.0, 40.0),
    "jump_training": False,
  })
  return cfg


def unitree_go2_self_pull_joint_robust_jump_v20_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Exact 90/180-degree direction changes with magnitude/Z jumps and release."""
  cfg = unitree_go2_self_pull_deployment_observable_v20_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params.update({
    "force_frame": "world_fixed",
    "write_follow_command": False,
    "static_ramp_release": False,
    "jump_training": True,
    "jump_cycle_s": 6.0,
    "jump_pre_s": 2.0,
    "jump_post_s": 2.0,
    "jump_force_range": (5.0, 40.0),
    "jump_targeted_turns": True,
  })
  return cfg


def unitree_go2_self_pull_traction_actor_v23_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Traction actor curriculum with all ten hard behavior families."""
  cfg = unitree_go2_self_pull_deployment_observable_v20_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params.update({
    "force_frame": "world_fixed",
    "write_follow_command": False,
    "static_ramp_release": False,
    "jump_training": False,
    "traction_actor_curriculum": True,
    "traction_cycle_s": 8.0,
  })
  cfg.rewards["body_orientation_l2"].weight = -1.5
  cfg.rewards["body_ang_vel"].weight = -0.10
  cfg.rewards["stand_still"].weight = -2.0
  cfg.rewards["is_terminated"].weight = -300.0
  return cfg


def unitree_go2_self_pull_traction_actor_v24_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Hard-family continuation from v23; the frozen estimator is unchanged."""
  cfg = unitree_go2_self_pull_traction_actor_v23_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_hard_focus"] = True
  cfg.events["tether_pull"].params["traction_cycle_s"] = 8.0
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["body_ang_vel"].weight = -0.15
  cfg.rewards["stand_still"].weight = -2.5
  cfg.rewards["is_terminated"].weight = -350.0
  return cfg


def unitree_go2_self_pull_traction_actor_v25_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Extreme stability continuation for v24 certification failures."""
  cfg = unitree_go2_self_pull_traction_actor_v24_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_extreme_focus"] = True
  cfg.rewards["body_orientation_l2"].weight = -2.5
  cfg.rewards["body_ang_vel"].weight = -0.20
  cfg.rewards["stand_still"].weight = -3.0
  cfg.rewards["is_terminated"].weight = -450.0
  cfg.rewards["soft_landing"].weight = -0.01
  return cfg


def unitree_go2_self_pull_traction_actor_v26_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Jump-checkpoint continuation focused on recovery and release behavior."""
  cfg = unitree_go2_self_pull_traction_actor_v23_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_recovery_focus"] = True
  cfg.events["tether_pull"].params["traction_cycle_s"] = 8.0
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["body_ang_vel"].weight = -0.15
  cfg.rewards["stand_still"].weight = -2.5
  cfg.rewards["is_terminated"].weight = -350.0
  cfg.rewards["soft_landing"].weight = -0.006
  return cfg


def unitree_go2_self_pull_traction_actor_v31_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Balanced contact-stability continuation based on v30 attribution.

  All ten strict-certification families remain in the sampling distribution;
  the four families that caused illegal contacts receive one extra copy.
  The frozen 82-D estimator and the shared command shaping are unchanged.
  """
  cfg = unitree_go2_self_pull_traction_actor_v23_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_contact_focus"] = True
  cfg.rewards["body_orientation_l2"].weight = -2.5
  cfg.rewards["body_ang_vel"].weight = -0.20
  cfg.rewards["angular_momentum"].weight = -0.05
  cfg.rewards["stand_still"].weight = -3.0
  cfg.rewards["is_terminated"].weight = -500.0
  cfg.rewards["action_rate_l2"].weight = -0.12
  cfg.rewards["soft_landing"].weight = -0.01
  return cfg


def unitree_go2_self_pull_traction_actor_v32_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Contact-focused continuation with a dense pre-collapse signal."""
  cfg = unitree_go2_self_pull_traction_actor_v31_env_cfg(play=play)
  cfg.rewards["base_to_feet_clearance"] = RewardTermCfg(
    func=self_pull_mdp.base_to_feet_clearance_l2,
    weight=-150.0,
    params={
      "minimum_clearance": 0.165,
      "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL")),
    },
  )
  cfg.rewards["pose"].weight = 2.0
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.12,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.28,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.38,
  }
  return cfg


def unitree_go2_self_pull_traction_actor_v33_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Earlier dense collapse warning calibrated from v32 telemetry."""
  cfg = unitree_go2_self_pull_traction_actor_v32_env_cfg(play=play)
  cfg.rewards["base_to_feet_clearance"].params["minimum_clearance"] = 0.22
  cfg.rewards["base_to_feet_clearance"].weight = -100.0
  return cfg


def unitree_go2_self_pull_traction_actor_v34_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Dense non-foot contact avoidance, calibrated from v30-v33 failures."""
  cfg = unitree_go2_self_pull_traction_actor_v33_env_cfg(play=play)
  cfg.rewards["nonfoot_contact_history"] = RewardTermCfg(
    func=self_pull_mdp.self_collision_cost,
    weight=-8.0,
    params={"sensor_name": "nonfoot_ground_touch", "force_threshold": 5.0},
  )
  return cfg


def unitree_go2_self_pull_traction_actor_v36_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Switch-stability continuation with stronger joint motion regularization."""
  cfg = unitree_go2_self_pull_traction_actor_v31_env_cfg(play=play)
  cfg.rewards["action_rate_l2"].weight = -0.45
  cfg.rewards["joint_acc_l2"].weight = -2.0e-6
  cfg.rewards["body_orientation_l2"].weight = -3.0
  cfg.rewards["body_ang_vel"].weight = -0.30
  cfg.rewards["angular_momentum"].weight = -0.08
  cfg.rewards["is_terminated"].weight = -550.0
  return cfg


def unitree_go2_self_pull_traction_actor_v37_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Joint acute-turn/contact stability continuation from the v36 actor."""
  cfg = unitree_go2_self_pull_traction_actor_v34_env_cfg(play=play)
  cfg.rewards["action_rate_l2"].weight = -0.55
  cfg.rewards["joint_acc_l2"].weight = -3.0e-6
  cfg.rewards["body_orientation_l2"].weight = -4.0
  cfg.rewards["body_ang_vel"].weight = -0.40
  cfg.rewards["angular_momentum"].weight = -0.10
  cfg.rewards["is_terminated"].weight = -650.0
  return cfg


def unitree_go2_self_pull_traction_actor_v40_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Low-force response, zero-command stop and non-foot-contact continuation.

  This keeps the frozen 82-D estimator and shared force-command contract
  untouched.  It addresses v39's measured failures with dense objectives:
  low 0.06--0.16 commands must move, zero commands must not drift, and rear
  thigh/calf or head ground contacts receive a stronger penalty.
  """
  cfg = unitree_go2_self_pull_traction_actor_v37_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v40_focus"] = True
  cfg.rewards["track_low_linear_velocity"] = RewardTermCfg(
    func=self_pull_mdp.track_low_linear_velocity,
    weight=2.5,
    params={
      "command_name": "twist",
      "std": 0.10,
      "command_min": 0.06,
      "command_max": 0.16,
    },
  )
  cfg.rewards["zero_command_velocity_l2"] = RewardTermCfg(
    func=self_pull_mdp.zero_command_velocity_l2,
    weight=-6.0,
    params={"command_name": "twist", "command_threshold": 0.04},
  )
  cfg.rewards["nonfoot_contact_history"].weight = -30.0
  cfg.rewards["base_to_feet_clearance"].weight = -180.0
  cfg.rewards["base_to_feet_clearance"].params["minimum_clearance"] = 0.235
  cfg.rewards["pose"].weight = 2.5
  cfg.rewards["body_orientation_l2"].weight = -5.0
  cfg.rewards["body_ang_vel"].weight = -0.50
  cfg.rewards["angular_momentum"].weight = -0.12
  cfg.rewards["is_terminated"].weight = -800.0
  # v37's very strong action-rate term suppressed the 0.1-command gait.
  # Dense contact/clearance rewards now provide the braking signal directly.
  cfg.rewards["action_rate_l2"].weight = -0.40
  return cfg


def unitree_go2_self_pull_traction_actor_v41_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Targeted final actor continuation from v40's best checkpoint."""
  cfg = unitree_go2_self_pull_traction_actor_v40_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v41_focus"] = True
  cfg.rewards["track_low_linear_velocity"].weight = 5.0
  cfg.rewards["zero_command_velocity_l2"].weight = -8.0
  cfg.rewards["nonfoot_contact_history"].weight = -45.0
  cfg.rewards["base_to_feet_clearance"].weight = -220.0
  cfg.rewards["base_to_feet_clearance"].params["minimum_clearance"] = 0.24
  cfg.rewards["body_orientation_l2"].weight = -6.0
  cfg.rewards["body_ang_vel"].weight = -0.60
  cfg.rewards["angular_momentum"].weight = -0.15
  cfg.rewards["is_terminated"].weight = -900.0
  return cfg


def unitree_go2_self_pull_traction_actor_v42_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stability continuation from v41/v47, focused on robust stop and turns."""
  cfg = unitree_go2_self_pull_traction_actor_v41_env_cfg(play=play)
  cfg.rewards["zero_command_velocity_l2"].weight = -24.0
  cfg.rewards["track_low_linear_velocity"].weight = 7.0
  cfg.rewards["nonfoot_contact_history"].weight = -55.0
  cfg.rewards["base_to_feet_clearance"].weight = -240.0
  cfg.rewards["base_to_feet_clearance"].params["minimum_clearance"] = 0.245
  cfg.rewards["pose"].weight = 3.0
  cfg.rewards["body_orientation_l2"].weight = -7.0
  cfg.rewards["body_ang_vel"].weight = -0.70
  cfg.rewards["angular_momentum"].weight = -0.18
  cfg.rewards["is_terminated"].weight = -1000.0
  cfg.rewards["action_rate_l2"].weight = -0.55
  return cfg


def unitree_go2_self_pull_traction_actor_v43_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Balanced stability continuation after v48 reward over-regularization."""
  cfg = unitree_go2_self_pull_traction_actor_v41_env_cfg(play=play)
  cfg.rewards["zero_command_velocity_l2"].weight = -10.0
  cfg.rewards["track_low_linear_velocity"].weight = 5.5
  cfg.rewards["nonfoot_contact_history"].weight = -47.0
  cfg.rewards["base_to_feet_clearance"].weight = -225.0
  cfg.rewards["base_to_feet_clearance"].params["minimum_clearance"] = 0.24
  cfg.rewards["pose"].weight = 2.7
  cfg.rewards["body_orientation_l2"].weight = -6.2
  cfg.rewards["body_ang_vel"].weight = -0.62
  cfg.rewards["angular_momentum"].weight = -0.16
  cfg.rewards["is_terminated"].weight = -920.0
  cfg.rewards["action_rate_l2"].weight = -0.45
  return cfg


def unitree_go2_self_pull_traction_actor_v44_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Targeted template-balance continuation after v49 failures."""
  cfg = unitree_go2_self_pull_traction_actor_v43_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v44_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v45_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Targeted continuation for v51 illegal-contact failures."""
  cfg = unitree_go2_self_pull_traction_actor_v44_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v45_focus"] = True
  cfg.rewards["nonfoot_contact_history"].weight = -52.0
  cfg.rewards["base_to_feet_clearance"].weight = -235.0
  cfg.rewards["base_to_feet_clearance"].params["minimum_clearance"] = 0.242
  cfg.rewards["zero_command_velocity_l2"].weight = -12.0
  cfg.rewards["track_low_linear_velocity"].weight = 5.8
  cfg.rewards["action_rate_l2"].weight = -0.47
  return cfg


def unitree_go2_self_pull_traction_actor_v46_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_self_pull_traction_actor_v45_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v46_focus"] = True
  cfg.rewards["nonfoot_contact_history"].weight = -56.0
  cfg.rewards["base_to_feet_clearance"].weight = -245.0
  cfg.rewards["base_to_feet_clearance"].params["minimum_clearance"] = 0.245
  cfg.rewards["zero_command_velocity_l2"].weight = -14.0
  cfg.rewards["track_low_linear_velocity"].weight = 6.2
  cfg.rewards["body_orientation_l2"].weight = -6.8
  cfg.rewards["body_ang_vel"].weight = -0.70
  cfg.rewards["is_terminated"].weight = -980.0
  return cfg


def unitree_go2_self_pull_joint_robust_recovery_v20_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Recovery fine-tune for abrupt turns, vertical jumps and release stops.

  This is deliberately a locomotion-only continuation from the frozen
  estimator contract.  The stronger stability terms teach the actor to brake
  and return to the official standing pose after a force disappears, while
  retaining the targeted 90/180-degree jump curriculum.
  """
  cfg = unitree_go2_self_pull_joint_robust_jump_v20_env_cfg(play=play)
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["body_ang_vel"].weight = -0.15
  cfg.rewards["angular_momentum"].weight = -0.05
  cfg.rewards["is_terminated"].weight = -300.0
  cfg.rewards["action_rate_l2"].weight = -0.08
  cfg.rewards["stand_still"].weight = -3.0
  cfg.rewards["track_linear_velocity"].weight = 1.5
  cfg.rewards["foot_gait"].weight = 0.6
  tether = cfg.events["tether_pull"]
  tether.params.update({
    "jump_cycle_s": 8.0,
    "jump_pre_s": 3.0,
    "jump_post_s": 3.0,
    "jump_force_range": (5.0, 40.0),
    "jump_targeted_turns": True,
  })
  return cfg


def unitree_go2_self_pull_gate5_wrench_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Gate 5: full 3D pull with randomized torso/head application wrench."""
  cfg = _unitree_go2_self_pull_expansion_env_cfg(
    play,
    direction_mode="random_horizontal",
    vertical_force_range=(-35.0, 35.0),
    random_application_point=True,
  )
  tether = cfg.events["tether_pull"]
  tether.params["horizontal_force_range"] = (12.0, 80.0)
  # 24 environment steps per PPO iteration: 40 N -> 60 N -> 80 N.
  tether.params["horizontal_force_curriculum_steps"] = (
    (12000, 40.0),
    (24000, 60.0),
    (48000, 80.0),
  )
  return cfg


def unitree_go2_self_pull_traction_actor_v47_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v54 continuation for release braking and abrupt reversal stability."""
  cfg = unitree_go2_self_pull_traction_actor_v46_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v47_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -26.0
  cfg.rewards["track_low_linear_velocity"].weight = 7.0
  cfg.rewards["nonfoot_contact_history"].weight = -62.0
  cfg.rewards["base_to_feet_clearance"].weight = -250.0
  cfg.rewards["body_orientation_l2"].weight = -7.4
  cfg.rewards["body_ang_vel"].weight = -0.78
  cfg.rewards["angular_momentum"].weight = -0.20
  cfg.rewards["is_terminated"].weight = -1050.0
  return cfg


def unitree_go2_self_pull_traction_actor_v48_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Balanced v55 continuation from v53 without v54 over-regularization."""
  cfg = unitree_go2_self_pull_traction_actor_v46_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v47_focus"] = True
  cfg.events["tether_pull"].params["traction_actor_v64_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -18.0
  cfg.rewards["track_low_linear_velocity"].weight = 6.4
  cfg.rewards["nonfoot_contact_history"].weight = -57.0
  cfg.rewards["base_to_feet_clearance"].weight = -245.0
  cfg.rewards["body_orientation_l2"].weight = -6.9
  cfg.rewards["body_ang_vel"].weight = -0.70
  cfg.rewards["angular_momentum"].weight = -0.17
  cfg.rewards["is_terminated"].weight = -990.0
  return cfg


def unitree_go2_self_pull_traction_actor_v65_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v65 actor continuation with an explicit hard-transition cohort.

  Unlike the earlier v25 task, this configuration actually enables the
  v65 template sampler.  It keeps the frozen estimator and shared command
  shaping unchanged while strengthening release braking and contact/attitude
  protection based on v64's failed gates.
  """
  cfg = unitree_go2_self_pull_traction_actor_v48_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v65_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -22.0
  cfg.rewards["track_low_linear_velocity"].weight = 7.2
  cfg.rewards["nonfoot_contact_history"].weight = -68.0
  cfg.rewards["base_to_feet_clearance"].weight = -270.0
  cfg.rewards["body_orientation_l2"].weight = -7.5
  cfg.rewards["body_ang_vel"].weight = -0.78
  cfg.rewards["angular_momentum"].weight = -0.20
  cfg.rewards["is_terminated"].weight = -1080.0
  return cfg


def unitree_go2_self_pull_traction_actor_v66_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Balanced retry after v65 over-regularization.

  Keep the v65 hard-transition cohort, but restore v48's balanced reward
  scale so the actor does not suppress the gait needed for sharp turns.
  """
  cfg = unitree_go2_self_pull_traction_actor_v48_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v65_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -18.0
  cfg.rewards["track_low_linear_velocity"].weight = 6.4
  cfg.rewards["nonfoot_contact_history"].weight = -57.0
  cfg.rewards["base_to_feet_clearance"].weight = -245.0
  cfg.rewards["body_orientation_l2"].weight = -6.9
  cfg.rewards["body_ang_vel"].weight = -0.70
  cfg.rewards["angular_momentum"].weight = -0.17
  cfg.rewards["is_terminated"].weight = -990.0
  return cfg


def unitree_go2_self_pull_traction_actor_v67_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Low-learning-rate targeted retry after v66 checkpoint sweep."""
  cfg = unitree_go2_self_pull_traction_actor_v48_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v67_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v68_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Very short polish for recurrent diagonal/high-reverse failures."""
  cfg = unitree_go2_self_pull_traction_actor_v48_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v68_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v69_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Contact-targeted continuation from the v67 checkpoint."""
  cfg = unitree_go2_self_pull_traction_actor_v48_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v69_focus"] = True
  cfg.rewards["nonfoot_contact_history"].weight = -64.0
  cfg.rewards["base_to_feet_clearance"].weight = -255.0
  cfg.rewards["body_orientation_l2"].weight = -7.1
  cfg.rewards["body_ang_vel"].weight = -0.72
  cfg.rewards["is_terminated"].weight = -1020.0
  return cfg


def unitree_go2_self_pull_traction_actor_v70_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Collision-focused continuation after v69 diagnostics."""
  cfg = unitree_go2_self_pull_traction_actor_v48_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v70_focus"] = True
  cfg.rewards["nonfoot_contact_history"].weight = -82.0
  cfg.rewards["base_to_feet_clearance"].weight = -265.0
  cfg.rewards["body_orientation_l2"].weight = -7.3
  cfg.rewards["body_ang_vel"].weight = -0.75
  cfg.rewards["angular_momentum"].weight = -0.18
  cfg.rewards["is_terminated"].weight = -1040.0
  return cfg


def unitree_go2_self_pull_traction_actor_v72_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Contact-learning continuation: keep contact cost, allow recovery during training."""
  cfg = unitree_go2_self_pull_traction_actor_v69_env_cfg(play=play)
  # Immediate illegal-contact resets provide almost no gradient for the
  # preceding action. Keep the measured contact penalty active and let the
  # actor learn recovery; strict certification still retains the reset.
  cfg.terminations.pop("illegal_contact", None)
  return cfg


def unitree_go2_self_pull_traction_actor_v77_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v73-base contact-targeted polish without changing the command contract."""
  cfg = unitree_go2_self_pull_traction_actor_v72_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v69_focus"] = True
  # Keep training recoverable as in v72; strict rollout still checks contact.
  return cfg


def unitree_go2_self_pull_traction_actor_v78_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v77 continuation with extra 3-D/high-reverse transition coverage."""
  cfg = unitree_go2_self_pull_traction_actor_v77_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v68_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v79_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v73-best continuation with one balanced, rate-limited curriculum.

  v79 intentionally uses one final focus selector. Earlier versions enabled
  several selectors even though their implementations overwrite one another.
  """
  cfg = unitree_go2_self_pull_traction_actor_v72_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v79_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v82_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v81 failure-targeted contact-stability curriculum."""
  cfg = unitree_go2_self_pull_traction_actor_v72_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v82_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v83_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v83 dense speed/rotation braking objective for contact prevention."""
  cfg = unitree_go2_self_pull_traction_actor_v72_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v82_focus"] = True
  cfg.rewards["velocity_overshoot_l2"] = RewardTermCfg(
    func=self_pull_mdp.velocity_overshoot_l2,
    weight=-4.0,
    params={"command_name": "twist", "margin": 0.10},
  )
  cfg.rewards["body_ang_vel"].weight = -0.90
  cfg.rewards["angular_momentum"].weight = -0.22
  cfg.rewards["nonfoot_contact_history"].weight = -72.0
  return cfg


def unitree_go2_self_pull_traction_actor_v88_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v85-baseline contact-stability candidate for targeted short-cycle training."""
  cfg = unitree_go2_self_pull_traction_actor_v83_env_cfg(play=play)
  cfg.rewards["velocity_overshoot_l2"].weight = -5.0
  cfg.rewards["body_ang_vel"].weight = -1.10
  cfg.rewards["angular_momentum"].weight = -0.30
  cfg.rewards["nonfoot_contact_history"].weight = -96.0
  return cfg




def unitree_go2_self_pull_traction_actor_v89_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v89 narrow curriculum for low-force 90-degree and reverse/Z transitions."""
  cfg = unitree_go2_self_pull_traction_actor_v83_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v89_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v93_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v93 targeted continuation from the v85/Jump actor checkpoint.

  Keeps the frozen estimator and unified command shaping, while oversampling
  low-force 90-degree switching and high reverse plus vertical transitions.
  """
  cfg = unitree_go2_self_pull_traction_actor_v83_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v93_focus"] = True
  cfg.rewards["velocity_overshoot_l2"].weight = -5.0
  cfg.rewards["body_ang_vel"].weight = -1.10
  cfg.rewards["angular_momentum"].weight = -0.30
  cfg.rewards["nonfoot_contact_history"].weight = -96.0
  return cfg


def unitree_go2_self_pull_traction_actor_v94_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v94 low-rate balanced continuation from the v85 actor."""
  cfg = unitree_go2_self_pull_traction_actor_v83_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v94_focus"] = True
  cfg.rewards["velocity_overshoot_l2"].weight = -4.5
  cfg.rewards["body_ang_vel"].weight = -1.00
  cfg.rewards["angular_momentum"].weight = -0.25
  cfg.rewards["nonfoot_contact_history"].weight = -80.0
  return cfg



def unitree_go2_self_pull_traction_actor_v95_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  "v95 robust false-start and low-force turn polish from v94."
  cfg = unitree_go2_self_pull_traction_actor_v83_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v95_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -24.0
  cfg.rewards["track_low_linear_velocity"].weight = 7.0
  cfg.rewards["velocity_overshoot_l2"].weight = -4.5
  cfg.rewards["body_ang_vel"].weight = -1.05
  cfg.rewards["angular_momentum"].weight = -0.27
  cfg.rewards["nonfoot_contact_history"].weight = -84.0
  return cfg


def unitree_go2_self_pull_traction_actor_v99_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v99 robust anchor restoration continuation from v98."""
  cfg = unitree_go2_self_pull_traction_actor_v98_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v99_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -18.0
  cfg.rewards["track_low_linear_velocity"].weight = 7.0
  cfg.rewards["velocity_overshoot_l2"].weight = -5.0
  cfg.rewards["body_ang_vel"].weight = -1.10
  cfg.rewards["angular_momentum"].weight = -0.30
  cfg.rewards["nonfoot_contact_history"].weight = -90.0
  return cfg


def unitree_go2_self_pull_traction_actor_v100_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v100 actor stability polish from v99: hard switches plus strict stop anchors."""
  cfg = unitree_go2_self_pull_traction_actor_v99_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v100_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -32.0
  cfg.rewards["track_low_linear_velocity"].weight = 9.0
  cfg.rewards["velocity_overshoot_l2"].weight = -6.5
  cfg.rewards["body_ang_vel"].weight = -1.20
  cfg.rewards["angular_momentum"].weight = -0.34
  cfg.rewards["nonfoot_contact_history"].weight = -105.0
  return cfg


def unitree_go2_self_pull_traction_actor_v102_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v102 retraining after validated 0.7 horizontal command cap ablation."""
  cfg = unitree_go2_self_pull_traction_actor_v99_env_cfg(play=play)
  return cfg


def unitree_go2_self_pull_traction_actor_v110_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Failure-targeted continuation from the validated v102 task.

  The v114 continuation keeps this task name for checkpoint compatibility;
  the event sampler above is intentionally biased toward the remaining
  high-reverse/vertical stability case.
  """
  cfg = unitree_go2_self_pull_traction_actor_v102_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v110_focus"] = True
  cfg.rewards["track_low_linear_velocity"].weight = 7.0
  cfg.rewards["zero_command_velocity_l2"].weight = -19.0
  cfg.rewards["velocity_overshoot_l2"].weight = -5.8
  cfg.rewards["body_ang_vel"].weight = -1.25
  cfg.rewards["angular_momentum"].weight = -0.36
  cfg.rewards["nonfoot_contact_history"].weight = -105.0
  cfg.rewards["body_orientation_l2"].weight = -12.0
  cfg.rewards["angular_momentum"].weight = -0.50
  cfg.rewards["is_terminated"].weight = -1500.0
  return cfg


def unitree_go2_self_pull_traction_actor_v118_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v118 narrow continuation targeting v117's two residual failures."""
  cfg = unitree_go2_self_pull_traction_actor_v110_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v118_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v119_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v119 stability/low-force continuation from the validated v117 actor.

  Keeps the v118 sampler targets, but strengthens upright/Angular-momentum
  shaping and the low-speed tracking signal instead of changing the estimator.
  """
  cfg = unitree_go2_self_pull_traction_actor_v118_env_cfg(play=play)
  cfg.rewards["track_low_linear_velocity"].weight = 8.5
  cfg.rewards["body_orientation_l2"].weight = -17.0
  cfg.rewards["body_ang_vel"].weight = -1.50
  cfg.rewards["angular_momentum"].weight = -0.70
  return cfg


def unitree_go2_self_pull_traction_actor_v120_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v120 continuation targeting the 40 N forward-hold matrix failure."""
  cfg = unitree_go2_self_pull_traction_actor_v119_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v120_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v121_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v121 continuation from validated v119 targeting residual transitions."""
  cfg = unitree_go2_self_pull_traction_actor_v119_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v121_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v130_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v130 core-only robust curriculum, anchored to the validated v119 setup."""
  cfg = unitree_go2_self_pull_traction_actor_v119_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_core_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v131_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v131: v130 curriculum with stronger stop and upright stability penalties."""
  cfg = unitree_go2_self_pull_traction_actor_v119_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_core_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -24.0
  cfg.rewards["velocity_overshoot_l2"].weight = -7.0
  cfg.rewards["body_orientation_l2"].weight = -28.0
  cfg.rewards["body_ang_vel"].weight = -2.0
  cfg.rewards["angular_momentum"].weight = -0.9
  cfg.rewards["nonfoot_contact_history"].weight = -115.0
  return cfg


def unitree_go2_self_pull_traction_actor_v132_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v132: v130 core curriculum with moderate stop/upright regularization."""
  cfg = unitree_go2_self_pull_traction_actor_v119_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_core_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -21.0
  cfg.rewards["velocity_overshoot_l2"].weight = -5.5
  cfg.rewards["body_orientation_l2"].weight = -22.0
  cfg.rewards["body_ang_vel"].weight = -1.8
  cfg.rewards["angular_momentum"].weight = -0.85
  cfg.rewards["nonfoot_contact_history"].weight = -105.0
  return cfg


def unitree_go2_self_pull_traction_actor_v133_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v133: v132 curriculum with corrected body-frame estimator contract."""
  return unitree_go2_self_pull_traction_actor_v132_env_cfg(play=play)


def unitree_go2_self_pull_traction_actor_v134_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v134: targeted residual stop and low-force-turn continuation."""
  cfg = unitree_go2_self_pull_traction_actor_v133_env_cfg(play=play)
  # Keep the frozen estimator and shared command shaping unchanged.  These
  # weights target v133's robust no-pull drift and low-force 90-degree turn.
  cfg.rewards["zero_command_velocity_l2"].weight = -30.0
  cfg.rewards["track_low_linear_velocity"].weight = 10.0
  cfg.rewards["velocity_overshoot_l2"].weight = -6.0
  cfg.rewards["stand_still"].weight = -4.0
  return cfg


def unitree_go2_self_pull_traction_actor_v135_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v135: focused low-force 90-degree startup/release continuation."""
  cfg = unitree_go2_self_pull_traction_actor_v134_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v135_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v141_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v141: bounded low-force direction correction from the v138 baseline.

  Keep the v135 scenario curriculum and stability weights.  The additional
  reward acts only on 0.06--0.16 horizontal commands and rewards signed
  directional projection rather than speed magnitude, preventing the
  three-dimensional instability observed after increasing the generic
  low-force tracking reward in v139/v140.
  """
  cfg = unitree_go2_self_pull_traction_actor_v135_env_cfg(play=play)
  # ``stand_still`` used a 0.10 command threshold.  A 10 N pull maps to an
  # exact 0.10 command, so this term penalized the same low-force motion that
  # the tracking rewards were trying to create.  Zero-command drift is
  # already covered more precisely by ``zero_command_velocity_l2`` (0.04).
  del cfg.rewards["stand_still"]
  cfg.rewards["track_low_linear_velocity"].weight = 6.0
  cfg.rewards["align_low_force_direction"] = RewardTermCfg(
    func=self_pull_mdp.align_low_force_direction,
    weight=2.0,
    params={
      "command_name": "twist",
      "command_min": 0.06,
      "command_max": 0.16,
      "scale": 0.08,
    },
  )
  # v143 targeted continuation: slightly stronger signed direction and upright
  # shaping for residual low-force 90-degree and 3-D transient failures.
  cfg.rewards["align_low_force_direction"].weight = 4.0
  cfg.rewards["body_orientation_l2"].weight = -28.0
  cfg.rewards["body_ang_vel"].weight = -2.1
  return cfg


def unitree_go2_self_pull_traction_actor_v145_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v145: v144 continuation with redundant reward ablation."""
  cfg = unitree_go2_self_pull_traction_actor_v141_env_cfg(play=play)
  cfg.rewards.pop("angular_momentum", None)
  return cfg


def unitree_go2_self_pull_traction_actor_v146_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v146: targeted continuation after v145 reward ablation."""
  cfg = unitree_go2_self_pull_traction_actor_v145_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v146_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v147_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v147: conservative targeted continuation after v146 regression."""
  cfg = unitree_go2_self_pull_traction_actor_v145_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v146_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v148_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v148: reward ablation removing redundant angular tracking terms."""
  cfg = unitree_go2_self_pull_traction_actor_v145_env_cfg(play=play)
  cfg.rewards.pop("track_angular_velocity", None)
  return cfg


def unitree_go2_self_pull_traction_actor_v149_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v149: mild targeted continuation from v148."""
  cfg = unitree_go2_self_pull_traction_actor_v148_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v146_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v150_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v150: balanced combined-force and diagonal curriculum from Jump base."""
  cfg = unitree_go2_self_pull_traction_actor_v148_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v150_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v151_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_self_pull_traction_actor_v148_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v151_focus"] = True
  return cfg


def _unitree_go2_self_pull_traction_actor_v152_env_cfg(
  stage: int, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Stable v152 continuation with an explicit three-stage force curriculum."""
  cfg = unitree_go2_self_pull_traction_actor_v148_env_cfg(play=play)
  tether = cfg.events["tether_pull"]
  tether.params["traction_actor_v152_stage"] = stage
  tether.params["traction_cycle_s"] = 8.0
  # During training, optionally sample traction directions in the body frame.
  # This provides yaw-invariant coverage while play/Viser remains world_fixed.
  if (not play) and os.environ.get("GUIDE_DOG_V152_FORCE_BODY_AUG") == "1":
    tether.params["force_frame"] = "legacy_body"
  # Keep the v148 gait, but make fall prevention and post-release braking
  # materially stronger without returning to the over-regularized v125 setup.
  cfg.rewards["body_orientation_l2"].weight = -30.0
  cfg.rewards["body_ang_vel"].weight = -2.20
  cfg.rewards["nonfoot_contact_history"].weight = -110.0
  cfg.rewards["base_to_feet_clearance"].weight = -258.0
  cfg.rewards["is_terminated"].weight = -1650.0
  cfg.rewards["zero_command_velocity_l2"].weight = -32.0
  cfg.rewards["velocity_overshoot_l2"].weight = -6.5
  cfg.rewards["track_low_linear_velocity"].weight = 6.5
  cfg.rewards["action_rate_l2"].weight = -0.48
  cfg.rewards["action_rate_l2"].weight = float(
    os.environ.get("GUIDE_DOG_V152_ACTION_RATE_WEIGHT", "-0.48")
  )
  cfg.rewards["foot_slip"].weight = float(
    os.environ.get("GUIDE_DOG_V152_FOOT_SLIP_WEIGHT", "-0.25")
  )
  if os.environ.get("GUIDE_DOG_V152_STABLE_START", "1") == "1":
    cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    cfg.events["base_com"].params["ranges"] = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
  if os.environ.get("GUIDE_DOG_V152_N45_STABILITY") == "1":
    cfg.rewards["n45_hold_stability_l2"] = RewardTermCfg(
      func=self_pull_mdp.n45_hold_stability_l2,
      weight=-8.0,
    )
  if os.environ.get("GUIDE_DOG_V152_HIGH_FORCE_STABILITY") == "1":
    cfg.rewards["high_force_hold_stability_l2"] = RewardTermCfg(
      func=self_pull_mdp.high_force_hold_stability_l2,
      weight=float(os.environ.get("GUIDE_DOG_V152_HIGH_FORCE_WEIGHT", "-8.0")),
    )
  if os.environ.get("GUIDE_DOG_V152_LATERAL_STABILITY") == "1":
    cfg.rewards["lateral_high_force_hold_stability_l2"] = RewardTermCfg(
      func=self_pull_mdp.lateral_high_force_hold_stability_l2,
      weight=float(os.environ.get("GUIDE_DOG_V152_LATERAL_WEIGHT", "-20.0")),
    )
  if os.environ.get("GUIDE_DOG_V152_NEG_Y_STABILITY") == "1":
    cfg.rewards["negative_lateral_high_force_hold_stability_l2"] = RewardTermCfg(
      func=self_pull_mdp.negative_lateral_high_force_hold_stability_l2,
      weight=float(os.environ.get("GUIDE_DOG_V152_NEG_Y_WEIGHT", "-40.0")),
    )
  if os.environ.get("GUIDE_DOG_V152_PY_STABILITY") == "1":
    cfg.rewards["py_hold_stability_l2"] = RewardTermCfg(
      func=self_pull_mdp.n45_hold_stability_l2,
      weight=float(os.environ.get("GUIDE_DOG_V152_PY_WEIGHT", "1.0")),
      params={
        "target_direction_xy": (0.0, 1.0),
        "angular_scale": float(os.environ.get("GUIDE_DOG_V152_PY_ANGULAR_SCALE", "0.15")),
        "action_rate_scale": float(os.environ.get("GUIDE_DOG_V152_PY_ACTION_RATE_SCALE", "0.03")),
      },
    )
  if os.environ.get("GUIDE_DOG_V152_FOLLOW_REWARD") == "1":
    # Explicitly train directional motion under 15--30 N pulls and suppress
    # command/velocity residue after release. These terms use the same
    # body-frame force label as the estimator target.
    cfg.rewards["traction_force_direction_cosine"] = RewardTermCfg(
      func=self_pull_mdp.traction_force_direction_cosine,
      weight=float(os.environ.get("GUIDE_DOG_V152_FOLLOW_COS_WEIGHT", "1.5")),
    )
    cfg.rewards["traction_force_velocity_projection"] = RewardTermCfg(
      func=self_pull_mdp.traction_force_velocity_projection,
      weight=float(os.environ.get("GUIDE_DOG_V152_FOLLOW_PROJ_WEIGHT", "2.5")),
      params={
        "target_speed": float(os.environ.get("GUIDE_DOG_V152_TARGET_SPEED", "0.18")),
        "scale": float(os.environ.get("GUIDE_DOG_V152_TARGET_SPEED_STD", "0.08")),
        "min_force_n": float(os.environ.get("GUIDE_DOG_V152_FOLLOW_MIN_FORCE_N", "10.0")),
        "max_force_n": float(os.environ.get("GUIDE_DOG_V152_FOLLOW_MAX_FORCE_N", "32.0")),
      },
    )
    cfg.rewards["traction_release_gate_l2"] = RewardTermCfg(
      func=self_pull_mdp.traction_release_gate_l2,
      weight=float(os.environ.get("GUIDE_DOG_V152_RELEASE_WEIGHT", "-4.0")),
    )
  if os.environ.get("GUIDE_DOG_V152_TROT_FOCUS") == "1":
    # Prevent the policy from satisfying stability objectives by crawling.
    cfg.rewards["foot_gait"].weight = float(
      os.environ.get("GUIDE_DOG_V152_TROT_GAIT_WEIGHT", "2.0")
    )
    cfg.rewards["foot_clearance"].weight = float(
      os.environ.get("GUIDE_DOG_V152_TROT_CLEARANCE_WEIGHT", "-2.0")
    )
  if os.environ.get("GUIDE_DOG_V152_TROT_BREAKOUT") == "1":
    # Directly distinguish diagonal trot from the measured all-stance crawl.
    gait_params = cfg.rewards["foot_gait"].params
    clearance_asset_cfg = cfg.rewards["foot_clearance"].params["asset_cfg"]
    cfg.rewards["traction_velocity_progress"] = RewardTermCfg(
      func=self_pull_mdp.traction_velocity_progress,
      weight=float(os.environ.get("GUIDE_DOG_V152_PROGRESS_WEIGHT", "12.0")),
      params={
        "target_speed": float(os.environ.get("GUIDE_DOG_V152_PROGRESS_TARGET", "0.14")),
      },
    )
    cfg.rewards["traction_trot_contact"] = RewardTermCfg(
      func=self_pull_mdp.traction_trot_contact_reward,
      weight=float(os.environ.get("GUIDE_DOG_V152_TROT_CONTACT_WEIGHT", "6.0")),
      params={
        "period": gait_params["period"],
        "offset": gait_params["offset"],
        "stance_fraction": gait_params["threshold"],
        "sensor_name": gait_params["sensor_name"],
      },
    )
    cfg.rewards["traction_trot_swing"] = RewardTermCfg(
      func=self_pull_mdp.traction_trot_swing_reward,
      weight=float(os.environ.get("GUIDE_DOG_V152_TROT_SWING_WEIGHT", "8.0")),
      params={
        "period": gait_params["period"],
        "offset": gait_params["offset"],
        "stance_fraction": gait_params["threshold"],
        "target_height": float(os.environ.get("GUIDE_DOG_V152_SWING_HEIGHT", "0.055")),
        "sensor_name": gait_params["sensor_name"],
        "asset_cfg": clearance_asset_cfg,
      },
    )
    if os.environ.get("GUIDE_DOG_V152_MIN_SWING") == "1":
      cfg.rewards["traction_trot_min_swing"] = RewardTermCfg(
        func=self_pull_mdp.traction_trot_min_swing_reward,
        weight=float(os.environ.get("GUIDE_DOG_V152_MIN_SWING_WEIGHT", "12.0")),
        params={
          "period": gait_params["period"],
          "offset": gait_params["offset"],
          "stance_fraction": gait_params["threshold"],
          "target_height": float(os.environ.get("GUIDE_DOG_V152_SWING_HEIGHT", "0.055")),
          "sensor_name": gait_params["sensor_name"],
          "asset_cfg": clearance_asset_cfg,
        },
      )
    cfg.rewards["traction_all_stance_l2"] = RewardTermCfg(
      func=self_pull_mdp.traction_all_stance_l2,
      weight=float(os.environ.get("GUIDE_DOG_V152_ALL_STANCE_WEIGHT", "-5.0")),
      params={"sensor_name": gait_params["sensor_name"]},
    )
    cfg.rewards["traction_effort_saturation_l2"] = RewardTermCfg(
      func=self_pull_mdp.traction_effort_saturation_l2,
      weight=float(os.environ.get("GUIDE_DOG_V152_EFFORT_SAT_WEIGHT", "-2.0")),
      params={"effort_limits": (23.5, 23.5, 45.0) * 4},
    )
  return cfg


def unitree_go2_self_pull_traction_actor_v152_s1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _unitree_go2_self_pull_traction_actor_v152_env_cfg(1, play)


def unitree_go2_self_pull_traction_actor_v152_s2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _unitree_go2_self_pull_traction_actor_v152_env_cfg(2, play)


def unitree_go2_self_pull_traction_actor_v152_s3_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _unitree_go2_self_pull_traction_actor_v152_env_cfg(3, play)


def unitree_go2_self_pull_traction_actor_v123_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_self_pull_traction_actor_v119_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v123_focus"] = True
  return cfg


def unitree_go2_self_pull_traction_actor_v124_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v124 conservative actor recovery from the validated v119 distribution."""
  cfg = unitree_go2_self_pull_traction_actor_v119_env_cfg(play=play)
  cfg.rewards["body_orientation_l2"].weight = -22.0
  cfg.rewards["body_ang_vel"].weight = -1.80
  cfg.rewards["angular_momentum"].weight = -0.85
  cfg.rewards["zero_command_velocity_l2"].weight = -21.0
  return cfg


def unitree_go2_self_pull_traction_actor_v125_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v125: stability-only continuation from v124."""
  cfg = unitree_go2_self_pull_traction_actor_v119_env_cfg(play=play)
  cfg.rewards["body_orientation_l2"].weight = -35.0
  cfg.rewards["body_ang_vel"].weight = -2.50
  cfg.rewards["angular_momentum"].weight = -1.20
  cfg.rewards["zero_command_velocity_l2"].weight = -30.0
  cfg.rewards["velocity_overshoot_l2"].weight = -8.0
  return cfg


def unitree_go2_self_pull_traction_actor_v107_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Balanced failure-targeted continuation from the frozen v103 actor."""
  cfg = unitree_go2_self_pull_traction_actor_v99_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v107_focus"] = True
  # Small changes preserve the v103 gait while increasing the dense signal
  # for low-force startup, reversal braking and upright recovery.
  cfg.rewards["track_low_linear_velocity"].weight = 7.5
  cfg.rewards["zero_command_velocity_l2"].weight = -19.0
  cfg.rewards["velocity_overshoot_l2"].weight = -5.2
  cfg.rewards["body_ang_vel"].weight = -1.15
  cfg.rewards["angular_momentum"].weight = -0.32
  cfg.rewards["nonfoot_contact_history"].weight = -92.0
  return cfg


def unitree_go2_self_pull_traction_actor_v101_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v101 ablation: v100 hard-switch sampling with v99 reward weights."""
  cfg = unitree_go2_self_pull_traction_actor_v99_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v100_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -18.0
  cfg.rewards["track_low_linear_velocity"].weight = 7.0
  cfg.rewards["velocity_overshoot_l2"].weight = -5.0
  cfg.rewards["body_ang_vel"].weight = -1.10
  cfg.rewards["angular_momentum"].weight = -0.30
  cfg.rewards["nonfoot_contact_history"].weight = -90.0
  return cfg


def unitree_go2_self_pull_traction_actor_v98_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v98 targeted stability continuation from v97."""
  cfg = unitree_go2_self_pull_traction_actor_v96_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v98_focus"] = True
  cfg.rewards["velocity_overshoot_l2"].weight = -5.5
  cfg.rewards["body_ang_vel"].weight = -1.20
  cfg.rewards["angular_momentum"].weight = -0.32
  cfg.rewards["nonfoot_contact_history"].weight = -100.0
  return cfg


def unitree_go2_self_pull_traction_actor_v96_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """v96 conservative continuation from the v94 stable actor."""
  cfg = unitree_go2_self_pull_traction_actor_v83_env_cfg(play=play)
  cfg.events["tether_pull"].params["traction_actor_v94_focus"] = True
  cfg.rewards["zero_command_velocity_l2"].weight = -14.0
  cfg.rewards["track_low_linear_velocity"].weight = 6.2
  cfg.rewards["velocity_overshoot_l2"].weight = -4.5
  cfg.rewards["body_ang_vel"].weight = -1.00
  cfg.rewards["angular_momentum"].weight = -0.25
  cfg.rewards["nonfoot_contact_history"].weight = -80.0
  return cfg
