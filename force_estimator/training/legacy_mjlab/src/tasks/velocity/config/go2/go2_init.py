from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import (
  GuideDogCalibratedFadeOnPolicyRunner,
  GuideDogCalibratedTeacherOnPolicyRunner,
  GuideDogFadeOnPolicyRunner,
  GuideDogGate2FyRunner,
  GuideDogGate21FyActiveRunner,
  GuideDogGate3PlanarRunner,
  GuideDogGate31PlanarBalancedRunner,
  GuideDogGate43DRunner,
  GuideDogGate41SlowdownRunner,
  GuideDogDirectionFinetuneRunner,
  GuideDogGate5WrenchRunner,
  GuideDogJointRobustActorRunner,
  GuideDogOnPolicyRunner,
  GuideDogTeacherOnPolicyRunner,
  VelocityOnPolicyRunner,
)

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_self_pull_env_cfg,
  unitree_go2_self_pull_fx_only_env_cfg,
  unitree_go2_self_pull_fx_bilateral_env_cfg,
  unitree_go2_self_pull_fx_teacher20_env_cfg,
  unitree_go2_self_pull_fx_teacher20_polish_env_cfg,
  unitree_go2_self_pull_gate2_fy_env_cfg,
  unitree_go2_self_pull_gate21_fy_active_env_cfg,
  unitree_go2_self_pull_gate3_planar_env_cfg,
  unitree_go2_self_pull_gate31_planar_balanced_env_cfg,
  unitree_go2_self_pull_gate4_3d_env_cfg,
  unitree_go2_self_pull_gate41_slowdown_env_cfg,
  unitree_go2_self_pull_gate41_world_env_cfg,
  unitree_go2_self_pull_direction_finetune_env_cfg,
  unitree_go2_self_pull_continuous_jump_env_cfg,
  unitree_go2_self_pull_deployment_observable_v20_env_cfg,
  unitree_go2_self_pull_joint_robust_matrix_v20_env_cfg,
  unitree_go2_self_pull_joint_robust_ramp_v20_env_cfg,
  unitree_go2_self_pull_joint_robust_jump_v20_env_cfg,
  unitree_go2_self_pull_joint_robust_recovery_v20_env_cfg,
  unitree_go2_self_pull_gate5_wrench_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull",
  env_cfg=unitree_go2_self_pull_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-FxOnly",
  env_cfg=unitree_go2_self_pull_fx_only_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_fx_only_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-FxBilateral",
  env_cfg=unitree_go2_self_pull_fx_bilateral_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_fx_bilateral_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-FxTeacher20",
  env_cfg=unitree_go2_self_pull_fx_teacher20_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_fx_teacher20_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-FxTeacher20-Polish",
  env_cfg=unitree_go2_self_pull_fx_teacher20_polish_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_fx_teacher20_polish_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogTeacherOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-FxTeacher20-Fade",
  env_cfg=unitree_go2_self_pull_fx_teacher20_polish_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_fx_teacher20_polish_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogFadeOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-FxTeacher20-Calibrate",
  env_cfg=unitree_go2_self_pull_fx_teacher20_polish_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_fx_teacher20_polish_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogCalibratedTeacherOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-FxTeacher20-CalibratedFade",
  env_cfg=unitree_go2_self_pull_fx_teacher20_polish_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_fx_teacher20_polish_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogCalibratedFadeOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Gate2-Fy",
  env_cfg=unitree_go2_self_pull_gate2_fy_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_gate2_fy_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogGate2FyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Gate21-FyActive",
  env_cfg=unitree_go2_self_pull_gate21_fy_active_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_gate21_fy_active_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogGate21FyActiveRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Gate3-Planar",
  env_cfg=unitree_go2_self_pull_gate3_planar_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_gate3_planar_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogGate3PlanarRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Gate31-PlanarBalanced",
  env_cfg=unitree_go2_self_pull_gate31_planar_balanced_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_gate31_planar_balanced_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogGate31PlanarBalancedRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Gate4-3D",
  env_cfg=unitree_go2_self_pull_gate4_3d_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_gate4_3d_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogGate43DRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Gate41-Slowdown",
  env_cfg=unitree_go2_self_pull_gate41_slowdown_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_gate41_slowdown_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogGate41SlowdownRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Gate41-World",
  env_cfg=unitree_go2_self_pull_gate41_world_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_gate41_world_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogGate41SlowdownRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Direction-Finetune",
  env_cfg=unitree_go2_self_pull_direction_finetune_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_direction_finetune_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogDirectionFinetuneRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Continuous-Jump-v19",
  env_cfg=unitree_go2_self_pull_continuous_jump_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_continuous_jump_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogDirectionFinetuneRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Deployment-Observable-v20",
  env_cfg=unitree_go2_self_pull_deployment_observable_v20_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_deployment_observable_v20_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogCalibratedFadeOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Gate5-Wrench",
  env_cfg=unitree_go2_self_pull_gate5_wrench_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_gate5_wrench_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=GuideDogGate5WrenchRunner,
)
_joint_robust_rl_cfg = unitree_go2_ppo_runner_cfg()
_joint_robust_rl_cfg.algorithm.learning_rate = 1.0e-5
_joint_robust_rl_cfg.algorithm.entropy_coef = 0.001

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-JointRobustActor-v20",
  env_cfg=unitree_go2_self_pull_joint_robust_matrix_v20_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_joint_robust_matrix_v20_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-JointRobustRamp-v20",
  env_cfg=unitree_go2_self_pull_joint_robust_ramp_v20_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_joint_robust_ramp_v20_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-JointRobustJump-v20",
  env_cfg=unitree_go2_self_pull_joint_robust_jump_v20_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_joint_robust_jump_v20_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-JointRobustRecovery-v20",
  env_cfg=unitree_go2_self_pull_joint_robust_recovery_v20_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_joint_robust_recovery_v20_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)
