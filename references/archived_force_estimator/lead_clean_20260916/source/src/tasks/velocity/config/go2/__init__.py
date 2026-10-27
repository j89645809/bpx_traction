import os

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
  GuideDogV152Stage1Runner,
  GuideDogV152Stage2Runner,
  GuideDogV152Stage3Runner,
  GuideDogOnPolicyRunner,
  GuideDogResidualRunner,
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
  unitree_go2_self_pull_traction_actor_v23_env_cfg,
  unitree_go2_self_pull_traction_actor_v24_env_cfg,
  unitree_go2_self_pull_traction_actor_v25_env_cfg,
  unitree_go2_self_pull_traction_actor_v26_env_cfg,
  unitree_go2_self_pull_traction_actor_v31_env_cfg,
  unitree_go2_self_pull_traction_actor_v32_env_cfg,
  unitree_go2_self_pull_traction_actor_v33_env_cfg,
  unitree_go2_self_pull_traction_actor_v34_env_cfg,
  unitree_go2_self_pull_traction_actor_v36_env_cfg,
  unitree_go2_self_pull_traction_actor_v37_env_cfg,
  unitree_go2_self_pull_traction_actor_v40_env_cfg,
  unitree_go2_self_pull_traction_actor_v42_env_cfg,
  unitree_go2_self_pull_traction_actor_v41_env_cfg,
  unitree_go2_self_pull_traction_actor_v43_env_cfg,
  unitree_go2_self_pull_traction_actor_v44_env_cfg,
  unitree_go2_self_pull_traction_actor_v45_env_cfg,
  unitree_go2_self_pull_traction_actor_v46_env_cfg,
  unitree_go2_self_pull_traction_actor_v47_env_cfg,
  unitree_go2_self_pull_traction_actor_v48_env_cfg,
  unitree_go2_self_pull_traction_actor_v65_env_cfg,
  unitree_go2_self_pull_traction_actor_v66_env_cfg,
  unitree_go2_self_pull_traction_actor_v67_env_cfg,
  unitree_go2_self_pull_traction_actor_v68_env_cfg,
  unitree_go2_self_pull_traction_actor_v69_env_cfg,
  unitree_go2_self_pull_traction_actor_v70_env_cfg,
  unitree_go2_self_pull_traction_actor_v72_env_cfg,
  unitree_go2_self_pull_traction_actor_v77_env_cfg,
  unitree_go2_self_pull_traction_actor_v78_env_cfg,
  unitree_go2_self_pull_traction_actor_v79_env_cfg,
  unitree_go2_self_pull_traction_actor_v82_env_cfg,
  unitree_go2_self_pull_traction_actor_v83_env_cfg,
  unitree_go2_self_pull_traction_actor_v88_env_cfg,
  unitree_go2_self_pull_traction_actor_v89_env_cfg,
  unitree_go2_self_pull_traction_actor_v93_env_cfg,
  unitree_go2_self_pull_traction_actor_v94_env_cfg,
  unitree_go2_self_pull_traction_actor_v95_env_cfg,
  unitree_go2_self_pull_traction_actor_v96_env_cfg,
  unitree_go2_self_pull_traction_actor_v98_env_cfg,
  unitree_go2_self_pull_traction_actor_v99_env_cfg,
  unitree_go2_self_pull_traction_actor_v100_env_cfg,
  unitree_go2_self_pull_traction_actor_v101_env_cfg,
  unitree_go2_self_pull_traction_actor_v102_env_cfg,
  unitree_go2_self_pull_traction_actor_v110_env_cfg,
  unitree_go2_self_pull_traction_actor_v118_env_cfg,
  unitree_go2_self_pull_traction_actor_v119_env_cfg,
  unitree_go2_self_pull_traction_actor_v120_env_cfg,
  unitree_go2_self_pull_traction_actor_v121_env_cfg,
  unitree_go2_self_pull_traction_actor_v130_env_cfg,
  unitree_go2_self_pull_traction_actor_v131_env_cfg,
  unitree_go2_self_pull_traction_actor_v132_env_cfg,
  unitree_go2_self_pull_traction_actor_v133_env_cfg,
  unitree_go2_self_pull_traction_actor_v134_env_cfg,
  unitree_go2_self_pull_traction_actor_v135_env_cfg,
  unitree_go2_self_pull_traction_actor_v141_env_cfg,
  unitree_go2_self_pull_traction_actor_v145_env_cfg,
  unitree_go2_self_pull_traction_actor_v146_env_cfg,
  unitree_go2_self_pull_traction_actor_v147_env_cfg,
  unitree_go2_self_pull_traction_actor_v148_env_cfg,
  unitree_go2_self_pull_traction_actor_v149_env_cfg,
  unitree_go2_self_pull_traction_actor_v150_env_cfg,
  unitree_go2_self_pull_traction_actor_v151_env_cfg,
  unitree_go2_self_pull_traction_actor_v152_s1_env_cfg,
  unitree_go2_self_pull_traction_actor_v152_s2_env_cfg,
  unitree_go2_self_pull_traction_actor_v152_s3_env_cfg,
  unitree_go2_self_pull_traction_actor_v123_env_cfg,
  unitree_go2_self_pull_traction_actor_v124_env_cfg,
  unitree_go2_self_pull_traction_actor_v125_env_cfg,
  unitree_go2_self_pull_traction_actor_v123_env_cfg,
  unitree_go2_self_pull_traction_actor_v124_env_cfg,
  unitree_go2_self_pull_traction_actor_v123_env_cfg,
  unitree_go2_self_pull_traction_actor_v124_env_cfg,
  unitree_go2_self_pull_traction_actor_v107_env_cfg,
  unitree_go2_self_pull_gate5_wrench_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg
from .residual_cfg import (
  unitree_go2_residual_ppo_runner_cfg,
  unitree_go2_self_pull_residual_env_cfg,
)


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
_joint_robust_rl_cfg.wandb_project = os.environ.get("WANDB_PROJECT", "mjlab")

_v152_rl_cfg = unitree_go2_ppo_runner_cfg()
_v152_rl_cfg.algorithm.learning_rate = 5.0e-7
_v152_rl_cfg.algorithm.entropy_coef = 0.0
_v152_rl_cfg.algorithm.schedule = "fixed"
_v152_rl_cfg.save_interval = 250
_v152_rl_cfg.wandb_project = os.environ.get("WANDB_PROJECT", "guide-dog")

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

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v23",
  env_cfg=unitree_go2_self_pull_traction_actor_v23_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v23_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v24",
  env_cfg=unitree_go2_self_pull_traction_actor_v24_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v24_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v25",
  env_cfg=unitree_go2_self_pull_traction_actor_v25_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v25_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v26",
  env_cfg=unitree_go2_self_pull_traction_actor_v26_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v26_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v31",
  env_cfg=unitree_go2_self_pull_traction_actor_v31_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v31_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v32",
  env_cfg=unitree_go2_self_pull_traction_actor_v32_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v32_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v33",
  env_cfg=unitree_go2_self_pull_traction_actor_v33_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v33_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v34",
  env_cfg=unitree_go2_self_pull_traction_actor_v34_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v34_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v36",
  env_cfg=unitree_go2_self_pull_traction_actor_v36_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v36_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v37",
  env_cfg=unitree_go2_self_pull_traction_actor_v37_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v37_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v40",
  env_cfg=unitree_go2_self_pull_traction_actor_v40_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v40_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v41",
  env_cfg=unitree_go2_self_pull_traction_actor_v41_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v41_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v42",
  env_cfg=unitree_go2_self_pull_traction_actor_v42_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v42_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v43",
  env_cfg=unitree_go2_self_pull_traction_actor_v43_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v43_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v44",
  env_cfg=unitree_go2_self_pull_traction_actor_v44_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v44_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v45",
  env_cfg=unitree_go2_self_pull_traction_actor_v45_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v45_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v46",
  env_cfg=unitree_go2_self_pull_traction_actor_v46_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v46_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v47",
  env_cfg=unitree_go2_self_pull_traction_actor_v47_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v47_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v48",
  env_cfg=unitree_go2_self_pull_traction_actor_v48_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v48_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v65",
  env_cfg=unitree_go2_self_pull_traction_actor_v65_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v65_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v66",
  env_cfg=unitree_go2_self_pull_traction_actor_v66_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v66_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v67",
  env_cfg=unitree_go2_self_pull_traction_actor_v67_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v67_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v68",
  env_cfg=unitree_go2_self_pull_traction_actor_v68_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v68_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v69",
  env_cfg=unitree_go2_self_pull_traction_actor_v69_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v69_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v72",
  env_cfg=unitree_go2_self_pull_traction_actor_v72_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v72_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v77",
  env_cfg=unitree_go2_self_pull_traction_actor_v77_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v77_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v78",
  env_cfg=unitree_go2_self_pull_traction_actor_v78_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v78_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v79",
  env_cfg=unitree_go2_self_pull_traction_actor_v79_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v79_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v82",
  env_cfg=unitree_go2_self_pull_traction_actor_v82_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v82_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v83",
  env_cfg=unitree_go2_self_pull_traction_actor_v83_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v83_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v70",
  env_cfg=unitree_go2_self_pull_traction_actor_v70_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v70_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v88",
  env_cfg=unitree_go2_self_pull_traction_actor_v88_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v88_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v89",
  env_cfg=unitree_go2_self_pull_traction_actor_v89_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v89_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v93",
  env_cfg=unitree_go2_self_pull_traction_actor_v93_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v93_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v94",
  env_cfg=unitree_go2_self_pull_traction_actor_v94_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v94_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v95",
  env_cfg=unitree_go2_self_pull_traction_actor_v95_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v95_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v96",
  env_cfg=unitree_go2_self_pull_traction_actor_v96_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v96_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v98",
  env_cfg=unitree_go2_self_pull_traction_actor_v98_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v98_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v99",
  env_cfg=unitree_go2_self_pull_traction_actor_v99_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v99_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v100",
  env_cfg=unitree_go2_self_pull_traction_actor_v100_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v100_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v101",
  env_cfg=unitree_go2_self_pull_traction_actor_v101_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v101_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v102",
  env_cfg=unitree_go2_self_pull_traction_actor_v102_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v102_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v107",
  env_cfg=unitree_go2_self_pull_traction_actor_v107_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v107_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v110",
  env_cfg=unitree_go2_self_pull_traction_actor_v110_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v110_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v118",
  env_cfg=unitree_go2_self_pull_traction_actor_v118_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v118_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v119",
  env_cfg=unitree_go2_self_pull_traction_actor_v119_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v119_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v120",
  env_cfg=unitree_go2_self_pull_traction_actor_v120_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v120_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v121",
  env_cfg=unitree_go2_self_pull_traction_actor_v121_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v121_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v130",
  env_cfg=unitree_go2_self_pull_traction_actor_v130_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v130_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v131",
  env_cfg=unitree_go2_self_pull_traction_actor_v131_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v131_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v132",
  env_cfg=unitree_go2_self_pull_traction_actor_v132_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v132_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v133",
  env_cfg=unitree_go2_self_pull_traction_actor_v133_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v133_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v134",
  env_cfg=unitree_go2_self_pull_traction_actor_v134_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v134_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v135",
  env_cfg=unitree_go2_self_pull_traction_actor_v135_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v135_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v141",
  env_cfg=unitree_go2_self_pull_traction_actor_v141_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v141_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v145",
  env_cfg=unitree_go2_self_pull_traction_actor_v145_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v145_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)


register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v146",
  env_cfg=unitree_go2_self_pull_traction_actor_v146_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v146_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v147",
  env_cfg=unitree_go2_self_pull_traction_actor_v147_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v147_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v148",
  env_cfg=unitree_go2_self_pull_traction_actor_v148_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v148_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v149",
  env_cfg=unitree_go2_self_pull_traction_actor_v149_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v149_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v150",
  env_cfg=unitree_go2_self_pull_traction_actor_v150_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v150_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v151",
  env_cfg=unitree_go2_self_pull_traction_actor_v151_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v151_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v152-s1",
  env_cfg=unitree_go2_self_pull_traction_actor_v152_s1_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v152_s1_env_cfg(play=True),
  rl_cfg=_v152_rl_cfg,
  runner_cls=GuideDogV152Stage1Runner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v152-s2",
  env_cfg=unitree_go2_self_pull_traction_actor_v152_s2_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v152_s2_env_cfg(play=True),
  rl_cfg=_v152_rl_cfg,
  runner_cls=GuideDogV152Stage2Runner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v152-s3",
  env_cfg=unitree_go2_self_pull_traction_actor_v152_s3_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v152_s3_env_cfg(play=True),
  rl_cfg=_v152_rl_cfg,
  runner_cls=GuideDogV152Stage3Runner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v123",
  env_cfg=unitree_go2_self_pull_traction_actor_v123_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v123_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v124",
  env_cfg=unitree_go2_self_pull_traction_actor_v124_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v124_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-TractionActor-v125",
  env_cfg=unitree_go2_self_pull_traction_actor_v125_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_traction_actor_v125_env_cfg(play=True),
  rl_cfg=_joint_robust_rl_cfg,
  runner_cls=GuideDogJointRobustActorRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-SelfPull-Residual-v1",
  env_cfg=unitree_go2_self_pull_residual_env_cfg(),
  play_env_cfg=unitree_go2_self_pull_residual_env_cfg(play=True),
  rl_cfg=unitree_go2_residual_ppo_runner_cfg(),
  runner_cls=GuideDogResidualRunner,
)
