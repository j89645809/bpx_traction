"""BPX-Pro 行走训练用的 ArticulationCfg。

和辨识环境（pace_bpx_pro 的 `bpx_pro_pace_env_cfg.py`）的区别：
  * 自由机身 + 地面，不是 `fix_root_link=True` 的悬空
  * PD 用部署的 kp40 / kd1（辨识时是 kp30 / kd1），力矩限幅 30 N·m 对齐板子
  * 初始姿态 = 板子上的 `DEFAULT_POSE`，也是策略的 q0

关节级参数（armature / 粘滞 / 库伦摩擦）在这里填辨识标称值；逐回合的随机化由
`mdp.events` 覆盖。零偏标称 0、延迟由事件写入。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from . import spec
from .actuator import PaceDCMotorCfg
from .params import TrainingParams, get_params


def _per_joint_dict(values: list[float], joint_names: list[str]) -> dict[str, float]:
    """把 12 维（SDK 顺序）的值挑出指定关节，变成 {关节名: 值}。"""
    idx = {n: i for i, n in enumerate(spec.JOINT_NAMES)}
    return {n: float(values[idx[n]]) for n in joint_names}


def _actuator_cfg(p: TrainingParams, joint_names: list[str], jt: str) -> PaceDCMotorCfg:
    """按关节类型（hip_roll / hip_pitch / knee）建一组执行器。

    `saturation_effort` 用 URDF 峰值 × 饱和系数（堵转力矩），
    `effort_limit` 用部署限幅 30 N·m —— 板子上的 `torque_limited_pd` 就是这么裁的。
    """
    return PaceDCMotorCfg(
        joint_names_expr=joint_names,
        stiffness={n: p.kp for n in joint_names},
        damping={n: p.kd for n in joint_names},
        effort_limit=p.effort_limit_deploy,
        saturation_effort=spec.EFFORT_LIMIT[jt] * p.saturation_factor,
        velocity_limit=spec.VELOCITY_LIMIT[jt],
        armature=_per_joint_dict(p.armature.nominal, joint_names),
        friction=_per_joint_dict(p.frictionloss.nominal, joint_names),
        dynamic_friction=_per_joint_dict(p.frictionloss.nominal, joint_names),
        viscous_friction=_per_joint_dict(p.damping.nominal, joint_names),
        encoder_bias={n: 0.0 for n in joint_names},
        max_delay=p.max_delay_steps,
    )


def make_robot_cfg(params: TrainingParams | None = None) -> ArticulationCfg:
    p = params or get_params()
    by_type: dict[str, list[str]] = {jt: [] for jt in spec.JOINT_TYPES}
    for n in spec.JOINT_NAMES:
        by_type[spec.joint_type(n)].append(n)

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=spec.default_usd_path(),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # 真机站立时机身高度 0.39–0.44 m（走路记录里的 init_z），0.45 留一点余量落下去
            pos=(0.0, 0.0, 0.45),
            joint_pos={n: q for n, q in zip(spec.JOINT_NAMES, p.default_joint_pos)},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "hip_roll": _actuator_cfg(p, by_type["hip_roll"], "hip_roll"),
            "hip_pitch": _actuator_cfg(p, by_type["hip_pitch"], "hip_pitch"),
            "knee": _actuator_cfg(p, by_type["knee"], "knee"),
        },
    )
