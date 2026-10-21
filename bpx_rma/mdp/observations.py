"""特权观测（只给 teacher 看）。

这些就是辨识辨出来的那些量 —— 让 teacher 直接看到真值、student 只能从 25 帧历史里
猜。如果辨识是对的，student 在真机上估出来的隐含参数应该落在辨识值附近；这正是
"用策略验证辨识结果" 的核心证据。

策略那 45 维观测不在这里，用的是 IsaacLab 自带的 mdp 项（``joint_pos_rel`` /
``joint_vel`` / ``base_ang_vel`` / ``projected_gravity`` / ``last_action`` /
``generated_commands``），顺序由 `velocity_env_cfg.PolicyFrameCfg` 固定成和板子一致。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .. import spec
from .state import get_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def actuator_armature_ratio(env: ManagerBasedRLEnv) -> torch.Tensor:
    """这一回合的 armature / 标称值 - 1，(N, 12)。0 表示正好是辨识值。"""
    st = get_state(env)
    return st.armature / st.armature_nominal - 1.0


def actuator_viscous_ratio(env: ManagerBasedRLEnv) -> torch.Tensor:
    """粘滞摩擦 / 标称值 - 1，(N, 12)。"""
    st = get_state(env)
    return st.viscous / st.viscous_nominal - 1.0


def actuator_friction_ratio(env: ManagerBasedRLEnv) -> torch.Tensor:
    """库伦摩擦 / 标称值 - 1，(N, 12)。膝关节在负载模型下每步都在变。"""
    st = get_state(env)
    return st.friction / st.friction_nominal - 1.0


def knee_friction_abs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """膝关节当前实际写进 PhysX 的库伦摩擦 [N·m]，(N, 4)。"""
    st = get_state(env)
    idx = torch.tensor(spec.KNEE_INDICES, device=env.device)
    return st.friction[:, idx]


def knee_load(env: ManagerBasedRLEnv) -> torch.Tensor:
    """膝关节负载估计 L [N·m]，(N, 4)。负载模型关掉时恒为 0。"""
    return get_state(env).knee_load


def encoder_bias(env: ManagerBasedRLEnv) -> torch.Tensor:
    """编码器零偏 [rad]，(N, 12)。"""
    return get_state(env).encoder_bias


def torque_delay_steps(env: ManagerBasedRLEnv) -> torch.Tensor:
    """力矩延迟 [仿真步]，(N, 1)。500 Hz 下 1 步 = 2 ms。"""
    return get_state(env).delay_steps.unsqueeze(-1)


def mass_offset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """整机质量相对 USD 原值的偏移 [kg]，(N, 1)。"""
    return get_state(env).mass_offset.unsqueeze(-1)


def total_mass(env: ManagerBasedRLEnv) -> torch.Tensor:
    """整机质量 [kg]，(N, 1)。"""
    return get_state(env).total_mass.unsqueeze(-1)


def feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=spec.FOOT_BODIES),
) -> torch.Tensor:
    """四只脚是否触地（0/1），(N, 4)，顺序 fl, fr, hl, hr。"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]  # (N, T, 4, 3)
    return (forces.norm(dim=-1).max(dim=1)[0] > 1.0).float()


def feet_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=spec.FOOT_BODIES),
) -> torch.Tensor:
    """四只脚的竖直接触力 [N]，(N, 4)。"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]


def joint_vel_filtered(env: ManagerBasedRLEnv) -> torch.Tensor:
    """执行器内一阶低通后的关节速度，(N, 12)，SDK 顺序。

    真机 SDK 返回的 dq 是板载滤波值（带滞后），策略观测和 500 Hz PD 阻尼都用它。
    仿真里滤波在 `PaceDCMotor.compute` 每物理步更新，这里**只读**它的
    `filtered_vel` —— 幂等，policy / history 两个组各调一次也不会把滤波器打乱。
    不开滤波（alpha=1）时它逐位等于真速度，与旧观测完全一致。
    """
    st = get_state(env)
    asset: Articulation = env.scene["robot"]
    st.resolve_joint_ids(asset)
    out = torch.zeros(env.num_envs, len(st.joint_ids_sdk), device=env.device)
    for name, cols in st.actuator_sdk_cols:
        out[:, cols] = asset.actuators[name].filtered_vel
    return out


def joint_applied_torque(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=spec.JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    """实际施加的关节力矩 [N·m]，(N, 12)，SDK 顺序。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]
