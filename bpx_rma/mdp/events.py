"""BPX-Pro 的域随机化事件。

写进仿真的量和 PACE 辨识时 `CMAESOptimizer.update_simulator` 写的是同一组，保证
"辨识出来的数" 和 "训练用的数" 走同一条路：

  armature      -> ``write_joint_armature_to_sim``
  粘滞摩擦       -> 关节 viscous friction（不是 PD 的 kd）
  库伦摩擦       -> 关节 static = dynamic friction
  编码器零偏     -> ``PaceDCMotor.update_encoder_bias``
  力矩延迟       -> ``PaceDCMotor.update_time_lags``

膝承重摩擦有两种做法，由 yaml 的 ``knee_friction.load_model.enabled`` 切换：

  **方案 A**（enabled: false）每回合抽一个常数摩擦，区间 1.5–3.5 N·m
  **方案 B**（enabled: true） 每个控制步按负载更新 F = f0 + mu·L，
      L = |tau - F·sign(v)| 的低通（时间常数 tau_s，只在关节在动时更新）

方案 B 的规则和 MuJoCo 侧 `mujoco_floor.FloorSim.set_load_friction` /
`isaac_friction_check.LoadFriction` 完全一致，这样仿真、重放、训练三边对得上。

注意：``interval`` 事件在 `ManagerBasedRLEnv.step()` 里是**物理步进之后**才触发的，
所以这一步算出来的摩擦下一个控制步（20 ms 后）才生效。负载估计本身是 tau_s=0.2 s
的低通，这点滞后可以忽略。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .. import spec
from .state import get_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _uniform(lo: torch.Tensor, hi: torch.Tensor, shape: tuple[int, ...], device) -> torch.Tensor:
    return torch.rand(shape, device=device) * (hi - lo) + lo


def _write_friction(asset: Articulation, static, dynamic, viscous, joint_ids, env_ids) -> None:
    """一次写静 / 动 / 粘滞摩擦。

    PhysX 要求静摩擦 >= 动摩擦，分三次写会出现中间态违规；Isaac Sim 5.x 的组合接口
    一次 get/set 就不会。
    """
    asset.write_joint_friction_coefficient_to_sim(
        static,
        joint_dynamic_friction_coeff=dynamic,
        joint_viscous_friction_coeff=viscous,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )


def randomize_actuator_params(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dq_filter_tau_range: tuple[float, float] | None = None,
) -> None:
    """reset 事件：抽这一回合的 armature / 粘滞 / 库伦摩擦 / 零偏 / 延迟。

    区间全部来自 ``data/bpx_pro_training_params.yaml``：
    hip_roll / hip_pitch 可辨识性差，±100%；knee 三套方法一致，armature / 粘滞 ±40%，
    摩擦按承重实测区间 1.5–3.5 N·m 给绝对值。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    st = get_state(env)
    st.resolve_joint_ids(asset)
    p = st.params
    dev = env.device

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=dev)
    n = len(env_ids)
    jids = st.joint_ids_sdk
    shape = (n, spec.NUM_JOINTS)

    def band(ranges) -> torch.Tensor:
        lo = torch.tensor(ranges.low, device=dev)
        hi = torch.tensor(ranges.high, device=dev)
        return _uniform(lo, hi, shape, dev)

    armature = band(p.armature)
    viscous = band(p.damping)
    friction = band(p.frictionloss)

    st.armature[env_ids] = armature
    st.viscous[env_ids] = viscous
    st.friction[env_ids] = friction

    asset.write_joint_armature_to_sim(armature, joint_ids=jids, env_ids=env_ids)
    _write_friction(asset, friction, friction, viscous, jids, env_ids)

    # 零偏：标称 0，逐回合逐关节 ±0.03 rad
    lo, hi = p.encoder_bias_range
    bias = torch.rand(shape, device=dev) * (hi - lo) + lo
    st.encoder_bias[env_ids] = bias

    # 延迟：2–8 ms -> 500 Hz 下 1–4 步（整数步）
    # DelayBuffer 的时滞缓冲是 int32，randint 默认出 int64，必须转，否则 index_put_ 报 dtype 不匹配
    d_lo, d_hi = p.delay_steps_range
    delay = torch.randint(int(round(d_lo)), int(round(d_hi)) + 1, (n,), device=dev).to(torch.int)
    st.delay_steps[env_ids] = delay.float()

    # dq 滤波滞后（模拟板载滤波）：每回合每 env 抽一个时间常数，**全部关节共用**
    # （真机是一个滤波器管所有关节）。不传该参数 = 不滤（旧行为）。
    tau = None
    if dq_filter_tau_range is not None:
        lo_t, hi_t = dq_filter_tau_range
        tau = torch.rand(n, device=dev) * (hi_t - lo_t) + lo_t

    # 写进执行器：每组执行器只拿自己那几个关节的那一列
    sdk_index = {name: i for i, name in enumerate(spec.JOINT_NAMES)}
    for actuator in asset.actuators.values():
        if not hasattr(actuator, "update_encoder_bias"):
            continue
        cols = torch.tensor([sdk_index[n] for n in actuator.joint_names], device=dev, dtype=torch.long)
        actuator.update_encoder_bias(bias[:, cols], env_ids=env_ids)
        actuator.update_time_lags(delay, env_ids=env_ids)
        if tau is not None and hasattr(actuator, "update_vel_filter"):
            actuator.update_vel_filter(tau, env_ids, dt=env.physics_dt)

    # 膝负载模型这一回合的 f0 / mu，并把负载估计清零
    kf = p.knee_load_friction
    k = len(spec.KNEE_JOINT_NAMES)
    if kf.enabled:
        st.knee_f0[env_ids] = _uniform(
            torch.tensor(kf.f0_range[0], device=dev), torch.tensor(kf.f0_range[1], device=dev), (n, k), dev
        )
        st.knee_mu[env_ids] = _uniform(
            torch.tensor(kf.mu_range[0], device=dev), torch.tensor(kf.mu_range[1], device=dev), (n, k), dev
        )
        # 负载初值：用站立时膝的静态负载量级（真机前膝 5.8 / 后膝 9.1 N·m）起步，
        # 避免第一步摩擦掉到 f0 导致站不住
        st.knee_load[env_ids] = 7.0
    else:
        st.knee_f0[env_ids] = 0.0
        st.knee_mu[env_ids] = 0.0
        st.knee_load[env_ids] = 0.0


def knee_load_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """interval 事件（每控制步）：按负载更新膝摩擦 F = f0 + mu·L（方案 B）。

    yaml 里 ``load_model.enabled: false`` 时直接返回，退化成方案 A 的常数摩擦。
    """
    st = get_state(env)
    kf = st.params.knee_load_friction
    if not kf.enabled:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    st.resolve_joint_ids(asset)
    kids = st.knee_ids_sdk
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    tau = asset.data.applied_torque[env_ids][:, kids]
    vel = asset.data.joint_vel[env_ids][:, kids]
    f_prev = st.friction[env_ids][:, torch.tensor(spec.KNEE_INDICES, device=env.device)]

    # L = |tau - F·sign(v)|，只在关节在动的时候以 a = dt/tau_s 低通更新
    l_inst = (tau - f_prev * torch.sign(vel)).abs()
    a = min(1.0, env.step_dt / kf.tau_s)
    moving = vel.abs() > kf.v_move
    load = st.knee_load[env_ids]
    load = torch.where(moving, load + (l_inst - load) * a, load)
    st.knee_load[env_ids] = load

    f_new = (st.knee_f0[env_ids] + st.knee_mu[env_ids] * load).clamp(kf.clamp[0], kf.clamp[1])
    # 记回 SDK 顺序的 12 维里，特权观测和下一步的 f_prev 都从这里读
    full = st.friction[env_ids]
    full[:, torch.tensor(spec.KNEE_INDICES, device=env.device)] = f_new
    st.friction[env_ids] = full

    visc = st.viscous[env_ids][:, torch.tensor(spec.KNEE_INDICES, device=env.device)]
    _write_friction(asset, f_new, f_new, visc, kids, env_ids)


def randomize_total_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[spec.TORSO_BODY]),
) -> None:
    """reset 事件：把整机质量抽到 yaml 的区间（默认 15.1 ± 1 kg），只改 torso，惯量同比例缩放。

    USD 是 17.55 kg，比真机重约 2.45 kg。和 `landing_compare.set_total_mass` /
    `isaac_friction_check.set_total_mass` 同一套做法。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    st = get_state(env)
    lo, hi = st.params.mass_range_kg
    dev = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=dev)

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        raise ValueError("randomize_total_mass 需要指定 body_names=[torso]")
    tid = int(body_ids[0])

    env_ids_cpu = env_ids.detach().cpu()
    masses = asset.root_physx_view.get_masses()          # (num_envs, num_bodies)，CPU 张量
    inertias = asset.root_physx_view.get_inertias()

    total = masses[env_ids_cpu].sum(dim=1)
    old_torso = masses[env_ids_cpu, tid]
    target = torch.rand(len(env_ids_cpu)) * (hi - lo) + lo
    new_torso = old_torso + (target - total)
    if bool((new_torso < 1.0).any()):
        raise ValueError(f"目标质量 {lo}~{hi} kg 太小：光四条腿就有 {float((total - old_torso).max()):.2f} kg")

    scale = (new_torso / old_torso).unsqueeze(-1)
    masses[env_ids_cpu, tid] = new_torso
    inertias[env_ids_cpu, tid] = inertias[env_ids_cpu, tid] * scale

    asset.root_physx_view.set_masses(masses, env_ids_cpu)
    asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

    st.total_mass[env_ids] = target.to(dev)
    st.mass_offset[env_ids] = (target - total).to(dev)
