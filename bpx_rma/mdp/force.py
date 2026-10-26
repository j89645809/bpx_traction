"""力感知 RMA（ForceAware 任务）：牵引外力 + 力估计器误差模型 + 两个新观测项。

接口约定见 ``docs/force_aware_rma_interface.md``。这里做三件事：

1. **施力**（每控制步的 interval 事件 :func:`tether_force_step`）：世界系方向固定的绳拉力，
   按"段"生成 —— 每段 1–3 s，随机选 置零 / 新目标（突加或斜坡）/ 释放 / 反向，
   大小 0–45 N、方向全 3D。作用在 torso 上一个每回合随机的挂点
   （x ±0.10 / y ±0.06 / z 0–0.08 m，机身系），力矩 = p × F。
2. **估计器误差模型**：仿真里没有真的估计器，teacher 看到的"估计力"是真值加上
   延迟、大小/方向噪声、回合偏置、限幅和遮蔽段；置信度 = 1 − clip(|误差| / 15 N, 0, 1)。
   这样 student 换成真估计器输出时分布不会突变。
3. **观测**：:func:`estimated_force_obs` / :func:`force_confidence_obs` 只读缓存。

**为什么噪声在事件里采、观测只读缓存**：板子上 history[-1] 就是同一帧 policy，逐位相等
（见 `velocity_env_cfg` 模块 docstring）。policy 和 history 两个组各调一次观测函数，
如果观测函数里采噪声，两组就会拿到不同的随机数。现在每步只在事件里采一次，
两组读到的是同一个张量。

时序：interval 事件在物理步进之后、观测计算之前触发。这一步算出的估计力对应的是
刚刚仿真完那一步实际受的力，然后再更新下一步要施加的力。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

try:  # IsaacLab 2.x 新名字；老版本只有 quat_rotate_inverse
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # pragma: no cover
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from .. import spec

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_ATTR = "_bpx_force_state"

#: 误差模型默认值（接口约定第 5 节）。改这里等于改 teacher 看到的估计力分布。
ERROR_MODEL_DEFAULTS: dict = {
    "mag_noise_abs": 2.0,              # 大小噪声 σ = 2 N + 5 %·|F|
    "mag_noise_rel": 0.05,
    "dir_noise_deg": 5.0,              # 方向噪声 σ = 5°，|F| < 5 N 时 15°
    "dir_noise_deg_low": 15.0,
    "low_force_threshold": 5.0,
    "bias_xyz": 3.0,                   # 每回合偏置：每轴 ±3 N，z 再加 0–5 N
    "bias_z_extra": (0.0, 5.0),
    "delay_steps": (1, 5),             # 控制步（50 Hz 下 20–100 ms），每回合抽一次
    "clip": spec.FORCE_OBS_SCALE,      # 每轴限幅 ±45 N
    "mask_episode_fraction": 0.10,     # 10 % 的回合有一段遮蔽（conf 0、力 0）
    "mask_duration_s": (0.2, 1.0),
    "conf_err_scale": 15.0,            # conf = 1 − clip(|err| / 15 N, 0, 1)
    "warmup_steps": spec.HISTORY_LENGTH,   # 复位后估计器历史没填满：conf 0、力 0
}

#: 施力剖面默认值。
PROFILE_DEFAULTS: dict = {
    "force_range": (0.0, 45.0),        # |F| [N]
    "segment_s": (1.0, 3.0),
    "ramp_s": (0.0, 1.0),
    "sudden_prob": 0.3,                # 新目标直接跳变（不走斜坡）的概率
    # 每段的模式概率：置零 / 新目标 / 释放（斜坡回 0）/ 反向
    "mode_probs": (0.2, 0.4, 0.2, 0.2),
    "attach_range": ((-0.10, 0.10), (-0.06, 0.06), (0.0, 0.08)),   # 挂点 [m]，机身系
}


class BpxForceState:
    """逐 env 的牵引力和估计力缓存。"""

    def __init__(self, env: ManagerBasedEnv):
        n, dev = env.num_envs, env.device
        max_delay = ERROR_MODEL_DEFAULTS["delay_steps"][1]

        def z(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, device=dev)

        # --- 真实施力 ---
        self.force_w = z(n, 3)           # 当前施加的力（世界系）[N]
        self.attach_b = z(n, 3)          # 挂点（机身系）[m]
        self.seg_left = z(n)             # 当前段剩余时间 [s]
        self.ramp_left = z(n)            # 斜坡剩余时间 [s]
        self.ramp_total = z(n)           # 斜坡总时长 [s]
        self.ramp_from_w = z(n, 3)
        self.target_w = z(n, 3)

        # --- 误差模型的回合级参数 ---
        self.delay = torch.ones(n, dtype=torch.long, device=dev)
        self.bias_b = z(n, 3)
        self.mask_begin = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.mask_end = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.step_count = torch.zeros(n, dtype=torch.long, device=dev)
        #: 机身系真值的环形缓存，[:, 0] 最新。长度 max_delay + 1。
        self.true_hist_b = z(n, max_delay + 1, 3)

        # --- 观测缓存（每步只写一次，policy / history 两组共享）---
        self.true_force_b = z(n, 3)      # 刚仿真完那一步的真值（机身系）[N]
        self.est_force_b = z(n, 3)       # 估计力（机身系）[N]，无效时 0
        self.confidence = z(n, 1)        # [0, 1]


def get_force_state(env: ManagerBasedEnv) -> BpxForceState:
    st = getattr(env, _ATTR, None)
    if st is None:
        st = BpxForceState(env)
        setattr(env, _ATTR, st)
    return st


def _uniform(lo: float, hi: float, shape, device) -> torch.Tensor:
    return torch.rand(shape, device=device) * (hi - lo) + lo


def _random_unit(n: int, device) -> torch.Tensor:
    v = torch.randn(n, 3, device=device)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _write_wrench(env, asset: Articulation, body_ids, st: BpxForceState, env_ids: torch.Tensor) -> None:
    """把 force_w 换到机身系、算 p × F，写进 PhysX（下一次物理步进生效）。"""
    quat = asset.data.root_quat_w[env_ids]
    f_b = quat_apply_inverse(quat, st.force_w[env_ids])
    tau_b = torch.cross(st.attach_b[env_ids], f_b, dim=-1)
    asset.set_external_force_and_torque(
        f_b.unsqueeze(1), tau_b.unsqueeze(1), body_ids=body_ids, env_ids=env_ids
    )


def tether_force_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[spec.TORSO_BODY]),
) -> None:
    """reset 事件：清零外力，重抽这一回合的挂点 / 延迟 / 偏置 / 遮蔽段。"""
    st = get_force_state(env)
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, dev = len(env_ids), env.device
    em, pr = ERROR_MODEL_DEFAULTS, PROFILE_DEFAULTS

    for i in ("force_w", "ramp_from_w", "target_w", "true_force_b", "est_force_b", "confidence"):
        getattr(st, i)[env_ids] = 0.0
    st.true_hist_b[env_ids] = 0.0
    st.seg_left[env_ids] = 0.0           # 下一步立刻开新段
    st.ramp_left[env_ids] = 0.0
    st.ramp_total[env_ids] = 0.0
    st.step_count[env_ids] = 0

    st.attach_b[env_ids] = torch.stack(
        [_uniform(lo, hi, (n,), dev) for lo, hi in pr["attach_range"]], dim=-1
    )
    lo, hi = em["delay_steps"]
    st.delay[env_ids] = torch.randint(lo, hi + 1, (n,), device=dev)
    b = _uniform(-em["bias_xyz"], em["bias_xyz"], (n, 3), dev)
    b[:, 2] += _uniform(*em["bias_z_extra"], (n,), dev)
    st.bias_b[env_ids] = b

    # 遮蔽段：落在预热之后、回合结束之前
    has_mask = torch.rand(n, device=dev) < em["mask_episode_fraction"]
    max_len = int(env.max_episode_length)
    dur = (_uniform(*em["mask_duration_s"], (n,), dev) / env.step_dt).long().clamp_min(1)
    begin = torch.randint(em["warmup_steps"], max(em["warmup_steps"] + 1, max_len), (n,), device=dev)
    st.mask_begin[env_ids] = torch.where(has_mask, begin, torch.full_like(begin, -1))
    st.mask_end[env_ids] = torch.where(has_mask, begin + dur, torch.full_like(begin, -1))

    _write_wrench(env, asset, asset_cfg.body_ids, st, env_ids)


def _estimate(st: BpxForceState, env_ids: torch.Tensor, device) -> None:
    """误差模型：延迟 → 大小 / 方向噪声 → 偏置 → 限幅 → 置信度 → 预热 / 遮蔽 / 无效置 0。"""
    em = ERROR_MODEL_DEFAULTS
    n = len(env_ids)
    rows = torch.arange(n, device=device)
    f_true = st.true_force_b[env_ids]
    f_del = st.true_hist_b[env_ids][rows, st.delay[env_ids]]          # 延迟后的真值

    mag = f_del.norm(dim=-1)
    unit = torch.where(
        (mag > 1e-6).unsqueeze(-1), f_del / mag.clamp_min(1e-9).unsqueeze(-1), _random_unit(n, device)
    )
    sigma_m = em["mag_noise_abs"] + em["mag_noise_rel"] * mag
    mag_n = (mag + sigma_m * torch.randn(n, device=device)).clamp_min(0.0)

    # 方向：绕一个和 unit 垂直的随机轴转 N(0, σ) 角
    sigma_d = torch.where(
        mag < em["low_force_threshold"],
        torch.full_like(mag, math.radians(em["dir_noise_deg_low"])),
        torch.full_like(mag, math.radians(em["dir_noise_deg"])),
    )
    perp = torch.randn(n, 3, device=device)
    perp = perp - (perp * unit).sum(-1, keepdim=True) * unit
    perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    ang = (sigma_d * torch.randn(n, device=device)).unsqueeze(-1)
    unit_n = torch.cos(ang) * unit + torch.sin(ang) * perp

    est = mag_n.unsqueeze(-1) * unit_n + st.bias_b[env_ids]
    est = est.clamp(-em["clip"], em["clip"])

    conf = 1.0 - ((est - f_true).norm(dim=-1) / em["conf_err_scale"]).clamp(0.0, 1.0)

    k = st.step_count[env_ids]
    invalid = (
        (k < em["warmup_steps"])
        | ((k >= st.mask_begin[env_ids]) & (k < st.mask_end[env_ids]))
        | ~torch.isfinite(est).all(dim=-1)
    )
    est = torch.where(invalid.unsqueeze(-1), torch.zeros_like(est), est)
    conf = torch.where(invalid, torch.zeros_like(conf), conf)

    st.est_force_b[env_ids] = est
    st.confidence[env_ids] = conf.unsqueeze(-1)


def _advance_profile(env, st: BpxForceState, env_ids: torch.Tensor) -> None:
    """推进施力剖面一个控制步：到段尾就开新段，斜坡中就线性插值。"""
    pr, dt, dev = PROFILE_DEFAULTS, env.step_dt, env.device

    new_seg = env_ids[st.seg_left[env_ids] <= 0.0]
    if len(new_seg):
        n = len(new_seg)
        cur = st.force_w[new_seg]
        mode = torch.multinomial(torch.tensor(pr["mode_probs"], device=dev), n, replacement=True)
        fresh = _random_unit(n, dev) * _uniform(*pr["force_range"], (n, 1), dev)
        target = torch.zeros_like(cur)                         # 0：置零（突变到 0）
        target = torch.where((mode == 1).unsqueeze(-1), fresh, target)   # 1：新目标
        target = torch.where((mode == 2).unsqueeze(-1), torch.zeros_like(cur), target)  # 2：释放
        target = torch.where((mode == 3).unsqueeze(-1), -cur, target)    # 3：反向
        ramp = _uniform(*pr["ramp_s"], (n,), dev)
        sudden = (mode == 0) | ((mode == 1) & (torch.rand(n, device=dev) < pr["sudden_prob"]))
        ramp = torch.where(sudden, torch.zeros_like(ramp), ramp)
        st.ramp_from_w[new_seg] = cur
        st.target_w[new_seg] = target
        st.ramp_left[new_seg] = ramp
        st.ramp_total[new_seg] = ramp
        st.seg_left[new_seg] = _uniform(*pr["segment_s"], (n,), dev) + ramp
        # 第一步就要动：斜坡长度为 0 的直接到位
        st.force_w[new_seg] = torch.where((ramp <= 0.0).unsqueeze(-1), target, cur)

    ramping = env_ids[st.ramp_left[env_ids] > 0.0]
    if len(ramping):
        left = (st.ramp_left[ramping] - dt).clamp_min(0.0)
        frac = (1.0 - left / st.ramp_total[ramping].clamp_min(1e-9)).unsqueeze(-1)
        st.force_w[ramping] = st.ramp_from_w[ramping] + frac * (st.target_w[ramping] - st.ramp_from_w[ramping])
        st.ramp_left[ramping] = left

    st.seg_left[env_ids] -= dt


def tether_force_step(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[spec.TORSO_BODY]),
) -> None:
    """interval 事件（每控制步）：记录刚受的力 → 算估计力 / 置信度 → 推进并写入下一步的力。"""
    st = get_force_state(env)
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # 1. 刚仿真完那一步实际受的力（机身系，用当前姿态换算）
    f_b = quat_apply_inverse(asset.data.root_quat_w[env_ids], st.force_w[env_ids])
    st.true_force_b[env_ids] = f_b
    hist = st.true_hist_b[env_ids]
    st.true_hist_b[env_ids] = torch.cat([f_b.unsqueeze(1), hist[:, :-1]], dim=1)

    # 2. 估计器误差模型（每步只采一次随机数）
    _estimate(st, env_ids, env.device)
    st.step_count[env_ids] += 1

    # 3. 下一步的力
    _advance_profile(env, st, env_ids)
    _write_wrench(env, asset, asset_cfg.body_ids, st, env_ids)


# ---------------------------------------------------------------------------
# 观测（只读缓存，幂等）
# ---------------------------------------------------------------------------
def estimated_force_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """估计力，机身系，clip(F_b / 45, ±1)，(N, 3)。"""
    return (get_force_state(env).est_force_b / spec.FORCE_OBS_SCALE).clamp(-1.0, 1.0)


def force_confidence_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """估计力置信度 [0, 1]，(N, 1)。"""
    return get_force_state(env).confidence.clamp(0.0, 1.0)


def true_force_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """特权：真实外力，机身系，/45（不限幅，让 teacher 看到超范围），(N, 3)。"""
    return get_force_state(env).true_force_b / spec.FORCE_OBS_SCALE


def force_attach_point(env: ManagerBasedEnv) -> torch.Tensor:
    """特权：这一回合的挂点 [m]，机身系，(N, 3)。"""
    return get_force_state(env).attach_b
