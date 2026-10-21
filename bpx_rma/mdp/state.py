"""逐 env 的域随机化状态。

事件（`events.py`）往这里写，特权观测（`observations.py`）从这里读。挂在 env 上，
第一次访问时惰性创建。所有 12 维张量按 :data:`bpx_rma.spec.JOINT_NAMES`（SDK 顺序）排列，
4 维的按 :data:`bpx_rma.spec.KNEE_JOINT_NAMES`（fl, fr, hl, hr）排列。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import spec
from ..params import TrainingParams, get_params

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_ATTR = "_bpx_domain_state"


class BpxDomainState:
    """这一回合每个 env 实际用的动力学参数。"""

    def __init__(self, env: ManagerBasedEnv, params: TrainingParams):
        n, j, k = env.num_envs, spec.NUM_JOINTS, len(spec.KNEE_JOINT_NAMES)
        dev = env.device
        self.params = params

        def z(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, device=dev)

        # 关节级参数（已写进 PhysX 的值）
        self.armature = z(n, j)
        self.viscous = z(n, j)
        self.friction = z(n, j)          # 库伦摩擦静=动，膝关节在负载模型下逐步变
        self.encoder_bias = z(n, j)
        self.delay_steps = z(n)

        # 整机质量
        self.total_mass = z(n)
        self.mass_offset = z(n)          # 相对 USD 原始质量的偏移

        # 膝承重摩擦的负载模型
        self.knee_load = z(n, k)         # L 的低通估计
        self.knee_f0 = z(n, k)
        self.knee_mu = z(n, k)

        # 便于观测归一化：标称值（1 维，broadcast）
        self.armature_nominal = torch.tensor(params.armature.nominal, device=dev)
        self.viscous_nominal = torch.tensor(params.damping.nominal, device=dev)
        self.friction_nominal = torch.tensor(params.frictionloss.nominal, device=dev)

        # 关节索引映射：articulation 的关节顺序 -> SDK 顺序
        self.joint_ids_sdk: torch.Tensor | None = None
        self.knee_ids_sdk: torch.Tensor | None = None
        # (执行器名, 该执行器各关节在 SDK 12 维里的列) —— joint_vel_filtered 观测用
        self.actuator_sdk_cols: list[tuple[str, torch.Tensor]] | None = None

    def resolve_joint_ids(self, asset) -> None:
        """第一次用的时候解析 SDK 顺序对应的 articulation 关节下标。"""
        if self.joint_ids_sdk is None:
            ids, _ = asset.find_joints(spec.JOINT_NAMES, preserve_order=True)
            self.joint_ids_sdk = torch.tensor(ids, device=self.armature.device, dtype=torch.long)
            kids, _ = asset.find_joints(spec.KNEE_JOINT_NAMES, preserve_order=True)
            self.knee_ids_sdk = torch.tensor(kids, device=self.armature.device, dtype=torch.long)
            sdk_index = {n: i for i, n in enumerate(spec.JOINT_NAMES)}
            self.actuator_sdk_cols = [
                (name, torch.tensor([sdk_index[n] for n in act.joint_names],
                                    device=self.armature.device, dtype=torch.long))
                for name, act in asset.actuators.items()
            ]


def get_state(env: ManagerBasedEnv) -> BpxDomainState:
    """取（必要时创建）这个 env 的域随机化状态。"""
    st = getattr(env, _ATTR, None)
    if st is None:
        st = BpxDomainState(env, get_params())
        setattr(env, _ATTR, st)
    return st
