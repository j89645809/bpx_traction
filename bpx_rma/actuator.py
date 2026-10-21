"""带编码器零偏和力矩延迟的 DC 电机执行器。

从 pace-sim2real 的 `pace_sim2real/utils/pace_actuator.py` 原样搬过来（Apache-2.0，
© 2025 ETH Zurich, Robotic Systems Lab, author Filip Bjelonic），这样 bpx 这个项目
不依赖 pace-sim2real 的安装。辨识用的就是这个执行器模型，训练必须用同一个，否则
辨识出来的 armature / 粘滞 / 摩擦 / 延迟 对不上。

模型：
  1. PD 在**编码器坐标**上算：tau = kp·(q_des - (q - bias)) - kd·dq
  2. DCMotor 的力矩-速度饱和（堵转力矩 saturation_effort，上限 effort_limit）
  3. 算出来的力矩延迟 `time_lag` 个仿真步才施加

关节级的 armature / 粘滞摩擦 / 库伦摩擦不在这里，它们是 PhysX 的关节属性，
由 :mod:`bpx_rma.mdp.events` 写进去（和 PACE 的 `CMAESOptimizer.update_simulator` 一致）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    pass


class PaceDCMotor(DCMotor):
    """DCMotor + 编码器零偏 + 力矩延迟。"""

    cfg: PaceDCMotorCfg

    def __init__(self, cfg: PaceDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)) and len(cfg.encoder_bias) != self.num_joints:
            raise ValueError(
                f"encoder_bias 要么是标量/字典，要么长度 = {self.num_joints}，给的是 {len(cfg.encoder_bias)}"
            )
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)

        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

        # --- 关节速度一阶低通（模拟真机板载 dq 滤波的滞后）-------------------
        # 真机 SDK 返回的 dq 噪声底 3e-05 rad/s，必然经过重滤波；滤波的代价是相位滞后。
        # 真机 500 Hz PD 的阻尼项和策略观测用的都是这个滤波值，所以仿真里也在
        # 执行器内（每物理步）滤：PD 用滤波值，观测端通过 `filtered_vel` 读同一份。
        # alpha = 1 表示不滤（v_f = v），是默认值 —— 不开启时行为与旧版逐位相同。
        self.vel_filter_alpha = torch.ones(self._num_envs, 1, device=self._device)
        self.filtered_vel = torch.zeros(self._num_envs, self.num_joints, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self.torques_delay_buffer.reset(env_ids)
        self.filtered_vel[env_ids] = 0.0

    def update_vel_filter(self, tau_s: torch.Tensor, env_ids: Sequence[int], dt: float):
        """逐 env 设置滤波时间常数（秒）。tau <= 0 表示不滤。"""
        alpha = torch.where(tau_s > 0, 1.0 - torch.exp(-dt / tau_s.clamp(min=1e-5)), torch.ones_like(tau_s))
        self.vel_filter_alpha[env_ids] = alpha.unsqueeze(-1)

    def update_encoder_bias(self, encoder_bias: torch.Tensor, env_ids: Sequence[int] | None = None):
        """逐 env 改编码器零偏。``env_ids`` 为 None 时整体替换。"""
        if env_ids is None:
            self.encoder_bias = encoder_bias
        else:
            self.encoder_bias[env_ids] = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        """逐 env 改力矩延迟（单位：仿真步）。"""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # 速度先过低通（alpha=1 时 filtered_vel ≡ joint_vel，零开销等价旧行为）
        self.filtered_vel += self.vel_filter_alpha * (joint_vel - self.filtered_vel)
        # PD 看到的是编码器读数（真值减零偏）和滤波后的速度 —— 与真机板载一致
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, self.filtered_vel)
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(control_action_sim.joint_efforts)
        return control_action_sim


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """:class:`PaceDCMotor` 的配置。"""

    class_type: type = PaceDCMotor

    encoder_bias: dict[str, float] | float | None = 0.0
    """编码器零偏 [rad]。控制器看到的是 ``q - encoder_bias``。"""

    max_delay: int = 0
    """力矩延迟缓冲的最大长度（仿真步）。必须 >= 随机化的延迟上界。"""
