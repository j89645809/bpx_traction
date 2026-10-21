"""rsl_rl 配置：RMA 的两个阶段。

**阶段一（PPO）**：actor 是 :class:`~bpx_rma.models.RmaModel` —— 特权参数经编码器 μ
压成 ``latent_dim`` 维的 z，再和 45 维本体感知一起进 base 策略 π。critic 是普通 MLP，
特权信息直接全给它（critic 不部署，不需要瓶颈）。

**阶段二（蒸馏）**：teacher 从阶段一的 ``actor_state_dict`` 加载并冻住；
student 同样是 `RmaModel`，但编码器换成吃 25 帧历史的 adaptation module φ，
而且 **base 策略 π 是从 teacher 原样复制过来冻住的**（见 `RmaDistillation.load`）。
所以阶段二只训 φ —— 这就是 RMA 原文的做法，ẑ 和 z 在同一个空间里，可以直接比。

两阶段必须对齐（否则 teacher 权重加载形状不匹配）：
  * PPO 的 ``actor`` 观测集 == 蒸馏的 ``teacher`` 观测集
  * actor / teacher 的 ``hidden_dims``、``latent_dim``、``encoder_hidden_dims`` 全一致
  * student 的 ``hidden_dims`` / ``latent_dim`` 也要和 teacher 一致（base 要共享）
  * ``experiment_name`` 相同（蒸馏默认从同名实验目录里找最新 checkpoint）

``obs_normalization=True``：归一化统计量学在网络里，导出 ONNX 时一并烤进计算图，
所以板子上送的仍是原始物理量，部署代码一行都不用改。
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from .. import spec

EXPERIMENT_NAME = "bpx_pro_flat"

#: base 策略 π 的形状。teacher 和 student 共享它，两阶段必须一致。
BASE_HIDDEN_DIMS = [512, 256, 128]
#: 隐变量 z 的维度。RMA 原文用 8；太大就失去瓶颈意义，太小装不下 50 个参数。
LATENT_DIM = 8
#: μ：65 维特权参数 -> z
TEACHER_ENCODER_HIDDEN_DIMS = [128, 64]
#: φ：1125 维历史 -> ẑ，比 μ 宽，因为要从时序里反推
STUDENT_ENCODER_HIDDEN_DIMS = [512, 256, 128]

#: 和板子上 `ACTION_CLIP` 一致：raw action 裁到 ±4 再进环境
CLIP_ACTIONS = spec.ACTION_CLIP


@configclass
class RslRlRmaModelCfg(RslRlMLPModelCfg):
    """:class:`~bpx_rma.models.RmaModel` 的配置：在 MLP 之外多了编码器那几项。"""

    class_name: str = "bpx_rma.models:RmaModel"
    latent_dim: int = LATENT_DIM
    encoder_hidden_dims: list[int] = None  # type: ignore[assignment]
    encoder_activation: str = "elu"


@configclass
class BpxProFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """阶段一：特权 PPO，产出 teacher。"""

    num_steps_per_env = 24          # 50 Hz 下 0.48 s
    max_iterations = 3000
    save_interval = 100
    experiment_name = EXPERIMENT_NAME
    run_name = "teacher"
    clip_actions = CLIP_ACTIONS
    empirical_normalization = False  # 已废弃字段，归一化在 model 的 obs_normalization 上

    # 第一个组进 base 策略，其余进编码器 —— 所以 policy(45) -> π，privileged(65) -> μ
    obs_groups = {"actor": ["policy", "privileged"], "critic": ["policy", "privileged"]}

    actor = RslRlRmaModelCfg(
        hidden_dims=BASE_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        latent_dim=LATENT_DIM,
        encoder_hidden_dims=TEACHER_ENCODER_HIDDEN_DIMS,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    # critic 不部署，不需要瓶颈，特权信息直接全给
    critic = RslRlMLPModelCfg(
        hidden_dims=BASE_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BpxProFlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """阶段二：把 teacher 蒸馏进只看历史的 student（RMA）。"""

    num_steps_per_env = 48          # 蒸馏不需要 GAE，rollout 长一点更省
    max_iterations = 1500
    save_interval = 100
    experiment_name = EXPERIMENT_NAME
    run_name = "student"
    clip_actions = CLIP_ACTIONS
    empirical_normalization = False
    #: teacher 从哪个 run 加载。PPO 和蒸馏共用 experiment_name，靠 run_name 后缀区分，
    #: 不加这个限制会把上一次的 student run 当成 teacher 捡起来。
    load_run = ".*_teacher"

    obs_groups = {"student": ["policy", "history"], "teacher": ["policy", "privileged"]}

    #: base 策略（hidden_dims / latent_dim）必须和 teacher 完全一致 —— 它是从 teacher
    #: 复制过来冻住的，只有编码器 φ 不同
    student = RslRlRmaModelCfg(
        hidden_dims=BASE_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        latent_dim=LATENT_DIM,
        encoder_hidden_dims=STUDENT_ENCODER_HIDDEN_DIMS,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlRmaModelCfg(
        hidden_dims=BASE_HIDDEN_DIMS,
        activation="elu",
        obs_normalization=True,
        latent_dim=LATENT_DIM,
        encoder_hidden_dims=TEACHER_ENCODER_HIDDEN_DIMS,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        #: 自定义算法：加载 teacher 之后把 base 策略复制进 student 并冻住
        class_name="bpx_rma.models:RmaDistillation",
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )
