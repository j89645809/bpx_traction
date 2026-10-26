# 力感知 RMA 上下游接口约定（v0.2）

上游 = 力估计器（输出机身系外力和置信度）；下游 = BPX 力感知 RMA 策略。
本文件是两边对接的唯一依据，代码里的常数都以这里为准（`bpx_rma/spec.py` 的 ForceAware 段）。

原 45 / 1125 契约和所有已有任务**一律不动**；力感知是新增的变体任务：

| 任务 ID | 用途 |
|---|---|
| `Isaac-Velocity-Flat-BpxPro-ForceAware-v0` | 训练（teacher PPO / student 蒸馏） |
| `Isaac-Velocity-Flat-BpxPro-ForceAware-Play-v0` | 回放 / 导出 / 契约检查 |

## 1. 上游：力估计器输出

| 项 | 约定 |
|---|---|
| 模型 | TorchScript / ONNX，冻结，不和策略一起更新 |
| 输入历史 | 25 × 64（估计器自己的契约，与策略无关） |
| 力输出 | `F_w` [N]，世界系 |
| 置信度输出 | `confidence` [B, 4]，**第 3 列 = 力置信度** |
| 频率 | 50 Hz，与策略同拍 |

## 2. 适配层（估计器 → 策略）

```
F_b   = R_wb^T · F_w                           # 世界系 → 机身系
force = clip(F_b / 45, -1, 1)                  # 45 N = 估计器验收上限
conf  = clip(confidence[:, 3], 0, 1)
```

硬规则（任一成立 → `conf = 0` 且 `force = 0`）：

* 输入无效：估计器输出含 NaN / Inf，或输入传感器字段缺失
* 过期：估计器结果的时间戳落后当前控制步（stale）
* 预热：复位 / 估计器重启后历史没填满 25 帧

不能把牛顿单位的原始力直接喂网络。

## 3. 下游：策略观测契约

单帧 49 维，新两项**追加在原 45 维末尾**，前 45 维顺序和分段完全不变：

| 段 | 维度 | 切片 |
|---|---|---|
| q − q0 | 12 | 0:12 |
| dq | 12 | 12:24 |
| ω（机身系） | 3 | 24:27 |
| 投影重力 | 3 | 27:30 |
| last_action | 12 | 30:42 |
| velocity_command | 3 | 42:45 |
| **estimated_force**（机身系 /45，±1） | 3 | 45:48 |
| **force_confidence**（[0, 1]） | 1 | 48:49 |

* 历史：25 帧，最旧在前，**term-major** 展平 = 25 × 49 = **1225**
* ONNX：输入 `policy` [1, 49] + `adaptation_history` [1, 1225]，输出 `actions` [1, 12]
* 动作、PD、频率（50 Hz 策略 / 500 Hz 内环）与原契约相同

仿真里估计力和置信度每步**只采一次**、policy 与 history 两组共享缓存，
保证 `history[-1] == policy` 逐位相等（板子上就是这样）。

## 4. 仿真施力（ForceAware 任务）

* 世界系方向固定的绳拉力，作用在 torso，力矩 = p × F
* 挂点每回合随机（机身系）：x ±0.10 m，y ±0.06 m，z 0–0.08 m
* 分段剖面：每段 1–3 s，模式 置零 0.2 / 新目标 0.4 / 释放 0.2 / 反向 0.2；
  新目标 |F| ∈ 0–45 N、方向全 3D，30 % 突加、其余 0–1 s 斜坡
* 覆盖 Ramp / Release / Reverse / Full3D；高力 Fz、2–5 N 方向尾部的加权采样留到训练配置里调

## 5. teacher 看到的估计器误差模型（默认值）

| 项 | 默认 |
|---|---|
| 大小噪声 | σ = 2 N + 5 %·\|F\| |
| 方向噪声 | σ = 5°；\|F\| < 5 N 时 15° |
| 回合偏置 | 每轴 ±3 N，z 另加 0–5 N |
| 延迟 | 1–5 控制步（20–100 ms），每回合抽一次 |
| 限幅 | 每轴 ±45 N |
| 遮蔽 | 10 % 回合有一段 0.2–1 s：conf 0、force 0 |
| teacher 置信度 | conf = 1 − clip(\|F_est − F_true\| / 15 N, 0, 1) |

特权观测额外给 teacher：真实外力（机身系 /45，不限幅）和挂点。
student 只看 policy + history（即带误差的估计力和置信度）。

## 6. 训练与验收（后续步骤，本次未做）

1. 输入契约：`python scripts/check_contract.py --headless --task Isaac-Velocity-Flat-BpxPro-ForceAware-Play-v0`
2. Torch / ONNX 一致性（`export_onnx.py` 需改成按 ForceAware 维度导出）
3. 无外力回归：原 Flat 场景性能不下降
4. 全场景闭环：0–15 N、15–45 N、Fz ±、Full3D、Ramp、Release、Reverse、2–5 N 方向尾部
5. student 蒸馏加 `L_old` 锚点损失，锚点批占 40 %

## 7. 已定事项（v0.2）

* 置信度 = `clip(confidence[:, 3], 0, 1)`，加第 2 节硬规则
* 力 = `clip(R_wb^T F_w / 45, ±1)`
* 新项追加在原 45 维末尾，顺序不变；历史 25 × 49 = 1225，term-major
* 保留所有 45 维任务和 spec 不动，新增 ForceAware 变体
* 挂点范围 x ±0.10 / y ±0.06 / z 0–0.08 m
* teacher 误差模型取第 5 节默认值
