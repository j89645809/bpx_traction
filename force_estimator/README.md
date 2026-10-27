# BPX Force Estimator

这是本项目的力估计器上游工程，供 `bpx_rma` 的 ForceAware policy 使用。它把历史本体感知输入转换为外力估计和置信度，再交给 BPX 策略；它和 BPX 策略是两个明确的模块，不把 Go2 仿真代码伪装成 BPX 运行时。

## 项目关系

```text
25 帧本体感知历史 [B, 25, 64]
        │
        ▼
MissingStateTCN
        │ values [B, 15] + confidence [B, 4]
        │ values[:, 12:15] = world-frame force (N)
        ▼
world → body 坐标变换（由上层适配器完成）
        │
        ▼
force_body / 45，clip 到 [-1, 1] + force confidence
        │
        ▼
BPX ForceAware RMA：policy 49 维，history 1225 维
```

另有一个保留原始训练结构的 `ForceEstimator`，输入 `[B, 25, input_dim]`、输出 `[B, 3]`，用于旧版 GuideDog/82 维估计器路线。正式 BPX handoff 使用 `MissingStateTCN` 的 world-force 输出和 `config/world_force.yaml` 中的契约。

## 目录

```text
force_estimator/
├── config/
│   ├── normalization.npz       # v2 的 64 通道 mean/std
│   └── world_force.yaml        # 坐标系、时序、输出和安全阈值
├── provenance/SOURCES.md       # 每个文件的远端来源
├── src/force_estimator/
│   ├── contracts.py            # 维度和输出切片
│   ├── preprocessing.py        # shape 检查、归一化、策略力输入
│   ├── model.py                # 旧版 temporal ForceEstimator
│   └── world_model.py          # causal MissingStateTCN
└── tests/
```

历史 Go2/mjlab 事件、Gate41 配置、runner、协议和报告保留在 [`../references/archived_force_estimator`](../references/archived_force_estimator)，用于溯源和核对，不是这个包的运行依赖。

## 安装和验证

在有 PyTorch 的 Python 3.10+ 环境中运行：

```bash
cd force_estimator
python -m pip install -e '.[test]'
pytest -q
```

没有 PyTorch 时，契约和归一化测试仍可运行；模型测试会自动跳过。最小的纯 NumPy 检查是：

```bash
PYTHONPATH=src pytest -q tests/test_contracts.py
```

模型的输入输出契约：

| 模块 | 输入 | 输出 |
|---|---|---|
| `ForceEstimator` | `[B, 25, input_dim]` | force `[B, 3]` |
| `MissingStateTCN` | `[B, T, 64]` | values `[B, 15]`, confidence `[B, 4]` |
| BPX handoff | body force in N `[B, 3]` | `clip(force / 45, -1, 1)` + confidence |

`MissingStateTCN` 的 `values` 顺序是 velocity(3)、height(1)、contact(4)、load(4)、force(3)，最后三维是世界坐标系外力；置信度第 3 列（零基索引）对应 force confidence。因果 TCN 只使用当前及过去时间步。

## 重要边界

- `normalization.npz` 只适用于 64 通道输入；代码会检查 mean/std 的形状和正值。
- 世界坐标到机身坐标的变换必须由适配器完成，并在日志中明确记录 frame，不能把 `force` 作为含义不明的字段。
- 45 N 是 BPX policy 的归一化尺度，80 N 是 world model 输出的仿真范围；两者不是同一个阈值。
- 仓库不包含 `.pt`、`.onnx`、数据集、机器人资产或服务器凭据。
