# Force estimator training and validation source

这部分是从 `zy-server` 的旧代码中恢复的训练链路，补齐“模型代码 + 数据处理 + 训练 + 导出 + 验收”的可审查源码。它对应 `releases/acceptance_v2_selected/` 中的最佳候选：第 6320 轮 checkpoint，10 个全新种子闭环复验。

## 目录

```text
training/
├── source/
│   ├── train_convergence_v1.py          # 第 6320 轮候选的收敛训练入口
│   ├── evaluate_convergence_v1.py       # 起点/候选盲测与 Torch/ONNX 一致性
│   ├── run_acceptance_v2.py              # 30101–30110 的冻结成对采集
│   ├── verify_acceptance_v2.py           # 20 条完整轨迹的验收复核
│   ├── scripts/
│   │   ├── train_estimator.py            # 从 train/validation.npz 训练 MissingStateTCN
│   │   ├── collect_sim_estimator_dataset.py
│   │   ├── evaluate_estimator.py
│   │   ├── verify_estimator_export.py
│   │   └── ...                           # 数据、导出、场景和报告工具
│   ├── world_force/                      # 模型、坐标系和安全监督器
│   ├── contracts/                        # 输入、坐标和观测契约
│   ├── config/                           # 方向矩阵、场景和 world-force 配置
│   └── tests/                            # 契约、坐标、安全测试
└── legacy_mjlab/src/tasks/velocity/      # 采集数据所需的 Go2/mjlab 任务源码
```

## 训练入口

通用的 `MissingStateTCN` 训练入口需要一个数据目录：

```text
DATASET/
├── train.npz
├── validation.npz
└── test.npz / ood.npz
```

每个 NPZ 至少包含：

- 旧版 Go2 数据：`sensor_history`: `[N, 25, 64]`
- `privileged_targets`: `[N, 15]`

BPX 数据格式：`sensor_history`: `[N,25,45]`，`force_world`: `[N,3]`；由 `collect_bpx_force_dataset.py` 生成。

运行：

```bash
cd force_estimator/training/source
PYTHONPATH=. python scripts/train_estimator.py \
  --dataset /path/to/DATASET \
  --output /path/to/output \
  --epochs 10

PYTHONPATH=. python scripts/evaluate_estimator.py \
  --dataset /path/to/DATASET \
  --model /path/to/output/estimator.onnx \
  --output /path/to/output/evaluation.json
```

`train_estimator.py` 会保存 checkpoint、TorchScript、ONNX、归一化统计量和训练报告。`evaluate_estimator.py` 会报告外力 RMSE、方向误差、置信度和推理延迟。

## 最佳版本复现链路

最佳候选不是普通的单次训练，而是以下固定链路：

1. `train_convergence_v1.py` 从 `causal_v3_extended/aligned_window_720.pt` 继续训练，固定随机种子、数据 hash、FP32 和 AdamW 阶段性学习率。
2. `evaluate_convergence_v1.py` 在新种子上比较起点与候选，检查物理审计、在线预测、Torch/ONNX 一致性和恢复时间。
3. `run_acceptance_v2.py` 固定 10 个新种子、GPU 分配、模型顺序和施力计划，采集 20 条完整成对轨迹。
4. `verify_acceptance_v2.py` 重新计算统计量、同时区间、删失恢复时间和 hash，不能用结果反向放宽门槛。

这些脚本依赖 IsaacLab/mjlab、PyTorch、NumPy、SciPy、ONNX Runtime 和 Go2 仿真资产；本仓库只上传源码和结果，不上传数据集、机器人资产、日志缓存或服务器路径下的运行产物。

## 来源与边界

源码恢复自 `zy@111.0.18.168`：

```text
/home/zy/old_data/lead_clean_20260916/source
```

`legacy_mjlab/` 只保留采集和任务所需的 Python 源码，已去掉旧备份、缓存、权重、数据和资产。要在服务器复现，需要根据本机环境把 `training/source` 和 `training/legacy_mjlab` 加入 `PYTHONPATH`，并提供对应的仿真依赖与数据。

结果和冻结模型见 [`../releases/acceptance_v2_selected/`](../releases/acceptance_v2_selected/)。


## BPX 45-D 训练

`source/collect_bpx_force_dataset.py` 在 ForceAware BPX 环境中采集 `[N,25,45]` policy 历史和 world-frame force 标签；`source/train_bpx_force_estimator.py` 训练并导出 `bpx_force.onnx`、`bpx_force.ts` 和归一化统计量。

```bash
conda activate bpx_rma
python source/collect_bpx_force_dataset.py --checkpoint /path/to/force_aware_teacher.pt --output /path/to/bpx_dataset --headless --device cuda:0
PYTHONPATH=../../src python source/train_bpx_force_estimator.py --dataset /path/to/bpx_dataset --output /path/to/bpx_force_release
```

## 服务器 GPU 调度约定

后续训练、数据采集和验收默认优先使用 GPU 1、2、3；启动前检查占用情况，按 `CUDA_VISIBLE_DEVICES=1`（或 2、3）绑定空闲卡。GPU 5、7 仅作为当前这次已启动任务的临时卡位，不作为后续默认选择。
