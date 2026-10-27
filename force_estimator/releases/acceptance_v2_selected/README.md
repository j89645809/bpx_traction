# Acceptance v2 selected release

这是当前本地验收记录中精度最好的冻结候选版本，来源于 `/Users/zhaoyi/Desktop/力估计/力估计器完整打包_20261021_v2.tar.gz`，生成日期为 2026-10-21（Asia/Shanghai）。

## 版本身份

- 选择依据：第 6320 轮权重，训练在第 6410 轮停止
- PyTorch checkpoint SHA256：`92b52bf4fe68a4f08ca258c518aa00f5e5232d85621b1ba13cf50d381de55639`
- 输入：单帧 64 维，最近 25 帧，20 ms 采样周期
- 输出：三维外力估计；另有 TorchScript/ONNX 部署文件
- 评估：10 个全新种子（30101–30110），20 条完整 MuJoCo 闭环轨迹，5 个场景，0–45 N 分层覆盖，共 12,880,000 个有效帧

## 最好版本的结果

| 指标 | 10 种子汇总 |
|---|---:|
| 合力大小 MAE | **0.445 N** |
| 合力大小 RMSE | **1.090 N** |
| 方向平均误差 | **2.886°** |
| 方向 P95 | **7.426°** |
| Fz RMSE | **1.283 N** |
| 原项目门槛逐种子通过 | **9/10** |

95% 近似 Student-t 区间见 `reports/力估计器新版重复验收报告.md` 和 `reports/acceptance_v2_summary.json`。

## 文件说明

- `models/convergence_v1_best_epoch6320.pt`：最佳冻结 PyTorch checkpoint。
- `models/acceptance_v2_selected.onnx`：82 维历史精度候选，不能直接接入 BPX。
- `models/missing_state_world_v2_estimator.ts`：TorchScript 模型。
- `models/missing_state_world_v2_estimator.onnx` + `.onnx.data`：64 维历史 ONNX 候选，仅作溯源。
- `config/missing_state_world_v2_normalization.npz`：64 通道 mean/std。
- `reports/`：验收汇总、完整报告、统计 JSON 和 Fz 诊断。
- `experiments/`：验收协议和远程复现实验协议。
- `SHA256SUMS.txt`：当前 GitHub 目录内文件的校验值。
- `ORIGINAL_SHA256SUMS.txt`：原始 2026-10-21 压缩包的校验值，用于溯源。

训练、数据采集、导出和验收源码不在这个发布目录内，位于同一分支的 [`../../training/`](../../training/)。因此本目录是“冻结模型与结果发布包”，要重跑训练应从 `training/` 进入；要做部署推理则使用 `models/` 中的 TorchScript 或 ONNX 文件。

## BPX 直接接入

BPX 使用同事仓库定义的 45 维 policy 历史（25 帧）。正式训练产物应为 `bpx_force.onnx`，由上层适配器完成世界系到机身系旋转、`/45 N` 归一化、`[-1, 1]` 裁剪和 confidence 提取。仓库中的 `force_estimator/src/force_estimator/bpx_adapter.py` 和 `force_estimator/config/bpx_deploy.yaml` 是正式入口；本目录内的 64 维模型仅作历史溯源，上方的 1.090 N 只属于 82 维 Go2/mjlab 基线。

## 结论和限制

这个版本是目前记录中离线/闭环精度最好的候选，但不能写成“正式合格”或“已通过实机认证”：

- Full3D 的 0–15 N 和 15–45 N 主分组均未通过同时区间判定。
- 高力 Fz、Full3D、低力方向尾部和 Reverse 瞬态仍有风险。
- Release/Reverse 的切换瞬态误差明显高于稳定段。
- 原始报告明确写着默认部署模型没有被替换。

因此，本目录用于同事审查、复现实验和选择后续训练起点；它不是实机部署授权。
