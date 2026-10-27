# BPX 45D 重训练结果（2026-10-27）

本目录保存按同事 `bpx_traction` 的 45 维 policy 观测、25 帧历史重新训练的力估计器。教师策略在 zy-server GPU 5/7 完成 1200 轮训练，末段平均回报约 39.6–40.2；数据采集和估计器训练随后使用 GPU 1。

## 模型契约

- 输入：`[B,25,45]`，连续 25 帧 BPX policy 观测。
- 输出：`force_world [B,3]`（N）和 `force_confidence [B,1]`。
- 部署适配：见 `force_estimator/src/force_estimator/bpx_adapter.py`。

## 独立测试结果

测试集 28,877 个样本，训练/验证/测试严格分开：

| 指标 | 结果 |
|---|---:|
| 合力大小 MAE | 8.14 N |
| 合力大小 RMSE | 10.23 N |
| 方向平均误差（真值大于 2 N） | 65.45° |
| 方向 P95 | 145.12° |
| 分量 MAE（Fx/Fy/Fz） | 8.01 / 7.83 / 4.94 N |

当前结果**未通过精度验收，不建议部署**。训练损失已下降并在后期趋于平台，但泛化误差仍大，下一轮应先审计 force 标签与 45D 历史的时间对齐、施力覆盖和 BPX 观测是否包含足够的可辨识信息。

文件：`bpx_force.onnx`、`bpx_force.ts`、`bpx_force_best.pt`、`normalization.npz`、`training_report.json`、`evaluation_torch.json`。
