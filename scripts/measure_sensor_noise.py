#!/usr/bin/env python3
"""从真机记录里量出传感器噪声底，给训练的观测噪声定幅度。

训练里的观测噪声幅度不能拍脑袋 —— IsaacLab 四足的通用值（关节角 ±0.01 rad、
关节速度 ±1.5 rad/s、陀螺 ±0.2 rad/s）是别人家机器的。这里从板子记的 500 Hz
原始数据里量 BPX-Pro 自己的。

**怎么把噪声和真实运动分开**：机身和关节的真实动态在 20 Hz 以下，传感器噪声是宽带的。
用二阶差分 d2[n] = x[n] - 2x[n-1] + x[n-2]：白噪声下 var(d2) = 6σ²，而平滑信号
在 500 Hz 采样下对 d2 的贡献很小。所以 σ ≈ std(d2) / sqrt(6)。
同时报告静止窗口里的原始标准差作为上界对照 —— 它包含真实的微小晃动，
所以一定比噪声底大；两者差得太多说明窗口选得不够静。

只读 pace 目录下的 .bin，不修改任何东西。

    conda activate bpx_rma
    python scripts/measure_sensor_noise.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_REC = Path.home() / "intern" / "ZGYD" / "pace" / "rec_20260418_031853_joint500.bin"
LEGS = ("fl", "fr", "hl", "hr")
JOINT_SUFFIX = ("roll", "pitch", "knee")


def second_diff_sigma(x: np.ndarray) -> np.ndarray:
    """白噪声标准差估计：var(x[n] - 2x[n-1] + x[n-2]) = 6 sigma^2。x 形状 (T, C)。"""
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return d2.std(axis=0) / np.sqrt(6.0)


def proj_gravity(quat_xyzw: np.ndarray) -> np.ndarray:
    """(T,4) 的 (x,y,z,w) -> (T,3) 投影重力，和板子 `projected_gravity_xyzw()` 同公式。"""
    x, y, z, w = (quat_xyzw[:, i] for i in range(4))
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.stack([2 * (w * y - x * z), -2 * (w * x + y * z), -(1 - 2 * (x * x + y * y))], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="从真机记录量传感器噪声底")
    ap.add_argument("--rec", default=str(DEFAULT_REC), help="板子记的 *_joint500.bin")
    ap.add_argument("--quiet-dq", type=float, default=0.15,
                    help="判定静止的关节速度阈值 rad/s（12 个关节都要低于它）")
    ap.add_argument("--min-window", type=float, default=2.0, help="静止窗口最短时长 s")
    args = ap.parse_args()

    rec = Path(args.rec).expanduser()
    cols = json.loads(Path(str(rec)[:-4] + ".json").read_text())["columns"]
    data = np.fromfile(rec, "<f8").reshape(-1, len(cols))
    idx = {c: i for i, c in enumerate(cols)}
    t = data[:, idx["t"]]
    fs = 1.0 / np.median(np.diff(t))
    print(f"[rec] {rec.name}  {len(data)} 行  {t[-1] - t[0]:.1f} s  采样率 {fs:.0f} Hz")

    q = data[:, [idx[f"q_{l}_{s}"] for l in LEGS for s in JOINT_SUFFIX]]
    dq = data[:, [idx[f"dq_{l}_{s}"] for l in LEGS for s in JOINT_SUFFIX]]
    omega = data[:, [idx[f"omega_{a}"] for a in "xyz"]]
    quat = data[:, [idx[f"quat_{a}"] for a in ("x", "y", "z", "w")]]
    grav = proj_gravity(quat)
    mode = data[:, idx["mode"]]

    # --- 找静止窗口：策略模式(4) 或 关节模式(3)，且 12 个关节速度都很小 ---
    still = (np.abs(dq).max(axis=1) < args.quiet_dq) & np.isin(mode, (3, 4))
    # 取最长的一段连续静止
    edges = np.diff(np.concatenate([[0], still.view(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    if len(starts) == 0:
        raise SystemExit(f"找不到静止段（阈值 {args.quiet_dq} rad/s），放宽 --quiet-dq 再试")
    lens = (ends - starts) / fs
    k = int(np.argmax(lens))
    if lens[k] < args.min_window:
        raise SystemExit(f"最长静止段只有 {lens[k]:.2f} s，不足 {args.min_window} s")
    sl = slice(starts[k], ends[k])
    print(f"[静止段] {t[starts[k]]:.1f}–{t[ends[k] - 1]:.1f} s（{lens[k]:.1f} s，{ends[k] - starts[k]} 个样本），"
          f"模式 {int(mode[starts[k]])}")

    # 记录是 500 Hz 定时写的，机器人状态偶尔没更新就会写进重复帧（这里约 9.5%）。
    # 重复帧的二阶差分恒为 0，会把噪声估计压低，必须按 seq 去重。
    seq = data[:, idx["seq"]][sl]
    fresh = np.concatenate([[True], np.diff(seq) != 0])
    print(f"[去重] 唯一 seq {fresh.sum()} / {len(seq)}（重复帧 {1 - fresh.mean():.1%}），"
          f"真实状态率 ≈ {fresh.sum() / lens[k]:.0f} Hz")
    age = data[:, idx["state_age"]][sl]
    print(f"[时延] state_age 中位 {np.median(age) * 1e3:.2f} ms，"
          f"95% {np.quantile(age, 0.95) * 1e3:.2f} ms，最大 {age.max() * 1e3:.2f} ms")

    print(f"\n{'='*76}")
    print(f"{'通道':<22}{'噪声底 σ':>14}{'静止段原始 std':>16}{'训练用 ±幅度':>16}")
    print(f"{'(二阶差分估计)':<22}{'':>14}{'(含真实微动)':>16}{'(= √3·σ)':>16}")
    print("-" * 76)

    out: dict[str, float] = {}
    for name, arr, unit, grp in [
        ("关节角 q", q, "rad", "joint_pos"),
        ("关节速度 dq", dq, "rad/s", "joint_vel"),
        ("陀螺 omega", omega, "rad/s", "base_ang_vel"),
        ("投影重力", grav, "-", "projected_gravity"),
    ]:
        seg = arr[sl][fresh]        # 去掉重复帧，否则噪声被严重低估
        sigma = second_diff_sigma(seg)
        raw = seg.std(axis=0)
        s_mean, s_max = float(sigma.mean()), float(sigma.max())
        # 均匀分布 U(-a, a) 的标准差是 a/√3，所以等效半宽 a = √3·σ
        amp = np.sqrt(3.0) * s_mean
        out[grp] = float(amp)
        print(f"{name:<18}{unit:>4}{s_mean:>10.4f}(max {s_max:.4f}){raw.mean():>14.4f}{amp:>16.4f}")

    print("-" * 76)
    print("二阶差分估计 = 噪声底（高频成分）；原始 std 含真实微动，是上界。")
    print("训练用的是均匀分布 U(-a, a)，a = √3·σ 使其方差与实测噪声一致。")

    print(f"\n建议写进 velocity_env_cfg.py 的噪声配置（对比 IsaacLab 四足通用值）：")
    generic = {"joint_pos": 0.01, "joint_vel": 1.5, "base_ang_vel": 0.2, "projected_gravity": 0.05}
    for k2, v in out.items():
        g = generic[k2]
        print(f"  {k2:<20} 实测 ±{v:.4f}   通用值 ±{g:<6}   实测 / 通用 = {v / g:.2f}x")

    print(f"\n{json.dumps(out, indent=1)}")


if __name__ == "__main__":
    main()
