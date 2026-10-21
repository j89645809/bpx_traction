#!/usr/bin/env python3
"""100 米计时：从静止起步，一路给定速度指令，记到跑完 100 米。

和 `sim2sim.py` 同一个 MuJoCo corrected 参考模型。只读引用 pace_bpx_pro，不需要 Isaac Sim。

    conda activate bpx_rma
    python scripts/dash_100m.py --cmd 3.0 \
        --policy 辨识=logs/rsl_rl/bpx_pro_fast/*_student/exported/policy.onnx
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from bpx_rma import spec  # noqa: E402
from sim2sim import OnnxPolicy, build_reference_sim, import_pace, quat_to_proj_gravity, settle  # noqa: E402


def dash(sim, policy: OnnxPolicy, cmd_vx: float, distance: float, timeout_s: float = 120.0,
         ramp_rate: float = 0.0) -> dict:
    """ramp_rate > 0 时指令从 1.0 m/s 起线性升到 cmd_vx（模拟逐级推杆；
    从静止直接给高速指令会越出训练分布，策略会选择原地站住）。"""
    q0 = np.array(spec.DEFAULT_JOINT_POS)
    settle(sim, q0)
    policy.reset()
    cmd = np.array([cmd_vx, 0.0, 0.0], dtype=np.float32)

    start_xy = sim.base_pos()[:2].copy()
    prev_xy = start_xy.copy()
    path = 0.0                      # 走过的路程（含横向漂移）
    t = 0.0
    splits: list[tuple[float, float]] = []   # (距离, 时间)
    next_split = 10.0
    peak = 0.0
    fell_at = None

    while path < distance and t < timeout_s:
        if ramp_rate > 0.0:
            cmd[0] = min(cmd_vx, 1.0 + ramp_rate * t)
        qw, qx, qy, qz = sim.data.qpos[sim.free_qpos + 3: sim.free_qpos + 7]
        grav = quat_to_proj_gravity(qw, qx, qy, qz)
        target = policy.act(policy.frame(sim.q_enc(), sim.dq(), sim.base_gyro(), grav, cmd))
        for _ in range(spec.DECIMATION):
            sim.step(target, kp=spec.KP_POLICY, kd=spec.KD_POLICY)
        t += 1.0 / spec.POLICY_HZ

        xy = sim.base_pos()[:2]
        step_len = float(np.linalg.norm(xy - prev_xy))
        path += step_len
        peak = max(peak, step_len * spec.POLICY_HZ)
        prev_xy = xy.copy()

        tilt = float(np.degrees(np.arccos(np.clip(-grav[2], -1.0, 1.0))))
        if fell_at is None and (tilt > 60.0 or sim.base_pos()[2] < 0.15):
            fell_at = t
        while next_split <= path <= distance:
            splits.append((next_split, t))
            next_split += 10.0

    net = float(np.linalg.norm(sim.base_pos()[:2] - start_xy))
    return {"time": t, "path": path, "net": net, "splits": splits,
            "peak": peak, "avg": path / t if t else 0.0, "fell_at": fell_at,
            "finished": path >= distance}


def main() -> None:
    p = argparse.ArgumentParser(description="100 米计时")
    p.add_argument("--policy", action="append", required=True, help="标签=路径（可含通配符）")
    p.add_argument("--cmd", type=float, nargs="*", default=[3.0], help="速度指令，可给多个")
    p.add_argument("--distance", type=float, default=100.0)
    p.add_argument("--ramp", type=float, default=0.0, help="指令渐升速率 m/s^2（0=直接给目标值）")
    p.add_argument("--pace-dir", default=None)
    p.add_argument("--ref-params", default=str(PROJECT_DIR / "data" / "bpx_pro_training_params.yaml"))
    p.add_argument("--mass", type=float, default=15.1)
    p.add_argument("--knee-fric", default="1.0,0.26")
    args = p.parse_args()

    mj_floor, pace_dir = import_pace(args.pace_dir)
    print(f"[dash] 参考模型 = MuJoCo corrected（只读引用 {pace_dir}）\n")

    entries = []
    for item in args.policy:
        label, _, pat = item.partition("=")
        hits = sorted(glob.glob(pat or item))
        if not hits:
            raise SystemExit(f"找不到策略：{pat or item}")
        entries.append((label or Path(hits[-1]).parent.parent.name, Path(hits[-1])))

    print(f"{'策略':<14}{'指令':>7}{'100m 用时':>11}{'平均速度':>10}{'峰值':>8}{'横向漂移':>10}  备注")
    print("-" * 74)
    for label, path in entries:
        for c in args.cmd:
            sim, _ = build_reference_sim(mj_floor, args)
            r = dash(sim, OnnxPolicy(path), c, args.distance, ramp_rate=args.ramp)
            note = "" if r["finished"] else f"只跑了 {r['path']:.1f} m"
            if r["fell_at"]:
                note = (note + f" 第 {r['fell_at']:.1f}s 摔倒").strip()
            drift = r["path"] - r["net"]
            print(f"{label:<14}{c:>6.1f}{r['time']:>10.2f}s{r['avg']:>9.2f}"
                  f"{r['peak']:>8.2f}{drift:>9.1f}m  {note}")
            if r["splits"]:
                marks = [f"{d:.0f}m@{tt:.1f}s" for d, tt in r["splits"] if d % 20 == 0]
                print(f"{'':14}分段 " + "  ".join(marks))
    print("-" * 74)
    print("速度单位 m/s；横向漂移 = 路程 − 直线位移（直线跑得越正越小）。")


if __name__ == "__main__":
    main()
