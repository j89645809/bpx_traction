#!/usr/bin/env python3
"""拿板子的**原始代码**验证我们的实现，而不是验证我对它的理解。

`check_contract.py` 和 `sim2sim.py` 都是照着读 `bpx_dual_pad.py` 写出来的。
如果我把板子读错了，那两处会一起错、还互相印证。这个脚本堵住这个洞：
给 `bpx_sdk` 打桩，把板子的源码**原样**加载进来，直接调它自己的
`build_obs_frame` / `flatten_term_major_history` / `run_policy`，
用随机状态和真实导出的 ONNX，逐位比对。

不连机器人、不需要 SDK、不修改板子存档的任何文件。

    conda activate bpx_rma
    python scripts/verify_against_board.py
"""

from __future__ import annotations

import argparse
import glob
import sys
import types
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bpx_rma import spec  # noqa: E402

DEFAULT_BOARD = Path.home() / "intern" / "ZGYD" / "pace" / "robot_cat" / "bpx_dual_pad_flush.py"


def load_board_module(path: Path) -> types.ModuleType:
    """给 bpx_sdk 打桩后，把板子脚本当模块加载（不执行 main）。"""
    if "bpx_sdk" not in sys.modules:
        stub = types.ModuleType("bpx_sdk")
        stub.__getattr__ = lambda name: types.SimpleNamespace()  # type: ignore[attr-defined]
        sys.modules["bpx_sdk"] = stub
    src = path.read_text()
    mod = types.ModuleType("board")
    mod.__file__ = str(path)
    mod.__name__ = "board"          # != "__main__"，所以末尾的 main() 不会跑
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="用板子原始代码验证本项目的部署契约实现")
    ap.add_argument("--board", default=str(DEFAULT_BOARD), help="板子脚本存档路径")
    ap.add_argument("--policy", default="logs/rsl_rl/bpx_pro_flat/*_student/exported/policy.onnx")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    board = load_board_module(Path(args.board).expanduser())
    print(f"[board] 已加载 {args.board}")

    fails: list[str] = []

    def check(name: str, ours, theirs, tol: float = 0.0) -> None:
        a, b = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
        if a.shape != b.shape:
            fails.append(f"{name}: 形状 {a.shape} != {b.shape}")
            print(f"  ✗ {name:<28} 形状不符 {a.shape} vs {b.shape}")
            return
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        ok = d <= tol
        print(f"  {'✓' if ok else '✗'} {name:<28} 最大差 {d:.3e}")
        if not ok:
            fails.append(f"{name}: 最大差 {d:.3e}")

    # ---- 1. 常数 ----
    print("\n[1] 常数（板子写死的，我们必须照抄）")
    # 板子把这两个存成 np.float32，我们是 Python float（float64），
    # 所以容差给到 float32 的分辨率，不是真的不一致
    check("DEFAULT_POSE / q0", spec.DEFAULT_JOINT_POS, board.DEFAULT_POSE, tol=1e-6)
    check("MODEL_LIMITS", spec.JOINT_POS_LIMITS, board.MODEL_LIMITS, tol=1e-6)
    check("KP_POLICY", spec.KP_POLICY, board.KP_POLICY)
    check("KD_POLICY", spec.KD_POLICY, board.KD_POLICY)
    check("ACTION_SCALE", spec.ACTION_SCALE, board.ACTION_SCALE)
    check("ACTION_CLIP", spec.ACTION_CLIP, board.ACTION_CLIP)
    check("POLICY_FRAME_SIZE", spec.OBS_FRAME_SIZE, board.POLICY_FRAME_SIZE)
    check("POLICY_HISTORY_LENGTH", spec.HISTORY_LENGTH, board.POLICY_HISTORY_LENGTH)
    check("HISTORY_INPUT_SIZE", spec.HISTORY_INPUT_SIZE, board.POLICY_HISTORY_INPUT_SIZE)
    check("POLICY_TERM_SLICES", spec.obs_term_slices(), board.POLICY_TERM_SLICES)
    check("HZ_POLICY", spec.POLICY_HZ, board.HZ_POLICY)
    check("HZ_JOINT", spec.JOINT_HZ, board.HZ_JOINT)
    check("EFFORT_LIMIT", 30.0, board.EFFORT_LIMIT)

    # ---- 2. 投影重力 ----
    print("\n[2] 投影重力（板子 xyzw 顺序 vs 我们的 wxyz）")
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))
    from sim2sim import quat_to_proj_gravity  # noqa: PLC0415

    rng = np.random.default_rng(args.seed)
    worst = 0.0
    for _ in range(args.trials):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        x, y, z, w = q
        worst = max(worst, float(np.max(np.abs(
            quat_to_proj_gravity(w, x, y, z) - board.projected_gravity_xyzw([x, y, z, w])
        ))))
    print(f"  {'✓' if worst <= 1e-6 else '✗'} {'projected_gravity':<28} 最大差 {worst:.3e}（{args.trials} 组随机四元数）")
    if worst > 1e-6:
        fails.append(f"projected_gravity: {worst:.3e}")

    # ---- 3. 45 维帧 + 历史展平 + 完整推理 ----
    hits = sorted(glob.glob(args.policy))
    if not hits:
        raise SystemExit(f"找不到 ONNX：{args.policy}")
    onnx = Path(hits[-1])
    print(f"\n[3] 完整推理链（ONNX: {onnx}）")

    from sim2sim import OnnxPolicy  # noqa: PLC0415

    ours = OnnxPolicy(onnx)
    # 把板子的 POLICY_ONNX 指到我们导出的文件，让**它自己的**校验代码来验：
    # 2 输入 1 输出、名称、dtype、shape 全部由 make_policy_session 把关
    board.POLICY_ONNX = onnx
    session, input_names = board.App.make_policy_session(types.SimpleNamespace())
    print(f"  ✓ {'板子 make_policy_session 校验':<28} 通过（输入 {sorted(input_names)}）")

    worst_frame = worst_flat = worst_target = 0.0
    ours.reset()
    last_action = np.zeros(12, np.float32)
    hist_board = None
    for k in range(args.trials):
        quat = rng.standard_normal(4)
        quat /= np.linalg.norm(quat)
        snap = {
            "q": rng.uniform(-1.0, 1.0, 12).astype(np.float32),
            "dq": rng.uniform(-5.0, 5.0, 12).astype(np.float32),
            "omega": rng.uniform(-2.0, 2.0, 3).astype(np.float32),
            "quat": quat.astype(np.float32),
            "fresh_age": 0.0,
        }
        cmd = rng.uniform(-1.0, 1.0, 3).astype(np.float32)

        # 板子的帧
        f_board = board.App.build_obs_frame(None, cmd, last_action, snap=snap)
        # 我们的帧（投影重力要先按板子的顺序算）
        x, y, z, w = quat
        grav = quat_to_proj_gravity(w, x, y, z)
        ours.last_action = last_action
        f_ours = ours.frame(snap["q"], snap["dq"], snap["omega"], grav, cmd)
        worst_frame = max(worst_frame, float(np.max(np.abs(f_board - f_ours))))

        # 历史（两边都用板子的滚动方式）
        if hist_board is None:
            hist_board = np.repeat(f_board.reshape(1, -1), spec.HISTORY_LENGTH, axis=0)
        else:
            hist_board[:-1] = hist_board[1:]
            hist_board[-1] = f_board
        flat_board = board.App.flatten_term_major_history(hist_board)
        flat_ours = np.concatenate(
            [hist_board[:, b:e].reshape(-1) for b, e in spec.obs_term_slices()]
        ).astype(np.float32)
        worst_flat = max(worst_flat, float(np.max(np.abs(flat_board - flat_ours))))

        # 完整推理 + 目标角
        raw_b, target_b = board.App.run_policy(session, input_names, f_board, hist_board)
        target_o = ours.act(f_ours)          # 内部自己滚历史，和板子同步
        worst_target = max(worst_target, float(np.max(np.abs(target_b - target_o))))
        last_action = raw_b

    for name, d in [("build_obs_frame", worst_frame),
                    ("flatten_term_major_history", worst_flat),
                    ("run_policy -> 关节目标角", worst_target)]:
        ok = d <= 1e-6
        print(f"  {'✓' if ok else '✗'} {name:<28} 最大差 {d:.3e}（{args.trials} 步）")
        if not ok:
            fails.append(f"{name}: {d:.3e}")

    print()
    if fails:
        print(f"✗ {len(fails)} 项不符：")
        for f in fails:
            print(f"    {f}")
        raise SystemExit(1)
    print("✓ 全部逐位一致 —— 本项目的实现和板子原始代码等价，不是我的转述。")


if __name__ == "__main__":
    main()
