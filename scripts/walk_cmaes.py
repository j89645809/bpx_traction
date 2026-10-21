#!/usr/bin/env python3
"""真机走路切片上的 CMA-ES 参数拟合。

方法 = 把 `walk_refit` 的网格换成 CMA-ES，目标函数、切片构造、起点剔除全部**只读复用**
pace 的 `walk_replay.short_horizon`（支撑检测 + FK 初始化 + 不可靠起点跳过），
保证和 4 月落地/走路分析同一口径。

优化变量 θ（质量钉死 15.1 kg，手册值，避免与 μ 耦合）：
  f0          膝承重摩擦截距 [N·m]       先验 1.0（落地辨识）
  mu          摩擦-负载斜率              先验 0.26
  delay       力矩延迟 [仿真步@500Hz]    先验 2（悬空辨识 4 ms）
  damp_scale  膝粘滞阻尼倍率             先验 1.0

目标 = 所有切片 0.5 s 开环重放的关节 RMS（与 walk_refit 相同指标）。
站立段不进目标（无激励），拟合完单独验收。

    conda activate bpx_rma
    python scripts/walk_cmaes.py logs/real/rec_20260418_093426_joint500.bin
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BPX = Path(__file__).resolve().parents[1]
PACE = Path.home() / "intern" / "ZGYD" / "pace" / "pace_bpx_pro"
sys.path.insert(0, str(PACE / "scripts" / "real"))
sys.path.insert(0, str(PACE))

import walk_replay as wr  # noqa: E402  只读复用
from pace_bpx_pro.mujoco_floor import FloorSim, load_final_yaml  # noqa: E402

KNEE = wr.GROUPS["knee"]
MASS_KG = 15.1

#           f0    mu   delay damp_scale
PRIOR = np.array([1.0, 0.26, 2.0, 1.0])
LO    = np.array([0.0, 0.00, 0.0, 0.3])
HI    = np.array([3.5, 0.60, 6.0, 2.5])
SIGMA0 = np.array([0.6, 0.12, 1.2, 0.35])


def build_sim(final: dict, theta: np.ndarray) -> FloorSim:
    f0, mu, delay, dscale = theta
    p = dict(final)
    p["delay_steps"] = int(round(delay))
    sim = FloorSim(p, kp=40.0, kd=1.0, dt=0.002)
    sim.set_hard_friction()
    wr.set_total_mass(sim, MASS_KG)
    sim.model.dof_damping[sim.dof_idx[KNEE]] = np.asarray(final["viscous"])[KNEE] * dscale
    sim.set_load_friction(KNEE, float(f0), float(mu))
    return sim


# 模块级目标函数：Linux fork 下子进程直接继承 _G，Pool.map 只传 theta
_G: dict = {}


def eval_theta(theta: np.ndarray) -> float:
    theta = np.clip(theta, LO, HI)
    sim = build_sim(_G["final"], theta)
    quiet = lambda *a, **k: None
    s = wr.short_horizon({"m": sim}, _G["rec"], _G["k_starts"], [_G["horizon"]], log=quiet)["m"]
    if _G["horizon"] not in s:
        return 1e3
    return s[_G["horizon"]]["q_rms_all"]


def cma_es(f, x0, sigma0, lo, hi, pop=10, iters=28, workers=8, log=print):
    """最小可用 CMA-ES（Hansen 标准步骤，对角起步的全协方差版）。"""
    n = len(x0)
    mean = x0.copy()
    sigma = 1.0
    scales = sigma0.copy()                     # 各维尺度（把问题归一化）
    C = np.eye(n); pc = np.zeros(n); ps = np.zeros(n)
    mu_w = pop // 2
    w = np.log(mu_w + 0.5) - np.log(np.arange(1, mu_w + 1)); w /= w.sum()
    mu_eff = 1.0 / (w ** 2).sum()
    cc = 4 / (n + 4); cs = (mu_eff + 2) / (n + mu_eff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff); cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    ds = 1 + cs + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    from multiprocessing import Pool
    hist = []
    best_x, best_f = x0.copy(), f(x0)
    log(f"[cma] 先验 θ={np.round(x0,3).tolist()}  目标 {best_f:.4f}")
    with Pool(workers) as pool:
        for it in range(iters):
            A = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            Z = np.random.randn(pop, n)
            X = mean + sigma * (Z @ A.T) * scales
            X = np.clip(X, lo, hi)
            F = pool.map(f, list(X))
            order = np.argsort(F)
            if F[order[0]] < best_f:
                best_f = F[order[0]]; best_x = X[order[0]].copy()
            Xs = (X[order[:mu_w]] - mean) / (sigma * scales)
            mean_new = mean + sigma * scales * (w @ Xs)
            y = (mean_new - mean) / (sigma * scales)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * np.linalg.solve(A, y)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1.4 + 2 / (n + 1)) * chiN
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C) \
                + cmu * sum(wi * np.outer(zi, zi) for wi, zi in zip(w, Xs))
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            mean = mean_new
            hist.append({"iter": it, "best": float(best_f), "gen_best": float(F[order[0]]),
                         "mean": mean.tolist(), "sigma": float(sigma)})
            log(f"[cma] it{it:2d}  代最优 {F[order[0]]:.4f}  全局最优 {best_f:.4f}  "
                f"θ*={np.round(best_x,3).tolist()}  σ={sigma:.2f}")
    return best_x, best_f, hist


def build_slices(rec: dict, horizon: float, step_s: float = 0.4, min_vx: float = 0.25):
    """行走起点（每 step_s 一个）+ 起步过渡起点（站→走切换前 0.2 s，对 f0 敏感）。

    注意 rec["cmd"] 是 12 维关节位置指令（qcmd），速度指令在 rec["vcmd"]。
    """
    pol = rec["sub"] == 2
    walk = pol & (np.abs(rec["vcmd"][:, 0]) > min_vx)
    step = int(round(step_s / 0.002))
    idx = np.flatnonzero(walk)
    k_walk = list(idx[::step])
    on = np.flatnonzero(np.diff(walk.astype(int)) > 0)
    k_trans = [int(k) - 100 for k in on if k > 200]
    k_starts = sorted(set(k_walk + k_trans))
    nmax = int(round(horizon / 0.002)) + 1
    k_starts = [k for k in k_starts if k + nmax < len(rec["t"])]
    return k_starts, len(k_walk), len(k_trans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("joint_bin")
    ap.add_argument("--final-yaml", default=str(PACE / "data" / "bpx_pro" / "bpx_pro_final_params.yaml"))
    ap.add_argument("--horizon", type=float, default=0.5)
    ap.add_argument("--step-s", type=float, default=0.4)
    ap.add_argument("--min-vx", type=float, default=0.25, help="只取 |cmd_vx| 高于此的行走起点")
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--iters", type=int, default=28)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(BPX / "logs" / "walk_cmaes"))
    args = ap.parse_args()

    rec = wr.load_rec(Path(args.joint_bin).expanduser())
    final = load_final_yaml(Path(args.final_yaml).expanduser())
    k_starts, n_walk, n_trans = build_slices(rec, args.horizon, args.step_s, args.min_vx)
    print(f"[data] 行走起点 {n_walk} + 过渡起点 {n_trans} -> {len(k_starts)} 个切片，"
          f"0.5 s 开环，质量钉 {MASS_KG} kg")

    _G.update(final=final, rec=rec, k_starts=k_starts, horizon=args.horizon)
    f = eval_theta
    t0 = time.time()
    best_x, best_f, hist = cma_es(f, PRIOR.copy(), SIGMA0, LO, HI,
                                  pop=args.pop, iters=args.iters, workers=args.workers)
    print(f"[cma] 用时 {time.time()-t0:.0f} s")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    prior_f = hist[0] and f(PRIOR)  # 复算先验（同一实现路径）
    names = ["f0", "mu", "delay_steps", "knee_damp_scale"]
    result = {
        "rec": args.joint_bin, "mass_fixed_kg": MASS_KG, "horizon_s": args.horizon,
        "n_slices": len(k_starts),
        "prior": dict(zip(names, PRIOR.tolist())), "prior_q_rms": float(prior_f),
        "best": dict(zip(names, np.round(best_x, 4).tolist())), "best_q_rms": float(best_f),
        "improvement_pct": float(100 * (prior_f - best_f) / prior_f),
        "history": hist,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
