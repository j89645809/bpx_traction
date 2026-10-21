#!/usr/bin/env python3
"""walk_cmaes 结果可视化：CMA-ES 收敛曲线 + 真机走路切片上先验 vs θ* 的开环重放对比。

    conda activate bpx_rma
    python scripts/plot_walk_cmaes.py logs/real/rec_20260418_093426_joint500.bin

产出（logs/walk_cmaes/）：
  cmaes_convergence.png   目标值 / 步长 σ / 参数均值随代数的变化
  walk_fit_clips.png      两个代表性切片（匀速走 + 起步过渡）real vs 先验 vs θ*
  walk_fit_stats.png      222 个切片逐起点 RMS 散点（先验 vs θ*）+ 分组 RMS
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BPX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BPX / "scripts"))

import walk_cmaes as wc  # noqa: E402  复用 build_sim / build_slices / PRIOR
from walk_cmaes import wr  # noqa: E402
from pace_bpx_pro.mujoco_floor import load_final_yaml  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "HYZhongYuanB5", "AR PL UKai CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_REAL, C_PRIOR, C_BEST = "#222222", "#1f77b4", "#ff7f0e"
NAMES = ["f0", "mu", "delay_steps", "knee_damp_scale"]
LABELS = ["f0 [N·m]", "μ", "延迟 [步@500Hz]", "膝阻尼倍率"]


def style(ax):
    ax.grid(color="#e6e8eb", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_convergence(res: dict, path: Path):
    hist = res["history"]
    it = [h["iter"] for h in hist]
    gen = [h["gen_best"] for h in hist]
    best = [h["best"] for h in hist]
    sig = [h["sigma"] for h in hist]
    means = np.array([h["mean"] for h in hist])

    fig, ax = plt.subplots(3, 1, figsize=(8.5, 8.5), sharex=True, constrained_layout=True)
    ax[0].axhline(res["prior_q_rms"], color=C_PRIOR, ls="--", lw=1.2,
                  label=f"先验 θ 的目标值 {res['prior_q_rms']:.4f}")
    ax[0].plot(it, gen, "o", ms=4, color="#bbbbbb", label="每代最优")
    ax[0].plot(it, best, color=C_BEST, lw=1.8, label=f"累计最优 → {res['best_q_rms']:.4f}")
    ax[0].set_ylabel("0.5 s 开环关节 RMS [rad]")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title(f"CMA-ES 收敛（{res['n_slices']} 个走路切片，质量钉 {res['mass_fixed_kg']} kg）"
                    f" — 相对先验仅改善 {res['improvement_pct']:.1f}%：目标面平坦", fontsize=10)
    style(ax[0])

    ax[1].plot(it, sig, color="#2ca02c", lw=1.6)
    ax[1].set_ylabel("全局步长 σ")
    ax[1].set_title("σ 单调收缩 → 优化器正常收敛（不是没拟合出来）", fontsize=10)
    style(ax[1])

    lo, hi = wc.LO, wc.HI
    for j, (nm, lb) in enumerate(zip(NAMES, LABELS)):
        ax[2].plot(it, (means[:, j] - lo[j]) / (hi[j] - lo[j]), lw=1.5, label=lb)
        ax[2].axhline((wc.PRIOR[j] - lo[j]) / (hi[j] - lo[j]), ls=":", lw=0.8,
                      color=ax[2].lines[-1].get_color())
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].set_ylabel("参数均值（归一化到搜索区间）")
    ax[2].set_xlabel("代数")
    f0, mu = res["best"]["f0"], res["best"]["mu"]
    ax[2].set_title(f"虚线 = 先验位置；f0 与 μ 互相补偿（等效摩擦 f0+7.5μ：先验 2.95 vs θ* "
                    f"{f0 + 7.5 * mu:.2f}）→ 只有组合可辨识", fontsize=10)
    ax[2].legend(frameon=False, fontsize=9, ncol=4)
    style(ax[2])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pick_clips(rec: dict, k_starts: list[int], sim) -> list[tuple[int, str]]:
    """挑两个代表性切片：匀速走（|cmd_vx| 最大的可靠起点）+ 起步过渡（vcmd 刚上来）。"""
    def reliable(k, n):
        if (rec["kp"][k:k + n] <= 0).any() or (rec["hold"][k:k + n] > 5).any():
            return False
        s = wr.initial_state(sim, rec, k)
        return s["stance"].sum() >= 2 and np.linalg.norm(s["v"]) <= 2.5

    n = 251
    vx = np.abs(rec["vcmd"][:, 0])
    steady = sorted((k for k in k_starts if vx[k] > 0.25), key=lambda k: -vx[k])
    walk_on = np.flatnonzero(np.diff((vx > 0.25).astype(int)) > 0)
    trans = [k for k in k_starts if any(abs(k - (o - 100)) < 3 for o in walk_on)]
    clips = []
    for pool, label in [(steady, "匀速行走"), (trans, "站→走起步")]:
        for k in pool:
            if k + n < len(rec["t"]) and reliable(k, n):
                clips.append((int(k), f"{label}（cmd_vx={rec['vcmd'][k + 150, 0]:.2f} m/s）"))
                break
    return clips


def fig_clips(rec, final, theta_best, clips, path: Path):
    n = 251  # 0.5 s
    panels = [(1, "FL 髋俯仰 q [rad]", 1.0), (2, "FL 膝 q [rad]", 1.0), (5, "FR 膝 q [rad]", 1.0)]
    fig, axes = plt.subplots(len(panels), len(clips), figsize=(11, 8),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_2d(axes.reshape(len(panels), len(clips)))
    for ci, (k0, label) in enumerate(clips):
        sims = {"先验": wc.build_sim(final, wc.PRIOR), "θ*": wc.build_sim(final, theta_best)}
        outs = {nm: wr.rollout(s, rec, k0, k0 + n, tau0=rec["tau_pd"][k0]) for nm, s in sims.items()}
        t = rec["t"][k0:k0 + n] - rec["t"][k0]
        for pi, (j, yl, sc) in enumerate(panels):
            a = axes[pi, ci]
            a.plot(t, rec["q"][k0:k0 + n, j] * sc, color=C_REAL, lw=1.8, label="真机")
            a.plot(t, outs["先验"]["q"][:, j] * sc, color=C_PRIOR, lw=1.2, label="先验 θ（落地辨识）")
            a.plot(t, outs["θ*"]["q"][:, j] * sc, color=C_BEST, lw=1.2, ls="--", label="CMA-ES θ*")
            if ci == 0:
                a.set_ylabel(yl)
            style(a)
        axes[0, ci].set_title(label, fontsize=10)
        axes[-1, ci].set_xlabel("切片内时间 [s]")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("真机走路切片 0.5 s 开环重放：两组参数几乎重合 → 目标对 θ 不敏感", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_stats(rec, final, theta_best, k_starts, horizon, path: Path):
    sims = {"prior": wc.build_sim(final, wc.PRIOR), "best": wc.build_sim(final, theta_best)}
    s = wr.short_horizon(sims, rec, k_starts, [horizon], log=lambda *a, **k: None)
    sp, sb = s["prior"][horizon], s["best"][horizon]
    rp = np.array(sp["per_start_q_rms"])
    rb = np.array(sb["per_start_q_rms"])

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    lim = max(rp.max(), rb.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], color="#999999", lw=1.0, ls="--")
    ax[0].scatter(rp, rb, s=14, alpha=0.55, color="#6a51a3", edgecolors="none")
    ax[0].set_xlabel("先验 θ 的逐起点 RMS [rad]")
    ax[0].set_ylabel("θ* 的逐起点 RMS [rad]")
    ax[0].set_title(f"逐起点对比（候选 {len(k_starts)}，起始状态可靠 n={sp['n_starts']}）：点贴在对角线上\n"
                    f"总 RMS {np.sqrt((rp**2).mean()):.4f} → {np.sqrt((rb**2).mean()):.4f}", fontsize=10)
    ax[0].set_xlim(0, lim)
    ax[0].set_ylim(0, lim)
    style(ax[0])

    groups = list(wr.GROUPS)
    x = np.arange(len(groups))
    gp = [sp["q_rms_group"][g] for g in groups]
    gb = [sb["q_rms_group"][g] for g in groups]
    ax[1].bar(x - 0.18, gp, 0.36, color=C_PRIOR, label="先验 θ")
    ax[1].bar(x + 0.18, gb, 0.36, color=C_BEST, label="CMA-ES θ*")
    ax[1].set_xticks(x, ["髋滚转", "髋俯仰", "膝"])
    ax[1].set_ylabel("分组 RMS [rad]")
    ax[1].set_title("分组误差：三组都只小幅下降（3–7%），θ 的影响远小于模型本身的开环误差", fontsize=10)
    ax[1].legend(frameon=False, fontsize=9)
    style(ax[1])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"per_start": (rp, rb), "groups": (gp, gb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("joint_bin")
    ap.add_argument("--result", default=str(BPX / "logs" / "walk_cmaes" / "result.json"))
    ap.add_argument("--out", default=str(BPX / "logs" / "walk_cmaes"))
    args = ap.parse_args()

    res = json.loads(Path(args.result).read_text())
    rec = wr.load_rec(Path(args.joint_bin).expanduser())
    final = load_final_yaml(Path(BPX.parent / "pace" / "pace_bpx_pro" / "data" / "bpx_pro" /
                                 "bpx_pro_final_params.yaml"))
    theta_best = np.array([res["best"][n] for n in NAMES])
    k_starts, n_walk, n_trans = wc.build_slices(rec, res["horizon_s"])
    print(f"[data] 切片 {len(k_starts)}（走 {n_walk} + 过渡 {n_trans}）")

    out = Path(args.out)
    fig_convergence(res, out / "cmaes_convergence.png")
    print("[fig] cmaes_convergence.png")

    probe = wc.build_sim(final, wc.PRIOR)
    clips = pick_clips(rec, k_starts, probe)
    print(f"[fig] 代表切片: {clips}")
    fig_clips(rec, final, theta_best, clips, out / "walk_fit_clips.png")
    print("[fig] walk_fit_clips.png")

    fig_stats(rec, final, theta_best, k_starts, res["horizon_s"], out / "walk_fit_stats.png")
    print("[fig] walk_fit_stats.png")


if __name__ == "__main__":
    main()
