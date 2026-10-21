#!/bin/bash
# Sprint4 对照组（厂家参数，无辨识）：逐步复刻辨识组的完整训练路径。
#
# 辨识组路径（2026-10-15 ~ 10-19）：
#   sprint3 teacher 从零 4200 迭代（Sprint3 cfg）
#   -> 微调 r1: Sprint4 cfg, 二次型 -2.0, +800
#   -> r2: 二次型 -6.0, +600
#   -> r3: 线性亏空 -1.5, +500
#   -> r4: 线性亏空 -3.0, +600      （最终 teacher）
#   -> 蒸馏 1500 -> 导出
# 本脚本用 BPX_S4_CLEARANCE 环境变量逐轮复刻同样的剂量，唯一差异 = 参数 yaml。
#
#   nohup systemd-inhibit --what=sleep --why="bpx sprint4 baseline" \
#       bash scripts/chain_sprint4_baseline.sh > logs/chain_sprint4_baseline.log 2>&1 &
#
# 不要加 set -u：Isaac Sim 的 setup_conda_env.sh 无保护引用 $ZSH_VERSION，会静默打死脚本。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda activate 失败"; exit 1; }

EXP=bpx_pro_sprint4_baseline
YAML=data/bpx_pro_baseline_params.yaml
T3=Isaac-Velocity-Flat-BpxPro-Sprint3-v0
T4=Isaac-Velocity-Flat-BpxPro-Sprint4-v0

run_teacher() {  # run_teacher <task> <iters> <resume(0/1)> <clearance 或 "">
    local extra=""
    [ "$3" = "1" ] && extra="--resume --load-run .*_teacher"
    BPX_S4_CLEARANCE="$4" BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/train.py --headless \
        --task "$1" --num_envs 4096 --max_iterations "$2" ${extra} \
        --experiment-name "${EXP}" --run-name teacher >> logs/train_teacher_s4b.log 2>&1
    local rc=$?
    echo "[chain] $(date '+%F %T') teacher 段 task=$1 iters=$2 clear='$4' 退出码 ${rc}"
    [ ${rc} -ne 0 ] && exit 1
    sleep 20
}

echo "[chain] $(date '+%F %T') ${EXP} 开始：阶段1 从零 4200（Sprint3 cfg）"
run_teacher "$T3" 4200 0 ""
echo "[chain] 阶段2 微调 r1 二次型-2.0 +800"
run_teacher "$T4" 800 1 "quad:-2.0"
echo "[chain] 阶段3 微调 r2 二次型-6.0 +600"
run_teacher "$T4" 600 1 "quad:-6.0"
echo "[chain] 阶段4 微调 r3 线性-1.5 +500"
run_teacher "$T4" 500 1 "lin:-1.5"
echo "[chain] 阶段5 微调 r4 线性-3.0 +600"
run_teacher "$T4" 600 1 "lin:-3.0"

echo "[chain] $(date '+%F %T') 蒸馏 1500"
BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/train.py --headless \
    --task "$T4" --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --experiment-name "${EXP}" --run-name student > logs/train_student_s4b.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') student 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20

BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint4-Play-v0 --experiment-name "${EXP}" \
    > logs/chain_s4b_export.log 2>&1
echo "[chain] $(date '+%F %T') 导出退出码 $?  对照组全部完成"
