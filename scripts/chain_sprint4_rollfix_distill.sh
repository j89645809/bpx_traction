#!/bin/bash
# Sprint4 后半条链：蒸馏 + 导出（teacher 已经微调定稿：r4 = 线性亏空 -3.0，真峰值 6.2 cm）。
#
#   nohup systemd-inhibit --what=sleep --why="bpx sprint4 distill" \
#       bash scripts/chain_sprint4_distill.sh > logs/chain_sprint4_distill.log 2>&1 &
#
# 不要加 set -u：Isaac Sim 的 setup_conda_env.sh 无保护引用 $ZSH_VERSION，会静默打死脚本。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda activate 失败"; exit 1; }

TASK=Isaac-Velocity-Flat-BpxPro-Sprint4-v0
EXP=bpx_pro_sprint4
YAML=data/bpx_pro_training_params.yaml

echo "[chain] $(date '+%F %T') ${EXP} / student (1500 迭代，teacher=2026-10-21_01-04-53_teacher_rollfix)"
BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/train.py --headless \
    --task "${TASK}" --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --load-run ".*_rollfix" \
    --experiment-name "${EXP}" --run-name student > logs/train_student_s4rf.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') student 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20

BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint4-Play-v0 --experiment-name "${EXP}" \
    > logs/chain_s4rf_export.log 2>&1
echo "[chain] $(date '+%F %T') 导出退出码 $?  蒸馏链完成"
