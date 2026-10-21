#!/bin/bash
# 冲刺组（指令上限 4.5 m/s，放开奔跑步态的奖励约束，电机按手册 33 N·m）。
# 只训辨识参数这一组 —— 这是速度探索，不是辨识对照实验。
#
#   nohup bash scripts/chain_sprint.sh > logs/chain_sprint.log 2>&1 &
#
# 不要加 set -u：Isaac Sim 的 setup_conda_env.sh 无保护引用 $ZSH_VERSION，会静默打死脚本。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda activate 失败"; exit 1; }

TASK=Isaac-Velocity-Flat-BpxPro-Sprint-v0
EXP=bpx_pro_sprint
YAML=data/bpx_pro_training_params.yaml

echo "[chain] $(date '+%F %T') ${EXP} / teacher (1800 迭代，高速更难学，给足预算)"
BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/train.py --headless \
    --task "${TASK}" --num_envs 4096 --max_iterations 1800 \
    --experiment-name "${EXP}" --run-name teacher > logs/train_teacher_sprint.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') teacher 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20

if [ -z "$(ls -v logs/rsl_rl/${EXP}/*_teacher/model_*.pt 2>/dev/null | tail -1)" ]; then
    echo "[chain] 中止：没有 teacher checkpoint"; exit 1
fi

echo "[chain] $(date '+%F %T') ${EXP} / student (1500 迭代)"
BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/train.py --headless \
    --task "${TASK}" --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --experiment-name "${EXP}" --run-name student > logs/train_student_sprint.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') student 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20

BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint-Play-v0 --experiment-name "${EXP}" \
    > logs/chain_sprint_export.log 2>&1
echo "[chain] $(date '+%F %T') 导出退出码 $?  全部完成"
