#!/bin/bash
# baseline 第一阶段跑完后自动接上蒸馏。
# 用 nohup 脱离终端跑，不受任何工具超时限制：
#   nohup bash scripts/chain_baseline.sh <第一阶段的PID> > logs/chain_baseline.log 2>&1 &
# 注意：这里**不能**开 `set -u`。Isaac Sim 的 setup_conda_env.sh（conda activate 时被
# activate.d 调用）第 19 行无保护地引用 $ZSH_VERSION，在 bash 下该变量不存在，
# set -u 会直接把脚本打死，第二阶段静默不启动。
PHASE1_PID="${1:?用法: chain_baseline.sh <第一阶段PID>}"
cd /home/wangyun/intern/ZGYD/bpx || exit 1

echo "[chain] $(date '+%F %T') 等待第一阶段 PID ${PHASE1_PID} 结束…"
while kill -0 "${PHASE1_PID}" 2>/dev/null; do sleep 30; done
echo "[chain] $(date '+%F %T') 第一阶段已结束"

# 确认真的产出了 teacher，没有就不要盲目启动第二阶段
CKPT=$(ls -v logs/rsl_rl/bpx_pro_flat_baseline/*_teacher/model_*.pt 2>/dev/null | tail -1)
if [ -z "${CKPT}" ]; then
    echo "[chain] 中止：baseline 第一阶段没有产出 checkpoint，第二阶段没有 teacher 可用"
    exit 1
fi
echo "[chain] teacher checkpoint: ${CKPT}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma

echo "[chain] $(date '+%F %T') 启动 baseline 蒸馏（1500 迭代，与辨识组同预算）"
BPX_PRO_PARAMS_YAML=data/bpx_pro_baseline_params.yaml \
python -u scripts/train.py --headless --num_envs 4096 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --experiment-name bpx_pro_flat_baseline --run-name student \
    > logs/train_student_baseline.log 2>&1
RC=$?
echo "[chain] $(date '+%F %T') baseline 蒸馏结束，退出码 ${RC}"
exit ${RC}
