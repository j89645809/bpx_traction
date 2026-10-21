#!/bin/bash
# Sprint4：sprint3 + 足端离地高度奖励（目标 ~7 cm）+ 倒车扩到 -2.0 m/s。
# 只训辨识参数这一组。
#
#   nohup systemd-inhibit --what=sleep --why="bpx sprint4 training" \
#       bash scripts/chain_sprint4.sh > logs/chain_sprint4.log 2>&1 &
#
# 不要加 set -u：Isaac Sim 的 setup_conda_env.sh 无保护引用 $ZSH_VERSION，会静默打死脚本。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda activate 失败"; exit 1; }

TASK=Isaac-Velocity-Flat-BpxPro-Sprint4-v0
EXP=bpx_pro_sprint4
YAML=data/bpx_pro_training_params.yaml

echo "[chain] $(date '+%F %T') ${EXP} / teacher 微调 (从 sprint3 种子续训 800 迭代)"
BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/train.py --headless \
    --task "${TASK}" --num_envs 4096 --max_iterations 800 \
    --resume --load-run ".*_s3seed_teacher" \
    --experiment-name "${EXP}" --run-name teacher > logs/train_teacher_sprint4.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') teacher 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20

if [ -z "$(ls -v logs/rsl_rl/${EXP}/*_teacher/model_*.pt 2>/dev/null | tail -1)" ]; then
    echo "[chain] 中止：没有 teacher checkpoint"; exit 1
fi

echo "[chain] $(date '+%F %T') ${EXP} / student (1500 迭代)"
BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/train.py --headless \
    --task "${TASK}" --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --experiment-name "${EXP}" --run-name student > logs/train_student_sprint4.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') student 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20

BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint4-Play-v0 --experiment-name "${EXP}" \
    > logs/chain_sprint4_export.log 2>&1
echo "[chain] $(date '+%F %T') 导出退出码 $?  全部完成"
