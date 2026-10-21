#!/bin/bash
# 续跑被休眠搞死的 sprint2 蒸馏（从 model_600 继续到 1500），然后导出。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda activate 失败"; exit 1; }
echo "[chain] $(date '+%F %T') 续跑 sprint2 蒸馏（resume from .*_student 最新 checkpoint）"
BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml python -u scripts/train.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-v0 --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --experiment-name bpx_pro_sprint2 --run-name student --resume --load-run ".*_student" \
    > logs/train_student_sprint2_resume.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') 蒸馏退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-Play-v0 --experiment-name bpx_pro_sprint2 \
    > logs/chain_sprint2_export.log 2>&1
echo "[chain] $(date '+%F %T') 导出退出码 $?  全部完成"
