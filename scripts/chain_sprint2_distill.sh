#!/bin/bash
# sprint2 蒸馏（从头，teacher = 最新续训完的 model_4197）+ 导出。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda activate 失败"; exit 1; }
echo "[chain] $(date '+%F %T') sprint2 蒸馏 1500 迭代（teacher 自动取最新 .*_teacher）"
BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml python -u scripts/train.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-v0 --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --experiment-name bpx_pro_sprint2 --run-name student \
    > logs/train_student_sprint2b.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') 蒸馏退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-Play-v0 --experiment-name bpx_pro_sprint2 \
    > logs/chain_sprint2_export.log 2>&1
echo "[chain] $(date '+%F %T') 导出退出码 $?  全部完成"
