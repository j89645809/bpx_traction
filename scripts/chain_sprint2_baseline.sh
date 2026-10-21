#!/bin/bash
# sprint2 的厂家参数对照组：配置与 sprint2 完全相同（含站立修复 + dq 滞后建模），
# 只换参数 yaml。teacher 给与 sprint2 同等的 4200 迭代预算（公平对照）。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda activate 失败"; exit 1; }
EXP=bpx_pro_sprint2_baseline
YAML=data/bpx_pro_baseline_params.yaml
echo "[chain] $(date '+%F %T') ${EXP} / teacher (4200 迭代，与 sprint2 同预算)"
BPX_PRO_PARAMS_YAML=${YAML} python -u scripts/train.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-v0 --num_envs 4096 --max_iterations 4200 \
    --experiment-name ${EXP} --run-name teacher > logs/train_teacher_s2b.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') teacher 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
[ -z "$(ls -v logs/rsl_rl/${EXP}/*_teacher/model_*.pt 2>/dev/null | tail -1)" ] && { echo "[chain] 中止：无 teacher checkpoint"; exit 1; }
echo "[chain] $(date '+%F %T') ${EXP} / student (1500 迭代)"
BPX_PRO_PARAMS_YAML=${YAML} python -u scripts/train.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-v0 --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --experiment-name ${EXP} --run-name student > logs/train_student_s2b.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') student 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
BPX_PRO_PARAMS_YAML=${YAML} python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-Play-v0 --experiment-name ${EXP} \
    > logs/chain_s2b_export.log 2>&1
echo "[chain] $(date '+%F %T') 导出退出码 $?  全部完成"
