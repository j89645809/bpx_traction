#!/bin/bash
# 续跑 sprint2_baseline：teacher 从 model_3500 续到 4200，再蒸馏 1500 + 导出。
# 整条链包在 systemd-inhibit 里由外层调用（阻止自动挂起）。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda activate 失败"; exit 1; }
EXP=bpx_pro_sprint2_baseline
YAML=data/bpx_pro_baseline_params.yaml
echo "[chain] $(date '+%F %T') teacher 续训 +700（3500 -> 4200）"
BPX_PRO_PARAMS_YAML=${YAML} python -u scripts/train.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-v0 --num_envs 4096 --max_iterations 700 \
    --experiment-name ${EXP} --run-name teacher --resume --load-run ".*_teacher" \
    > logs/train_teacher_s2b_cont.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') teacher 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
echo "[chain] $(date '+%F %T') student 1500 迭代"
BPX_PRO_PARAMS_YAML=${YAML} python -u scripts/train.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-v0 --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --experiment-name ${EXP} --run-name student > logs/train_student_s2b.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') student 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
BPX_PRO_PARAMS_YAML=${YAML} python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint2-Play-v0 --experiment-name ${EXP} \
    > logs/chain_s2b_export.log 2>&1
echo "[chain] $(date '+%F %T') 导出退出码 $?  baseline 完成"

# ---- 接 sprint3（蜘蛛腿修正，辨识参数）----
echo "[chain] $(date '+%F %T') sprint3 冒烟"
BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml python -u scripts/train.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint3-v0 --num_envs 64 --max_iterations 2 \
    --experiment-name sprint3_smoke --run-name t > logs/sprint3_smoke.log 2>&1
rc=$?; echo "[chain] sprint3 冒烟退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
rm -rf logs/rsl_rl/sprint3_smoke; sleep 20
echo "[chain] $(date '+%F %T') sprint3 teacher 4200"
BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml python -u scripts/train.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint3-v0 --num_envs 4096 --max_iterations 4200 \
    --experiment-name bpx_pro_sprint3 --run-name teacher > logs/train_teacher_s3.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') sprint3 teacher 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
echo "[chain] $(date '+%F %T') sprint3 student 1500"
BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml python -u scripts/train.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint3-v0 --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point \
    --experiment-name bpx_pro_sprint3 --run-name student > logs/train_student_s3.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') sprint3 student 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint3-Play-v0 --experiment-name bpx_pro_sprint3 \
    > logs/chain_s3_export.log 2>&1
echo "[chain] $(date '+%F %T') sprint3 导出退出码 $?  全部完成"
