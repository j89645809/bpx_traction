#!/bin/bash
# 高速组（指令上限 vx 2.0 / vy 0.6 / yaw 1.5）的完整流水线，两组各跑两阶段。
# 顺序串行，任何一步失败就停，不往下走。
#
#   nohup bash scripts/chain_fast.sh > logs/chain_fast.log 2>&1 &
#
# 注意：**不要**加 set -u —— Isaac Sim 的 setup_conda_env.sh 第 19 行无保护引用
# $ZSH_VERSION，在 bash 下会把脚本静默打死（之前踩过）。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda activate 失败"; exit 1; }

TASK=Isaac-Velocity-Flat-BpxPro-Fast-v0
NENV=4096

run_stage () {   # run_stage <实验名> <参数yaml> <阶段> <迭代数> <日志>
    local exp="$1" yaml="$2" stage="$3" iters="$4" log="$5"
    local extra=""
    local run_name="teacher"
    if [ "${stage}" = "student" ]; then
        extra="--agent rsl_rl_distillation_cfg_entry_point"
        run_name="student"
        # 开蒸馏前确认 teacher 真的存在，否则直接失败而不是跑一个没有老师的蒸馏
        if [ -z "$(ls -v logs/rsl_rl/${exp}/*_teacher/model_*.pt 2>/dev/null | tail -1)" ]; then
            echo "[chain] 中止：${exp} 没有 teacher checkpoint"
            return 1
        fi
    fi
    echo "[chain] $(date '+%F %T') ${exp} / ${stage}  (${iters} 迭代)"
    BPX_PRO_PARAMS_YAML="${yaml}" python -u scripts/train.py --headless \
        --task "${TASK}" --num_envs "${NENV}" --max_iterations "${iters}" \
        --experiment-name "${exp}" --run-name "${run_name}" ${extra} > "${log}" 2>&1
    local rc=$?
    echo "[chain] $(date '+%F %T') ${exp} / ${stage} 退出码 ${rc}"
    # Isaac Sim 退出后要留点时间释放 GPU，否则下一个实例会在启动时段错误
    sleep 20
    return ${rc}
}

ID_YAML=data/bpx_pro_training_params.yaml
BL_YAML=data/bpx_pro_baseline_params.yaml

run_stage bpx_pro_fast          "${ID_YAML}" teacher 1200 logs/train_teacher_fast.log          || exit 1
run_stage bpx_pro_fast          "${ID_YAML}" student 1500 logs/train_student_fast.log          || exit 1
run_stage bpx_pro_fast_baseline "${BL_YAML}" teacher 1200 logs/train_teacher_fast_baseline.log || exit 1
run_stage bpx_pro_fast_baseline "${BL_YAML}" student 1500 logs/train_student_fast_baseline.log || exit 1

echo "[chain] $(date '+%F %T') 四个阶段全部完成，开始导出"
for exp in bpx_pro_fast bpx_pro_fast_baseline; do
    yaml="${ID_YAML}"; [ "${exp}" = "bpx_pro_fast_baseline" ] && yaml="${BL_YAML}"
    BPX_PRO_PARAMS_YAML="${yaml}" python -u scripts/export_onnx.py --headless \
        --task Isaac-Velocity-Flat-BpxPro-Fast-Play-v0 --experiment-name "${exp}" \
        >> logs/chain_fast_export.log 2>&1
    echo "[chain] ${exp} 导出退出码 $?"
    sleep 20
done
echo "[chain] $(date '+%F %T') 全部完成"
