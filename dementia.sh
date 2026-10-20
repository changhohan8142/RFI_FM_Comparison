#!/usr/bin/env bash
# Training launch script — iterates over all adaptation strategies × backbones (3 × 7 = 21 configurations)
#
# Runs torchrun jobs sequentially for each (ft, arch) combination; a failed job is reported and the loop continues.
# Each job trains on the development dataset, keeps the checkpoint with the highest validation AUROC and then saves
# per-image predictions for the validation set, the test set and the future-incidence cohort (see trainer.py).
# GPU assignment and paths can be overridden via environment variables before calling this script.
#
# Hyperparameters per adaptation strategy (fixed a priori, identical across backbones; 8 GPUs):
#   linear  : lr 1e-3, batch size 32 per GPU (classification head only)
#   partial : lr 1e-4, batch size 8 per GPU, final transformer block + final norm (--ft_blks 1)
#   lora    : lr 1e-4, batch size 8 per GPU, rank-4 LoRA on query/value of every block (--ft_blks full --lora_rank 4)
#   all     : AdamW, weight decay 1e-6, mixed precision; up to 80 epochs, LR halved after 5 epochs without
#             improvement in validation AUROC, stop after 10 such epochs (trainer.py defaults)
#
# Batch sizes are per GPU, so the effective batch size changes with the number of GPUs.
# Override examples:
#   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" bash dementia.sh
#   DET_CSV=/data/dementia_detection.csv PRED_CSV=/data/dementia_prediction.csv ROOT_DIR=./ckpts bash dementia.sh


# -------------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------------
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}


# -------------------------------------------------------------------------
# Paths — update to match your environment
# -------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT=${SCRIPT:-${SCRIPT_DIR}/trainer.py}
DET_CSV=${DET_CSV:-/path/to/dementia_detection.csv}
PRED_CSV=${PRED_CSV:-/path/to/dementia_prediction.csv}
ROOT_DIR=${ROOT_DIR:-./ckpts}


# -------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------
NP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
WEIGHT_DECAY=1e-6

# Backbones / adaptation strategies (remove entries from the lists to skip them)
ARCHS=(
    "retfound"
    "retfound_dinov2"
    "mae"
    "openclip"
    "dinov2"
    "dinov3"
    "dinov3_vith16plus"
)

FT_METHODS=(
    "linear"
    "partial"
    "lora"
)


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------
for FT in "${FT_METHODS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do

        case "$FT" in
            linear)
                LR=1e-3;  BS=32
                EXTRA_ARGS=()
                ;;
            partial)
                LR=1e-4;  BS=8
                EXTRA_ARGS=(--ft_blks 1)
                ;;
            lora)
                LR=1e-4;  BS=8
                EXTRA_ARGS=(--ft_blks "full" --lora_rank 4)
                ;;
        esac

        echo "=============================================================="
        echo "Running: arch=${ARCH} | ft=${FT} | lr=${LR} | batch=${BS} x ${NP} GPUs ${EXTRA_ARGS[*]}"
        echo "=============================================================="

        torchrun --standalone --nproc_per_node="${NP}" \
            "${SCRIPT}" \
            --arch "${ARCH}" \
            --ft "${FT}" \
            --lr "${LR}" \
            --batch_size "${BS}" \
            --weight_decay "${WEIGHT_DECAY}" \
            --det_csv "${DET_CSV}" \
            --pred_csv "${PRED_CSV}" \
            --root_dir "${ROOT_DIR}" \
            "${EXTRA_ARGS[@]}" \
            --enable_amp \
            || echo "FAILED: arch=${ARCH} ft=${FT} — continuing with the next configuration"

    done
done

echo "All jobs finished."
