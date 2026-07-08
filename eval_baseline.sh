
# Evaluates a trained checkpoint's BLER/CE loss over a km sweep via inference.py.
# Defaults to the unconditioned baseline checkpoint already committed in this repo
# (see checkpoints/hardware_v2_diag_.../baseline_it=40000_..._s=2).
#
# Usage: bash eval_baseline.sh [checkpoint_dir] [eval_kms] [val_size] [gpu_id]
#   e.g. bash eval_baseline.sh "" "0,10,20,30,40,50,55" 2000000 0

cd "$(dirname "$0")" || exit 1

DEFAULT_CKPT="checkpoints/hardware_v2_diag_train=(0, 55)_eval=(0, 55)/baseline_it=40000_vs=616000_opt=adam_lr=0.0005_bs=1012_s=2"

CKPT=${1:-$DEFAULT_CKPT}
KMS=${2:-"0,5,10,15,20,25,30,35,40,45,50,55"}
VAL_SIZE=${3:-1000000}
GPU_ID=${4:-0}

python inference.py --gpu_id $GPU_ID --checkpoint_dir "$CKPT" --eval_kms $KMS --val_size $VAL_SIZE
