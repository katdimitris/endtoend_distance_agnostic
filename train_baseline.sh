
# Trains the unconditioned baseline: receiver_type=baseline, transmitter_type=baseline,
# so length is only ever seen by the channel physics, never by the networks -- trained
# with lengths sampled uniformly at random over the full 0-55km range each step.
#
# Defaults reproduce the exact config of the baseline checkpoint already committed in
# this repo (checkpoints/hardware_v2_diag_.../baseline_it=40000_..._s=2), so re-running
# with the same GPU_ID/SEED overwrites it with a fresh, identically-configured run.
#
# Usage: bash train_baseline.sh [gpu_id] [seed]

cd "$(dirname "$0")" || exit 1

GPU_ID=${1:-0}
SEED=${2:-2}

EXP_DIR=hardware_v2_diag
TRAIN_LENGTHS=0,55
TRAIN_ITER=40000

python train.py --gpu_id $GPU_ID --seed $SEED --exp_dir $EXP_DIR \
  --train_lengths $TRAIN_LENGTHS --train_iter $TRAIN_ITER --no_eval True \
  --receiver_type baseline --transmitter_type baseline
