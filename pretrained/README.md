# Pretrained checkpoints

Best-seed (lowest mean BLER over its km sweep, out of 5 seeds) checkpoints for the
`non_photonic_activations_sigmoid` activation profile (relu everywhere, sigmoid transmitter
output), n=40, trained/evaluated over two fiber-length ranges: `10-60km` and `20-60km`.

Four methods per range:

| Folder | Method |
|---|---|
| `unconditioned` | Unconditioned baseline (no length conditioning) |
| `karanov_baseline` | One-hot length encoding, beta-only FiLM receiver, mlp=1, preact |
| `proposed_receiver_only` | Scalar length encoding, gamma-only FiLM receiver, mlp=2, preact |
| `proposed_receiver_transmitter` | Same as above, FiLM applied to receiver AND transmitter |

Each checkpoint folder contains `transmitter.pt`, `channel.pt`, `receiver.pt`, `parameters.pt`,
`parameters.json`. `parameters.pt` is the source of truth for reconstructing the exact model
architecture (including the transmitter/receiver activation functions) -- `inference.py` reads
it directly and does not need any of these CLI flags re-supplied.

## Loading and evaluating

From the repo root:

```bash
python inference.py --gpu_id 0 \
  --checkpoint_dir pretrained/10-60km/proposed_receiver_transmitter \
  --eval_kms 10,20,30,40,50,60 \
  --val_size 1000000 \
  --output_csv eval_out.csv
```

Swap `--checkpoint_dir` for any of the 8 folders (`pretrained/{10-60km,20-60km}/{unconditioned,
karanov_baseline,proposed_receiver_only,proposed_receiver_transmitter}`). `--eval_kms` should stay
within the folder's training range (10-60 or 20-60) for meaningful results.

## Figures

`figures/` has the mean-over-5-seeds and best-seed BLER-vs-km comparison plots (across all 4
methods) for both ranges, under this same activation profile.
