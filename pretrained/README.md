# Pretrained checkpoints

Best-seed (lowest mean BLER over its km sweep, out of 5 seeds) checkpoints for the
`non_photonic_activations_sigmoid` activation profile (relu everywhere, sigmoid transmitter
output), n=40, in eight sets:

| Set | Train/eval range | AWGN noise model |
|---|---|---|
| `10-60km` | 10-60 km | previous σ table (20-80 km), single least-squares log-linear fit |
| `20-60km` | 20-60 km | previous σ table (20-80 km), single least-squares log-linear fit |
| `20-50km-sigmav2` | 20-50 km | measured σ table, exact piecewise log-linear interpolation (below) |
| `20-50km-fixedsigma` | 20-50 km | constant σ = 0.015 at every length (noise does not grow with km) |
| `10-60km-new` | 10-60 km | constant σ = 0.015 at every length (same noise model as `20-50km-fixedsigma`) |
| `10-60km-sigmav2-x1.5` | 10-60 km | measured σ table with every point ×1.5 (+50%) |
| `10-60km-sigmav2-x2.0` | 10-60 km | measured σ table with every point ×2.0 (+100%) |
| `10-60km-sigmav2-x2.5` | 10-60 km | measured σ table with every point ×2.5 (+150%) |

Measured σ table used by `20-50km-sigmav2`: km [25, 29, 36, 40, 47] → σ [0.0273, 0.0329, 0.0476,
0.0554, 0.1058], interpolated linearly in log10(σ) between points and extended by the end segments
outside 25-47 km (σ ≈ 0.0216 at 20 km, 0.1396 at 50 km).

The `10-60km-sigmav2-x*` sets multiply every table point by the factor, which scales the whole σ(km) curve,
extensions included. Unscaled, the curve reaches σ ≈ 0.0136 at 10 km and 0.352 at 60 km, so at the far end of
10-60 km these sets are very noisy (×1.5: 0.020 to 0.53; ×2.0: 0.027 to 0.70; ×2.5: 0.034 to 0.88).

Four methods per set:

| Folder | Method |
|---|---|
| `unconditioned` | Unconditioned baseline (no length conditioning) |
| `karanov_baseline` | One-hot length encoding, beta-only FiLM receiver, mlp=1, preact |
| `proposed_receiver_only` | Scalar length encoding, gamma-only FiLM receiver, mlp=2, preact |
| `proposed_receiver_transmitter` | Same as above, FiLM applied to receiver AND transmitter |

All sets use the identical configuration for each method (same flags, 20k iterations, CE
loss); only the range and the noise model differ.

Seed picked per method:

| Set | unconditioned | karanov_baseline | proposed_receiver_only | proposed_receiver_transmitter |
|---|---|---|---|---|
| `10-60km` | 1 | 1 | 2 | 4 |
| `20-60km` | 4 | 1 | 2 | 1 |
| `20-50km-sigmav2` | 1 | 0 | 1 | 1 |
| `20-50km-fixedsigma` | 2 | 1 | 0 | 4 |
| `10-60km-new` | 3 | 1 | 2 | 3 |
| `10-60km-sigmav2-x1.5` | 4 | 0 | 0 | 4 |
| `10-60km-sigmav2-x2.0` | 3 | 1 | 0 | 3 |
| `10-60km-sigmav2-x2.5` | 2 | 2 | 1 | 3 |

Each checkpoint folder contains `transmitter.pt`, `channel.pt`, `receiver.pt`, `parameters.pt`,
`parameters.json`. `parameters.pt` is the source of truth for reconstructing the exact model
architecture (including the transmitter/receiver activation functions) and the channel's noise
model (`std_km_table` + `std_interp`) -- `inference.py` reads it directly and does not need any of
these CLI flags re-supplied.

## Loading and evaluating

From the repo root:

```bash
python inference.py --gpu_id 0 \
  --checkpoint_dir pretrained/10-60km/proposed_receiver_transmitter \
  --eval_kms 10,20,30,40,50,60 \
  --val_size 1000000 \
  --output_csv eval_out.csv
```

Swap `--checkpoint_dir` for any of the 32 folders (`pretrained/{10-60km,20-60km,20-50km-sigmav2,
20-50km-fixedsigma,10-60km-new,10-60km-sigmav2-x1.5,10-60km-sigmav2-x2.0,10-60km-sigmav2-x2.5}/{unconditioned,karanov_baseline,proposed_receiver_only,
proposed_receiver_transmitter}`). `--eval_kms` should stay within the folder's training range
(10-60, 20-60 or 20-50) for meaningful results.

## Figures

`figures/` has the mean-over-5-seeds (`avg_*`) and best-seed (`best_seed_*`) BLER-vs-km comparison
plots (across all 4 methods) for every set, under this same activation profile. Evaluation: every
2 km over the training range, val_size=1,000,000. Seeds whose mean BLER over the sweep marks them
as never having trained are left out of the average and named in its legend (Karanov seed 4 in both
20-50km sets, Karanov seed 3 in `10-60km-new` and in `10-60km-sigmav2-x2.0`).

## Other folders (single models, not best-of-5)

| Folder | Contents |
|---|---|
| `10-60km-fixedsigma-0.04` | seed 1 (not the best seed), constant σ = 0.04: `karanov_baseline`, `proposed_receiver_transmitter` |
| `10-60km-fixedsigma-0.025` | seed 1, constant σ = 0.025, all 4 methods |
| `baseline-25km-0.04` | `unconditioned`, trained and evaluated at 25 km only, constant σ = 0.04, seed 1 |
| `25km-ablation` | 25 km-only models at σ = 0.025: ER / SNR ablations and two models trained with the eakirtas code; see its README |
