# Re-trained receivers: `sigma-0.025-kirtas-photonic` on its lab capture

Receiver re-training (`retrain.py`) of [`pretrained/25km-ablation/sigma-0.025-kirtas-photonic`](../../../pretrained/25km-ablation/sigma-0.025-kirtas-photonic)
(eakirtas code, photonic sigmoid, 25 km only, σ = 0.025) on the lab capture of its transmitter at 25 km
(`2026_10_20_E2E_baseline_25km_kirtas_photonic_sigma_0025_auth.npy`, 79 loops × 25,600 messages; targets `targets[5:25605]`).
Only the receiver is trained; the transmitter and channel files are copies of the pretrained ones.

- **Validation:** the last 2,560 message positions of the pattern (10%), in all 79 loops (202,240 messages), held out
  from training; 5 positions on each side of the block are dropped from training as an ISI guard.
- **Recipe:** Adam, cosine learning-rate schedule; "pretrained" starts from the pretrained receiver, "random init"
  from a freshly initialised receiver with the same architecture.
- **Batch:** 92 messages, as for every other capture; `bs23` = `retrain.py`'s default for this checkpoint
  (its 253-message training batch / 11).

| Run | Iterations | lr | Batch | Start | Val BLER before | **Val BLER, best** | Val BER, best | Best iteration |
|---|---|---|---|---|---|---|---|---|
| `lr5e-04_it20000_bs23_val-block10pct_s0` | 20k | 5e-04 | 23 | pretrained | 0.371 | **4.27e-03** | 2.28e-03 | 19,200 |
| `lr5e-04_it20000_bs23_val-block10pct_scratch-baseline_s0` | 20k | 5e-04 | 23 | random init | 0.987 | **3.80e-03** | 2.05e-03 | 19,100 |
| `lr5e-04_it20000_bs92_val-block10pct_s0` | 20k | 5e-04 | 92 | pretrained | 0.371 | **3.53e-03** | 1.87e-03 | 18,300 |
| `lr5e-04_it20000_bs92_val-block10pct_scratch-baseline_s0` | 20k | 5e-04 | 92 | random init | 0.987 | **3.35e-03** | 1.82e-03 | 18,300 |
| `lr5e-04_it50000_bs92_val-block10pct_s0` | 50k | 5e-04 | 92 | pretrained | 0.371 | **3.17e-03** | 1.70e-03 | 46,900 |
| `lr5e-04_it50000_bs92_val-block10pct_scratch-baseline_s0` | 50k | 5e-04 | 92 | random init | 0.987 | **3.06e-03** | 1.65e-03 | 45,100 |
| `lr1e-03_it50000_bs92_val-block10pct_s0` | 50k | 1e-03 | 92 | pretrained | 0.371 | **3.31e-03** | 1.80e-03 | 45,300 |
| `lr1e-03_it50000_bs92_val-block10pct_scratch-baseline_s0` | 50k | 1e-03 | 92 | random init | 0.987 | **3.30e-03** | 1.77e-03 | 46,900 |

## Files per run

| File | Content |
|---|---|
| `receiver.pt` | best receiver by validation loss (same as `receiver_best.pt`) |
| `receiver_final.pt` | receiver after the last iteration |
| `transmitter.pt`, `channel.pt` | unchanged copies of the pretrained model, so the folder loads like a checkpoint |
| `parameters.pt` / `.json` | model parameters plus a `finetune` entry with the full re-training settings |
| `finetune_log.csv`, `finetune_curve.png` | loss, BER and BLER (training and validation) every 100 iterations |
