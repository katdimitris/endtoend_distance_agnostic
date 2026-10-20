# 25 km ablation

Single-model checkpoints trained and evaluated at **25 km only** (no length conditioning: plain transmitter and
receiver), seed 1, 20,000 optimisation steps, constant AWGN σ = 0.025. They test single changes to the channel or
the training code at one fixed length.

## Channel: identical for every model

| Parameter | Value |
|---|---|
| Samples per message `n` | 40 |
| Sample rate | 320 GSa/s, so 8 GBd message rate (6 bits per message, M = 64) |
| Tx and Rx low-pass filter | 400-tap windowed sinc, cut-off `fc` = 0.1 × 320 GSa/s = 32 GHz |
| Fibre | 25 km, D = 17 ps/nm/km at 1550 nm (dispersion only) |
| AWGN after detection | σ = 0.025 at every length (`std_interp = 'constant'`) |
| DAC / ADC noise | SNR 38 dB, unless the folder name says otherwise |
| Modulator extinction ratio | 20 dB on the field min-max range, unless the folder name says otherwise |
| Sequence | 11 messages, centre one decoded |

The two `kirtas` models were trained with the code of
[eakirtas/endtoend_plain](https://github.com/eakirtas/endtoend_plain) (commit `da223e5`, `v4_haetae` channel, which runs
the same operations as `channel.Channel_v3`). That repository's own defaults are n = 48 at 336 GSa/s (7 GBd), a single
40 km length and σ = 0.03936. **None of these were used**: they were set to the values in the table above so that
only the networks and the training procedure differ from the models trained here.

## Models

| Folder | Code | Tx activations (hidden, hidden, output) | Rx activations | Tx hidden width | Batch (messages) | Learning rate | ER / SNR (dB) |
|---|---|---|---|---|---|---|---|
| `sigma-0.025` | this repo | relu, relu, sigmoid | relu, relu | 64 | 1012 | 1e-3, cosine | 20 / 38 |
| `sigma-0.025-ER-13.1` | this repo | relu, relu, sigmoid | relu, relu | 64 | 1012 | 1e-3, cosine | **13.1** / 38 |
| `sigma-0.025-SNR-22.7` | this repo | relu, relu, sigmoid | relu, relu | 64 | 1012 | 1e-3, cosine | 20 / **22.7** |
| `sigma-0.025-kirtas-photonic` | eakirtas | **photonic sigmoid** ×3 | **photonic sigmoid** ×2 | 128 | 253 | 5e-4 → 2e-7, step | 20 / 38 |
| `sigma-0.025-kirtas-photonic-ER-13.1` | eakirtas | **photonic sigmoid** ×3 | **photonic sigmoid** ×2 | 128 | 253 | 5e-4 → 2e-7, step | **13.1** / 38 |
| `sigma-0.025-kirtas-non-photonic` | eakirtas | relu, relu, sigmoid | relu, relu | 128 | 253 | 5e-4 → 2e-7, step | 20 / 38 |

- **eakirtas training:** 1,000 epochs × 20 batches of 253 messages (23 sequences), freshly drawn every epoch. The
  learning rate steps down every 100 epochs (5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 2e-7) and Adam
  is re-created at every epoch. The weights were converted with `import_kirtas.py`.
- **This repo's training:** 20,000 steps of 1012 messages (92 sequences), Adam with a cosine schedule.
- **Photonic sigmoid:** output 0.06 to 1.005, about 30× steeper than a standard sigmoid. The transmitter's raw
  samples are mostly at the rails, but they alternate faster than the 32 GHz transmitter filter passes: after the
  filter (the signal that drives the modulator) only about 11% of the drive is at the rails, so it is not a
  two-level signal.
- **Evaluation batch:** the modulator's min-max scaling uses whole-batch statistics, so the `kirtas` checkpoints
  store `eval_batch = 253` and `inference.py` uses it (an explicit `--batch_size` overrides it).

`parameters.pt` in each folder is the complete record; `parameters.json` is the same in readable form.

## Simulated performance at 25 km

| Folder | BLER (message error rate) | Tx output samples at the rails |
|---|---|---|
| `sigma-0.025` | 1.7e-6 (2.3M messages) | 10% |
| `sigma-0.025-ER-13.1` | 2.6e-6 (2.3M messages) | – |
| `sigma-0.025-SNR-22.7` | BER 8.1e-2 | – |
| `sigma-0.025-kirtas-photonic` | 7.8e-6 (2.3M messages) | 84% |
| `sigma-0.025-kirtas-photonic-ER-13.1` | 4.3e-6 (2.3M messages) | 84% |
| `sigma-0.025-kirtas-non-photonic` | 4.9e-4 (2.3M messages) | 29% |

"At the rails" = raw transmitter output within 0.1 of the lowest or highest value the output activation can produce
(before the 32 GHz transmitter filter).

## Loading

```bash
python inference.py --checkpoint_dir pretrained/25km-ablation/sigma-0.025-kirtas-photonic --eval_kms 25 --seed 1
```
