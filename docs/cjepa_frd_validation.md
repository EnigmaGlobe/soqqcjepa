**C-JEPA FRD — Validation (plain-English, step-by-step)**

Purpose
- Explain, simply and without code, how we validate a trained C-JEPA model and what the outputs mean.

Overview (one sentence)
- Validation runs a saved checkpoint on held-out slot data, compares predicted future slot vectors to the true future, runs a few simple intervention probes, and writes numeric and example-artifact outputs.

Inputs you must provide
- Staged slot pickles under `checkpoints/validation_staged/` (examples: `stage_0_slots.pkl`, `stage_1_slots.pkl`).
- A trained checkpoint file to evaluate (example used in this project: `C:\Users\infra\.stable_worldmodel\local_run_bs128_ep50_weights.ckpt`).

Step-by-step validation process (what we do and why)
1) Load: the validator loads the chosen checkpoint and the staged slot pickles.
2) Build inputs: for each sample it assembles the history of slot vectors and the aligned action/proprio inputs.
3) Predict: the model produces future slot vectors from each input history.
4) Measure: compute average squared error between predictions and ground truth (MSE) and aggregate into headline metrics.
5) Probe: run simple counterfactuals (replace actions, ablate slots) and measure how predictions change to test action sensitivity.
6) Save: write numeric results and example artifacts to `outputs/` (common files: `outputs/val_results.json`, `outputs/tmp_val_examples.npz`, images under `outputs/val_examples/` and `outputs/val_error_maps/`).

What you will find in `outputs/` (plain)
- `val_results.json`: headline numbers (MSE, probe deltas).
- `tmp_val_examples.npz`: small saved arrays of `preds` and `targs` for reproducible examples.
- PNGs in `outputs/val_examples/` and `outputs/val_error_maps/`: visual examples and error heatmaps.

One-line summary you can copy
- "Validation ran on `<checkpoint>` using staged pickles in `checkpoints/validation_staged/`. Results written to `outputs/` (metrics + example images)."

End of FRD — validation-focused, non-technical

How we calculate MSE, RMSE, and NRMSE (plain language)
- MSE (Mean Squared Error): take the difference between each predicted slot vector and the true slot vector, square those differences, and average them across examples, timesteps, slots, and vector dimensions. In short: average of squared errors.
- RMSE (Root Mean Squared Error): the square root of MSE; gives error in the same units as slot vectors.
- NRMSE (Normalized RMSE): divide RMSE by the typical variability of the target (standard deviation) so the number is scale-free and comparable across datasets.

Simple formula view (optional)
- MSE: $\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$
- RMSE: $\mathrm{RMSE} = \sqrt{\mathrm{MSE}}$
- NRMSE: $\mathrm{NRMSE} = \mathrm{RMSE} / \sigma_{y}$

How to judge whether a value is "good" or "bad" (rules of thumb)
- NRMSE &lt; 0.5 — good: model error is much smaller than typical variation in the data.
- 0.5 ≤ NRMSE ≤ 1.0 — borderline: model captures some structure but leaves substantial unexplained variability.
- NRMSE &gt; 1.0 — poor: model is no better than (or worse than) predicting the mean; inspect data alignment or normalization.

Practical checks to interpret metrics
1. Always report NRMSE alongside raw MSE so scale differences do not mislead reviewers.
2. Compare model MSE to simple baselines: zero prediction and per-slot mean — if the model doesn't beat the mean, there's an issue.
3. Look at per-slot NRMSE and the error heatmap (slots × timesteps) to find whether a few slots or timesteps drive poor performance.

Where this is computed in the repo (for completeness)
- The validator uses an averaged MSE internally (implemented as `F.mse_loss(pred, tgt)` in the validation code). A helper script `scripts/compute_nrmse_from_npz.py` computes per-slot RMSE/NRMSE from saved `preds` and `targs` in `outputs/tmp_val_examples.npz`.

End of FRD — validation metrics explanation
Verification steps I run
- Print loader batch shapes to confirm `pixels_embed` is `(B, T, S, D)`.
- Overfit single-episode test: confirm training MSE falls near zero when training on one episode.
- Check normalization: if comparing validation to training MSE, re-normalize validation with training stats.

- Data formats:
  - Slot pickle: `{ 'train': {...}, 'val': {...} }` with values `ndarray (T, S, D)`.
  - Action/proprio pickles: per-split dicts mapping video_id → `(T, A)` and `(T, P)`.
- Tensor shapes and concrete config values (from `configs/config_train_causal_pusht_slot.yaml`):
  - Slots: `(T, S, D)` = `(T, 4, 128)`.
  - Batched input: `pixels_embed` → `(B, T_hist, S, D)` with `T_hist=5`.
  - Predictions: `(B, T_pred, S, D)` with `T_pred=3`.
  - Actions: `(B, T, A)` with `A=2`. Proprio: `(B, T, P)` with `P=4`.
Technical details (compact)