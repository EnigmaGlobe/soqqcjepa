**C-JEPA FRD — Metrics (plain-English, step-by-step)**

Purpose
- Explain what the validation numbers mean for the trained model and how we compute the key metrics in this repos.

Overview (one sentence)
- Metrics measure how close the model's predicted slot vectors are to the real future slot vectors; we compute MSE as the basic error, RMSE for interpretability, and NRMSE for scale-free comparisons.

Key concepts (plain)
- MSE (Mean Squared Error): average squared difference between predicted and true slot vectors — lower is better.
- RMSE: square root of MSE — gives error in the same units as slot vectors.
- NRMSE: RMSE divided by the target's typical variability (std) — lets you compare performance across datasets or scales.

Step-by-step: how metrics are produced here
1) Run validation: the validator runs the checkpoint on staged pickles and saves predictions and targets for a small set of examples (`outputs/tmp_val_examples.npz`).
2) Compute MSE: the validator computes MSE between predictions and targets and writes headline numbers to `outputs/val_results.json`.
3) Compute RMSE & NRMSE: for interpretability we take sqrt(MSE) → RMSE, and divide RMSE by the target std (per-slot or global) → NRMSE.
4) Produce diagnostics: we summarize per-slot RMSE/NRMSE and create visual heatmaps of error over slots × timesteps (saved under `outputs/val_error_maps/`).

What you will find in `outputs/` (plain)
- `val_results.json`: headline error numbers (MSE, probe deltas).
- `tmp_val_examples.npz`: saved `preds` and `targs` for reproducible checks.
- `nrmse_report.json` or CSV: per-slot RMSE and NRMSE summaries.
- `val_error_maps/`: PNGs visualizing per-slot × time errors.

How to explain metrics to a non-technical reviewer
- Headline: "NRMSE = 0.X" with one sentence: "This means model error is X times the typical variation in the data; lower is better." 
- Add one example prediction image and one heatmap so reviewers can see what errors look like in practice.

Short troubleshooting notes (plain)
- If raw MSE looks unexpectedly large: check whether validation used the same normalization metadata as training (see `checkpoints/local_action_meta.pkl`).
- If specific slots have much higher NRMSE: those slots may need more data or model capacity.

End of FRD — metrics-focused, non-technical
