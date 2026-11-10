# Curated A6000 experiment summaries

These three files are byte-for-byte copies of the requested local summaries. No raw logs or per-step/per-request records are included.

- [CUDA Graph inference](pi05_optimized_20261107/optimized_inference.csv), from `/home/hengyi/pi05_optimized_20261107/optimized_inference.csv`.
- [Microbatch-4 training breakdown](pi05_optimized_20261107/training_mb4.csv), from `/home/hengyi/pi05_optimized_20261107/training_mb4.csv`.
- [A0/A1/A2 half-page report](pi05_a1_baseline_20261108/半页报告.md), from `/home/hengyi/pi05_a1_baseline_20261108/半页报告.md`.

The A0/A1/A2 report retains its original references to local raw records and `transfer_distributions.json`; those artifacts are not uploaded. Its co-running A1 measurement starts with an empty cache and spans three 60-second windows, so the 8.34× A2/A1 ratio is not a warmed steady-state comparison.

## Completed 2000-step comparison and 4000-step curves (2026-11-09)

- [Full / LoRA three-seed comparison](pi05_extended_20261109/comparison_2000_three_seeds.csv): counts extracted from decisions.json; 50 episodes per task.
- [Learning curves](pi05_extended_20261109/learning_curves.csv): three distinct tasks plus the old drawer-shift appendix.
- [Summary with confidence intervals](pi05_extended_20261109/汇总报告.md): local artifact links are retained; raw logs and checkpoints are excluded.
