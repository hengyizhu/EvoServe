# Curated A6000 experiment summaries

These three files are byte-for-byte copies of the requested local summaries. No raw logs or per-step/per-request records are included.

- [CUDA Graph inference](pi05_optimized_20261107/optimized_inference.csv), from `/home/hengyi/pi05_optimized_20261107/optimized_inference.csv`.
- [Microbatch-4 training breakdown](pi05_optimized_20261107/training_mb4.csv), from `/home/hengyi/pi05_optimized_20261107/training_mb4.csv`.
- [A0/A1/A2 half-page report](pi05_a1_baseline_20261108/半页报告.md), from `/home/hengyi/pi05_a1_baseline_20261108/半页报告.md`.

The A0/A1/A2 report retains its original references to local raw records and `transfer_distributions.json`; those artifacts are not uploaded. Its co-running A1 measurement starts with an empty cache and spans three 60-second windows, so the 8.34× A2/A1 ratio is not a warmed steady-state comparison.
