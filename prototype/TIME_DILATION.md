# Time-dilated and real-time on-policy runs

Use the machine-A Python environment and the measurement host's calibrated r32 controller table. Each serving GPU also hosts its training worker. The simulated operator is placed on the selected GPU with the most free memory at startup. All other GPUs are monitored read-only. Operators consume the serving request's VLM prefix. Distinct robots attach to camera and texture experts; larger expert counts create independent replicas of those tasks.

```bash
python prototype/tools/dilated_run.py --gpus 0 --k 3 --robots-per-gpu 4 --experts 2 --seconds 900 --out /var/tmp/dilation-smoke
python prototype/tools/dilated_run.py --gpus 0 1 --k 1 --robots-per-gpu 1 --experts 2 --out /var/tmp/realtime-system
```

A request is due every 250 × k ms and has the same deadline. Reset and admission waits are excluded. The physics control rate is 20 Hz. The ten-action fallback holds the last gripper command when the block is exhausted. Training uses the on-policy cache pool, relative arrival rate and timeout feedback; it has no configured steps/s ceiling. Pass a hardware-specific JSON with `--calibration` for future hardware.

Half-open measurement windows use nominal releases and count unanswered requests as timeouts. `system_5min.jsonl` gives per-GPU P99, SLO, expert steps/s, mean merged experts per executed update, commanded duty and GPU 2/3 utilization. `overall_slo.json` covers all complete five-minute windows. `activations.jsonl` identifies the first episode beginning on a promoted production version. `switch_latency.json` records the surrounding ±30 seconds of requests. `host_gpu_monitor.jsonl` records hardware and host conditions through the run. To finish a continuing run and flush its final summary, create `stop_requested` in its output directory.
