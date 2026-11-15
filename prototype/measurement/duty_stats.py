"""Time-weighted commanded duty; feedback state persists across warmup."""
from pathlib import Path
import json
import statistics


def mean_duty(out, result):
    out = Path(out)
    cfg = json.loads((out / 'measurement_schedule.json').read_text())
    begin = cfg['origin_ns']
    end = begin + round(result['duration_s'] * 1e9)
    means = []
    for pair in out.glob('pair*'):
        if not pair.is_dir():
            continue
        events = []
        for phase in ['warmup', 'measurement']:
            path = pair / phase / 'controller.jsonl'
            if path.exists():
                events.extend(json.loads(line) for line in path.read_text().splitlines())
        if not events:
            continue
        value, previous, area = 1.0, begin, 0.0
        for row in sorted(events, key=lambda r: r['observed_ns']):
            observed = row['observed_ns']
            if observed <= begin:
                value = row['duty_after']
                continue
            if observed >= end:
                break
            area += (observed - previous) * value
            previous, value = observed, row['duty_after']
        area += (end - previous) * value
        means.append(area / (end - begin))
    return statistics.mean(means) if means else None
