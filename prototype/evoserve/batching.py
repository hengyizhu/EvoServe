"""Bounded collection window; GPU execution still enforces per-request deadlines."""
import queue,time

def collect_batch(first, requests, maximum, wait_ms):
    items=[first];barrier=None
    until=time.monotonic()+wait_ms/1000
    while len(items)<maximum:
        remaining=until-time.monotonic()
        try:item=requests.get(timeout=max(0,remaining)) if remaining>0 else requests.get_nowait()
        except queue.Empty:break
        if item[0]['op']!='infer':barrier=item;break
        items.append(item)
    return items,barrier
