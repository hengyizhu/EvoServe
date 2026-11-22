"""Read-only inventory of neighboring GPU utilization throughout latency runs."""
import csv
import subprocess
import threading
import time

class ExternalLoad:
    def __init__(self, out):
        self.path = out / 'external_gpu_load.csv'
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        with self.path.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['unix_time','gpu','util_pct','used_MiB','power_W'])
            while not self.stop.is_set():
                now = time.time()
                result = subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                for row in csv.reader(result.stdout.splitlines(),skipinitialspace=True):
                    if len(row)==4 and row[0] in ('2','3'):
                        writer.writerow([now,*row])
                        self.rows.append((now,*row))
                f.flush()
                self.stop.wait(5)

    def summary(self, after):
        readings = [r for r in self.rows if r[0]>=after]
        parts=[]
        for gpu in ('2','3'):
            v=[r for r in readings if r[1]==gpu]
            util=[float(r[2]) for r in v if r[2].isdigit()]
            mem=[int(r[3]) for r in v if r[3].isdigit()]
            if util and mem:
                parts.append(f'GPU {gpu}: {sum(util)/len(util):.1f}% avg, {max(mem)} MiB peak (n={len(util)})')
        return ', '.join(parts) or 'No GPU 2/3 samples'

    def close(self):
        self.stop.set()
        self.thread.join()
