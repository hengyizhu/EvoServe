"""Reconstruct frame counts and robot time from the archived observation windows."""
from pathlib import Path
import csv
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'raw/paired_validation_20261120/camera_curves.csv').open()))
with (root/'b3_camera_times.csv').open('w') as f:
    writer=None
    for group in ('sequential','fixed50'):
        data=[r for r in rows if r['group']==group]
        assert len(data)==18
        for scope,g in [('final_window',data[-1:]),('all_windows',data)]:
            frames=sum(int(r['frames']) for r in g)
            def counts(a,b):
                values=[float(r[a])*int(r[b]) for r in g]
                assert all(abs(v-round(v))<1e-6 for v in values)
                return sum(map(round,values))
            took=counts('takeover_fraction','frames')
            row=dict(group=group,scope=scope,frames=frames,takeover_frames=took,takeover_fraction=took/frames,
                     autonomous=counts('autonomous','production_episodes'),production_episodes=sum(int(r['production_episodes']) for r in g),
                     camera_robot_hours=frames/20/3600,mean_active_hours_per_camera_robot=frames/20/3600/4)
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow(row)
