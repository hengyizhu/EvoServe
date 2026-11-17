"""Fixed causal rule. Signed100-step gain; strict thresholds;50-step checks."""
import math
def decision(rows,skip_initial=False):
 d={int(x['step']):float(x['representation_cosine_distance_parent']) for x in rows}
 assert all(math.isfinite(x) and x>=0 for x in d.values())
 checks=[];streak=0
 for t in sorted(d):
  if t<100:continue
  if t==100:
   fire=d[t]<.03 and not skip_initial;checks.append(dict(step=t,distance=d[t],reason='low_initial_distance',threshold=.03,triggered=fire,consecutive=0))
   if fire:return checks[-1],checks
  elif t%50==0:
   assert t-100 in d,(t,sorted(d))
   gain=d[t]-d[t-100];flat=gain<.1*d[t];streak=streak+1 if flat else 0
   checks.append(dict(step=t,distance=d[t],previous_100_distance=d[t-100],gain_100=gain,threshold=.1*d[t],below_threshold=flat,consecutive=streak,reason='two_consecutive_plateau_checks',triggered=streak>=2))
   if streak>=2:return checks[-1],checks
 return None,checks
