"""Find real, time-overlapping CUDA kernel intervals in separate profiler captures."""
import json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def union(intervals):
 out=[]
 for a,b in sorted(intervals):
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out

def intersection(a,b):
 i=j=0;total=0.
 while i<len(a) and j<len(b):
  total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<=b[j][1]:i+=1
  else:j+=1
 return total

def analyze():
 result={}
 for path in (ROOT/'gpu3_phases').glob('*_representative_trace.json'):
  data=json.loads(path.read_text());streams=collections.defaultdict(list)
  for e in data.get('traceEvents',[]):
   if e.get('cat')=='kernel' and e.get('ph')=='X' and e.get('dur',0)>0:
    args=e.get('args',{});key=str(args.get('device','?'))+':'+str(args.get('stream',e.get('tid')))
    streams[key].append([e['ts'],e['ts']+e['dur']])
  merged={k:union(v) for k,v in streams.items()};pairs=[]
  for i,a in enumerate(sorted(merged)):
   for b in sorted(merged)[i+1:]:
    if a.split(':')[0]==b.split(':')[0]:pairs.append({'stream_a':a,'stream_b':b,'overlapping_kernel_us':intersection(merged[a],merged[b])})
  result[path.name]={'source':'CUPTI GPU kernel timestamps; CPU enqueue overlap excluded','kernel_counts':{k:len(v) for k,v in streams.items()},'stream_pairs':pairs,'observed_kernel_overlap':any(p['overlapping_kernel_us']>0 for p in pairs),'scope':'Two separate representative pair captures per profile, not all 100 trials and not an end-to-end gain claim.'}
 if result:(ROOT/'trace_overlap.json').write_text(json.dumps(result,indent=2))
 return result
if __name__=='__main__':print(json.dumps(analyze(),indent=2))
