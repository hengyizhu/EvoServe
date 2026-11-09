import pathlib,time,json,subprocess,datetime,zoneinfo,traceback
import report,importlib
from report import read
R=pathlib.Path('/home/hengyi/pi05_measure_20261107')
while True:
 try:
  importlib.reload(report).main();states={str(p.relative_to(R)):read(p) for p in [R/'training/status.json',R/'inference/status.json',R/'interference/action/status.json',R/'interference/action_vision_lora/status.json']}
  result={'updated_at':datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).isoformat(),'states':states};done=all(x and x.get('state') in ['complete','failed','finished'] for x in states.values());result['all_terminal']=done
  result['failures']=[str(p.relative_to(R)) for p in (R/'interference').glob('*/*/failure.json')]
  tmp=R/'status.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(R/'status.json')
  if done:break
 except Exception:print(traceback.format_exc(),flush=True)
 time.sleep(15)
