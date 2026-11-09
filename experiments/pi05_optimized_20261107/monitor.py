import pathlib,time,json,importlib,datetime,zoneinfo,traceback
import report
R=pathlib.Path('/home/hengyi/pi05_optimized_20261107')
while True:
 try:
  n=importlib.reload(report).main();states={str(p.relative_to(R)):report.read(p) for p in [R/'inference/status.json',R/'training_mb4/status.json',R/'matrix/action/status.json',R/'matrix/action_vision_lora/status.json',R/'matrix_replicates/action/status.json',R/'matrix_replicates/action_vision_lora/status.json']};done=all(x and x.get('state') in ['complete','finished','failed','blocked_prerequisite'] for x in states.values());d={'updated':datetime.datetime.now(zoneinfo.ZoneInfo('America/New_York')).isoformat(),'conditions_complete':n,'conditions_target':76,'states':states,'all_terminal':done};tmp=R/'status.tmp';tmp.write_text(json.dumps(d,indent=2));tmp.replace(R/'status.json')
  if done:break
 except Exception:print(traceback.format_exc(),flush=True)
 time.sleep(15)
