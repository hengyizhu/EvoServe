import os,json,pathlib,subprocess,hashlib,time
p=pathlib.Path(__file__).resolve().parent
python='/home/hengyi/vla_followup_20261020/venv/bin/python'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
assert not active.strip(),'GPU compute processes appeared; refusing to launch'
manifest={'created_client_date':'2026-11-04','gpus':raw,'source':'/home/hengyi/vla_followup_20261020','original_training':'/home/hengyi/vla_ladder_20261020','protocol':'protocol.txt','modules':{'L2':'action expert plus action input/output and time modules; original frozen VLM','L3':'L2 plus vision; language remains frozen, activation checkpointing enabled'},'precision':'original selective bf16; fp32 optimizer masters and moments','effective_batch':8,'microbatch':1,'optimizer':'AdamW lr=2.5e-5 betas=(0.9,0.95) eps=1e-8 weight_decay=1e-10 clip=1','production':'fixed parent checkpoint','candidate':'original respective tier resume.pt at step 1000, same reset per condition','control':{'denoising_steps':5,'horizon':10,'execute_steps':5,'control_hz':20,'action_coverage_ms':250},'compile':False,'threads':4,'cpu_affinity':{str(i):list(range(i*8,i*8+8)) for i in range(4)},'training_cache':'first 16 optimizer batches of original sample_sequence.json, 128 pretransformed microbatches cached on CPU, cyclic; identical cursor reset per policy','rng':'explicit independent per-input generators for serving noise, training noise, augmentation and Beta time; no global seed mutation','residency':'Both independent models plus initialized optimizer present for all S0-S4 modes','full_training_label':'L3 vision+action training; not full-model or LoRA','workers':[]}
for gpu,tier in [(0,'L2'),(1,'L2'),(2,'L3')]:
 env=os.environ.copy();env['VLA_GPU']=str(gpu)
 with (p/f'logs/gpu{gpu}.log').open('w') as log:
  process=subprocess.Popen([python,str(p/'replay.py'),'--tier',tier],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=p)
 manifest['workers'].append({'gpu':gpu,'tier':tier,'pid':process.pid})
manifest['sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('*.py')}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest['workers']))
