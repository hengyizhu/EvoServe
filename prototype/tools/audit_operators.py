"""CPU-only provenance check; no model weights are copied into Git."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoserve.operators import load_operator,sha256
from evoserve.io import save
p=argparse.ArgumentParser();p.add_argument('--experiments',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=Path(a.experiments)
sources={
 3:('drawer','pi05_night_20261110/training/p01_drawer_s08_action_d100',.86),
 5:('book','pi05_extended_20261109/training/loadscan_lora_book_s08_4000',.98),
 9:('microwave','pi05_night_20261110/training/p02_microwave_s08_action_d100',.60),
 1:('X','pi05_adaptive_validate_20261111/training/new_X_task1_action_s08_train',.70),
 6:('camera','pi05_followon_20261112/B/training/new_camera_action_s08',.84)}
records={};expected=None
for eid,(name,folder,historical) in sources.items():
 path=r/folder/'ema_4000.pt';parameters,ckpt=load_operator(path,expected,eid);expected=set(parameters)
 config_path=r/folder/'config.json';config=json.loads(config_path.read_text()) if config_path.exists() else {}
 split_path=r/folder.split('/training/')[0]/'splits'/Path(config['split']).name
 split=json.loads(split_path.read_text()) if split_path.exists() else {}
 episodes=config.get('adaptation_episodes',len(split['target_train']) if 'target_train' in split else None)
 fraction=config.get('fraction',1 if episodes and 'target_train' in split else None)
 assert fraction==1,('operator is not full adaptation-data training',name)
 records[str(eid)]=dict(name=name,task_id=eid,checkpoint=str(path),sha256=sha256(path),step=ckpt['step'],weights_kind=ckpt['weights_kind'],trainable_parameters=sum(x.numel() for x in parameters.values()),family=ckpt['family'],parent_revision=ckpt['parent']['revision'],frozen_visual_verified=True,ignored_zero_effect_visual_parameters=len(ckpt['parameters'])-len(parameters),adaptation_data_fraction=fraction,adaptation_episodes=episodes,source_split=str(split_path),source_config=str(config_path),perturb=ckpt['perturb'],historical_success_rate=historical,historical_denoising_steps=5,current_operator_denoising_steps=10)
 print(name,ckpt['step'],records[str(eid)]['sha256'])
save(a.out,dict(operators=records,stagnation_distance_m=.002,stagnation_control_steps=20,time_fraction=.7,source='simulated correction using preexisting full-adaptation-data action experts',reinforcement_learning_enabled=False))
