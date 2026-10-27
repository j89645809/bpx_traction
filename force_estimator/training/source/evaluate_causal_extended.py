from direction_balance_common import *
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import hashlib,onnxruntime as ort
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends(allow_tf32=False,deterministic=True);torch.set_num_threads(2)
R=Path('logs/causal_v3_extended');O=Path('logs/rsl_rl/go2_velocity/causal_v3_extended')
import time
while not Path('logs/rsl_rl/go2_velocity/causal_v3_extended/selection.json').exists():time.sleep(10)
selection=json.load(open(O/'selection.json'));ck=O/(selection['selected']+'.pt')
results={};audits={};events={};parity={};response={}
def response_metrics(d):
 out={}
 for name,t0 in [('release',1601),('reverse',3001)]:
  p=d.pred[t0:t0+100]*80;y=d.y[t0:t0+100]*80
  mask=np.linalg.norm(d.y[t0-1]*80,axis=-1)>=2
  err=np.linalg.norm(p-y,axis=-1);pm=np.linalg.norm(p,axis=-1);ym=np.linalg.norm(y,axis=-1)
  angle=np.degrees(np.arccos(np.clip((p*y).sum(-1)/np.maximum(pm*ym,1e-12),-1,1)))
  ok=err<=1.5
  if name=='reverse':ok&=angle<=10
  latency=[]
  for e in np.where(mask)[0]:
   hits=[k for k in range(91) if ok[k:k+10,e].all()]
   latency.append(hits[0]*.02 if hits else None)
  finite=[v for v in latency if v is not None]
  out[name]={'eligible_envs':int(mask.sum()),'settled_within2s':len(finite),'censored':len(latency)-len(finite),
   'median_settling_s_among_settled':float(np.median(finite)) if finite else None,
   'settling_s':latency,'vector_rmse_by_step_N':np.sqrt((err[:,mask]**2).mean(1)).tolist(),
   'angle_p95_by_step_deg':np.percentile(angle[:,mask],95,axis=1).tolist() if name=='reverse' else None,
   'criterion':'vector error<=1.5N, reverse also angle<=10deg, sustained10frames; all event frames incl resets'}
 return out
for name in ['start','selected']:
 for seed in [29701,29702]:
  d=Data(seed,R/f'{name}_{seed}.npz');p=d.pred.reshape(-1,3)[d.ids]
  results.setdefault(name,{})[str(seed)]=breakdown(p,d);audits[f'{name}_{seed}']=d.audit
  response[f'{name}_{seed}']=response_metrics(d)
  event=json.load(open(R/f'{name}_{seed}.events.json'))
  events[f'{name}_{seed}']={'total':len(event),'terms':dict(Counter(k for row in event for k,v in row['terms'].items() if v)),'unique_envs':len(set(row['env'] for row in event))}
  w=SummaryWriter(str(O/f'blind_{name}_{seed}'))
  for g,row in results[name][str(seed)].items():
   for key in ['mag_mae','mag_rmse','angle_mean','angle_p95']:
    if row.get(key) is not None:w.add_scalar(f'acceptance/{g}/{key}',row[key],0)
   if row['n']:w.add_scalar(f'acceptance/{g}/fz_rmse',row['component_rmse'][2],0)
  w.add_scalar('acceptance/resets',len(event),0);w.close()
  reference=np.load(R/f'start_{seed}.npz');assert np.array_equal(d.applied,reference['applied'])
  if name=='selected':
   meta=np.load(R/f'{name}_{seed}.npz');m=model(ck).eval();configure_torch_backends(allow_tf32=False,deterministic=True)
   with torch.no_grad():replay=np.stack([m(torch.from_numpy(h).cuda()).cpu().numpy() for h in meta['history_samples']])
   error=float(abs(replay-meta['prediction_samples']).max())
   parity[str(seed)]={'online_replay_max_normalized':error,'pass':error<1e-5,'max_N':80*error}
   configure_torch_backends(allow_tf32=False)
# CPU export agreement on actual online histories, no model selection from blind metrics.
m=model(ck).cpu().eval();sample=np.load(R/'selected_29701.npz')['history_samples'][::10].reshape(-1,25,82)
torch.onnx.export(m,torch.from_numpy(sample[:1]),str(O/'selected.onnx'),input_names=['history'],output_names=['force'],dynamic_axes={'history':{0:'batch'},'force':{0:'batch'}},opset_version=17,dynamo=False)
session=ort.InferenceSession(str(O/'selected.onnx'),providers=['CPUExecutionProvider'])
with torch.no_grad():a=m(torch.from_numpy(sample)).numpy()
b=session.run(None,{'history':sample})[0];error=float(abs(a-b).max())
parity['onnx']={'max_normalized':error,'max_N':80*error,'pass':error<1e-5,'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}
diagnosis={}
for seed in [29701,29702]:
 for name in ['start','selected','oracle']:
  ee=json.load(open(R/f'{name}_{seed}.events.json'))
  diagnosis[f'{name}_{seed}']={'total':len(ee),'terms':dict(Counter(k for r in ee for k,v in r['terms'].items() if v)),'by_env':dict(Counter(r['env'] for r in ee)),'by_scene':dict(Counter(r['step']//1200 for r in ee))}
  side=json.load(open(R/f'{name}_{seed}.json'))
  assert side['physics_audit']['substeps']==24000 and side['physics_audit']['max_wrench_error_N']<1e-6
  audits[f'{name}_{seed}/physics']=side
json.dump({'selection':selection,'results':results,'audits':audits,'events':events,'parity':parity,'diagnosis':diagnosis,'force_plans_exactly_matched':True,'response':response},open(O/'acceptance.json','w'),indent=2)
tb=Path('logs/rsl_rl/go2_velocity/force_estimator_ft_20261014_v2/tensorboard')
for d in O.iterdir():
 if d.is_dir() and list(d.glob('events.out.tfevents.*')):
  link=tb/('causal_v3_extended_'+d.name)
  if not link.exists():link.symlink_to(d.resolve(),target_is_directory=True)
print('EVALUATION_COMPLETE',parity,diagnosis,flush=True)
