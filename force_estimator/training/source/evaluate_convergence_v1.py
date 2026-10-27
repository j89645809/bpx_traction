from direction_balance_common import *
from train_causal_extended import CausalData
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from mjlab.utils.torch import configure_torch_backends
import onnxruntime as ort,hashlib
configure_torch_backends(allow_tf32=False,deterministic=True);torch.set_num_threads(2)
O=Path('logs/rsl_rl/go2_velocity/convergence_v1');R=Path('logs/convergence_v1');c=json.load(open(O/'convergence.json'))
assert c['converged'];ck=O/'best.pt';assert hashlib.sha256(ck.read_bytes()).hexdigest()==c['selected_sha256']
results={};audits={};events={};parity={};response={}
def responses(d):
 out={}
 for name,t0 in [('release',1601),('reverse',3001)]:
  p=d.pred[t0:t0+100]*80;y=d.y[t0:t0+100]*80;mask=np.linalg.norm(d.y[t0-1]*80,axis=-1)>=2
  error=np.linalg.norm(p-y,axis=-1);pm=np.linalg.norm(p,axis=-1);ym=np.linalg.norm(y,axis=-1)
  angle=np.degrees(np.arccos(np.clip((p*y).sum(-1)/np.maximum(pm*ym,1e-12),-1,1)))
  ok=error<=1.5
  if name=='reverse':ok&=angle<=10
  times=[]
  for e in np.where(mask)[0]:
   hit=[i for i in range(91) if ok[i:i+10,e].all()];times.append(hit[0]*.02 if hit else None)
  finite=[t for t in times if t is not None]
  out[name]={'eligible_envs':int(mask.sum()),'settled_within2s':len(finite),'censored':len(times)-len(finite),'median_settling_s_among_settled':float(np.median(finite)) if finite else None,'settling_s':times}
 return out
for name in ['start','selected']:
 path=Path('logs/rsl_rl/go2_velocity/causal_v3_extended/aligned_window_720.pt') if name=='start' else ck
 m=model(path).eval()
 for seed in [29901,29902]:
  key=f'{name}_{seed}';d=Data(seed,R/f'{key}.npz');p=d.pred.reshape(-1,3)[d.ids];rows=breakdown(p,d)
  results.setdefault(name,{})[str(seed)]=rows;response[key]=responses(d)
  side=json.load(open(R/f'{key}.json'));assert side['physics_audit']['substeps']==24000 and side['physics_audit']['max_wrench_error_N']<1e-6
  assert side['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest();audits[key]={'data':d.audit,'physical':side}
  ev=json.load(open(R/f'{key}.events.json'));events[key]={'total':len(ev),'terms':dict(Counter(k for r in ev for k,v in r['terms'].items() if v)),'by_scene':dict(Counter(r['step']//1200 for r in ev)),'by_env':dict(Counter(r['env'] for r in ev))}
  ref=np.load(R/f'start_{seed}.npz');assert np.array_equal(d.applied,ref['applied'])
  meta=np.load(R/f'{key}.npz');configure_torch_backends(allow_tf32=False,deterministic=True)
  with torch.no_grad():replay=np.stack([m(torch.from_numpy(h).cuda()).cpu().numpy() for h in meta['history_samples']])
  err=float(abs(replay-meta['prediction_samples']).max());parity[key]={'max_normalized':err,'max_N':err*80,'pass':err<1e-5}
  wr=SummaryWriter(str(O/f'blind_{key}'))
  for g,row in rows.items():
   for k in ['mag_mae','mag_rmse','angle_mean','angle_p95']:
    if row.get(k) is not None:wr.add_scalar(f'acceptance/{g}/{k}',row[k],c['selected_epoch'] if name=='selected' else 240)
   if row['n']:
    for i,axis in enumerate(['fx','fy','fz']):wr.add_scalar(f'acceptance/{g}/{axis}_rmse',row['component_rmse'][i],c['selected_epoch'] if name=='selected' else 240)
  wr.add_scalar('acceptance/terminations',len(ev),c['selected_epoch'] if name=='selected' else 240)
  for scene,rr in response[key].items():
   wr.add_scalar(f'acceptance/{scene}/unsettled_within2s',rr['censored'],c['selected_epoch'] if name=='selected' else 240)
   if rr['median_settling_s_among_settled'] is not None:wr.add_scalar(f'acceptance/{scene}/median_settling_s_among_settled',rr['median_settling_s_among_settled'],c['selected_epoch'] if name=='selected' else 240)
  wr.close()
# Full fixed validation: comparison only; selected checkpoint is already frozen.
v=CausalData(29201);full={}
for name,path in [('start',Path('logs/rsl_rl/go2_velocity/causal_v3_extended/aligned_window_720.pt')),('selected',ck)]:
 m=model(path);full[name]=breakdown(predict(m,v,v.ids),v)
json.dump(full,open(O/'full_validation.json','w'),indent=2)
m=model(ck).cpu().eval();sample=np.load(R/'selected_29901.npz')['history_samples'][::10].reshape(-1,25,82)
torch.onnx.export(m,torch.from_numpy(sample[:1]),str(O/'selected.onnx'),input_names=['history'],output_names=['force'],dynamic_axes={'history':{0:'batch'},'force':{0:'batch'}},opset_version=17,dynamo=False)
sess=ort.InferenceSession(str(O/'selected.onnx'),providers=['CPUExecutionProvider'])
with torch.no_grad():a=m(torch.from_numpy(sample)).numpy()
b=sess.run(None,{'history':sample})[0];err=float(abs(a-b).max());parity['onnx']={'max_normalized':err,'max_N':err*80,'pass':err<1e-5}
scenes=['ramp','release','reverse','sweep','full3d'];numeric=all(x['pass'] for x in parity.values());accuracy=all(rows[g]['pass'] for rows in results['selected'].values() for g in scenes)
strict_groups=['overall']+scenes+[g+'/force_'+b for g in scenes for b in ['0_15.0001','15.0001_45.001']]+['force_0_15.0001','force_15.0001_45.001','release/switch','release/recovery','reverse/switch','reverse/recovery']
def strict_pass(r):
 if not r['n']:return False
 size=max(r['component_mae'])<=1 and max(r['component_rmse'])<=1.5 and r['mag_mae']<=1.5 and r['mag_rmse']<=1.5
 return bool(size and (r['angle_n']==0 or (r['angle_mean']<=5 and r['angle_p95']<=10)))
strict_accuracy=all(strict_pass(rows[g]) for rows in results['selected'].values() for g in strict_groups)
stability=all(events[f'selected_{s}']['total']<=events[f'start_{s}']['total'] for s in [29901,29902])
decision={'practical_validation_convergence':c['converged'],'numerical_pass':numeric,'all_scene_aggregate_accuracy_pass':accuracy,'strict_full_coverage_accuracy_pass':strict_accuracy,'no_paired_termination_regression':stability,'default_deployment_changed':False,'promote':False,'note':'Diagnostic closed-loop contract only; convergence is not full accuracy or deployment certification.'}
json.dump({'convergence':{k:v for k,v in c.items() if k!='history'},'results':results,'audits':audits,'events':events,'parity':parity,'response':response,'force_plans_exactly_matched':True,'decision':decision},open(O/'acceptance.json','w'),indent=2)
json.dump(decision,open(O/'promotion_decision.json','w'),indent=2)
wr=SummaryWriter(str(O/'decision'))
wr.add_text('decision/checkpoint',json.dumps({'checkpoint':c['selected_checkpoint'],'sha256':c['selected_sha256'],'selected_epoch':c['selected_epoch'],'stop_epoch':c['last_epoch'],'optimizer_restart_at_epoch':240},indent=2),c['last_epoch'])
for k in ['practical_validation_convergence','numerical_pass','all_scene_aggregate_accuracy_pass','strict_full_coverage_accuracy_pass','no_paired_termination_regression','promote']:wr.add_scalar('decision/'+k,int(decision[k]),c['last_epoch'])
wr.close();tb=Path('logs/rsl_rl/go2_velocity/force_estimator_ft_20261014_v2/tensorboard')
for d in O.iterdir():
 if d.is_dir() and list(d.glob('events.out.tfevents.*')):
  link=tb/('convergence_v1_'+d.name)
  if not link.exists():link.symlink_to(d.resolve(),target_is_directory=True)
print('EVALUATION_COMPLETE',decision,events,parity,flush=True)
