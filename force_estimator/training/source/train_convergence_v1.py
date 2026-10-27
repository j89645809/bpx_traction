from train_causal_extended import CausalData,windows,validate,START as ORIGINAL
from direction_balance_common import *
from transient_v2_train import rank,eligible,KEYS,value
from torch.utils.tensorboard import SummaryWriter
from mjlab.utils.torch import configure_torch_backends
import hashlib,time,os,random
O=Path('logs/rsl_rl/go2_velocity/convergence_v1');O.mkdir(exist_ok=True,parents=True)
START=Path('logs/rsl_rl/go2_velocity/causal_v3_extended/aligned_window_720.pt')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 configure_torch_backends(allow_tf32=False,deterministic=True);torch.set_num_threads(2)
 torch.manual_seed(20261020);random.seed(20261020);rng=np.random.default_rng(20261020)
 ds=[CausalData(s) for s in [29101,29102]];v=CausalData(29201);ww=[windows(d) for d in ds]
 ids=np.unique(np.concatenate([v.ids[::10],windows(v)]));base=validate(model(ORIGINAL),v,ids)
 m=model(START);anchor=validate(m,v,ids);opt=torch.optim.AdamW(m.parameters(),lr=3e-6,weight_decay=1e-5)
 rates=[3e-6,1e-6,3e-7];stage=0;stage_ep=240;last_significant=240;best=rank(anchor,base);significant=best;be=240;history=[];ep0=240
 writer=SummaryWriter(str(O/'training'));tb=Path('logs/rsl_rl/go2_velocity/force_estimator_ft_20261014_v2/tensorboard/convergence_v1_training')
 if not tb.exists():tb.symlink_to((O/'training').resolve(),target_is_directory=True)
 def save(path,ep,r):
  torch.save({'model':{k:t.detach().cpu().clone() for k,t in m.state_dict().items()},'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':rng.bit_generator.state,'python_rng':random.getstate(),'epoch':ep,'stage':stage,'stage_epoch':stage_ep,'last_significant':last_significant,'significant_score':significant,'best_score':best,'best_epoch':be,'history':history,'validation':r,'history_length':25,'input_dim':82,'hidden_dim':32,'label_contract':'integrated force, current body coordinates','start_sha256':digest(START),'optimizer_restart_at_epoch':240},path)
 if (O/'last.pt').exists():
  c=torch.load(O/'last.pt',weights_only=False);m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);rng.bit_generator.state=c['numpy_rng'];random.setstate(c['python_rng']);ep0=c['epoch'];stage=c['stage'];stage_ep=c['stage_epoch'];last_significant=c['last_significant'];significant=c['significant_score'];best=c['best_score'];be=c['best_epoch'];history=c['history']
 else:
  json.dump({'start_sha256':digest(START),'audits':[d.audit for d in ds+[v]],'anchor':anchor,'original':base,'validation_ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'data_hashes':{str(d.path):digest(d.path) for d in ds+[v]},'torch':torch.__version__,'optimizer_restart':True},open(O/'audit.json','w'),indent=2)
  save(O/'best.pt',240,anchor)
 reason='safety_limit_not_converged';converged=False;t0=time.time()
 for ep in range(ep0+1,10001):
  m.train();ls=[]
  for it in range(32):
   di=(ep+it)%2;d=ds[di];ix=np.concatenate([rng.choice(d.ids,448),rng.choice(ww[di],64)])
   x,y=d.get(ix);p=m(torch.from_numpy(x).cuda());l=loss(p,torch.from_numpy(y).cuda());opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();ls.append(float(l.detach()))
  writer.add_scalar('loss/train',np.mean(ls),ep);writer.add_scalar('optimization/lr',opt.param_groups[0]['lr'],ep)
  if ep%10:continue
  r=validate(m,v,ids);vl=evaluate_loss(m,v.get,ids);score=rank(r,base);ok=eligible(r,base) and eligible(r,anchor)
  row={'epoch':ep,'score':score,'validation_loss':vl,'train_loss':float(np.mean(ls)),'guard':ok,'stage':stage,'metrics':r};history.append(row)
  for g,rr in r.items():
   for k in ['mag_mae','mag_rmse','angle_mean','angle_p95']:
    if rr[k] is not None:writer.add_scalar(f'validation/{g}/{k}',rr[k],ep)
   for i,k in enumerate(['fx','fy','fz']):writer.add_scalar(f'validation/{g}/{k}_rmse',rr['component_rmse'][i],ep)
  writer.add_scalar('validation/loss',vl,ep);writer.add_scalar('validation/score',score,ep);writer.add_scalar('validation/guard_pass',ok,ep)
  if ok and score<best:best=score;be=ep;save(O/'best.pt',ep,r)
  if ok and score<significant*.998:significant=score;last_significant=ep
  status={'epoch':ep,'optimizer_updates_since_240':(ep-240)*32,'best_epoch':be,'best_score':best,'stage':stage,'lr':opt.param_groups[0]['lr'],'last_significant_epoch':last_significant,'elapsed_s':time.time()-t0}
  stable=False
  if len(history)>=12:
   recent=history[-6:];prior=history[-12:-6]
   def med(rows,key):return float(np.median([x[key] for x in rows]))
   gains={k:1-med(recent,k)/med(prior,k) for k in ['score','validation_loss']}
   gains.update({g+'/'+k:1-np.median([value(x['metrics'],g,k) for x in recent])/np.median([value(x['metrics'],g,k) for x in prior]) for g,k in KEYS})
   stable=all(gains[k]<.005 and abs(gains[k])<=.03 for k in ['score','validation_loss']) and all(gains[g+'/'+k]<.01 for g,k in KEYS)
   status['last60_vs_prior60_relative_improvements']=gains
  if ep-stage_ep>=160 and ep-last_significant>=120 and stage<2:
   stage+=1;stage_ep=ep;last_significant=ep;significant=best
   for group in opt.param_groups:group['lr']=rates[stage]
   print('LR_REDUCED',ep,rates[stage],flush=True)
  elif stage==2 and ep-stage_ep>=180 and ep-last_significant>=180 and stable:
   converged=True;reason='practical_fixed_validation_plateau';status['converged']=True
  json.dump(history,open(O/'history.json','w'),indent=2);json.dump(status,open(O/'status.json','w'),indent=2)
  if ep%20==0 or converged:save(O/'last.pt',ep,r)
  if ep%20==0:save(O/f'epoch_{ep}.pt',ep,r)
  print('PROGRESS',json.dumps(status),flush=True);writer.flush()
  if converged:break
 json.dump({'converged':converged,'reason':reason,'last_epoch':ep,'selected_epoch':be,'selected_checkpoint':str(O/'best.pt'),'selected_sha256':digest(O/'best.pt'),'selected_validation_score':best,'starting_validation_score':rank(anchor,base),'optimizer_restart_at_epoch':240,'history':history,'status':status},open(O/'convergence.json','w'),indent=2)
 writer.add_scalar('decision/converged',converged,ep);writer.close();print('COMPLETE',reason,ep,be,flush=True)
if __name__=='__main__':main()
