from direction_balance_common import *
from transient_v2_train import rank,eligible
from torch.utils.tensorboard import SummaryWriter
import hashlib
O=Path('logs/rsl_rl/go2_velocity/causal_v3');O.mkdir(parents=True,exist_ok=True)
START=Path('logs/rsl_rl/go2_velocity/direction_balance_v1/augmented_717.pt')

class CausalData(Data):
 def __init__(self,seed):
  super().__init__(seed)
  z=np.load(self.path);world=z['wrench_world'].astype(float)
  # The environment integrates first, then updates the external-force buffer.
  # At observation t, wrench recorded at t-1 is the force integrated into state t.
  integrated=np.concatenate([np.zeros_like(world[:1]),world[:-1]])
  qq=self.q.astype(float);v=qq[...,1:];w=qq[...,:1]
  self.y=((integrated-2*w*np.cross(v,integrated)+2*np.cross(v,np.cross(v,integrated)))/80).astype('float32')
  t=self.ids//self.n;e=self.ids%self.n
  self.ids=self.ids[(t>0)&(~self.done[np.maximum(t-1,0),e])]
  err=float(abs(np.linalg.norm(self.y[2:]*80,axis=-1)-np.linalg.norm(self.applied[:-2],axis=-1)).max())
  assert err<1e-3
  self.audit['physical_label_vs_command_two_steps_before_max_N']=err
  self.audit['reset_first_frames_excluded_from_training_validation']=True
  self.audit['label_contract']='preceding integrated world wrench rotated into current body frame; source-loop ordering proof, not best-fit lag'

def windows(d):
 t=d.ids//d.n;e=d.ids%d.n
 return d.ids[(((t>=1577)&(t<1652))|((t>=2977)&(t<3052)))&(t>d.start[t,e])]

def validate(m,d,ids):
 p=predict(m,d,ids);t=ids//d.n;e=ids%d.n;y=d.y[t,e];mag=np.linalg.norm(y*80,axis=1)
 masks={'overall':np.ones(len(ids),bool),'high':mag>30,'ramp':t//1200==0,'sweep':t//1200==3,
 'release_switch':(t>=1602)&(t<1607),'reverse_recovery':(t>=3007)&(t<3032)}
 return {k:metric(p[v],y[v]) for k,v in masks.items()}

def main():
 from mjlab.utils.torch import configure_torch_backends
 configure_torch_backends(allow_tf32=False);torch.set_num_threads(2)
 ds=[CausalData(s) for s in [29101,29102]];v=CausalData(29201);ww=[windows(d) for d in ds]
 ids=np.unique(np.concatenate([v.ids[::10],windows(v)]));base=validate(model(START),v,ids)
 json.dump({'baseline':base,'audits':[d.audit for d in ds+[v]],'excluded_legacy_dual':'no saved integrated wrench provenance; retained full-scene anchors from audited29101/29102'},open(O/'audit.json','w'),indent=2)
 results={}
 for seed in [719,720]:
  for arm in ['aligned_uniform','aligned_window']:
   torch.manual_seed(seed);rng=np.random.default_rng(seed);m=model(START);opt=torch.optim.AdamW(m.parameters(),lr=3e-6,weight_decay=1e-5)
   writer=SummaryWriter(str(O/f'{arm}_{seed}'));best=1.;be=0
   torch.save({'model':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'epoch':0,'arm':arm,'seed':seed},O/f'{arm}_{seed}.pt')
   for ep in range(1,61):
    m.train();ls=[]
    for it in range(32):
     di=(ep+it)%2;d=ds[di];ix=rng.choice(d.ids,512 if arm=='aligned_uniform' else 448)
     if arm=='aligned_window':ix=np.concatenate([ix,rng.choice(ww[di],64)])
     x,y=d.get(ix);p=m(torch.from_numpy(x).cuda());l=loss(p,torch.from_numpy(y).cuda())
     opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();ls.append(float(l))
    writer.add_scalar('loss/train',np.mean(ls),ep)
    if ep%5==0:
     r=validate(m,v,ids);score=rank(r,base);ok=eligible(r,base)
     for g,row in r.items():
      for k in ['mag_rmse','angle_p95']:
       if row[k] is not None:writer.add_scalar(f'validation/{g}/{k}',row[k],ep)
     writer.add_scalar('validation/score',score,ep);writer.add_scalar('validation/guard_pass',int(ok),ep)
     if ok and score<best:
      best=score;be=ep;torch.save({'model':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'epoch':ep,'arm':arm,'seed':seed,'validation':r,'history_length':25,'input_dim':82,'hidden_dim':32,'label_contract':'integrated force, current body coordinates'},O/f'{arm}_{seed}.pt')
     print('TRAIN',arm,seed,ep,score,ok,'best',be,flush=True)
    writer.flush()
   writer.close();results[f'{arm}_{seed}']={'score':best,'epoch':be,'sha256':hashlib.sha256((O/f'{arm}_{seed}.pt').read_bytes()).hexdigest()}
   json.dump(results,open(O/'training_results.json','w'),indent=2)
 selected=min(results,key=lambda x:results[x]['score'])
 json.dump({'selected':selected,'results':results,'criterion':'causally aligned validation and same guards; no blind access'},open(O/'selection.json','w'),indent=2)
 print('COMPLETE',selected,flush=True)
if __name__=='__main__':main()
