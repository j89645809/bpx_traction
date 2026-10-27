from direction_balance_common import *
from torch.utils.tensorboard import SummaryWriter
import hashlib

DEST=Path('logs/rsl_rl/go2_velocity/transient_v2')
START=Path('logs/rsl_rl/go2_velocity/direction_balance_v1/augmented_717.pt')

def event_ids(d):
 t=d.ids//d.n;e=d.ids%d.n
 event=((t>=1576)&(t<1651))|((t>=2976)&(t<3051))
 return d.ids[event&(t>d.start[t,e])]

def temporal_loss(p,previous,y,previous_y):
 return nn.functional.smooth_l1_loss(p-previous,y-previous_y)

def validation(m,d,ids):
 p=predict(m,d,ids);t=ids//d.n;e=ids%d.n;y=d.y[t,e]
 mag=np.linalg.norm(y*80,axis=1)
 masks={'overall':np.ones(len(ids),bool),'high':mag>30,
 'ramp':t//1200==0,'sweep':t//1200==3,
 'release_switch':(t>=1601)&(t<1606),
 'reverse_recovery':(t>=3006)&(t<3031)}
 return {k:metric(p[v],y[v]) for k,v in masks.items()}

KEYS=[('overall','mag_rmse'),('overall','angle_p95'),('high','fz'),
 ('release_switch','mag_rmse'),('reverse_recovery','mag_rmse'),('reverse_recovery','angle_p95')]
GUARDS=[('ramp','mag_rmse'),('ramp','angle_p95'),('sweep','mag_rmse'),('sweep','angle_p95'),
 ('release_switch','mag_rmse'),('reverse_recovery','mag_rmse'),('reverse_recovery','angle_p95')]
def value(r,g,k):return r[g]['component_rmse'][2] if k=='fz' else r[g][k]
def rank(r,baseline):return float(np.mean([value(r,g,k)/max(value(baseline,g,k),1e-9) for g,k in KEYS]))
def eligible(r,baseline):return all(value(r,g,k)<=1.1*value(baseline,g,k) for g,k in GUARDS)

def main():
 from mjlab.utils.torch import configure_torch_backends
 configure_torch_backends(allow_tf32=False);torch.set_num_threads(2);DEST.mkdir(parents=True,exist_ok=True)
 train=[Data(s) for s in [29101,29102]];val=Data(29201)
 windows=[event_ids(d) for d in train];x,y=olddata('train')
 # Fixed validation subsample plus ALL event windows, so first100ms cannot disappear.
 vi=np.unique(np.concatenate([val.ids[::10],event_ids(val)]))
 m=model(START).eval();baseline=validation(m,val,vi)
 json.dump({'baseline':baseline,'n':len(vi),'training_audits':[d.audit for d in train],
  'event_counts':[len(w) for w in windows],'start_sha256':hashlib.sha256(START.read_bytes()).hexdigest()},open(DEST/'audit.json','w'),indent=2)
 results={}
 for seed in [719,720]:
  for arm in ['extend','window','phase']:
   torch.manual_seed(seed);rng=np.random.default_rng(seed);m=model(START)
   opt=torch.optim.AdamW(m.parameters(),lr=3e-6,weight_decay=1e-5)
   writer=SummaryWriter(str(DEST/f'{arm}_{seed}'));best=1.;best_ep=0
   ck={'model':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'history_length':25,'input_dim':82,'hidden_dim':32,'epoch':0,'arm':arm,'seed':seed}
   torch.save(ck,DEST/f'{arm}_{seed}.pt')
   for ep in range(1,61):
    losses=[];m.train()
    for it in range(32):
     ix=rng.integers(len(x),size=256);di=(ep+it)%2;d=train[di]
     # Separate draws are identical in window/phase arms under matched optimizer seed.
     ids=rng.choice(d.ids,256 if arm=='extend' else 192)
     if arm!='extend':ids=np.concatenate([ids,rng.choice(windows[di],64)])
     nx,ny=d.get(ids);bx=np.concatenate([x[ix],nx]);by=np.concatenate([y[ix],ny])
     pred=m(torch.from_numpy(bx).cuda());target=torch.from_numpy(by).cuda();l=loss(pred,target)
     if arm=='phase':
      pairids=ids[-64:];px,py=d.get(pairids-d.n)
      previous=m(torch.from_numpy(px).cuda());previous_y=torch.from_numpy(py).cuda()
      l=l+.5*loss(pred[-64:],target[-64:])+.25*temporal_loss(pred[-64:],previous,target[-64:],previous_y)
     opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(l.detach()))
    writer.add_scalar('loss/train',np.mean(losses),ep)
    if ep%5==0:
     r=validation(m,val,vi);score=rank(r,baseline);ok=eligible(r,baseline)
     for g,row in r.items():
      for k in ['mag_rmse','angle_p95']:
       if row[k] is not None:writer.add_scalar(f'validation/{g}/{k}',row[k],ep)
     writer.add_scalar('validation/score',score,ep);writer.add_scalar('validation/guard_pass',int(ok),ep)
     if ok and score<best:
      best=score;best_ep=ep;torch.save({'model':{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},'history_length':25,'input_dim':82,'hidden_dim':32,'epoch':ep,'arm':arm,'seed':seed,'validation':r,'score':score},DEST/f'{arm}_{seed}.pt')
     print('TRAIN',arm,seed,ep,'score',round(score,5),'guard',ok,'best',best_ep,flush=True)
    writer.flush()
   writer.close();results[f'{arm}_{seed}']={'score':best,'epoch':best_ep,'sha256':hashlib.sha256((DEST/f'{arm}_{seed}.pt').read_bytes()).hexdigest()}
   json.dump(results,open(DEST/'training_results.json','w'),indent=2)
 chosen=min(results,key=lambda k:results[k]['score'])
 json.dump({'selected':chosen,'results':results,'criterion':'validation only; predeclared guards; no blind access'},open(DEST/'selection.json','w'),indent=2)
 print('COMPLETE',chosen,flush=True)

if __name__=='__main__':main()
