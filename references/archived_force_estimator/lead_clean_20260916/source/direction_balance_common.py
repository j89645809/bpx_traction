import json
from pathlib import Path
import numpy as np,torch
from torch import nn
from src.tasks.velocity.rl.force_estimator import ForceEstimator
ROOT=Path('logs/direction_balance_v1');OUT=Path('logs/rsl_rl/go2_velocity/direction_balance_v1');OUT.mkdir(parents=True,exist_ok=True)
def loss(p,y):
 return nn.functional.smooth_l1_loss(p,y)+.25*nn.functional.smooth_l1_loss(p.norm(dim=-1),y.norm(dim=-1))+.002*((1-nn.functional.cosine_similarity(p,y,dim=-1))*(y.norm(dim=-1)>=2/80)).mean()
def model(path):
 m=ForceEstimator(82,25,32).cuda();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);return m
BASE=Path('logs/rsl_rl/go2_velocity/input_fair_v2/latest25_s717.pt')
class Data:
 def __init__(self,seed,path=None):
  self.path=Path(path) if path else ROOT/f'baseline_{seed}.npz';d=np.load(self.path)
  self.f=d['frames'];self.y=d['labels'];self.done=d['dones'];self.q=d['quat'];self.pos=d['position'];self.pred=d['prediction'];self.applied=d['applied'];self.seed=seed
  t,n,_=self.f.shape;times=np.arange(t)[:,None];reset=np.concatenate([np.zeros((1,n),bool),self.done[:-1]])
  self.start=np.maximum.accumulate(np.where(reset,times,0),axis=0)
  self.ids=np.flatnonzero(np.broadcast_to((times%1200)>=50,(t,n)).ravel());self.n=n
  # Reconstructed histories must match every saved online history, including resets.
  hs=d['history_samples'];mx=0.
  for i in range(len(hs)):
   ids=i*100*n+np.arange(n);mx=max(mx,float(abs(self.get(ids)[0]-hs[i]).max()))
  assert mx==0,('history mismatch',self.path,mx)
  world=d['wrench_world'];qq=self.q.astype(float);v=qq[...,1:];w=qq[...,:1];world=world.astype(float)
  rotated=world-2*w*np.cross(v,world)+2*np.cross(v,np.cross(v,world))
  labelerr=float(abs(rotated-self.y*80).max());assert labelerr<1e-3
  # Norm survives a coordinate rotation; compare label timing to previous command.
  normerr=float(abs(np.linalg.norm(self.y[1:]*80,axis=-1)-np.linalg.norm(self.applied[:-1],axis=-1)).max());assert normerr<1e-3
  self.audit={'history_reconstruction_max':mx,'label_rotation_max_N':labelerr,'previous_command_norm_max_N':normerr,'resets':int(self.done.sum()),'frames':int(len(self.ids))}
  yy=self.y.reshape(-1,3)[self.ids]*80;mag=np.linalg.norm(yy,axis=1);cos=yy[:,2]/np.maximum(mag,1e-12);gravity=self.f.reshape(-1,82)[self.ids,9:12];tilt=np.degrees(np.arccos(np.clip(-gravity[:,2]/np.linalg.norm(gravity,axis=1),-1,1)))
  self.audit['coverage']={'fz_min_N':float(yy[:,2].min()),'fz_max_N':float(yy[:,2].max()),'up_frames':int(((cos>.5)&(mag>=2)).sum()),'down_frames':int(((cos<-.5)&(mag>=2)).sum()),'abs_fz_above20_frames':int((abs(yy[:,2])>20).sum()),'tilt_ge5_frames':int((tilt>=5).sum()),'tilt_ge10_frames':int((tilt>=10).sum()),'tilt_max_deg':float(tilt.max())}

 def get(self,ids):
  t=ids//self.n;e=ids%self.n;idx=t[:,None]-np.arange(25);h=self.f[np.maximum(idx,0),e[:,None]].copy();h[idx<self.start[t,e,None]]=0
  return h,self.y[t,e]
def olddata(split):
 xx=[];yy=[]
 env=slice(0,8) if split=='train' else slice(8,10)
 for fp in sorted(Path('logs/paired_v6v7').glob('dual_*.npz')):
  d=np.load(fp);xx.append(d['history50'][50:,env,:25].reshape(-1,25,82).copy());yy.append(d['labels'][50:,env].reshape(-1,3).copy())
 return np.concatenate(xx),np.concatenate(yy)
def evaluate_loss(m,get,ids):
 total=0.;m.eval()
 with torch.no_grad():
  for j in range(0,len(ids),1024):
   ix=ids[j:j+1024];x,y=get(ix);p=m(torch.from_numpy(x).cuda());l=loss(p,torch.from_numpy(y).cuda());total+=float(l)*len(ix)
 return total/len(ids)
def predict(m,data,ids):
 m.eval();out=[]
 with torch.no_grad():
  for j in range(0,len(ids),1024):out.append(m(torch.from_numpy(data.get(ids[j:j+1024])[0]).cuda()).cpu().numpy())
 return np.concatenate(out)
def metric(p,y):
 p=p.astype(float)*80;y=y.astype(float)*80;e=p-y;pm=np.linalg.norm(p,axis=-1);tm=np.linalg.norm(y,axis=-1);valid=tm>=2;ang=np.degrees(np.arccos(np.clip((p*y).sum(-1)/np.maximum(pm*tm,1e-12),-1,1)))
 if not len(y):return {'n':0,'pass':False}
 r={'n':len(y),'angle_n':int(valid.sum()),'component_mae':abs(e).mean(0).tolist(),'component_rmse':np.sqrt((e*e).mean(0)).tolist(),'fz_bias':float(e[:,2].mean()),'mag_mae':float(abs(pm-tm).mean()),'mag_rmse':float(np.sqrt(((pm-tm)**2).mean())),'angle_mean':float(ang[valid].mean()) if valid.any() else None,'angle_p95':float(np.percentile(ang[valid],95)) if valid.any() else None}
 r['pass_size_direction']=bool(valid.any() and r['mag_mae']<=1.5 and r['mag_rmse']<=1.5 and r['angle_mean']<=5 and r['angle_p95']<=10)
 r['pass']=bool(r['pass_size_direction'] and max(r['component_mae'])<=1 and max(r['component_rmse'])<=1.5);return r
def breakdown(p,data):
 ids=data.ids;t=ids//data.n;e=ids%data.n;y=data.y[t,e];mag=np.linalg.norm(y*80,axis=1);cos=y[:,2]/np.maximum(mag/80,1e-12);g=data.f[t,e,9:12];tilt=np.degrees(np.arccos(np.clip(-g[:,2]/np.linalg.norm(g,axis=1),-1,1)))
 masks={'overall':np.ones(len(ids),bool)}
 for k,s in enumerate(['ramp','release','reverse','sweep','full3d']):masks[s]=t//1200==k
 for lo,hi in [(0,15.0001),(15.0001,45.001),(0,5),(5,15.0001),(15.0001,30.0001),(30.0001,45.001)]:masks[f'force_{lo:g}_{hi:g}']=(mag>=lo)&(mag<hi)
 for s,m in [('up',(cos>.5)&(mag>=2)),('down',(cos<-.5)&(mag>=2)),('horizontal',(abs(cos)<=.5)&(mag>=2)),('zero',mag<2)]:masks['direction_'+s]=m
 for lo,hi in [(0,5),(5,10),(10,20),(20,181)]:masks[f'tilt_{lo}_{hi}']=(tilt>=lo)&(tilt<hi)
 for k,s in enumerate(['ramp','release','reverse','sweep','full3d']):
  for lo,hi in [(0,15.0001),(15.0001,45.001)]:masks[f'{s}/force_{lo:g}_{hi:g}']=(t//1200==k)&(mag>=lo)&(mag<hi)
 for scene_id,scene,event in [(1,'release',401),(2,'reverse',601)]:
  local=t%1200;in_scene=t//1200==scene_id
  for phase,lo,hi in [('before',event-25,event),('switch',event,event+5),('recovery',event+5,event+30),('later',event+30,1200)]:masks[f'{scene}/{phase}']=in_scene&(local>=lo)&(local<hi)
 return {k:metric(p[q],y[q]) for k,q in masks.items()}
