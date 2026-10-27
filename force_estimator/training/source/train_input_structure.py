import os,json,hashlib
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.tensorboard import SummaryWriter
from src.tasks.velocity.rl.force_estimator import ForceEstimator
ROOT=Path('logs/paired_v6v7'); OUT=Path('logs/rsl_rl/go2_velocity/force_estimator_input_structure_20261016'); OUT.mkdir(parents=True,exist_ok=True)
BASE=torch.load('logs/rsl_rl/go2_velocity/force_estimator_ft_20261016_v8/model_fullrange_v8a.pt',map_location='cpu',weights_only=False)['force_estimator_state_dict']
class Est50(ForceEstimator):
 def __init__(self): super().__init__(82,50,32); self.pool=nn.AdaptiveAvgPool1d(1)
 def forward(self,x): return self.head(self.pool(self.temporal(self.encoder(x).transpose(1,2))).squeeze(-1))
class EstDelta(ForceEstimator):
 def __init__(self): super().__init__(164,25,32)
 def forward(self,x):
  d=torch.zeros_like(x); d[:,1:]=x[:,1:]-x[:,:-1]; return super().forward(torch.cat([x,d],-1))
def loadset(envs, mode):
 X=[];Y=[];S=[];A=[];SC=[]
 for fp in sorted(ROOT.glob('dual_*.npz')):
  d=np.load(fp); h=d['history50'][:,envs]; y=d['labels'][:,envs]; a=d['applied'][:,envs]; sc=str(d['scenario'])
  X.append(h);Y.append(y);A.append(a);S.append(np.broadcast_to(d['step'][:,None],(h.shape[0],len(envs))));SC += [sc]*h.shape[0]
 return np.concatenate(X,axis=0),np.concatenate(Y,axis=0),np.concatenate(A,axis=0),np.concatenate(S,axis=0),np.array(SC)

def flatten(h,y,a,step,sc,mode):
 n,e,L,D=h.shape; x=h.reshape(n*e,L,D); yy=y.reshape(n*e,3); aa=a.reshape(n*e,3); ss=step.reshape(n*e); scenes=np.repeat(sc,e)
 if mode=='baseline25' or mode=='delta25': x=x[:,-25:]
 return x.astype('float32'),yy.astype('float32'),aa.astype('float32'),ss.astype('int64'),scenes
Htr,Ytr,Atr,Str,ScTr=loadset(range(8),'train'); Hv,Yv,Av,Sv,ScV=loadset(range(8,10),'val'); Hb,Yb,Ab,Sb,ScB=loadset(range(10,12),'blind')

def prep(H,Y,A,S,SC,mode): return flatten(H,Y,A,S,SC,mode)
# weights emphasize reverse transition and full3d low/mid while retaining every scenario

def weights(A,S,SC):
 mag=np.linalg.norm(A,axis=-1); w=np.ones(len(S),np.float32); trans=np.zeros(len(S),bool)
 for i,sc in enumerate(SC):
  k=S[i]
  if sc=='reverse': trans[i]=(470<=k<535)
  elif sc=='release': trans[i]=(235<=k<275)
  elif sc=='ramp': trans[i]=(k<40)
  elif sc=='sweep': trans[i]=(k%125<15)
  elif sc=='full3d': w[i]=1+2*((mag[i]>=2)&(mag[i]<30))
 w=np.where(trans,8,w); return w.astype('float32'),trans.astype('float32')
def make_model(kind):
 if kind=='baseline25': m=ForceEstimator(82,25,32); m.load_state_dict(BASE)
 elif kind=='history50':
  m=Est50(); sd=m.state_dict();
  for k,v in BASE.items(): sd[k]=v.clone()
  m.load_state_dict(sd)
 else:
  m=EstDelta(); sd=m.state_dict()
  for k,v in BASE.items():
   if k=='encoder.0.weight': sd[k][:,:82]=v; sd[k][:,82:]=0
   else: sd[k]=v.clone()
  m.load_state_dict(sd)
 return m
def loss_parts(p,y,yp,yprev,w,t):
 comp=torch.nn.functional.smooth_l1_loss(p,y,reduction='none').mean(1); pm=torch.linalg.norm(p,dim=1); tm=torch.linalg.norm(y,dim=1); mag=torch.nn.functional.smooth_l1_loss(pm,tm,reduction='none'); cos=(p*y).sum(1)/(pm*tm).clamp_min(1e-6); dire=torch.nn.functional.smooth_l1_loss(1-cos,torch.zeros_like(cos),reduction='none')*(tm>5/80); delta=torch.nn.functional.smooth_l1_loss(p-yp,y-yprev,reduction='none').mean(1); return (comp+.25*mag+.02*dire+t*delta)*w

def train(kind):
 x,y,a,s,sc=prep(Htr,Ytr,Atr,Str,ScTr,kind); xv,yv,av,sv,scv=prep(Hv,Yv,Av,Sv,ScV,kind); wb,tb=weights(a,s,sc); wv,tv=weights(av,sv,scv)
 m=make_model(kind).cuda().train(); opt=torch.optim.AdamW(m.parameters(),lr=3e-6,weight_decay=1e-5); ds=TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(wb),torch.from_numpy(tb)); dl=DataLoader(ds,1024,shuffle=True); wr=SummaryWriter(str(OUT/kind)); best=1e9; bestsd=None
 for ep in range(1,16):
  tot=den=0.; m.train()
  for xb,yb,ww,tt in dl:
   xb,yb,ww,tt=[z.cuda() for z in (xb,yb,ww,tt)]; p=m(xb); z=torch.zeros_like(p); l=(loss_parts(p,yb,z,z,ww,tt)).mean(); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); tot+=float(l)*len(xb); den+=len(xb)
  m.eval(); vals=[]
  with torch.no_grad():
   for i in range(0,len(xv),4096):
    xb=torch.from_numpy(xv[i:i+4096]).cuda(); yb=torch.from_numpy(yv[i:i+4096]).cuda(); ww=torch.from_numpy(wv[i:i+4096]).cuda(); tt=torch.from_numpy(tv[i:i+4096]).cuda(); p=m(xb); z=torch.zeros_like(p); vals.append(loss_parts(p,yb,z,z,ww,tt).detach().cpu().numpy())
  val=float(np.concatenate(vals).mean()); wr.add_scalar('loss/train',tot/den,ep); wr.add_scalar('loss/validation',val,ep)
  if val<best: best=val; bestsd={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 wr.close(); m.load_state_dict(bestsd); path=OUT/(kind+'.pt'); torch.save({'model':m.state_dict(),'kind':kind,'best_val_loss':best,'input_dim':164 if kind=='delta25' else 82,'history_length':25 if kind!='history50' else 50},path); return m,best,str(path)
def metrics(m,kind):
 x,y,a,s,sc=prep(Hb,Yb,Ab,Sb,ScB,kind); m.eval(); pp=[]
 with torch.no_grad():
  for i in range(0,len(x),4096): pp.append(m(torch.from_numpy(x[i:i+4096]).cuda()).cpu().numpy())
 p=np.concatenate(pp)*80; t=y*80; mag=np.linalg.norm(a.reshape(-1,3),axis=1); tm=np.linalg.norm(t,axis=1); pm=np.linalg.norm(p,axis=1); err=p-t; ang=np.degrees(np.arccos(np.clip((p*t).sum(1)/(pm*tm).clip(1e-6),-1,1)))
 out={'overall':{'component_mae':np.mean(np.abs(err),0).tolist(),'component_rmse':np.sqrt(np.mean(err**2,0)).tolist(),'mag_mae':float(np.mean(abs(pm-tm))),'mag_rmse':float(np.sqrt(np.mean((pm-tm)**2))),'angle_mean_deg':float(np.mean(ang[tm>=2])),'angle_p95_deg':float(np.percentile(ang[tm>=2],95))}}
 scenes=np.repeat(ScB,2)
 for sc in sorted(set(scenes)):
  q=scenes==sc; out[sc]={'mag_rmse':float(np.sqrt(np.mean((pm[q]-tm[q])**2))),'angle_p95_deg':float(np.percentile(ang[q & (tm>=2)],95)),'n':int(q.sum())}
 return out
results={}
for kind in ['baseline25','history50','delta25']:
 m,best,path=train(kind); results[kind]={'best_val_loss':best,'checkpoint':path,'blind':metrics(m,kind)}
json.dump({'split':{'train_envs':'0-7','val_envs':'8-9','blind_envs':'10-11'},'scenarios':['ramp','release','reverse','sweep','full3d'],'train_samples':int(Htr.shape[0]*Htr.shape[1]),'val_samples':int(Hv.shape[0]*Hv.shape[1]),'blind_samples':int(Hb.shape[0]*Hb.shape[1]),'results':results},open(OUT/'input_structure_results.json','w'),indent=2)
print(json.dumps(results,indent=2))
