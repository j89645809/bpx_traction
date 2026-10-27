"""Pure NumPy metrics. All inputs/outputs are physical newtons/seconds/degrees."""
import numpy as np
DT=.02
EVENTS={'release':1601,'reverse':3001,'load_release':1251,'load_reverse':2451,'load_full3d':4851}
def angle_deg(p,y):
 m=np.linalg.norm(y,axis=-1);pm=np.linalg.norm(p,axis=-1)
 return np.degrees(np.arccos(np.clip((p*y).sum(-1)/np.maximum(pm*m,1e-12),-1,1)))
def score(p,y):
 p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=float);n=len(y)
 if not n:return {'n':0}
 m=np.linalg.norm(y,axis=-1);pm=np.linalg.norm(p,axis=-1);e=p-y;v=np.linalg.norm(e,axis=-1);de=pm-m;good=m>=2;ang=angle_deg(p,y);z=m<=1e-4
 r={'n':n,'angle_n':int(good.sum()),'component_mae':np.abs(e).mean(0).tolist(),'component_rmse':np.sqrt((e*e).mean(0)).tolist(),'mag_mae':float(abs(de).mean()),'mag_rmse':float(np.sqrt((de*de).mean())),
 'vector_rmse':float(np.sqrt((v*v).mean())),'vector_error_p95':float(np.percentile(v,95)),'angle_mean':float(ang[good].mean()) if good.any() else None,'angle_p95':float(np.percentile(ang[good],95)) if good.any() else None}
 r['relative_n']=int(good.sum())
 if good.any():
  rm=de[good]/m[good];rv=v[good]/m[good]
  r.update({'relative_mag_mae_pct':float(abs(rm).mean()*100),'relative_mag_rms_pct':float(np.sqrt((rm*rm).mean())*100),'relative_vector_mae_pct':float(rv.mean()*100),'relative_vector_rms_pct':float(np.sqrt((rv*rv).mean())*100),'vector_nrmse_pct':float(np.sqrt((v[good]**2).mean()/(m[good]**2).mean())*100)})
 else:
  r.update({k:None for k in ['relative_mag_mae_pct','relative_mag_rms_pct','relative_vector_mae_pct','relative_vector_rms_pct','vector_nrmse_pct']})
 r['absolute_pass']=bool(max(r['component_mae'])<=1 and max(r['component_rmse'])<=1.5 and r['mag_mae']<=1.5 and r['mag_rmse']<=1.5 and (not good.any() or (r['angle_mean']<=5 and r['angle_p95']<=10)))
 r['zero_n']=int(z.sum())
 if z.any():
  r['zero_pred_median_N']=float(np.median(pm[z]));r['zero_pred_p95_N']=float(np.percentile(pm[z],95));r['zero_pred_max_N']=float(pm[z].max());r['zero_false_positive_pct']={str(b):float((pm[z]>b).mean()*100) for b in [1,1.5,2,5]}
 return r

def masks(y,done):
 t,n,_=y.shape;time=np.broadcast_to(np.arange(t)[:,None],(t,n));local=time%1200;scene=time//1200;valid=local>=50;m=np.round(np.linalg.norm(y,axis=-1),3)
 ranges={'low':m<=15,'high':(m>15)&(m<=45),'0_2':m<2,'2_5':(m>=2)&(m<=5),'5_15':(m>5)&(m<=15),'15_30':(m>15)&(m<=30),'30_45':(m>30)&(m<=45)}
 out={'overall':valid}
 for k,v in ranges.items():out['force/'+k]=valid&v
 for i,s in enumerate(['ramp','release','reverse','sweep','full3d']):
  out[s]=valid&(scene==i)
  for k in ['low','high']:out[s+'/'+k]=valid&(scene==i)&ranges[k]
 # 1 second after reaching hold / after step change; these do not replace full results.
 steady=(((scene==0)&(local>=351))|((scene==1)&(((local>=101)&(local<401))|(local>=451)))|((scene==2)&(((local>=101)&(local<601))|(local>=651))))&valid
 out['settled_hold']=steady;out['continuous_dynamic']=valid&((scene==3)|(scene==4));out['other_transition_or_startup']=valid&~steady&~out['continuous_dynamic']
 for k in ['low','high']:out['settled_hold/'+k]=steady&ranges[k];out['continuous_dynamic/'+k]=out['continuous_dynamic']&ranges[k]
 reset=np.concatenate([np.zeros((1,n),bool),done[:-1]],axis=0);out['post_reset_first_observation']=valid&reset;out['not_post_reset_first_observation']=valid&~reset
 for s,t0 in [('release',1601),('reverse',3001)]:
  for phase,lo,hi in [('switch',0,5),('recovery',5,30),('later',30,1200-(t0%1200))]:out[s+'/'+phase]=valid&(time>=t0+lo)&(time<t0+hi)
 return out

def longest(a):
 # Input time x environments; longest contiguous true interval, in seconds.
 run=np.zeros(a.shape[1],int);mx=run.copy()
 for row in a:run=np.where(row,run+1,0);mx=np.maximum(mx,run)
 return mx*DT

def recovery(e,a,valid_angle,bound):
 ok=(e<=bound)&((~valid_angle)|(a<=10));n=e.shape[1];times=np.full(n,np.inf)
 for i in range(len(e)-9):
  hit=ok[i:i+10].all(0)&np.isinf(times);times[hit]=i*DT
 return times

def event_metrics(p,y,done):
 out={};mag=np.linalg.norm(y,axis=-1);reset=np.concatenate([np.zeros((1,done.shape[1]),bool),done[:-1]],axis=0)
 for name,t0 in EVENTS.items():
  eligible=mag[t0-1]>=2 if name in ['release','reverse'] else mag[t0]>=2
  pp=p[t0:t0+100,eligible].astype(float);yy=y[t0:t0+100,eligible].astype(float);rr=reset[t0:t0+100,eligible];m=np.linalg.norm(yy,axis=-1);pm=np.linalg.norm(pp,axis=-1);e=np.linalg.norm(pp-yy,axis=-1);de=abs(pm-m);a=angle_deg(pp,yy);av=m>=2
  ref=np.maximum(mag[t0-1,eligible],mag[t0,eligible]);times=recovery(e,a,av,1.5)
  def arr(v):return [None if not np.isfinite(x) else float(x) for x in v]
  per={'env':np.where(eligible)[0].tolist(),'reference_force_N':ref.tolist(),'peak_vector_100ms_N':e[:5].max(0).tolist(),'peak_mag_100ms_N':de[:5].max(0).tolist(),'peak_vector_1s_N':e[:50].max(0).tolist(),'peak_mag_1s_N':de[:50].max(0).tolist(),
   'peak_vector_100ms_reference_pct':(e[:5].max(0)/ref*100).tolist(),'integral_vector_error_2s_Ns':(e.sum(0)*DT).tolist(),'exceed_vector_1p5N_total_s':((e>1.5).sum(0)*DT).tolist(),'exceed_vector_1p5N_longest_s':longest(e>1.5).tolist(),'exceed_angle_10deg_total_s':(((a>10)&av).sum(0)*DT).tolist(),
   'settling_start_s':arr(times),'recovery_confirmed_s':arr(times+.2),'physical_recovery_start_s':arr(times+DT),'physical_confirmation_s':arr(times+.2),'any_reset_within2s':rr.any(0).tolist()}
  per['peak_angle_100ms_deg']=[float(a[:5,i][av[:5,i]].max()) if av[:5,i].any() else None for i in range(len(times))]
  per['recovered_without_reset_before_confirmation']=[bool(np.isfinite(v) and not rr[:int(round(v/DT))+10,i].any()) for i,v in enumerate(times)]
  valid_times=np.sort(times)
  def qtime(q):
   val=valid_times[max(0,int(np.ceil(q*len(times)))-1)]+DT;return None if not np.isfinite(val) else float(val)
  summary={'eligible_envs':len(times),'recovered':int(np.isfinite(times).sum()),'censored':int(np.isinf(times).sum()),'recovered_without_reset':int(sum(per['recovered_without_reset_before_confirmation'])),'settling_start_median_s_all_events':qtime(.5),'settling_start_p95_s_all_events':qtime(.95),'time_origin':'actual force onset; recovery start includes first20ms physics interval; legacy per-environment settling_start_s is relative to first observation','any_reset_envs':int(rr.any(0).sum())}
  for k in ['peak_vector_100ms_N','peak_mag_100ms_N','peak_vector_1s_N','peak_mag_1s_N','peak_vector_100ms_reference_pct','integral_vector_error_2s_Ns','exceed_vector_1p5N_total_s','exceed_vector_1p5N_longest_s']:
   z=np.array(per[k]);summary[k]={'mean':float(z.mean()),'p95':float(np.percentile(z,95)),'max':float(z.max())}
  summary['peak_angle_100ms_deg']={'p95':float(np.percentile([x for x in per['peak_angle_100ms_deg'] if x is not None],95)) if any(x is not None for x in per['peak_angle_100ms_deg']) else None}
  deadlines=[.1,.2,.3,.5,1,2];grid={}
  for b in [.5,1,1.5,2,3]:
   rt=recovery(e,a,av,b)+DT;grid[str(b)]={str(t):float((rt<=t+1e-9).mean()*100) for t in deadlines}
  out[name]={'summary':summary,'per_environment':per,'recovery_start_deadline_budget_grid_pct':grid}
 return out
