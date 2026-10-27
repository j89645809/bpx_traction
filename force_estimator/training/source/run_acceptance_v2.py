import os,sys,json,subprocess,hashlib,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
O=Path('logs/rsl_rl/go2_velocity/acceptance_v2');R=Path('logs/acceptance_v2');O.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
PY=os.environ.get('FORCE_ESTIMATOR_PYTHON', sys.executable);candidate=Path('logs/rsl_rl/go2_velocity/convergence_v1/best.pt');base=Path('logs/rsl_rl/go2_velocity/causal_v3_extended/aligned_window_720.pt')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(candidate)=='92b52bf4fe68a4f08ca258c518aa00f5e5232d85621b1ba13cf50d381de55639'
seeds=list(range(30101,30111));rng=np.random.default_rng(20261020);first=set(rng.choice(seeds,5,replace=False).tolist());gpu_assignment=[4,4,4,0,4,4,4,0,4,4]
plan=[{'seed':s,'gpu':gpu_assignment[i],'order':['selected','start'] if s in first else ['start','selected']} for i,s in enumerate(seeds)]
lock={'plan':plan,'model_sha256':{'selected':sha(candidate),'start':sha(base)},'collector_sha256':sha('collect_causal_fp32.py'),'policy_sha256':sha('logs/rsl_rl/go2_velocity/force_estimator_ft_20261014_fullrange_v6/model_fullrange_v6.pt'),'protocol_sha256':sha(O/'PROTOCOL.md'),'numpy':np.__version__,'application_budgets':'undetermined'}
if (O/'lock.json').exists():assert json.load(open(O/'lock.json'))==lock
else:json.dump(lock,open(O/'lock.json','w'),indent=2)
# Per-device semaphores keep at most3 GPU4 collectors and1 GPU0 collector; no shared outputs.
sems={4:threading.Semaphore(3),0:threading.Semaphore(1)}
def pair(row):
 with sems[row['gpu']]:
  for name in row['order']:
   path=candidate if name=='selected' else base;out=R/f'{name}_{row["seed"]}.npz'
   if out.exists() and out.with_suffix('.json').exists() and out.with_suffix('.events.json').exists():continue
   logfile=R/f'{name}_{row["seed"]}.log'
   if logfile.exists():logfile.rename(logfile.with_suffix(f'.previous_{int(time.time())}.log'))
   print('START',row,name,flush=True)
   with open(logfile,'w') as f:
    subprocess.run([PY,'collect_causal_fp32.py',name,str(row['seed']),str(out)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(row['gpu']),BALANCE_MODEL=str(path)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
   print('DONE',name,row['seed'],flush=True)
with ThreadPoolExecutor(max_workers=10) as pool:
 jobs=[pool.submit(pair,row) for row in plan]
 for job in jobs:job.result()
(O/'COLLECTION_COMPLETE').write_text('20 frozen-model runs collected; not a pass decision.\n')
print('ALL_COLLECTION_COMPLETE',flush=True)
