from pathlib import Path
import sys
import os,json,time,subprocess,hashlib
O=Path('logs/rsl_rl/go2_velocity/convergence_v1');R=Path('logs/convergence_v1');R.mkdir(exist_ok=True,parents=True)
PY=os.environ.get('FORCE_ESTIMATOR_PYTHON', sys.executable)
# Freeze force-plan seeds before collecting any result; never feed these into optimizer.
lock={'seeds':[29901,29902],'comparison':'240-epoch start vs validation-selected continuation','gpu':0,'policy':'force_estimator_ft_20261014_fullrange_v6/model_fullrange_v6.pt','use':'final diagnostics, not training or selection'}
json.dump(lock,open(O/'blind_plan.json','w'),indent=2)
for name in ['start','selected']:
 if name=='selected':
  while not (O/'convergence.json').exists():time.sleep(10)
  c=json.load(open(O/'convergence.json'))
  if not c['converged']:raise RuntimeError('Training stopped without convergence; no final-converged claim allowed')
 path=Path('logs/rsl_rl/go2_velocity/causal_v3_extended/aligned_window_720.pt') if name=='start' else O/'best.pt'
 for seed in lock['seeds']:
  with open(R/f'{name}_{seed}.log','w') as f:
   subprocess.run([PY,'collect_causal_fp32.py',name,str(seed),str(R/f'{name}_{seed}.npz')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',BALANCE_MODEL=str(path)),stdout=f,stderr=subprocess.STDOUT,check=True)
(O/'COLLECTION_COMPLETE').write_text('Final paired trajectories collected; inspect evaluation before acceptance.\n')
