from pathlib import Path
import json,re
O=Path('logs/rsl_rl/go2_velocity/acceptance_v2');R=Path('logs/acceptance_v2');done=[];active={}
for seed in range(30101,30111):
 for name in ['selected','start']:
  k=f'{name}_{seed}'
  if (R/f'{k}.events.json').exists():done.append(k);continue
  f=R/f'{k}.log'
  if f.exists():
   s=f.read_text(errors='replace');m=re.findall(r'PROGRESS (\d+) 6000',s);active[k]=int(m[-1]) if m else 0
print(json.dumps({'complete':len(done),'planned':20,'active_steps':active,'paired_complete':sum(all(f'{n}_{s}' in done for n in ['selected','start']) for s in range(30101,30111))}))
for log in ['run.log','analyze.log']:
 p=O/log
 if p.exists() and 'Traceback' in p.read_text(errors='replace'):print('ERROR',log,p.read_text(errors='replace')[-2500:])
