"""Closed-loop BPX ForceAware acceptance: estimator output drives the locomotion policy."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--estimator',required=True); p.add_argument('--normalization',required=True); p.add_argument('--output',required=True); p.add_argument('--task',default='Isaac-Velocity-Flat-BpxPro-ForceAware-Play-v0'); p.add_argument('--steps',type=int,default=800); AppLauncher.add_app_launcher_args(p); args=p.parse_args()
app=AppLauncher(args).app
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import bpx_rma
from bpx_rma.mdp.force import get_force_state

def main():
    env_cfg=load_cfg_from_registry(args.task,'env_cfg_entry_point'); agent_cfg=load_cfg_from_registry(args.task,'rsl_rl_cfg_entry_point')
    env=gym.make(args.task,cfg=env_cfg); env=RslRlVecEnvWrapper(env,clip_actions=agent_cfg.clip_actions)
    cfg=agent_cfg.to_dict(); allowed={'class_name','hidden_dims','activation','obs_normalization','distribution_cfg','encoder_hidden_dims','encoder_activation','latent_dim'}
    cfg['actor']={k:v for k,v in cfg.get('actor',{}).items() if k in allowed}; cfg['critic']={k:v for k,v in cfg.get('critic',{}).items() if k in allowed}
    runner=OnPolicyRunner(env,cfg,log_dir=None,device=agent_cfg.device); runner.load(args.checkpoint); policy=runner.get_inference_policy(device=agent_cfg.device)
    model=torch.jit.load(args.estimator,map_location=env.unwrapped.device).eval(); z=np.load(args.normalization); mean=torch.as_tensor(z['mean'],device=env.unwrapped.device); std=torch.as_tensor(z['std'],device=env.unwrapped.device)
    base=env.unwrapped; st=get_force_state(base); n=base.num_envs; hist=torch.zeros(n,25,45,device=base.device); obs=env.get_observations(); err=[]; true=[]; pred=[]; rewards=[]; falls=0
    with torch.inference_mode():
      for step in range(args.steps):
        frame=obs['policy'][:,:45]; hist[:,:-1]=hist[:,1:]; hist[:,-1]=frame
        f,c=model((hist-mean)/std); st.est_force_b[:]=f; st.confidence[:]=c.clamp(0,1)
        obs=env.get_observations(); action=policy(obs); obs,reward,done,_=env.step(action)
        t=st.true_force_b.detach().cpu().numpy(); q=f.detach().cpu().numpy(); err.append(np.abs(q-t)); true.append(t); pred.append(q); rewards.append(reward.detach().cpu().numpy()); falls += int(done.sum().item()) if torch.is_tensor(done) else 0
        if torch.is_tensor(done) and done.any(): hist[done]=0
    env.close(); e=np.concatenate(err); t=np.concatenate(true); q=np.concatenate(pred); r=np.concatenate(rewards); mag=np.linalg.norm(t,axis=1); pm=np.linalg.norm(q,axis=1); mask=mag>=2; co=(q[mask]*t[mask]).sum(1)/(np.maximum(np.linalg.norm(q[mask],axis=1)*mag[mask],1e-6)); ang=np.degrees(np.arccos(np.clip(co,-1,1)))
    out={'steps':args.steps,'envs':n,'frames':int(len(t)),'component_mae_N':e.mean(0).tolist(),'magnitude_mae_N':float(np.abs(pm-mag).mean()),'direction_mean_deg_above_2N':float(ang.mean()),'direction_p95_deg_above_2N':float(np.percentile(ang,95)),'mean_reward':float(r.mean()),'done_events':falls}
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
app.close()
