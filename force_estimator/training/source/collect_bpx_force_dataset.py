#!/usr/bin/env python3
"""Collect BPX 45-D policy histories and body-frame force labels in IsaacLab."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--task", default="Isaac-Velocity-Flat-BpxPro-ForceAware-Play-v0"); parser.add_argument("--num-envs", type=int, default=512); parser.add_argument("--steps", type=int, default=1200); AppLauncher.add_app_launcher_args(parser); args = parser.parse_args()
app = AppLauncher(args).app
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import bpx_rma
from bpx_rma.mdp.force import get_force_state

def world_to_body(force_world: torch.Tensor, q_wb: torch.Tensor) -> torch.Tensor:
    q = q_wb / q_wb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    r00 = 1 - 2*(y*y+z*z); r01 = 2*(x*y-z*w); r02 = 2*(x*z+y*w)
    r10 = 2*(x*y+z*w); r11 = 1 - 2*(x*x+z*z); r12 = 2*(y*z-x*w)
    r20 = 2*(x*z-y*w); r21 = 2*(y*z+x*w); r22 = 1 - 2*(x*x+y*y)
    return torch.stack((r00*force_world[:,0]+r10*force_world[:,1]+r20*force_world[:,2], r01*force_world[:,0]+r11*force_world[:,1]+r21*force_world[:,2], r02*force_world[:,0]+r12*force_world[:,1]+r22*force_world[:,2]), dim=-1)

def main():
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point"); env_cfg.scene.num_envs = args.num_envs
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg); env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    cfg_dict = agent_cfg.to_dict(); cfg_dict["actor"] = {k: v for k, v in cfg_dict.get("actor", {}).items() if k in {"class_name", "hidden_dims", "activation", "obs_normalization", "distribution_cfg", "encoder_hidden_dims", "encoder_activation", "latent_dim"}}
    cfg_dict["critic"] = {k: v for k, v in cfg_dict.get("critic", {}).items() if k in {"class_name", "hidden_dims", "activation", "obs_normalization", "distribution_cfg", "encoder_hidden_dims", "encoder_activation", "latent_dim"}}
    runner = OnPolicyRunner(env, cfg_dict, log_dir=None, device=agent_cfg.device); runner.load(args.checkpoint); policy = runner.get_inference_policy(device=agent_cfg.device)
    base = env.unwrapped; state = get_force_state(base); robot = base.scene["robot"]
    history = torch.zeros(args.num_envs, 25, 45, device=base.device); xs=[]; ys=[]; qs=[]
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args.steps):
            frame = obs["policy"][:, :45]
            history[:, :-1] = history[:, 1:]; history[:, -1] = frame
            action = policy(obs); obs, _, _, _ = env.step(action)
            if step >= 24:
                q_now = robot.data.root_quat_w
                xs.append(history.cpu().numpy().astype(np.float32)); ys.append(world_to_body(state.force_w, q_now).cpu().numpy().astype(np.float32)); qs.append(q_now.cpu().numpy().astype(np.float32))
            done = getattr(base, "reset_buf", None)
            if done is not None: history[done] = 0
    env.close(); x=np.concatenate(xs); y=np.concatenate(ys); q=np.concatenate(qs); n=len(x); cut1=int(n*.7); cut2=int(n*.85); out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for name, sl in (("train", slice(0,cut1)), ("validation", slice(cut1,cut2)), ("test", slice(cut2,None))): np.savez_compressed(out/f"{name}.npz", sensor_history=x[sl], force_body=y[sl], q_wb=q[sl])
    (out/"manifest.json").write_text(json.dumps({"schema":"bpx-policy-history-v2","input_shape":[25,45],"target":"body_force_N","history_order":"oldest_to_newest","num_samples":n,"checkpoint":args.checkpoint}, indent=2)+"\n")

if __name__ == "__main__": main()
    
app.close()
