"""Inject a real TorchScript force estimator into ForceAware observations."""
from __future__ import annotations
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from .mdp.force import get_force_state

class ForceEstimatorObservationWrapper(gym.Wrapper):
    """Update ``est_force_b`` before policy observations are assembled.

    The wrapped manager environment still owns force application and true-force
    bookkeeping. This wrapper only replaces the synthetic estimator cache with
    the deployed estimator output, preserving the policy's 49-D contract.
    """
    def __init__(self, env, model_path: str, normalization_path: str):
        super().__init__(env)
        self.base = env.unwrapped
        self.model = torch.jit.load(model_path, map_location=self.base.device).eval()
        z = np.load(normalization_path)
        self.mean = torch.as_tensor(z["mean"], device=self.base.device, dtype=torch.float32)
        self.std = torch.as_tensor(z["std"], device=self.base.device, dtype=torch.float32)
        self.history = torch.zeros(self.base.num_envs, 25, 45, device=self.base.device)
        self._last_frame = None

    def _inject(self, obs):
        frame = obs["policy"][:, :45]
        # Avoid shifting twice if a caller requests observations twice in a step.
        if self._last_frame is None or not torch.equal(frame, self._last_frame):
            self.history[:, :-1] = self.history[:, 1:]
            self.history[:, -1] = frame
            self._last_frame = frame.clone()
        with torch.inference_mode():
            force, confidence = self.model((self.history - self.mean) / self.std)
        st = get_force_state(self.base)
        st.est_force_b[:] = force.clamp(-45.0, 45.0)
        st.confidence[:] = confidence.clamp(0.0, 1.0)
        return self.env.get_observations()

    def get_observations(self):
        return self._inject(self.env.get_observations())

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.history.zero_(); self._last_frame = None
        if isinstance(out, tuple):
            return self._inject(out[0]), out[1]
        return self._inject(out)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return self._inject(obs), reward, terminated, truncated, info
        obs, reward, done, info = out
        return self._inject(obs), reward, done, info
