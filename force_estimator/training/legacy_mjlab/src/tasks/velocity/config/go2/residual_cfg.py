from __future__ import annotations

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from src.tasks.velocity import mdp

from .env_cfgs import unitree_go2_self_pull_env_cfg
from .rl_cfg import unitree_go2_ppo_runner_cfg


def unitree_go2_self_pull_residual_env_cfg(play: bool = False):
    cfg = unitree_go2_self_pull_env_cfg(play=play)
    cfg.observations["residual_gate"] = ObservationGroupCfg(
        terms={
            "residual_gate": ObservationTermCfg(func=mdp.residual_gate_placeholder)
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
    )
    cfg.observations["emergency_features"] = ObservationGroupCfg(
        terms={
            "emergency_features": ObservationTermCfg(
                func=mdp.emergency_features_placeholder
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
    )
    cfg.observations["emergency_gate"] = ObservationGroupCfg(
        terms={
            "emergency_gate": ObservationTermCfg(func=mdp.emergency_gate_placeholder)
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
    )
    tether = cfg.events["tether_pull"].params
    tether.update(
        {
            "vertical_force_range": (0.0, 0.0),
            "gate_aligned_training": True,
            "gate_force_range": (45.0, 50.0),
            "gate_force_replay": ((15.0, 30.0, 0.30), (30.0, 45.0, 0.30)),
            "gate_recovery_state_fraction": 0.40,
            "gate_hard_direction_weights": (
                0.14,
                0.14,
                0.14,
                0.14,
                0.10,
                0.10,
                0.10,
                0.14,
            ),
        }
    )
    return cfg


def unitree_go2_residual_ppo_runner_cfg():
    cfg = unitree_go2_ppo_runner_cfg()
    cfg.obs_groups["actor"] = (
        "actor",
        "residual_gate",
        "emergency_features",
        "emergency_gate",
    )
    cfg.actor.class_name = "src.tasks.velocity.rl.residual_policy:ResidualMLPModel"
    cfg.seed = 20260917
    cfg.algorithm.learning_rate = 5e-5
    cfg.algorithm.schedule = "fixed"
    cfg.algorithm.num_learning_epochs = 1
    cfg.algorithm.num_mini_batches = 1
    cfg.save_interval = 15
    cfg.wandb_project = "lead_clean1"
    return cfg
