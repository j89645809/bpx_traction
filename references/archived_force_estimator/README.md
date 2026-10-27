# Archived force-estimator references

This directory contains the source files requested for reproducing and checking the upstream force-estimation work that feeds the BPX force-aware RMA project. The files were recovered read-only from `zy@111.0.18.168` on 2026-10-27:

- `/home/zy/old_data/lead_clean_20260916/source`
- `/home/zy/old_data/unitree_rl_mjlab_self_pull`

The original relative paths are preserved below each source directory. These are Go2/mjlab reference modules; they are intentionally kept outside `bpx_rma/` and are not imported by the BPX training package. They provide the missing implementation and provenance for the force-event generator, estimator, world-force adapter, Gate41 configuration, runner command shaping, protocols, metrics, and normalization statistics.

Key references:

- `lead_clean_20260916/source/src/tasks/velocity/mdp/tether_events.py`: tether-force event implementation and force application details.
- `lead_clean_20260916/source/src/tasks/velocity/config/go2/env_cfgs.py`: Gate41 World/Slowdown environment configuration.
- `lead_clean_20260916/source/src/tasks/velocity/mdp/rewards.py`: corresponding reward terms.
- `lead_clean_20260916/source/src/tasks/velocity/rl/runner.py`: `_follow_command` and `_phase_from_command` implementations.
- `unitree_rl_mjlab_self_pull/src/tasks/velocity/rl/force_estimator.py`: archived 82-dimension force-estimator implementation.
- `*/experiments/world_force_adapter_v1/world_force/model.py`: `MissingStateTCN` implementation.
- `configs/missing_state_world_v2_normalization.npz`: normalization statistics for the v2 missing-state estimator.

The archive is reference material rather than a standalone installable package because it depends on the original Go2/mjlab environment and assets.
