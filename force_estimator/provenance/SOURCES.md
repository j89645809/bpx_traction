# Source map

The clean package was reconstructed from read-only files on `zy-server` (2026-10-27). The original files are represented under `../training/legacy_mjlab/` and the clean package sources.

| Clean file | Source file | Role |
|---|---|---|
| `src/force_estimator/model.py` | `unitree_rl_mjlab_self_pull/src/tasks/velocity/rl/force_estimator.py` | 25-frame temporal force estimator |
| `src/force_estimator/world_model.py` | `lead_clean_20260916/source/experiments/world_force_adapter_v1/world_force/model.py` | causal `MissingStateTCN` |
| `config/world_force.yaml` | `lead_clean_20260916/source/experiments/world_force_adapter_v1/config/world_force.yaml` | frames, timing, confidence and safety settings |
| `config/normalization.npz` | `unitree_rl_mjlab_self_pull/estimators/missing_state_world_v2/normalization.npz` | 64-channel mean/std |
| `provenance/` references | `src/tasks/velocity/mdp/tether_events.py`, Gate41 configs, runner, protocols and reports | simulator/event context and validation evidence |

The clean package intentionally has no dependency on the original Go2/mjlab repository. The simulator files remain reference-only because they require the original robot assets and environment.
