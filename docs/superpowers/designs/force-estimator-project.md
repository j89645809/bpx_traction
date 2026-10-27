# Force Estimator Project Design

**Goal:** make the force-estimation work visible and reproducible to the upstream BPX collaborator from one clean directory in this fork.

**Audience:** the collaborator reviewing `feat/force-aware-rma`, plus anyone reproducing the estimator without the original Go2/mjlab checkout.

**Scope:** package the temporal force estimator and the missing-state world-force estimator, their exact tensor contracts, normalization statistics, configuration, validation tests, and provenance. Keep simulator-specific event code as clearly marked reference material.

**Non-goals:** do not claim that the archived Go2/mjlab modules are a standalone BPX runtime; do not add model checkpoints, ONNX weights, datasets, robot assets, or credentials.

**Public structure:**

- `force_estimator/README.md` — entry point, pipeline, contracts, setup, validation, provenance.
- `force_estimator/pyproject.toml` — minimal CPU install and test metadata.
- `force_estimator/src/force_estimator/model.py` — cleaned temporal `ForceEstimator`.
- `force_estimator/src/force_estimator/world_model.py` — cleaned causal `MissingStateTCN`.
- `force_estimator/src/force_estimator/contracts.py` — named dimensions and output semantics.
- `force_estimator/src/force_estimator/preprocessing.py` — normalization and shape checks.
- `force_estimator/config/world_force.yaml` — frame, timing, confidence, and safety contract.
- `force_estimator/config/normalization.npz` — 64-channel mean/std from v2.
- `force_estimator/tests/` — deterministic shape, causality, bounds, and normalization tests.
- `force_estimator/provenance/` — source mapping and archived simulator references.

The top-level BPX training package remains separate. The force estimator is an upstream producer of `[Fx, Fy, Fz]` plus confidence; the BPX ForceAware policy consumes normalized body-frame force and confidence as the final four observation values.
