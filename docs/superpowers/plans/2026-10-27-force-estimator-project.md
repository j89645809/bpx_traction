# Force Estimator Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** publish a clean, self-contained force-estimation project inside the BPX fork so the upstream collaborator can inspect, validate, and reuse it.

**Architecture:** create a small PyTorch package under `force_estimator/`, separating model definitions, tensor contracts, preprocessing, configuration, tests, and provenance. Keep the existing BPX RMA package unchanged except for documentation links; preserve the old archive as evidence, while making the new package the review entry point.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, PyYAML, pytest.

**Spec:** `docs/superpowers/designs/force-estimator-project.md`

## Global Constraints

- The estimator outputs base-frame force in newtons and confidence in `[0, 1]`.
- The world model consumes `[B, 25, 64]` and returns values `[B, 15]` plus confidence `[B, 4]`.
- The causal TCN must not use future timesteps when producing the last timestep output.
- The BPX policy consumes normalized body-frame force `[Fx, Fy, Fz] / 45`, clipped to `[-1, 1]`, plus force confidence.
- No checkpoints, ONNX weights, datasets, robot assets, secrets, or server credentials are added.

## Review Focus

- Input shape and output semantics are explicit and tested.
- Normalization rejects wrong feature dimensions and protects against zero standard deviation.
- Causal output is invariant to changes after the queried timestep.
- Bounded outputs match the documented physical ranges.
- Provenance distinguishes runnable package code from Go2/mjlab-only references.

### Task 1: Package contracts and preprocessing

**Files:**
- Create: `force_estimator/src/force_estimator/contracts.py`
- Create: `force_estimator/src/force_estimator/preprocessing.py`
- Test: `force_estimator/tests/test_contracts.py`

- [ ] Define named dimensions and output slices.
- [ ] Implement normalization with NumPy/Torch-compatible shape checks.
- [ ] Test valid and invalid shapes and zero-std handling.

### Task 2: Clean model package

**Files:**
- Create: `force_estimator/src/force_estimator/model.py`
- Create: `force_estimator/src/force_estimator/world_model.py`
- Create: `force_estimator/src/force_estimator/__init__.py`
- Test: `force_estimator/tests/test_models.py`

- [ ] Port the archived `ForceEstimator` and `MissingStateTCN` without changing tensor semantics.
- [ ] Test output shapes, bounds, confidence range, and causal behavior.

### Task 3: Config, statistics, provenance, and README

**Files:**
- Create: `force_estimator/config/world_force.yaml`
- Create: `force_estimator/config/normalization.npz`
- Create: `force_estimator/provenance/SOURCES.md`
- Create: `force_estimator/README.md`
- Modify: `README.md`

- [ ] Add the recovered configuration and statistics with source paths.
- [ ] Document installation, quick checks, model contracts, BPX handoff, limitations, and source provenance.
- [ ] Link the new project from the root README.

### Task 4: Verify and publish

- [ ] Run package tests and Python compilation.
- [ ] Check repository for secrets, checkpoints, and untracked generated files.
- [ ] Commit and push to `feat/force-aware-rma`.
