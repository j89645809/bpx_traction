"""State-hidden, physically applied tether disturbances for pull-following."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse


class apply_tether_pull:
  """Apply a sustained random 3D tether force without exposing it to the policy."""

  def __init__(self, cfg, env):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._num_bodies = len(self._body_ids) if isinstance(self._body_ids, list) else 1
    self._remaining = torch.zeros(self._num_envs, device=self._device)
    self._cooldown = torch.zeros(self._num_envs, device=self._device)
    self._active = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
    self.manual_enabled = False
    # Manual certification may request a different world/body force per
    # vectorized environment. A single 3-vector remains supported and is
    # broadcast by set_manual_force().
    self.manual_force_b = torch.zeros((self._num_envs, 3), device=self._device)
    self.last_force_b = torch.zeros((self._num_envs, 3), device=self._device)
    self.last_force_w = torch.zeros((self._num_envs, 3), device=self._device)
    self.force_frame = None
    self.frame_id = 0
    self._bilateral_cycle_index = -1
    # One independently sampled Fx per environment, refreshed each pull cycle.
    self._bilateral_force_x = torch.zeros(self._num_envs, device=self._device)
    self._static_cycle_index = -1
    self._static_target_force = torch.zeros((self._num_envs, 3), device=self._device)
    self._jump_cycle_index = -1
    self._jump_force_before = torch.zeros((self._num_envs, 3), device=self._device)
    self._jump_force_after = torch.zeros((self._num_envs, 3), device=self._device)
    self._traction_cycle_index = -1
    self._traction_rotation = torch.zeros(self._num_envs, device=self._device)
    # The certification matrix has 192 cases because each of these 96 unique
    # force vectors is tested once while held and once with a mid-run release.
    # Training already alternates pull and cooldown, so cycling the unique
    # vectors here gives both the held-force and post-release portions without
    # duplicating otherwise identical force samples.
    matrix: list[tuple[float, float, float]] = []
    directions = (
      (1.0, 0.0),
      (-1.0, 0.0),
      (0.0, 1.0),
      (0.0, -1.0),
      (2.0**-0.5, 2.0**-0.5),
      (2.0**-0.5, -(2.0**-0.5)),
      (-(2.0**-0.5), 2.0**-0.5),
      (-(2.0**-0.5), -(2.0**-0.5)),
    )
    for dx, dy in directions:
      for magnitude in (5.0, 10.0, 20.0, 30.0, 40.0):
        matrix.append((dx * magnitude, dy * magnitude, 0.0))
    for force_z in (5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0):
      matrix.append((0.0, 0.0, force_z))
    for dx, dy in directions:
      for horizontal, vertical in ((10.0, 5.0), (20.0, 10.0), (30.0, 15.0)):
        matrix.append((dx * horizontal, dy * horizontal, vertical))
        matrix.append((dx * horizontal, dy * horizontal, -vertical))
    self._certification_force_matrix = torch.tensor(
      matrix, dtype=torch.float32, device=self._device
    )
    if self._certification_force_matrix.shape != (96, 3):
      raise RuntimeError("certification training matrix must contain 96 unique vectors")
    hard_40n = torch.isclose(
      torch.linalg.vector_norm(self._certification_force_matrix[:, :2], dim=1),
      torch.tensor(40.0, device=self._device),
    ) & torch.isclose(
      self._certification_force_matrix[:, 2],
      torch.tensor(0.0, device=self._device),
    )
    # A short polish stage repeats each of the eight pure-horizontal 40 N
    # vectors three additional times. All original 96 vectors remain present,
    # so improving the boundary does not discard lower-force or 3-D coverage.
    self._certification_hard40_matrix = torch.cat(
      (
        self._certification_force_matrix,
        self._certification_force_matrix[hard_40n].repeat(3, 1),
      ),
      dim=0,
    )
    # Environment i starts at vector i mod 96. With >=96 environments every
    # force vector is present from the first training step onward.
    self._certification_sample_index = torch.arange(
      self._num_envs, device=self._device, dtype=torch.long
    ) % len(matrix)
    self._certification_hard40_initialized = False
    # Gate 2 keeps a force-free locomotion cohort so pull training cannot erase
    # the active gait learned in Gate 1.
    self.retention_mask = torch.zeros(
      self._num_envs, dtype=torch.bool, device=self._device
    )
    self.retention_command = torch.zeros(
      (self._num_envs, 2), device=self._device
    )

  def __call__(
    self,
    env,
    env_ids,
    horizontal_force_range: tuple[float, float],
    vertical_force_range: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    force_frame: str,
    body_point_offset_ranges: tuple[
      tuple[float, float], tuple[float, float], tuple[float, float]
    ] = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    fixed_force_b: tuple[float, float, float] | None = None,
    fixed_cycle_s: float | None = None,
    bilateral_fx_curriculum: bool = False,
    write_follow_command: bool = True,
    force_free_warmup_steps: int = 0,
    bilateral_force_magnitude_range: tuple[float, float] | None = None,
    retention_env_fraction: float = 0.0,
    retention_command_range: tuple[float, float] = (0.2, 0.5),
    force_direction_mode: str = "random_horizontal",
    fx_retention_fraction: float = 0.0,
    horizontal_force_curriculum_steps: tuple[tuple[int, float], ...] | None = None,
    direct_lateral_retention_fraction: float = 0.0,
    direct_lateral_command_range: tuple[float, float] = (0.2, 0.4),
    vertical_endpoint_fraction: float = 0.0,
    static_ramp_release: bool = False,
    static_cycle_s: float = 4.0,
    static_target_force_range: tuple[float, float] = (10.0, 40.0),
    static_fixed_force: tuple[float, float, float] | None = None,
    jump_training: bool = False,
    jump_cycle_s: float = 4.0,
    jump_pre_s: float = 1.0,
    jump_post_s: float = 1.5,
    jump_force_range: tuple[float, float] = (12.0, 40.0),
    jump_fixed_before: tuple[float, float, float] | None = None,
    jump_fixed_after: tuple[float, float, float] | None = None,
    jump_targeted_turns: bool = False,
    traction_actor_curriculum: bool = False,
    traction_actor_hard_focus: bool = False,
    traction_actor_extreme_focus: bool = False,
    traction_actor_recovery_focus: bool = False,
    traction_actor_contact_focus: bool = False,
    traction_actor_v40_focus: bool = False,
    traction_actor_v41_focus: bool = False,
    traction_actor_v44_focus: bool = False,
    traction_actor_v45_focus: bool = False,
    traction_actor_v46_focus: bool = False,
    traction_actor_v47_focus: bool = False,
    traction_actor_v63_focus: bool = False,
    traction_actor_v64_focus: bool = False,
    traction_actor_v65_focus: bool = False,
    traction_actor_v67_focus: bool = False,
    traction_actor_v68_focus: bool = False,
    traction_actor_v69_focus: bool = False,
    traction_actor_v70_focus: bool = False,
    traction_actor_v79_focus: bool = False,
    traction_actor_v82_focus: bool = False,
    traction_actor_v93_focus: bool = False,
    traction_actor_v94_focus: bool = False,
    traction_actor_v95_focus: bool = False,
    traction_actor_v98_focus: bool = False,
    traction_actor_v99_focus: bool = False,
    traction_actor_v100_focus: bool = False,
    traction_cycle_s: float = 8.0,
  ) -> None:
    del env_ids, asset_cfg
    if force_frame not in ("legacy_body", "world_fixed"):
      raise ValueError(
        "tether.force_frame must be explicit: legacy_body or world_fixed"
      )
    self.force_frame = force_frame
    if self.manual_enabled:
      ids = torch.arange(self._num_envs, device=self._device)
      force_requested = self.manual_force_b
      force_b, force_w_selected = self._resolve_force_frames(
        force_requested, self._asset.data.root_link_quat_w, force_frame
      )
      force_w = torch.zeros((self._num_envs, self._num_bodies, 3), device=self._device)
      force_w[:, 0] = force_w_selected
      zeros = torch.zeros_like(force_w)
      self._asset.write_external_wrench_to_sim(force_w, zeros, env_ids=ids, body_ids=self._body_ids)
      self.last_force_b[:] = force_b
      self.last_force_w[:] = force_w_selected
      if write_follow_command:
        self._write_follow_command(env, ids, force_b)
      return
    if traction_actor_curriculum:
      # Ten explicit physical templates cover stand, ramp, sudden pull,
      # release, +/-90, 180, high-force 3-D reversal and combined XY/Z jump.
      # Horizontal rotation is resampled each cycle so these are behavior
      # families rather than memorized world directions. Environment index is
      # divided by three because actor source cohorts are interleaved mod 3;
      # every teacher/nominal/robust cohort therefore sees all ten templates.
      ids = torch.arange(self._num_envs, device=self._device)
      cycle_steps = max(1, int(traction_cycle_s / self._step_dt))
      cycle_index = env.common_step_counter // cycle_steps
      if cycle_index != self._traction_cycle_index:
        self._traction_cycle_index = cycle_index
        self._traction_rotation.uniform_(-torch.pi, torch.pi)
      group = ((ids // 3) + cycle_index) % 10
      if traction_actor_hard_focus:
        # v24 oversamples the families that failed v23 certification while
        # retaining one ramp and one sudden-pull template for anchoring.
        hard_templates = torch.tensor((3, 5, 7, 8, 9, 1, 2, 3, 5, 9), device=self._device)
        group = hard_templates[group.long()]
      if traction_actor_extreme_focus:
        extreme_templates = torch.tensor((3, 7, 4, 7, 3, 9, 2, 7, 4, 3), device=self._device)
        group = extreme_templates[group.long()]
      if traction_actor_recovery_focus:
        # Recovery cohort: retain the zero-force anchor while oversampling
        # release, 90/180-degree turns and 3-D reversals that failed gates.
        recovery_templates = torch.tensor(
          (0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9),
          device=self._device,
        )
        recovery_index = ((ids // 3) + cycle_index) % recovery_templates.shape[0]
        group = recovery_templates[recovery_index.long()]
      if traction_actor_contact_focus:
        # v31 is driven by measured v30 failure attribution: every observed
        # termination was an illegal non-foot contact. Preserve all ten
        # behavior families while giving extra probability to the four
        # contact-triggering templates (ramp, diagonal 3-D, -90, high reverse)
        # and retaining no-pull/low-force release anchors against forgetting.
        contact_templates = torch.tensor(
          (0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 8, 9),
          device=self._device,
        )
        contact_index = ((ids // 3) + cycle_index) % contact_templates.shape[0]
        group = contact_templates[contact_index.long()]
      if traction_actor_v40_focus:
        # v40 targets the measured v39 failures: zero-command drift (0),
        # diagonal/release (3), lateral turns (4/5), high 3-D reversal (7),
        # and the low-force 90-degree start/release case (8).  Every original
        # family remains present to prevent catastrophic forgetting.
        v40_templates = torch.tensor(
          (0, 0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 7, 7, 8, 8, 9),
          device=self._device,
        )
        v40_index = ((ids // 3) + cycle_index) % v40_templates.shape[0]
        group = v40_templates[v40_index.long()]
      if traction_actor_v41_focus:
        # v41 starts from the best v40 checkpoint. Concentrate on the two
        # remaining actor failures (10 N +/-90/release and high reverse 3-D)
        # while retaining anchors for no-pull, ramp, sudden pull and 180 deg.
        v41_templates = torch.tensor(
          (0, 0, 1, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9),
          device=self._device,
        )
        v41_index = ((ids // 3) + cycle_index) % v41_templates.shape[0]
        group = v41_templates[v41_index.long()]
      if traction_actor_v45_focus:
        v45_templates = torch.tensor(
          (0, 0, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 8, 8, 8, 9),
          device=self._device,
        )
        v45_index = ((ids // 3) + cycle_index) % v45_templates.shape[0]
        group = v45_templates[v45_index.long()]
      if traction_actor_v46_focus:
        v46_templates = torch.tensor((0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7, 8, 8, 8, 9), device=self._device)
        v46_index = ((ids // 3) + cycle_index) % v46_templates.shape[0]
        group = v46_templates[v46_index.long()]
      if traction_actor_v47_focus:
        # v54: release braking, no-pull stand and abrupt 3-D reversals.
        # v58: rebalance release/no-pull with hard turns and 3-D reversals.
        # Keep every behavior represented while preventing robust-cohort
        # forgetting of active traction and high-reversal stabilization.
        v47_templates = torch.tensor((0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9), device=self._device)
        v47_index = ((ids // 3) + cycle_index) % v47_templates.shape[0]
        group = v47_templates[v47_index.long()]
      if traction_actor_v63_focus:
        # v63 concentrates on the failures observed in v62 while retaining
        # no-pull/ramp/release anchors. Groups 3-9 cover 3-D jumps, +/-90,
        # 180, high reverse and low-force release transitions.
        v63_templates = torch.tensor((0, 1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9), device=self._device)
        v63_index = ((ids // 3) + cycle_index) % v63_templates.shape[0]
        group = v63_templates[v63_index.long()]
      if traction_actor_v64_focus:
        # v64 targets aligned-v63 failures: 3-D release, +90, high reverse,
        # sudden-pull release, while retaining cardinal and no-pull anchors.
        v64_templates = torch.tensor((0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 6, 7, 7, 7, 7, 8, 9, 9), device=self._device)
        v64_index = ((ids // 3) + cycle_index) % v64_templates.shape[0]
        group = v64_templates[v64_index.long()]
      if traction_actor_v65_focus:
        # v65 keeps the v63 anchor distribution but explicitly repeats the
        # failure families seen in v64: 3-D release, +90 turn, sudden pull,
        # high reverse/vertical jump, and combined XY/Z sign jump.
        v65_templates = torch.tensor(
          (0, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 7, 7, 7, 7, 9, 9, 8, 5, 6, 0, 1, 2),
          device=self._device,
        )
        v65_index = ((ids // 3) + cycle_index) % v65_templates.shape[0]
        group = v65_templates[v65_index.long()]
      if traction_actor_v67_focus:
        # v67 narrows the hard cohort after v66 over-specialized. Keep
        # anchors (stand, ramp, sudden pull, low-force) frequent while still
        # revisiting each failed transition family.
        v67_templates = torch.tensor(
          (0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2),
          device=self._device,
        )
        v67_index = ((ids // 3) + cycle_index) % v67_templates.shape[0]
        group = v67_templates[v67_index.long()]
      if traction_actor_v68_focus:
        # v68 is a short polish pass for the two recurrent teacher failures:
        # diagonal 3-D release (3) and high reverse/Z jump (7). Anchors stop
        # the narrow update from forgetting stand, ramp and low-force motion.
        v68_templates = torch.tensor(
          (0, 1, 2, 3, 3, 3, 7, 7, 7, 8, 4, 5, 6, 9),
          device=self._device,
        )
        v68_index = ((ids // 3) + cycle_index) % v68_templates.shape[0]
        group = v68_templates[v68_index.long()]
      if traction_actor_v69_focus:
        # v69 targets measured non-foot contacts (thigh/calf/head) while
        # retaining enough anchors for gait and low-force standing.
        v69_templates = torch.tensor(
          (0, 1, 2, 3, 3, 6, 7, 7, 7, 9, 0, 1, 4, 5, 8),
          device=self._device,
        )
        v69_index = ((ids // 3) + cycle_index) % v69_templates.shape[0]
        group = v69_templates[v69_index.long()]
      if traction_actor_v70_focus:
        # v70 concentrates on the two deterministic non-foot collision
        # families identified from v69 diagnostics while retaining anchors.
        v70_templates = torch.tensor(
          (0, 1, 2, 3, 3, 3, 3, 7, 7, 7, 7, 7, 9, 0, 1, 4, 5, 6, 8),
          device=self._device,
        )
        v70_index = ((ids // 3) + cycle_index) % v70_templates.shape[0]
        group = v70_templates[v70_index.long()]
      if traction_actor_v79_focus:
        # A single explicit curriculum replaces the misleading combination of
        # multiple focus flags (those flags are sequential overrides, not a
        # mixture). Preserve every anchor while moderately oversampling the
        # four remaining failure families: sudden pull (2), diagonal 3-D (3),
        # high reverse/Z jump (7), and low-force turn/release (8).
        v79_templates = torch.tensor(
          (0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8, 8, 9),
          device=self._device,
        )
        v79_index = ((ids // 3) + cycle_index) % v79_templates.shape[0]
        group = v79_templates[v79_index.long()]
      if traction_actor_v82_focus:
        # v82 targets the measured body-contact failures from v81. Keep every
        # anchor, but expose diagonal 3-D and high reverse/Z transitions more
        # often than the other families.
        v82_templates = torch.tensor(
          (0, 1, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 7, 8, 9),
          device=self._device,
        )
        v82_index = ((ids // 3) + cycle_index) % v82_templates.shape[0]
        group = v82_templates[v82_index.long()]
      if traction_actor_v93_focus:
        # v93 is a narrow continuation from v85: keep all anchor families,
        # but expose low-force 90-degree switching and reverse/Z transitions
        # at equal frequency.  This targets the only recurring v85 failures.
        v93_templates = torch.tensor(
          (0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9),
          device=self._device,
        )
        v93_index = ((ids // 3) + cycle_index) % v93_templates.shape[0]
        group = v93_templates[v93_index.long()]
      if traction_actor_v94_focus:
        # v94 keeps the v82 broad curriculum and only moderately oversamples
        # the two remaining failure families, minimizing catastrophic forgetting.
        v94_templates = torch.tensor(
          (0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 9),
          device=self._device,
        )
        v94_index = ((ids // 3) + cycle_index) % v94_templates.shape[0]
        group = v94_templates[v94_index.long()]
      if traction_actor_v99_focus:
        # v99 restores no-pull/release anchors while retaining hard switch
        # families, improving robustness without changing the command chain.
        v99_templates = torch.tensor((0, 0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 8, 8, 9), device=self._device)
        v99_index = ((ids // 3) + cycle_index) % v99_templates.shape[0]
        group = v99_templates[v99_index.long()]
      if traction_actor_v100_focus:
        # v100 concentrates exposure on recurring +/-90-degree and reverse+vertical failures while retaining no-pull anchors.
        v100_templates = torch.tensor((0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 9, 9), device=self._device)
        v100_index = ((ids // 3) + cycle_index) % v100_templates.shape[0]
        group = v100_templates[v100_index.long()]
      if traction_actor_v98_focus:
        # v98 targets the remaining actor-stability failures while retaining
        # broad anchors to avoid catastrophic forgetting.
        v98_templates = torch.tensor(
          (0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 7, 8, 8, 9),
          device=self._device,
        )
        v98_index = ((ids // 3) + cycle_index) % v98_templates.shape[0]
        group = v98_templates[v98_index.long()]
      if traction_actor_v95_focus:
        # v95 targets robust false-starts in no-pull and low-force turns.
        v95_templates = torch.tensor(
          (0, 0, 0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8, 9, 9),
          device=self._device,
        )
        v95_index = ((ids // 3) + cycle_index) % v95_templates.shape[0]
        group = v95_templates[v95_index.long()]
      t = (env.common_step_counter % cycle_steps) * self._step_dt
      force_requested = torch.zeros(self._num_envs, 3, device=self._device)

      def assign(mask, before, after=None, switch=4.0, release=1.0e9):
        if not mask.any(): return
        b = torch.tensor(before, device=self._device, dtype=force_requested.dtype)
        a = b if after is None else torch.tensor(after, device=self._device, dtype=force_requested.dtype)
        if traction_actor_v79_focus or traction_actor_v82_focus or traction_actor_v93_focus or traction_actor_v94_focus or traction_actor_v98_focus or traction_actor_v99_focus or traction_actor_v100_focus:
          # Stage-one curriculum: retain the target endpoints but reach them
          # over 0.4 s. Strict certification remains instantaneous, so later
          # stages can shorten this ramp without changing the final gate.
          # Anneal the transition duration over training cycles. Early
          # cycles provide a stable gradient; later cycles expose the actor
          # to nearly instantaneous changes matching strict certification.
          if traction_actor_v99_focus:
            cycle_ramp = torch.clamp(0.22 - 0.002 * cycle_index.to(dtype=force_requested.dtype), min=0.12, max=0.22)
          elif traction_actor_v98_focus:
            cycle_ramp = torch.clamp(
              0.24 - 0.003 * cycle_index.to(dtype=force_requested.dtype),
              min=0.12,
              max=0.24,
            )
          elif traction_actor_v94_focus:
            cycle_ramp = torch.clamp(
              0.32 - 0.004 * cycle_index.to(dtype=force_requested.dtype),
              min=0.16,
              max=0.32,
            )
          elif traction_actor_v93_focus:
            cycle_ramp = torch.clamp(
              0.24 - 0.006 * cycle_index.to(dtype=force_requested.dtype),
              min=0.10,
              max=0.24,
            )
          elif traction_actor_v82_focus:
            cycle_ramp = torch.clamp(
              0.32 - 0.008 * cycle_index.to(dtype=force_requested.dtype),
              min=0.12,
              max=0.32,
            )
          else:
            cycle_ramp = torch.clamp(
              0.4 - 0.015 * cycle_index.to(dtype=force_requested.dtype),
              min=0.08,
              max=0.4,
            )
          blend = torch.clamp(
            torch.tensor((t - switch), device=self._device) / cycle_ramp, 0.0, 1.0
          )
          value = b + blend * (a - b)
          if release < 1.0e8:
            release_blend = torch.clamp(
              torch.tensor((t - release), device=self._device) / cycle_ramp, 0.0, 1.0
            )
            value = value * (1.0 - release_blend)
        else:
          value = b if t < switch else a
          if t >= release: value = torch.zeros_like(value)
        force_requested[mask] = value

      # 0 remains zero. 1 is a true slow ramp and release.
      ramp = group == 1
      if ramp.any():
        alpha = torch.clamp(torch.tensor((t - 1.0) / 2.5, device=self._device), 0.0, 1.0)
        if t >= 6.0: alpha = torch.zeros_like(alpha)
        force_requested[ramp, 0] = 30.0 * alpha
      assign(group == 2, (0.0, 0.0, 0.0), (30.0, 0.0, 0.0), 2.0, 6.0)
      assign(group == 3, (0.0, 0.0, 0.0), (21.213, 21.213, 15.0), 2.0, 6.0)
      assign(group == 4, (25.0, 0.0, 0.0), (0.0, 25.0, 0.0), 4.0)
      assign(group == 5, (25.0, 0.0, 0.0), (0.0, -25.0, 0.0), 4.0)
      assign(group == 6, (25.0, 0.0, 0.0), (-25.0, 0.0, 0.0), 4.0)
      assign(group == 7, (15.0, 0.0, -8.0), (-40.0, 0.0, 20.0), 4.0)
      assign(group == 8, (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), 3.0, 6.0)
      assign(group == 9, (20.0, 10.0, 10.0), (-10.0, 30.0, -15.0), 3.0, 6.0)
      angle = self._traction_rotation
      c, s = torch.cos(angle), torch.sin(angle)
      x, y = force_requested[:, 0].clone(), force_requested[:, 1].clone()
      force_requested[:, 0] = c * x - s * y
      force_requested[:, 1] = s * x + c * y
      force_b, force_w_selected = self._resolve_force_frames(
        force_requested, self._asset.data.root_link_quat_w, force_frame
      )
      force_w = torch.zeros((self._num_envs, self._num_bodies, 3), device=self._device)
      force_w[:, 0] = force_w_selected
      self._asset.write_external_wrench_to_sim(
        force_w, torch.zeros_like(force_w), env_ids=ids, body_ids=self._body_ids
      )
      self.last_force_b[:] = force_b
      self.last_force_w[:] = force_w_selected
      return
    # Dedicated estimator data: stand with zero force, slowly ramp a real
    # MuJoCo world-force, hold it, then release back to zero while remaining
    # upright. This is deliberately before the legacy random-cycle branches.
    if static_ramp_release:
      ids = torch.arange(self._num_envs, device=self._device)
      cycle_steps = max(1, int(static_cycle_s / self._step_dt))
      cycle_index = env.common_step_counter // cycle_steps
      if cycle_index != self._static_cycle_index:
        self._static_cycle_index = cycle_index
        if static_fixed_force is not None:
          self._static_target_force[:] = torch.tensor(static_fixed_force, device=self._device)
        else:
          angle = torch.empty(self._num_envs, device=self._device).uniform_(-torch.pi, torch.pi)
          magnitude = torch.empty(self._num_envs, device=self._device).uniform_(*static_target_force_range)
          vertical = torch.empty(self._num_envs, device=self._device).uniform_(-5.0, 5.0)
          self._static_target_force[:, 0] = magnitude * torch.cos(angle)
          self._static_target_force[:, 1] = magnitude * torch.sin(angle)
          self._static_target_force[:, 2] = vertical
      phase = (env.common_step_counter % cycle_steps) / float(cycle_steps)
      # 0-.25 zero, .25-.50 ramp, .50-.75 hold, .75-1.0 release.
      alpha = torch.clamp((torch.tensor(phase, device=self._device) - 0.25) / 0.25, 0.0, 1.0)
      release = torch.clamp((1.0 - torch.tensor(phase, device=self._device)) / 0.25, 0.0, 1.0)
      scale = torch.minimum(alpha, release)
      force_requested = self._static_target_force * scale
      force_b, force_w_selected = self._resolve_force_frames(
        force_requested, self._asset.data.root_link_quat_w, force_frame
      )
      force_w = torch.zeros((self._num_envs, self._num_bodies, 3), device=self._device)
      force_w[:, 0] = force_w_selected
      zeros = torch.zeros_like(force_w)
      self._asset.write_external_wrench_to_sim(force_w, zeros, env_ids=ids, body_ids=self._body_ids)
      self.last_force_b[:] = force_b
      self.last_force_w[:] = force_w_selected
      if write_follow_command:
        self._write_follow_command(env, ids, force_b)
      return
    # Targeted abrupt-change data: a real force is held before the jump,
    # switched instantaneously to a new direction/magnitude (including a
    # vertical component), held again, then released.  This branch is used
    # only for estimator fine-tuning and does not alter the normal task.
    if jump_training:
      ids = torch.arange(self._num_envs, device=self._device)
      cycle_steps = max(1, int(jump_cycle_s / self._step_dt))
      cycle_index = env.common_step_counter // cycle_steps
      if cycle_index != self._jump_cycle_index:
        self._jump_cycle_index = cycle_index
        if jump_fixed_before is not None and jump_fixed_after is not None:
          self._jump_force_before[:] = torch.tensor(jump_fixed_before, device=self._device)
          self._jump_force_after[:] = torch.tensor(jump_fixed_after, device=self._device)
        else:
          a0 = torch.empty(self._num_envs, device=self._device).uniform_(-torch.pi, torch.pi)
          if jump_targeted_turns:
            # Explicitly train the hard intent changes instead of relying on
            # continuous random angles to occasionally approximate them.
            turn_choices = torch.tensor(
              (-torch.pi, -0.5 * torch.pi, 0.5 * torch.pi, torch.pi),
              device=self._device,
            )
            turn_index = torch.randint(
              len(turn_choices), (self._num_envs,), device=self._device
            )
            a1 = a0 + turn_choices[turn_index]
          else:
            a1 = torch.empty(self._num_envs, device=self._device).uniform_(
              -torch.pi, torch.pi
            )
          m0 = torch.empty(self._num_envs, device=self._device).uniform_(*jump_force_range)
          m1 = torch.empty(self._num_envs, device=self._device).uniform_(*jump_force_range)
          z0 = torch.empty(self._num_envs, device=self._device).uniform_(-15.0, 15.0)
          z1 = torch.empty(self._num_envs, device=self._device).uniform_(-20.0, 20.0)
          self._jump_force_before[:, 0] = m0 * torch.cos(a0)
          self._jump_force_before[:, 1] = m0 * torch.sin(a0)
          self._jump_force_before[:, 2] = z0
          self._jump_force_after[:, 0] = m1 * torch.cos(a1)
          self._jump_force_after[:, 1] = m1 * torch.sin(a1)
          self._jump_force_after[:, 2] = z1
      t = (env.common_step_counter % cycle_steps) * self._step_dt
      if t < jump_pre_s:
        force_requested = self._jump_force_before
      elif t < jump_pre_s + jump_post_s:
        force_requested = self._jump_force_after
      else:
        force_requested = torch.zeros_like(self._jump_force_after)
      force_b, force_w_selected = self._resolve_force_frames(
        force_requested, self._asset.data.root_link_quat_w, force_frame
      )
      force_w = torch.zeros((self._num_envs, self._num_bodies, 3), device=self._device)
      force_w[:, 0] = force_w_selected
      zeros = torch.zeros_like(force_w)
      self._asset.write_external_wrench_to_sim(force_w, zeros, env_ids=ids, body_ids=self._body_ids)
      self.last_force_b[:] = force_b
      self.last_force_w[:] = force_w_selected
      if write_follow_command:
        self._write_follow_command(env, ids, force_b)
      return
    if bilateral_fx_curriculum and fixed_cycle_s is not None:
      ids = torch.arange(self._num_envs, device=self._device)
      if env.common_step_counter < force_free_warmup_steps:
        zeros = torch.zeros(
          (self._num_envs, self._num_bodies, 3), device=self._device
        )
        self._asset.write_external_wrench_to_sim(
          zeros, zeros, env_ids=ids, body_ids=self._body_ids
        )
        self.last_force_b.zero_()
        self.last_force_w.zero_()
        self._bilateral_force_x.zero_()
        return
      cycle_steps = max(1, int(fixed_cycle_s / self._step_dt))
      cycle_index = env.common_step_counter // cycle_steps
      # One second of pull followed by two full seconds of zero-force stance.
      pulling = cycle_index % 3 == 0
      if pulling and cycle_index != self._bilateral_cycle_index:
        if bilateral_force_magnitude_range is not None:
          min_force, max_force = bilateral_force_magnitude_range
        else:
          # 24 control steps per learning iteration in the current PPO config.
          min_force = 0.0
          if env.common_step_counter < 72000:       # iterations 1500-3000
            max_force = 20.0
          elif env.common_step_counter < 120000:    # iterations 3000-5000
            max_force = 40.0
          elif env.common_step_counter < 168000:    # iterations 5000-7000
            max_force = 60.0
          else:                                     # iterations 7000-10000
            max_force = 80.0
        magnitude = torch.empty(self._num_envs, device=self._device).uniform_(
          min_force, max_force
        )
        sign = torch.where(
          torch.rand(self._num_envs, device=self._device) < 0.5,
          -torch.ones_like(magnitude),
          torch.ones_like(magnitude),
        )
        self._bilateral_force_x.copy_(sign * magnitude)
        self.retention_mask.copy_(
          torch.rand(self._num_envs, device=self._device) < retention_env_fraction
        )
        retain_magnitude = torch.empty(
          self._num_envs, device=self._device
        ).uniform_(*retention_command_range)
        retain_sign = torch.where(
          torch.rand(self._num_envs, device=self._device) < 0.5,
          -torch.ones_like(retain_magnitude),
          torch.ones_like(retain_magnitude),
        )
        self.retention_command.zero_()
        self.retention_command[self.retention_mask, 0] = (
          retain_sign * retain_magnitude
        )[self.retention_mask]
        self._bilateral_cycle_index = cycle_index
      force_x = self._bilateral_force_x if pulling else torch.zeros_like(self._bilateral_force_x)
      force_x = torch.where(self.retention_mask, torch.zeros_like(force_x), force_x)
      force_requested = torch.stack((force_x, torch.zeros_like(force_x), torch.zeros_like(force_x)), dim=-1)
      force_b, force_w_selected = self._resolve_force_frames(
        force_requested, self._asset.data.root_link_quat_w, force_frame
      )
      force_w = torch.zeros((self._num_envs, self._num_bodies, 3), device=self._device)
      force_w[:, 0] = force_w_selected
      zeros = torch.zeros_like(force_w)
      self._asset.write_external_wrench_to_sim(force_w, zeros, env_ids=ids, body_ids=self._body_ids)
      self.last_force_b[:] = force_b
      self.last_force_w[:] = force_w_selected
      if write_follow_command:
        self._write_follow_command(env, ids, force_b)
      return
    if fixed_force_b is not None and fixed_cycle_s is not None:
      ids = torch.arange(self._num_envs, device=self._device)
      cycle_steps = max(1, int(fixed_cycle_s / self._step_dt))
      pulling = (env.common_step_counter // cycle_steps) % 2 == 0
      force_requested = torch.tensor(fixed_force_b if pulling else (0.0, 0.0, 0.0), device=self._device)
      force_requested = force_requested.expand(self._num_envs, -1)
      force_b, force_w_selected = self._resolve_force_frames(
        force_requested, self._asset.data.root_link_quat_w, force_frame
      )
      force_w = torch.zeros((self._num_envs, self._num_bodies, 3), device=self._device)
      force_w[:, 0] = force_w_selected
      zeros = torch.zeros_like(force_w)
      self._asset.write_external_wrench_to_sim(force_w, zeros, env_ids=ids, body_ids=self._body_ids)
      self.last_force_b[:] = force_b
      self.last_force_w[:] = force_w_selected
      if write_follow_command:
        self._write_follow_command(env, ids, force_b)
      return

    self._remaining[self._active] -= self._step_dt
    expired = self._active & (self._remaining <= 0.0)
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      zeros = torch.zeros((len(expired_ids), self._num_bodies, 3), device=self._device)
      self._asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=expired_ids, body_ids=self._body_ids
      )
      self._active[expired_ids] = False
      self.last_force_b[expired_ids] = 0.0
      self.last_force_w[expired_ids] = 0.0
      self.retention_mask[expired_ids] = False
      self.retention_command[expired_ids] = 0.0
      if write_follow_command:
        self._write_follow_command(env, expired_ids, self.last_force_b[expired_ids])
      self._cooldown[expired_ids] = self._sample(cooldown_s, len(expired_ids))

    self._cooldown -= self._step_dt
    trigger = (~self._active) & (self._cooldown <= 0.0)
    if not trigger.any():
      return

    ids = trigger.nonzero(as_tuple=False).squeeze(-1)
    count = len(ids)
    active_horizontal_range = horizontal_force_range
    if horizontal_force_curriculum_steps:
      max_force = horizontal_force_range[1]
      for step_limit, stage_max in horizontal_force_curriculum_steps:
        if env.common_step_counter < step_limit:
          max_force = stage_max
          break
      active_horizontal_range = (horizontal_force_range[0], max_force)
    horizontal = self._sample(active_horizontal_range, count)
    pure_vertical = torch.zeros(count, dtype=torch.bool, device=self._device)
    sign = torch.where(
      torch.rand(count, device=self._device) < 0.5,
      -torch.ones_like(horizontal),
      torch.ones_like(horizontal),
    )
    if force_direction_mode in ("certification_matrix", "certification_matrix_hard40"):
      if (
        force_direction_mode == "certification_matrix_hard40"
        and not self._certification_hard40_initialized
      ):
        self._certification_sample_index.copy_(
          torch.arange(self._num_envs, device=self._device)
          % self._certification_hard40_matrix.shape[0]
        )
        self._certification_hard40_initialized = True
      matrix_index = self._certification_sample_index[ids]
      active_matrix = (
        self._certification_hard40_matrix
        if force_direction_mode == "certification_matrix_hard40"
        else self._certification_force_matrix
      )
      matrix_index = matrix_index % active_matrix.shape[0]
      force_requested = active_matrix[matrix_index]
      self._certification_sample_index[ids] = (
        matrix_index + 1
      ) % active_matrix.shape[0]
      force_x, force_y, force_z = force_requested.unbind(dim=-1)
    elif force_direction_mode == "fy_bilateral":
      force_x = torch.zeros_like(horizontal)
      force_y = sign * horizontal
      if fx_retention_fraction > 0.0:
        use_fx = torch.rand(count, device=self._device) < fx_retention_fraction
        force_x = torch.where(use_fx, sign * horizontal, force_x)
        force_y = torch.where(use_fx, torch.zeros_like(force_y), force_y)
    elif force_direction_mode == "fx_bilateral":
      force_x = sign * horizontal
      force_y = torch.zeros_like(horizontal)
    elif force_direction_mode == "random_horizontal":
      angle = torch.rand(count, device=self._device) * (2.0 * torch.pi)
      force_x = horizontal * torch.cos(angle)
      force_y = horizontal * torch.sin(angle)
    elif force_direction_mode == "continuous_world_jump":
      # Most samples use a truly continuous azimuth.  A retained axial cohort
      # prevents catastrophic forgetting of the certified cardinal cases, and
      # a pure-vertical cohort preserves the ±Z behavior.
      angle = torch.rand(count, device=self._device) * (2.0 * torch.pi)
      bucket = torch.randint(10, (count,), device=self._device)
      angle = torch.where(bucket == 0, torch.zeros_like(angle), angle)
      angle = torch.where(bucket == 1, torch.full_like(angle, torch.pi), angle)
      angle = torch.where(bucket == 2, torch.full_like(angle, 0.5 * torch.pi), angle)
      angle = torch.where(bucket == 3, torch.full_like(angle, -0.5 * torch.pi), angle)
      force_x = horizontal * torch.cos(angle)
      force_y = horizontal * torch.sin(angle)
      pure_vertical = bucket == 4
      force_x = torch.where(pure_vertical, torch.zeros_like(force_x), force_x)
      force_y = torch.where(pure_vertical, torch.zeros_like(force_y), force_y)
    elif force_direction_mode == "planar_balanced":
      # Keep exact axial pulls well represented while still covering every
      # diagonal direction. This avoids forgetting the harder lateral gait.
      mode = torch.randint(3, (count,), device=self._device)
      angle = torch.rand(count, device=self._device) * (2.0 * torch.pi)
      force_x = torch.where(
        mode == 0,
        sign * horizontal,
        torch.where(mode == 1, torch.zeros_like(horizontal), horizontal * torch.cos(angle)),
      )
      force_y = torch.where(
        mode == 1,
        sign * horizontal,
        torch.where(mode == 0, torch.zeros_like(horizontal), horizontal * torch.sin(angle)),
      )
    else:
      raise ValueError(f"Unsupported force_direction_mode: {force_direction_mode}")
    if force_direction_mode not in ("certification_matrix", "certification_matrix_hard40"):
      force_z = self._sample(vertical_force_range, count)
      if force_direction_mode == "continuous_world_jump":
        # One quarter of the non-vertical cohort remains exactly horizontal;
        # the rest spans continuous signed vertical components.
        pure_horizontal = (~pure_vertical) & (torch.rand(count, device=self._device) < 0.25)
        force_z = torch.where(pure_horizontal, torch.zeros_like(force_z), force_z)
        # Pure-vertical samples must carry a meaningful nonzero signed load.
        vertical_mag = torch.empty(count, device=self._device).uniform_(5.0, 20.0)
        vertical_sign = torch.where(
          torch.rand(count, device=self._device) < 0.5,
          -torch.ones_like(vertical_mag), torch.ones_like(vertical_mag),
        )
        force_z = torch.where(pure_vertical, vertical_sign * vertical_mag, force_z)
      if vertical_endpoint_fraction > 0.0:
        endpoint_mask = (
          torch.rand(count, device=self._device) < vertical_endpoint_fraction
        )
        endpoint = torch.where(
          torch.rand(count, device=self._device) < 0.5,
          torch.full_like(force_z, vertical_force_range[0]),
          torch.full_like(force_z, vertical_force_range[1]),
        )
        force_z = torch.where(endpoint_mask, endpoint, force_z)
      force_requested = torch.stack((force_x, force_y, force_z), dim=-1)
    self.retention_mask[ids] = False
    self.retention_command[ids] = 0.0
    if direct_lateral_retention_fraction > 0.0:
      retain = torch.rand(count, device=self._device) < direct_lateral_retention_fraction
      retained_ids = ids[retain]
      if retained_ids.numel() > 0:
        retain_magnitude = self._sample(
          direct_lateral_command_range, retained_ids.numel()
        )
        retain_sign = torch.where(
          torch.rand(retained_ids.numel(), device=self._device) < 0.5,
          -torch.ones_like(retain_magnitude),
          torch.ones_like(retain_magnitude),
        )
        force_requested[retain] = 0.0
        self.retention_mask[retained_ids] = True
        self.retention_command[retained_ids, 1] = retain_sign * retain_magnitude
    # Pull direction is expressed in the base frame, while the selected link
    # and local application point are randomized independently.
    root_quat_w = self._asset.data.root_link_quat_w[ids]
    force_b, force_w_selected = self._resolve_force_frames(
      force_requested, root_quat_w, force_frame
    )
    selected_body = torch.randint(self._num_bodies, (count,), device=self._device)
    selected_quat_w = self._asset.data.body_com_quat_w[ids][:, self._body_ids][
      torch.arange(count, device=self._device), selected_body
    ]
    point_b = torch.stack(
      [self._sample(axis_range, count) for axis_range in body_point_offset_ranges],
      dim=-1,
    )
    offset_w_selected = quat_apply(selected_quat_w, point_b)
    torque_w_selected = torch.cross(offset_w_selected, force_w_selected, dim=-1)

    force_w = torch.zeros((count, self._num_bodies, 3), device=self._device)
    torque_w = torch.zeros_like(force_w)
    batch = torch.arange(count, device=self._device)
    force_w[batch, selected_body] = force_w_selected
    torque_w[batch, selected_body] = torque_w_selected
    self._asset.write_external_wrench_to_sim(
      force_w, torque_w, env_ids=ids, body_ids=self._body_ids
    )
    self.last_force_b[ids] = force_b
    self.last_force_w[ids] = force_w_selected
    if write_follow_command:
      self._write_follow_command(env, ids, force_b)
    self._remaining[ids] = self._sample(duration_s, count)
    self._active[ids] = True
    self._cooldown[ids] = 0.0

  def reset(self, env_ids=None) -> None:
    ids = torch.arange(self._num_envs, device=self._device) if env_ids is None else env_ids
    zeros = torch.zeros((len(ids), self._num_bodies, 3), device=self._device)
    self._asset.write_external_wrench_to_sim(zeros, zeros, env_ids=ids, body_ids=self._body_ids)
    self._remaining[ids] = 0.0
    self._cooldown[ids] = 0.0
    self._active[ids] = False
    self.last_force_b[ids] = 0.0
    self.last_force_w[ids] = 0.0
    self.retention_mask[ids] = False
    self.retention_command[ids] = 0.0

  def _curriculum_ranges(self, env) -> tuple[tuple[float, float], tuple[float, float]]:
    step = env.common_step_counter
    if step < 24000:
      return (0.0, 20.0), (0.0, 0.0)
    if step < 72000:
      return (0.0, 40.0), (-10.0, 10.0)
    if step < 144000:
      return (0.0, 60.0), (-20.0, 20.0)
    return (0.0, 80.0), (-35.0, 35.0)

  def _write_follow_command(self, env, env_ids: torch.Tensor, force_b: torch.Tensor) -> None:
    cmd = env.command_manager.get_term("twist").vel_command_b
    dead = 8.0
    target = torch.sign(force_b[:, :2]) * torch.clamp((torch.abs(force_b[:, :2]) - dead) * 0.02, min=0.0)
    target[:, 0] = torch.clamp(target[:, 0], -0.6, 0.8)
    target[:, 1] = torch.clamp(target[:, 1], -0.4, 0.4)
    vertical_scale = 1.0 - 0.6 * torch.clamp((torch.abs(force_b[:, 2]) - 10.0) / 25.0, 0.0, 1.0)
    target *= vertical_scale.unsqueeze(-1)
    cmd[env_ids, :2] = 0.85 * cmd[env_ids, :2] + 0.15 * target
    cmd[env_ids, 2] = 0.0

  def set_manual_force(self, force_b: torch.Tensor, enabled: bool) -> None:
    force = force_b.to(self._device)
    if force.ndim == 1:
      force = force.expand(self._num_envs, -1)
    if force.shape != (self._num_envs, 3):
      raise ValueError(
        f"manual force must have shape (3,) or ({self._num_envs}, 3), got {tuple(force.shape)}"
      )
    self.manual_force_b.copy_(force)
    self.manual_enabled = enabled

  @staticmethod
  def _resolve_force_frames(
    force_requested: torch.Tensor,
    root_quat_w: torch.Tensor,
    force_frame: str,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (F_b, F_w) without changing the requested physical semantics."""
    # Compute the adapter in float64 even though the simulator state is float32.
    # This keeps the 80 N-scale absolute round-trip error below 1e-5.
    requested64 = force_requested.to(dtype=torch.float64)
    quat64 = root_quat_w.to(dtype=torch.float64)
    quat64 = quat64 / torch.linalg.norm(quat64, dim=-1, keepdim=True).clamp_min(1.0e-12)
    if force_frame == "legacy_body":
      force_b64 = requested64
      force_w64 = quat_apply(quat64, force_b64)
    elif force_frame == "world_fixed":
      force_w64 = requested64
      force_b64 = quat_apply_inverse(quat64, force_w64)
    else:
      raise ValueError(f"Unsupported force_frame: {force_frame}")
    force_b = force_b64.to(dtype=force_requested.dtype)
    force_w = force_w64.to(dtype=force_requested.dtype)
    return force_b, force_w

  def _sample(self, value_range: tuple[float, float], count: int) -> torch.Tensor:
    low, high = value_range
    return low + (high - low) * torch.rand(count, device=self._device)
