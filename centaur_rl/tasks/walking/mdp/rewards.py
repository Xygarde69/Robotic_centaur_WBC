from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_orientation_roll_pitch(
  env: ManagerBasedRlEnv,
  roll_weight: float = 1.0,
  pitch_weight: float = 1.0,
  command_name: str | None = None,
  gravity: float = 9.81,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """body_orientation_l2 with separate roll and pitch weights, roll measured from the
  equilibrium lean of the commanded turn.

  Projected gravity in the body frame: its y component is sin(roll-ish lean), its x
  component grows with pitch. Walking a circle of radius r = 1 / curvature at speed v
  needs a centripetal acceleration v^2 / r, which is balanced by leaning into the turn by
  theta = atan(v^2 / (r g)). At that lean gravity has body-frame y component sin(theta)
  (positive = leaning left, for a left turn, curvature > 0), so the roll penalty is
  (g_y - sin(theta))^2. v is the measured forward speed at this instant, curvature is
  the walk command's (see WalkVelocityCommand). Without command_name the target is upright.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    g = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
  else:
    g = asset.data.projected_gravity_b
  roll_target = torch.zeros_like(g[:, 1])
  if command_name is not None:
    curvature = env.command_manager.get_term(command_name).curvature
    v = asset.data.root_link_lin_vel_b[:, 0]
    roll_target = torch.sin(torch.atan(v.square() * curvature / gravity))
  return pitch_weight * torch.square(g[:, 0]) + roll_weight * torch.square(
    g[:, 1] - roll_target
  )


def body_angular_velocity_roll_pitch(
  env: ManagerBasedRlEnv,
  roll_weight: float = 1.0,
  pitch_weight: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """body_angular_velocity_penalty with separate roll-rate and pitch-rate weights.

  Rates are taken in the body frame (the original uses world x/y, which only line up with
  roll/pitch when the robot faces world +x).
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  ang_vel = quat_apply_inverse(quat_w, ang_vel_w)
  return roll_weight * torch.square(ang_vel[:, 0]) + pitch_weight * torch.square(ang_vel[:, 1])


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def load_gait_schedule(
        path: str, foot_order: tuple[str, ...]
) -> tuple[list[float], list[list[int]]]:
    """Read a gait JSON into (phase_durations, stance_table) in foot_order."""
    import json

    with open(path) as f:
        gait = json.load(f)
    durations = [float(p["duration"]) for p in gait["phases"]]
    table = [[int(p["feet"][leg] == "ground") for leg in foot_order] for p in gait["phases"]]
    return durations, table


def gait_timing(
        phase_durations: list[float], stance_table: list[list[int]]
) -> tuple[float, float, float]:
    """Return (period, longest continuous stance, longest continuous swing) in seconds."""
    period = sum(phase_durations)
    longest = {0: 0.0, 1: 0.0}
    for foot in range(len(stance_table[0])):
        states = [row[foot] for row in stance_table]
        if len(set(states)) == 1:
            longest[states[0]] = period
            continue
        # Start right after a state change so cyclic runs are not split.
        start = next(i for i in range(len(states)) if states[i] != states[i - 1])
        run = 0.0
        for k in range(len(states)):
            i = (start + k) % len(states)
            run = run + phase_durations[i] if k and states[i] == states[i - 1] else phase_durations[i]
            longest[states[i]] = max(longest[states[i]], run)
    return period, longest[1], longest[0]


def gait_leg_clocks(
        phase_durations: list[float], stance_table: list[list[int]]
) -> tuple[float, float, list[float], list[float], list[float]]:
    """Turn a gait JSON timeline into trot-style per-leg clocks.

    The JSON is read as a linear timeline starting at t=0 from standing: a leg starts
    its cycle at its first lift-off (a foot already in the air at t=0 lifts off at 0),
    swings for its first air run, then stands until its next lift-off. Every leg runs
    the same cycle (one period), each started at a different time.

    Returns (period, stance_fraction, start_times, air_times, offsets), per-leg lists in
    stance_table column order. offsets are for feet_gait, where a leg is in stance while
    (t / period + offset) % 1 < stance_fraction, i.e. its clock starts at touchdown.
    """
    starts = [0.0]
    for d in phase_durations:
        starts.append(starts[-1] + d)
    total = starts[-1]
    n_legs = len(stance_table[0])
    columns = [[row[leg] for row in stance_table] for leg in range(n_legs)]
    lift_offs: list[list[float]] = []
    for leg, states in enumerate(columns):
        lifts = [starts[i] for i, s in enumerate(states) if s == 0 and (i == 0 or states[i - 1] == 1)]
        if not lifts:
            raise ValueError(f"Gait leg {leg} never leaves the ground.")
        lift_offs.append(lifts)
    # The period is the time between two lift-offs of the same leg.
    periods = [l[1] - l[0] for l in lift_offs if len(l) > 1]
    if not periods:
        raise ValueError("Gait JSON must contain a second lift-off of some leg to define the period.")
    period = periods[0]
    if any(abs(p - period) > 1e-6 for p in periods):
        raise ValueError(f"Inconsistent per-leg cycle lengths in gait JSON: {periods}")

    def state_at(states: list[int], t: float) -> int:
        while t >= total - 1e-9:  # Past the end of the JSON: the gait repeats every period.
            t -= period
        return states[max(i for i in range(len(states)) if starts[i] <= t + 1e-9)]

    # A swing lasts until the leg's next ground phase; ends are phase starts, possibly
    # one period on (a swing can run off the end of the JSON, e.g. RR in P6-P8).
    boundaries = sorted({s + k * period for s in starts[:-1] for k in (0, 1, 2)})
    air_times = []
    for states, lifts in zip(columns, lift_offs, strict=True):
        end = next(b for b in boundaries if b > lifts[0] + 1e-9 and state_at(states, b) == 1)
        air_times.append(end - lifts[0])
    if any(abs(a - air_times[0]) > 1e-6 for a in air_times):
        raise ValueError(f"feet_gait needs one air time for all legs, got {air_times}")
    stance_fraction = 1.0 - air_times[0] / period
    start_times = [l[0] for l in lift_offs]
    offsets = [(-(t + air_times[0]) / period) % 1.0 for t in start_times]
    return period, stance_fraction, start_times, air_times, offsets


def walk_gait_cycles(env: ManagerBasedRlEnv, command_name: str, lookahead: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(gait cycles since walking started, walking mask) from the walk command's clock.

    With lookahead, the clock is advanced one control step: rewards are computed before
    the clock ticks, so this matches the time the trot's feet_gait uses (one step past
    the observation the action was chosen from).
    """
    term = env.command_manager.get_term(command_name)
    walking = term.is_walking()
    cycles = term.gait_cycles
    if lookahead:
        cycles = cycles + torch.where(walking, env.step_dt / term.cycle_time, 0.0)
    return cycles, walking


class _WalkClockTerm:
    """Shared setup for terms on the walk clock: offsets and lift phases in sensor order.

    offset and lift_phase are given per foot_geoms entry and permuted here to the
    contact sensor's foot order (model order).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        foot_geoms = list(cfg.params["foot_geoms"])
        asset: Entity = env.scene[cfg.params.get("asset_name", "robot")]
        _, sensor_order = asset.find_geoms(foot_geoms)
        assert sorted(sensor_order) == sorted(foot_geoms), (sensor_order, foot_geoms)
        perm = [foot_geoms.index(name) for name in sensor_order]
        self.offset = torch.tensor(cfg.params["offset"], device=env.device)[perm].view(1, -1)
        self.lift_phase = torch.tensor(cfg.params["lift_phase"], device=env.device)[perm].view(1, -1)

    def _targets(
            self, env: ManagerBasedRlEnv, threshold: float, command_name: str, sensor_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(scheduled stance [B, N], actual contact [B, N], walking [B])."""
        sensor: ContactSensor = env.scene[sensor_name]
        is_contact = sensor.data.current_contact_time > 0
        cycles, walking = walk_gait_cycles(env, command_name, lookahead=True)
        cycles = cycles.unsqueeze(1)
        is_stance = (((cycles + self.offset) % 1.0) < threshold) | (cycles < self.lift_phase)
        return is_stance, is_contact, walking


class feet_gait_clock(_WalkClockTerm):
    """The trot's feet_gait, clocked by the walk command and started from rest.

    Every leg runs the same cycle (cycle time from the command): stance while
    (cycles + offset) % 1 < threshold. Starting from standing, a leg stays on the ground
    until its first lift-off (cycles < lift_phase), so the walk begins with FR's long
    first stride and the other legs join in turn, as in gait.json's P0. Zero while
    standing (as feet_gait).

    reward = (1 - exact_weight) * fraction of feet matching + exact_weight * [all match]

    exact_weight = 0 is the trot's feet_gait. The per-foot fraction alone cannot tell a
    walk from a trot: a best-aligned trot matches 75% of the walk's foot states, but all
    four feet only 45% of the time (a walk: 100%), so the all-match part is what pays
    for walking instead of trotting.
    """

    def __call__(
            self,
            env: ManagerBasedRlEnv,
            offset: list[float],
            lift_phase: list[float],
            threshold: float,
            command_name: str,
            sensor_name: str,
            foot_geoms: tuple[str, ...],
            exact_weight: float = 0.0,
    ) -> torch.Tensor:
        del offset, lift_phase, foot_geoms  # Permuted to sensor order in __init__.
        is_stance, is_contact, walking = self._targets(env, threshold, command_name, sensor_name)
        match = is_stance == is_contact
        exact = match.all(dim=1).float()
        if walking.any():
            env.extras["log"]["Metrics/gait_foot_match"] = match[walking].float().mean()
            env.extras["log"]["Metrics/gait_exact_match"] = exact[walking].mean()
        reward = (1.0 - exact_weight) * match.float().mean(dim=1) + exact_weight * exact
        return reward * walking.float()


class feet_contact_in_swing(_WalkClockTerm):
    """Fraction of feet on the ground while their clock says swing (use a negative weight).

    Same clock, offsets and startup hold as feet_gait_clock. Zero while standing.
    """

    def __call__(
            self,
            env: ManagerBasedRlEnv,
            offset: list[float],
            lift_phase: list[float],
            threshold: float,
            command_name: str,
            sensor_name: str,
            foot_geoms: tuple[str, ...],
    ) -> torch.Tensor:
        del offset, lift_phase, foot_geoms  # Permuted to sensor order in __init__.
        is_stance, is_contact, walking = self._targets(env, threshold, command_name, sensor_name)
        cost = (is_contact & ~is_stance).float().mean(dim=1)
        return cost * walking.float()


def gait_step_table(
        phase_durations: list[float], stance_table: list[list[int]], step_dt: float
) -> list[list[int]]:
    """Expand the phase schedule to one stance row per control step of the cycle.

    Integer step indexing keeps phase boundaries exact (no float modulo drift), so
    every phase duration must be a whole number of control steps.
    """
    rows: list[list[int]] = []
    for duration, stance in zip(phase_durations, stance_table, strict=True):
        steps = round(duration / step_dt)
        if steps < 1 or abs(steps * step_dt - duration) > 1e-6:
            raise ValueError(
                f"Gait phase duration {duration} s is not a multiple of the control step {step_dt} s."
            )
        rows.extend([list(stance)] * steps)
    return rows


def resolve_gait_step_table(params: dict, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-step stance table [steps_per_cycle, N] with columns in contact-sensor order.

    stance_table columns follow params["foot_geoms"]. The contact sensor orders its
    feet by model order, not by the order they were listed in, so the columns are
    permuted here with the same lookup the sensor uses.
    """
    rows = gait_step_table(params["phase_durations"], params["stance_table"], env.step_dt)
    table = torch.as_tensor(rows, device=env.device, dtype=torch.bool)
    foot_geoms = list(params["foot_geoms"])
    asset: Entity = env.scene[params.get("asset_name", "robot")]
    _, sensor_order = asset.find_geoms(foot_geoms)
    assert sorted(sensor_order) == sorted(foot_geoms), (sensor_order, foot_geoms)
    return table[:, [foot_geoms.index(name) for name in sensor_order]]


def _command_is_standing(
        env: ManagerBasedRlEnv, command_name: str, command_threshold: float
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    return torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) <= command_threshold


def gait_stance_target(
        env: ManagerBasedRlEnv,
        step_table: torch.Tensor,
        command_name: str,
        command_threshold: float,
        lookahead_steps: int = 0,
) -> torch.Tensor:
    """Scheduled stance (True = on the ground) per foot, [B, N].

    step_table is [steps_per_cycle, N]. Below the command threshold every foot is
    scheduled on the ground.
    """
    idx = (env.episode_length_buf + lookahead_steps) % step_table.shape[0]
    is_stance = step_table[idx]
    standing = _command_is_standing(env, command_name, command_threshold)
    return is_stance | standing.unsqueeze(1)


class feet_gait_schedule:
    """Reward foot contacts matching a piecewise phase schedule (mdp/gait.json).

    phase_durations[i] is the length (s) of phase i; stance_table[i][j] is 1 if
    the foot with collision geom foot_geoms[j] should be on the ground in phase i,
    0 if in the air.
    Below command_threshold the target is all feet on the ground.

    reward = (1 - exact_weight) * fraction of feet matching + exact_weight * [all match]

    The per-foot fraction gives a dense signal for each leg; the all-match part is
    only paid while the whole contact pattern is exactly the scheduled one, so
    keeping every foot planted (which already matches about half the schedule) is
    worth much less than executing the gait.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.step_table = resolve_gait_step_table(cfg.params, env)

    def __call__(
            self,
            env: ManagerBasedRlEnv,
            phase_durations: list[float],
            stance_table: list[list[int]],
            foot_geoms: tuple[str, ...],
            command_threshold: float,
            command_name: str,
            sensor_name: str,
            exact_weight: float = 0.5,
    ) -> torch.Tensor:
        del phase_durations, stance_table, foot_geoms  # Baked into step_table.
        sensor: ContactSensor = env.scene[sensor_name]
        is_contact = sensor.data.current_contact_time > 0
        is_stance = gait_stance_target(env, self.step_table, command_name, command_threshold)
        match = is_stance == is_contact
        exact = match.all(dim=1).float()
        env.extras["log"]["Metrics/gait_foot_match"] = match.float().mean()
        env.extras["log"]["Metrics/gait_exact_match"] = exact.mean()
        return (1.0 - exact_weight) * match.float().mean(dim=1) + exact_weight * exact


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward

