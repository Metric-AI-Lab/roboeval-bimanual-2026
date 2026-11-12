"""Observation-only LiftPot controller for the challenge EE-delta interface.

Input: raw standard observation and one reset-frame pot (x, y, yaw) estimate.
Output: 14-D action. This module never reads an environment, physics object,
contact state, object pose after reset, or privileged simulation data.

The static constants are calibrated from the released bimanual Panda and pot
model: pot handle centres in its body frame, reset commanded EE pose, pelvis
translation, and fingertip-pad midpoint in the EE frame. Robot FK below uses
the published Panda link transforms and joint positions in proprioception.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

DT = 1 / 20
V_AVG = .22
W_AVG = 1.6

# From env.reset and the static robot/prop models; neither depends on episode seed.
RESET_EE = np.array([
    .5544995, .3, 1.2245024, -2.2213323, 2.2215505, 0.,
    .5544995, -.3, 1.2245024, -2.2213323, 2.2215505, 0.,
], dtype=np.float64)
PELVIS_POS = np.array([.1, 0., 0.])
PAD_IN_EE = np.array([0., .00175, .1034])
HANDLE_CENTRES_IN_POT = np.array([[0., .2225, .035], [0., -.235, .035]])
POT_RESET_Z = .996
GRIP_THRESHOLD = .05

# Panda nohand_modified.xml, in arm-base coordinates; the arm bases are
# [0, +/-0.3, 0.6] relative to the pelvis. Each body transform precedes its
# hinge rotation about its local z axis.
LINK_POS = np.array([
    [0., 0., .333], [0., 0., 0.], [0., -.316, 0.], [.0825, 0., 0.],
    [-.0825, .384, 0.], [0., 0., 0.], [.088, 0., 0.],
])
LINK_QUAT = np.array([
    [1., 0., 0., 0.], [1., -1., 0., 0.], [1., 1., 0., 0.],
    [1., 1., 0., 0.], [1., -1., 0., 0.], [1., 1., 0., 0.],
    [1., 1., 0., 0.],
])
TIP_POS = np.array([0., 0., .107])
TIP_ROT = R.from_quat([.3826834, 0., 0., .9238795], scalar_first=True)
RESET_QCTRL = np.array([0., 0., 0., -1.5707900524, 0., 1.5707900524, -.7853000164] * 2)
MAX_JOINT_DELTA = 2.175 * .05  # 20 Hz control interval


def wrap(a: float, period: float) -> float:
    return -((-a + period / 2) % period - period / 2)


def min_jerk(s: float) -> float:
    s = min(max(s, 0.), 1.)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def measured_ee(obs: dict) -> np.ndarray:
    """Optional Panda FK from public arm qpos, useful for diagnostics."""
    qpos = np.asarray(obs["proprioception"], dtype=float)
    if qpos.shape != (36,):
        raise ValueError(f"expected 36 proprioception values, got {qpos.shape}")
    out = np.zeros(12, dtype=float)
    for i, start in enumerate((0, 9)):
        p = np.array([0., .3 if i == 0 else -.3, .6])
        rot = R.identity()
        for j in range(7):
            p += rot.apply(LINK_POS[j])
            rot = rot * R.from_quat(LINK_QUAT[j], scalar_first=True)
            rot = rot * R.from_euler("z", float(qpos[start + j]))
        p += rot.apply(TIP_POS)
        rot = rot * TIP_ROT
        out[6 * i:6 * i + 3] = p
        out[6 * i + 3:6 * i + 6] = rot.as_rotvec()
    return out


def arm_fk_jac(q: np.ndarray, side: int):
    """Static Panda site FK and geometric Jacobian, in pelvis coordinates."""
    p = np.array([0., .3 if side == 0 else -.3, .6])
    rot = R.identity()
    origins, axes = [], []
    for j in range(7):
        p += rot.apply(LINK_POS[j])
        rot = rot * R.from_quat(LINK_QUAT[j], scalar_first=True)
        origins.append(p.copy())
        axes.append(rot.apply([0., 0., 1.]))
        rot = rot * R.from_euler("z", float(q[j]))
    p += rot.apply(TIP_POS)
    rot = rot * TIP_ROT
    jac = np.empty((6, 7), float)
    for j in range(7):
        jac[:3, j] = np.cross(axes[j], p - origins[j])
        jac[3:, j] = axes[j]
    return p, rot, jac


def solve_arm_ik(q_start: np.ndarray, side: int, target_p: np.ndarray, target_r: R):
    """Match the challenge's 40-step site-pose solver using the static Jacobian.

    Its target is the commanded EE pose, not the measured arm pose. Failed IK
    keeps the previous joint target, as the challenge action mode does.
    """
    q = q_start.copy()
    for _ in range(40):
        p, rot, jac = arm_fk_jac(q, side)
        err_pos = target_p - p
        err_rot = (target_r * rot.inv()).as_rotvec()
        err_norm = np.linalg.norm(err_pos) + np.linalg.norm(err_rot)
        if err_norm < 1e-5:
            return q
        err = np.concatenate((err_pos, err_rot))
        hess = jac.T @ jac
        rhs = jac.T @ err
        if err_norm > .1:
            update = np.linalg.solve(hess + .03 * np.eye(7), rhs)
        else:
            update = np.linalg.lstsq(hess, rhs, rcond=-1)[0]
        norm = np.linalg.norm(update)
        if norm == 0 or err_norm / norm > 20.:
            break
        if norm > 2.:
            update *= 2. / norm
        q += update
    return q_start.copy()


class Seg:
    def __init__(self, p0, p1, y0, y1, grip, min_steps=6, speed=1., yaw_fraction=1.):
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.y0, self.y1, self.grip = float(y0), float(y1), float(grip)
        dur = max(np.linalg.norm(self.p1 - self.p0) / (V_AVG * speed),
                  abs(self.y1 - self.y0) / W_AVG)
        self.move = self.n = max(min_steps, int(np.ceil(dur / DT)))
        self.yaw_fraction = float(yaw_fraction)
        self.k = 0

    def next(self):
        self.k += 1
        s = min_jerk(self.k / self.move)
        syaw = min_jerk(self.k / (self.move * self.yaw_fraction))
        return self.p0 + (self.p1 - self.p0) * s, self.y0 + (self.y1 - self.y0) * syaw

    @property
    def done(self):
        return self.k >= self.n


class PotController:
    """No-environment policy component; call `act(obs)` once per 20 Hz step.

    The first action is the demonstration convention's grippers-open no-op.
    `pot_pose` is only the reset estimate (x, y, yaw, in metres/radians).
    """

    GRASP_DZ = -.012
    GRASP_DX = 0.
    LIFT_DX = -.08
    YAW_CAP = 1.40
    LIFT_SPEED = .70
    STAGES = ("over", "descend", "settle", "close", "verify", "micro", "lift", "verifylift", "lower", "done")

    def __init__(self, pot_pose, *, right_offset=(0., 0., 0.), right_yaw_offset=0.,
                 descent_speed=1., settled_frames=1, micro_lift=0., exact_kinematics=False,
                 yaw_cap=None, preclose_wait=0, retry_wait=0, retry_schedule=(),
                 descent_replans=0, replan_tol=.01, settle_error_m=0., settle_max_steps=8,
                 yaw_mode="signed", retry_yaw_alternate=False, hold_grasp_on_retry=False,
                 grasp_offset_local_x=0., far_arm_dx=0., far_arm_yaw_threshold=.12,
                 adaptive_yaw_ik_threshold=0., adaptive_negative_yaw_edge_x=0.,
                 over_height=.095, over_speed=1., high_over_negative_y=False,
                 adaptive_handle_x=False, adaptive_handle_x_strict=False,
                 over_yaw_fraction=1., over_yaw_gate=False):
        if yaw_cap is not None:
            self.YAW_CAP = float(yaw_cap)
        self.preclose_wait = int(preclose_wait)
        self.retry_wait = int(retry_wait)
        self.retry_schedule = tuple(tuple(map(float, row)) for row in retry_schedule)
        self.descent_replans = int(descent_replans)
        self.replan_tol = float(replan_tol)
        self.settle_error_m = float(settle_error_m)
        self.settle_max_steps = int(settle_max_steps)
        if yaw_mode not in ("signed", "cross_if_clipped", "nearest"):
            raise ValueError(f"unknown pot yaw mode {yaw_mode}")
        self.yaw_mode = yaw_mode
        self.retry_yaw_alternate = bool(retry_yaw_alternate)
        self.hold_grasp_on_retry = bool(hold_grasp_on_retry)
        self.grasp_offset_local_x = float(grasp_offset_local_x)
        self.far_arm_dx = float(far_arm_dx)
        self.far_arm_yaw_threshold = float(far_arm_yaw_threshold)
        self.adaptive_yaw_ik_threshold = float(adaptive_yaw_ik_threshold)
        self.adaptive_negative_yaw_edge_x = float(adaptive_negative_yaw_edge_x)
        self.over_height = float(over_height)
        self.over_speed = float(over_speed)
        if not np.isfinite(over_yaw_fraction) or not 0.1 <= over_yaw_fraction <= 1.:
            raise ValueError("over_yaw_fraction must be in [0.1, 1]")
        self.over_yaw_fraction = float(over_yaw_fraction)
        self._descent_replans = 0
        self.right_offset = np.asarray(right_offset, dtype=float)
        self.right_yaw_offset = float(right_yaw_offset)
        self.descent_speed = float(descent_speed)
        self.settled_frames = int(settled_frames)
        self.micro_lift = float(micro_lift)
        x, y, yaw = map(float, pot_pose)
        if over_yaw_gate and (yaw > -.2 or y < -.03):
            self.over_yaw_fraction = 1.
        if high_over_negative_y and y < 0. and yaw < 0.:
            self.over_height = .14
            self.over_speed = 1.2
            self.descent_speed = 1.0
        self.pot_yaw = yaw
        self.pot_rot = R.from_euler("z", yaw)
        c = self.pot_rot.apply(HANDLE_CENTRES_IN_POT)
        c += np.array([x, y, POT_RESET_Z])
        self.handles = [(c[0], yaw + np.pi), (c[1], yaw)]
        self.qctrl = RESET_QCTRL.copy()
        self.ee = RESET_EE.copy()
        self._robot = None
        if exact_kinematics:   # robot-only MuJoCo model: the action mode's own FK/IK (rdb.hybrid.book_kinematics)
            from rdb.hybrid.book_kinematics import RobotKinematics
            self._robot = RobotKinematics(PELVIS_POS)
            self.qctrl = self._robot.initial_qctrl.copy()
            self.ee = self._robot.command_pose(self.qctrl)
        self.R0 = [R.from_rotvec(self.ee[6 * i + 3:6 * i + 6]) for i in (0, 1)]
        self.retries = [0, 0]
        if self.adaptive_yaw_ik_threshold > 0. and abs(yaw) > self.far_arm_yaw_threshold:
            far = 1 if yaw > 0. else 0
            c, bar = self.handles[far]
            target = c + [self.GRASP_DX, 0., self.GRASP_DZ]
            candidate_yaw = self.handle_yaw(far, bar)
            edge = (self.adaptive_negative_yaw_edge_x > 0. and
                    x > self.adaptive_negative_yaw_edge_x and y < 0. and yaw < -.1)
            if edge or self._yaw_ik_cost(far, target, candidate_yaw) > self.adaptive_yaw_ik_threshold:
                self.YAW_CAP = 1.40
        self.handle_x_offsets = [0., 0.]
        if (adaptive_handle_x or adaptive_handle_x_strict) and abs(yaw) > self.far_arm_yaw_threshold:
            far = 1 if yaw > 0. else 0
            centre, bar = self.handles[far]
            target = centre + [self.GRASP_DX, 0., self.GRASP_DZ]
            target_yaw = self.handle_yaw(far, bar)
            if self._yaw_ik_cost(far, target, target_yaw) > .15:
                offsets = (-.04,) if adaptive_handle_x_strict else (-.02, -.03, -.04)
                for offset in offsets:
                    trial = target + self.pot_rot.apply([offset, 0., 0.])
                    if (self._yaw_ik_cost(far, trial, target_yaw) < .02 and
                            (not adaptive_handle_x_strict or
                             self._yaw_ik_cost(far, target + self.pot_rot.apply([-.03, 0., 0.]),
                                               target_yaw) > .3)):
                        self.handle_x_offsets[far] = offset
                        break
        self.yaw = [0., 0.]
        self.grip = [0., 0.]
        self._grip_obs = np.zeros(2)
        self._anchor_pads_z = None
        self.initial_z = POT_RESET_Z
        self.phase = "over"
        self.tick = self.age = self.stable = 0
        self._retry_hold = [False, False]
        self._first = True
        self.set_phase("over")

    def cmd_pads(self):
        return [self.ee[6 * i:6 * i + 3] + PELVIS_POS +
                R.from_rotvec(self.ee[6 * i + 3:6 * i + 6]).apply(PAD_IN_EE)
                for i in (0, 1)]

    def handle_yaw(self, i, bar):
        axis = wrap(bar + np.pi / 2, np.pi)
        if self.yaw_mode == "nearest":
            return float(np.clip(axis, -self.YAW_CAP, self.YAW_CAP))
        if i == 0 and axis > 0:
            axis -= np.pi
        if i == 1 and axis < 0:
            axis += np.pi
        if self.retry_yaw_alternate and self.retries[i] % 2:
            other = axis - np.copysign(np.pi, axis)
            return float(np.clip(other, -self.YAW_CAP, self.YAW_CAP))
        if self.yaw_mode == "cross_if_clipped" and abs(axis) > self.YAW_CAP:
            return float(np.clip(wrap(bar + np.pi / 2, np.pi), -self.YAW_CAP, self.YAW_CAP))
        return float(np.clip(axis, -self.YAW_CAP, -.50) if i == 0
                     else np.clip(axis + self.right_yaw_offset, .50, self.YAW_CAP))

    def _yaw_ik_cost(self, i, target_pad, yaw):
        """Predicted pad miss for one candidate wrist rotation, using only the static robot model."""
        target_r = R.from_euler("z", yaw) * self.R0[i]
        target_p = np.asarray(target_pad) - PELVIS_POS - target_r.apply(PAD_IN_EE)
        q0 = self.qctrl[7 * i:7 * i + 7]
        q1 = solve_arm_ik(q0, i, target_p, target_r)
        p, rot, _ = arm_fk_jac(q1, i)
        pad = p + PELVIS_POS + rot.apply(PAD_IN_EE)
        return float(np.linalg.norm(pad - target_pad) +
                     .02 * (target_r * rot.inv()).magnitude())

    def holding(self, i):
        return bool(self.grip[i] >= .5 and GRIP_THRESHOLD <= self._grip_obs[i] < .9)

    def estimated_z(self):
        if self._anchor_pads_z is None:
            return self.initial_z
        return float(self.initial_z + np.median(
            np.array([p[2] for p in self.cmd_pads()]) - self._anchor_pads_z))

    def ee_lag(self, obs):
        """Largest measured-to-commanded EE position gap from public arm joints."""
        measured = measured_ee(obs)
        return max(float(np.linalg.norm(measured[j:j + 3] - self.ee[j:j + 3]))
                   for j in (0, 6))

    def set_phase(self, phase):
        old = getattr(self, "phase", None)
        if phase == "over" and old == "lower":
            self._anchor_pads_z = None
        if phase == "over":
            self._descent_replans = 0
        self.phase = phase
        self.age = self.stable = 0
        ps = self.cmd_pads()
        if phase == "lift" and self._anchor_pads_z is None:
            self._anchor_pads_z = np.array([p[2] for p in ps])
        z_est = self.estimated_z() if phase == "lower" else self.initial_z
        goals = []
        for i in (0, 1):
            c, bar = self.handles[i]
            yaw = self.handle_yaw(i, bar)
            p = ps[i]
            c = c + [self.GRASP_DX, 0., 0.]
            c = c + self.pot_rot.apply([self.grasp_offset_local_x + self.handle_x_offsets[i], 0., 0.])
            if abs(self.pot_yaw) > self.far_arm_yaw_threshold and (
                    (self.pot_yaw > 0. and i == 1) or (self.pot_yaw < 0. and i == 0)):
                c = c + [self.far_arm_dx, 0., 0.]
            if i == 1:
                c = c + self.right_offset
            if self.retries[i] and self.retry_schedule:
                dx, dy, dz, dyaw = self.retry_schedule[min(self.retries[i] - 1,
                                                           len(self.retry_schedule) - 1)]
                c = c + self.pot_rot.apply([dx, dy, dz])
                yaw += dyaw
            if self.hold_grasp_on_retry and self._retry_hold[i] and phase in ("over", "descend", "settle"):
                goal = p; grip = 1.; yaw = self.yaw[i]
            elif phase == "over":
                goal = c + [0., 0., self.over_height]; grip = 0.
            elif phase == "descend":
                goal = c + [0., 0., self.GRASP_DZ]; grip = 0.
            elif phase == "settle":
                goal = p; grip = 0.; yaw = self.yaw[i]
            elif phase == "lift":
                goal = p + [self.LIFT_DX, 0., .16]; grip = 1.
            elif phase == "micro":
                goal = p + [0., 0., self.micro_lift]; grip = 1.
            elif phase == "lower":
                goal = p + [0., 0., min(0., self.initial_z - z_est)]; grip = 1.
            else:
                goal = p; grip = 1.; yaw = self.yaw[i]
            goals.append(Seg(p, goal, self.yaw[i], yaw, grip,
                             min_steps=8 if phase == "descend" else 6,
                             speed=self.LIFT_SPEED if phase == "lift" else
                             self.descent_speed if phase == "descend" else
                             self.over_speed if phase == "over" else 1.,
                             yaw_fraction=self.over_yaw_fraction if phase == "over" else 1.))
        n = max(s.move for s in goals)
        for s in goals:
            s.move = s.n = n
        self.segs = goals

    def _write(self, a, i, pos, yaw):
        self.yaw[i] = yaw
        target = R.from_euler("z", yaw) * self.R0[i]
        current = R.from_rotvec(self.ee[6 * i + 3:6 * i + 6])
        a[6 * i:6 * i + 3] = pos - self._pads[i]
        a[6 * i + 3:6 * i + 6] = (target * current.inv()).as_rotvec()

    def _integrate_action(self, a):
        """Predict target joints after IK and velocity clamp, then recompute EE.

        EE deltas compose on FK of the *commanded* joint targets in the real
        action mode. Simply adding the emitted deltas drifts on IK rejection.
        """
        if self._robot is not None:
            ee = self._robot.command_pose(self.qctrl).astype(np.float32)
            for i in (0, 1):
                j = 6 * i
                q0 = self.qctrl[7 * i:7 * i + 7].copy()
                target_p = (ee[j:j + 3] + a[j:j + 3]).astype(float)
                target_rv = (R.from_rotvec(a[j + 3:j + 6].astype(float)) *
                             R.from_rotvec(ee[j + 3:j + 6].astype(float))).as_rotvec().astype(np.float32)
                q1 = self._robot.solve(self.qctrl, i, target_p, R.from_rotvec(target_rv.astype(float)))
                self.qctrl[7 * i:7 * i + 7] = q0 + np.clip(q1 - q0, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
            self.ee = self._robot.command_pose(self.qctrl)
            return
        for i in (0, 1):
            j = 6 * i
            q0 = self.qctrl[7 * i:7 * i + 7]
            current_p, current_r, _ = arm_fk_jac(q0, i)
            target_p = current_p + a[j:j + 3]
            target_r = R.from_rotvec(a[j + 3:j + 6]) * current_r
            q1 = solve_arm_ik(q0, i, target_p, target_r)
            self.qctrl[7 * i:7 * i + 7] += np.clip(q1 - q0, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
            p, rot, _ = arm_fk_jac(self.qctrl[7 * i:7 * i + 7], i)
            self.ee[j:j + 3] = p
            self.ee[j + 3:j + 6] = rot.as_rotvec()

    def act(self, obs: dict) -> np.ndarray:
        if self._first:
            self._first = False
            return np.zeros(14, np.float32)
        previous_opening = self._grip_obs.copy()
        self._grip_obs = np.asarray(obs["proprioception_grippers"], float).copy()
        # qctrl and its FK are tracked from the policy's own actions. The
        # observation is still used for the grasp test below.
        self.tick += 1
        self.age += 1
        held = [self.holding(i) for i in (0, 1)]
        if self.phase in ("micro", "lift") and not all(held):
            self.set_phase("verifylift")
        if self.phase == "verifylift":
            if all(held):
                self.set_phase("lift")
            elif self.age >= 5:
                for i in (0, 1):
                    self.retries[i] += int(not held[i])
                if self.hold_grasp_on_retry:
                    self._retry_hold = held.copy()
                self.set_phase("lower")
        if self.phase in ("over", "descend", "micro", "lift", "lower") and all(s.done for s in self.segs):
            if self.phase == "descend":
                residual = max(np.linalg.norm(p - s.p1)
                               for p, s in zip(self.cmd_pads(), self.segs))
                if self._descent_replans < self.descent_replans and residual > self.replan_tol:
                    self._descent_replans += 1
                    self.set_phase("descend")
                else:
                    self._settle_duration = max(self.preclose_wait,
                                                self.retry_wait if any(self.retries) else 0)
                    self._settle_by_error = (self.settle_error_m > 0. and
                                             self.ee_lag(obs) > self.settle_error_m)
                    self.set_phase("settle" if self._settle_duration or self._settle_by_error else "close")
            else:
                self.set_phase(dict(over="descend", micro="lift", lift="done",
                                    lower="over")[self.phase])
        if self.phase == "settle" and self.age >= self._settle_duration:
            if (not self._settle_by_error or self.ee_lag(obs) <= self.settle_error_m or
                    self.age >= max(self._settle_duration, self.settle_max_steps)):
                self.set_phase("close")
        if self.phase == "done" and self.age >= 8:
            self.set_phase("lift" if all(held) else "lower")
        if self.phase == "close" and self.age >= 10:
            self.set_phase("verify")
        if self.phase == "verify":
            opening_settled = (self.settled_frames <= 1 or
                               np.max(np.abs(self._grip_obs - previous_opening)) <= .05)
            self.stable = self.stable + 1 if all(held) and opening_settled else 0
            if self.stable >= self.settled_frames:
                self.set_phase("micro" if self.micro_lift > 0. else "lift")
            elif self.age >= 10:
                for i in (0, 1):
                    self.retries[i] += int(not held[i])
                if self.hold_grasp_on_retry:
                    self._retry_hold = held.copy()
                self.set_phase("over")
        a = np.zeros(14, np.float32)
        self._pads = self.cmd_pads()
        for i, seg in enumerate(self.segs):
            pos, yaw = seg.next()
            self._write(a, i, pos, yaw)
            self.grip[i] = seg.grip
        a[12:] = self.grip
        self._integrate_action(a)
        return a
