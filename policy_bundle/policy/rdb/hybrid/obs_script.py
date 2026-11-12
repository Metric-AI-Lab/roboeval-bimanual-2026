"""Observation-only counterpart of rdb.gen.scripted_common.BimanualScript, for porting teachers to the pose hybrid.

Same stage / segment / until / advance machinery and the same Seg min-jerk moves, but every robot quantity comes
from legal inputs: the commanded joint targets are tracked from our own actions with a robot-only MuJoCo model
(rdb.hybrid.book_kinematics: same FK, float32 rounding and IK update as the EE-delta action mode), fingertip-pad midpoints from FK + the
finger joints in proprioception, and "holding" from the gripper opening while closed is commanded. Object
geometry comes from the pose regressor (subclasses). `pelvis` is the robot base position of the task's scene.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rdb.hybrid.book_kinematics import RobotKinematics
from rdb.hybrid.pot_controller import MAX_JOINT_DELTA, PAD_IN_EE, min_jerk

DT, V_AVG, W_AVG = 1 / 20, 0.22, 1.6


def wrap(a: float, period: float) -> float:
    return -((-a + period / 2) % period - period / 2)


class Seg:
    """rdb.gen.scripted_common.Seg (min-jerk move of grasp point, yaw and tilt)."""

    def __init__(self, p0, p1, y0, y1, grip, t0=None, t1=None, min_steps=6, hold=0, speed=1.0):
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.y0, self.y1, self.grip = y0, y1, grip
        self.t0 = np.zeros(3) if t0 is None else np.asarray(t0, float)
        self.t1 = self.t0 if t1 is None else np.asarray(t1, float)
        dur = max(np.linalg.norm(self.p1 - self.p0) / (V_AVG * speed), abs(y1 - y0) / W_AVG,
                  np.linalg.norm(self.t1 - self.t0) / W_AVG)
        self.move = max(min_steps, int(np.ceil(dur / DT)))
        self.n = self.move + hold
        self.k = 0

    def next(self):
        self.k += 1
        s = min_jerk(self.k / self.move)
        return (self.p0 + (self.p1 - self.p0) * s, self.y0 + (self.y1 - self.y0) * s,
                self.t0 + (self.t1 - self.t0) * s)

    @property
    def done(self) -> bool:
        return self.k >= self.n


class ArcSeg(Seg):
    """rdb.gen.scripted_common.ArcSeg: grasp point rotated about a world axis through `centre`."""

    def __init__(self, p0, centre, axis, angle, yaw, grip, tilt=None, w=0.9, min_steps=10, hold=0):
        super().__init__(p0, p0, yaw, yaw, grip, tilt, min_steps=min_steps, hold=hold)
        self.c, self.r0 = np.asarray(centre, float), np.asarray(p0, float) - np.asarray(centre, float)
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)
        self.angle = angle
        self.move = max(min_steps, int(np.ceil(abs(angle) / w / DT)))
        self.n = self.move + hold

    def next(self):
        self.k += 1
        s = min_jerk(self.k / self.move)
        return self.c + R.from_rotvec(self.axis * self.angle * s).apply(self.r0), self.y0, self.t0


class ObsScript:
    """Subclasses set self.stages in plan() and implement segment(); call act(obs) every step."""
    MAX_JOINT_DELTA = MAX_JOINT_DELTA   # action-mode clamp: 2.175 rad/s x 25 sub-steps x physics timestep

    def __init__(self, pelvis, grip_lo=.05, grip_hi=.9):
        self.pelvis = np.asarray(pelvis, float)
        self.grip_lo, self.grip_hi = grip_lo, grip_hi
        self._robot = RobotKinematics(self.pelvis)
        self.qctrl = self._robot.initial_qctrl.copy()
        self.finger = np.zeros(2)          # half the finger-joint difference per arm (pad midpoint shift)
        self.openings = np.zeros(2)
        ee = self.cmd_pose()
        self.R0 = [R.from_rotvec(ee[6 * i + 3:6 * i + 6]) for i in (0, 1)]
        self._pads = self.cmd_pads()
        self.home = [p.copy() for p in self._pads]
        self.yaw, self.tilt, self.grip = [0.0, 0.0], [np.zeros(3), np.zeros(3)], [0.0, 0.0]
        self.k = [0, 0]
        self.segs: List[Optional[Seg]] = [None, None]
        self.stages: List[List[str]] = [["done"], ["done"]]
        self.retries = [0, 0]
        self.tick = 0
        self._first = True
        self.plan()

    # ---- robot state (legal) --------------------------------------------------------------------
    def cmd_pose(self) -> np.ndarray:
        return self._robot.command_pose(self.qctrl)

    def cmd_pads(self) -> List[np.ndarray]:
        return [p + rot @ (PAD_IN_EE + [0., self.finger[i], 0.])
                for i, (p, rot) in enumerate(self._robot.poses(self.qctrl, world=True))]

    def pinch(self, i: int, ee=None) -> np.ndarray:
        return self._pads[i].copy()

    def holding(self, i: int, *_) -> bool:
        return bool(self.grip[i] >= .5 and self.grip_lo <= self.openings[i] < self.grip_hi)

    # ---- to implement ---------------------------------------------------------------------------
    def plan(self):
        raise NotImplementedError

    def segment(self, i: int, stage: str) -> Seg:
        raise NotImplementedError

    def until(self, i: int, stage: str) -> bool:
        return False

    def advance(self, i: int, stage: str) -> Optional[int]:
        return None

    # ---- stepping (BimanualScript.act) ----------------------------------------------------------
    def stage(self, i: int) -> str:
        return self.stages[i][self.k[i]]

    def hold_seg(self, i: int, grip=None, steps=1) -> Seg:
        p = self.pinch(i)
        return Seg(p, p, self.yaw[i], self.yaw[i], self.grip[i] if grip is None else grip, self.tilt[i],
                   min_steps=1, hold=steps - 1)

    def _next_stage(self, i: int):
        j = self.advance(i, self.stage(i))
        self.k[i] = j if j is not None else min(self.k[i] + 1, len(self.stages[i]) - 1)

    def _passed(self, i: int, sync_stage: str) -> bool:
        return sync_stage in self.stages[i] and self.stages[i].index(sync_stage) < self.k[i]

    def observe(self, obs):
        q = np.asarray(obs["proprioception"], float)
        self.openings = np.asarray(obs["proprioception_grippers"], float).copy()
        self.finger = np.array([.5 * (q[8] - q[7]), .5 * (q[17] - q[16])])

    def step(self) -> np.ndarray:
        a = np.zeros(14, np.float32)
        ee = self.cmd_pose()
        self._pads = self.cmd_pads()
        for i in (0, 1):
            if self.segs[i] is not None and not self.segs[i].done and self.until(i, self.stage(i)):
                self.segs[i].k = self.segs[i].n
            if self.segs[i] is not None and self.segs[i].done:
                self._next_stage(i)
                self.segs[i] = None
            st = self.stage(i)
            if st.startswith("sync:"):
                if self.stage(1 - i) == st or self._passed(1 - i, st):
                    self.k[i] += 1
                    st = self.stage(i)
                    self.segs[i] = None
                else:
                    self._write(a, ee, i, self.pinch(i), self.yaw[i], self.tilt[i])
                    continue
            if self.segs[i] is None:
                self.segs[i] = self.segment(i, st)
            pos, yaw, tilt = self.segs[i].next()
            self._write(a, ee, i, pos, yaw, tilt)
            self.grip[i] = self.segs[i].grip
        a[12:] = self.grip
        return a

    def _write(self, a, ee, i, pos, yaw, tilt):
        self.yaw[i], self.tilt[i] = yaw, tilt
        rot_t = R.from_rotvec(tilt) * R.from_rotvec([0, 0, yaw]) * self.R0[i]
        a[6 * i:6 * i + 3] = pos - self._pads[i]
        a[6 * i + 3:6 * i + 6] = (rot_t * R.from_rotvec(ee[6 * i + 3:6 * i + 6]).inv()).as_rotvec()

    def integrate(self, a: np.ndarray):
        """Advance the tracked joint targets as the action mode will (compose on the float32 commanded pose, IK,
        velocity clamp); as rdb.hybrid.book_controller.PickBookController._integrate_action."""
        ee = self.cmd_pose().astype(np.float32)
        for i in (0, 1):
            j = 6 * i
            q0 = self.qctrl[7 * i:7 * i + 7].copy()
            target_p = (ee[j:j + 3] + a[j:j + 3]).astype(float)
            target_rv = (R.from_rotvec(a[j + 3:j + 6].astype(float)) *
                         R.from_rotvec(ee[j + 3:j + 6].astype(float))).as_rotvec().astype(np.float32)
            q1 = self._robot.solve(self.qctrl, i, target_p, R.from_rotvec(target_rv.astype(float)))
            self.qctrl[7 * i:7 * i + 7] = q0 + np.clip(q1 - q0, -self.MAX_JOINT_DELTA, self.MAX_JOINT_DELTA)

    def act(self, obs) -> np.ndarray:
        self.observe(obs)
        if self._first:          # demonstration convention: first action is an open no-op
            self._first = False
            return np.zeros(14, np.float32)
        self.tick += 1
        a = self.postprocess(self.step())
        self.integrate(a)
        return a

    def postprocess(self, a: np.ndarray) -> np.ndarray:
        return a
