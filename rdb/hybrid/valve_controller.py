"""Observation-only RotateValve controller: rdb/gen/scripted_rotate_valve_h.py on legal inputs.

Input: the two valve hub-top points [(x, y, z), (x, y, z)] estimated at reset (valves are fixed to the table), then
the standard observation every step. The teacher reads the valve joint angle to decide whether a twist worked; the
policy cannot, so every grasped twist is followed by the top-off twist (0.73 + 0.32 rad) and a twist that loses
the grip (gripper opening) is retried from the approach.
"""
from __future__ import annotations

import numpy as np

from rdb.hybrid.obs_script import ObsScript, Seg

VALVE_PELVIS = np.zeros(3)   # calibrated: RotateValve robot base (see verify_valve_controller --calibrate)
SETTLE_DZ = 0.0              # hubs sit 2.8 mm lower once the scene settles after reset (measured, every seed)


class ValveController(ObsScript):
    MAX_JOINT_DELTA = 2.175 * .025   # RotateValve runs physics at 1 ms (others 2 ms): half the per-step clamp
    TWIST = 0.73
    CLOSE_HOLD = 7
    TWIST_STEPS = 30
    ACTION_SMOOTHING = .55

    SYNC_STEPS = dict(over=24, approach=10, descend=10)   # sync mode: both arms move in lockstep (coordination)

    def __init__(self, hub_tops, pelvis=VALVE_PELVIS, passes=1, sync=False, **kw):
        self.tops = [np.asarray(h, float) + [0., 0., SETTLE_DZ] for h in hub_tops]
        self.passes = int(passes)
        self.sync = bool(sync)
        super().__init__(pelvis, **kw)

    def plan(self):
        ys = [t[1] for t in self.tops]
        self.mine = [int(np.argmax(ys)), int(np.argmin(ys))]
        turn = ["approach", "descend", "close", "twist", "topoff", "release", "retreat"]
        # passes > 1: grasp and turn again (a grip can slip around the hub unnoticed; the angle is not observable)
        extra = [f"{s}~{k}" for k in range(1, self.passes) for s in turn]
        lead = [] if self.sync else ["wait"]
        self.stages = [lead + ["over"] + turn + extra + ["done"], ["over"] + turn + extra + ["done"]]
        self.previous_arm_action = np.zeros(12, np.float32)
        self.lost = [False, False]

    def hub_top(self, i):
        return self.tops[self.mine[i]].copy()

    def postprocess(self, a):
        for i in (0, 1):   # a grip lost during the twist (no valve angle available: use the gripper)
            if self.stage(i).split("~")[0] in ("twist", "topoff") and self.grip[i] >= .5 and not self.holding(i):
                self.lost[i] = True
        a[:12] = (1 - self.ACTION_SMOOTHING) * a[:12] + self.ACTION_SMOOTHING * self.previous_arm_action
        self.previous_arm_action = a[:12].copy()
        return a

    def segment(self, i, stage):
        stage = stage.split("~")[0]
        p, y = self.pinch(i), self.yaw[i]
        c = self.hub_top(i)
        if self.sync and stage in self.SYNC_STEPS:   # fixed duration, same for both arms
            n = self.SYNC_STEPS[stage]
            to = {"over": [c[0], c[1], max(p[2], c[2] + .065)], "approach": c + [0, 0, .035], "descend": c}[stage]
            return Seg(p, to, y, y if stage == "descend" else -.08, 0.0, min_steps=n, speed=50.)
        if stage == "wait":
            return self.hold_seg(i, 0.0, steps=10)
        if stage == "over":
            return Seg(p, [c[0], c[1], max(p[2], c[2] + .065)], y, -.08, 0.0,
                       min_steps=38 if i == 0 else 10, speed=.75 if i == 0 else .9)
        if stage == "approach":
            return Seg(p, c + [0, 0, .035], y, -.08, 0.0, min_steps=9, speed=.65 if i == 0 else .75)
        if stage == "descend":
            return Seg(p, c, y, y, 0.0, min_steps=10, speed=.55 if i == 0 else .6)
        if stage == "close":
            return Seg(p, p, y, y, 1.0, min_steps=1, hold=self.CLOSE_HOLD)
        if stage == "twist":
            self.lost[i] = False
            return Seg(p, p, y, y + self.TWIST, 1.0, min_steps=self.TWIST_STEPS)
        if stage == "topoff":
            return Seg(p, p, y, y + .32, 1.0, min_steps=18)
        if stage == "release":
            return Seg(p, p, y, y, 0.0, min_steps=1, hold=2)
        if stage == "retreat":
            return Seg(p, p + [0, 0, .020], y, y, 0.0, min_steps=8, speed=1.)
        return self.hold_seg(i, 0.0, steps=1000)

    def advance(self, i, stage):
        base, _, rep = stage.partition("~")
        approach = self.stages[i].index("approach" + (f"~{rep}" if rep else ""))
        if base == "close" and not self.holding(i):
            self.retries[i] += 1
            return approach
        if base in ("twist", "topoff") and (self.lost[i] or not self.holding(i)):
            self.retries[i] += 1
            self.grip[i] = 0.0
            return approach
        return None
