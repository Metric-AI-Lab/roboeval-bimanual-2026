"""Human-ordered, closed-loop RotateValve teacher for training data only.

The right hand turns its valve first. The left hand begins its open-hand
approach while the right finishes the turn. Both use short min-jerk wrist
rotations around the measured hub, with grasp and angle checks.
"""
from __future__ import annotations

import numpy as np

from rdb.gen.scripted_common import BimanualScript, Seg
from rdb.gen.scripted_pack_box_h import thermal_guard


class RotateValveHumanExpert(BimanualScript):
    TWIST = 0.73
    GRASP_DOWN = .012
    CLOSE_HOLD = 7
    TWIST_STEPS = 30
    ACTION_SMOOTHING = .55

    def plan(self):
        import mujoco

        self.hub = []
        for valve in self.env.valves:
            name = valve.body.mjcf.full_identifier
            ids = [g for g in self.geoms_of(name)
                   if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("unnamed_geom_2")]
            self.hub.append((valve, ids[0]))
        ys = [self.d.geom_xpos[g][1] for _, g in self.hub]
        self.mine = [int(np.argmax(ys)), int(np.argmin(ys))]
        self.stages = [
            ["wait", "over", "approach", "descend", "close", "twist", "topoff", "release", "retreat", "done"],
            ["over", "approach", "descend", "close", "twist", "topoff", "release", "retreat", "done"],
        ]
        self.tick = 0
        self.retries = [0, 0]
        self.trace = []
        self.previous_arm_action = np.zeros(12, np.float32)

    def hub_top(self, i):
        geom = self.hub[self.mine[i]][1]
        _, hi = self.geom_aabb(geom)
        centre = self.d.geom_xpos[geom].copy()
        centre[2] = hi[2] - self.GRASP_DOWN
        return centre

    def valve_state(self, i):
        return float(np.mean(self.hub[self.mine[i]][0].get_state()))

    def act(self):
        if self.tick % 20 == 0:
            thermal_guard()
            self.trace.append({"step": self.tick + 1,
                               "stages": [self.stage(i) for i in (0, 1)],
                               "valves": [self.valve_state(i) for i in (0, 1)]})
        self.tick += 1
        action = super().act()
        action[:12] = ((1 - self.ACTION_SMOOTHING) * action[:12]
                       + self.ACTION_SMOOTHING * self.previous_arm_action)
        self.previous_arm_action = action[:12].copy()
        return action

    def segment(self, i, stage):
        p, y = self.pinch(i), self.yaw[i]
        c = self.hub_top(i)
        if stage == "wait":
            return self.hold_seg(i, 0.0, steps=10)
        if stage == "over":
            return Seg(p, [c[0], c[1], max(p[2], c[2] + .065)], y, -.08,
                       0.0, min_steps=38 if i == 0 else 10,
                       speed=.75 if i == 0 else .9)
        if stage == "approach":
            return Seg(p, c + [0, 0, .035], y, -.08, 0.0,
                       min_steps=9,
                       speed=.65 if i == 0 else .75)
        if stage == "descend":
            return Seg(p, c, y, y, 0.0, min_steps=10,
                       speed=.55 if i == 0 else .6)
        if stage == "close":
            return Seg(p, p, y, y, 1.0, min_steps=1, hold=self.CLOSE_HOLD)
        if stage == "twist":
            s = Seg(p, p, y, y + self.TWIST, 1.0, min_steps=self.TWIST_STEPS)
            return s
        if stage == "topoff":
            return Seg(p, p, y, y + .32, 1.0, min_steps=18)
        if stage == "release":
            return Seg(p, p, y, y, 0.0, min_steps=1, hold=2)
        if stage == "retreat":
            return Seg(p, p + [0, 0, .020], y, y, 0.0,
                       min_steps=8, speed=1.)
        return self.hold_seg(i, 0.0, steps=1000)

    def advance(self, i, stage):
        if stage == "close" and not self.holding(i, self.hub[self.mine[i]][0].valve):
            self.retries[i] += 1
            return self.stages[i].index("approach")
        if stage == "twist":
            state = self.valve_state(i)
            if state >= .115:
                return self.stages[i].index("release")
            if state < .025:
                self.retries[i] += 1
                self.grip[i] = 0.0
                return self.stages[i].index("approach")
        if stage == "topoff" and self.valve_state(i) < .105:
            self.retries[i] += 1
            self.grip[i] = 0.0
            return self.stages[i].index("approach")
        return None


if __name__ == "__main__":
    from rdb.gen.scripted_h_runner import main
    main("RotateValve", RotateValveHumanExpert, "106400:106460")
