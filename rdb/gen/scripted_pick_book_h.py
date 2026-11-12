"""Human-style closed-loop book teacher; privileged training data only.

Nudge the book with an open, upright hand, release sideways before withdrawing,
then pitch and pinch the overhang. Grasp checks gate the lift; misses and slips
trigger a slower measured approach. Never use this teacher in a submission.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rdb.gen.scripted_books import BookExpert
from rdb.gen.scripted_common import Seg
from rdb.gen.scripted_pack_box_h import thermal_guard


class PickBookHumanExpert(BookExpert):
    PITCH = .90
    YAW = .80
    PUSH_STOP = .022
    PUSH_SPEED = .35
    TARGET_EDGE = -.005
    HEIGHT_DZ = -.002
    QUIET = BookExpert.QUIET + ('verifylift',)

    def plan(self):
        super().plan()
        self.stages = [['done'], ['done']]
        self.stages[self.arm] = ['side_over', 'side_down', 'push', 'clear', 'clearup',
                                 'over', 'front', 'insert', 'close', 'verifylift', 'lift', 'done']
        self.initial_x = float(self.book.body.get_position()[0])
        # A book close to the edge gets the wrist fully aligned above the work
        # surface first. With more support, combine the initial turn and descent.
        self.direct = self.initial_x > self.edge() + .05
        if self.direct:
            self.k[self.arm] = 1
        self.retries = [0, 0]
        self.tick = 0
        self.trace = []

    def target(self):
        """Mid-thickness pinch point over the measured robot-facing strip."""
        lo, hi = self.book_box()
        x = min(self.edge() + self.TARGET_EDGE, self.book.body.get_position()[0] - .015)
        mat = self.d.geom_xmat[self.bg].reshape(3, 3)
        centre, half = self.m.geom_aabb[self.bg][:3], self.m.geom_aabb[self.bg][3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + centre
        world = corners @ mat.T + self.d.geom_xpos[self.bg]
        ys = []
        for p in world:
            for q in world:
                if (p[0] - x) * (q[0] - x) <= 0 and abs(p[0] - q[0]) > 1e-8:
                    ys.append(float(p[1] + (x - p[0]) / (q[0] - p[0]) * (q[1] - p[1])))
        y = (min(ys) + max(ys)) / 2 if ys else self.book.body.get_position()[1]
        return np.array([x, y, (lo[2] + hi[2]) / 2 + self.HEIGHT_DZ])

    def orientation(self, i):
        yaw = -self.YAW if i == 0 else self.YAW
        tilt = Rotation.from_rotvec([0, 0, yaw]).apply([0, -self.PITCH, 0])
        return yaw, tilt

    def segment(self, i, stage):
        p, y, tilt = self.pinch(i), self.yaw[i], self.tilt[i]
        if i != self.arm:
            return self.hold_seg(i, 0., steps=1000)
        gy, gt = self.orientation(i)
        c = self.target()
        sign = 1 if i == 0 else -1
        lo, hi = self.book_box()
        bc = self.book.body.get_position()
        side = np.array([bc[0] + .02, hi[1] + .035 if sign > 0 else lo[1] - .035, bc[2] - .003])
        if stage == 'side_over':
            return Seg(p, side + [0, 0, .09], y, gy, 0., tilt, np.zeros(3))
        if stage == 'side_down':
            return Seg(p, side, y, gy, 0., tilt, np.zeros(3), min_steps=10, speed=1.1 if self.direct else .7)
        if stage == 'push':
            return Seg(p, p + [-.085, -sign * .085, 0], y, y, 0., tilt, tilt, min_steps=14, speed=self.PUSH_SPEED)
        if stage == 'clear':
            # Break side-pad friction before moving up; an immediate upward
            # withdrawal can lift one edge and tip the book off the counter.
            return Seg(p, p + [.03, sign * .055, .005], y, y, 0., tilt, tilt, min_steps=8, speed=.8)
        if stage == 'clearup':
            return Seg(p, p + [0, 0, .075], y, y, 0., tilt, tilt, min_steps=6)
        if stage == 'over':
            return Seg(p, c + [-.06, 0, .10], y, gy, 0., tilt, gt)
        if stage == 'front':
            return Seg(p, c + [-.06, 0, 0], y, gy, 0., tilt, gt, min_steps=10, speed=.7)
        if stage == 'insert':
            return Seg(p, c, y, gy, 0., tilt, gt, min_steps=10, speed=.6)
        if stage == 'close':
            return Seg(p, p, y, y, 1., tilt, tilt, min_steps=1, hold=10)
        if stage == 'verifylift':
            return self.hold_seg(i, 1., steps=6)
        if stage == 'lift':
            return Seg(p, p + [-.025, 0, .15], y, y, 1., tilt, tilt, min_steps=10, speed=.85)
        return self.hold_seg(i, 1., steps=12)

    def advance(self, i, stage):
        if i != self.arm:
            return None
        if stage == 'close':
            if self.holding(i, self.book):
                return self.stages[i].index('lift')
            self.retries[i] += 1
            was_direct, self.direct = self.direct, False
            # Re-push if the book moved back onto the counter. Otherwise repeat
            # the measured pitched approach with the existing overhang.
            if was_direct or self.book_box()[0][0] > self.edge() - .045:
                return 0
            return self.stages[i].index('over')
        if stage in ('verifylift', 'done'):
            if self.holding(i, self.book):
                return self.stages[i].index('lift')
            self.retries[i] += 1
            self.direct = False
            return 0
        return None

    def until(self, i, stage):
        if i != self.arm:
            return False
        if stage == 'push':
            return self.book.body.get_position()[0] <= self.initial_x - self.PUSH_STOP
        if stage == 'close' and self.direct and self.segs[i].k >= 6:
            return self.holding(i, self.book)
        return stage == 'verifylift' and self.holding(i, self.book)

    def act(self):
        if self.tick % 20 == 0:
            thermal_guard()
        self.tick += 1
        held = self.holding(self.arm, self.book)
        if self.stage(self.arm) == 'lift' and not held:
            self.k[self.arm] = self.stages[self.arm].index('verifylift')
            self.segs[self.arm] = None
        self.trace.append(dict(step=self.tick, phase=self.stage(self.arm), holding=bool(held),
                               object=self.book.body.get_position().tolist()))
        action = super().act()
        self.trace[-1].update(command_phase=self.stage(self.arm), careful=not self.direct)
        return action


if __name__ == '__main__':
    from rdb.gen.scripted_h_runner import main
    main('PickSingleBookFromTable', PickBookHumanExpert, '105400:105460')
