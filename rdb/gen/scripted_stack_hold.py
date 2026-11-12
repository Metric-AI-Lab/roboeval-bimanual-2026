"""StackTwoBlocks teacher that grasps BOTH blocks at once; privileged training data only.

The challenge's task progress for StackTwoBlocks has 3 stages: (1) a block grasped, (2) both blocks held at
the same moment, (3) stacked. One-arm stacking (the human demos and our policies) never reaches stage 2,
so task progress caps at 2/3 even on success. Here the arm nearer a block ("stacker") grasps it while the
other arm ("steady") grasps the base block in place; both hold together for a moment, the steady arm
releases and backs away, then the stacker lifts, carries and stacks. Grasp checks retry misses.

  python -m rdb.gen.scripted_stack_hold --seeds 105700:105720 --dry --noise 0
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from rdb.gen.scripted_stack_two_blocks import CUBE, Segment, StackExpert, wrap


class HoldStackExpert(StackExpert):
    QUIET = ('close', 'release', 'done', 'wait', 'hold', 'wait0')
    NEAR = 0.15          # blocks closer than this (m): sequential grasps
    ASIDE = 0.12

    def __init__(self, env, rng, noise=1.0):
        super().__init__(env, rng, noise, mirror=0.0)
        self.home = [self.pinch_cmd(i).copy() for i in range(2)]
        self.retries = [0, 0]
        self.trace = []
        self.tick = 0

    def plan(self):
        super().plan()                       # self.arm = stacker, self.pick / self.base = block indices
        self.steady = 1 - self.arm
        self.own = [None, None]
        self.own[self.arm], self.own[self.steady] = self.pick, self.base
        self.stages = [None, None]
        self.stages[self.arm] = ['approach', 'descend', 'close', 'wait', 'lift', 'carry', 'lower', 'release',
                                 'retreat', 'done']
        self.stages[self.steady] = ['approach', 'descend', 'close', 'hold', 'release', 'retreat', 'done']
        blk = self.blocks()
        self.near = np.linalg.norm((blk[0][0] - blk[1][0])[:2]) < self.NEAR
        if self.near:
            # Hands at adjacent blocks collide: the stacker lifts its block and moves it aside first.
            self.stages[self.arm] = ['approach', 'descend', 'close', 'lift', 'aside', 'wait', 'carry', 'lower',
                                     'release', 'retreat', 'done']
            self.stages[self.steady] = ['wait0'] + self.stages[self.steady]
        self.k = [0, 0]
        self.segs = [None, None]
        self.both_held = False

    def st(self, i):
        return self.stages[i][self.k[i]]

    def _seg(self, i):
        st, blk, p, y = self.st(i), self.blocks(), self.pinch_cmd(i), self.yaw[i]
        pa, ya = blk[self.own[i]]
        po = blk[self.own[1 - i]][0]
        gy = self._grasp_yaw(ya, i=i, pick=pa, other=po)
        if st == 'approach':
            return Segment(p, pa + [0, 0, 0.055], y, gy, 0.0)
        if st == 'descend':
            return Segment(p, pa + [0, 0, 0.006], y, gy, 0.0, min_steps=6)
        if st == 'close':
            return Segment(p, p, y, y, 1.0, min_steps=1, hold=6)
        if st == 'aside':
            sgn = 1.0 if self.sides[i].value == 'Left' else -1.0
            return Segment(p, p + [-0.03, sgn * self.ASIDE, 0.02], y, y, 1.0, min_steps=6)
        if st in ('wait', 'hold', 'wait0'):
            return Segment(p, p, y, y, 1.0, min_steps=1, hold=1)
        if st == 'release':
            return Segment(p, p, y, y, 0.0, min_steps=1, hold=4)
        if st == 'retreat' and i == self.steady:
            # back to its start pose, out of the stacker's way
            return Segment(p, self.home[i], y, 0.0, 0.0, min_steps=8)
        if st == 'retreat':
            return Segment(p, p + [0, 0, 0.06], y, y, 0.0, min_steps=6)
        base = blk[self.base]
        if st == 'lift':
            return Segment(p, p + [0, 0, max(0.05, base[0][2] + CUBE + 0.045 - p[2])], y, y, 1.0, min_steps=5)
        if st == 'carry':
            return Segment(p, p + (base[0] + [0, 0, CUBE + 0.03] - pa), y, y + wrap(base[1] - ya, np.pi / 2), 1.0)
        if st == 'lower':
            return Segment(p, p + (base[0] + [0, 0, CUBE + 0.003] - pa), y, y, 1.0, min_steps=5)
        return Segment(p, p, y, y, 0.0, min_steps=1, hold=1000)

    def _advance(self, i):
        st = self.st(i)
        if st == 'close' and not self.holding(i, self.own[i]):
            self.retries[i] += 1
            self.k[i] = 0
            self.grip[i] = 0.0
            return
        if st == 'wait0':
            if self.k[self.arm] < self.stages[self.arm].index('wait'):
                return
        if st == 'hold':
            if not self.both_held:
                if not self.holding(i, self.own[i]):          # lost it while waiting: grasp again
                    self.retries[i] += 1
                    self.k[i] = 0
                    self.grip[i] = 0.0
                return
        if st == 'wait':
            # stacker keeps its block on the table until both are held, then until the steady arm is clear
            if not self.both_held or self.st(self.steady) not in ('retreat', 'done'):
                if not self.holding(i, self.own[i]):
                    self.retries[i] += 1
                    self.k[i] = 0
                    self.grip[i] = 0.0
                return
        if st == 'lower' and not self.holding(i, self.own[i]):
            self.k[i] = self.stages[i].index('release')
            return
        if st in ('lift', 'aside', 'carry') and not self.holding(i, self.own[i]):
            self.retries[i] += 1
            self.k[i] = 0
            self.grip[i] = 0.0
            return
        self.k[i] = min(self.k[i] + 1, len(self.stages[i]) - 1)

    def act(self):
        if self.tick % 20 == 0:
            from rdb.gen.scripted_pack_box_h import thermal_guard
            thermal_guard()
        self.tick += 1
        if self.holding(self.arm, self.pick) and self.holding(self.steady, self.base):
            self.both_held = True
        a = np.zeros(14, np.float32)
        ee = self.cmd_pose()
        for i in range(2):
            if self.segs[i] is not None and self.segs[i].done:
                self._advance(i)
                self.segs[i] = None
            if self.segs[i] is None:
                self.segs[i] = self._seg(i)
            pos, yaw = self.segs[i].next()
            self.yaw[i] = yaw
            rot_t = R.from_rotvec([0, 0, yaw]) * self.R0[i]
            a[6 * i: 6 * i + 3] = pos - rot_t.apply(self.off[i]) - ee[6 * i: 6 * i + 3]
            a[6 * i + 3: 6 * i + 6] = (rot_t * R.from_rotvec(ee[6 * i + 3: 6 * i + 6]).inv()).as_rotvec()
            self.grip[i] = self.segs[i].grip
        a[12:] = self.grip
        self.stage = self.label
        self.trace.append(dict(step=self.tick, phase=self.label, both_held=self.both_held))
        return a

    @property
    def label(self):
        return '+'.join(self.st(i) for i in range(2))

    def perturb(self, a):
        if self.noise <= 0:
            return a
        e = a.copy()
        for i in range(2):
            if self.st(i) in self.QUIET:
                continue
            e[6 * i: 6 * i + 3] += self.rng.normal(0, 0.0025 * self.noise, 3)
            e[6 * i + 3: 6 * i + 6] += self.rng.normal(0, 0.015 * self.noise, 3)
        return e


if __name__ == '__main__':
    from rdb.gen.scripted_h_runner import main
    main('StackTwoBlocks', HoldStackExpert, '105700:105720')
