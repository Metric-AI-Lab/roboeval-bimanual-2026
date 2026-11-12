"""Scripted CubeHandover expert (privileged rod pose; training data only). See rdb.gen.scripted_common.

The arm nearer the rod ("giver") grasps the rod's end on its own side, fingers across the rod, lifts it
to a handover point in the middle while turning it so the rod lies along y with the free end toward
the other arm; the "taker" grasps the free end from above, then the giver releases and backs off.
Success = taker holds, giver doesn't, rod off the table.

  python -m rdb.gen.scripted_cube_handover --seeds 100000:100050 --dry
"""
from __future__ import annotations

import numpy as np

from rdb.gen.scripted_common import BimanualScript, Seg, cli, wrap

HALF = 0.10           # rod half length (box 0.04 x 0.2 x 0.04, long axis = local y)


class HandoverExpert(BimanualScript):
    END = 0.06        # grasp points this far from the rod centre along its axis
    H = (0.55, 0.0, 1.13)   # handover point (rod centre)
    CLOSE_HOLD = 6

    def plan(self):
        self.rod = self.env.cube
        self.rg = self.geoms_of(self.rod.body.mjcf.full_identifier)[0]
        c = self.rod.body.get_position()
        self.giver = 0 if c[1] >= 0 else 1                  # LEFT arm is at +y
        self.taker = 1 - self.giver
        self.sgn = 1.0 if self.giver == 0 else -1.0          # +y side of the giver
        self.stages = [None, None]
        self.stages[self.giver] = ["over", "approach", "descend", "close", "lift", "carry", "sync:at",
                                   "sync:taken", "release", "retreat", "done"]
        self.stages[self.taker] = ["sync:at", "over", "approach", "descend", "close", "sync:taken", "settle", "done"]

    def axis(self) -> np.ndarray:
        """Unit long axis of the rod (world), oriented toward the giver's side (+sgn y)."""
        u = self.d.geom_xmat[self.rg].reshape(3, 3)[:, 1].copy()
        u[2] = 0.0
        u /= np.linalg.norm(u)
        return u if u[1] * self.sgn >= 0 else -u

    def point(self, end: float) -> np.ndarray:
        """Point on the rod axis, `end` metres from the centre toward the giver (top surface height)."""
        c = self.d.geom_xpos[self.rg].copy()
        return c + self.axis() * end

    def segment(self, i: int, st: str) -> Seg:
        p, y = self.pinch(i), self.yaw[i]
        u = self.axis()
        across = np.arctan2(u[1], u[0]) + np.pi / 2          # finger closing direction: across the rod
        if i == self.giver:
            g = self.point(self.END)
            gy = self.pick_yaw(i, across, -1.0 if i == 0 else 1.0)
            if st == "over":
                return Seg(p, [g[0], g[1], max(p[2], g[2] + 0.10)], y, gy, 0.0)
            if st == "approach":
                return Seg(p, g + [0, 0, 0.04], y, gy, 0.0, speed=0.7)
            if st == "descend":
                return Seg(p, g, y, gy, 0.0, min_steps=6, speed=0.6)
            if st == "close":
                return Seg(p, p, y, y, 1.0, min_steps=1, hold=self.CLOSE_HOLD)
            if st == "lift":
                return Seg(p, p + [0, 0, 0.08], y, y, 1.0, min_steps=6)
            if st == "carry":                               # rod centre to H, rod along y (free end to taker)
                turn = wrap(np.arctan2(self.sgn, 0.0) - np.arctan2(u[1], u[0]), 2 * np.pi)
                tgt = np.array(self.H) + np.array([0, self.sgn * self.END, 0])
                return Seg(p, tgt, y, y + turn, 1.0)
            if st == "release":
                return Seg(p, p, y, y, 0.0, min_steps=1, hold=5)
            if st == "retreat":
                return Seg(p, p + [0, self.sgn * 0.08, 0.06], y, y, 0.0, min_steps=6)
            return Seg(p, p, y, y, self.grip[i], min_steps=1, hold=1000)
        # taker
        g = self.point(-self.END)
        gy = self.pick_yaw(i, across, -1.0 if i == 0 else 1.0)
        if st == "over":
            return Seg(p, [g[0], g[1], max(p[2], g[2] + 0.10)], y, gy, 0.0)
        if st == "approach":
            return Seg(p, g + [0, 0, 0.04], y, gy, 0.0, speed=0.7)
        if st == "descend":
            return Seg(p, g, y, gy, 0.0, min_steps=6, speed=0.6)
        if st == "close":
            return Seg(p, p, y, y, 1.0, min_steps=1, hold=self.CLOSE_HOLD)
        if st == "settle":
            return Seg(p, p + [0, 0, 0.02], y, y, 1.0, min_steps=8)
        return Seg(p, p, y, y, self.grip[i], min_steps=1, hold=1000)

    def advance(self, i: int, st: str):
        if st == "close" and not self.holding(i, self.rod):
            self.grip[i] = 0.0
            return self.stages[i].index("approach")
        return None


if __name__ == "__main__":
    cli("CubeHandover", {"*": HandoverExpert}, "100000:100050")
