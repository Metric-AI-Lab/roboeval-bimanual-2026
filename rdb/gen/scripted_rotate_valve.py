"""Scripted RotateValve expert (privileged valve poses; training data only). See rdb.gen.scripted_common.

Each arm takes the valve on its side: pinch the hub cap on the wheel axis, twist the hand
counter-clockwise (+z) by TWIST about the grasp point (joint range 0..270 deg, success needs > 10%,
i.e. > 27 deg on both valves), release, back off.

  python -m rdb.gen.scripted_rotate_valve --seeds 100000:100050 --dry
"""
from __future__ import annotations

import numpy as np

from rdb.gen.scripted_common import BimanualScript, Seg, cli


class RotateValveExpert(BimanualScript):
    TWIST = np.deg2rad(55)
    GRASP_DOWN = 0.012    # pads this far below the hub top
    CLOSE_HOLD = 6
    TWIST_SPEED = 0.6     # rad/s scale (W_AVG * this)

    def plan(self):
        import mujoco
        self.hub = []
        for v in self.env.valves:
            name = v.body.mjcf.full_identifier
            gs = [g for g in self.geoms_of(name)
                  if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("unnamed_geom_2")]
            self.hub.append((v, gs[0]))
        ys = [self.d.geom_xpos[g][1] for _, g in self.hub]
        self.mine = [int(np.argmax(ys)), int(np.argmin(ys))]   # LEFT arm (+y) -> valve with larger y
        self.stages = [["over", "approach", "descend", "close", "twist", "release", "retreat", "done"] for _ in range(2)]

    def hub_top(self, i: int) -> np.ndarray:
        g = self.hub[self.mine[i]][1]
        lo, hi = self.geom_aabb(g)
        c = self.d.geom_xpos[g].copy()
        c[2] = hi[2] - self.GRASP_DOWN
        return c

    def segment(self, i: int, st: str) -> Seg:
        p, y = self.pinch(i), self.yaw[i]
        c = self.hub_top(i)
        y0 = -self.TWIST / 2
        if st == "over":
            return Seg(p, [c[0], c[1], max(p[2], c[2] + 0.08)], y, y0, 0.0)
        if st == "approach":
            return Seg(p, c + [0, 0, 0.04], y, y0, 0.0, speed=0.7)
        if st == "descend":
            return Seg(p, c, y, y0, 0.0, min_steps=6, speed=0.6)
        if st == "close":
            return Seg(p, p, y, y, 1.0, min_steps=1, hold=self.CLOSE_HOLD)
        if st == "twist":
            s = Seg(p, p, y, y + self.TWIST, 1.0, min_steps=10)
            s.move = max(s.move, int(np.ceil(self.TWIST / (1.6 * self.TWIST_SPEED) * 20)))
            s.n = s.move
            return s
        if st == "release":
            return Seg(p, p, y, y, 0.0, min_steps=1, hold=4)
        if st == "retreat":
            return Seg(p, p + [0, 0, 0.06], y, y, 0.0, min_steps=6)
        return Seg(p, p, y, y, 0.0, min_steps=1, hold=1000)

    def advance(self, i: int, st: str):
        if st == "close" and not self.holding(i, self.hub[self.mine[i]][0].valve):
            self.grip[i] = 0.0
            return 1
        return None


if __name__ == "__main__":
    cli("RotateValve", {"*": RotateValveExpert}, "100000:100050")
