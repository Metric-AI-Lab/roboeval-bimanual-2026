"""Scripted LiftTray expert (privileged tray pose; training data only). See rdb.gen.scripted_common.

Each arm pinches the top of the tray's end wall on its side (fingers closing across the wall, i.e.
along the tray's long axis), both meet at a sync point and lift together, same displacement.

  python -m rdb.gen.scripted_lift_tray --seeds 100000:100050 --dry
"""
from __future__ import annotations

import numpy as np

from rdb.gen.scripted_common import BimanualScript, Seg, cli


class LiftTrayExpert(BimanualScript):
    GRASP_DOWN = 0.015    # pads this far below the wall top
    INSET = 0.0           # move the grasp point toward the tray centre (m)
    LIFT = 0.24           # commanded; the 34 kg tray makes the arms sag ~0.1 (success needs a real >= ~0.08 rise)
    CLOSE_HOLD = 8
    LIFT_SPEED = 0.5
    PULL = 0.28           # slide the tray this far toward the robot before/while lifting (humans: ~0.3)
    PULL_DZ = 0.03        # rise during the pull

    def plan(self):
        tray = self.env.breakfast_tray
        self.tray = tray
        name = tray.body.mjcf.full_identifier
        import mujoco
        walls = [g for g in self.geoms_of(name)
                 if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(("unnamed_geom_1", "unnamed_geom_3"))]
        assert len(walls) == 2, walls
        self.walls = walls
        self.stages = [["over", "approach", "descend", "close", "sync:grasped", "pull", "sync:pulled", "lift", "done"] for _ in range(2)]

    def wall(self, i: int):
        """(grasp point, long-axis angle) of the end wall on arm i's side (LEFT = larger y)."""
        g = max(self.walls, key=lambda w: self.d.geom_xpos[w][1]) if i == 0 else min(self.walls, key=lambda w: self.d.geom_xpos[w][1])
        other = [w for w in self.walls if w != g][0]
        c, o = self.d.geom_xpos[g].copy(), self.d.geom_xpos[other]
        axis = o[:2] - c[:2]
        lo, hi = self.geom_aabb(g)
        u = np.append(axis / np.linalg.norm(axis), 0)
        p = np.array([c[0], c[1], hi[2] - self.GRASP_DOWN]) + u * self.INSET
        return p, np.arctan2(axis[1], axis[0])

    def segment(self, i: int, st: str) -> Seg:
        p, y = self.pinch(i), self.yaw[i]
        c, ax = self.wall(i)
        gy = self.pick_yaw(i, ax, -1.0 if i == 0 else 1.0)
        if st == "over":
            return Seg(p, [c[0], c[1], max(p[2], c[2] + 0.12)], y, gy, 0.0)
        if st == "approach":
            return Seg(p, c + [0, 0, 0.05], y, gy, 0.0, speed=0.7)
        if st == "descend":
            return Seg(p, c, y, gy, 0.0, min_steps=8, speed=0.6)
        if st == "close":
            return Seg(p, p, y, y, 1.0, min_steps=1, hold=self.CLOSE_HOLD)
        if st == "pull":                                   # same displacement for both hands
            return Seg(p, p + [-self.PULL, 0, self.PULL_DZ], y, y, 1.0, min_steps=10, speed=self.LIFT_SPEED)
        if st == "lift":
            return Seg(p, p + [0, 0, self.LIFT], y, y, 1.0, min_steps=10, speed=self.LIFT_SPEED)
        return Seg(p, p, y, y, self.grip[i], min_steps=1, hold=1000)

    def advance(self, i: int, st: str):
        if st == "close" and not self.holding(i, self.tray):
            self.grip[i] = 0.0
            return 1
        return None


if __name__ == "__main__":
    cli("LiftTray", {"*": LiftTrayExpert}, "100000:100050")
