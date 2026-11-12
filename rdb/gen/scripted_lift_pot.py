"""Scripted LiftPot expert (privileged pot pose; training data only). See rdb.gen.scripted_common.

Each arm takes the loop handle on its side (the bar between the handle's two end spheres), fingers
closing across the bar; both arms meet at a sync point, lift together, and hold the pot level.

  python -m rdb.gen.scripted_lift_pot --seeds 100000:100050 --dry
  python -m rdb.gen.scripted_lift_pot --seeds 100000:101500
"""
from __future__ import annotations

import numpy as np

from rdb.gen.scripted_common import BimanualScript, Seg, cli, wrap

LIFT = 0.16          # success needs +0.10
YAW_MAX = np.deg2rad(95)
PULL_X = 0.62        # lift toward this pot x


class LiftPotExpert(BimanualScript):
    GRASP_DZ = -0.012    # grasp height relative to the bar centre (search 2026-10-17: best of 0/-0.006/-0.012)
    CLOSE_HOLD = 12      # steps the gripper closes before the lift
    LIFT_SPEED = 0.6     # fraction of the default move speed
    def plan(self):
        pot = self.env.kitchenpot
        name = pot.body.mjcf.full_identifier
        self.pot = pot
        self.spheres = [g for g in self.geoms_of(name) if self.m.geom_type[g] == 2 and self.m.geom_contype[g]]
        assert len(self.spheres) == 4, self.spheres
        self.stages = [["over", "approach", "descend", "close", "sync:grasped", "lift", "done"] for _ in range(2)]

    def handle(self, i: int):
        """(centre, bar yaw) of the handle on arm i's side (LEFT = +y)."""
        x = self.d.geom_xpos[self.spheres]
        order = np.argsort(x[:, 1])                        # two lowest y = right handle
        pair = x[order[2:]] if i == 0 else x[order[:2]]
        c = pair.mean(0)
        v = pair[1] - pair[0]
        return c, np.arctan2(v[1], v[0])

    def segment(self, i: int, st: str) -> Seg:
        p, y = self.pinch(i), self.yaw[i]
        c, bar = self.handle(i)
        gy = self.handle_yaw(i, bar)
        if st == "over":                                   # clear the rim (walls reach ~handle + 0.07)
            return Seg(p, [c[0], c[1], max(p[2], c[2] + 0.16)], y, gy, 0.0)
        if st == "approach":
            return Seg(p, c + [0, 0, 0.05], y, gy, 0.0, speed=0.7)
        if st == "descend":
            return Seg(p, c + [0, 0, self.GRASP_DZ], y, gy, 0.0, min_steps=8, speed=0.6)
        if st == "close":
            return Seg(p, p, y, y, 1.0, min_steps=1, hold=self.CLOSE_HOLD)
        if st == "lift":                                   # same displacement for both hands: pot stays level
            return Seg(p, p + self.lift_vec(), y, y, 1.0, min_steps=10, speed=self.LIFT_SPEED)
        return Seg(p, p, y, y, self.grip[i], min_steps=1, hold=1000)

    def lift_vec(self) -> np.ndarray:
        """Up, and toward the robot / centre line when the pot is far: lifting straight up from a
        far-out handle needs a fully stretched arm, which the IK refuses."""
        c = self.pot.body.get_position()
        return np.array([np.clip(PULL_X - c[0], -0.2, 0.0), np.clip(-c[1], -0.1, 0.1) * 0.5, LIFT])

    def handle_yaw(self, i: int, bar: float) -> float:
        """Fingers square across the bar. Of the two equivalent yaws (perp, perp -+ pi) take the reachable
        one (|yaw| <= YAW_MAX), preferring the mirrored turn the human demos use (left negative, right
        positive); if neither fits, the closer one clipped."""
        perp = wrap(bar + np.pi / 2, np.pi)                  # in (-90, 90]
        cands = [perp, perp - np.pi if perp > 0 else perp + np.pi]
        pref = -1.0 if i == 0 else 1.0
        ok = [c for c in cands if abs(c) <= YAW_MAX]
        if ok:
            return float(max(ok, key=lambda c: (np.sign(c) == pref, -abs(c))))
        return float(np.clip(min(cands, key=abs), -YAW_MAX, YAW_MAX))

    def advance(self, i: int, st: str):
        if st == "close" and not self.holding(i, self.pot):
            self.grip[i] = 0.0
            return 1                                       # missed: re-approach
        return None


if __name__ == "__main__":
    cli("LiftPot", {"*": LiftPotExpert}, "100000:100050")
