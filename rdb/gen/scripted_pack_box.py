"""Scripted PackBox expert (privileged box/flap state; training data only). See rdb.gen.scripted_common.

Each arm sweeps the flap on its side shut: closed fingertips start just behind the flap's outer face
R from the hinge, then follow a circular arc about the hinge axis that rotates the flap from its open
rest (leaning outward) past vertical down onto the box top (success: both hinge states within 0.1 of
closed, i.e. the joints' lower limits); gravity holds it shut. Left arm -> flap on the +y side.

  python -m rdb.gen.scripted_pack_box --seeds 100000:100050 --dry
"""
from __future__ import annotations

import numpy as np

from rdb.gen.scripted_common import ArcSeg, BimanualScript, Seg, cli


class PackBoxExpert(BimanualScript):
    R = 0.20              # push point distance from the hinge (leverage against the hinge damping)
    BEHIND = 0.03         # start this far behind the flap face (along the push direction)
    OVERSWEEP = 0.25      # sweep this far (rad) past the closed angle
    SWEEP_W = 0.3         # rad/s (hinge damping 10: torque grows with speed, the arms are weak)
    STAGGER = 0           # steps the second arm waits

    def plan(self):
        import mujoco
        pb = self.env.packing_box
        self.box = pb
        self.flaps = []
        for j, geom in zip(pb._joints, (pb.flap_1, pb.flap_2)):
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j.mjcf.full_identifier)
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, geom.mjcf.full_identifier)
            self.flaps.append((jid, gid))
        ys = [self.d.xanchor[jid][1] for jid, _ in self.flaps]
        self.mine = [int(np.argmax(ys)), int(np.argmin(ys))]            # LEFT arm (+y) -> +y flap
        self.stages = [["outside", "under", "start", "sweep", "settle", "retreat", "done"] for _ in range(2)]

    def flap(self, i: int):
        jid, gid = self.flaps[self.mine[i]]
        a = self.d.xanchor[jid].copy()
        ax = self.d.xaxis[jid].copy()
        q = self.d.qpos[self.m.jnt_qposadr[jid]]
        lo = self.m.jnt_range[jid][0]
        u = self.d.geom_xpos[gid] - a
        u -= ax * (u @ ax)
        u /= np.linalg.norm(u)
        return a, ax, q, lo, u

    def length(self, i: int) -> float:
        """Flap extent from the hinge along u (max over its bounding-box corners)."""
        a, ax, q, lo, u = self.flap(i)
        gid = self.flaps[self.mine[i]][1]
        h, cc = self.m.geom_aabb[gid][3:], self.m.geom_aabb[gid][:3]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + cc
        w = corners @ self.d.geom_xmat[gid].reshape(3, 3).T + self.d.geom_xpos[gid]
        return float(((w - a) @ u).max())

    def start_point(self, i: int):
        a, ax, q, lo, u = self.flap(i)
        v = np.cross(-ax, u)                       # motion direction of the flap point when q decreases
        return a + u * self.R - v * self.BEHIND, v

    def segment(self, i: int, st: str) -> Seg:
        p, y = self.pinch(i), self.yaw[i]
        if st in ("outside", "under"):             # around the flap's free edge, then below it
            a, ax, q, lo, u = self.flap(i)
            s0, v = self.start_point(i)
            beyond = a + u * (self.length(i) + 0.06)
            if st == "outside":
                return Seg(p, [beyond[0], beyond[1], max(p[2], beyond[2] + 0.05)], y, y, 1.0,
                           min_steps=8 + (self.STAGGER if i == 1 else 0))
            return Seg(p, beyond - v * 0.08, y, y, 1.0, min_steps=6)
        if st == "start":
            s, v = self.start_point(i)
            return Seg(p, s, y, y, 1.0, min_steps=6, speed=0.6)
        if st == "sweep":
            a, ax, q, lo, u = self.flap(i)
            return ArcSeg(p, a, -ax, (q - lo) + self.OVERSWEEP, y, 1.0, w=self.SWEEP_W)
        if st == "settle":
            return Seg(p, p, y, y, 1.0, min_steps=1, hold=8)
        if st == "retreat":
            return Seg(p, p + [0, 0, 0.08], y, y, 1.0, min_steps=6)
        return Seg(p, p, y, y, 1.0, min_steps=1, hold=1000)


if __name__ == "__main__":
    cli("PackBox", {"*": PackBoxExpert}, "100000:100050")
