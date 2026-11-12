"""Human-style, level-wrist LiftTray teacher; privileged training data only.

Both arms approach the end walls on a common clock, visibly close for a fixed
short interval, then draw the tray toward the robot and lift together. Failed
pinches/slips cause a controlled set-down and a fresh measured approach, never
an indefinite wait for an invisible contact predicate.
"""
from __future__ import annotations

import time

import numpy as np

from rdb.gen.scripted_common import BimanualScript, Seg
from rdb.gen.scripted_pack_box_h import thermal_guard


class LiftTrayHumanExpert(BimanualScript):
    GRASP_DOWN = .015
    PULL = .25
    LIFT = .22
    CLOSE_STEPS = 8
    QUIET = BimanualScript.QUIET

    def plan(self):
        import mujoco
        self.tray = self.env.breakfast_tray
        self.walls = [g for g in self.geoms_of(self.tray.body.mjcf.full_identifier)
                      if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or '')
                      .endswith(('unnamed_geom_1', 'unnamed_geom_3'))]
        assert len(self.walls) == 2, self.walls
        self.retries = [0, 0]
        self.trace = []
        self.tick = self.missing = 0
        self.level_bias = 0.
        self.rest_z = np.mean([self.wall(i)[0][2] for i in (0, 1)])
        self.carry_y = float(np.mean([p[1] for p in self.home]))
        self.phases = ['over', 'descend', 'close', 'pull', 'lift', 'lower', 'clear']
        self.stages = [self.phases[:], self.phases[:]]
        self.set_phase('over')

    def wall(self, i):
        g = (max if i == 0 else min)(self.walls, key=lambda w: self.d.geom_xpos[w][1])
        other = next(w for w in self.walls if w != g)
        c = self.d.geom_xpos[g].copy()
        axis = self.d.geom_xpos[other][:2] - c[:2]
        c[2] = self.geom_aabb(g)[1][2] - self.GRASP_DOWN
        return c, float(np.arctan2(axis[1], axis[0]))

    def set_phase(self, phase):
        self.phase, self.age, self.missing = phase, 0, 0
        self.k = [self.phases.index(phase)] * 2
        ps = self.cmd_pads()
        walls = [self.wall(i) for i in (0, 1)]
        plane = np.mean([c[2] for c, _ in walls])
        self.segs = []
        for i, p in enumerate(ps):
            c, axis = walls[i]
            goal, yaw, grip = p.copy(), self.yaw[i], 1.
            if phase in ('over', 'descend'):
                goal = c.copy()
                goal[2] = plane + (.10 if phase == 'over' else 0.)
                yaw = self.pick_yaw(i, axis, -1. if i == 0 else 1., yaw_max=2.0)
                grip = 0.
            elif phase == 'pull':
                # Human pulls also bring an off-centre tray between the shoulders.
                dy = np.clip(self.carry_y - self.tray.body.get_position()[1], -.18, .18)
                goal = p + [-self.PULL, dy, .055]
            elif phase == 'lift':
                goal = p + [0., 0., self.LIFT]
            elif phase == 'lower':
                goal[2] = self.rest_z
            elif phase == 'clear':
                goal[2] = max(p[2], plane) + .08
                grip = 0.
            if phase in ('pull', 'lift', 'lower', 'clear'):
                # Shared plane, even if DART or the load made the starting poses unequal.
                goal[2] = (np.mean([q[2] for q in ps]) +
                           dict(pull=.055, lift=self.LIFT, clear=.08).get(phase, 0.)) if phase != 'lower' else self.rest_z
            self.segs.append(Seg(p, goal, self.yaw[i], yaw, grip,
                                 self.tilt[i], np.zeros(3), min_steps=self.CLOSE_STEPS if phase == 'close' else 6,
                                 speed=.85 if phase in ('pull', 'lift') else .8))
        n = max(s.move for s in self.segs)
        for s in self.segs:
            s.move = s.n = n

    def retry(self, held):
        for i, ok in enumerate(held):
            if not ok:
                self.retries[i] += 1
        self.set_phase('lower')

    def act(self):
        if self.tick % 20 == 0:
            thermal_guard()
        # Yield this shared CPU between commands; simulated 20 Hz timing is unchanged.
        time.sleep(.02)
        self.tick += 1
        self.age += 1
        held = [self.holding(i, self.tray) for i in (0, 1)]
        wrists = [self.env.robot.grippers[s].wrist_position.copy() for s in self.sides]
        before = self.phase
        if self.phase in ('pull', 'lift'):
            self.missing = self.missing + 1 if not all(held) else 0
            if self.missing >= 3:
                self.retry(held)
        if all(s.done for s in self.segs):
            if self.phase == 'close' and not all(held):
                self.retry(held)
            else:
                self.set_phase(dict(over='descend', descend='close', close='pull',
                                    pull='lift', lift='lower', lower='clear', clear='over')[self.phase])
        a = np.zeros(14, np.float32)
        ee = self.cmd_pose()
        self._pads = self.cmd_pads()
        dz = float(wrists[0][2] - wrists[1][2])
        # Slow integral compensation balances unequal arm sag under the heavy tray.
        # Reset while open; retain it across the draw/lift transition.
        if self.phase in ('pull', 'lift', 'lower'):
            self.level_bias = float(np.clip(self.level_bias + .08 * dz, -.055, .055))
        else:
            self.level_bias *= .8
        correction = float(np.clip(.45 * dz + self.level_bias, -.065, .065))
        for i, seg in enumerate(self.segs):
            p, yaw, tilt = seg.next()
            p[2] += -correction if i == 0 else correction
            self._write(a, ee, i, p, yaw, tilt)
            self.grip[i] = seg.grip
        a[12:] = self.grip
        self.trace.append(dict(step=self.tick, phase=before, command_phase=self.phase,
                               phase_age=self.age, holding=held, grips=self.grip[:],
                               wrist_z=[float(w[2]) for w in wrists], wrist_dz=abs(dz)))
        return a


if __name__ == '__main__':
    from rdb.gen.scripted_h_runner import main
    main('LiftTray', LiftTrayHumanExpert, '106000:106060')
