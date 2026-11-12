"""Observation-only human-style rod handover; no task simulator access.

rod_pose is centre xyz + world wxyz quaternion. The rod's long axis is local y
(public box half-sizes .02,.10,.02). Robot quantities use standard proprioception
and our command-history FK/IK. Grasp checks use closed-command gripper opening.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from rdb.hybrid.obs_script import ObsScript, Seg, wrap


class HandoverController(ObsScript):
    END = .070
    H = np.array([.58, 0., 1.08])
    PHASES = ('over','descend','close_giver','carry','insert','close_taker',
              'release','verify_giver','verify_taker','clear','done')

    def __init__(self, rod_pose, *, speed=.75, sync_approach=False,
                 grip_lo=.05, grip_hi=.9):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('speed must be finite and positive')
        if not 0 <= grip_lo < grip_hi <= 1:
            raise ValueError('Require 0 <= grip_lo < grip_hi <= 1')
        self.speed, self.sync_approach = float(speed), bool(sync_approach)
        self.update_pose(rod_pose)
        # Public CubeHandover scene robot base (not the book scene base).
        super().__init__([.1, 0., 0.], grip_lo=grip_lo, grip_hi=grip_hi)

    def update_pose(self, pose):
        p = np.asarray(pose, float)
        if p.shape != (7,) or not np.isfinite(p).all() or np.linalg.norm(p[3:]) < 1e-8:
            raise ValueError('rod_pose must be finite xyz + nonzero wxyz quaternion')
        self.rod_xyz = p[:3].copy()
        self.rod_rotation = R.from_quat(p[3:], scalar_first=True)

    def centre(self):
        return self.rod_xyz.copy()

    def axis(self):
        u = self.rod_rotation.apply([0., 1., 0.])
        return u if u[1]*self.sgn >= 0 else -u

    def plan(self):
        self.giver = 0 if self.centre()[1] >= 0 else 1
        self.taker = 1-self.giver
        self.sgn = 1. if self.giver == 0 else -1.
        self.phases = list(self.PHASES)
        self.stages = [self.phases[:], self.phases[:]]
        self.trace = []
        self.set_phase('over')

    def set_phase(self, phase):
        self.phase = phase
        self.age = self.stable = 0
        self.k = [self.phases.index(phase)]*2
        ps = self.cmd_pads()
        c, u = self.centre(), self.axis()
        gy = wrap(np.arctan2(u[1],u[0])+np.pi/2, np.pi)
        self.segs = []
        for i,p in enumerate(ps):
            giver = i == self.giver
            goal, yaw, grip = p.copy(), self.yaw[i], self.grip[i]
            if giver and phase in ('over','descend'):
                goal = c+u*self.END
                goal[2] += .085 if phase == 'over' else -.003
                yaw,grip = gy,0.
            elif phase == 'close_giver' and giver:
                grip = 1.
            elif phase == 'carry':
                goal = self.H+[0,self.sgn*(self.END if giver else -.18),0]
                yaw,grip = 0.,float(giver)
            elif phase == 'insert' and not giver:
                goal = c-u*self.END
                yaw,grip = gy,0.
            elif phase == 'close_taker' and not giver:
                grip = 1.
            elif phase == 'release' and giver:
                grip = 0.
            elif phase == 'clear' and not giver:
                goal = p+[0,-self.sgn*.065,.015]
                grip = 0.
            elif phase == 'over' and not giver:
                goal = self.H+[0,-self.sgn*.18,0]
                yaw,grip = 0.,0.
            self.segs.append(Seg(p,goal,self.yaw[i],yaw,grip,self.tilt[i],np.zeros(3),
                                 min_steps=6,speed=self.speed))
        if phase == 'carry' or self.sync_approach and phase == 'over':
            n = max(s.move for s in self.segs)
            for s in self.segs:
                s.move = s.n = n

    def step(self):
        self.age += 1
        held = [self.holding(i) for i in (0,1)]
        g,t = held[self.giver],held[self.taker]
        self.trace.append(dict(step=self.tick,phase=self.phase,holding=held,
                               centre=self.centre().tolist(),giver=self.giver))
        if self.phase in ('carry','insert','close_taker') and not g:
            self.resume = self.phase
            self.set_phase('verify_giver')
        if self.phase == 'verify_giver':
            if g:
                self.set_phase(self.resume)
            elif t:
                self.set_phase('release')
            elif self.age >= 6:
                self.retries[self.giver] += 1
                self.set_phase('over')
        if self.phase == 'close_giver':
            self.stable = self.stable+1 if g else 0
            if self.stable >= 2:
                self.set_phase('carry')
            elif self.age >= 16:
                self.retries[self.giver] += 1
                self.set_phase('over')
        elif self.phase == 'close_taker':
            self.stable = self.stable+1 if t else 0
            if self.stable >= 2:
                self.set_phase('release')
            elif self.age >= 16:
                self.retries[self.taker] += 1
                self.set_phase('clear')
        elif self.phase == 'release' and not t:
            self.set_phase('verify_taker')
        elif self.phase == 'verify_taker':
            if t:
                self.set_phase('release')
            elif self.age >= 5:
                self.retries[self.taker] += 1
                self.set_phase('clear' if g else 'over')
        elif all(s.done for s in self.segs):
            nxt = dict(over='descend',descend='close_giver',carry='insert',
                       insert='close_taker',clear='insert',release='done').get(self.phase)
            if nxt:
                self.set_phase(nxt)
        a = np.zeros(14,np.float32)
        ee = self.cmd_pose()
        self._pads = self.cmd_pads()
        for i,s in enumerate(self.segs):
            p,y,tilt = s.next()
            self._write(a,ee,i,p,y,tilt)
            self.grip[i] = s.grip
        a[12:] = self.grip
        self.trace[-1].update(command_phase=self.phase,grips=self.grip[:])
        return a

    def integrate(self, action):
        self.qctrl = self._robot.update(self.qctrl,action,self.MAX_JOINT_DELTA)

    def act(self, obs, rod_pose):
        self.update_pose(rod_pose)
        return super().act(obs)
