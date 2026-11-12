"""Human-style closed-loop rod handover. Privileged training-data teacher only.

Approach the nearer rod end, lift/turn toward the centre while the other hand
prepares on its own side, then pinch the free end. Release requires a measured
taker grasp. Misses and slips re-plan from the rod; all moving phases use DART.
"""
from __future__ import annotations

import numpy as np

from rdb.gen.scripted_common import BimanualScript, Seg, wrap
from rdb.gen.scripted_pack_box_h import thermal_guard


class HandoverHumanExpert(BimanualScript):
    END = .070
    H = np.array([.58, 0., 1.08])
    QUIET = BimanualScript.QUIET + ('verify',)

    def plan(self):
        self.rod = self.env.cube
        self.rg = self.geoms_of(self.rod.body.mjcf.full_identifier)[0]
        self.giver = 0 if self.centre()[1] >= 0 else 1
        self.taker = 1-self.giver
        self.sgn = 1. if self.giver == 0 else -1.
        self.retries = [0, 0]
        self.trace = []
        self.tick = 0
        self.phases = ['over','descend','close_giver','carry','insert','close_taker',
                       'release','verify_giver','verify_taker','clear','done']
        self.stages = [self.phases[:], self.phases[:]]
        self.set_phase('over')

    def centre(self):
        return self.d.geom_xpos[self.rg].copy()

    def axis(self):
        u = self.d.geom_xmat[self.rg].reshape(3,3)[:,1].copy()
        u /= np.linalg.norm(u)
        return u if u[1]*self.sgn >= 0 else -u

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
                goal[2] += .085 if phase=='over' else -.003
                yaw, grip = gy, 0.
            elif phase=='close_giver' and giver:
                grip = 1.
            elif phase=='carry':
                goal = self.H + [0, self.sgn*(self.END if giver else -.18),0]
                yaw,grip = 0.,float(giver)
            elif phase=='insert' and not giver:
                goal = c-u*self.END
                yaw,grip = gy,0.
            elif phase=='close_taker' and not giver:
                grip=1.
            elif phase=='release' and giver:
                grip=0.
            elif phase=='clear' and not giver:
                goal=p+[0,-self.sgn*.065,.015];grip=0.
            elif phase=='over' and not giver:
                goal=self.H+[0,-self.sgn*.18,0]
                yaw,grip=0.,0.
            self.segs.append(Seg(p,goal,self.yaw[i],yaw,grip,self.tilt[i],np.zeros(3),
                                 min_steps=6,speed=.75))
        # Common clock for the coordinated carry, distinct paths on each side.
        if phase=='carry':
            n=max(s.move for s in self.segs)
            for s in self.segs:s.move=s.n=n

    def act(self):
        if self.tick%20==0:thermal_guard()
        self.tick+=1;self.age+=1
        held=[self.holding(i,self.rod) for i in (0,1)]
        g,t=held[self.giver],held[self.taker]
        self.trace.append(dict(step=self.tick,phase=self.phase,holding=held,
                               centre=self.centre().tolist(),giver=self.giver))
        if self.phase in ('carry','insert','close_taker') and not g:
            self.resume=self.phase;self.set_phase('verify_giver')
        if self.phase=='verify_giver':
            if g:self.set_phase(self.resume)
            elif t:self.set_phase('release')
            elif self.age>=6:
                self.retries[self.giver]+=1;self.set_phase('over')
        if self.phase=='close_giver':
            self.stable=self.stable+1 if g else 0
            if self.stable>=2:self.set_phase('carry')
            elif self.age>=16:
                self.retries[self.giver]+=1;self.set_phase('over')
        elif self.phase=='close_taker':
            self.stable=self.stable+1 if t else 0
            if self.stable>=2:self.set_phase('release')
            elif self.age>=16:
                self.retries[self.taker]+=1;self.set_phase('clear')
        elif self.phase=='release' and not t:
            self.set_phase('verify_taker')
        elif self.phase=='verify_taker':
            if t:self.set_phase('release')
            elif self.age>=5:
                self.retries[self.taker]+=1
                self.set_phase('clear' if g else 'over')
        elif all(s.done for s in self.segs):
            nxt=dict(over='descend',descend='close_giver',carry='insert',
                     insert='close_taker',clear='insert',release='done').get(self.phase)
            if nxt:self.set_phase(nxt)
        a=np.zeros(14,np.float32);ee=self.cmd_pose();self._pads=self.cmd_pads()
        for i,s in enumerate(self.segs):
            p,y,tilt=s.next();self._write(a,ee,i,p,y,tilt);self.grip[i]=s.grip
        a[12:]=self.grip
        self.trace[-1].update(command_phase=self.phase,grips=self.grip[:])
        return a


if __name__=='__main__':
    from rdb.gen.scripted_h_runner import main
    main('CubeHandover',HandoverHumanExpert,'105800:105860')
