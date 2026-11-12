"""Human-style, grasp-gated LiftPot teacher. Privileged state for data only.

Both open hands approach the handles together, close together and lift together
only while BOTH grippers hold. Misses are re-approached; slips lower the pot before
re-grasping. Run with MUJOCO_GL=egl.
"""
from __future__ import annotations

import numpy as np

from rdb.gen.scripted_common import Seg, wrap
from rdb.gen.scripted_lift_pot import LiftPotExpert
from rdb.gen.scripted_pack_box_h import thermal_guard


class LiftPotHumanExpert(LiftPotExpert):
    GRASP_DZ = -.012
    GRASP_DX = 0.
    LIFT_DX = -.08
    YAW_CAP = 1.40
    LIFT_SPEED = .70
    QUIET = LiftPotExpert.QUIET + ('verify',)

    def plan(self):
        super().plan()
        self.stages = [['over','descend','close','verify','lift','verifylift','lower','done'] for _ in range(2)]
        self.phase='over';self.tick=0;self.age=0;self.stable=0
        self.retries=[0,0];self.trace=[]
        self.initial_z=float(self.pot.body.get_position()[2])
        self.set_phase('over')

    def handle_yaw(self, i, bar):
        axis=wrap(bar+np.pi/2,np.pi)
        if i==0 and axis>0:axis-=np.pi
        if i==1 and axis<0:axis+=np.pi
        return float(np.clip(axis,-self.YAW_CAP,-.50) if i==0 else np.clip(axis,.50,self.YAW_CAP))

    def set_phase(self, phase):
        self.phase=phase;self.age=0;self.stable=0
        self.k=[self.stages[0].index(phase) if phase in self.stages[0] else 0]*2
        ps=self.cmd_pads(); goals=[]
        for i in (0,1):
            c,bar=self.handle(i);yaw=self.handle_yaw(i,bar);p=ps[i]
            c=c+[self.GRASP_DX,0,0]
            if phase=='over':goal=c+[0,0,.095];grip=0.
            elif phase=='descend':goal=c+[0,0,self.GRASP_DZ];grip=0.
            elif phase=='lift':goal=p+[self.LIFT_DX,0,.16];grip=1.
            elif phase=='lower':goal=p+[0,0,min(0.,self.initial_z-float(self.pot.body.get_position()[2]))];grip=1.
            else:goal=p;grip=1.;yaw=self.yaw[i]
            goals.append(Seg(p,goal,self.yaw[i],yaw,grip,min_steps=8 if phase=='descend' else 6,
                             speed=self.LIFT_SPEED if phase=='lift' else 1.))
        # Equal-length segments give the arms a common clock, even on offset layouts.
        n=max(s.move for s in goals)
        for s in goals:s.move=s.n=n
        self.segs=goals

    def act(self):
        if self.tick%20==0:thermal_guard()
        self.tick+=1;self.age+=1
        held=[self.holding(i,self.pot) for i in (0,1)]
        z=float(self.pot.body.get_position()[2])
        self.trace.append(dict(step=self.tick,phase=self.phase,holding=held,z=z))
        if self.phase=='lift' and not all(held):
            self.set_phase('verifylift')
        if self.phase=='verifylift':
            if all(held):self.set_phase('lift')
            elif self.age>=5:
                for i in (0,1):self.retries[i]+=int(not held[i])
                self.set_phase('lower')
        if self.phase in ('over','descend','lift','lower') and all(s.done for s in self.segs):
            self.set_phase(dict(over='descend',descend='close',lift='done',lower='over')[self.phase])
        if self.phase=='done' and self.age>=8:
            self.set_phase('lift' if all(held) else 'lower')
        if self.phase=='close' and self.age>=10:self.set_phase('verify')
        if self.phase=='verify':
            self.stable=self.stable+1 if all(held) else 0
            if self.stable>=1:self.set_phase('lift')
            elif self.age>=10:
                for i in (0,1):self.retries[i]+=int(not held[i])
                self.set_phase('over')
        a=np.zeros(14,np.float32);ee=self.cmd_pose();self._pads=self.cmd_pads()
        for i,s in enumerate(self.segs):
            p,y,t=s.next();self._write(a,ee,i,p,y,t);self.grip[i]=s.grip
        a[12:]=self.grip
        self.trace[-1].update(command_phase=self.phase,command_dz=[float(a[2]),float(a[8])])
        return a


if __name__=='__main__':
    from rdb.gen.scripted_h_runner import main
    main('LiftPot',LiftPotHumanExpert,'105300:105360')
