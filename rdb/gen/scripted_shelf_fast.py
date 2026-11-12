"""Human-speed, closed-loop shelf teacher; privileged training-data use only.

Use the measured fast open-hand nudge and pitched pinch, then keep the closed
hand moving through the rise, insertion and placement. Open while withdrawing.
Contact failures trigger a measured retry; no long closed-and-still phase.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rdb.gen.scripted_common import BimanualScript, Seg
from rdb.gen.scripted_pack_box_h import thermal_guard
from rdb.gen.scripted_pick_book_fast import PickBookFastExpert


class ShelfFastExpert(PickBookFastExpert):
    PLACE = True
    CARRY_Z = 1.10
    SHELF_GAP = .005
    TARGET_EDGE = -.005
    HEIGHT_DZ = -.002
    NUDGE_PITCH = .45
    NUDGE_CLEARANCE = .010

    def plan(self):
        super().plan()
        seq = self.stages[self.arm]
        self.stages[self.arm] = seq[:seq.index('lift')+1] + [
            'raise', 'slide', 'place', 'release', 'back', 'done']
        self.missing = 0
        self.raise_checks = 0
        self.repush = False
        self.push_passes = 0
        # Orient above the work first; the measured pad frame differs from
        # the robot's reported wrist frame, so do not assume a rigid offset.
        self.k[self.arm] = 0
        import mujoco
        hand = f'hand_{self.sides[self.arm].value.lower()}/fingertip_pad_collision_'
        self.pad_geoms = [g for g in range(self.m.ngeom)
                          if hand in (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,g) or '')]
        assert len(self.pad_geoms) == 2
        wrist = self.env.robot.grippers[self.sides[self.arm]]._wrist_site.mjcf.full_identifier
        self.wrist_site = mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_SITE,wrist)
        assert self.wrist_site >= 0

    def pad_floor(self):
        return min(self.geom_aabb(g)[0][2] for g in self.pad_geoms)

    def until(self,i,stage):
        if i == self.arm and stage == 'push':
            return self.book_box()[0][0] <= self.edge()-.030
        if i == self.arm and stage == 'close':
            return False
        return super().until(i,stage)

    def segment(self, i, stage):
        if i != self.arm:
            return super().segment(i, stage)
        p, y, t = self.pinch(i), self.yaw[i], self.tilt[i]
        blo, bhi = self.geom_aabb(self.board)
        if stage == 'verifylift':
            return Seg(p, p+[-.004,0,.004],y,y,1.,t,t,min_steps=6)
        if stage in ('side_over','side_down'):
            seg = super().segment(i,stage)
            if stage == 'side_over':
                seg.p1[2] = self.book_box()[1][2]+.11
            else:
                lower = min(self.pad_geoms,key=lambda g:self.d.geom_xpos[g][2])
                offset = self.d.geom_xpos[lower]-self.pad_mid(i)
                lo, hi = self.book_box()
                bc = self.book.body.get_position()
                sign = 1 if i == 0 else -1
                desired = np.array([bc[0]+.02,hi[1]+.015 if sign>0 else lo[1]-.015,0.])
                seg.p1[:2] = (desired-offset)[:2]
                seg.p1[2] = self.pad_mid(i)[2]+self.top()+self.NUDGE_CLEARANCE-self.pad_floor()
            gy, _ = self.orientation(i)
            seg.t1 = Rotation.from_rotvec([0,0,gy]).apply([0,-self.NUDGE_PITCH,0])
            return seg
        if stage == 'push':
            sign = 1 if i == 0 else -1
            return Seg(p,p+[-.09,-sign*.085,self.top()+self.NUDGE_CLEARANCE-self.pad_floor()],
                       y,y,0.,t,t,min_steps=12,speed=.65)
        if stage == 'close':
            return Seg(p,p+[0,0,.002],y,y,1.,t,t,min_steps=11)
        if stage == 'lift':
            return Seg(p,p+[0,0,.16],y,y,1.,t,t,min_steps=12,speed=.8)
        if stage == 'raise':
            lo, hi = self.book_box()
            z = max(1.11,p[2]+bhi[2]+.025-lo[2])
            safe_x = blo[0]-.025-(hi[0]-p[0])
            return Seg(p,[min(p[0],safe_x),p[1],min(z,1.25)],
                       y,y,1.,t,Rotation.from_rotvec([0,0,y]).apply([0,-1.10,0]),
                       min_steps=8,speed=1.15)
        if stage == 'slide':
            lo, _ = self.book_box()
            z = p[2]+bhi[2]+.015-lo[2]
            return Seg(p,[blo[0]-self.SHELF_GAP,p[1],z],
                       y,y,1.,t,t,min_steps=8,speed=1.15)
        if stage == 'place':
            lo, _ = self.book_box()
            drop = max(lo[2]-bhi[2]-.003,0.)
            return Seg(p,p+[0,0,-drop],y,y,1.,t,t,min_steps=6,speed=.8)
        if stage == 'release':
            return Seg(p,p+[-.025,0,.006],y,y,0.,t,t,min_steps=6)
        if stage in ('back','done'):
            return Seg(p,p+[-.10,0,.025],y,y,0.,t,t,min_steps=8)
        return super().segment(i,stage)

    def advance(self,i,stage):
        if i != self.arm:
            return None
        if stage == 'push' and self.book_box()[0][0] > self.edge()-.024 and self.push_passes < 2:
            self.repush = True
            self.push_passes += 1
            self.retries[i] += 1
        if stage == 'clearup' and self.repush:
            self.repush = False
            return 0
        if stage == 'raise':
            _, bhi = self.geom_aabb(self.board)
            if self.book_box()[0][2] < bhi[2]+.012 and self.raise_checks < 3:
                self.raise_checks += 1
                return self.stages[i].index('raise')
        if stage in ('raise','slide','place') and not self.holding(i,self.book):
            self.retries[i] += 1
            self.direct = False
            self.grip[i] = 0.
            return 0
        if stage == 'done':
            self.retries[i] += 1
            self.direct = False
            return 0
        if stage in ('release','back'):
            return None
        return super().advance(i,stage)

    def act(self):
        if self.tick % 20 == 0:
            thermal_guard()
        self.tick += 1
        held = self.holding(self.arm,self.book)
        before = self.stage(self.arm)
        if before in ('lift','raise','slide','place'):
            self.missing = self.missing+1 if not held else 0
            if self.missing >= 3:
                self.retries[self.arm] += 1
                self.direct = False
                self.k[self.arm] = 0
                self.segs[self.arm] = None
                self.grip[self.arm] = 0.
                self.missing = 0
        else:
            self.missing = 0
        a = BimanualScript.act(self)
        lo, hi = self.book_box()
        self.trace.append(dict(step=self.tick,phase=before,holding=bool(held),
                               object=self.book.body.get_position().tolist(),
                               command_phase=self.stage(self.arm),arm=self.arm,grips=self.grip[:],
                               pad_floor=float(self.pad_floor()),
                               pad=self.d.geom_xpos[self.pad_geoms].mean(0).tolist(),
                               wrist=self.d.site_xpos[self.wrist_site].copy().tolist(),
                               wrist_quat=Rotation.from_matrix(self.d.site_xmat[self.wrist_site].reshape(3,3)).as_quat(scalar_first=True).tolist(),
                               book_box=[lo.tolist(),hi.tolist()],
                               on_shelf=bool(self.book.is_colliding(self.env.book_shelf.lower_shelf_body)
                                             or self.book.is_colliding(self.env.book_shelf.upper_shelf_body))))
        return a


if __name__ == '__main__':
    from rdb.gen.scripted_shelf_fast_runner import main
    main('StackSingleBookShelf',ShelfFastExpert,'106200:106260')
