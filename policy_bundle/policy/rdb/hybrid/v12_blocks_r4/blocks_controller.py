"""Observation-only StackTwoBlocks controller: the grab-both teacher (HoldStackExpertV2) on legal inputs.

Input: the reset estimate of both blocks [(xyz, yaw), (xyz, yaw)] (from the pose regressor) and, every step, the
standard observation. Output: one 14-D EE-delta/gripper action. No environment, physics, contact or live object
state: before its grasp a block is where the reset estimate put it (nothing touches it), after the grasp it moves
rigidly with the commanded pinch point. Grasps are checked from the gripper opening (proprioception). The commanded
joint targets are tracked with the robot's own FK/IK (rdb.hybrid.pot_controller), as the action mode does.

Teacher reference: rdb/gen/scripted_stack_hold.py + scripted_stack_hold2.py (privileged, training data only).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .pot_controller import MAX_JOINT_DELTA, RESET_QCTRL, arm_fk_jac, min_jerk, solve_arm_ik, wrap

CUBE = .04
V_AVG, W_AVG, DT = .22, 1.6, 1 / 20
PAD_IN_EE = np.array([0., .001752, .103428])   # pinch point in the wrist frame (fingers open), calibrated at reset
NEAR, ASIDE, CLEAR_STEPS = .15, .12, 6
REST_Z = .97   # blocks spawn above the table and drop within a few steps; x, y and yaw do not change (measured)


class Segment:
    """Min-jerk move of one arm's pinch point and yaw (the teacher's Segment)."""

    def __init__(self, p0, p1, y0, y1, grip, min_steps=6, hold=0, speed=1.):
        self.p0, self.p1, self.y0, self.y1, self.grip = np.asarray(p0, float), np.asarray(p1, float), y0, y1, grip
        dur = max(np.linalg.norm(self.p1 - self.p0) / V_AVG, abs(y1 - y0) / W_AVG) / speed
        self.n = max(int(np.ceil(min_steps / speed)), int(np.ceil(dur / DT))) + hold
        self.move = self.n - hold
        self.k = 0

    def next(self):
        self.k += 1
        s = min_jerk(self.k / self.move)
        return self.p0 + (self.p1 - self.p0) * s, self.y0 + (self.y1 - self.y0) * s

    @property
    def done(self):
        return self.k >= self.n


class BlocksController:
    def __init__(self, blocks, grip_lo=.05, grip_hi=.9, speed=1., one_arm=False, steady_home=True, approach_dz=.055,
                 home_frac=1., retreat_dz=.06, steady_retreat_out=0., steady_retreat_x=0.,
                 steady_retreat_base_x_max=np.inf, short_clear=0., clear_mode='away', clear_dz=.03,
                 direct_approach=False, merge_carry=False, carry_clearance=.03,
                 swap_roles=False, steady_side_grasp=False, steady_side_angle=np.pi/2,
                 clear_lift_first=0., lift_min=.05, lift_clearance=.045, steady_speed=None,
                 steady_late_tilt=False, withdraw_discrete_stride=1, withdraw_discrete_far_only=False):
        if int(withdraw_discrete_stride) != withdraw_discrete_stride or not 1 <= withdraw_discrete_stride <= 4:
            raise ValueError("Withdrawal stride must be integer1–4")
        self.withdraw_discrete_stride = int(withdraw_discrete_stride)
        self.withdraw_discrete_far_only = bool(withdraw_discrete_far_only)
        self.withdraw_pause_steps = 0
        self._withdraw_tick = 0
        if (not np.isfinite([short_clear,clear_dz,carry_clearance]).all() or short_clear<0
                or clear_dz<0 or carry_clearance<.003 or clear_mode not in ('away','back','home')):
            raise ValueError('Invalid block path/clearance settings')
        self.short_clear, self.clear_mode, self.clear_dz = float(short_clear),clear_mode,float(clear_dz)
        self.direct_approach, self.merge_carry = bool(direct_approach),bool(merge_carry)
        self.carry_clearance, self.swap_roles = float(carry_clearance),bool(swap_roles)
        self.steady_side_grasp = bool(steady_side_grasp)
        if steady_speed is not None and (not np.isfinite(steady_speed) or steady_speed <= 0):
            raise ValueError('steady_speed must be positive')
        self.steady_speed = steady_speed
        self.steady_late_tilt = bool(steady_late_tilt)
        if (not np.isfinite([steady_side_angle,clear_lift_first,lift_min,lift_clearance]).all()
                or not 0<steady_side_angle<=np.pi/2 or clear_lift_first<0 or lift_min<0):
            raise ValueError('Invalid side grasp or lift parameters')
        self.steady_side_angle = float(steady_side_angle)
        self.clear_lift_first, self.lift_min, self.lift_clearance = clear_lift_first,lift_min,lift_clearance
        self.tilt = [0.,0.]
        self.approach_dz = float(approach_dz)   # hover height above a block before descending (path length)
        self.home_frac = float(home_frac)   # steady arm returns this fraction of the way home after releasing
        self.retreat_dz = float(retreat_dz)   # lift of the stacking arm after releasing
        self.steady_retreat_out = float(steady_retreat_out)
        self.steady_retreat_x = float(steady_retreat_x)
        self.steady_retreat_base_x_max = float(steady_retreat_base_x_max)
        if (not np.isfinite([self.steady_retreat_out, self.steady_retreat_x]).all()
                or np.isnan(self.steady_retreat_base_x_max) or self.steady_retreat_out < 0.):
            raise ValueError('steady retreat offsets must be finite and outward distance nonnegative')
        self.speed = float(speed)   # >1 shortens every move (efficiency: completion time)
        self.steady_home = bool(steady_home)   # False: the steady arm only lifts clear after releasing (path length)
        self.one_arm = bool(one_arm)   # the other arm stays home instead of steadying the base block (path length)
        self.blk0 = [(np.r_[np.asarray(p, float)[:2], REST_Z], float(y)) for p, y in blocks]
        self.grip_lo, self.grip_hi = grip_lo, grip_hi
        self.qctrl = RESET_QCTRL.astype(float).copy()
        self.R0 = [arm_fk_jac(self.qctrl[7 * i:7 * i + 7], i)[1] for i in (0, 1)]
        self.home = [self.pinch_cmd(i) for i in (0, 1)]
        self.yaw, self.grip = [0., 0.], [0., 0.]
        self.retries = [0, 0]
        self.openings = np.zeros(2)
        self.grasped = [None, None]   # per arm: (pinch, yaw) at the moment its block was grasped
        self._first = True
        self.plan()
        if self.withdraw_discrete_far_only and self.near:
            self.withdraw_discrete_stride = 1

    # ---- state -------------------------------------------------------------------------------------
    def cmd_pose(self, i):
        p, rot, _ = arm_fk_jac(self.qctrl[7 * i:7 * i + 7], i)
        return p, rot

    def pinch_cmd(self, i):
        p, rot = self.cmd_pose(i)
        return p + rot.apply(PAD_IN_EE)

    def holding(self, i):
        return bool(self.grip[i] >= .5 and self.grip_lo <= self.openings[i] < self.grip_hi)

    def blocks(self):
        """Live block estimates: reset pose until grasped, then rigid with the holding arm's commanded pinch."""
        out = [list(b) for b in self.blk0]
        for i in (0, 1):
            if self.grasped[i] is not None and self.own[i] is not None:
                b = self.own[i]
                p0, y0 = self.grasped[i]
                out[b] = [self.blk0[b][0] + (self.pinch_cmd(i) - p0), self.blk0[b][1] + (self.yaw[i] - y0)]
        return [tuple(b) for b in out]

    # ---- plan --------------------------------------------------------------------------------------
    def plan(self):
        blk, best = self.blk0, None
        for i in (0, 1):
            p = self.pinch_cmd(i)
            for a in (0, 1):
                c = np.linalg.norm(p[:2] - blk[a][0][:2])
                if best is None or c < best[0]:
                    best = (c, i, a)
        _, self.arm, self.pick = best
        if self.swap_roles:
            self.arm, self.pick = 1-self.arm,1-self.pick
        self.base = 1 - self.pick
        self.steady = 1 - self.arm
        self.own = [None, None]
        self.own[self.arm], self.own[self.steady] = self.pick, self.base
        self.stages = [None, None]
        self.stages[self.arm] = ['approach', 'descend', 'close', 'wait', 'lift', 'carry', 'lower', 'release',
                                 'retreat', 'done']
        self.stages[self.steady] = ['approach', 'descend', 'close', 'hold', 'release', 'retreat', 'done']
        self.near = np.linalg.norm((blk[0][0] - blk[1][0])[:2]) < NEAR
        if self.near:
            self.stages[self.arm] = ['approach', 'descend', 'close', 'lift', 'aside', 'wait', 'carry', 'lower',
                                     'release', 'retreat', 'done']
            self.stages[self.steady] = ['wait0'] + self.stages[self.steady]
        if self.one_arm:
            self.stages[self.arm] = [s for s in self.stages[self.arm] if s != 'wait']
            self.stages[self.steady] = ['done']
        if self.direct_approach and not self.near:
            self.stages = [[s for s in stages if s!='descend'] for stages in self.stages]
        if self.merge_carry and not self.near:
            self.stages[self.arm] = [s for s in self.stages[self.arm] if s!='lift']
        if self.short_clear and self.clear_lift_first:
            j=self.stages[self.steady].index('retreat')
            self.stages[self.steady].insert(j,'ungrasp_lift')
        self.side_angle = 0.
        if self.steady_side_grasp:
            pa,ya = blk[self.base]
            gy = self._grasp_yaw(ya,self.steady,pa,blk[self.pick][0])
            side = (R.from_rotvec([0,0,gy])*self.R0[self.steady]).apply([1,0,0])
            toward_home = self.home[self.steady]-pa
            self.side_angle = -self.steady_side_angle if side@toward_home>=0 else self.steady_side_angle
            if self.steady_late_tilt:
                j = self.stages[self.steady].index('retreat')
                self.stages[self.steady].insert(j, 'pivot')
        self.k = [0, 0]
        self.segs = [None, None]
        self.both_held = False

    def st(self, i):
        return self.stages[i][self.k[i]]

    def _grasp_yaw(self, block_yaw, i, pick, other):
        y = self.yaw[i]
        c = y + wrap(block_yaw - y, np.pi / 2)
        if np.linalg.norm((other - pick)[:2]) < 0.12:
            d = (other - pick)[:2] / np.linalg.norm((other - pick)[:2])
            cands = [x for x in (c, c + np.pi / 2, c - np.pi / 2) if abs(x - y) <= np.pi * 0.75] or [c]
            c = min(cands, key=lambda x: (round(abs(np.cos(x) * d[0] + np.sin(x) * d[1]), 2), abs(x - y)))
        return c

    def _seg(self, i):
        self._planning_arm = i
        st, blk, p, y = self.st(i), self.blocks(), self.pinch_cmd(i), self.yaw[i]
        pa, ya = blk[self.own[i]]
        po = blk[self.own[1 - i]][0]
        gy = self._grasp_yaw(ya, i, pa, po)
        if st == 'approach':
            dz = .006 if self.direct_approach and not self.near else self.approach_dz
            return self._move(p, pa + [0, 0, dz], y, gy, 0.0)
        if st == 'descend':
            return self._move(p, pa + [0, 0, 0.006], y, gy, 0.0, min_steps=6)
        if st == 'close':
            return self._move(p, p, y, y, 1.0, min_steps=1, hold=6)
        if st == 'aside':
            sgn = 1.0 if i == 0 else -1.0
            return self._move(p, p + [-0.03, sgn * ASIDE, 0.02], y, y, 1.0, min_steps=6)
        if st in ('wait', 'hold', 'wait0'):
            return self._move(p, p, y, y, 1.0, min_steps=1, hold=1)
        if st == 'release':
            return self._move(p, p, y, y, 0.0, min_steps=1, hold=4)
        if st == 'ungrasp_lift':
            return self._move(p,p+[0,0,self.clear_lift_first],y,y,0.,min_steps=4)
        if st == 'pivot':
            n = int(np.ceil(abs(self.side_angle) / W_AVG / (self.steady_speed or self.speed) / DT))
            return self._move(p,p,y,y,0.,min_steps=max(6,n*self.speed))
        if st == 'retreat' and i == self.steady and self.short_clear:
            direction = self.blk0[self.base][0]-self.blk0[self.pick][0]
            direction[2] = 0.
            direction = direction/max(np.linalg.norm(direction),1e-9)
            if self.clear_mode == 'back':direction=np.array([-1.,0.,0.])
            if self.clear_mode == 'home':
                direction = self.home[i]-p
                direction[2] = 0.
                direction /= max(np.linalg.norm(direction),1e-9)
            return self._move(p,p+direction*self.short_clear+[0,0,self.clear_dz],y,y,0.,min_steps=8)
        if (st == 'retreat' and i == self.steady and self.steady_retreat_out > 0.
                and self.blk0[self.base][0][0] < self.steady_retreat_base_x_max):
            sign = 1. if i == 0 else -1.
            return self._move(p, p + [self.steady_retreat_x, sign * self.steady_retreat_out, self.retreat_dz],
                              y, y, 0.0, min_steps=8)
        if st == 'retreat' and i == self.steady and self.steady_home:
            f = self.home_frac
            return self._move(p, p + f * (self.home[i] - p) + [0, 0, (1 - f) * .06], y, f * 0.0 + (1 - f) * y, 0.0,
                              min_steps=8)
        if st == 'retreat':
            return self._move(p, p + [0, 0, self.retreat_dz], y, y, 0.0, min_steps=6)
        base = blk[self.base]
        if st == 'lift':
            return self._move(p, p + [0, 0, max(self.lift_min, base[0][2] + CUBE + self.lift_clearance - p[2])], y, y, 1.0, min_steps=5)
        if st == 'carry':
            return self._move(p, p + (base[0] + [0, 0, CUBE + self.carry_clearance] - pa), y, y + wrap(base[1] - ya, np.pi / 2), 1.0)
        if st == 'lower':
            return self._move(p, p + (base[0] + [0, 0, CUBE + 0.003] - pa), y, y, 1.0, min_steps=5)
        return self._move(p, p, y, y, 0.0, min_steps=1, hold=1000)

    def _move(self, *a, **kw):
        speed = self.speed
        if self.steady_speed is not None and self._planning_arm == self.steady:
            speed = self.steady_speed
        seg = Segment(*a, speed=speed, **kw)
        if self.steady_side_grasp and self._planning_arm == self.steady:
            seg.tilt0 = self.tilt[self.steady]
            stage = self.st(self.steady)
            rotate = stage != 'wait0' and (not self.steady_late_tilt or stage in ('pivot','retreat','done'))
            seg.tilt1 = self.side_angle if rotate else seg.tilt0
        return seg

    def _regrasp(self, i):
        self.retries[i] += 1
        self.k[i] = 0
        self.grip[i] = 0.0
        self.grasped[i] = None

    def _advance(self, i):
        st = self.st(i)
        if st == 'close':
            if not self.holding(i):
                return self._regrasp(i)
            self.grasped[i] = (self.pinch_cmd(i), self.yaw[i])
        # HoldStackExpertV2: wait/hold only on visible events
        if st == 'wait':
            j = self.steady
            clear = self.st(j) == 'done' or (self.st(j) == 'retreat' and self.segs[j] is not None
                                             and self.segs[j].k >= CLEAR_STEPS)
            if self.short_clear and self.st(j)=='retreat':
                distance=np.linalg.norm((self.pinch_cmd(j)-self.blk0[self.base][0])[:2])
                clear = distance >= min(.085,self.short_clear*.85)
            if not clear:
                if not self.holding(i):
                    self._regrasp(i)
                return
            self.k[i] += 1
            return
        if st == 'hold':
            a = self.arm
            if self.k[a] <= self.stages[a].index('close'):
                if not self.holding(i):
                    self._regrasp(i)
                return
            self.k[i] += 1
            return
        if st == 'wait0':
            if self.k[self.arm] < self.stages[self.arm].index('wait'):
                return
        if st == 'lower' and not self.holding(i):
            self.k[i] = self.stages[i].index('release')
            return
        if st in ('lift', 'aside', 'carry') and not self.holding(i):
            return self._regrasp(i)
        if st == 'release':
            self.grasped[i] = None
        self.k[i] = min(self.k[i] + 1, len(self.stages[i]) - 1)

    def _integrate_action(self, a):
        for i in (0, 1):
            j = 6 * i
            q0 = self.qctrl[7 * i:7 * i + 7]
            p, rot, _ = arm_fk_jac(q0, i)
            q1 = solve_arm_ik(q0, i, p + a[j:j + 3], R.from_rotvec(a[j + 3:j + 6]) * rot)
            self.qctrl[7 * i:7 * i + 7] += np.clip(q1 - q0, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)

    def act(self, obs):
        self.openings = np.asarray(obs["proprioception_grippers"], float).copy()
        if self._first:
            self._first = False
            return np.zeros(14, np.float32)
        if self.holding(self.arm) and self.holding(self.steady):
            self.both_held = True
        a = np.zeros(14, np.float32)
        for i in (0, 1):
            if self.segs[i] is not None and self.segs[i].done:
                self._advance(i)
                self.segs[i] = None
            if self.segs[i] is None:
                self.segs[i] = self._seg(i)
            pos, yaw = self.segs[i].next()
            self.yaw[i] = yaw
            ee_p, ee_r = self.cmd_pose(i)
            rot_t = R.from_rotvec([0, 0, yaw]) * self.R0[i]
            if hasattr(self.segs[i],'tilt0'):
                seg=self.segs[i]
                self.tilt[i]=seg.tilt0+(seg.tilt1-seg.tilt0)*min_jerk(seg.k/seg.move)
                rot_t=rot_t*R.from_rotvec([0,self.tilt[i],0])
            a[6 * i:6 * i + 3] = pos - rot_t.apply(PAD_IN_EE) - ee_p
            a[6 * i + 3:6 * i + 6] = (rot_t * ee_r.inv()).as_rotvec()
            self.grip[i] = self.segs[i].grip
        a[12:] = self.grip
        if self.withdraw_discrete_stride > 1:
            j = self.steady
            if self.st(j) == 'retreat' and self.short_clear and self.grip[j] == 0.:
                distance = np.linalg.norm((self.pinch_cmd(j)-self.blk0[self.base][0])[:2])
                if distance >= min(.085,self.short_clear*.85):
                    self._withdraw_tick += 1
                    if self._withdraw_tick % self.withdraw_discrete_stride:
                        seg = self.segs[j]
                        seg.k -= 1
                        t = min_jerk(seg.k/seg.move)
                        self.yaw[j] = seg.y0+(seg.y1-seg.y0)*t
                        if hasattr(seg,'tilt0'):
                            self.tilt[j] = seg.tilt0+(seg.tilt1-seg.tilt0)*t
                        a[6*j:6*j+6] = 0.
                        self.withdraw_pause_steps += 1
        self._integrate_action(a)
        return a
