"""Observation-only pick/shelf book controllers, with live vision poses.

v12 Pick snapshot: byte copy of the v10/v11 submission bundle's
policy/rdb/hybrid/book_controller.py plus default-off lift_idle_yaw_* options
(idle wrist turns about the vertical while the active arm lifts the grasped
book, its joint-step norm matched to a fraction of the active arm's). With the
options off the actions are identical to the bundle controller.

act(obs, book_pose) consumes standard proprioception and a *predicted* world
book pose [x,y,z,qw,qx,qy,qz]. It has no environment/physics/contact access.
Static geometry below comes from the public book/shelf/robot assets, not seed
lookups. Commanded joints are tracked using the robot's own FK/IK model.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from rdb.hybrid.pot_controller import PAD_IN_EE
from rdb.hybrid.book_kinematics import RobotKinematics

# Pick/shelf use 25 physics steps at 2 ms. Other tasks can use a different dt.
MAX_JOINT_DELTA = 2.175 * .05


class Seg:
    """Pure min-jerk grasp-point/yaw/tilt segment; no simulator dependency."""

    def __init__(self, p0, p1, y0, y1, grip, t0=None, t1=None, min_steps=6, hold=0, speed=1.):
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.y0, self.y1, self.grip = y0, y1, grip
        self.t0 = np.zeros(3) if t0 is None else np.asarray(t0, float)
        self.t1 = self.t0 if t1 is None else np.asarray(t1, float)
        dur = max(np.linalg.norm(self.p1-self.p0)/(.22*speed), abs(y1-y0)/1.6,
                  np.linalg.norm(self.t1-self.t0)/1.6)
        self.move = max(min_steps, int(np.ceil(dur/.05)))
        self.n, self.k = self.move+hold, 0

    def next(self):
        self.k += 1
        u = min(max(self.k/self.move, 0.), 1.)
        s = u**3*(10-15*u+6*u*u)
        return self.p0+(self.p1-self.p0)*s, self.y0+(self.y1-self.y0)*s, self.t0+(self.t1-self.t0)*s

    @property
    def done(self):
        return self.k >= self.n

# place_books.yaml robot position; LiftPot's base is different.
BOOK_PELVIS_POS = np.array([-.15, 0., -.1])
# Bookshelf reset fixture, derived from the public asset/preset. A caller can
# replace these with a camera-derived reset estimate through counter_bounds.
COUNTER_BOUNDS = np.array([
    [.3458550035953516, -.464118987321855, .6728999975323677],
    [.7295080035924917, .5952209830284133, .6900779965519905],
])
# Compiled mesh-local AABB corners transformed to book-body coordinates. This
# matches teacher geom_aabb (a principal-axis bounding box, not a body AABB).
BOOK_CORNERS = np.array([
    [.04751408085889091, -.07748469261937146, .0014217536112781926],
    [.04707823194172883, .08679249063444894, .00012634096344609456],
    [-.04665429749828523, -.07773571793764279, .0012715725292497763],
    [-.04709014641544731, .08654146531617761, -.000023840118582321712],
    [.04755680984442709, -.07769310494137281, -.025022339437508093],
    [.047120960927265004, .0865840783124476, -.026317752085340193],
    [-.04661156851274905, -.07794413025964414, -.025172520519536508],
    [-.047047417429911136, .08633305299417628, -.026467933167368608],
])


class PickBookController:
    """Fast teacher phase logic on legal inputs; first action is an open no-op.

    book_pose must be updated from cameras on EVERY call, including after a
    push or missed grasp. Quaternions are world-frame wxyz. This component does
    not load a regressor or integrate itself into a submission.
    """

    PITCH, YAW = .90, .80
    TARGET_EDGE, HEIGHT_DZ, PUSH_SPEED = -.005, -.002, .35
    SPEEDUP = dict(side_over=1.8, side_down=2., push=1.5, clear=1.35,
                   clearup=1.75, over=1.8, front=1.25, insert=1.5)
    MIN_FRAMES = dict(side_over=8, side_down=9, push=10, clear=6,
                      clearup=4, over=8, front=10, insert=7)
    SEQUENCE = ('side_over', 'side_down', 'push', 'clear', 'clearup',
                'over', 'front', 'insert', 'close', 'verifylift', 'lift', 'done')

    def __init__(self, initial_book_pose, counter_bounds=None, pelvis_pos=None,
                 idle_wrist_z=None, idle_wrist_speed=.15, lift_dx=-.025,
                 direct_margin=.05, direct_edge_gap=None,
                 push_stop_left=.014, push_stop_right=.018,
                 idle_after_grasp=False, short_push_right=None,
                 short_push_target_y_max=.025, late_lift_after=0, late_lift_frames=6,
                 late_lift_min_book_move=0., front_clearance_x=0.,
                 front_clearance_book_x_max=.375, lift_idle_yaw_ratio=0.,
                 lift_idle_yaw_cap=1.2, lift_idle_yaw_sign=1., lift_idle_yaw_step=.1,
                 lift_idle_phases=('lift',), lift_idle_first_only=False,
                 lift_idle_delay=0, lift_idle_book_dz=None):
        if (not np.isfinite([lift_idle_yaw_ratio, lift_idle_yaw_cap, lift_idle_yaw_step]).all()
                or not 0 <= lift_idle_yaw_ratio <= 1.5 or not 0 < lift_idle_yaw_cap <= 2.
                or not 0 < lift_idle_yaw_step <= MAX_JOINT_DELTA or lift_idle_yaw_sign not in (-1, 1)
                or not set(lift_idle_phases) <= set(self.SEQUENCE)):
            raise ValueError('lift_idle_yaw options out of range')
        self.lift_idle_yaw_ratio = float(lift_idle_yaw_ratio)
        self.lift_idle_yaw_cap, self.lift_idle_yaw_step = float(lift_idle_yaw_cap), float(lift_idle_yaw_step)
        self.lift_idle_yaw_sign, self.lift_idle_phases = float(lift_idle_yaw_sign), tuple(lift_idle_phases)
        self._idle_yaw, self._idle_ref = 0., None
        self.lift_idle_first_only = bool(lift_idle_first_only)
        if int(lift_idle_delay) != lift_idle_delay or not 0 <= lift_idle_delay <= 20:
            raise ValueError('lift_idle_delay must be an integer in [0, 20]')
        self.lift_idle_delay, self._lift_ticks = int(lift_idle_delay), 0
        if lift_idle_book_dz is not None and (not np.isfinite(lift_idle_book_dz)
                                              or not 0 <= lift_idle_book_dz <= .1):
            raise ValueError('lift_idle_book_dz must be in [0, .1] metres')
        self.lift_idle_book_dz, self._lift_book_z = lift_idle_book_dz, None
        self.counter_bounds = np.array(COUNTER_BOUNDS if counter_bounds is None else counter_bounds, float)
        self.pelvis_pos = np.array(BOOK_PELVIS_POS if pelvis_pos is None else pelvis_pos, float)
        if idle_wrist_z is not None and not np.isfinite(idle_wrist_z):
            raise ValueError('idle_wrist_z must be finite')
        if (int(late_lift_after) != late_lift_after or int(late_lift_frames) != late_lift_frames
                or late_lift_after < 0 or late_lift_frames < 4
                or not np.isfinite(late_lift_min_book_move) or late_lift_min_book_move < 0):
            raise ValueError('late lift requires nonnegative trigger and at least four frames')
        self.late_lift_after = int(late_lift_after)
        self.late_lift_frames = int(late_lift_frames)
        self.late_lift_min_book_move = float(late_lift_min_book_move)
        if not np.isfinite(front_clearance_x) or not 0 <= front_clearance_x <= .05:
            raise ValueError('front_clearance_x must be in [0, .05] metres')
        self.front_clearance_x = float(front_clearance_x)
        if not np.isfinite(front_clearance_book_x_max):
            raise ValueError('front_clearance_book_x_max must be finite')
        self.front_clearance_book_x_max = float(front_clearance_book_x_max)
        self._close_book_xy = None
        if (not np.isfinite(idle_wrist_speed) or idle_wrist_speed <= 0 or
                not np.isfinite(lift_dx) or not np.isfinite(direct_margin) or
                (direct_edge_gap is not None and not np.isfinite(direct_edge_gap)) or
                not np.isfinite(push_stop_left) or not np.isfinite(push_stop_right) or
                min(push_stop_left, push_stop_right) <= 0 or
                (short_push_right is not None and
                 (not np.isfinite(short_push_right) or short_push_right <= 0)) or
                not np.isfinite(short_push_target_y_max)):
            raise ValueError('idle_wrist_speed must be positive; lift_dx and direct_margin finite')
        self.idle_wrist_z = idle_wrist_z
        self.idle_wrist_speed = float(idle_wrist_speed)
        self._idle_wrist_goal = None
        self.idle_after_grasp = bool(idle_after_grasp)
        self._idle_wrist_started = False
        self.lift_dx = float(lift_dx)
        self.direct_margin = float(direct_margin)
        self.direct_edge_gap = None if direct_edge_gap is None else float(direct_edge_gap)
        self.push_stop_left = float(push_stop_left)
        self.push_stop_right = float(push_stop_right)
        self.short_push_right = None if short_push_right is None else float(short_push_right)
        self.short_push_target_y_max = float(short_push_target_y_max)
        if self.counter_bounds.shape != (2, 3) or not np.isfinite(self.counter_bounds).all() or np.any(self.counter_bounds[0] >= self.counter_bounds[1]):
            raise ValueError('counter_bounds must be finite [lo,hi] world xyz bounds')
        if self.pelvis_pos.shape != (3,) or not np.isfinite(self.pelvis_pos).all():
            raise ValueError('pelvis_pos must be finite xyz')
        self._robot = RobotKinematics(self.pelvis_pos)
        self.qctrl = self._robot.initial_qctrl.copy()
        self.grip = [0., 0.]
        self.yaw = [0., 0.]
        self.tilt = [np.zeros(3), np.zeros(3)]
        self.retries = [0, 0]
        self.k = [0, 0]
        self.segs = [None, None]
        self._pad_offsets = [PAD_IN_EE.copy(), PAD_IN_EE.copy()]
        self._openings = np.zeros(2)
        ee = self.cmd_pose()
        self.R0 = [R.from_rotvec(ee[6*i+3:6*i+6]) for i in (0, 1)]
        self.tick = 0
        self._first, self._planned = True, False
        self.update_pose(initial_book_pose)
        self.arm = 0 if self.book_xyz[1] >= 0 else 1
        self.stages = [['done'], ['done']]
        self.stages[self.arm] = list(self.SEQUENCE)

    def update_pose(self, pose):
        pose = np.asarray(pose, float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
            raise ValueError('book pose must be finite xyz + nonzero wxyz quaternion')
        self.book_xyz = pose[:3].copy()
        self.book_rotation = R.from_quat(pose[3:], scalar_first=True)
        self.world_corners = self.book_rotation.apply(BOOK_CORNERS) + self.book_xyz
        self._book_bounds = (self.world_corners.min(0), self.world_corners.max(0))

    def cmd_pose(self):
        return self._robot.command_pose(self.qctrl)

    def cmd_pads(self):
        out = []
        for i, (p, rot) in enumerate(self._robot.poses(self.qctrl, world=True)):
            out.append(p + rot @ self._pad_offsets[i])
        return out

    def pinch(self, i):
        return self._pads[i].copy()

    def edge(self):
        return float(self.counter_bounds[0, 0])

    def book_box(self):
        return self._book_bounds

    def holding(self, i):
        return bool(self.grip[i] >= .5 and .05 <= self._openings[i] < .9)

    def stage(self, i):
        return self.stages[i][self.k[i]]

    @property
    def phase(self):
        return self.stage(self.arm)

    def target(self):
        lo, hi = self.book_box()
        x = min(self.edge() + self.TARGET_EDGE, self.book_xyz[0] - .015)
        ys = []
        for p in self.world_corners:
            for q in self.world_corners:
                if (p[0]-x)*(q[0]-x) <= 0 and abs(p[0]-q[0]) > 1e-8:
                    ys.append(float(p[1] + (x-p[0])/(q[0]-p[0])*(q[1]-p[1])))
        y = (min(ys)+max(ys))/2 if ys else self.book_xyz[1]
        return np.array([x, y, (lo[2]+hi[2])/2 + self.HEIGHT_DZ])

    def hold_seg(self, i, grip, steps):
        p = self.pinch(i)
        return Seg(p, p, self.yaw[i], self.yaw[i], grip, self.tilt[i], min_steps=1, hold=steps-1)

    def segment(self, i, stage):
        p, y, tilt = self.pinch(i), self.yaw[i], self.tilt[i]
        if i != self.arm:
            return self.hold_seg(i, 0., 1000)
        gy = -self.YAW if i == 0 else self.YAW
        gt = R.from_rotvec([0, 0, gy]).apply([0, -self.PITCH, 0])
        c = self.target()
        front_clearance = (self.front_clearance_x if
                           self.book_xyz[0] <= self.front_clearance_book_x_max else 0.)
        sign = 1 if i == 0 else -1
        lo, hi = self.book_box()
        bc = self.book_xyz
        side = np.array([bc[0]+.02, hi[1]+.035 if sign > 0 else lo[1]-.035, bc[2]-.003])
        if stage == 'side_over':
            seg = Seg(p, side+[0,0,.09], y, gy, 0., tilt, np.zeros(3))
        elif stage == 'side_down':
            seg = Seg(p, side, y, gy, 0., tilt, np.zeros(3), min_steps=10, speed=1.1 if self.direct else .7)
        elif stage == 'push':
            seg = Seg(p, p+[-.085,-sign*.085,0], y, y, 0., tilt, tilt, min_steps=14, speed=self.PUSH_SPEED)
        elif stage == 'clear':
            seg = Seg(p, p+[.03,sign*.055,.005], y, y, 0., tilt, tilt, min_steps=8, speed=.8)
        elif stage == 'clearup':
            seg = Seg(p, p+[0,0,.075], y, y, 0., tilt, tilt, min_steps=6)
        elif stage == 'over':
            seg = Seg(p, c+[-.06-front_clearance,0,.10], y, gy, 0., tilt, gt)
        elif stage == 'front':
            seg = Seg(p, c+[-.06-front_clearance,0,0], y, gy, 0., tilt, gt, min_steps=10, speed=.7)
        elif stage == 'insert':
            seg = Seg(p, c, y, gy, 0., tilt, gt, min_steps=10, speed=.6)
        elif stage == 'close':
            self._close_book_xy = self.book_xyz[:2].copy()
            seg = Seg(p, p, y, y, 1., tilt, tilt, min_steps=1, hold=10)
        elif stage == 'verifylift':
            seg = self.hold_seg(i, 1., 6)
        elif stage == 'lift':
            seg = Seg(p, p+[self.lift_dx,0,.15], y, y, 1., tilt, tilt, min_steps=10, speed=.85)
        else:
            seg = self.hold_seg(i, 1., 12)
        if stage in self.SPEEDUP:
            hold = seg.n-seg.move
            seg.move = max(self.MIN_FRAMES[stage], math.ceil(seg.move/self.SPEEDUP[stage]))
            seg.n = seg.move+hold
        moved_with_grip = (self._close_book_xy is not None and
                           np.linalg.norm(self.book_xyz[:2] - self._close_book_xy) >= self.late_lift_min_book_move)
        if stage == 'lift' and self.late_lift_after and self.tick >= self.late_lift_after and moved_with_grip:
            seg.move = min(seg.move, self.late_lift_frames)
            seg.n = seg.move
        return seg

    def until(self, i, stage):
        if i != self.arm:
            return False
        if stage == 'push':
            return self.book_xyz[0] <= self.initial_x-self.PUSH_STOP
        if stage == 'close' and self.direct:
            return self.segs[i].k >= 6 and self.holding(i)
        return stage == 'verifylift' and self.holding(i)

    def advance(self, i, stage):
        if i != self.arm:
            return None
        target = None
        if stage == 'close':
            if self.holding(i):
                target = self.stages[i].index('lift')
            else:
                self.retries[i] += 1
                was_direct, self.direct = self.direct, False
                target = 0 if was_direct or self.book_box()[0][0] > self.edge()-.045 else self.stages[i].index('over')
        elif stage in ('verifylift', 'done'):
            if self.holding(i):
                target = self.stages[i].index('lift')
            else:
                self.retries[i] += 1
                self.direct = False
                target = 0
        if self.retries[i]:
            self.PUSH_STOP = .030
        return target

    def _integrate_action(self, action):
        self.qctrl = self._robot.update(self.qctrl, action, MAX_JOINT_DELTA)

    def observe(self, obs):
        qpos = np.asarray(obs['proprioception'], float)
        self._openings = np.asarray(obs['proprioception_grippers'], float).copy()
        if qpos.shape != (36,) or self._openings.shape != (2,) or not np.isfinite(qpos).all() or not np.isfinite(self._openings).all():
            raise ValueError('expected finite standard proprioception (36) and grippers (2)')
        for i, start in enumerate((0, 9)):
            self._pad_offsets[i] = PAD_IN_EE.copy()
            self._pad_offsets[i][1] += .5*(qpos[start+8]-qpos[start+7])
        self._qpos = qpos.copy()

    def plan(self):
        self.arm = 0 if self.book_xyz[1] >= 0 else 1
        self.stages = [['done'], ['done']]
        self.stages[self.arm] = list(self.SEQUENCE)
        self.initial_x = float(self.book_xyz[0])
        self.direct = (self.initial_x > self.edge()+self.direct_margin or
                       self.direct_edge_gap is not None and
                       self.book_box()[0][0] <= self.edge()+self.direct_edge_gap)
        self.k[self.arm] = 1 if self.direct else 0
        self.PUSH_STOP = self.push_stop_left if self.arm == 0 else self.push_stop_right
        if (self.arm == 1 and not self.direct and self.short_push_right is not None and
                self.target()[1] <= self.short_push_target_y_max):
            self.PUSH_STOP = self.short_push_right
        self._planned = True

    def before_action(self):
        if self.phase == 'lift' and not self.holding(self.arm):
            self.k[self.arm] = self.stages[self.arm].index('verifylift')
            self.segs[self.arm] = None

    def adjust_action(self, action, ee):
        if self.idle_wrist_z is not None:
            if self.phase in ('lift', 'done'):
                self._idle_wrist_started = True
            if self.idle_after_grasp and not self._idle_wrist_started:
                return action
            idle = 1 - self.arm
            current = float(ee[6*idle+2] + self.pelvis_pos[2])
            if self._idle_wrist_goal is None:
                self._idle_wrist_goal = current
            self._idle_wrist_goal += .25 * (self.idle_wrist_z - self._idle_wrist_goal)
            action[6*idle:6*idle+3] = 0.
            action[6*idle+2] = np.clip(self._idle_wrist_goal-current,
                                       -self.idle_wrist_speed*.05, self.idle_wrist_speed*.05)
        if self.lift_idle_yaw_ratio:
            action = self._lift_idle_yaw(action, ee)
        return action

    def _lift_idle_yaw(self, action, ee):
        """Turn the idle wrist about world z while the active arm moves a held book.

        Per step the idle yaw increment is ratio * the active arm's commanded
        joint-step norm (from this controller's own IK of this step's action),
        so both arms' joint speeds are similar. Once engaged, the idle wrist
        holds its commanded position and the accumulated yaw for the rest of
        the episode (no return motion). With lift_idle_first_only the yaw
        stops for good after the first retry (a slipped or missed grasp); with
        lift_idle_delay=k it starts only after k consecutive holding lift steps;
        with lift_idle_book_dz the estimated book must have risen that much
        above its pre-lift estimate (book carried, not slipping). Inputs: commanded joints, gripper
        proprioception (holding), the book pose estimate and the phase clock only.
        """
        idle, a = 1-self.arm, self.arm
        step = 0.
        holding = self.phase in self.lift_idle_phases and self.holding(a)
        self._lift_ticks = self._lift_ticks+1 if holding else 0
        if self.phase not in self.lift_idle_phases:
            self._lift_book_z = float(self.book_xyz[2])  # estimate just before the lift
        if self.lift_idle_book_dz is not None and (
                self._lift_book_z is None or self.book_xyz[2]-self._lift_book_z < self.lift_idle_book_dz):
            holding = False  # the estimated book is not rising with the hand (yet)
        if (holding and self._lift_ticks > self.lift_idle_delay
                and self._idle_yaw < self.lift_idle_yaw_cap
                and not (self.lift_idle_first_only and any(self.retries))):
            probe = action.copy()
            probe[6*idle:6*idle+6] = 0.
            q = self._robot.update(self.qctrl, probe, MAX_JOINT_DELTA)
            motion = float(np.linalg.norm(q[7*a:7*a+7]-self.qctrl[7*a:7*a+7]))
            step = min(self.lift_idle_yaw_ratio*motion, self.lift_idle_yaw_step,
                       self.lift_idle_yaw_cap-self._idle_yaw)
        if self._idle_ref is None:
            if step <= 0:
                return action
            j = 6*idle
            self._idle_ref = (ee[j:j+3].astype(float).copy(), R.from_rotvec(ee[j+3:j+6].astype(float)))
        self._idle_yaw += step
        j = 6*idle
        position, rotation = self._idle_ref
        target = R.from_rotvec([0., 0., self.lift_idle_yaw_sign*self._idle_yaw])*rotation
        action[j:j+3] = position-ee[j:j+3]
        action[j+3:j+6] = (target*R.from_rotvec(ee[j+3:j+6].astype(float)).inv()).as_rotvec()
        return action

    def act(self, obs, book_pose):
        self.update_pose(book_pose)
        self.observe(obs)
        if self._first:
            self._first = False
            return np.zeros(14, np.float32)
        if not self._planned:
            self.plan()
        self.tick += 1
        self.before_action()
        ee, self._pads = self.cmd_pose(), self.cmd_pads()
        action = np.zeros(14, np.float32)
        for i in (0, 1):
            seg = self.segs[i]
            if seg is not None and not seg.done and self.until(i, self.stage(i)):
                seg.k = seg.n
            if seg is not None and seg.done:
                target = self.advance(i, self.stage(i))
                self.k[i] = target if target is not None else min(self.k[i]+1, len(self.stages[i])-1)
                self.segs[i] = None
            if self.segs[i] is None:
                self.segs[i] = self.segment(i, self.stage(i))
            pos, yaw, tilt = self.segs[i].next()
            self.yaw[i], self.tilt[i] = yaw, tilt
            target_rot = R.from_rotvec(tilt) * R.from_rotvec([0,0,yaw]) * self.R0[i]
            action[6*i:6*i+3] = pos-self._pads[i]
            action[6*i+3:6*i+6] = (target_rot * R.from_rotvec(ee[6*i+3:6*i+6]).inv()).as_rotvec()
            self.grip[i] = self.segs[i].grip
        action[12:] = self.grip
        action = self.adjust_action(action, ee)
        self._integrate_action(action)
        return action


# Public fixture geometry in assembly-body coordinates. The compiled counter
# and board geom boxes are axis aligned in this frame (principal axes permute).
COUNTER_IN_SHELF = np.array([
    [-.12950800359249215, -.5952209830284148, -.10710000246763234],
    [.2541449964046488, .46411898732185647, -.08992200344800945],
])
BOARD_IN_SHELF = np.array([
    [-.121734999120236, -.5723159909248379, .22869700193405146],
    [.08440700173378046, .44121399521827964, .24587500095367437],
])
BOX_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
PAD_HALF_SIZES = np.array([[.01, .002, .012], [.01, .003, .012]])
BOOK_COM = np.array([-.004260220856839033,-.011251038636037136,-.014426348154117884])


def transformed_bounds(bounds, pose):
    """Known fixture box through a predicted world assembly pose."""
    lo, hi = bounds
    corners = (lo+hi)/2 + BOX_SIGNS*(hi-lo)/2
    world = R.from_quat(pose[3:], scalar_first=True).apply(corners) + pose[:3]
    return np.stack((world.min(0), world.max(0)))


class ShelfBookController(PickBookController):
    """ShelfFastExpert phases with live book vision and frozen reset shelf pose.

    shelf_pose is [x,y,z,qw,qx,qy,qz] of the assembly body. It is frozen from
    initial_shelf_pose, or the first act() call, then reused throughout episode.
    Completion and phase transitions never read evaluator success/contact.
    """

    SHELF_GAP, NUDGE_PITCH, NUDGE_CLEARANCE = .005, .45, .010
    SEQUENCE = PickBookController.SEQUENCE[:-1] + ('raise', 'slide', 'place', 'release', 'back', 'done')

    def __init__(self, initial_book_pose, initial_shelf_pose=None, pelvis_pos=None,
                 pinch_edge=-.005, pinch_height=-.002, carry_pitch=1.10,
                 raise_clearance=.025, slide_clearance=.015, place_clearance=.003,
                 push_overhang=.030, repush_overhang=.024,
                 free_speed=1., carry_speed=1., lift_speed=1.,
                 wrist_follow=0., wrist_follow_speed=.15,
                 support_margin=0., reset_push_passes=False, ensure_overhang=0.,
                 grasp_pitch=.9, grasp_yaw=.8, close_steps=11, idle_wrist_z=None,
                 com_yaw_blend=0., push_speed=.65, push_dx=.09,
                 phase_speeds=None, pose_filter=None):
        super().__init__(initial_book_pose, pelvis_pos=pelvis_pos)
        values = (pinch_edge,pinch_height,carry_pitch,raise_clearance,slide_clearance,
                  place_clearance,push_overhang,repush_overhang,free_speed,carry_speed,
                  lift_speed,wrist_follow,wrist_follow_speed,support_margin,ensure_overhang,grasp_pitch,grasp_yaw,com_yaw_blend,push_speed,push_dx)
        if not np.isfinite(values).all() or min(free_speed,carry_speed,lift_speed,wrist_follow_speed,push_speed)<=0 or not 0<=wrist_follow<=1:
            raise ValueError('Require finite shelf parameters, positive speeds, wrist_follow in [0,1]')
        self.TARGET_EDGE, self.HEIGHT_DZ = pinch_edge, pinch_height
        self.PITCH, self.YAW = grasp_pitch, grasp_yaw
        self.grasp_yaw, self.com_yaw_blend = grasp_yaw, com_yaw_blend
        self.close_steps = int(close_steps)
        if self.close_steps < 1 or (idle_wrist_z is not None and not np.isfinite(idle_wrist_z)):
            raise ValueError('Require positive close_steps and a finite idle wrist height')
        self.idle_wrist_z = idle_wrist_z
        self.carry_pitch = carry_pitch
        self.raise_clearance, self.slide_clearance = raise_clearance, slide_clearance
        self.place_clearance = place_clearance
        self.push_overhang, self.repush_overhang = push_overhang, repush_overhang
        self.carry_speed, self.lift_speed = carry_speed, lift_speed
        self.push_speed = push_speed
        self.push_dx = push_dx
        self.phase_speeds = dict(phase_speeds or {})
        if any(stage not in self.SEQUENCE or not np.isfinite(speed) or speed <= 0
               for stage, speed in self.phase_speeds.items()):
            raise ValueError('phase_speeds requires shelf phase names and finite positive multipliers')
        self.SPEEDUP = dict(self.SPEEDUP)
        for stage in ('side_over','clear','clearup','over'):
            self.SPEEDUP[stage] *= free_speed
        self.wrist_follow, self.wrist_follow_speed = wrist_follow, wrist_follow_speed
        self.support_margin, self.ensure_overhang = support_margin, ensure_overhang
        self.reset_push_passes = bool(reset_push_passes)
        self._wrist_height_goal = None
        self._shelf_pose = None
        self.missing = self.raise_checks = self.push_passes = 0
        self.repush = False
        self._pose_filter = None
        if pose_filter:
            from rdb.hybrid.pose_filter import BookPoseFilter
            self._pose_filter = BookPoseFilter(**pose_filter)
        if initial_shelf_pose is not None:
            self.freeze_shelf(initial_shelf_pose)

    def freeze_shelf(self, pose):
        if self._shelf_pose is not None:
            return
        pose = np.asarray(pose, float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
            raise ValueError('shelf pose must be finite xyz + nonzero wxyz quaternion')
        self._shelf_pose = pose.copy()
        self.counter_bounds = transformed_bounds(COUNTER_IN_SHELF, pose)
        self.board_bounds = transformed_bounds(BOARD_IN_SHELF, pose)

    def observe(self, obs):
        super().observe(obs)
        centres, bounds, wrist_z = [], [], []
        joints = np.r_[self._qpos[:7], self._qpos[9:16]]
        for arm, (p, rot) in enumerate(self._robot.poses(joints, world=True)):
            start = 0 if arm == 0 else 9
            # Hand and right finger each have a pi-z frame transform. Pad
            # centres below are expressed directly in the attachment/EE frame.
            fingers = np.array([[0., -self._qpos[start+7]-.002, .1034],
                                [0., self._qpos[start+8]+.0055, .1034]])
            world = p + fingers @ rot.T
            radius = PAD_HALF_SIZES @ np.abs(rot).T
            centres.append(world)
            bounds.append(np.stack((world-radius, world+radius), axis=1))
            wrist_z.append(float(p[2]))
        self._measured_pad_centres = np.array(centres)
        self._measured_pad_bounds = np.array(bounds)
        self._measured_wrist_z = wrist_z

    def adjust_action(self, action, ee):
        if self.wrist_follow <= 0 and self.idle_wrist_z is None:
            return action
        active, idle = self.arm, 1-self.arm
        current = ee[6*idle+2]+self.pelvis_pos[2]
        desired = max(self.top()+.16, self._measured_wrist_z[active] if self.idle_wrist_z is None else self.idle_wrist_z)
        if self._wrist_height_goal is None:
            self._wrist_height_goal = current
        self._wrist_height_goal += .25*(desired-self._wrist_height_goal)
        delta = (self.wrist_follow if self.idle_wrist_z is None else 1.)*(self._wrist_height_goal-current)
        # The idle hand stays at its home x/y, upright and open. This height
        # floor leaves its fingertip pads safely above the work surface.
        action[6*idle:6*idle+3] = 0.
        action[6*idle+2] = np.clip(delta,-self.wrist_follow_speed*.05,self.wrist_follow_speed*.05)
        return action

    def plan(self):
        super().plan()
        self.k[self.arm] = 0  # Shelf always aligns above the work first.

    def before_action(self):
        if self.phase in ('lift', 'raise', 'slide', 'place'):
            self.missing = self.missing+1 if not self.holding(self.arm) else 0
            if self.missing >= 3:
                self.retries[self.arm] += 1
                self.direct = False
                self.k[self.arm] = 0
                self.segs[self.arm] = None
                self.grip[self.arm] = 0.
                self.missing = 0
                if self.reset_push_passes:
                    self.push_passes = 0
        else:
            self.missing = 0

    def pad_mid(self, i):
        return self._measured_pad_centres[i].mean(0)

    def pad_floor(self):
        return float(self._measured_pad_bounds[self.arm, :, 0, 2].min())

    def top(self):
        return float(self.counter_bounds[1, 2])

    def until(self, i, stage):
        if i == self.arm and stage == 'push':
            return self.book_box()[0][0] <= self.edge()-self.push_goal()
        if i == self.arm and stage == 'close':
            return False
        return super().until(i, stage)

    def push_goal(self):
        if self.support_margin <= 0:
            return self.push_overhang
        # More overhang gives a deeper pinch and a shorter gravity lever arm.
        # Cap it by the estimated centre of mass's remaining table support.
        com = self.book_rotation.apply(BOOK_COM)+self.book_xyz
        safe = com[0]-self.book_box()[0][0]-self.support_margin
        return min(self.push_overhang,max(.015,float(safe)))

    def segment(self, i, stage):
        seg = self._shelf_segment(i, stage)
        speed = self.phase_speeds.get(stage, 1.) if i == self.arm else 1.
        if speed != 1.:
            hold = seg.n-seg.move
            seg.move = max(1, int(np.ceil(seg.move/speed)))
            seg.n = seg.move+hold
        return seg

    def _shelf_segment(self, i, stage):
        if i != self.arm:
            return super().segment(i, stage)
        if self.com_yaw_blend > 0:
            if stage in ('side_over','side_down'):
                self.YAW = self.grasp_yaw
            elif stage in ('over','front','insert'):
                com = self.book_rotation.apply(BOOK_COM)+self.book_xyz
                direction = com[:2]-self.target()[:2]
                preferred = (-1 if i == 0 else 1)*np.arctan2(direction[1],direction[0])
                preferred = float(np.clip(preferred,0.,1.3))
                self.YAW = (1-self.com_yaw_blend)*self.grasp_yaw+self.com_yaw_blend*preferred
        p, y, t = self.pinch(i), self.yaw[i], self.tilt[i]
        blo, bhi = self.board_bounds
        if stage == 'verifylift':
            return Seg(p, p+[-.004,0,.004], y, y, 1., t, t, min_steps=6)
        if stage in ('side_over', 'side_down'):
            seg = super().segment(i, stage)
            if stage == 'side_over':
                seg.p1[2] = self.book_box()[1][2]+.11
            else:
                lower = self._measured_pad_centres[i, :, 2].argmin()
                offset = self._measured_pad_centres[i, lower]-self.pad_mid(i)
                lo, hi = self.book_box()
                sign = 1 if i == 0 else -1
                desired = np.array([self.book_xyz[0]+.02, hi[1]+.015 if sign > 0 else lo[1]-.015, 0.])
                seg.p1[:2] = (desired-offset)[:2]
                seg.p1[2] = self.pad_mid(i)[2]+self.top()+self.NUDGE_CLEARANCE-self.pad_floor()
            gy = -self.YAW if i == 0 else self.YAW
            seg.t1 = R.from_rotvec([0,0,gy]).apply([0,-self.NUDGE_PITCH,0])
            return seg
        if stage == 'push':
            sign = 1 if i == 0 else -1
            return Seg(p, p+[-self.push_dx,-sign*.085,self.top()+self.NUDGE_CLEARANCE-self.pad_floor()],
                       y, y, 0., t, t, min_steps=12, speed=self.push_speed)
        if stage == 'close':
            return Seg(p, p+[0,0,.002], y, y, 1., t, t, min_steps=self.close_steps)
        if stage == 'lift':
            return Seg(p, p+[0,0,.16], y, y, 1., t, t, min_steps=12, speed=.8*self.lift_speed)
        if stage == 'raise':
            lo, hi = self.book_box()
            z = max(1.11, p[2]+bhi[2]+self.raise_clearance-lo[2])
            safe_x = blo[0]-.025-(hi[0]-p[0])
            return Seg(p, [min(p[0],safe_x),p[1],min(z,1.25)], y, y, 1., t,
                       R.from_rotvec([0,0,y]).apply([0,-self.carry_pitch,0]), min_steps=8, speed=1.15*self.carry_speed)
        if stage == 'slide':
            lo, _ = self.book_box()
            z = p[2]+bhi[2]+self.slide_clearance-lo[2]
            return Seg(p, [blo[0]-self.SHELF_GAP,p[1],z], y, y, 1., t, t, min_steps=8, speed=1.15*self.carry_speed)
        if stage == 'place':
            lo, _ = self.book_box()
            drop = max(lo[2]-bhi[2]-self.place_clearance, 0.)
            return Seg(p, p+[0,0,-drop], y, y, 1., t, t, min_steps=6, speed=.8)
        if stage == 'release':
            return Seg(p, p+[-.025,0,.006], y, y, 0., t, t, min_steps=6)
        if stage in ('back', 'done'):
            return Seg(p, p+[-.10,0,.025], y, y, 0., t, t, min_steps=8)
        return super().segment(i, stage)

    def advance(self, i, stage):
        if i != self.arm:
            return None
        repush_goal = self.repush_overhang
        minimum_overhang = self.ensure_overhang
        if self.support_margin > 0:
            repush_goal = min(repush_goal,self.push_goal()-.006)
            # A narrow book may reach the COM support limit with <20 mm
            # exposed. Do not reject an overhang the safe nudge cannot exceed.
            minimum_overhang = min(minimum_overhang,self.push_goal()-.006)
        if stage == 'push' and self.book_box()[0][0] > self.edge()-repush_goal and self.push_passes < 2:
            self.repush = True
            self.push_passes += 1
            self.retries[i] += 1
        if stage == 'clearup' and self.repush:
            self.repush = False
            return 0
        if stage in ('clearup','over') and minimum_overhang > 0 and self.book_box()[0][0] > self.edge()-minimum_overhang:
            self.retries[i] += 1
            self.direct = False
            if self.reset_push_passes:
                self.push_passes = 0
            return 0
        if stage == 'raise':
            if self.book_box()[0][2] < self.board_bounds[1,2]+.012 and self.raise_checks < 3:
                self.raise_checks += 1
                return self.stages[i].index('raise')
        if stage in ('raise', 'slide', 'place') and not self.holding(i):
            self.retries[i] += 1
            self.direct = False
            self.grip[i] = 0.
            if self.reset_push_passes:
                self.push_passes = 0
            return 0
        if stage == 'done':
            self.retries[i] += 1
            self.direct = False
            return 0
        if stage in ('release', 'back'):
            return None
        target = super().advance(i, stage)
        if self.reset_push_passes and stage == 'close' and target == 0:
            self.push_passes = 0
        return target

    def act(self, obs, book_pose, shelf_pose):
        self.freeze_shelf(shelf_pose)
        if self._pose_filter is not None:
            book_pose = self._pose_filter(book_pose, obs, self)
        return super().act(obs, book_pose)
