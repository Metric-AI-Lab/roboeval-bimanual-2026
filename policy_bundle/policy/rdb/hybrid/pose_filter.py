"""Optional book-pose filters using camera predictions and standard proprioception.

No environment, contacts, evaluator labels or simulator objects are available here.
Each controller owns one filter, so history cannot cross episode boundaries.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R


class BookPoseFilter:
    HELD_PHASES = ('lift', 'raise', 'slide', 'place')

    def __init__(self, mode='ema', alpha=.5, window=3, rotation_alpha=None,
                 phases=None, jump_mm=0., max_reject=2, held_height=.03,
                 support_tilt_deg=10., support_height=.02, support_margin=.005,
                 support_motion_mm=4., grip_gap_min=.022, grip_gap_max=.028,
                 grip_angle_deg=15., grip_max_shift=.015):
        if mode not in ('ema', 'median', 'hand_ema', 'support', 'carry_plane'):
            raise ValueError('Unknown book pose filter mode')
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('alpha is the current measurement weight in (0,1]')
        if int(window) != window or not 1 <= window <= 9 or window % 2 != 1:
            raise ValueError('window must be an odd integer from 1 through 9')
        if rotation_alpha is None:
            rotation_alpha = 1. if mode == 'hand_ema' else alpha
        if not np.isfinite(rotation_alpha) or not 0 < rotation_alpha <= 1:
            raise ValueError('rotation_alpha must be in (0,1]')
        if not np.isfinite(jump_mm) or jump_mm < 0:
            raise ValueError('jump_mm must be nonnegative')
        if int(max_reject) != max_reject or max_reject < 0:
            raise ValueError('max_reject must be a nonnegative integer')
        if not np.isfinite(held_height) or not 0 <= held_height <= .2:
            raise ValueError('held_height must be in [0,.2] metres')
        if (not np.isfinite([support_tilt_deg,support_height,support_margin,support_motion_mm]).all()
                or not 0 < support_tilt_deg <= 20 or not 0 < support_height <= .05
                or not 0 <= support_margin <= .05 or support_motion_mm <= 0):
            raise ValueError('Require bounded, finite support-plane guards')
        if (not np.isfinite([grip_gap_min,grip_gap_max,grip_angle_deg,grip_max_shift]).all()
                or not .01 <= grip_gap_min < grip_gap_max <= .05
                or not 0 < grip_angle_deg <= 30 or not 0 < grip_max_shift <= .03):
            raise ValueError('Require bounded, finite grasp-plane guards')
        self.mode, self.alpha = mode, float(alpha)
        self.rotation_alpha = float(rotation_alpha)
        self.phases = None if phases is None else frozenset(phases)
        self.jump, self.max_reject = jump_mm/1000, int(max_reject)
        self.held_height = held_height
        self.support_cos = np.cos(np.deg2rad(support_tilt_deg))
        self.support_height, self.support_margin = support_height, support_margin
        self.support_motion = support_motion_mm/1000
        self.grip_gap_min, self.grip_gap_max = grip_gap_min, grip_gap_max
        self.grip_cos = np.cos(np.deg2rad(grip_angle_deg))
        self.grip_max_shift = grip_max_shift
        self.history = deque(maxlen=int(window))
        self.last = self.anchor = self.lift_z = None
        self.previous_phase = None
        self.rejected = self.rejections = 0

    def _rotation(self, old, raw):
        if self.rotation_alpha == 1.:
            return raw.copy()
        before = R.from_quat(old, scalar_first=True)
        innovation = before.inv()*R.from_quat(raw, scalar_first=True)
        return (before*R.from_rotvec(self.rotation_alpha*innovation.as_rotvec())).as_quat(scalar_first=True)

    def __call__(self, raw, obs, controller):
        raw = np.asarray(raw, float)
        if raw.shape != (7,) or not np.isfinite(raw).all() or np.linalg.norm(raw[3:]) < 1e-8:
            raise ValueError('Require finite xyz and nonzero wxyz')
        phase = controller.phase
        if (phase in ('side_over', 'over') and
                self.previous_phase in ('close', 'lift', 'raise', 'slide', 'place', 'done')):
            # A failed grasp may have moved the book. Do not smooth a recovery
            # target toward a pose from the preceding failed carry.
            self.history.clear()
            self.last = self.anchor = self.lift_z = None
            self.rejected = 0
        if self.phases is not None and phase not in self.phases:
            self.history.clear()
            self.anchor = self.lift_z = None
            self.last, self.previous_phase = None, phase
            self.rejected = 0
            return raw.copy()
        used = raw.copy()
        if self.mode == 'carry_plane':
            segment = controller.segs[controller.arm]
            ready = (phase in ('slide','place') or
                     phase == 'raise' and segment is not None and segment.done)
            if ready and controller.grip[controller.arm] >= .5:
                from rdb.hybrid.book_controller import BOOK_CORNERS
                q = np.asarray(obs['proprioception'],float)
                arm=controller.arm;start=0 if arm == 0 else 9
                position,hand = controller._robot.poses(np.r_[q[:7],q[9:16]],world=True)[arm]
                # Midpoint of the inner pad faces, accounting for unequal pad
                # half-widths (.002/.003 m), not just their centre midpoint.
                inner = position+hand@np.array([0,.5*(q[start+8]-q[start+7])+.00125,.1034])
                gap = q[start+7]+q[start+8]+.0025
                rotation = R.from_quat(raw[3:],scalar_first=True)
                normal = rotation.apply([0,0,1])
                lower = (rotation.apply(BOOK_CORNERS)+raw[:3])[:,2].min()
                top = controller.counter_bounds[1,2]
                shift = normal@(inner-raw[:3])-BOOK_CORNERS[:,2].mean()
                if (self.grip_gap_min <= gap <= self.grip_gap_max and
                        abs(normal@(hand@np.array([0,1,0]))) >= self.grip_cos and
                        lower >= top+.03 and inner[2] >= top+.08 and
                        np.linalg.norm(raw[:3]-inner) < .12 and abs(shift) <= self.grip_max_shift):
                    # Keep camera orientation and tangential position. A tight,
                    # near-parallel grasp constrains only the thickness midplane.
                    used[:3] += normal*shift
        elif self.mode == 'support':
            from rdb.hybrid.book_controller import BOOK_CORNERS, BOOK_COM
            self.history.append(raw.copy())
            rotation = R.from_quat(raw[3:], scalar_first=True)
            corners = rotation.apply(BOOK_CORNERS)+raw[:3]
            com = rotation.apply(BOOK_COM)+raw[:3]
            top = controller.counter_bounds[1,2]
            quiet = (len(self.history) >= 3 and
                np.linalg.norm(np.diff(np.array(self.history)[:, :3],axis=0),axis=1).max() <= self.support_motion)
            if (phase in ('side_over','side_down','over','front','insert') and
                    controller.grip[controller.arm] < .5 and quiet and
                    rotation.apply([0,0,1])[2] >= self.support_cos and
                    abs(corners[:,2].min()-top) <= self.support_height and
                    np.all(com[:2] >= controller.counter_bounds[0,:2]+self.support_margin) and
                    np.all(com[:2] <= controller.counter_bounds[1,:2]-self.support_margin)):
                # Public book/counter geometry and the camera-derived fixture
                # constrain a quiet, ungrasped, supported book to the table plane.
                yaw = rotation.as_euler('xyz')[2]
                flat = R.from_euler('z',yaw)
                used[3:] = flat.as_quat(scalar_first=True)
                used[2] = top-flat.apply(BOOK_CORNERS)[:,2].min()
        elif self.mode == 'hand_ema':
            # A world-frame freeze would prevent the estimated book from moving
            # with the hand. Average in the measured hand frame instead.
            if phase in self.HELD_PHASES and self.previous_phase not in self.HELD_PHASES:
                self.lift_z = float(raw[2])
            opening = float(obs['proprioception_grippers'][controller.arm])
            held = (phase in self.HELD_PHASES and controller.grip[controller.arm] >= .5
                    and .05 <= opening < .9 and self.lift_z is not None
                    and raw[2] >= self.lift_z+self.held_height)
            if held:
                q = np.asarray(obs['proprioception'], float)
                joints = np.r_[q[:7], q[9:16]]
                position, rotation = controller._robot.poses(joints, world=True)[controller.arm]
                local = rotation.T@(raw[:3]-position)
                if self.anchor is None:
                    self.anchor = local
                else:
                    jump = np.linalg.norm(local-self.anchor)
                    if self.jump and jump > self.jump and self.rejected < self.max_reject:
                        self.rejected += 1
                        self.rejections += 1
                    elif self.jump and jump > self.jump:
                        # Never keep an indefinitely stale grasp hypothesis.
                        self.anchor, self.rejected = local, 0
                    else:
                        self.anchor = (1-self.alpha)*self.anchor+self.alpha*local
                        self.rejected = 0
                used[:3] = position+rotation@self.anchor
                if self.last is not None:
                    used[3:] = self._rotation(self.last[3:], raw[3:])
            else:
                self.anchor = None
                self.rejected = 0
                if phase not in self.HELD_PHASES:
                    self.lift_z = None
        elif self.mode == 'median':
            self.history.append(raw.copy())
            history = np.array(self.history)
            used[:3] = np.median(history[:, :3], axis=0)
            quats = history[:, 3:]/np.linalg.norm(history[:, 3:], axis=1, keepdims=True)
            # Quaternion medoid handles sign ambiguity without averaging two modes.
            index = np.argmin((1-np.abs(quats@quats.T)).sum(axis=1))
            used[3:] = history[index, 3:]
        elif self.last is not None:
            used[:3] = (1-self.alpha)*self.last[:3]+self.alpha*raw[:3]
            used[3:] = self._rotation(self.last[3:], raw[3:])
        self.last, self.previous_phase = used.copy(), phase
        return used
