"""Observation-only PackBox controller: rdb/gen/scripted_pack_box_h.py (PackBoxHumanExpert) on legal inputs.

Inputs: the packing-box pose (x, y, z, yaw) estimated at reset — the flap hinges are rigid in the box frame — and, on
every step, the two flap angles estimated from the cameras (`flap_rem` = q − closed limit for the flap the left (0) /
right (1) arm folds), plus the standard observation. Everything the teacher read from the simulator is rebuilt from
these: hinge anchors (with the teacher's reach-friendly shift along the hinge), axes, flap directions and lengths.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from rdb.hybrid.obs_script import ObsScript, Seg

BOX_PELVIS = np.array([.2, .3, .2])   # calibrated: PackBox robot base (see verify_box_controller --calibrate)
# Flap geometry in the packing-box body frame (measured, identical for every seed). Arm 0 (left) folds the +x flap.
HINGE_X = (.2178, -.2100)          # hinge line x per arm
HINGE_Z = (.1440, .1443)           # hinge line z per arm
HINGE_Y0 = .0033                   # flap centre along the hinge line
HALF_WIDTH = .2106                 # flap half-width along the hinge
FLAP_LEN = (.2089, .2188)          # flap extent from the hinge
REST_REM = 2.793                   # q − lower limit at reset (both flaps, every seed)
U_REST = ((.938, 0., .346), (-.938, 0., .348))   # flap direction at reset (box frame)
AXIS = ((0., 1., 0.), (0., -1., 0.))             # hinge axis (box frame)
HOME_Y = (.6, 0.)                  # world y of the left / right pinch point at reset (robot base at y 0.3)


class FollowFlap(Seg):
    """Track the estimated flap with a bounded tangential pushing lead (the teacher's FollowFlap)."""

    def __init__(self, ctl, arm, floor=0.0):
        self.ctl, self.arm, self.floor = ctl, arm, floor
        p, y = ctl.pinch(arm), ctl.yaw[arm]
        super().__init__(p, p, y, y, 0.0, min_steps=1, hold=180)
        self.last = p.copy()
        self.velocity = np.zeros(3)
        self.history = []
        self.stalled = False

    def next(self):
        self.k += 1
        c = self.ctl
        a, axis, q, lo, u = c.flap(self.arm)
        remaining = q - lo
        self.history.append(float(remaining))
        floor = self.floor
        if self.arm == 1 and c.k[0] < c.stages[0].index("watch"):
            floor = c.GATE
        goal = max(floor, remaining - c.LEAD)
        if len(self.history) >= c.STALL_STEPS and remaining > max(1.6, floor + .2):
            self.stalled = self.history[-c.STALL_STEPS] - remaining < .035
        direction = R.from_rotvec(-axis * (remaining - goal)).apply(u)
        target = a + c.R * direction - np.array([0., 0., c.pad_dz[self.arm]])
        target -= np.array([0., 0., c.PRESS_DZ]) if remaining < .5 else 0.
        delta = target - self.last
        delta *= min(1., c.STEP / max(np.linalg.norm(delta), 1e-9))
        self.velocity = .65 * self.velocity + .35 * delta
        self.last += self.velocity
        yaw = (-.30 if self.arm == 0 else .30) * np.clip((2.8 - remaining) / 2.8, 0., 1.)
        return self.last.copy(), float(yaw), self.t0


class TrackGrasp(Seg):
    """rdb/gen/scripted_pack_box_grasp.py TrackGrasp without its contact-sensing corrections: track the flap edge
    with a closing gripper (a brief flap grasp; the task's progress stages count a held flap)."""

    def __init__(self, ctl, arm):
        self.ctl, self.arm = ctl, arm
        p, y = ctl.pinch(arm), ctl.yaw[arm]
        super().__init__(p, p, y, y, 1., ctl.tilt[arm], min_steps=1, hold=70)
        self.last = p.copy()

    def next(self):
        self.k += 1
        c = self.ctl
        a, axis, q, lo, u = c.flap(self.arm)
        normal = np.cross(-axis, u)
        target = a + u * c.GRASP_RADIUS + normal * c.GRASP_LEAD
        delta = target - self.last
        self.last += delta * min(1., c.GRASP_STEP / max(np.linalg.norm(delta), 1e-9))
        axis_yaw = np.arctan2(axis[1], axis[0])
        goal = self.y0 + ((axis_yaw - self.y0 + np.pi / 2) % np.pi - np.pi / 2)     # fingers close along the hinge
        yaw = self.y0 + np.clip(goal - self.y0, -.04 * self.k, .04 * self.k)
        return self.last.copy(), yaw, self.t0


class BoxController(ObsScript):
    R = .185
    BEHIND = .025
    LEAD = .30
    STEP = .006
    PRESS_DZ = .012
    PAD_DZ = .085
    CLOSED = .21
    GATE = 1.15
    STALL_STEPS = 24
    GRASP_LEAD, GRASP_STEP, GRASP_RADIUS = .05, .018, .185
    GRASP_STEPS = 40                      # fixed grasp duration (no contact sensing); 40 holds both flaps under pose noise (60/60)

    RETRY_SHIFTS = (.03, -.03, .06, -.06)   # anchor shift along the hinge on retry k (a failed grasp/fold never repeats)

    def __init__(self, box_pose, pelvis=BOX_PELVIS, grasp=False, grasp_steps=None, vary_retries=False, reach_anchor=False,
                 extreme_pad_dz=False, extreme_grasp_steps=None,
                 coord_watch_z=None, coord_watch_speed=.05,
                 **kw):
        if coord_watch_z is not None and (not np.isfinite(coord_watch_z) or
                                          not np.isfinite(coord_watch_speed) or coord_watch_speed <= 0):
            raise ValueError("coord_watch_z must be finite and coord_watch_speed positive")
        self.coord_watch_z = coord_watch_z
        self.coord_watch_speed = float(coord_watch_speed)
        self.reach_anchor = bool(reach_anchor)
        self.vary_retries = bool(vary_retries)
        self.shift = [0., 0.]
        if grasp_steps is not None:
            self.GRASP_STEPS = int(grasp_steps)
        self.grasp = bool(grasp)          # brief flap grasps before the fold (scripted_pack_box_grasp)
        x, y, z, yaw = map(float, box_pose)
        yaw = yaw % np.pi                     # two-fold symmetric estimate; the box's +x flap faces +y (yaw in (0, pi))
        self.pad_dz = [self.PAD_DZ, self.PAD_DZ]
        extreme = False
        if extreme_pad_dz:
            # Keep the pushing fingers on the reachable side of the flap in
            # extreme layouts. This uses only the estimated reset box pose.
            if yaw > 1.75 and x > .74:
                self.pad_dz[1] = 0.
                extreme = True
            if yaw < 1.25 and x > .77 and y > .32:
                self.pad_dz[0] = .17
                extreme = True
        if extreme and extreme_grasp_steps is not None:
            self.GRASP_STEPS = int(extreme_grasp_steps)
        self.box_p, self.box_r = np.array([x, y, z]), R.from_euler("z", yaw)
        self.rem = [REST_REM, REST_REM]
        self.geom, self.t = [], []
        for i in (0, 1):
            axis = self.box_r.apply(AXIS[i])
            line = self.box_p + self.box_r.apply([HINGE_X[i], HINGE_Y0, HINGE_Z[i]])
            preferred = (.65 - line[0]) / axis[0] if abs(axis[0]) > 1e-9 else 0.
            if self.reach_anchor:   # hinge point nearest (0.65, the arm's home y): slides the grasp toward its arm
                goal = np.array([.65, HOME_Y[i]])
                preferred = float((goal - line[:2]) @ axis[:2] / max(axis[:2] @ axis[:2], 1e-9))
            t = np.clip(preferred, -HALF_WIDTH + .045, HALF_WIDTH - .045)
            self.t.append(float(t))
            u_rest = self.box_r.apply(U_REST[i])
            u_rest = u_rest - axis * (u_rest @ axis)
            u_rest /= np.linalg.norm(u_rest)
            closed = R.from_rotvec(-axis * REST_REM).apply(u_rest)
            self.geom.append((line, axis, closed))
        super().__init__(pelvis, **kw)

    def update_flaps(self, flap_rem):
        self.rem = [float(r) for r in flap_rem]

    def flap(self, i):
        line, axis, closed = self.geom[i]
        t = np.clip(self.t[i] + self.shift[i], -HALF_WIDTH + .045, HALF_WIDTH - .045)
        rem = self.rem[i]
        return line + axis * t, axis.copy(), rem, 0., R.from_rotvec(axis * rem).apply(closed)

    def _retry(self, i):
        self.retries[i] += 1
        if self.vary_retries:
            self.shift[i] = self.RETRY_SHIFTS[(self.retries[i] - 1) % len(self.RETRY_SHIFTS)]

    def length(self, i):
        return FLAP_LEN[i]

    def start_point(self, i):
        a, axis, q, lo, u = self.flap(i)
        v = np.cross(-axis, u)
        return a + u * self.R - v * self.BEHIND, v

    def plan(self):
        grasp = ["grasp", "release", "reset"] if self.grasp else []
        self.stages = [["outside", "under", "engage"] + grasp + ["fold", "clear", "watch", "done"] for _ in range(2)]

    def segment(self, i, stage):
        p, y = self.pinch(i), self.yaw[i]
        a, axis, q, lo, u = self.flap(i)
        start, v = self.start_point(i)
        if stage == "outside":
            beyond = a + u * (self.length(i) + .07)
            return Seg(p, [beyond[0], beyond[1], max(p[2], beyond[2] + .07)], y, y, 0., min_steps=8, speed=1.2)
        if stage == "under":
            beyond = a + u * (self.length(i) + .07)
            return Seg(p, beyond - v * .05 - [0., 0., self.pad_dz[i]], y, y, 0., min_steps=8, speed=1.)
        if stage == "engage":
            return Seg(p, start - [0., 0., self.pad_dz[i]], y, y, 0., min_steps=8, speed=1.)
        if stage == "grasp":
            return TrackGrasp(self, i)
        if stage == "release":
            return Seg(p, p, y, y, 0., self.tilt[i], min_steps=1, hold=4)
        if stage == "reset":
            return Seg(p, a + u * self.R - [0., 0., self.pad_dz[i]], y, 0., 0., self.tilt[i], np.zeros(3), min_steps=8)
        if stage == "fold":
            return FollowFlap(self, i)
        if stage == "clear":
            return Seg(p, p + np.array([0., 0., .12]), y, y, 0., min_steps=10, speed=1.)
        return self.hold_seg(i, 0., steps=10 if stage == "watch" else 1000)

    def until(self, i, stage):
        if stage == "grasp":
            return self.segs[i].k >= self.GRASP_STEPS
        return stage == "fold" and (self.rem[i] < self.CLOSED or self.segs[i].stalled)

    def advance(self, i, stage):
        if stage == "fold" and self.rem[i] >= self.CLOSED:
            self._retry(i)
            return 0
        if stage == "watch":
            if self.rem[i] >= .26:
                self._retry(i)
                return 0
            return self.stages[i].index("watch")
        return None

    def act(self, obs, flap_rem=None):
        if flap_rem is not None:
            self.update_flaps(flap_rem)
        return super().act(obs)

    def postprocess(self, action):
        """Optionally bring a finished arm toward the other wrist's height.

        The watch arm has already cleared its flap. The lower bound preserves
        its clearance; all positions come from our commanded joint history.
        """
        if self.coord_watch_z is None:
            return action
        ee = self.cmd_pose()
        for i in (0, 1):
            if self.stage(i) not in ("watch", "done"):
                continue
            other = 1 - i
            if self.stage(other) in ("watch", "done"):
                continue
            if other == 0 and self.retries[other]:
                continue
            target_z = max(float(ee[6 * other + 2]), self.coord_watch_z)
            action[6 * i + 2] = np.clip(target_z - float(ee[6 * i + 2]),
                                         -self.coord_watch_speed * .05,
                                         self.coord_watch_speed * .05)
        return action
