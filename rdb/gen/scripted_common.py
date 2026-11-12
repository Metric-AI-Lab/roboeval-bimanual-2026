"""Shared machinery for privileged-state scripted experts (training-data generators only).

A task expert subclasses `BimanualScript`: it lists a stage sequence per arm and, in `segment(i, stage)`,
returns where that arm's grasp point (midpoint of the fingertip pads) and yaw go next, re-planned from
the true object poses at every segment start (closed loop). Stage "sync:<tag>" waits until the other
arm reaches the same sync tag, so two-arm moves (lifts) start on the same step; equal displacements
then take equal time. Actions are EE deltas from the commanded pose (FK of the joint targets, which the
EE-delta action mode composes onto) to the next min-jerk point, like rdb.gen.scripted_stack_two_blocks.

Orientation: the hand keeps its reset (top-down) orientation, rotated by `yaw` about world z and
optionally tilted by `tilt` (a rotvec applied in the world frame after the yaw).

Only the generator reads privileged state; recorded observations are the challenge's standard ones.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

DT = 1 / 20
V_AVG = 0.22                    # m/s mean speed of a min-jerk move (peak ~1.9x)
W_AVG = 1.6                     # rad/s mean rotation speed
MIN_SEED = 100_000


def wrap(a: float, period: float) -> float:
    """Wrap to (-period/2, period/2]."""
    return -((-a + period / 2) % period - period / 2)


def yaw_of(quat_wxyz) -> float:
    return R.from_quat(np.asarray(quat_wxyz), scalar_first=True).as_euler("xyz")[2]


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


class Seg:
    """Min-jerk move of one arm's grasp point, yaw and tilt; `grip` is held during the move."""

    def __init__(self, p0, p1, y0, y1, grip, t0=None, t1=None, min_steps=6, hold=0, speed=1.0):
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.y0, self.y1, self.grip = y0, y1, grip
        self.t0 = np.zeros(3) if t0 is None else np.asarray(t0, float)
        self.t1 = self.t0 if t1 is None else np.asarray(t1, float)
        dur = max(np.linalg.norm(self.p1 - self.p0) / (V_AVG * speed), abs(y1 - y0) / W_AVG,
                  np.linalg.norm(self.t1 - self.t0) / W_AVG)
        self.move = max(min_steps, int(np.ceil(dur / DT)))
        self.n = self.move + hold
        self.k = 0

    def next(self):
        self.k += 1
        s = min_jerk(self.k / self.move)
        return (self.p0 + (self.p1 - self.p0) * s, self.y0 + (self.y1 - self.y0) * s,
                self.t0 + (self.t1 - self.t0) * s)

    @property
    def done(self) -> bool:
        return self.k >= self.n


class ArcSeg(Seg):
    """Grasp point rotated about a world axis through `centre` by `angle` (min-jerk in the angle):
    for sweeping hinged parts (flaps, doors). Yaw/tilt held."""

    def __init__(self, p0, centre, axis, angle, yaw, grip, tilt=None, w=0.9, min_steps=10, hold=0):
        super().__init__(p0, p0, yaw, yaw, grip, tilt, min_steps=min_steps, hold=hold)
        self.c, self.r0 = np.asarray(centre, float), np.asarray(p0, float) - np.asarray(centre, float)
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)
        self.angle = angle
        self.move = max(min_steps, int(np.ceil(abs(angle) / w / DT)))
        self.n = self.move + hold

    def next(self):
        self.k += 1
        s = min_jerk(self.k / self.move)
        p = self.c + R.from_rotvec(self.axis * self.angle * s).apply(self.r0)
        return p, self.y0, self.t0


class BimanualScript:
    """Base class. Subclasses set `self.stages = [[...], [...]]` in `plan()` and implement `segment`."""

    NOISE_POS, NOISE_ROT = 0.0025, 0.015
    QUIET = ("close", "release", "done", "wait")       # no DART noise in these stages (prefix match)

    def __init__(self, env, rng: np.random.Generator, noise: float = 1.0):
        self.env, self.rng, self.noise = env, rng, noise
        self.sides = list(env.robot.grippers.keys())    # [LEFT (+y), RIGHT (-y)]: action blocks 0:6 / 6:12, grips 12 / 13
        ee = self.cmd_pose()
        self.R0 = [R.from_rotvec(ee[6 * i + 3: 6 * i + 6]) for i in range(2)]
        self.off = [self.R0[i].inv().apply(self.pad_mid(i) - ee[6 * i: 6 * i + 3]) for i in range(2)]
        self.home = [self.pinch(i, ee) for i in range(2)]
        self.yaw = [0.0, 0.0]
        self.tilt = [np.zeros(3), np.zeros(3)]
        self.grip = [0.0, 0.0]
        self.k = [0, 0]
        self.segs: List[Optional[Seg]] = [None, None]
        self.stages: List[List[str]] = [["done"], ["done"]]
        self.plan()

    # ---- robot state ----------------------------------------------------------------------------
    @property
    def m(self):
        return self.env._mojo.physics.model._model

    @property
    def d(self):
        return self.env._mojo.physics.data._data

    def cmd_pose(self) -> np.ndarray:
        rb = self.env.robot
        return rb.forward_kinematics(rb.get_initial_joint_positions()).astype(np.float64)

    def pad_mid(self, i: int) -> np.ndarray:
        import mujoco
        hand = f"hand_{self.sides[i].value.lower()}/fingertip_pad_collision_"
        ids = [g for g in range(self.m.ngeom)
               if hand in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        assert len(ids) == 2, ids
        return self.d.geom_xpos[ids].mean(0).astype(np.float64)

    def pinch(self, i: int, ee=None) -> np.ndarray:
        """Commanded grasp point of arm i: fingertip-pad midpoint with the arm at its joint targets.
        Computed by kinematics, not as a fixed offset from the reported EE pose: the pads are not rigid
        in that frame (they also move with a joint outside it), so a fixed offset is only right near
        the reset yaw."""
        return self.cmd_pads()[i]

    def cmd_pads(self) -> List[np.ndarray]:
        import mujoco
        rb, phys = self.env.robot, self.env._mojo.physics
        joints = [a.joint for a in rb.limb_actuators]
        backup = [phys.bind(j).qpos.copy() for j in joints]
        for j, q in zip(joints, rb.get_initial_joint_positions()):
            phys.bind(j).qpos = q
        mujoco.mj_kinematics(self.m, self.d)
        out = [self.pad_mid(i) for i in range(2)]
        for j, q in zip(joints, backup):
            phys.bind(j).qpos = q
        mujoco.mj_kinematics(self.m, self.d)
        return out

    def holding(self, i: int, prop) -> bool:
        return self.env.robot.is_gripper_holding_object(prop, self.sides[i])

    def geoms_of(self, body_name_prefix: str) -> List[int]:
        """Geom ids whose body (or an ancestor) name starts with the prefix."""
        import mujoco
        out = []
        for g in range(self.m.ngeom):
            b = self.m.geom_bodyid[g]
            while b > 0:
                if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(body_name_prefix):
                    out.append(g)
                    break
                b = self.m.body_parentid[b]
        return out

    def geom_aabb(self, g: int):
        """World-frame axis-aligned bounds (lo, hi) of geom g (from its local AABB corners)."""
        c, h = self.m.geom_aabb[g][:3], self.m.geom_aabb[g][3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
        w = corners @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
        return w.min(0), w.max(0)

    def pick_yaw(self, i: int, axis: float, pref: float, yaw_max: float = np.deg2rad(95)) -> float:
        """Yaw making the fingers close along world angle `axis` (period pi), within +-yaw_max,
        preferring sign `pref` (human demos turn the left hand negative, the right positive)."""
        a = wrap(axis, np.pi)
        cands = [a, a - np.pi if a > 0 else a + np.pi]
        ok = [c for c in cands if abs(c) <= yaw_max]
        if ok:
            return float(max(ok, key=lambda c: (np.sign(c) == pref, -abs(c))))
        return float(np.clip(min(cands, key=abs), -yaw_max, yaw_max))

    def grasp_yaw(self, i: int, axis_yaw: float, period: float = np.pi) -> float:
        """Yaw that makes the fingers close along a world direction at angle `axis_yaw` (fingers close
        along world x at yaw 0); of the symmetric solutions, the one closest to the current yaw."""
        return self.yaw[i] + wrap(axis_yaw - self.yaw[i], period)

    # ---- to implement -----------------------------------------------------------------------------
    def plan(self):
        raise NotImplementedError

    def segment(self, i: int, stage: str) -> Seg:
        raise NotImplementedError

    def until(self, i: int, stage: str) -> bool:
        """Checked every step: True ends arm i's current segment early (e.g. a push that reached its goal)."""
        return False

    def advance(self, i: int, stage: str) -> Optional[int]:
        """Called when arm i finishes `stage`; return a stage index to jump to (e.g. retry), or None."""
        return None

    # ---- stepping ----------------------------------------------------------------------------------
    def stage(self, i: int) -> str:
        return self.stages[i][self.k[i]]

    def hold_seg(self, i: int, grip=None, steps=1) -> Seg:
        p = self.pinch(i)
        return Seg(p, p, self.yaw[i], self.yaw[i], self.grip[i] if grip is None else grip, self.tilt[i],
                   min_steps=1, hold=steps - 1)

    def _next_stage(self, i: int):
        st = self.stage(i)
        j = self.advance(i, st)
        self.k[i] = j if j is not None else min(self.k[i] + 1, len(self.stages[i]) - 1)

    def act(self) -> np.ndarray:
        a = np.zeros(14, np.float32)
        ee = self.cmd_pose()
        self._pads = self.cmd_pads()
        for i in range(2):
            if self.segs[i] is not None and not self.segs[i].done and self.until(i, self.stage(i)):
                self.segs[i].k = self.segs[i].n
            if self.segs[i] is not None and self.segs[i].done:
                self._next_stage(i)
                self.segs[i] = None
            st = self.stage(i)
            if st.startswith("sync:"):
                if self.stage(1 - i) == st or self._passed(1 - i, st):
                    self.k[i] += 1                     # both arms here: go on together
                    st = self.stage(i)
                    self.segs[i] = None
                else:                                  # wait in place this step
                    pos, yaw, tilt = self.pinch(i), self.yaw[i], self.tilt[i]
                    self._write(a, ee, i, pos, yaw, tilt)
                    continue
            if self.segs[i] is None:
                self.segs[i] = self.segment(i, st)
            pos, yaw, tilt = self.segs[i].next()
            self._write(a, ee, i, pos, yaw, tilt)
            self.grip[i] = self.segs[i].grip
        a[12:] = self.grip
        return a

    def _write(self, a, ee, i, pos, yaw, tilt):
        """Arm i's EE delta that moves the grasp point to `pos` with orientation (tilt, yaw)."""
        self.yaw[i], self.tilt[i] = yaw, tilt
        rot_t = R.from_rotvec(tilt) * R.from_rotvec([0, 0, yaw]) * self.R0[i]
        # translate by the grasp point's error; the shift a rotation in this step causes is corrected
        # on the next step (closed loop)
        a[6 * i: 6 * i + 3] = pos - self._pads[i]
        a[6 * i + 3: 6 * i + 6] = (rot_t * R.from_rotvec(ee[6 * i + 3: 6 * i + 6]).inv()).as_rotvec()

    def _passed(self, i: int, sync_stage: str) -> bool:
        """Arm i is at or beyond `sync_stage` in its own list."""
        return sync_stage in self.stages[i] and self.stages[i].index(sync_stage) < self.k[i]

    def perturb(self, a: np.ndarray) -> np.ndarray:
        """DART: noise on moving arms' deltas (never the grippers)."""
        if self.noise <= 0:
            return a
        e = a.copy()
        for i in range(2):
            if self.stage(i).startswith(self.QUIET) or self.stage(i).startswith("sync:"):
                continue
            e[6 * i: 6 * i + 3] += self.rng.normal(0, self.NOISE_POS * self.noise, 3)
            e[6 * i + 3: 6 * i + 6] += self.rng.normal(0, self.NOISE_ROT * self.noise, 3)
        return e

    @property
    def label(self) -> str:
        return "+".join(self.stage(i) for i in range(2))


def run_episode(env, family: str, expert_cls, seed: int, noise: float, recorder=None, debug=False):
    from roboeval.challenge import spec
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 7)
    max_steps = spec.max_steps(family, 20)
    zero = np.zeros(14, np.float32)
    if recorder is not None:                            # step 0: the no-op every episode starts with
        recorder.add(obs, zero)
    obs, _, term, trunc, info = env.step(zero)
    expert = expert_cls(env, rng, noise)
    steps, status, last = 1, "budget", None
    if env.success:
        status = "success"
    while status == "budget" and steps < max_steps:
        a = expert.act()
        if debug and expert.label != last:
            last = expert.label
            print(f"  step {steps}: {last}", flush=True)
        e = expert.perturb(a)
        if recorder is not None:
            recorder.add(obs, a, e)
        obs, _, term, trunc, info = env.step(e)
        steps += 1
        if env.success:
            status = "success"
        elif term:
            status = "fail"
        elif trunc:
            status = "unhealthy"
    return status, steps, info, expert


def cli(family: str, experts: Dict[str, type], default_seeds: str):
    """Generator CLI shared by the task scripts. `experts` maps variation -> expert class ('*' = any)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--variation", default="PositionAndOrientation")
    ap.add_argument("--seeds", default=default_seeds, help="a:b range (>= 100000)")
    ap.add_argument("--noise", type=float, default=1.0, help="DART noise scale (0 = off)")
    ap.add_argument("--dry", action="store_true", help="no cameras, nothing written: success rate + metrics")
    ap.add_argument("--debug", action="store_true", help="print stage changes")
    ap.add_argument("--out", default="data/gen/scripted")
    a = ap.parse_args()

    from roboeval.challenge import results as res
    from rdb.gen.episode_io import EpisodeRecorder, episode_path, make_env, variant_name

    s0, s1 = map(int, a.seeds.split(":"))
    assert s0 >= MIN_SEED, "generation seeds must be >= 100000 (never the eval/dev seeds)"
    cls = experts.get(a.variation, experts.get("*"))
    env = make_env(family, a.variation, render=not a.dry)
    vname = variant_name(family, a.variation)
    stats, mets, fails = {}, [], []
    t0 = time.time()
    for seed in range(s0, s1):
        path = episode_path(a.out, "", vname, seed)
        if not a.dry and path.exists():
            continue
        rec = None if a.dry else EpisodeRecorder()
        if a.debug:
            print(f"seed {seed}")
        try:
            status, steps, info, ex = run_episode(env, family, cls, seed, a.noise, rec, a.debug)
            where = ex.label
        except Exception as err:  # noqa: BLE001 - a broken episode is skipped
            status, steps, info, where = f"error:{type(err).__name__}:{err}"[:80], 0, {}, ""
            env.close()
            env = make_env(family, a.variation, render=not a.dry)
        stats[status] = stats.get(status, 0) + 1
        if status == "success":
            mets.append(res.episode_metrics(res.to_jsonable(info), steps, 20))
        else:
            fails.append((seed, status, where))
        if rec is not None and status == "success":
            rec.save(path, {"source": "scripted", "family": family, "variation": a.variation, "variant": vname,
                            "seed": seed, "success": True, "steps": steps, "noise": a.noise})
        if (seed - s0) % 25 == 24:
            print(f"{seed - s0 + 1} eps {time.time() - t0:.0f}s {stats}", flush=True)
    print("status", stats)
    print("fails", fails[:12])
    if mets:
        keys = mets[0].keys()
        print(json.dumps({k: round(float(np.median([m[k] for m in mets])), 4) for k in keys}))
