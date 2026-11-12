"""Scripted StackTwoBlocks expert that plans from simulator ground truth (block poses), for training data.

One arm (the one closest to a block) picks that block and puts it on the other, like the human
demos: approach above the block with the fingers aligned to its faces, descend, close, lift, carry
over the other block while matching its yaw, lower, release, retreat. Waypoints are joined with
minimum-jerk profiles and re-planned from the true poses at every segment start; each step's action
is the EE delta from the commanded pose (FK of the joint targets, what the EE-delta action mode
composes onto) to the next point of the profile, so the expert is closed loop.

DART noise (--noise, default on): the executed action gets Gaussian noise on the arm deltas, the
recorded label is the clean expert action at the visited state, so the data shows corrections.

Only the data generator reads privileged state; the recorded observations are the challenge's
standard cameras + proprioception (rdb.gen.episode_io). Seeds must be >= 100000 (never the
official 1000-1049 or our dev seeds).

  python -m rdb.gen.scripted_stack_two_blocks --seeds 100000:100050 --dry      # success/metrics, no rendering
  python -m rdb.gen.scripted_stack_two_blocks --seeds 100000:101500 --out data/gen/scripted
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

FAMILY = "StackTwoBlocks"
MIN_SEED = 100_000
CUBE = 0.04                     # edge length
V_AVG = 0.22                    # m/s mean speed of a min-jerk move (peak ~1.9x)
W_AVG = 1.6                     # rad/s mean yaw speed
DT = 1 / 20


def wrap(a: float, period: float) -> float:
    """Wrap to (-period/2, period/2]."""
    return -((-a + period / 2) % period - period / 2)


def yaw_of(quat_wxyz: np.ndarray) -> float:
    return R.from_quat(quat_wxyz, scalar_first=True).as_euler("xyz")[2]


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


class Segment:
    """Min-jerk move of one arm's pinch point and yaw (relative to its reset orientation)."""

    def __init__(self, p0, p1, y0, y1, grip, min_steps=6, hold=0):
        self.p0, self.p1, self.y0, self.y1, self.grip = np.asarray(p0), np.asarray(p1), y0, y1, grip
        dur = max(np.linalg.norm(self.p1 - self.p0) / V_AVG, abs(y1 - y0) / W_AVG)
        self.n = max(min_steps, int(np.ceil(dur / DT))) + hold
        self.move = self.n - hold
        self.k = 0

    def next(self):
        self.k += 1
        s = min_jerk(self.k / self.move)
        return self.p0 + (self.p1 - self.p0) * s, self.y0 + (self.y1 - self.y0) * s

    @property
    def done(self) -> bool:
        return self.k >= self.n


class StackExpert:
    def __init__(self, env, rng: np.random.Generator, noise: float = 1.0, mirror: float = 0.4):
        self.env, self.rng, self.noise, self.mirror = env, rng, noise, mirror
        rb = env.robot
        self.sides = list(rb.grippers.keys())           # [LEFT, RIGHT] = action blocks 0:6 / 6:12, grips 12 / 13
        self.R0 = [R.from_rotvec(self.cmd_pose()[6 * i + 3: 6 * i + 6]) for i in range(2)]
        # grasp point = midpoint of the two fingertip pads (the gripper's "pinch" site sits at the wrist),
        # relative to the commanded wrist pose, in the wrist frame (fixed for a rigid hand)
        ee = self.cmd_pose()
        self.off = [self.R0[i].inv().apply(self.pad_mid(i) - ee[6 * i: 6 * i + 3]) for i in range(2)]
        self.z0 = [ee[2], ee[8]]                        # reset wrist heights
        self.yaw = [0.0, 0.0]                           # commanded yaw offset per arm
        self.grip = [0.0, 0.0]
        self.plan()

    # ---- state -------------------------------------------------------------------------------------
    def cmd_pose(self) -> np.ndarray:
        rb = self.env.robot
        return rb.forward_kinematics(rb.get_initial_joint_positions()).astype(np.float64)

    def pad_mid(self, i: int) -> np.ndarray:
        import mujoco
        m, d = self.env._mojo.physics.model._model, self.env._mojo.physics.data._data
        hand = f"hand_{self.sides[i].value.lower()}/fingertip_pad_collision_"
        ids = [g for g in range(m.ngeom)
               if hand in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        assert len(ids) == 2, ids
        return d.geom_xpos[ids].mean(0).astype(np.float64)

    def pinch_cmd(self, i: int, ee=None) -> np.ndarray:
        ee = self.cmd_pose() if ee is None else ee
        return ee[6 * i: 6 * i + 3] + R.from_rotvec(ee[6 * i + 3: 6 * i + 6]).apply(self.off[i])

    def blocks(self):
        return [(b.body.get_position().copy(), yaw_of(b.body.get_quaternion())) for b in self.env.blocks]

    def holding(self, i: int, b: int) -> bool:
        return self.env.robot.is_gripper_holding_object(self.env.blocks[b], self.sides[i])

    # ---- plan --------------------------------------------------------------------------------------
    def plan(self):
        blk = self.blocks()
        best = None
        for i in range(2):
            p = self.pinch_cmd(i)
            for a in range(2):
                c = np.linalg.norm(p[:2] - blk[a][0][:2])
                if best is None or c < best[0]:
                    best = (c, i, a)
        _, self.arm, self.pick = best
        self.base = 1 - self.pick
        self.stage = "approach"
        self.seg = None

    def _grasp_yaw(self, block_yaw: float, i: int | None = None, pick=None, other=None) -> float:
        """Fingers close along world x at yaw 0, so any face normal (90 deg symmetry) will do; when the
        other block is close, take the pair of faces that keeps the fingers out of the gap."""
        y = self.yaw[self.arm if i is None else i]
        c = y + wrap(block_yaw - y, np.pi / 2)
        if pick is not None and np.linalg.norm((other - pick)[:2]) < 0.12:
            d = (other - pick)[:2] / np.linalg.norm((other - pick)[:2])
            cands = [c, c + np.pi / 2, c - np.pi / 2]
            cands = [x for x in cands if abs(x - y) <= np.pi * 0.75] or [c]
            c = min(cands, key=lambda x: (round(abs(np.cos(x) * d[0] + np.sin(x) * d[1]), 2), abs(x - y)))
        return c

    def _new_segment(self):
        i, blk = self.arm, self.blocks()
        p = self.pinch_cmd(i)
        (pa, ya), (pb, yb) = blk[self.pick], blk[self.base]
        y = self.yaw[i]
        st = self.stage
        if st == "approach":
            tgt = pa + [0, 0, 0.055]
            self.seg = Segment(p, tgt, y, self._grasp_yaw(ya, pick=pa, other=pb), 0.0)
        elif st == "descend":
            self.seg = Segment(p, pa + [0, 0, 0.006], y, self._grasp_yaw(ya, pick=pa, other=pb), 0.0, min_steps=6)
        elif st == "close":
            self.seg = Segment(p, p, y, y, 1.0, min_steps=1, hold=6)
        elif st == "lift":
            self.seg = Segment(p, p + [0, 0, max(0.05, pb[2] + CUBE + 0.045 - p[2])], y, y, 1.0, min_steps=5)
        elif st == "carry":
            # move so the carried block's centre ends above the base block, block faces aligned
            tgt = p + (pb + [0, 0, CUBE + 0.03] - pa)
            dy = wrap(yb - ya, np.pi / 2)
            self.seg = Segment(p, tgt, y, y + dy, 1.0)
        elif st == "lower":
            self.seg = Segment(p, p + (pb + [0, 0, CUBE + 0.003] - pa), y, y, 1.0, min_steps=5)
        elif st == "release":
            self.seg = Segment(p, p, y, y, 0.0, min_steps=1, hold=4)
        elif st == "retreat":
            self.seg = Segment(p, p + [0, 0, 0.06], y, y, 0.0, min_steps=6)
        elif st == "done":
            self.seg = Segment(p, p, y, y, 0.0, min_steps=1, hold=1000)

    NEXT = {"approach": "descend", "descend": "close", "close": "lift", "lift": "carry", "carry": "lower",
            "lower": "release", "release": "retreat", "retreat": "done", "done": "done"}

    def act(self) -> np.ndarray:
        """Clean expert action for the current state."""
        if self.seg is None or self.seg.done:
            if self.seg is not None:
                if self.stage == "close" and not self.holding(self.arm, self.pick):
                    self.stage = "regrasp"
                elif self.stage == "lower" and not self.holding(self.arm, self.pick):
                    self.stage = "release"          # slipped onto the stack early: let go
                else:
                    self.stage = self.NEXT[self.stage]
            if self.stage == "regrasp":            # missed: open, go back up, try again
                self.grip[self.arm] = 0.0
                self.stage = "approach"
            self._new_segment()
        pos, yaw = self.seg.next()
        self.yaw[self.arm] = yaw
        i, ee = self.arm, self.cmd_pose()
        a = np.zeros(14, np.float32)
        rot_t = R.from_rotvec([0, 0, yaw]) * self.R0[i]
        site_t = pos - rot_t.apply(self.off[i])
        a[6 * i: 6 * i + 3] = site_t - ee[6 * i: 6 * i + 3]
        a[6 * i + 3: 6 * i + 6] = (rot_t * R.from_rotvec(ee[6 * i + 3: 6 * i + 6]).inv()).as_rotvec()
        # the idle arm follows a fraction of the active wrist's height change (lower wrist-height
        # difference and arm-speed difference, the two coordination metrics)
        j = 1 - i
        a[6 * j + 2] = self.z0[j] + self.mirror * (site_t[2] - self.z0[i]) - ee[6 * j + 2]
        self.grip[i] = self.seg.grip
        a[12:] = self.grip
        return a

    def perturb(self, a: np.ndarray) -> np.ndarray:
        """DART: noise on the moving arm's delta (not the grippers), off while the gripper closes/opens."""
        if self.noise <= 0 or self.stage in ("close", "release", "done"):
            return a
        e = a.copy()
        i = self.arm
        e[6 * i: 6 * i + 3] += self.rng.normal(0, 0.0025 * self.noise, 3)
        e[6 * i + 3: 6 * i + 6] += self.rng.normal(0, 0.015 * self.noise, 3)
        return e


class TwoArmStackExpert(StackExpert):
    """Blocks too far apart for one arm (the Static / Orientation variations put them at y = +-0.35):
    both arms grasp their own block at once; the "giver" sets its block down in the middle and backs
    off, the "stacker" waits beside the middle with its block lifted, then stacks it."""

    REACH = 0.45                                        # max |y| travel of one arm from its home

    @staticmethod
    def needed(env) -> bool:
        ys = sorted(b.body.get_position()[1] for b in env.blocks)
        return ys[1] - ys[0] > TwoArmStackExpert.REACH + 0.1

    def plan(self):
        blk = self.blocks()
        left_block = int(np.argmax([b[0][1] for b in blk]))   # LEFT arm is at +y
        self.own = [left_block, 1 - left_block]
        self.giver = int(self.rng.integers(2))
        self.stacker = 1 - self.giver
        sign = [1.0, -1.0]                                    # +y side of each arm
        pg, ps = blk[self.own[self.giver]][0], blk[self.own[self.stacker]][0]
        self.mid = np.array([(pg[0] + ps[0]) / 2, 0.04 * sign[self.giver], pg[2]])
        self.hover_y = 0.12 * sign[self.stacker]
        self.stages = [["approach", "descend", "close", "lift", "carry", "lower", "release", "retreat", "done"]] * 2
        self.stages = [list(s) for s in self.stages]
        self.stages[self.stacker] = ["approach", "descend", "close", "lift", "hover", "wait", "over", "lower",
                                     "release", "retreat", "done"]
        self.k = [0, 0]
        self.segs = [None, None]
        self.stage = "approach"

    def _seg(self, i: int):
        st = self.stages[i][self.k[i]]
        blk = self.blocks()
        p = self.pinch_cmd(i)
        pa, ya = blk[self.own[i]]
        y = self.yaw[i]
        gy = y + wrap(ya - y, np.pi / 2)
        base = blk[self.own[self.giver]][0]
        if st == "approach":
            return Segment(p, pa + [0, 0, 0.055], y, gy, 0.0)
        if st == "descend":
            return Segment(p, pa + [0, 0, 0.006], y, gy, 0.0)
        if st == "close":
            return Segment(p, p, y, y, 1.0, min_steps=1, hold=6)
        if st == "lift":
            h = 0.05 if i == self.giver else CUBE + 0.07
            return Segment(p, p + [0, 0, h], y, y, 1.0, min_steps=5)
        if st == "carry":                                     # giver: block centre above the middle spot
            return Segment(p, p + (self.mid + [0, 0, 0.04] - pa), y, y, 1.0)
        if st == "lower" and i == self.giver:
            return Segment(p, p + (self.mid + [0, 0, 0.003] - pa), y, y, 1.0, min_steps=5)
        if st == "retreat" and i == self.giver:
            home = 0.3 if self.sides[i].value == "Left" else -0.3
            return Segment(p, p + [0, (home - p[1]) * 0.5, 0.06], y, y, 0.0, min_steps=6)
        if st == "hover":                                     # stacker: lifted block beside the middle
            tgt = np.array([self.mid[0], self.hover_y, base[2] + CUBE + 0.07])
            return Segment(p, p + (tgt - pa), y, y, 1.0)
        if st == "wait":
            return Segment(p, p, y, y, 1.0, min_steps=1)
        if st == "over":
            dy = wrap(blk[self.own[self.giver]][1] - ya, np.pi / 2)
            return Segment(p, p + (base + [0, 0, CUBE + 0.03] - pa), y, y + dy, 1.0)
        if st == "lower":
            return Segment(p, p + (base + [0, 0, CUBE + 0.003] - pa), y, y, 1.0, min_steps=5)
        if st == "release":
            return Segment(p, p, y, y, 0.0, min_steps=1, hold=4)
        if st == "retreat":
            return Segment(p, p + [0, 0, 0.06], y, y, 0.0, min_steps=6)
        return Segment(p, p, y, y, 0.0, min_steps=1, hold=1000)   # done

    def _advance(self, i: int):
        st = self.stages[i][self.k[i]]
        if st == "close" and not self.holding(i, self.own[i]):
            self.k[i] = 0                                     # missed: open and re-approach
            self.grip[i] = 0.0
            return
        if st == "wait":
            g = self.stages[self.giver][self.k[self.giver]]
            if g != "done":
                return                                        # keep waiting
        if st == "lower" and not self.holding(i, self.own[i]):
            self.k[i] = self.stages[i].index("release")
            return
        self.k[i] = min(self.k[i] + 1, len(self.stages[i]) - 1)

    def act(self) -> np.ndarray:
        a = np.zeros(14, np.float32)
        ee = self.cmd_pose()
        for i in range(2):
            if self.segs[i] is not None and self.segs[i].done:
                self._advance(i)
                self.segs[i] = None
            if self.segs[i] is None:
                self.segs[i] = self._seg(i)
            pos, yaw = self.segs[i].next()
            self.yaw[i] = yaw
            rot_t = R.from_rotvec([0, 0, yaw]) * self.R0[i]
            a[6 * i: 6 * i + 3] = pos - rot_t.apply(self.off[i]) - ee[6 * i: 6 * i + 3]
            a[6 * i + 3: 6 * i + 6] = (rot_t * R.from_rotvec(ee[6 * i + 3: 6 * i + 6]).inv()).as_rotvec()
            self.grip[i] = self.segs[i].grip
        a[12:] = self.grip
        self.stage = "+".join(self.stages[i][self.k[i]] for i in range(2))
        return a

    def perturb(self, a: np.ndarray) -> np.ndarray:
        if self.noise <= 0:
            return a
        e = a.copy()
        for i in range(2):
            if self.stages[i][self.k[i]] in ("close", "release", "done", "wait"):
                continue
            e[6 * i: 6 * i + 3] += self.rng.normal(0, 0.0025 * self.noise, 3)
            e[6 * i + 3: 6 * i + 6] += self.rng.normal(0, 0.015 * self.noise, 3)
        return e


def run_episode(env, seed: int, noise: float, recorder=None, max_steps: int | None = None, mirror: float = 0.4):
    from roboeval.challenge import spec
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 7)
    max_steps = max_steps or spec.max_steps(FAMILY, 20)
    zero = np.zeros(14, np.float32)
    # step 0: the no-op every episode starts with (as the human demos and our deployed policy)
    if recorder is not None:
        recorder.add(obs, zero)
    obs, _, term, trunc, info = env.step(zero)
    expert = (TwoArmStackExpert(env, rng, noise) if TwoArmStackExpert.needed(env)
              else StackExpert(env, rng, noise, mirror))
    steps, status = 1, "budget"
    if env.success:
        status = "success"
    while status == "budget" and steps < max_steps:
        a = expert.act()
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
    return status, steps, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variation", default="PositionAndOrientation", help="PositionAndOrientation|Position|Orientation|Static")
    ap.add_argument("--seeds", default="100000:100050", help="a:b range")
    ap.add_argument("--noise", type=float, default=1.0, help="DART noise scale (0 = off)")
    ap.add_argument("--mirror", type=float, default=0.4, help="idle arm follows this fraction of the active wrist height")
    ap.add_argument("--dry", action="store_true", help="no cameras, nothing written: success rate + metrics")
    ap.add_argument("--out", default="data/gen/scripted", help="writes <out>/<Variant>/ep_<seed>.npz")
    ap.add_argument("--keep_failed", action="store_true")
    a = ap.parse_args()

    from roboeval.challenge import results as res
    from rdb.gen.episode_io import EpisodeRecorder, episode_path, make_env, variant_name

    s0, s1 = map(int, a.seeds.split(":"))
    assert s0 >= MIN_SEED, "generation seeds must be >= 100000 (never the eval/dev seeds)"
    env = make_env(FAMILY, a.variation, render=not a.dry)
    vname = variant_name(FAMILY, a.variation)
    stats, mets = {}, []
    t0 = time.time()
    for seed in range(s0, s1):
        path = episode_path(a.out, "", vname, seed)
        if not a.dry and path.exists():
            continue
        rec = None if a.dry else EpisodeRecorder()
        try:
            status, steps, info = run_episode(env, seed, a.noise, rec, mirror=a.mirror)
        except Exception as ex:  # noqa: BLE001 - a broken episode is just skipped
            status, steps, info = f"error:{type(ex).__name__}", 0, {}
            env.close()
            env = make_env(FAMILY, a.variation, render=not a.dry)
        stats[status] = stats.get(status, 0) + 1
        if status == "success":
            mets.append(res.episode_metrics(res.to_jsonable(info), steps, 20))
        if rec is not None and (status == "success" or a.keep_failed):
            rec.save(path, {"source": "scripted", "family": FAMILY, "variation": a.variation, "variant": vname, "seed": seed,
                            "success": status == "success", "steps": steps, "noise": a.noise, "mirror": a.mirror})
        if (seed - s0) % 25 == 24:
            print(f"{seed - s0 + 1} eps {time.time() - t0:.0f}s {stats}", flush=True)
    print("status", stats)
    if mets:
        keys = mets[0].keys()
        print(json.dumps({k: round(float(np.median([m[k] for m in mets])), 4) for k in keys}))


if __name__ == "__main__":
    main()
