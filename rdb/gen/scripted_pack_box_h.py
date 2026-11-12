"""Closed-loop, open-hand PackBox expert. Privileged state: training data only.

    MUJOCO_GL=egl python -m rdb.gen.scripted_pack_box_h \
        --seeds 105200:105260 --dry

Human demonstrations keep both grippers open, engage the flaps from outside and
below, then fold inward. Stagger the final closure to avoid a symmetric tent.
"""
from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation

from rdb.gen.scripted_common import Seg, run_episode
from rdb.gen.scripted_pack_box import PackBoxExpert


def thermal_guard():
    """Pause our own simulation while the CPU is above 87 C (needs lm-sensors; does nothing without it)."""
    waiting = False
    while True:
        try:
            readings = json.loads(subprocess.check_output(["sensors", "-j"], text=True, stderr=subprocess.DEVNULL))
        except (OSError, subprocess.CalledProcessError, ValueError):
            return
        values = [v for chip in readings.values() for sensor in chip.values() if isinstance(sensor, dict)
                  for key, v in sensor.items() if key == "temp1_input"]
        if not values:
            return
        temperature = max(values)
        if temperature <= (84 if waiting else 87):
            return
        if not waiting:
            print(f"thermal backoff: {temperature:.1f} C; waiting for <=84 C", flush=True)
        waiting = True
        time.sleep(5)


class FollowFlap(Seg):
    """Track the measured flap with a bounded tangential pushing lead."""
    def __init__(self, expert, arm, floor=0.0):
        self.expert, self.arm, self.floor = expert, arm, floor
        p, y = expert.pinch(arm), expert.yaw[arm]
        super().__init__(p, p, y, y, 0.0, min_steps=1, hold=180)
        self.last = p.copy()
        self.velocity = np.zeros(3)
        self.history = []
        self.stalled = False

    def next(self):
        self.k += 1
        e = self.expert
        a, axis, q, lo, u = e.flap(self.arm)
        remaining = q - lo
        self.history.append(float(remaining))
        floor = self.floor
        if self.arm == 1 and e.k[0] < e.stages[0].index("watch"):
            floor = e.GATE
        goal = max(floor, remaining - e.LEAD)
        if len(self.history) >= e.STALL_STEPS and remaining > max(1.6, floor + .2):
            self.stalled = self.history[-e.STALL_STEPS] - remaining < .035
        radius = e.R
        direction = Rotation.from_rotvec(-axis * (remaining - goal)).apply(u)
        target = a + radius * direction - np.array([0., 0., e.PAD_DZ])
        # A hand-sized pushing offset keeps contact without hard-coding a trajectory.
        target -= np.array([0., 0., e.PRESS_DZ]) if remaining < .5 else 0.
        delta = target - self.last
        delta *= min(1., e.STEP / max(np.linalg.norm(delta), 1e-9))
        self.velocity = .65 * self.velocity + .35 * delta
        self.last += self.velocity
        yaw = (-.30 if self.arm == 0 else .30) * np.clip((2.8 - remaining) / 2.8, 0., 1.)
        return self.last.copy(), float(yaw), self.t0


class PackBoxHumanExpert(PackBoxExpert):
    R = .185
    BEHIND = .025
    LEAD = .30
    STEP = .006
    PRESS_DZ = .012
    PAD_DZ = .085
    CLOSED = .21
    GATE = 1.15
    STALL_STEPS = 24
    QUIET = PackBoxExpert.QUIET + ("watch",)

    def flap(self, i):
        a, axis, q, lo, u = super().flap(i)
        gid = self.flaps[self.mine[i]][1]
        centre = self.d.geom_xpos[gid] + self.d.geom_xmat[gid].reshape(3, 3) @ self.m.geom_aabb[gid][:3]
        # The XML hinge anchors are at arbitrary corners, not the panel midpoint.
        # Shift along the same hinge line to push centrally and keep the arm reachable.
        mat = self.d.geom_xmat[gid].reshape(3, 3)
        half = self.m.geom_aabb[gid][3:]
        width = np.abs(axis @ mat) @ half
        offset = (centre - a) @ axis
        preferred = (.65 - a[0]) / axis[0]
        offset = np.clip(preferred, offset - width + .045, offset + width - .045)
        a = a + axis * offset
        return a, axis, q, lo, u

    def plan(self):
        super().plan()
        self.stages = [["outside", "under", "engage", "fold", "clear", "watch", "done"] for _ in range(2)]
        self.tick = 0
        self.retries = [0, 0]
        self.trace = []

    def act(self):
        if self.tick % 20 == 0:
            thermal_guard()
        self.trace.append({"step": self.tick + 1, "angles": [float(self.flap(i)[2] - self.flap(i)[3]) for i in (0, 1)],
                           "stages": [self.stage(i) for i in (0, 1)]})
        self.tick += 1
        return super().act()

    def segment(self, i, stage):
        p, y = self.pinch(i), self.yaw[i]
        a, axis, q, lo, u = self.flap(i)
        start, v = self.start_point(i)
        if stage == "outside":
            beyond = a + u * (self.length(i) + .07)
            return Seg(p, [beyond[0], beyond[1], max(p[2], beyond[2] + .07)], y, y, 0., min_steps=8, speed=1.2)
        if stage == "under":
            # Descend outside the flap, then engage its underside with open fingers.
            beyond = a + u * (self.length(i) + .07)
            return Seg(p, beyond - v * .05 - [0., 0., self.PAD_DZ], y, y, 0., min_steps=8, speed=1.)
        if stage == "engage":
            return Seg(p, start - [0., 0., self.PAD_DZ], y, y, 0., min_steps=8, speed=1.)
        if stage == "fold":
            return FollowFlap(self, i)
        if stage == "clear":
            return Seg(p, p + np.array([0., 0., .12]), y, y, 0., min_steps=10, speed=1.)
        return self.hold_seg(i, 0., steps=10 if stage == "watch" else 1000)

    def until(self, i, stage):
        return stage == "fold" and (self.flap(i)[2] - self.flap(i)[3] < self.CLOSED or self.segs[i].stalled)

    def advance(self, i, stage):
        if stage == "fold" and self.flap(i)[2] - self.flap(i)[3] >= self.CLOSED:
            self.retries[i] += 1
            return 0
        if stage == "watch":
            if self.flap(i)[2] - self.flap(i)[3] >= .26:
                self.retries[i] += 1
                return 0
            return self.stages[i].index("watch")
        return None


def main():
    """Shared runner/storage conventions, with an audit trail and human-style noise."""
    from rdb.gen.episode_io import EpisodeRecorder, episode_path, make_env, variant_name
    from roboeval.challenge import results as res
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variation", default="PositionAndOrientation")
    ap.add_argument("--seeds", default="105200:105260")
    ap.add_argument("--noise", type=float, default=.25, help="DART scale (human-style default: 0.25)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out", default="data/gen/scripted_h")
    ap.add_argument("--stats", type=Path)
    args = ap.parse_args()
    start, stop = map(int, args.seeds.split(":"))
    if start < 100000 or stop <= start or not np.isfinite(args.noise) or args.noise < 0:
        ap.error("Require a nonempty generation/search range >=100000 and finite nonnegative noise")
    if args.variation != "PositionAndOrientation":
        ap.error("This expert is validated only for PositionAndOrientation")
    variant = variant_name("PackBox", args.variation)
    stats_path = args.stats or Path(f"runs/packbox/{'dry' if args.dry else 'generation'}_{start}_{stop}.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records, skipped = [], []
    thermal_guard()
    env = make_env("PackBox", args.variation, render=not args.dry)
    started = time.time()

    def save_stats():
        successes = [r for r in records if r["status"] == "success"]
        durations = [r["steps"] / 20 for r in successes]
        summary = {"seeds": [start, stop], "noise": args.noise, "dry": args.dry, "script_sha256": code_sha,
                   "attempted": len(records), "skipped": skipped, "successes": len(successes),
                   "success_rate": len(successes) / len(records) if records else None,
                   "success_time_s": {"median": float(np.median(durations)), "mean": float(np.mean(durations)),
                       "p90": float(np.percentile(durations, 90))} if durations else {},
                   "wall_s": time.time() - started, "episodes": records}
        tmp = stats_path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(summary, indent=2)); tmp.replace(stats_path)
        return summary

    try:
        for seed in range(start, stop):
            path = episode_path(args.out, "", variant, seed)
            if not args.dry and path.exists():
                with np.load(path) as saved:
                    meta = json.loads(str(saved["meta"]))
                if (meta.get("script_sha256") != code_sha or meta.get("seed") != seed or
                        meta.get("noise") != args.noise or meta.get("source") != "scripted_h"):
                    raise ValueError(f"Existing episode has different provenance: {path}")
                skipped.append(seed)
                continue
            thermal_guard()
            rec = None if args.dry else EpisodeRecorder()
            try:
                status, steps, info, expert = run_episode(env, "PackBox", PackBoxHumanExpert, seed, args.noise, rec, args.debug)
                metrics = res.episode_metrics(res.to_jsonable(info), steps, 20)
                record = {"seed": seed, "status": status, "steps": steps, "metrics": metrics,
                          "retries": expert.retries, "trace": expert.trace}
                if rec is not None and status == "success":
                    rec.save(path, {"source": "scripted_h", "expert": "PackBoxHumanExpert", "family": "PackBox",
                        "variation": args.variation, "variant": variant, "seed": seed, "success": True,
                        "steps": steps, "noise": args.noise, "script_sha256": code_sha, "retries": expert.retries})
            except Exception as exc:
                record = {"seed": seed, "status": "error", "steps": 0, "error": repr(exc)}
                env.close()
                thermal_guard()
                env = make_env("PackBox", args.variation, render=not args.dry)
            records.append(record)
            if len(records) % 25 == 0:
                summary = save_stats()
                print(f"{len(records)} attempted: {summary['successes']} successes; {summary['success_time_s']}; "
                      f"{summary['wall_s']:.0f}s wall", flush=True)
    finally:
        env.close()
        summary = save_stats()
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    thermal_guard()
    main()
