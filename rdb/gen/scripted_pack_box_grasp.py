"""Human-style PackBox teacher with brief flap grasps before the open-hand fold.

Only the generator reads privileged flap geometry and contact state. Recorded
observations use the challenge interface.

  MUJOCO_GL=egl python -m rdb.gen.scripted_pack_box_grasp \
      --seeds 105200:105260 --dry
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rdb.gen.scripted_common import Seg, run_episode
from rdb.gen.scripted_pack_box_h import PackBoxHumanExpert, thermal_guard


class TrackGrasp(Seg):
    """Track a moving flap with a partly closed gripper until both pads touch."""

    def __init__(self, expert, arm):
        self.expert, self.arm = expert, arm
        p, y = expert.pinch(arm), expert.yaw[arm]
        super().__init__(p, p, y, y, expert.GRASP_GRIP, t0=expert.tilt[arm], min_steps=1, hold=70)
        self.last = p.copy()

    def next(self):
        self.k += 1
        a, axis, q, lo, u = self.expert.flap(self.arm)
        a = a + axis * (self.expert.contact_bias[self.arm] / axis[0])
        if self.expert.GRASP_CLOSE_ANGLE is not None:
            self.grip = float(q - lo <= self.expert.GRASP_CLOSE_ANGLE)
        if self.expert.GRASP_X is not None:
            a = a + axis * ((self.expert.GRASP_X - a[0]) / axis[0])
        normal = np.cross(-axis, u)
        lead = (.02 if 1 <= self.expert.search_steps[self.arm] <= 20 else
                self.expert.GRASP_LEAD + self.expert.normal_bias[self.arm])
        target = a + u * self.expert.GRASP_RADIUS + normal * lead
        delta = target - self.last
        self.last += delta * min(1., self.expert.GRASP_STEP / max(np.linalg.norm(delta), 1e-9))
        if self.expert.GRASP_YAW == "hinge":
            goal_yaw = self.expert.grasp_yaw(self.arm, np.arctan2(axis[1], axis[0]))
        else:
            goal_yaw = (self.expert.GRASP_YAW[self.arm] if isinstance(self.expert.GRASP_YAW, tuple)
                        else self.expert.GRASP_YAW)
        yaw = self.y0 + np.clip(goal_yaw - self.y0, -.04 * self.k, .04 * self.k)
        tilt = self.t0 + np.array([0., np.clip(self.expert.GRASP_TILT_Y - self.t0[1],
                                               -.04 * self.k, .04 * self.k), 0.])
        return self.last.copy(), yaw, tilt


class PackBoxGraspExpert(PackBoxHumanExpert):
    """Touch each flap with both finger pads, release, then push it closed."""

    QUIET = PackBoxHumanExpert.QUIET + ("grasp", "release")
    GRASP_GRIP = 1.
    GRASP_LEAD = .05
    GRASP_STEP = .018
    GRASP_X = None
    GRASP_RADIUS = .185
    GRASP_YAW = "hinge"
    GRASP_TILT_Y = 0.
    GRASP_CLOSE_ANGLE = None

    def plan(self):
        super().plan()
        self.stages = [["outside", "under", "engage", "grasp", "release", "reset",
                        "fold", "clear", "watch", "done"] for _ in range(2)]
        self.ever_held = [False, False]
        self.contact_bias = [0., 0.]
        self.normal_bias = [0., 0.]
        self.search_steps = [0, 0]
        physics = self.env._mojo.physics
        self.pad_geom_ids = []
        for side in self.sides:
            bodies = self.env.robot.grippers[side]._pad_bodies
            self.pad_geom_ids.append([
                set(physics.bind([g.mjcf for g in body.geoms if g.is_collidable()]).element_id)
                for body in bodies
            ])

    def target_flap(self, i):
        return self.box.flap_2 if i == 0 else self.box.flap_1

    def act(self):
        held = [bool(self.holding(i, self.target_flap(i))) for i in range(2)]
        self.ever_held = [x or y for x, y in zip(self.ever_held, held)]
        for i in range(2):
            if self.stage(i) != "grasp":
                continue
            flap_id = self.flaps[self.mine[i]][1]
            hits = [any(c.dist <= 1e-8 and
                        ((c.geom1 == flap_id and c.geom2 in ids) or
                         (c.geom2 == flap_id and c.geom1 in ids))
                        for c in self.d.contact) for ids in self.pad_geom_ids[i]]
            if hits[0] != hits[1] and self.search_steps[i] == 0:
                self.search_steps[i] = 1
            elif 1 <= self.search_steps[i] <= 20:
                self.search_steps[i] += 1
            if not any(hits) and self.flap(i)[2] - self.flap(i)[3] < 1.2:
                self.normal_bias[i] = min(.018, self.normal_bias[i] + .003)
            if hits[0] != hits[1]:
                x0 = self.d.geom_xpos[list(self.pad_geom_ids[i][0]), 0].mean()
                x1 = self.d.geom_xpos[list(self.pad_geom_ids[i][1]), 0].mean()
                direction = np.sign(x0 - x1) * (-1. if hits[0] else 1.)
                self.contact_bias[i] = float(np.clip(self.contact_bias[i] + .004 * direction, -.025, .025))
        action = super().act()
        self.trace[-1]["held"] = held
        self.trace[-1]["contact_bias"] = self.contact_bias.copy()
        self.trace[-1]["normal_bias"] = self.normal_bias.copy()
        self.trace[-1]["search_steps"] = self.search_steps.copy()
        return action

    def segment(self, i, stage):
        if stage == "grasp":
            self.search_steps[i] = 0
            return TrackGrasp(self, i)
        if stage in ("release", "reset"):
            p, y = self.pinch(i), self.yaw[i]
            a, axis, q, lo, u = self.flap(i)
            point = a + u * self.R
            if stage == "release":
                return Seg(p, p, y, y, 0., t0=self.tilt[i], min_steps=1, hold=4)
            return Seg(p, point - [0., 0., self.PAD_DZ], y, 0., 0.,
                       t0=self.tilt[i], t1=np.zeros(3), min_steps=8)
        return super().segment(i, stage)

    def until(self, i, stage):
        if stage == "grasp":
            return self.holding(i, self.target_flap(i))
        return stage == "fold" and super().until(i, stage)

    def advance(self, i, stage):
        if stage in ("fold", "watch"):
            return super().advance(i, stage)
        return None


def main():
    """Run search or data generation with a resumable per-episode audit."""
    from rdb.gen.episode_io import EpisodeRecorder, episode_path, make_env, variant_name
    from roboeval.challenge import results as res
    from rdb.gen import scripted_common, scripted_pack_box_h, episode_io

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variation", default="PositionAndOrientation", choices=["PositionAndOrientation"])
    ap.add_argument("--seeds", default="105200:105260")
    ap.add_argument("--noise", type=float, default=.25)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out", default="data/gen/scripted_hg")
    ap.add_argument("--stats", type=Path)
    args = ap.parse_args()
    start, stop = map(int, args.seeds.split(":"))
    if start < 100000 or stop <= start or not np.isfinite(args.noise) or args.noise < 0:
        ap.error("Require a nonempty seed range >=100000 and finite nonnegative noise")

    deps = [Path(__file__), Path(scripted_pack_box_h.__file__), Path(scripted_common.__file__),
            Path(episode_io.__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}
    code_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    variant = variant_name("PackBox", args.variation)
    stats_path = args.stats or Path(f"runs/packbox_grasp/{'dry' if args.dry else 'generation'}_{start}_{stop}.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    prior_wall = 0.
    if stats_path.exists():
        prior = json.loads(stats_path.read_text())
        if (prior["script_sha256"] != code_sha or prior["seeds"] != [start, stop] or
                prior["noise"] != args.noise or prior["dry"] != args.dry):
            raise ValueError(f"Incompatible audit file; choose another --stats path: {stats_path}")
        records = {r["seed"]: r for r in prior["episodes"]}
        prior_wall = prior["wall_s"]
    started = time.time()

    def save():
        rows = list(records.values())
        successes = [r for r in rows if r["status"] == "success"]
        duration = [r["steps"] / 20 for r in successes]
        summary = {"seeds": [start, stop], "noise": args.noise, "dry": args.dry,
                   "script_sha256": code_sha, "dependencies": hashes, "attempted": len(rows),
                   "successes": len(successes), "success_rate": len(successes) / len(rows) if rows else None,
                   "success_time_s": {"mean": float(np.mean(duration)), "median": float(np.median(duration)),
                                      "p90": float(np.percentile(duration, 90))} if duration else {},
                   "wall_s": prior_wall + time.time() - started, "episodes": rows}
        tmp = stats_path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(stats_path)
        return summary

    thermal_guard()
    env = make_env("PackBox", args.variation, render=not args.dry)
    try:
        for seed in range(start, stop):
            path = episode_path(args.out, "", variant, seed)
            if not args.dry and path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    meta = json.loads(str(saved["meta"]))
                expected = {"script_sha256": code_sha, "seed": seed, "noise": args.noise,
                            "source": "scripted_hg", "family": "PackBox", "success": True}
                if any(meta.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Existing episode has incompatible provenance: {path}")
                if seed not in records:
                    raise ValueError(f"Existing episode missing audit record: {path}")
                continue
            if seed in records and (args.dry or records[seed]["status"] not in ("success", "error")):
                continue
            thermal_guard()
            rec = None if args.dry else EpisodeRecorder()
            try:
                status, steps, info, expert = run_episode(env, "PackBox", PackBoxGraspExpert,
                                                          seed, args.noise, rec, args.debug)
                expert.ever_held = [old or expert.holding(i, expert.target_flap(i))
                                    for i, old in enumerate(expert.ever_held)]
                record = {"seed": seed, "status": status, "steps": steps,
                          "metrics": res.episode_metrics(res.to_jsonable(info), steps, 20),
                          "held": expert.ever_held, "retries": expert.retries, "trace": expert.trace}
                if rec is not None and status == "success":
                    rec.save(path, {"source": "scripted_hg", "expert": "PackBoxGraspExpert",
                                    "family": "PackBox", "variation": args.variation, "variant": variant,
                                    "seed": seed, "success": True, "steps": steps, "noise": args.noise,
                                    "script_sha256": code_sha, "retries": expert.retries,
                                    "held": expert.ever_held})
            except Exception as exc:
                record = {"seed": seed, "status": "error", "steps": 0, "error": repr(exc)}
                print(record, flush=True)
                env.close()
                thermal_guard()
                env = make_env("PackBox", args.variation, render=not args.dry)
            records[seed] = record
            summary = save()
            print(f"seed {seed}: {record['status']} {record['steps']/20:.2f}s; "
                  f"{summary['successes']}/{summary['attempted']} successes; "
                  f"{summary['wall_s']:.0f}s wall", flush=True)
    finally:
        env.close()
        summary = save()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("episodes", "dependencies")}), flush=True)


if __name__ == "__main__":
    thermal_guard()
    main()
