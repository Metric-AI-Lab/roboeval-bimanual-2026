"""CPU-only book pose feasibility screen; never a submission policy.

Run one process per --task, with
OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1, MUJOCO_GL=egl.
Object truth enters only PoseFeed (synthetic perception) and evaluator labels.
Robot FK remains the original teacher implementation. No cameras or learning.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from rdb.gen.episode_io import make_env
from rdb.gen.scripted_common import BimanualScript
from rdb.gen.scripted_pick_book_fast import PickBookFastExpert
from rdb.gen.scripted_shelf_fast import ShelfFastExpert

FAMILIES = {"pick": "PickSingleBookFromTable", "shelf": "StackSingleBookShelf"}
TEACHERS = {"pick": PickBookFastExpert, "shelf": ShelfFastExpert}
XY_CM, YAW_DEG, Z_CM = (0., .5, 1., 2.), (0., 2., 5.), (0., .5, 1.)
SEED_START, SEED_STOP, HZ = 2_210_000, 2_211_000, 20
CORNERS = np.array(list(itertools.product((-1, 1), repeat=3)))


class PoseFeed:
    """One coherent noisy body/geometry estimate per observed frame.

    Bias is constant ERROR, not frozen book pose: true moving poses still update.
    The shelf/counter receive one common rigid error about the shelf body origin.
    In iid mode their error also changes per frame, as requested by the screen;
    freezing a reset shelf estimate would remove this source of temporal jitter.
    Independent object and shelf draws; same standardized draws across settings.
    """

    def __init__(self, env, seed, mode, xy_cm, yaw_deg, z_cm):
        import mujoco
        self.env = env
        self.m = env._mojo.physics.model._model
        self.d = env._mojo.physics.data._data
        self.body = env.books[0].body
        prefix = self.body.mjcf.full_identifier
        # Match the teacher's geom selection exactly (including body ancestors).
        self.bg = BimanualScript.geoms_of(self, prefix)[0]
        self.counter = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                       env.book_shelf.counter.mjcf.full_identifier)
        self.board = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                     env.book_shelf.lower_shelf_body.mjcf.full_identifier)
        self.anchor = np.asarray(env.book_shelf.body.get_position(), float).copy()
        self.static = {g: (self.d.geom_xpos[g].copy(), self.d.geom_xmat[g].reshape(3, 3).copy())
                       for g in (self.counter, self.board)}
        self.local_corners = {g: CORNERS * self.m.geom_aabb[g][3:] + self.m.geom_aabb[g][:3]
                              for g in (self.bg, self.counter, self.board)}
        self.rng = np.random.default_rng(seed + 37_000_003)
        self.mode = mode
        self.scale = np.array([xy_cm / 100, xy_cm / 100, z_cm / 100, np.deg2rad(yaw_deg)])
        self.bias = self.rng.normal(size=(2, 4))
        self.update()

    def update(self):
        error = (self.bias if self.mode == "bias" else self.rng.normal(size=(2, 4))) * self.scale
        dr = R.from_euler("z", error[0, 3]).as_matrix()
        true_pos = np.asarray(self.body.get_position(), float)
        self.pos = true_pos + error[0, :3]
        true_rot = R.from_quat(np.asarray(self.body.get_quaternion()), scalar_first=True).as_matrix()
        self.rot = dr @ true_rot
        # Rotate both geom centre offset and orientation consistently about body.
        self.geoms = {self.bg: (self.pos + dr @ (self.d.geom_xpos[self.bg] - true_pos),
                               dr @ self.d.geom_xmat[self.bg].reshape(3, 3))}
        sr = R.from_euler("z", error[1, 3]).as_matrix()
        for g, (pos, rot) in self.static.items():
            self.geoms[g] = (self.anchor + error[1, :3] + sr @ (pos - self.anchor), sr @ rot)
        self.bounds = {}
        for g, (pos, rot) in self.geoms.items():
            world = self.local_corners[g] @ rot.T + pos
            self.bounds[g] = (world.min(0), world.max(0))


class BodyView:
    def __init__(self, body, feed):
        self.mjcf, self.feed = body.mjcf, feed

    def get_position(self):
        return self.feed.pos.copy()

    def get_quaternion(self):
        return R.from_matrix(self.feed.rot).as_quat(scalar_first=True)


class BookView:
    def __init__(self, book, feed):
        self.body = BodyView(book.body, feed)

    def is_colliding(self, other):
        # ShelfFastExpert uses this only for its on_shelf diagnostic trace.
        return False


class EnvView:
    def __init__(self, env, feed):
        self._env = env
        self.books = [BookView(env.books[0], feed)]

    def __getattr__(self, name):
        return getattr(self._env, name)


class RobotGeometryCache:
    """Cache static pad IDs and unchanged command FK within one action.

    No physics step occurs inside teacher.act(); command targets and measured
    gripper joints are constant throughout it. Results are numerically identical
    to rerunning these lookups/FK at every pinch() call.
    """

    def pad_mid(self, i):
        import mujoco
        if not hasattr(self, "_pad_ids"):
            self._pad_ids = {}
        if i not in self._pad_ids:
            hand = f"hand_{self.sides[i].value.lower()}/fingertip_pad_collision_"
            model = self.m
            ids = [g for g in range(model.ngeom)
                   if hand in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
            assert len(ids) == 2, ids
            self._pad_ids[i] = ids
        return self.d.geom_xpos[self._pad_ids[i]].mean(0).astype(np.float64)

    def cmd_pads(self):
        cached = getattr(self, "_command_pads", None)
        if cached is None:
            cached = super().cmd_pads()
            self._command_pads = cached
        return [p.copy() for p in cached]

    def act(self):
        self._command_pads = None
        return super().act()


class CachedPick(RobotGeometryCache, PickBookFastExpert):
    pass


class CachedShelf(RobotGeometryCache, ShelfFastExpert):
    pass


class ScreenInputs(RobotGeometryCache):
    """Only overrides teacher input geometry/contact; phase logic is inherited."""

    def __init__(self, env, rng, feed, obs):
        self.feed = feed
        self.observe(obs)
        super().__init__(EnvView(env, feed), rng, noise=0.)

    def observe(self, obs):
        self.openings = np.asarray(obs["proprioception_grippers"], float).copy()

    def holding(self, i, prop):
        return bool(self.grip[i] >= .5 and .05 <= self.openings[i] < .9)

    def geom_aabb(self, g):
        if g in self.feed.bounds:
            lo, hi = self.feed.bounds[g]
            return lo.copy(), hi.copy()
        # Robot pad bounds use unmodified robot kinematics.
        return super().geom_aabb(g)

    def target(self):
        # PickBookHumanExpert.target, using the estimated book geom transform.
        lo, hi = self.book_box()
        x = min(self.edge() + self.TARGET_EDGE, self.book.body.get_position()[0] - .015)
        pos, mat = self.feed.geoms[self.bg]
        world = self.feed.local_corners[self.bg] @ mat.T + pos
        ys = []
        for p in world:
            for q in world:
                if (p[0] - x) * (q[0] - x) <= 0 and abs(p[0] - q[0]) > 1e-8:
                    ys.append(float(p[1] + (x - p[0]) / (q[0] - p[0]) * (q[1] - p[1])))
        y = (min(ys) + max(ys)) / 2 if ys else self.book.body.get_position()[1]
        return np.array([x, y, (lo[2] + hi[2]) / 2 + self.HEIGHT_DZ])


class ScreenPick(ScreenInputs, PickBookFastExpert):
    pass


class ScreenShelf(ScreenInputs, ShelfFastExpert):
    pass


def run_episode(env, task, seed, setting, max_steps):
    obs, _ = env.reset(seed=seed)
    # Same initial no-op as the training teacher and final policy.
    obs, _, term, trunc, _ = env.step(np.zeros(14, np.float32))
    baseline = setting["mode"] == "privileged"
    rng = np.random.default_rng(seed + 7)
    feed = None if baseline else PoseFeed(env, seed, **setting)
    expert = ({"pick": CachedPick, "shelf": CachedShelf}[task](env, rng, noise=0.) if baseline else
              {"pick": ScreenPick, "shelf": ScreenShelf}[task](env, rng, feed, obs))
    floor_drop, status, steps = False, "budget", 1
    first_success_steps = None
    confusion = {k: 0 for k in ("tp", "fp", "fn", "tn")}
    confusion_by_phase = {}
    while steps < max_steps:
        if feed is not None:
            # Constructor's estimate is reused for first action (one draw/frame).
            if steps > 1:
                feed.update()
            expert.observe(obs)
        # Evaluator-only contacts: never pass these labels back to expert.
        if expert.grip[expert.arm] >= .5:
            pred = .05 <= float(obs["proprioception_grippers"][expert.arm]) < .9
            truth = bool(env.robot.is_gripper_holding_object(env.books[0], expert.sides[expert.arm]))
            key = "tp" if pred and truth else "fp" if pred else "fn" if truth else "tn"
            confusion[key] += 1
            phase_counts = confusion_by_phase.setdefault(expert.stage(expert.arm), dict.fromkeys(confusion, 0))
            phase_counts[key] += 1
        action = expert.act()
        obs, _, term, trunc, _ = env.step(action)
        steps += 1
        floor_drop |= bool(env.books[0].is_colliding(env._floor))
        if env.success:
            if first_success_steps is None:
                first_success_steps = steps
            # Count shelf success only with a physically commanded opening.
            # Evaluator success must NEVER force a phase/action transition:
            # the teacher follows its own place/release/back sequence.
            if task == "pick" or expert.grip[expert.arm] < .5:
                status = "success"
                break
        elif term or trunc:
            status = "fail" if term else "unhealthy"
            break
    return dict(seed=seed, **setting, status=status, steps=steps, time_s=steps / HZ,
                floor_drop=floor_drop, retries=expert.retries, arm=expert.arm,
                phase=expert.stage(expert.arm), first_success_steps=first_success_steps,
                grip_confusion=confusion, grip_confusion_by_phase=confusion_by_phase)


def setting_key(row):
    return tuple(row[k] for k in ("mode", "xy_cm", "yaw_deg", "z_cm"))


def summarize(rows):
    wins = [r["time_s"] for r in rows if r["status"] == "success"]
    return dict(zip(("mode", "xy_cm", "yaw_deg", "z_cm"), setting_key(rows[0])),
                n=len(rows), successes=len(wins), success_rate=len(wins) / len(rows),
                floor_drops=sum(r["floor_drop"] for r in rows),
                median_success_s=float(np.median(wins)) if wins else None,
                median_all_s=float(np.median([r["time_s"] for r in rows])),
                failures=[r["seed"] for r in rows if r["status"] != "success"])


def provenance():
    from rdb.gen import scripted_common, scripted_books, scripted_pick_book_h, scripted_pick_book_fast, scripted_shelf_fast, scripted_pack_box_h, episode_io
    paths = [Path(__file__)] + [Path(inspect.getfile(m)) for m in
             (scripted_common, scripted_books, scripted_pick_book_h, scripted_pick_book_fast, scripted_shelf_fast, scripted_pack_box_h, episode_io)]
    return {str(p.relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    from absl import logging
    from roboeval.challenge import spec
    logging.set_verbosity(logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=FAMILIES, required=True)
    ap.add_argument("--seeds-per-setting", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=SEED_START)
    ap.add_argument("--xy-cm", type=float, nargs="+", default=XY_CM)
    ap.add_argument("--yaw-deg", type=float, nargs="+", default=YAW_DEG)
    ap.add_argument("--z-cm", type=float, nargs="+", default=Z_CM)
    ap.add_argument("--modes", nargs="+", choices=("iid", "bias"), default=("iid", "bias"))
    ap.add_argument("--out", type=Path, default=Path("runs/hybrid_books/screen_v3"))
    args = ap.parse_args()
    if not (1 <= args.seeds_per_setting <= 30 and SEED_START <= args.seed_start
            and args.seed_start + args.seeds_per_setting <= SEED_STOP):
        ap.error("Require 1–30 seeds inside registered range 2210000–2210999")
    if any(v not in allowed for vals, allowed in ((args.xy_cm, XY_CM), (args.yaw_deg, YAW_DEG), (args.z_cm, Z_CM)) for v in vals):
        ap.error("Noise scales must belong to the prescribed grid")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.task}.jsonl"
    meta_path = args.out / f"{args.task}_meta.json"
    meta = dict(family=FAMILIES[args.task], variation="PositionAndOrientation", hz=HZ,
                seed_start=args.seed_start, seeds_per_setting=args.seeds_per_setting,
                action_noise=0., shelf_noise="common rigid transform about shelf body origin",
                shelf_completion="success with opening commanded; no evaluator-driven transitions",
                proprio_grip_rule="commanded close and 0.05 <= opening < 0.9",
                dependencies=provenance())
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("Output provenance differs: use a fresh --out")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    prior = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    records = {(setting_key(r), r["seed"]): r for r in prior}
    if len(records) != len(prior):
        raise ValueError("Duplicate output rows")
    settings = [dict(mode="privileged", xy_cm=0., yaw_deg=0., z_cm=0.)]
    settings += [dict(mode=mode, xy_cm=xy, yaw_deg=yaw, z_cm=z)
                 for mode, xy, yaw, z in itertools.product(args.modes, args.xy_cm, args.yaw_deg, args.z_cm)]
    env = make_env(FAMILIES[args.task], "PositionAndOrientation", render=False)
    started = time.monotonic()
    try:
        with path.open("a", buffering=1) as output, open(os.devnull, "w") as sink:
            for setting in settings:
                rows = []
                for seed in range(args.seed_start, args.seed_start + args.seeds_per_setting):
                    key = (setting_key(setting), seed)
                    if key not in records:
                        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                            row = run_episode(env, args.task, seed, setting, spec.max_steps(FAMILIES[args.task], HZ))
                        output.write(json.dumps(row) + "\n")
                        records[key] = row
                    rows.append(records[key])
                summary = summarize(rows)
                print(json.dumps(dict(task=args.task, **summary, elapsed_wall_s=round(time.monotonic() - started, 1))), flush=True)
            summaries = [summarize([records[(setting_key(s), seed)]
                         for seed in range(args.seed_start, args.seed_start + args.seeds_per_setting)]) for s in settings]
            summary_path = args.out / f"{args.task}_summary.json"
            tmp = summary_path.with_suffix(".tmp.json")
            tmp.write_text(json.dumps(summaries, indent=2) + "\n")
            tmp.replace(summary_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
