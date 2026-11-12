"""Collect live book pose labels from fast teacher or pick/shelf-controller rollouts.

Training data only. Raw policy observations (CHW uint8 256x256, unchanged
proprioception) are aligned with world-frame pose/contact labels at that instant.
One atomic compressed NPZ episode shard, including unsuccessful trajectories.
Run one EGL process per task, with one BLAS thread. No training here.
`--policy ctl` instead drives the task's default hybrid controller on per-frame iid
3 mm xyz / 1 degree yaw privileged book pose noise (labels remain exact).
`--mix-yaw-close` uses yaw 1.1 / close 17 on odd seeds, baseline on even seeds.
Controller training seeds: shelf 2268000–2268999, pick 2270000–2270999.
The optional yaw/close mixture is shelf-only. Defaults have no idle-wrist changes.
Shelf `--ctl-kwargs JSON` enables custom controller settings on 2272000–2272999 and 2285000–2285999 (v9 preset);
`--idle-wrist-z-range MIN MAX` samples a deterministic per-episode idle height.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from rdb.gen.episode_io import make_env
from rdb.hybrid.book_noise_screen import CachedPick, CachedShelf, FAMILIES

SEEDS = {"pick": (2_211_000, 2_215_000), "shelf": (2_215_000, 2_216_000)}
CTL_SEEDS = {"shelf": (2_268_000, 2_269_000), "pick": (2_270_000, 2_271_000)}
CUSTOM_SHELF_SEEDS = (2_272_000, 2_273_000)
CUSTOM_SHELF_SEEDS_V9 = (2_285_000, 2_286_000)  # v9 controller preset
IDLE_HEIGHT_SEED_OFFSET = 91_000_019
CAMERAS = ("head", "left_wrist", "right_wrist")
TEACHERS = {"pick": CachedPick, "shelf": CachedShelf}


def record(obs, env, task, seed, step):
    frame = {}
    for cam in CAMERAS:
        key = f"rgb_{cam}"
        rgb = np.asarray(obs[key])
        if rgb.shape != (3, 256, 256) or rgb.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 CHW 256x256; got {rgb.shape} {rgb.dtype}")
        frame[key] = rgb.copy()
    for key, shape in (("proprioception", (36,)), ("proprioception_grippers", (2,))):
        value = np.asarray(obs[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {key}: {value.shape}")
        frame[key] = value.copy()  # Preserve policy observation values and dtype.
    book = env.books[0]
    frame["book_xyz"] = np.asarray(book.body.get_position(), np.float32).copy()
    frame["book_quat_wxyz"] = np.asarray(book.body.get_quaternion(), np.float32).copy()
    frame["holding"] = np.array([env.robot.is_gripper_holding_object(book, side)
                                 for side in env.robot.grippers], dtype=bool)
    if task == "shelf":
        frame["shelf_xyz"] = np.asarray(env.book_shelf.body.get_position(), np.float32).copy()
        frame["shelf_quat_wxyz"] = np.asarray(env.book_shelf.body.get_quaternion(), np.float32).copy()
    frame["seed"] = np.int64(seed)
    frame["step"] = np.int32(step)
    return frame


def source_hashes(policy="teacher"):
    if policy == "ctl":
        from rdb.hybrid import book_controller, book_controller_noise, book_kinematics, book_robot_model, pot_controller
        from rdb.gen import episode_io, scripted_pack_box_h
        modules = (book_controller, book_controller_noise, book_kinematics, book_robot_model,
                   pot_controller, episode_io, scripted_pack_box_h)
        paths = [Path(__file__)] + [Path(inspect.getfile(m)) for m in modules]
        return {str(p.relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    from rdb.gen import scripted_common, scripted_books, scripted_pick_book_h, scripted_pick_book_fast, scripted_shelf_fast, scripted_pack_box_h, episode_io
    from rdb.hybrid import book_noise_screen
    paths = [Path(__file__)] + [Path(inspect.getfile(m)) for m in
             (scripted_common, scripted_books, scripted_pick_book_h, scripted_pick_book_fast,
              scripted_shelf_fast, scripted_pack_box_h, episode_io, book_noise_screen)]
    return {str(p.relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def collect(env, task, seed, noise, max_steps):
    obs, _ = env.reset(seed=seed)
    frames = [record(obs, env, task, seed, 0)]
    obs, _, term, trunc, _ = env.step(np.zeros(14, np.float32))
    teacher = TEACHERS[task](env, np.random.default_rng(seed + 7), noise=noise)
    steps, status, floor_drop = 1, "budget", False
    while steps < max_steps:
        action = teacher.perturb(teacher.act())
        obs, _, term, trunc, _ = env.step(action)
        steps += 1
        if steps % 2 == 0:
            frames.append(record(obs, env, task, seed, steps))
        floor_drop |= bool(env.books[0].is_colliding(env._floor))
        # No evaluator-driven phase changes, even though privileged teachers are
        # allowed for data. Shelf follows its own release sequence.
        if env.success and (task == "pick" or teacher.grip[teacher.arm] < .5):
            status = "success"
            break
        if not env.success and (term or trunc):
            status = "fail" if term else "unhealthy"
            break
    return frames, dict(seed=seed, status=status, success=status == "success", steps=steps,
                        frames=len(frames), time_s=steps / 20, floor_drop=floor_drop,
                        arm=teacher.arm, retries=teacher.retries)


def collect_ctl(env, seed, max_steps, mix_yaw_close=False, task="shelf",
                ctl_kwargs=None, idle_wrist_z_range=None):
    """Drive the observation-only controller with noisy inputs, exact labels.

    Stride/alignment is identical to collect(): reset frame 0 followed by each
    even post-action observation. Controller act() supplies its own first no-op.
    """
    from rdb.hybrid.book_controller import PickBookController, ShelfBookController
    from rdb.hybrid.book_controller_noise import PoseNoise
    from rdb.gen.scripted_pack_box_h import thermal_guard

    obs, _ = env.reset(seed=seed)
    if not np.isclose(env._mojo.physics.model.opt.timestep, .002):
        raise ValueError("Book controllers require a 2 ms physics step")
    body = env.books[0].body
    noise = PoseNoise(seed, "iid", 3., 1., 3.)
    estimate = noise(np.r_[body.get_position(), body.get_quaternion()])
    shelf = np.r_[env.book_shelf.body.get_position(), env.book_shelf.body.get_quaternion()] if task == "shelf" else None
    kwargs = dict(ctl_kwargs) if ctl_kwargs is not None else (
        dict(grasp_yaw=1.1, close_steps=17) if mix_yaw_close and seed % 2 else {})
    preset = "custom" if ctl_kwargs is not None else "yaw11_close17" if kwargs else "baseline"
    if idle_wrist_z_range is not None:
        kwargs["idle_wrist_z"] = float(np.random.default_rng(seed+IDLE_HEIGHT_SEED_OFFSET).uniform(*idle_wrist_z_range))
        preset = "custom_idle_uniform"
    ctl = ShelfBookController(estimate, shelf, **kwargs) if task == "shelf" else PickBookController(estimate)
    frames = [record(obs, env, task, seed, 0)]
    steps, status, floor_drop, max_qerr = 0, "budget", False, 0.
    while steps < max_steps:
        if steps % 20 == 0:
            thermal_guard()
        if steps:
            estimate = noise(np.r_[body.get_position(), body.get_quaternion()])
        action = ctl.act(obs, estimate, shelf) if task == "shelf" else ctl.act(obs, estimate)
        obs, _, term, trunc, _ = env.step(action)
        steps += 1
        if steps % 2 == 0:
            frames.append(record(obs, env, task, seed, steps))
        floor_drop |= bool(env.books[0].is_colliding(env._floor))
        actual = np.asarray(env.robot.get_initial_joint_positions(), float).reshape(-1)
        max_qerr = max(max_qerr, float(np.max(np.abs(actual-ctl.qctrl))))
        if max_qerr > 1e-6:
            raise RuntimeError(f"Command target drift {max_qerr:g}, seed {seed}, step {steps}")
        # Evaluator success/contact never changes controller phase/action.
        if env.success and (task == "pick" or ctl.grip[ctl.arm] < .5):
            status = "success"
            break
        if not env.success and (term or trunc):
            status = "fail" if term else "unhealthy"
            break
    return frames, dict(seed=seed, status=status, success=status == "success", steps=steps,
                        frames=len(frames), time_s=steps/20, floor_drop=floor_drop,
                        arm=ctl.arm, retries=ctl.retries,
                        controller_preset=preset,
                        controller_kwargs=kwargs, max_qctrl_error_rad=max_qerr)


def save_episode(path, frames, metadata):
    arrays = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
    arrays["meta"] = np.array(json.dumps(metadata))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main():
    from absl import logging
    from roboeval.challenge import spec
    logging.set_verbosity(logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=FAMILIES, required=True)
    ap.add_argument("--policy", choices=("teacher", "ctl"), default="teacher")
    ap.add_argument("--mix-yaw-close", action="store_true",
                    help="shelf ctl: odd seeds use yaw 1.1 / close 17; even seeds use defaults")
    ap.add_argument("--ctl-kwargs", type=json.loads,
                    help="shelf ctl: JSON constructor kwargs; custom training seeds 2272000–2272999 or 2285000–2285999")
    ap.add_argument("--idle-wrist-z-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                    help="shelf ctl: seeded uniform idle_wrist_z per episode, overriding --ctl-kwargs")
    ap.add_argument("--first-seed", type=int)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=.5,
                    help="Teacher-only DART scale: .5 = 1.25 mm translation, .0075 rad rotation")
    ap.add_argument("--out", type=Path, default=Path("data/hybrid/books"))
    args = ap.parse_args()
    if args.mix_yaw_close and (args.policy != "ctl" or args.task != "shelf"):
        ap.error("--mix-yaw-close requires --task shelf --policy ctl")
    custom = args.ctl_kwargs is not None or args.idle_wrist_z_range is not None
    if custom:
        if args.task != "shelf" or args.policy != "ctl" or args.mix_yaw_close:
            ap.error("Custom kwargs/idle heights require shelf ctl without --mix-yaw-close")
        if args.ctl_kwargs is None:
            args.ctl_kwargs = {}
        if not isinstance(args.ctl_kwargs, dict):
            ap.error("--ctl-kwargs must be a JSON object")
        from rdb.hybrid.book_controller import ShelfBookController
        try:
            inspect.signature(ShelfBookController).bind(None, None, **args.ctl_kwargs)
        except TypeError as exc:
            ap.error(str(exc))
        if args.idle_wrist_z_range is not None:
            low, high = args.idle_wrist_z_range
            if not np.isfinite([low,high]).all() or low >= high:
                ap.error("Require finite idle-wrist bounds with MIN < MAX")
    lo, hi = CTL_SEEDS[args.task] if args.policy == "ctl" else SEEDS[args.task]
    if custom:
        lo, hi = CUSTOM_SHELF_SEEDS
        if args.first_seed is not None and CUSTOM_SHELF_SEEDS_V9[0] <= args.first_seed < CUSTOM_SHELF_SEEDS_V9[1]:
            lo, hi = CUSTOM_SHELF_SEEDS_V9
    first = lo if args.first_seed is None else args.first_seed
    if not (lo <= first < hi and 1 <= args.count <= hi - first and np.isfinite(args.noise) and args.noise >= 0):
        ap.error(f"Require nonnegative finite noise and seeds in {lo}–{hi-1}")
    family = FAMILIES[args.task]
    out = args.out / f"{family}PositionAndOrientation"
    out.mkdir(parents=True, exist_ok=True)
    config = dict(source="hybrid_book_pose", family=family, variation="PositionAndOrientation",
                  task=args.task, noise=args.noise, hz=20, frame_stride=2,
                  observation="raw policy observations; CHW uint8 RGB 256x256, unchanged proprioception",
                  quaternion="wxyz, world frame", holding_order=["left", "right"],
                  keep_failures=True, dependencies=source_hashes(args.policy))
    if args.policy == "ctl":
        config.update(source=f"hybrid_{args.task}_ctl_pose", policy="ctl", noise=None,
                      pose_noise=dict(temporal="iid", xyz_std_mm=[3.,3.,3.], yaw_std_deg=1.,
                                      labels="exact unperturbed simulator pose",
                                      shelf="exact, frozen at reset" if args.task == "shelf" else "not used"),
                      controller_presets={"baseline": {}, "yaw11_close17": dict(grasp_yaw=1.1, close_steps=17)}
                          if args.mix_yaw_close else {"baseline": {}},
                      controller_assignment="odd seed: yaw11_close17; even seed: baseline"
                          if args.mix_yaw_close else "baseline only",
                      perturb_actions=False, idle_wrist_motion=False)
        if custom:
            config.update(controller_presets={"custom": args.ctl_kwargs},
                          controller_assignment="explicit constructor kwargs",
                          idle_wrist_motion=bool(args.idle_wrist_z_range is not None or
                              args.ctl_kwargs.get("idle_wrist_z") is not None or args.ctl_kwargs.get("wrist_follow",0)))
            if args.idle_wrist_z_range is not None:
                config["idle_wrist_z_sampling"] = dict(distribution="uniform", bounds=args.idle_wrist_z_range,
                                                     seed_offset=IDLE_HEIGHT_SEED_OFFSET)
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Dataset config/provenance differs; use a fresh --out")
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    started = time.monotonic()
    env = make_env(family, "PositionAndOrientation", render=True)
    completed, successes, total_frames = 0, 0, 0
    try:
        with (out / "episodes.jsonl").open("a", buffering=1) as audit, open(os.devnull, "w") as sink:
            for seed in range(first, first + args.count):
                path = out / f"ep_{seed}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        meta = json.loads(str(saved["meta"]))
                    if meta.get("config") != config or meta.get("seed") != seed:
                        raise ValueError(f"Shard provenance differs: {path}")
                    completed += 1
                    successes += int(meta["success"])
                    total_frames += meta["frames"]
                    continue
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    if args.policy == "ctl":
                        frames, result = collect_ctl(env, seed, spec.max_steps(family, 20), args.mix_yaw_close,
                                                     args.task, args.ctl_kwargs, args.idle_wrist_z_range)
                    else:
                        frames, result = collect(env, args.task, seed, args.noise, spec.max_steps(family, 20))
                meta = dict(config=config, **result)
                save_episode(path, frames, meta)
                completed += 1
                successes += int(result["success"])
                total_frames += len(frames)
                row = dict(task=args.task, path=str(path), **result, completed=completed,
                           successes=successes, total_frames=total_frames,
                           elapsed_wall_s=round(time.monotonic() - started, 1))
                audit.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                del frames
    finally:
        env.close()
    print(json.dumps(dict(task=args.task, complete=True, episodes=completed, successes=successes,
                          frames=total_frames, elapsed_wall_s=round(time.monotonic() - started, 1))), flush=True)


if __name__ == "__main__":
    main()
