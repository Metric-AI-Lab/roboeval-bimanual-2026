"""Oracle-pose closed-loop check of the observation-only RotateValve controller (CPU, no cameras).

  OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python -m rdb.hybrid.verify_valve_controller --count 10 [--calibrate]

--calibrate prints the robot base offset (simulated pad midpoints minus our FK pads at reset). The true hub-top
points stand in for the pose regressor (verifier input only). Seeds 2241000-2241999 (registered).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os

import numpy as np
from absl import logging as absl_logging

from rdb.gen.episode_io import make_env
from rdb.gen.scripted_rotate_valve_h import RotateValveHumanExpert
from rdb.hybrid.valve_controller import ValveController

HZ = 20


def hub_tops(env):
    """The teacher's hub_top points for env.valves order (privileged: verifier / label use only)."""
    t = RotateValveHumanExpert(env, np.random.default_rng(0), noise=0)
    out = []
    for valve, g in t.hub:
        _, hi = t.geom_aabb(g)
        c = t.d.geom_xpos[g].copy()
        c[2] = hi[2] - t.GRASP_DOWN
        out.append(c)
    return out, t


def main():
    from roboeval.challenge import spec
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-seed", type=int, default=2_241_000)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--noise-xy-m", type=float, default=0.0, help="constant Gaussian xy error per hub (pose-model stand-in)")
    args = ap.parse_args()
    if not (2_241_000 <= args.first_seed and args.first_seed + args.count <= 2_242_000):
        ap.error("seeds must stay in 2241000-2241999")
    absl_logging.set_verbosity(absl_logging.ERROR)
    env = make_env("RotateValve", "PositionAndOrientation", render=False)
    max_steps = spec.max_steps("RotateValve", HZ)
    rows = []
    try:
        with open(os.devnull, "w") as sink:
            for seed in range(args.first_seed, args.first_seed + args.count):
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    obs, _ = env.reset(seed=seed)
                    tops, teacher = hub_tops(env)
                    rng = np.random.default_rng(seed)
                    tops = [t + np.r_[rng.normal(0, args.noise_xy_m, 2), 0.] for t in tops]
                    ctl = ValveController(tops, passes=args.passes, sync=args.sync)
                    if args.calibrate:
                        sim = [teacher.pad_mid(i) for i in (0, 1)]
                        print("pelvis offset", [np.round(sim[i] - ctl.cmd_pads()[i], 6).tolist() for i in (0, 1)],
                              file=__import__("sys").__stderr__)
                    steps, dz, dv = 0, [], []
                    while steps < max_steps:
                        q0 = ctl.qctrl.copy()
                        obs, _, term, trunc, _ = env.step(ctl.act(obs))
                        steps += 1
                        dz.append(abs(ctl._pads[0][2] - ctl._pads[1][2]))       # proxies for bgvd / bavd
                        dq = (ctl.qctrl - q0) * HZ
                        dv.append(abs(np.linalg.norm(dq[:7]) - np.linalg.norm(dq[7:])))
                        if env.success or term or trunc:
                            break
                    states = [float(np.mean(v.get_state())) for v in env.valves]
                row = dict(seed=seed, success=bool(env.success), steps=steps, time_s=steps / HZ,
                           retries=ctl.retries, z_diff=round(float(np.mean(dz)), 4), vel_diff=round(float(np.mean(dv)), 3), stages=[ctl.stage(0), ctl.stage(1)], valve_states=states)
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        env.close()
    ok = [r for r in rows if r["success"]]
    print("SUMMARY " + json.dumps(dict(n=len(rows), success=len(ok),
                                       median_success_s=float(np.median([r["time_s"] for r in ok])) if ok else None,
                                       z_diff=float(np.mean([r["z_diff"] for r in rows])),
                                       vel_diff=float(np.mean([r["vel_diff"] for r in rows])),
                                       failures=[r["seed"] for r in rows if not r["success"]])), flush=True)


if __name__ == "__main__":
    main()
