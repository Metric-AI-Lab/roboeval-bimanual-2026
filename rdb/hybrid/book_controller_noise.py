"""Millimetre pose-noise screen of the observation-only pick controller.

One camera-free worker per --mode.
Only book pose is perturbed; counter geometry is exact. Noise is standard
 deviation per world x/y coordinate, world z, and world yaw. Bias errors stay
constant while the book's actual pose continues moving. Truth is supplied by
the local verifier; it is never available inside the policy component.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.spatial.transform import Rotation as R

from rdb.gen.episode_io import make_env
from rdb.gen.scripted_pack_box_h import thermal_guard
from rdb.hybrid.book_controller import PickBookController, MAX_JOINT_DELTA, PAD_IN_EE
from rdb.hybrid.verify_book_controller import live_pose


class PoseNoise:
    def __init__(self, seed, mode, xy_mm, yaw_deg, z_mm):
        self.rng = np.random.default_rng(seed+47_000_003)
        self.mode = mode
        self.scales = np.array([xy_mm/1000, xy_mm/1000, z_mm/1000, np.deg2rad(yaw_deg)])
        self.bias = self.rng.normal(size=4)

    def __call__(self, true_pose):
        if not self.scales.any():
            return np.asarray(true_pose, float).copy()
        error = (self.bias if self.mode == 'bias' else self.rng.normal(size=4))*self.scales
        pose = np.asarray(true_pose, float).copy()
        pose[:3] += error[:3]
        rot = R.from_euler('z', error[3])*R.from_quat(pose[3:], scalar_first=True)
        pose[3:] = rot.as_quat(scalar_first=True)
        return pose


def run_episode(env, seed, mode, xy_mm, yaw_deg, z_mm):
    from roboeval.challenge import spec
    obs, _ = env.reset(seed=seed)
    if not np.isclose(env._mojo.physics.model.opt.timestep, .002):
        raise ValueError('Pick requires a 2 ms physics step and 0.05 s joint clamp')
    noise = PoseNoise(seed, mode, xy_mm, yaw_deg, z_mm)
    estimate = noise(live_pose(env))
    policy = PickBookController(estimate)
    floor, status, max_qerr = False, 'budget', 0.
    phases = []
    for step in range(spec.max_steps('PickSingleBookFromTable', 20)):
        if step % 20 == 0:
            thermal_guard()
        if step:
            estimate = noise(live_pose(env))
        action = policy.act(obs, estimate)
        obs, _, term, trunc, _ = env.step(action)
        if not phases or phases[-1]['phase'] != policy.phase:
            phases.append(dict(step=step+1, phase=policy.phase))
        floor |= bool(env.books[0].is_colliding(env._floor))
        actual = np.asarray(env.robot.get_initial_joint_positions(), float).reshape(-1)
        max_qerr = max(max_qerr, float(np.max(np.abs(actual-policy.qctrl))))
        if env.success or term or trunc:
            status = 'success' if env.success else 'fail' if term else 'unhealthy'
            break
    return dict(seed=seed, mode=mode, xy_mm=xy_mm, yaw_deg=yaw_deg, z_mm=z_mm,
                status=status, steps=step+1, time_s=(step+1)/20, floor_drop=floor,
                arm=policy.arm, retries=policy.retries, phase=policy.phase, phases=phases,
                max_qctrl_error_rad=max_qerr,
                finished_at=datetime.now(ZoneInfo('Asia/Dubai')).isoformat(timespec='seconds'))


def key(row):
    return tuple(row[n] for n in ('xy_mm','yaw_deg','z_mm'))


def summary(rows):
    wins=[r['time_s'] for r in rows if r['status']=='success']
    return dict(mode=rows[0]['mode'], xy_mm=rows[0]['xy_mm'], yaw_deg=rows[0]['yaw_deg'], z_mm=rows[0]['z_mm'],
                n=len(rows), successes=len(wins), success_rate=len(wins)/len(rows),
                floor_drops=sum(r['floor_drop'] for r in rows),
                median_success_s=statistics.median(wins) if wins else None,
                median_all_s=statistics.median(r['time_s'] for r in rows),
                max_qctrl_error_rad=max(r['max_qctrl_error_rad'] for r in rows),
                completed_at=max(r['finished_at'] for r in rows),
                failures=[r['seed'] for r in rows if r['status']!='success'])


def main():
    from absl import logging
    logging.set_verbosity(logging.ERROR)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',required=True,choices=('iid','bias'))
    ap.add_argument('--count',type=int,default=30)
    ap.add_argument('--xy-mm',type=float,nargs='+',default=(1.,2.))
    ap.add_argument('--yaw-deg',type=float,nargs='+',default=(1.,2.))
    ap.add_argument('--z-mm',type=float,nargs='+',default=(.5,1.))
    ap.add_argument('--out',type=Path,default=Path('runs/hybrid_books/controller_noise_mm'))
    args=ap.parse_args()
    if not 1 <= args.count <= 30:
        ap.error('Require 1–30 paired seeds starting at 2210000')
    for values,allowed in ((args.xy_mm,(1.,2.)),(args.yaw_deg,(1.,2.)),(args.z_mm,(.5,1.))):
        if any(v not in allowed for v in values):
            ap.error('Noise scales must belong to the requested mm/degree grid')
    args.out.mkdir(parents=True,exist_ok=True)
    dependencies={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [Path(__file__)]+[Path('rdb/hybrid/'+n) for n in
                    ('book_controller.py','book_kinematics.py','book_robot_model.py','verify_book_controller.py')]}
    meta=dict(mode=args.mode, family='PickSingleBookFromTable', variation='PositionAndOrientation',
              seed_start=2210000,count=args.count,hz=20,physics_dt_s=.002,
              max_joint_delta_rad=MAX_JOINT_DELTA,pad_in_ee=PAD_IN_EE.tolist(),
              pose_noise='book only; per-axis world xy/z mm and world yaw deg; full roll/pitch retained',
              dependencies=dependencies)
    meta_path=args.out/f'{args.mode}_meta.json'
    if meta_path.exists() and json.loads(meta_path.read_text())!=meta:
        raise ValueError('Output provenance differs; use a fresh --out')
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    path=args.out/f'{args.mode}.jsonl'
    prior=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    records={(key(r),r['seed']):r for r in prior}
    if len(records)!=len(prior):
        raise ValueError('Duplicate output rows')
    settings=[(0.,0.,0.)]+list(itertools.product(args.xy_mm,args.yaw_deg,args.z_mm))
    env=make_env('PickSingleBookFromTable','PositionAndOrientation',render=False)
    started=time.monotonic()
    summaries=[]
    try:
        with path.open('a',buffering=1) as output,open(os.devnull,'w') as sink:
            for xy,yaw,z in settings:
                rows=[]
                for seed in range(2210000,2210000+args.count):
                    identity=((xy,yaw,z),seed)
                    if identity not in records:
                        with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
                            result=run_episode(env,seed,args.mode,xy,yaw,z)
                        output.write(json.dumps(result)+'\n')
                        records[identity]=result
                    rows.append(records[identity])
                result=summary(rows)
                summaries.append(result)
                print(json.dumps(dict(**result,elapsed_wall_s=round(time.monotonic()-started,1))),flush=True)
                target=args.out/f'{args.mode}_summary.json'
                tmp=target.with_suffix('.tmp.json')
                tmp.write_text(json.dumps(summaries,indent=2)+'\n')
                tmp.replace(target)
    finally:
        env.close()


if __name__=='__main__':
    main()
