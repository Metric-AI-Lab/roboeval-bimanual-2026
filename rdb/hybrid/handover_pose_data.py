"""Training-only live rod-pose shards from HandoverController on noisy inputs.

Raw CHW uint8 RGB256 and unchanged proprioception, exact world xyz/wxyz labels.
One atomically replaced compressed NPZ per episode; failures are retained.
Only training seeds 2287000..2287999. No model training or challenge submission.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from rdb.hybrid.handover_controller import HandoverController

FAMILY = 'CubeHandover'
CAMERAS = ('head','left_wrist','right_wrist')


def rod_pose(env):
    """Privileged labels/feed only; public rod geom has zero body offset."""
    return np.r_[env.cube.body.get_position(),env.cube.body.get_quaternion()]


class RodNoise:
    """IID world xyz and world rotation-vector noise, per coordinate/axis."""
    def __init__(self, seed, xyz_mm=3., angle_deg=1.):
        if not np.isfinite([xyz_mm,angle_deg]).all() or min(xyz_mm,angle_deg)<0:
            raise ValueError('Require finite nonnegative noise standard deviations')
        self.rng = np.random.default_rng(seed+57_000_003)
        self.xyz, self.angle = xyz_mm/1000,np.deg2rad(angle_deg)

    def __call__(self, exact):
        p = np.asarray(exact,float).copy()
        draws = self.rng.normal(size=6)
        p[:3] += draws[:3]*self.xyz
        p[3:] = (R.from_rotvec(draws[3:]*self.angle)*R.from_quat(p[3:],scalar_first=True)).as_quat(scalar_first=True)
        return p


def record(obs, exact_pose, seed, step):
    frame = {}
    for cam in CAMERAS:
        key = f'rgb_{cam}'
        image = np.asarray(obs[key])
        if image.shape != (3,256,256) or image.dtype != np.uint8:
            raise ValueError(f'{key}: require raw CHW uint8 RGB256, got {image.shape}/{image.dtype}')
        frame[key] = image.copy()
    for key,shape in [('proprioception',(36,)),('proprioception_grippers',(2,))]:
        value = np.asarray(obs[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid standard observation {key}')
        frame[key] = value.copy()
    frame.update(rod_xyz=np.asarray(exact_pose[:3],np.float32).copy(),
                 rod_quat_wxyz=np.asarray(exact_pose[3:],np.float32).copy(),
                 seed=np.int64(seed),step=np.int32(step))
    return frame


def source_hashes():
    names = ['handover_controller.py','handover_pose_data.py','obs_script.py',
             'book_kinematics.py','book_robot_model.py','pot_controller.py']
    paths = [Path(__file__).parent/n for n in names]
    paths += [Path(__file__).parents[1]/'gen/episode_io.py']
    return {str(p.relative_to(Path(__file__).parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def collect(env, seed, max_steps, xyz_mm=3., angle_deg=1., ctl_kwargs=None):
    from rdb.gen.scripted_pack_box_h import thermal_guard
    from roboeval.challenge import results,scoring
    obs,_ = env.reset(seed=seed)
    if not np.isclose(env._mojo.physics.model.opt.timestep,.002):
        raise ValueError('Require 2 ms physics step / 20 Hz control')
    exact = rod_pose(env)
    noise = RodNoise(seed,xyz_mm,angle_deg)
    estimate = noise(exact)
    ctl = HandoverController(estimate,**(ctl_kwargs or {}))
    frames = [record(obs,exact,seed,0)]
    qerr,status = 0.,'budget'
    for k in range(max_steps):
        if k%20 == 0:
            thermal_guard()
        if k:
            estimate = noise(rod_pose(env))
        action = ctl.act(obs,estimate)
        obs,_,term,trunc,info = env.step(action)
        steps = k+1
        if steps%2 == 0:
            frames.append(record(obs,rod_pose(env),seed,steps))
        actual = np.asarray(env.robot.get_initial_joint_positions(),float).reshape(-1)
        qerr = max(qerr,float(np.abs(actual-ctl.qctrl).max()))
        if qerr>1e-6:
            raise RuntimeError(f'Command target mismatch seed {seed}, step {steps}: {qerr:g}')
        if env.success or term or trunc:
            status = 'success' if env.success else 'fail' if term else 'unhealthy'
            break
    metrics = results.episode_metrics(results.to_jsonable(info),steps,20)
    return frames,dict(seed=seed,status=status,success=status=='success',steps=steps,
                       frames=len(frames),time_s=steps/20,giver=ctl.giver,retries=ctl.retries,
                       last_phase=ctl.phase,max_qctrl_error_rad=qerr,metrics=metrics,
                       scores=scoring.task_scores(FAMILY,metrics))


def atomic_text(path, text):
    fd,tmp = tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            f.write(text)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def save_episode(path,frames,metadata):
    arrays = {key:np.stack([f[key] for f in frames]) for key in frames[0]}
    arrays['meta'] = np.array(json.dumps(metadata))
    fd,tmp = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp.npz',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,**arrays)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main():
    from absl import logging
    from roboeval.challenge import spec
    from rdb.gen.episode_io import make_env
    logging.set_verbosity(logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy',choices=['ctl'],default='ctl')
    ap.add_argument('--first-seed',type=int,default=2287000)
    ap.add_argument('--count',type=int,default=1000)
    ap.add_argument('--xyz-mm',type=float,default=3.)
    ap.add_argument('--angle-deg',type=float,default=1.)
    ap.add_argument('--ctl-kwargs',type=json.loads,default={})
    ap.add_argument('--out',type=Path,default=Path('data/hybrid/handover_ctl'))
    a = ap.parse_args()
    if not 2287000 <= a.first_seed < a.first_seed+a.count <= 2288000:
        ap.error('Training data must stay within 2287000–2287999; count must be positive')
    if not isinstance(a.ctl_kwargs,dict):ap.error('--ctl-kwargs must be a JSON object')
    RodNoise(a.first_seed,a.xyz_mm,a.angle_deg)
    HandoverController([.5,0,1,1,0,0,0],**a.ctl_kwargs)
    out = a.out/'CubeHandoverPositionAndOrientation'
    out.mkdir(parents=True,exist_ok=True)
    config = dict(source='hybrid_handover_ctl_pose',family=FAMILY,variation='PositionAndOrientation',
                  policy='ctl',hz=20,frame_stride=2,labels=['rod_xyz','rod_quat_wxyz'],
                  quaternion='exact world wxyz',observation='raw CHW uint8 RGB256; unchanged proprioception',
                  noise=dict(temporal='iid',xyz_std_mm_per_axis=a.xyz_mm,
                             world_rotvec_std_deg_per_axis=a.angle_deg,labels='exact, never perturbed'),
                  controller_kwargs=a.ctl_kwargs,keep_failures=True,dependencies=source_hashes())
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise ValueError('Dataset provenance changed; choose a fresh --out')
    atomic_text(manifest,json.dumps(config,indent=2)+'\n')
    env = make_env(FAMILY,'PositionAndOrientation',render=True)
    start = time.monotonic()
    try:
        with (out/'episodes.jsonl').open('a',buffering=1) as audit,open(os.devnull,'w') as sink:
            for seed in range(a.first_seed,a.first_seed+a.count):
                path = out/f'ep_{seed}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:meta=json.loads(str(z['meta']))
                    if meta.get('config')!=config or meta.get('seed')!=seed:
                        raise ValueError(f'Existing shard provenance mismatch: {path}')
                    print(json.dumps(dict(seed=seed,resumed=True)),flush=True)
                    continue
                with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
                    frames,row = collect(env,seed,spec.max_steps(FAMILY,20),a.xyz_mm,a.angle_deg,a.ctl_kwargs)
                save_episode(path,frames,dict(config=config,**row))
                row.update(path=str(path),elapsed_wall_s=time.monotonic()-start)
                audit.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    finally:
        env.close()


if __name__ == '__main__':main()
