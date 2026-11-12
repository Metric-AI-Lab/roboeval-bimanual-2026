"""Offline phase-port checks or CPU oracle-pose verification for the book controllers.

--offline uses stored pose-data shards, without creating an environment.
Default mode is one local, camera-free simulation worker on registered screen
seeds. True live book pose is verifier input ONLY; policy component takes an
estimated pose and standard observations. Never runs challenge portal calls.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from rdb.gen.scripted_common import BimanualScript
from rdb.gen.scripted_pick_book_fast import PickBookFastExpert
from rdb.gen.scripted_shelf_fast import ShelfFastExpert
from rdb.hybrid.book_controller import PickBookController, ShelfBookController

FAMILIES = {'pick': 'PickSingleBookFromTable', 'shelf': 'StackSingleBookShelf'}


class PhaseReference(PickBookFastExpert):
    """Actual teacher phase methods on supplied snapshots, without physics."""

    def __init__(self, policy):
        self.__dict__.update(copy.deepcopy(policy.__dict__))
        self.source = policy
        self.book = SimpleNamespace(body=SimpleNamespace(get_position=lambda: self.source.book_xyz.copy()))

    def cmd_pose(self):
        return self.source.cmd_pose()

    def cmd_pads(self):
        return [p.copy() for p in self.source._pads]

    def pinch(self, i, ee=None):
        return self.source._pads[i].copy()

    def book_box(self):
        return self.source.book_box()

    def target(self):
        return self.source.target()

    def edge(self):
        return self.source.edge()

    def holding(self, i, prop):
        return bool(self.grip[i] >= .5 and .05 <= self.source._openings[i] < .9)

    def act(self):
        # Teacher act(), removing only host thermal polling and diagnostic trace.
        if self.stage(self.arm) == 'lift' and not self.holding(self.arm, self.book):
            self.k[self.arm] = self.stages[self.arm].index('verifylift')
            self.segs[self.arm] = None
        return BimanualScript.act(self)


class ShelfPhaseReference(PhaseReference, ShelfFastExpert):
    def __init__(self, policy):
        super().__init__(policy)
        self.pad_geoms, self.board = [0, 1], 2

    @property
    def d(self):
        return SimpleNamespace(geom_xpos=self.source._measured_pad_centres[self.arm])

    def geom_aabb(self, geom):
        if geom == self.board:
            return self.source.board_bounds
        return self.source._measured_pad_bounds[self.arm, geom]

    def pad_mid(self, arm):
        return self.source.pad_mid(arm)

    def top(self):
        return self.source.top()

    def act(self):
        if self.stage(self.arm) in ('lift', 'raise', 'slide', 'place'):
            self.missing = self.missing+1 if not self.holding(self.arm, self.book) else 0
            if self.missing >= 3:
                self.retries[self.arm] += 1
                self.direct = False
                self.k[self.arm] = 0
                self.segs[self.arm] = None
                self.grip[self.arm] = 0.
                self.missing = 0
        else:
            self.missing = 0
        return BimanualScript.act(self)


def offline(data, count, task):
    shards = sorted(data.glob('ep_*.npz'))[:count]
    if not shards:
        raise ValueError(f'No pose-data shards in {data}')
    actions, phases = 0, set()
    started = time.monotonic()
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            poses = np.concatenate((z['book_xyz'], z['book_quat_wxyz']), axis=1)
            shelf_pose = np.r_[z['shelf_xyz'][0], z['shelf_quat_wxyz'][0]] if task == 'shelf' else None
            observations = [dict(proprioception=z['proprioception'][i],
                                 proprioception_grippers=z['proprioception_grippers'][i]) for i in range(len(poses))]
        policy = ShelfBookController(poses[0], shelf_pose) if task == 'shelf' else PickBookController(poses[0])
        def act(obs, pose):
            return policy.act(obs, pose, shelf_pose) if task == 'shelf' else policy.act(obs, pose)
        np.testing.assert_array_equal(act(observations[0], poses[0]), np.zeros(14, np.float32))
        # Initialize the reference after one ordinary action, before any phase
        # transitions. Subsequent phase state evolves independently.
        act(observations[0], poses[0])
        reference = ShelfPhaseReference(policy) if task == 'shelf' else PhaseReference(policy)
        for index in range(2*len(poses)):
            frame = min(index//2, len(poses)-1)
            obs, pose = observations[frame], poses[frame]
            policy.update_pose(pose)
            policy.observe(obs)
            policy._pads = policy.cmd_pads()
            expected = reference.act()
            actual = act(obs, pose)
            np.testing.assert_array_equal(actual, expected, err_msg=f'{path.name} step {index}')
            assert policy.k == reference.k and policy.retries == reference.retries
            actions += 1
            phases.add(policy.phase)
        print(json.dumps(dict(shard=path.name, arm=policy.arm, compared_actions=2*len(poses))), flush=True)
    print('SUMMARY '+json.dumps(dict(task=task, offline=True, shards=len(shards), exact_actions=actions,
                                    phases=sorted(phases), wall_s=round(time.monotonic()-started, 2))), flush=True)


def live_pose(env):
    body = env.books[0].body
    return np.r_[np.asarray(body.get_position(), float), np.asarray(body.get_quaternion(), float)]


def online(first_seed, count, out, task):
    from absl import logging
    from roboeval.challenge import spec
    from rdb.gen.episode_io import make_env
    from rdb.gen.scripted_pack_box_h import thermal_guard
    logging.set_verbosity(logging.ERROR)
    env = make_env(FAMILIES[task], 'PositionAndOrientation', render=False)
    max_steps = spec.max_steps(FAMILIES[task], 20)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.monotonic()
    try:
        with out.open('w', buffering=1) as output, open(os.devnull, 'w') as sink:
            for seed in range(first_seed, first_seed+count):
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    obs, _ = env.reset(seed=seed)
                    shelf_pose = (np.r_[env.book_shelf.body.get_position(), env.book_shelf.body.get_quaternion()]
                                  if task == 'shelf' else None)
                    policy = ShelfBookController(live_pose(env), shelf_pose) if task == 'shelf' else PickBookController(live_pose(env))
                    status, floor_drop, steps = 'budget', False, 0
                    phases, max_qctrl_error = [], 0.
                    for step in range(max_steps):
                        if step % 20 == 0:
                            thermal_guard()
                        action = (policy.act(obs, live_pose(env), shelf_pose) if task == 'shelf'
                                  else policy.act(obs, live_pose(env)))
                        obs, _, term, trunc, _ = env.step(action)
                        steps += 1
                        if not phases or phases[-1]['phase'] != policy.phase:
                            phases.append(dict(step=steps, phase=policy.phase))
                        floor_drop |= bool(env.books[0].is_colliding(env._floor))
                        actual_qctrl = np.asarray(env.robot.get_initial_joint_positions(), float).reshape(-1)
                        max_qctrl_error = max(max_qctrl_error, float(np.max(np.abs(actual_qctrl-policy.qctrl))))
                        if env.success and (task == 'pick' or policy.grip[policy.arm] < .5):
                            status = 'success'
                            break
                        if not env.success and (term or trunc):
                            status = 'fail' if term else 'unhealthy'
                            break
                    row = dict(seed=seed, status=status, steps=steps, time_s=steps/20,
                               floor_drop=floor_drop, retries=policy.retries, arm=policy.arm,
                               phase=policy.phase, phases=phases, max_qctrl_error_rad=max_qctrl_error)
                output.write(json.dumps(row)+'\n')
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        env.close()
    wins = [r['time_s'] for r in rows if r['status']=='success']
    print('SUMMARY '+json.dumps(dict(task=task, n=len(rows), successes=len(wins), floor_drops=sum(r['floor_drop'] for r in rows),
                                    median_success_s=float(np.median(wins)) if wins else None,
                                    median_all_s=float(np.median([r['time_s'] for r in rows])),
                                    elapsed_wall_s=round(time.monotonic()-started, 1))), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=FAMILIES, default='pick')
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--data', type=Path)
    ap.add_argument('--first-seed', type=int, default=2210000)
    ap.add_argument('--count', type=int, default=30)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if not (1 <= args.count <= 30 and 2210000 <= args.first_seed and args.first_seed+args.count <= 2211000):
        ap.error('Use 1–30 seeds inside registered 2210000–2210999')
    if args.offline:
        offline(args.data or Path(f'data/hybrid/books/{FAMILIES[args.task]}PositionAndOrientation'), args.count, args.task)
    else:
        online(args.first_seed, args.count, args.out or Path(f'runs/hybrid_books/{args.task}_controller_oracle.jsonl'), args.task)


if __name__ == '__main__':
    main()
