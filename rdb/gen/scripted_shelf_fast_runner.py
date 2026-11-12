"""Shelf-specific runner with full dependency provenance and terminal observations.

Stores a real terminal observation plus one explicitly unused hold placeholder,
so the existing converter retains the physical release action. Shared code is
unchanged; all generated observations use the standard challenge interface.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from rdb.gen.scripted_pack_box_h import thermal_guard


def run_episode(env, family, expert_cls, seed, noise, recorder=None, debug=False):
    """Execute a physical opening before stopping; preserve every actual label.

    The environment permits stepping after its first success predicate. If it
    fires with the hand still commanded closed, immediately open/withdraw and
    require success again. Completion includes this conservative extra motion.
    """
    from roboeval.challenge import spec
    assert seed >= 100000
    obs, info = env.reset(seed=seed)
    zero = np.zeros(14,np.float32)
    if recorder is not None:
        recorder.add(obs,zero)
    obs, _, term, trunc, info = env.step(zero)
    expert = expert_cls(env,np.random.default_rng(seed+7),noise)
    expert.first_success_steps = None
    steps, status, last = 1, 'budget', None
    while status == 'budget' and steps < spec.max_steps(family,20):
        a = expert.act()
        if debug and expert.label != last:
            last = expert.label
            print(f'  step {steps}: {last}',flush=True)
        e = expert.perturb(a)
        if recorder is not None:
            recorder.add(obs,a,e)
        obs, _, term, trunc, info = env.step(e)
        steps += 1
        counts = {k: float(info.get(spec.RAW_METRIC_KEYS[k],0.)) for k in ('ecc','scc','sc')}
        prior = expert.trace[-2].get('metric_counts',{}) if len(expert.trace)>1 else {}
        expert.trace[-1]['metric_counts'] = counts
        if any(counts[k]>prior.get(k,0.) for k in counts):
            import mujoco
            pairs = []
            for c in expert.d.contact:
                names = [mujoco.mj_id2name(expert.m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or ''
                         for g in (c.geom1,c.geom2)]
                if c.dist < 0 and any('panda' in name for name in names):
                    pairs.append(dict(geoms=names,distance=float(c.dist)))
            expert.trace[-1]['contacts_at_metric_event'] = pairs
        if env.success:
            if expert.first_success_steps is None:
                expert.first_success_steps = steps
            if expert.grip[expert.arm] < .5:
                status = 'success'
            else:
                expert.k[expert.arm] = expert.stages[expert.arm].index('release')
                expert.segs[expert.arm] = None
        elif term:
            status = 'fail'
        elif trunc:
            status = 'unhealthy'
    return status,steps,info,expert


def main(family, expert_cls, search_range):
    from rdb.gen.episode_io import EpisodeRecorder, episode_path, make_env, variant_name
    from roboeval.challenge import results as res
    ap = argparse.ArgumentParser(description=expert_cls.__doc__)
    ap.add_argument('--variation', default='PositionAndOrientation', choices=['PositionAndOrientation'])
    ap.add_argument('--seeds', default=search_range)
    ap.add_argument('--noise', type=float, default=.25)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--out', default='data/gen/scripted_hs')
    ap.add_argument('--stats', type=Path)
    args = ap.parse_args()
    start, stop = map(int, args.seeds.split(':'))
    if start < 100000 or stop <= start or not np.isfinite(args.noise) or args.noise < 0:
        ap.error('Require a nonempty seed range >=100000 and finite nonnegative noise')
    # Hash controller and its local dependencies, including unchanged shared machinery.
    files = {Path(__file__), Path(inspect.getfile(expert_cls))}
    from rdb.gen import scripted_common, episode_io, scripted_lift_pot, scripted_books, scripted_pack_box_h, scripted_pick_book_h, scripted_pick_book_fast
    files.update(Path(m.__file__) for m in (scripted_common, episode_io, scripted_lift_pot, scripted_books, scripted_pack_box_h, scripted_pick_book_h, scripted_pick_book_fast))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    code_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    variant = variant_name(family, args.variation)
    stats_path = args.stats or Path(f'runs/shelf_fast/{family}_{"dry" if args.dry else "generation"}_{start}_{stop}.json')
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    if stats_path.exists():
        prior = json.loads(stats_path.read_text())
        if prior['script_sha256'] != code_sha or prior['noise'] != args.noise or prior['seeds'] != [start, stop] or prior['dry'] != args.dry:
            raise ValueError(f'Incompatible audit file; choose a new --stats: {stats_path}')
        records = {r['seed']: r for r in prior['episodes']}
    started = time.time()
    wall_prior = prior['wall_s'] if stats_path.exists() else 0

    def save():
        rows = list(records.values()); success = [r for r in rows if r['status'] == 'success']
        durations = [r['steps'] / 20 for r in success]
        summary = dict(family=family, seeds=[start, stop], noise=args.noise, dry=args.dry,
                       script_sha256=code_sha, dependencies=hashes, attempted=len(rows), successes=len(success),
                       success_rate=len(success)/len(rows) if rows else None,
                       success_time_s={k: float(v) for k, v in zip(('median','mean','p90'),
                           (np.median(durations),np.mean(durations),np.percentile(durations,90)))} if durations else {},
                       wall_s=wall_prior+time.time()-started, episodes=rows)
        tmp = stats_path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(summary, indent=2)); tmp.replace(stats_path)
        return summary

    thermal_guard()
    env = make_env(family, args.variation, render=not args.dry)
    try:
        for seed in range(start, stop):
            path = episode_path(args.out, '', variant, seed)
            if not args.dry and path.exists():
                with np.load(path, allow_pickle=False) as z:
                    meta = json.loads(str(z['meta']))
                if any(meta.get(k) != v for k,v in dict(script_sha256=code_sha,seed=seed,noise=args.noise,source='scripted_hs',family=family,success=True).items()):
                    raise ValueError(f'Existing episode has incompatible provenance: {path}')
                if seed not in records:
                    raise ValueError(f'Existing episode missing audit record: {path}')
                continue
            if seed in records and records[seed]['status'] != 'error' and (args.dry or records[seed]['status'] != 'success'):
                continue
            thermal_guard();rec = None if args.dry else EpisodeRecorder()
            try:
                status, steps, info, ex = run_episode(env, family, expert_cls, seed, args.noise, rec, args.debug)
                record = dict(seed=seed,status=status,steps=steps,retries=ex.retries,trace=ex.trace,
                              first_success_steps=ex.first_success_steps,
                              metrics=res.episode_metrics(res.to_jsonable(info),steps,20))
                if rec is not None and status == 'success':
                    terminal = env.get_observation()
                    placeholder = np.zeros(14, np.float32)
                    placeholder[12:] = ex.grip
                    rec.add(terminal, placeholder)
                    rec.save(path,dict(source='scripted_hs',expert=expert_cls.__name__,family=family,variation=args.variation,
                                       variant=variant,seed=seed,success=True,steps=steps,noise=args.noise,
                                       script_sha256=code_sha,retries=ex.retries,stored_steps=steps+1,
                                       terminal_observation=True,unused_last_action=True,
                                       first_success_steps=ex.first_success_steps))
            except Exception as exc:
                record=dict(seed=seed,status='error',steps=0,error=repr(exc))
                print(record,flush=True);env.close();thermal_guard();env=make_env(family,args.variation,render=not args.dry)
            records[seed]=record
            summary=save()
            print(f'{family} seed {seed}: {record["status"]} {record["steps"]/20:.2f}s; '
                  f'{summary["successes"]}/{summary["attempted"]} successes; {summary["wall_s"]:.0f}s wall',flush=True)
    finally:
        env.close();summary=save()
    print(json.dumps({k:v for k,v in summary.items() if k not in ('episodes','dependencies')}),flush=True)
