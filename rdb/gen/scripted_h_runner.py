"""Audited CLI for human-style privileged teachers; never a submission policy."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from rdb.gen.scripted_common import run_episode
from rdb.gen.scripted_pack_box_h import thermal_guard


def main(family, expert_cls, search_range):
    from rdb.gen.episode_io import EpisodeRecorder, episode_path, make_env, variant_name
    from roboeval.challenge import results as res
    ap = argparse.ArgumentParser(description=expert_cls.__doc__)
    ap.add_argument('--variation', default='PositionAndOrientation', choices=['PositionAndOrientation'])
    ap.add_argument('--seeds', default=search_range)
    ap.add_argument('--noise', type=float, default=.25)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--out', default='data/gen/scripted_h')
    ap.add_argument('--stats', type=Path)
    args = ap.parse_args()
    start, stop = map(int, args.seeds.split(':'))
    if start < 100000 or stop <= start or not np.isfinite(args.noise) or args.noise < 0:
        ap.error('Require a nonempty seed range >=100000 and finite nonnegative noise')
    # Hash controller and its local dependencies, including unchanged shared machinery.
    files = {Path(__file__), Path(inspect.getfile(expert_cls))}
    from rdb.gen import scripted_common, episode_io, scripted_lift_pot, scripted_books, scripted_pack_box_h
    files.update(Path(m.__file__) for m in (scripted_common, episode_io, scripted_lift_pot, scripted_books, scripted_pack_box_h))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    code_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    variant = variant_name(family, args.variation)
    stats_path = args.stats or Path(f'runs/pot_book/{family}_{"dry" if args.dry else "generation"}_{start}_{stop}.json')
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
                if any(meta.get(k) != v for k,v in dict(script_sha256=code_sha,seed=seed,noise=args.noise,source='scripted_h',family=family,success=True).items()):
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
                              metrics=res.episode_metrics(res.to_jsonable(info),steps,20))
                if rec is not None and status == 'success':
                    rec.save(path,dict(source='scripted_h',expert=expert_cls.__name__,family=family,variation=args.variation,
                                       variant=variant,seed=seed,success=True,steps=steps,noise=args.noise,
                                       script_sha256=code_sha,retries=ex.retries))
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
