"""Roll out trained policies on fresh seeds and keep the successful episodes (filtered self-imitation).

  MUJOCO_GL=egl python -m rdb.gen.collect_rollouts \
      --submissions subs/v1_flow subs/v1_flow_b128 --out data/gen --source rollout_v1 \
      --plan StackTwoBlocks:300 PackBox:300 LiftPot:150

Each family is run on its evaluated variation (PositionAndOrientation) with the challenge's own
env builder and policy loader (roboeval.challenge.submission), so rollouts match evaluation.
Seeds are 200000 + 100000*family_index + k (never the official 1000-1049 or seeds/dev50.json);
episode k uses submission k % len(submissions). Every attempt is appended to
<out>/<source>/<Variation>/attempts.jsonl, which also makes reruns resume where they stopped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from roboeval.challenge import spec
from roboeval.challenge.evaluate import _seed_policy_rngs
from roboeval.challenge.submission import EpisodeSpec, load_interface, load_metadata, load_policy, make_env, task_instruction

from rdb.common import FAMILIES
from rdb.gen.episode_io import episode_path, obs_frame, save_episode

ROOT = Path(__file__).resolve().parents[2]
VARIATION = "PositionAndOrientation"
SEED_BASE, SEED_FAMILY_STRIDE = 200_000, 100_000


def forbidden_seeds() -> set:
    dev = ROOT / "seeds" / "dev50.json"
    return set(spec.EVAL_SEEDS) | (set(json.loads(dev.read_text())) if dev.exists() else set())


def rollout(env, policy, episode: EpisodeSpec):
    """Same loop as roboeval.challenge.evaluate.run_episode, also recording what the policy saw."""
    frames, actions, termination = [], [], "budget"
    obs, info = env.reset(seed=episode.seed)
    _seed_policy_rngs(episode.policy_seed, episode.seed)
    policy.reset(episode)
    for _ in range(episode.max_steps):
        frames.append(obs_frame(obs))
        action = np.asarray(policy.act(obs), dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        actions.append(action)
        if env.success:
            termination = "success"
            break
        if terminated:
            termination = "fail"
            break
        if truncated:
            termination = "unhealthy"
            break
    return frames, actions, termination


def collect_family(family: str, target: int, max_attempts: int, subs, out: Path, source: str, k0: int = 0) -> None:
    variation_dir = f"{family}{VARIATION}"
    log_path = out / source / variation_dir / "attempts.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = [json.loads(l) for l in log_path.read_text().splitlines()] if log_path.exists() else []
    tried = {d["seed"] for d in done}
    n_ok = sum(d["success"] for d in done)
    bad = forbidden_seeds()

    interface = load_interface(subs[0][0])
    env = make_env(family, VARIATION, interface)
    max_steps = spec.max_steps(family, interface.control_frequency)
    instruction = task_instruction(family)
    base = SEED_BASE + SEED_FAMILY_STRIDE * FAMILIES.index(family)
    k, t0 = k0, time.time()
    while n_ok < target and len(tried) < max_attempts:
        seed = base + k
        k += 1
        if seed in tried:
            continue
        assert seed not in bad and seed >= SEED_BASE
        sub_dir, policy, policy_seed = subs[(seed - base) % len(subs)]
        episode = EpisodeSpec(family=family, variation=VARIATION, env_name=type(env).__name__,
                              instruction=instruction, episode_index=seed - base, seed=seed,
                              policy_seed=policy_seed, control_frequency=interface.control_frequency,
                              max_steps=max_steps, action_space=env.action_space)
        try:
            frames, actions, term = rollout(env, policy, episode)
        except Exception as e:  # a broken sim state: log it, rebuild the env, move on
            frames, actions, term = [], [], f"error: {e!r}"[:200]
            env.close()
            env = make_env(family, VARIATION, interface)
        ok = term == "success"
        if ok:
            save_episode(episode_path(out, source, variation_dir, seed), frames, actions, {
                "source": source, "family": family, "variation": VARIATION, "seed": seed,
                "policy_seed": policy_seed, "policy": sub_dir.name, "success": True, "steps": len(actions),
                "obs_convention": "obs[t] seen when choosing actions[t]"})
        rec = {"seed": seed, "policy": sub_dir.name, "success": ok, "termination": term, "steps": len(actions)}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        tried.add(seed)
        n_ok += ok
        print(f"{family} seed {seed} {sub_dir.name:14s} {term:9s} {len(actions):4d} steps | "
              f"{n_ok}/{target} ok of {len(tried)} tried | {time.time() - t0:.0f}s", flush=True)
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", nargs="+", required=True, help="built submission dirs (rdb.build_submission)")
    ap.add_argument("--out", default=str(ROOT / "data" / "gen"))
    ap.add_argument("--source", default="rollout_v1")
    ap.add_argument("--plan", nargs="+", required=True, help="Family:target_successes, run in order")
    ap.add_argument("--max_attempts_factor", type=float, default=4.0)
    ap.add_argument("--k0", type=int, default=0, help="first seed offset within the family's range (new rounds: new layouts)")
    a = ap.parse_args()

    subs = []
    for s in a.submissions:
        d = Path(s).resolve()
        meta = load_metadata(d)
        subs.append((d, load_policy(d, meta), int(meta.get("policy_seed", 0))))
    for item in a.plan:
        family, _, target = item.partition(":")
        target = int(target)
        collect_family(family, target, int(target * a.max_attempts_factor), subs, Path(a.out), a.source, a.k0)


if __name__ == "__main__":
    main()
