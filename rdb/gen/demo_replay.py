"""Nearest-demo replay expert for any task family (generalises rdb.gen.scripted_books.NearestDemoBookExpert).

Privileged state is used only to pick the human demonstration whose task object started closest
(x, y, yaw) to the new episode's; its recorded 20 Hz actions are then replayed open loop on the new
seed. Only successful replays are kept by the shared runner, so this yields human-style episodes with
new observations and dynamics. Training data only: the policy never sees privileged state.

  MUJOCO_GL=egl python -m rdb.gen.demo_replay --family PackBox --variation PositionAndOrientation \
      --seeds 105000:105030 --dry
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from rdb.gen.scripted_common import cli, yaw_of

OBJECT = {   # env attribute holding the task object whose start pose varies
    "PackBox": "packing_box", "LiftPot": "kitchenpot", "LiftTray": "breakfast_tray", "CubeHandover": "cube",
    "StackSingleBookShelf": "books", "PickSingleBookFromTable": "books",
}
YAW_WEIGHT = 0.04   # metres per radian when comparing start poses
_LIBS: dict = {}


def object_pose(env, family: str) -> np.ndarray:
    obj = getattr(env, OBJECT[family])
    body = (obj[0] if isinstance(obj, (list, tuple)) else obj).body
    return np.r_[body.get_position()[:2], yaw_of(body.get_quaternion())]


def demo_library(family: str) -> list:
    """(episode index, start pose, actions) for every successful human demo of the evaluated variation."""
    if family in _LIBS:
        return _LIBS[family]
    from rdb.gen.episode_io import make_env

    root = Path(__file__).resolve().parents[2]
    refs = json.loads((root / "RoboEval/roboeval/challenge/human_reference_episodes.json").read_text())[family]
    folder = root / "data/s128" / f"{family}PositionAndOrientation"
    episodes = np.load(folder / "episode.npy", mmap_mode="r")
    actions = np.load(folder / "action.npy", mmap_mode="r")
    if int(episodes.max()) + 1 != len(refs):
        raise ValueError(f"human reference and actions disagree for {family}")
    env = make_env(family, "PositionAndOrientation", render=False)
    lib = []
    try:
        for j, ref in enumerate(refs):
            if not ref["success"]:
                continue
            env.reset(seed=int(ref["seed"]))
            traj = np.asarray(actions[episodes == j], np.float32)
            if len(traj) != ref["dataset_steps"]:
                raise ValueError(f"human trajectory length disagrees for {family} episode {j}")
            lib.append((j, object_pose(env, family), traj))
    finally:
        env.close()
    _LIBS[family] = lib
    return lib


def make_expert(family: str):
    class NearestDemoExpert:
        FAMILY = family

        def __init__(self, env, rng: np.random.Generator, noise: float = 0.0):
            self.rng, self.noise = rng, noise
            pose = object_pose(env, family)
            lib = demo_library(family)
            d = [np.linalg.norm(p[:2] - pose[:2]) + YAW_WEIGHT * abs(np.angle(np.exp(1j * (p[2] - pose[2]))))
                 for _, p, _ in lib]
            self.source, _, self.actions = lib[int(np.argmin(d))]
            self.index = 1   # run_episode already issued the demo's initial no-op

        @property
        def label(self) -> str:
            return f"demo:{self.source}"

        def act(self) -> np.ndarray:
            if self.index < len(self.actions):
                self.index += 1
                return self.actions[self.index - 1].copy()
            a = np.zeros(14, np.float32)
            a[12:] = self.actions[-1, 12:]   # hold the final gripper state
            return a

        def perturb(self, action: np.ndarray) -> np.ndarray:
            return action

    return NearestDemoExpert


if __name__ == "__main__":
    fam = "PackBox"
    if "--family" in sys.argv:
        k = sys.argv.index("--family")
        fam = sys.argv[k + 1]
        del sys.argv[k:k + 2]
    if not any(a == "--noise" or a.startswith("--noise=") for a in sys.argv):
        sys.argv.extend(["--noise", "0"])
    cli(fam, {"*": make_expert(fam)}, "105000:105030")
