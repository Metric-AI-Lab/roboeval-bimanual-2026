"""Book-task data generators using privileged book poses and demonstrated scripts.

The default controller chooses the closest successful human demonstration by
the book's starting x, y and yaw, then runs that action script on the new reset.
This produces new physical rollouts on new seeds. Earlier hand-designed grasp
controllers remain below as experimental alternatives.

  python -m rdb.gen.scripted_books --family PickSingleBookFromTable --seeds 105000:105030 --dry
  python -m rdb.gen.scripted_books --family StackSingleBookShelf --seeds 105000:105030 --dry
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from rdb.gen.scripted_common import BimanualScript, Seg, cli, yaw_of

PITCH = np.array([0.0, -np.pi / 2, 0.0])     # tilt: hand points +x, fingers close along z


class BookExpert(BimanualScript):
    PLACE = False         # True for the shelf task
    OVERHANG = 0.06       # push until the book overhangs the counter edge by this much
    PRESS = 0.012         # press the closed fingers this far into the book top while dragging
    DRAG_X = 0.43         # drag contact x (a vertical hand can't get lower than the shelf board beyond ~0.47)
    DRAG_AIM = 0.20       # position-controlled arm: aiming past the goal gives the force to overcome friction
    DRAG_SPEED = 0.4
    GRIP_IN = 0.025       # pinch point this far inside the book's near edge
    LIFT = 0.12
    CARRY_Z = 1.10        # between the boards (lower top 1.026, upper bottom 1.255)
    SHELF_GAP = 0.012     # fingers this far in front of the lower board's front edge
    CLOSE_HOLD = 8

    def plan(self):
        import mujoco
        bs = self.env.book_shelf
        gid = lambda o: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, o.mjcf.full_identifier)
        self.counter = gid(bs.counter)
        self.board = gid(bs.lower_shelf_body) if self.PLACE else None
        self.book = self.env.books[0]
        self.bg = self.geoms_of(self.book.body.mjcf.full_identifier)[0]
        self.arm = 0 if self.book.body.get_position()[1] >= 0 else 1
        seq = ["turn", "over", "behind", "lower", "push", "up", "pitch", "front", "insert", "close", "lift"]
        if self.PLACE:
            seq += ["raise", "slide", "place", "release", "back"]
        self.stages = [["done"], ["done"]]
        self.stages[self.arm] = seq + ["done"]

    # ---- geometry --------------------------------------------------------------------------------
    def book_box(self):
        return self.geom_aabb(self.bg)

    def edge(self) -> float:
        return float(self.geom_aabb(self.counter)[0][0])        # counter's near edge (x)

    def top(self) -> float:
        return float(self.geom_aabb(self.counter)[1][2])

    def segment(self, i: int, st: str) -> Seg:
        p, y, t = self.pinch(i), self.yaw[i], self.tilt[i]
        lo, hi = self.book_box()
        c = self.book.body.get_position()
        zt = self.top()
        mid_z = (lo[2] + hi[2]) / 2
        side = -1.0 if i == 0 else 1.0                         # mirrored, like the human demos
        if st == "turn":                                       # hand's long side along y: clears the shelf board
            return Seg(p, p + [-0.03, 0, 0], y, side * np.pi / 2, 1.0, t, np.zeros(3))
        if st == "over":                                       # above the book (reachable part), gripper closed
            return Seg(p, [self.drag_x(), c[1], max(min(p[2], 0.95), hi[2] + 0.08)], y, y, 1.0, t, t)
        if st == "behind":
            return Seg(p, [self.drag_x(), c[1], hi[2] + 0.03], y, y, 1.0, t, t, speed=0.7)
        if st == "lower":                                      # press the closed fingers onto the book top
            return Seg(p, [p[0], p[1], hi[2] - self.PRESS], y, y, 1.0, t, t, min_steps=6, speed=0.5)
        if st == "push":                                       # drag toward the robot (aim past the goal, stop at it)
            return Seg(p, p + [-self.DRAG_AIM, 0, -0.004], y, y, 1.0, t, t, min_steps=10, speed=self.DRAG_SPEED)
        if st == "up":
            return Seg(p, p + [0.04, 0, 0.10], y, y, 0.0, t, t, min_steps=6)
        if st == "pitch":                                      # pitch in free space, in front of the book
            return Seg(p, [lo[0] - 0.12, c[1], mid_z + 0.06], y, 0.0, 0.0, t, PITCH)
        if st == "front":
            return Seg(p, [lo[0] - 0.08, self.grip_y(), mid_z], y, y, 0.0, t, PITCH, speed=0.7)
        if st == "insert":                                     # fingers straddle the overhang
            return Seg(p, [lo[0] + self.GRIP_IN, self.grip_y(), mid_z], y, y, 0.0, t, PITCH, min_steps=8, speed=0.5)
        if st == "close":
            return Seg(p, p, y, y, 1.0, t, t, min_steps=1, hold=self.CLOSE_HOLD)
        if st == "lift":
            return Seg(p, p + [-0.02, 0, self.LIFT], y, y, 1.0, t, t, min_steps=8, speed=0.6)
        if st == "raise":
            return Seg(p, [p[0], p[1], self.CARRY_Z], y, y, 1.0, t, t)
        if st == "slide":                                      # book over the board, fingers in front of it
            blo, bhi = self.geom_aabb(self.board)
            return Seg(p, [blo[0] - self.SHELF_GAP, p[1], self.CARRY_Z], y, y, 1.0, t, t, speed=0.6)
        if st == "place":                                      # book bottom onto the board
            blo, bhi = self.geom_aabb(self.board)
            drop = (lo[2] - bhi[2]) - 0.004
            return Seg(p, p + [0, 0, -max(drop, 0.0)], y, y, 1.0, t, t, min_steps=6, speed=0.4)
        if st == "release":
            return Seg(p, p, y, y, 0.0, t, t, min_steps=1, hold=6)
        if st == "back":
            return Seg(p, p + [-0.12, 0, 0.0], y, y, 0.0, t, t, min_steps=8)
        return Seg(p, p, y, y, self.grip[i], t, t, min_steps=1, hold=1000)

    def drag_x(self) -> float:
        lo, hi = self.book_box()
        return float(min(self.DRAG_X, hi[0] - 0.03))

    def grip_y(self) -> float:
        """y of the overhanging part's middle (points of the book in front of the counter edge)."""
        lo, hi = self.book_box()
        e = self.edge()
        R = self.d.geom_xmat[self.bg].reshape(3, 3)
        h = self.m.geom_aabb[self.bg][3:]
        cc = self.m.geom_aabb[self.bg][:3]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + cc
        w = corners @ R.T + self.d.geom_xpos[self.bg]
        front = w[w[:, 0] < e - 0.01]
        return float(front[:, 1].mean()) if len(front) else float(self.book.body.get_position()[1])

    def until(self, i: int, st: str) -> bool:
        return st == "push" and self.book_box()[0][0] <= self.edge() - self.OVERHANG

    def advance(self, i: int, st: str):
        if st == "close" and not self.holding(i, self.book):
            self.grip[i] = 0.0
            return self.stages[i].index("front")
        return None


HOOK = np.array([0.0, -np.pi / 4, 0.0])      # tilt for hooking: fingers point forward-down


class HookBookExpert(BookExpert):
    """Pull the book toward the robot with a hand pitched -45 deg reaching over it (a vertical hand's
    housing hits the shelf board): closed fingertips go down just behind the book's far face and pull,
    until it overhangs the counter edge by OVERHANG; then the pitched pinch of BookExpert."""

    HOOK_BEHIND = 0.02
    HOOK_Z = 0.010        # fingertip height above the counter top

    def plan(self):
        super().plan()
        seq = ["hookturn", "hookover", "hookdown", "push", "up", "pitch", "front", "insert", "close", "lift"]
        if self.PLACE:
            seq += ["raise", "slide", "place", "release", "back"]
        self.stages = [["done"], ["done"]]
        self.stages[self.arm] = seq + ["done"]

    def segment(self, i: int, st: str) -> Seg:
        p, y, t = self.pinch(i), self.yaw[i], self.tilt[i]
        lo, hi = self.book_box()
        c = self.book.body.get_position()
        if st == "hookturn":
            return Seg(p, [lo[0] - 0.02, c[1], hi[2] + 0.15], y, 0.0, 1.0, t, HOOK)
        if st == "hookover":
            return Seg(p, [hi[0] + self.HOOK_BEHIND, c[1], hi[2] + 0.05], y, y, 1.0, t, t, speed=0.7)
        if st == "hookdown":
            return Seg(p, [p[0], p[1], self.top() + self.HOOK_Z], y, y, 1.0, t, t, min_steps=6, speed=0.5)
        if st == "push":
            return Seg(p, p + [-self.DRAG_AIM, 0, 0], y, y, 1.0, t, t, min_steps=10, speed=self.DRAG_SPEED)
        if st == "up":
            return Seg(p, p + [0.0, 0, 0.08], y, y, 0.0, t, t, min_steps=6)
        return super().segment(i, st)


class RollBookExpert(BookExpert):
    """Pinch the strip of the book that already overhangs the counter edge, with the hand pointing
    sideways (toward the robot's centre line) and fingers closing vertically: only a finger's width,
    not its length, has to fit in front of the edge. Left: yaw -90, roll -90 deg; right: mirrored."""

    STRIP_IN = 0.022      # pinch this far in from the strip's outer side (along y)

    def plan(self):
        super().plan()
        seq = ["turn", "side", "insert", "close", "lift"]
        if self.PLACE:
            seq += ["raise", "slide", "place", "release", "back"]
        self.stages = [["done"], ["done"]]
        self.stages[self.arm] = seq + ["done"]

    def strip(self):
        """(x, y_outer, y_inner) of the overhanging strip: x mid-strip, y extent of book points in front
        of the counter edge, outer = the arm's side."""
        e = self.edge()
        R = self.d.geom_xmat[self.bg].reshape(3, 3)
        h, cc = self.m.geom_aabb[self.bg][3:], self.m.geom_aabb[self.bg][:3]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + cc
        w = corners @ R.T + self.d.geom_xpos[self.bg]
        lo = w[:, 0].min()
        x = (lo + e) / 2
        # y range of the book's cross-section at x (sample the projected outline)
        pts = []
        for a in range(8):
            for b in range(8):
                if a == b:
                    continue
                p, q = w[a], w[b]
                if (p[0] - x) * (q[0] - x) <= 0 and p[0] != q[0]:
                    s = (x - p[0]) / (q[0] - p[0])
                    pts.append(p[1] + s * (q[1] - p[1]))
        ys = np.array(pts) if pts else np.array([self.book.body.get_position()[1]])
        sgn = 1.0 if self.arm == 0 else -1.0
        outer = ys.max() if sgn > 0 else ys.min()
        inner = ys.min() if sgn > 0 else ys.max()
        return x, outer, inner, e - lo

    def segment(self, i: int, st: str) -> Seg:
        p, y, t = self.pinch(i), self.yaw[i], self.tilt[i]
        sgn = 1.0 if i == 0 else -1.0
        roll = np.array([-np.pi / 2 if i == 0 else np.pi / 2, 0.0, 0.0])
        lo, hi = self.book_box()
        mid_z = (lo[2] + hi[2]) / 2
        if st in ("turn", "side", "insert"):
            x, outer, inner, over = self.strip()
            if st == "turn":                                   # roll in free space, beside and above the strip
                return Seg(p, [x, outer + sgn * 0.10, mid_z + 0.10], y, -sgn * np.pi / 2, 0.0, t, roll)
            if st == "side":
                return Seg(p, [x, outer + sgn * 0.06, mid_z], y, y, 0.0, t, t, speed=0.6)
            return Seg(p, [x, outer - sgn * self.STRIP_IN, mid_z], y, y, 0.0, t, t, min_steps=8, speed=0.4)
        if st == "close":
            return Seg(p, p, y, y, 1.0, t, t, min_steps=1, hold=self.CLOSE_HOLD)
        if st == "lift":
            return Seg(p, p + [-0.04, 0, self.LIFT], y, y, 1.0, t, t, min_steps=8, speed=0.5)
        return super().segment(i, st)

    def advance(self, i: int, st: str):
        if st == "close" and not self.holding(i, self.book):
            self.grip[i] = 0.0
            return self.stages[i].index("side")
        return None


class ShelfExpert(BookExpert):
    PLACE = True


class HookShelfExpert(HookBookExpert):
    PLACE = True


class RollShelfExpert(RollBookExpert):
    PLACE = True


_DEMO_LIBRARIES = {}


def _demo_library(family: str):
    """Index successful human trajectories by their seeded starting book poses.

    The reference metadata and training actions use the same episode order.  This
    index is built once per process, using a separate camera-free environment so
    the episode being controlled is never reset by the lookup.
    """
    if family in _DEMO_LIBRARIES:
        return _DEMO_LIBRARIES[family]

    from rdb.gen.episode_io import make_env

    root = Path(__file__).resolve().parents[2]
    refs = json.loads((root / "RoboEval/roboeval/challenge/human_reference_episodes.json").read_text())[family]
    folder = root / "data/s128" / f"{family}PositionAndOrientation"
    episodes = np.load(folder / "episode.npy", mmap_mode="r")
    actions = np.load(folder / "action.npy", mmap_mode="r")
    if int(episodes.max()) + 1 != len(refs):
        raise ValueError(f"Human reference and actions disagree for {family}")

    reference_env = make_env(family, "PositionAndOrientation", render=False)
    examples = []
    try:
        for j, ref in enumerate(refs):
            if not ref["success"]:
                continue
            reference_env.reset(seed=int(ref["seed"]))
            book = reference_env.books[0].body
            pose = np.r_[book.get_position()[:2], yaw_of(book.get_quaternion())]
            trajectory = np.asarray(actions[episodes == j], np.float32)
            if len(trajectory) != ref["dataset_steps"]:
                raise ValueError(f"Human trajectory length disagrees for {family} episode {j}")
            examples.append((j, pose, trajectory))
    finally:
        reference_env.close()
    _DEMO_LIBRARIES[family] = examples
    return examples


class NearestDemoBookExpert:
    """Seeded privileged-state retrieval followed by a demonstrated action script.

    Replaying on a new reset seed gives a new physical rollout and camera record.
    The source trajectory is selected only from training demonstrations that
    succeed when replayed under the challenge rules.
    """

    FAMILY = "PickSingleBookFromTable"

    def __init__(self, env, rng: np.random.Generator, noise: float = 0.0):
        self.rng, self.noise = rng, noise
        book = env.books[0].body
        pose = np.r_[book.get_position()[:2], yaw_of(book.get_quaternion())]
        examples = _demo_library(self.FAMILY)
        scores = []
        for _, source_pose, _ in examples:
            xy = np.linalg.norm(source_pose[:2] - pose[:2])
            yaw = abs(np.angle(np.exp(1j * (source_pose[2] - pose[2]))))
            scores.append(xy + 0.04 * yaw)
        self.source, self.source_pose, self.actions = examples[int(np.argmin(scores))]
        self.index = 1  # run_episode has already issued the demonstration's initial no-op

    @property
    def label(self) -> str:
        return f"demo:{self.source}"

    def act(self) -> np.ndarray:
        if self.index < len(self.actions):
            action = self.actions[self.index].copy()
            self.index += 1
            return action
        action = np.zeros(14, np.float32)
        action[12:] = self.actions[-1, 12:]
        return action

    def perturb(self, action: np.ndarray) -> np.ndarray:
        if self.noise <= 0:
            return action
        executed = action.copy()
        for i in range(2):
            if np.linalg.norm(action[6 * i:6 * i + 6]) > 0.002:
                executed[6 * i:6 * i + 3] += self.rng.normal(0, 0.0005 * self.noise, 3)
                executed[6 * i + 3:6 * i + 6] += self.rng.normal(0, 0.003 * self.noise, 3)
        return executed


class NearestDemoShelfExpert(NearestDemoBookExpert):
    FAMILY = "StackSingleBookShelf"


if __name__ == "__main__":
    fam = "PickSingleBookFromTable"
    if "--family" in sys.argv:
        k = sys.argv.index("--family")
        fam = sys.argv[k + 1]
        del sys.argv[k:k + 2]
    if not any(arg == "--noise" or arg.startswith("--noise=") for arg in sys.argv):
        # Direct trajectory replay is most reliable without DART perturbations.
        sys.argv.extend(["--noise", "0"])
    cli(fam, {"*": NearestDemoShelfExpert if fam == "StackSingleBookShelf" else NearestDemoBookExpert},
        "100000:100050")
