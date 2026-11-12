"""Human-style closed-loop shelf teacher; privileged training data only.

The pick part is PickBookHumanExpert (side nudge to an overhang, pitched pinch,
grasp checks with retries). Then carry the book between the shelf boards, slide
it over the lower board, lower, release and withdraw. A dropped book restarts
the pick. Never use this teacher in a submission.

  python -m rdb.gen.scripted_shelf_h --seeds 105600:105620 --dry --noise 0
"""
from __future__ import annotations

from rdb.gen.scripted_common import Seg
from rdb.gen.scripted_pick_book_h import PickBookHumanExpert


class ShelfHumanExpert(PickBookHumanExpert):
    PLACE = True
    CARRY_Z = 1.10        # between the boards (lower top 1.026, upper bottom 1.255)
    SHELF_GAP = 0.012
    QUIET = PickBookHumanExpert.QUIET + ('release',)

    def plan(self):
        super().plan()
        self.board = self._board()
        seq = self.stages[self.arm]
        self.stages[self.arm] = seq[:seq.index('lift') + 1] + ['raise', 'slide', 'place', 'release', 'back', 'done']

    def _board(self):
        import mujoco
        o = self.env.book_shelf.lower_shelf_body
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, o.mjcf.full_identifier)

    def segment(self, i, stage):
        if i != self.arm or stage not in ('raise', 'slide', 'place', 'release', 'back', 'done'):
            return super().segment(i, stage)
        p, y, t = self.pinch(i), self.yaw[i], self.tilt[i]
        blo, bhi = self.geom_aabb(self.board)
        if stage == 'raise':
            return Seg(p, [min(p[0], blo[0] - .10), p[1], self.CARRY_Z], y, y, 1., t, t, min_steps=12, speed=.45)
        if stage == 'slide':
            return Seg(p, [blo[0] - self.SHELF_GAP, p[1], self.CARRY_Z], y, y, 1., t, t, speed=.6)
        if stage == 'place':
            lo, _ = self.book_box()
            drop = max((lo[2] - bhi[2]) - .004, 0.)
            return Seg(p, p + [0, 0, -drop], y, y, 1., t, t, min_steps=6, speed=.4)
        if stage == 'release':
            return Seg(p, p, y, y, 0., t, t, min_steps=1, hold=6)
        if stage == 'back':
            return Seg(p, p + [-.12, 0, 0.], y, y, 0., t, t, min_steps=8)
        return self.hold_seg(i, 0., steps=12)

    def advance(self, i, stage):
        if i == self.arm and stage in ('raise', 'slide', 'place') and not self.holding(i, self.book):
            self.retries[i] += 1
            self.direct = False
            return 0
        if i == self.arm and stage == 'done' and not self.env.success:
            # Book is not on the board (fell back or missed it): start the pick again.
            self.retries[i] += 1
            self.direct = False
            return 0
        if i == self.arm and stage in ('release', 'back', 'done'):
            return None
        return super().advance(i, stage)

    def act(self):
        st = self.stage(self.arm)
        if st in ('raise', 'slide') and not self.holding(self.arm, self.book):
            self.retries[self.arm] += 1
            self.direct = False
            self.k[self.arm] = 0
            self.segs[self.arm] = None
        return super().act()


if __name__ == '__main__':
    from rdb.gen.scripted_h_runner import main
    main('StackSingleBookShelf', ShelfHumanExpert, '105600:105620')
