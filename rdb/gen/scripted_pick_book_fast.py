"""Fast, human-timed, closed-loop table-book teacher for training data only.

Keep the measured side nudge and pitched overhang pinch of PickBookHumanExpert,
but execute the open-hand approach and transfer at demonstrator-like speed.
The inherited contact, grasp, slip, and retry checks remain in force.
"""
from __future__ import annotations

import math

from rdb.gen.scripted_pick_book_h import PickBookHumanExpert


class PickBookFastExpert(PickBookHumanExpert):
    PUSH_STOP = .018
    # A compact approach gives the robot time to follow each target while
    # removing the long pauses between phases in the original controller.
    SPEEDUP = {
        'side_over': 1.8,
        'side_down': 2.0,
        'push': 1.5,
        'clear': 1.35,
        'clearup': 1.75,
        'over': 1.8,
        'front': 1.25,
        'insert': 1.5,
    }
    MIN_FRAMES = {
        'side_over': 8,
        'side_down': 9,
        'push': 10,
        'clear': 6,
        'clearup': 4,
        'over': 8,
        'front': 10,
        'insert': 7,
    }

    def plan(self):
        super().plan()
        # The left-arm side nudge carries farther after the command stops;
        # stopping it earlier keeps the book supported during the pinch.
        self.PUSH_STOP = .014 if self.arm == 0 else .018

    def advance(self, i, stage):
        target = super().advance(i, stage)
        if i == self.arm and self.retries[i]:
            # A missed pinch often shifts the book back onto the counter.
            # Give the recovery push enough travel to restore the overhang.
            self.PUSH_STOP = .030
        return target

    def segment(self, i, stage):
        seg = super().segment(i, stage)
        if i != self.arm:
            return seg
        if stage in self.SPEEDUP:
            hold = seg.n - seg.move
            seg.move = max(self.MIN_FRAMES[stage], math.ceil(seg.move / self.SPEEDUP[stage]))
            seg.n = seg.move + hold
        return seg

    def until(self, i, stage):
        if i == self.arm and stage == 'close' and self.direct:
            # A settled frontal approach permits the demonstrated short
            # close interval while still requiring measured pad contact.
            return self.segs[i].k >= 6 and self.holding(i, self.book)
        return super().until(i, stage)


if __name__ == '__main__':
    from rdb.gen.scripted_h_runner import main
    main('PickSingleBookFromTable', PickBookFastExpert, '105400:105460')
