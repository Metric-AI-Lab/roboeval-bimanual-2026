"""StackTwoBlocks grab-both teacher, v2: no waits on hidden conditions; privileged training data only.

v1 (`scripted_stack_hold.py`) held each hand still with a closed gripper until "both blocks held" became true. That
condition is not visible in the cameras, and a policy trained on v1 learned "closed and still -> stay still": it
froze until the time limit in 60 % of dev tries (`ab_stk_holdonly`, success 0.40). Here the steady hand releases
as soon as the stacker's gripper has visibly closed (so both are held at that moment), and the stacker only waits
for a VISIBLE event: the steady hand moving back home.

  python -m rdb.gen.scripted_stack_hold2 --seeds 105700:105760 --dry
"""
from __future__ import annotations

from rdb.gen.scripted_stack_hold import HoldStackExpert


class HoldStackExpertV2(HoldStackExpert):
    CLEAR_STEPS = 6       # stacker resumes once the steady hand has retreated this many steps

    def _advance(self, i):
        st = self.st(i)
        if st == 'wait':
            j = self.steady
            clear = self.st(j) == 'done' or (self.st(j) == 'retreat' and self.segs[j] is not None
                                             and self.segs[j].k >= self.CLEAR_STEPS)
            if not clear:
                if not self.holding(i, self.own[i]):
                    self.retries[i] += 1
                    self.k[i] = 0
                    self.grip[i] = 0.0
                return
            self.k[i] += 1
            return
        if st == 'hold':
            a = self.arm
            if self.k[a] <= self.stages[a].index('close'):      # stacker's gripper not closed yet: keep holding
                if not self.holding(i, self.own[i]):
                    self.retries[i] += 1
                    self.k[i] = 0
                    self.grip[i] = 0.0
                return
            self.k[i] += 1
            return
        super()._advance(i)


if __name__ == '__main__':
    from rdb.gen.scripted_h_runner import main
    main('StackTwoBlocks', HoldStackExpertV2, '105700:105760')
