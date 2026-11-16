"""
IntelliSignal - Stage 2
rl/reward.py

Defines WHAT THE AGENT IS REWARDED FOR.

The reward is deliberately simple, because a simple reward is one that can be
explained and defended:

    reward = -scale * (  waiting_time_weight * total_waiting_time
                       + queue_weight        * total_queue_length
                       + fairness_weight     * worst_approach_waiting )

  * total_waiting_time   summed current waiting time over the four approaches,
                         in vehicle-seconds
  * total_queue_length   halting vehicles over the four approaches
  * worst_approach_waiting  the waiting time of the single worst approach;
                         the fairness term, switched off by default (weight 0)

The reward is always negative or zero: congestion is a cost, and the best the
agent can do is reduce it towards zero. Less waiting and shorter queues mean a
reward closer to zero, so maximising the reward means reducing congestion.

Averaged over the interval
--------------------------
The values are averaged over every simulation second of the decision interval,
not sampled once at the end. A single snapshot would be noisy - a queue can
empty in the last second of a green - and a noisy reward makes learning
harder. Averaging costs nothing and gives a far more stable signal.

`scale` only keeps the numbers small (a raw sum can be in the hundreds).
It does not change which action is best, because it multiplies every reward
by the same positive constant.

Why max_green matters here
--------------------------
With fairness_weight left at 0 the reward alone could, in principle, be
maximised by serving only the busiest direction. The environment prevents
that structurally: rl/action.py forces a switch once a green has run for
max_green seconds. Fairness is therefore guaranteed by the environment rather
than bought with a reward term, which keeps the reward easy to explain. The
fairness weight is available for experiments in a later stage.
"""

from __future__ import annotations

from typing import Any, Dict

from rl.state import APPROACH_ORDER


class RewardFunction:
    """Turns averaged traffic measurements into a single number."""

    def __init__(self, waiting_time_weight: float = 1.0,
                 queue_weight: float = 1.0,
                 fairness_weight: float = 0.0,
                 scale: float = 0.01):
        self.waiting_time_weight = float(waiting_time_weight)
        self.queue_weight = float(queue_weight)
        self.fairness_weight = float(fairness_weight)
        self.scale = float(scale)

    # ------------------------------------------------------------------
    def __call__(self, averaged: Dict[str, Any]) -> Dict[str, float]:
        """Compute the reward and its components.

        `averaged` is {"queues": {approach: value}, "waiting": {...}} holding
        the mean over the decision interval.
        """
        queues = averaged["queues"]
        waiting = averaged["waiting"]

        total_queue = float(sum(queues[a] for a in APPROACH_ORDER))
        total_waiting = float(sum(waiting[a] for a in APPROACH_ORDER))
        worst_waiting = float(max(waiting[a] for a in APPROACH_ORDER))

        waiting_penalty = self.waiting_time_weight * total_waiting
        queue_penalty = self.queue_weight * total_queue
        fairness_penalty = self.fairness_weight * worst_waiting

        reward = -self.scale * (waiting_penalty + queue_penalty + fairness_penalty)

        return {
            "reward": float(reward),
            "total_waiting_time": total_waiting,
            "total_queue_length": total_queue,
            "worst_approach_waiting": worst_waiting,
            "waiting_penalty": float(-self.scale * waiting_penalty),
            "queue_penalty": float(-self.scale * queue_penalty),
            "fairness_penalty": float(-self.scale * fairness_penalty),
        }

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return ("reward = -{s} x ({w} x waiting + {q} x queue + {f} x worst_approach)"
                .format(s=self.scale, w=self.waiting_time_weight,
                        q=self.queue_weight, f=self.fairness_weight))
