"""
IntelliSignal - Stage 3
agents/replay_buffer.py

THE REPLAY BUFFER - the agent's memory of what happened.

What it stores
--------------
One entry per decision the agent made, called a transition:

    state       the 10-number observation before the decision
    action      0 = keep the current green, 1 = switch
    reward      the congestion reward that followed
    next_state  the observation after the decision interval
    done        True when the episode ended on this transition

Why it exists
-------------
Consecutive traffic observations are almost identical - the queue one decision
later looks very much like the queue now. Training a network on such a stream
in order makes the updates strongly correlated and the learning unstable.
The buffer breaks that in two ways:

  1. Sampling a RANDOM mini-batch mixes decisions from different moments and
     different episodes, so one gradient step is not dominated by one traffic
     situation.
  2. Every transition can be used more than once, so a single SUMO run - which
     is the expensive part - teaches the network several times.

It is a fixed-size ring buffer: when it is full the oldest transition is
dropped, so memory use is bounded no matter how long training runs.
50,000 transitions of 10 floats each is a few megabytes.
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Optional, Tuple

import numpy as np

#: One stored decision. Plain and inspectable, which keeps the tests readable.
Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))

#: What `sample()` returns: five NumPy arrays, all with the batch as dimension 0.
Batch = namedtuple("Batch",
                   ("states", "actions", "rewards", "next_states", "dones"))


class ReplayBuffer:
    """A fixed-capacity memory of transitions with uniform random sampling."""

    def __init__(self, capacity: int = 50000, seed: Optional[int] = None):
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive, got {}.".format(capacity))
        self.capacity = int(capacity)
        self._storage = deque(maxlen=self.capacity)
        self._random = random.Random(seed)

    # ------------------------------------------------------------------
    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        """Store one transition. Oldest is dropped once the buffer is full."""
        self._storage.append(Transition(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    #: `remember` reads better from the agent, `push` from the tests.
    remember = push

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Batch:
        """A uniformly random mini-batch, as five arrays.

            states      (batch_size, observation_size)  float32
            actions     (batch_size,)                   int64
            rewards     (batch_size,)                   float32
            next_states (batch_size, observation_size)  float32
            dones       (batch_size,)                   float32  (0.0 / 1.0)

        Sampling is WITHOUT replacement inside one batch, so the same
        transition cannot be counted twice in a single gradient step.
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive, got {}.".format(batch_size))
        if batch_size > len(self._storage):
            raise ValueError(
                "Cannot sample {} transitions from a buffer holding {}. "
                "Training must not start before the buffer has enough "
                "experience - see training.learning_starts."
                .format(batch_size, len(self._storage)))

        transitions = self._random.sample(list(self._storage), batch_size)
        return Batch(
            states=np.stack([t.state for t in transitions]).astype(np.float32),
            actions=np.asarray([t.action for t in transitions], dtype=np.int64),
            rewards=np.asarray([t.reward for t in transitions], dtype=np.float32),
            next_states=np.stack([t.next_state for t in transitions]).astype(np.float32),
            dones=np.asarray([t.done for t in transitions], dtype=np.float32),
        )

    # ------------------------------------------------------------------
    def can_sample(self, batch_size: int) -> bool:
        return len(self._storage) >= int(batch_size)

    def clear(self) -> None:
        self._storage.clear()

    @property
    def is_full(self) -> bool:
        return len(self._storage) >= self.capacity

    def __len__(self) -> int:
        return len(self._storage)

    def __iter__(self):
        return iter(self._storage)

    def __repr__(self) -> str:
        return "<ReplayBuffer {}/{} transitions>".format(
            len(self._storage), self.capacity)

    # ------------------------------------------------------------------
    def statistics(self) -> Tuple[float, float]:
        """(mean reward, min reward) of everything stored - a sanity check."""
        if not self._storage:
            return 0.0, 0.0
        rewards = [t.reward for t in self._storage]
        return float(np.mean(rewards)), float(np.min(rewards))
