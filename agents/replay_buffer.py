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
    next_action_mask  actions allowed by the next state's signal timing

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
from typing import Optional, Sequence, Tuple

import numpy as np

#: One stored decision. Plain and inspectable, which keeps the tests readable.
#:
#: `duration` is how many SECONDS of simulation the decision covered. It is
#: 1.0 for callers that do not model it, which is every Stage 3 call site, so
#: nothing those callers do changes. It matters when decisions differ in
#: length - choosing a green duration, or serving the clearance outside the
#: decision interval - because then the transition is semi-Markov: a decision
#: that occupies eleven seconds must not be discounted like one that occupies
#: one. See `DQNConfig.gamma_per_second`.
#: It defaults to 1.0 so that code constructing a Transition directly - the
#: tests do - keeps working unchanged.
Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done",
                         "next_action_mask", "duration"),
                        defaults=(1.0,))

#: What `sample()` returns: arrays with the batch as dimension 0.
Batch = namedtuple("Batch",
                   ("states", "actions", "rewards", "next_states", "dones",
                    "next_action_masks", "durations"))


class ReplayBuffer:
    """A fixed-capacity memory of transitions with uniform random sampling."""

    def __init__(self, capacity: int = 50000, seed: Optional[int] = None,
                 num_actions: int = 2):
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive, got {}.".format(capacity))
        if int(num_actions) <= 0:
            raise ValueError("num_actions must be positive, got {}.".format(num_actions))
        self.capacity = int(capacity)
        self.num_actions = int(num_actions)
        self._storage = deque(maxlen=self.capacity)
        self._random = random.Random(seed)

    # ------------------------------------------------------------------
    def push(self, state, action: int, reward: float, next_state, done: bool,
             next_valid_actions: Optional[Sequence[int]] = None,
             duration: float = 1.0) -> None:
        """Store one transition. Oldest is dropped once the buffer is full.

        When omitted, `next_valid_actions` defaults to the full action space
        for compatibility with callers that do not model timing constraints,
        and `duration` defaults to one second for callers whose decisions all
        cover the same amount of simulated time.
        """
        duration = float(duration)
        if not duration > 0.0:
            raise ValueError(
                "duration must be positive, got {!r}. A transition covering "
                "no simulated time cannot be discounted.".format(duration))
        if next_valid_actions is None:
            next_action_mask = np.ones(self.num_actions, dtype=np.bool_)
        else:
            next_action_mask = np.zeros(self.num_actions, dtype=np.bool_)
            valid = [int(value) for value in next_valid_actions]
            if not valid:
                raise ValueError("next_valid_actions must contain an action")
            if any(value < 0 or value >= self.num_actions for value in valid):
                raise ValueError("next_valid_actions contains an action outside the action space")
            next_action_mask[valid] = True
        self._storage.append(Transition(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            next_action_mask,
            duration,
        ))

    #: `remember` reads better from the agent, `push` from the tests.
    remember = push

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Batch:
        """A uniformly random mini-batch, including next-action masks.

            states      (batch_size, observation_size)  float32
            actions     (batch_size,)                   int64
            rewards     (batch_size,)                   float32
            next_states (batch_size, observation_size)  float32
            dones       (batch_size,)                   float32  (0.0 / 1.0)
            next_action_masks (batch_size, num_actions) bool
            durations   (batch_size,)                   float32  (seconds)

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
            next_action_masks=np.stack(
                [t.next_action_mask for t in transitions]).astype(np.bool_),
            durations=np.asarray([t.duration for t in transitions],
                                 dtype=np.float32),
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
