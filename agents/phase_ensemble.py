"""
IntelliSignal
agents/phase_ensemble.py

Several independently trained networks answering as one controller.

A single DQN run is one draw from a noisy process: the same recipe with a
different random seed lands on a slightly different policy, and which one
looks best on six validation workloads is partly luck. Averaging the
Q-values of several runs keeps what they agree on and cancels much of what
is peculiar to any one of them.

The ensemble is deliberately boring. Every member must be the same kind of
controller - same action space, same greens, same observation contract -
and the answer is the plain mean of their Q-values. It quacks like a
`PhaseDQNAgent` for everything that only needs to ACT (`q_values`, `act`,
`durations`, `observation_contract`, `action_space`, `extension`), so the
environments, the rollout and the harness run it unchanged. It cannot be
trained.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.phase_dqn_agent import PhaseDQNAgent


class EnsembleError(ValueError):
    """The members do not describe one and the same controller."""


class PhaseEnsemble:
    """The mean of several `PhaseDQNAgent` Q-functions."""

    is_ensemble = True

    def __init__(self, members: Sequence[PhaseDQNAgent]):
        members = list(members)
        if not members:
            raise EnsembleError("An ensemble needs at least one member.")
        first = members[0]
        for member in members[1:]:
            for name in ("num_actions", "observation_size", "action_space"):
                if getattr(member, name) != getattr(first, name):
                    raise EnsembleError(
                        "Members disagree on {}: {!r} against {!r}.".format(
                            name, getattr(member, name), getattr(first, name)))
            if [float(d) for d in member.durations] != \
                    [float(d) for d in first.durations]:
                raise EnsembleError("Members choose from different greens.")
            if abs(float(member.extension) - float(first.extension)) > 1e-9:
                raise EnsembleError("Members extend by different amounts.")
            if dict(member.observation_contract) != \
                    dict(first.observation_contract):
                raise EnsembleError(
                    "Members read the junction through different "
                    "observation contracts.")

        self.members: List[PhaseDQNAgent] = members
        self.num_actions = int(first.num_actions)
        self.observation_size = int(first.observation_size)
        self.durations = [float(d) for d in first.durations]
        self.observation_contract = dict(first.observation_contract)
        self.action_space = str(first.action_space)
        self.extension = float(first.extension)
        self.gamma_per_second = float(first.gamma_per_second)
        self.scenario = str(first.scenario)
        self.extra: Dict[str, Any] = dict(getattr(first, "extra", {}) or {})
        self.epsilon = 0.0
        self.model_path = ";".join(str(m.model_path or "") for m in members)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, paths: Sequence[str], device: str = "cpu") -> "PhaseEnsemble":
        return cls([PhaseDQNAgent.load(path, device=device) for path in paths])

    @property
    def member_paths(self) -> List[str]:
        return [str(member.model_path or "") for member in self.members]

    def __len__(self) -> int:
        return len(self.members)

    # ------------------------------------------------------------------
    def member_q_values(self, observation) -> np.ndarray:
        """One row of Q-values per member."""
        return np.stack([member.q_values(observation)
                         for member in self.members])

    def q_values(self, observation) -> np.ndarray:
        return self.member_q_values(observation).mean(axis=0)

    def act(self, observation, training: bool = False,
            valid_actions: Optional[Sequence[int]] = None) -> int:
        """The greedy action of the averaged Q-values. Never explores."""
        if training:
            raise EnsembleError("An ensemble is for acting, not training.")
        values = self.q_values(observation)
        if valid_actions is not None:
            allowed = [int(a) for a in valid_actions]
            if not allowed:
                raise EnsembleError("No valid action to choose from.")
            return int(max(allowed, key=lambda action: values[action]))
        return int(np.argmax(values))

    def agreement(self, observation,
                  valid_actions: Optional[Sequence[int]] = None) -> float:
        """Share of members whose own greedy action is the ensemble's."""
        chosen = self.act(observation, valid_actions=valid_actions)
        votes = [member.act(observation, training=False,
                            valid_actions=valid_actions)
                 for member in self.members]
        return sum(1 for vote in votes if int(vote) == chosen) / len(votes)

    # ------------------------------------------------------------------
    def architecture(self) -> str:
        return "{} x {}".format(len(self.members),
                                self.members[0].architecture())

    def __repr__(self) -> str:
        return "<PhaseEnsemble of {} | {} | {} actions>".format(
            len(self.members), self.action_space, self.num_actions)


def load_policy(paths) -> Any:
    """One path gives the network itself; several give their ensemble."""
    if isinstance(paths, (str, os.PathLike)):
        return PhaseDQNAgent.load(str(paths))
    paths = [str(path) for path in paths]
    if len(paths) == 1:
        return PhaseDQNAgent.load(paths[0])
    return PhaseEnsemble.load(paths)
