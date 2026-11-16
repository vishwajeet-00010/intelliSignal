"""
IntelliSignal - Stage 3
agents/dqn_network.py

THE NEURAL NETWORK. This is the "Deep" in Deep Q-Network.

What it computes
----------------
A Q-value for every action the agent can take, from one traffic observation:

    observation (10 numbers)                      Q-values (2 numbers)
    -------------------------                     -------------------
    north/east/south/west queue                   Q(keep)
    north/east/south/west waiting time   ----->    Q(switch)
    current phase
    how long the current green has run

    Linear(10, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 2)

A Q-value is the total discounted future reward the agent expects if it takes
that action now and then keeps following its own policy. Because the reward is
negative congestion, a Q-value closer to zero means "less waiting ahead".
The agent acts by taking the action with the larger Q-value.

Why the sizes are not hard-coded
--------------------------------
`input_size` comes from `env.observation_space.shape` and `output_size` from
`env.action_space.n`. If rl/state.py ever adds an eleventh observation, or
rl/action.py a third action, the network follows automatically and no number
in this file has to change. Only the hidden layers are configured, in
config/dqn/training.yaml.

Deliberately small
------------------
Two hidden layers of 64 units is roughly 5,000 parameters. The observation is
10 numbers and the decision is binary, so a larger network would only be
slower to train on a student laptop without learning anything more.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """A small fully connected network: observation in, one Q-value per action."""

    def __init__(self, input_size: int, output_size: int,
                 hidden_layers: Sequence[int] = (64, 64)):
        super().__init__()

        if int(input_size) <= 0:
            raise ValueError("input_size must be positive, got {}.".format(input_size))
        if int(output_size) <= 0:
            raise ValueError("output_size must be positive, got {}.".format(output_size))
        hidden = [int(h) for h in hidden_layers]
        if not hidden or any(h <= 0 for h in hidden):
            raise ValueError(
                "hidden_layers must be a non-empty sequence of positive "
                "integers, got {!r}.".format(hidden_layers))

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_layers = hidden

        layers = []
        previous = self.input_size
        for size in hidden:
            layers.append(nn.Linear(previous, size))
            layers.append(nn.ReLU())
            previous = size
        layers.append(nn.Linear(previous, self.output_size))
        self.model = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Q-values for a single observation or for a batch.

        Accepts shape (input_size,) or (batch, input_size) and returns
        (output_size,) or (batch, output_size) respectively.
        """
        if observation.dim() == 1:
            return self.model(observation.unsqueeze(0)).squeeze(0)
        return self.model(observation)

    # ------------------------------------------------------------------
    @classmethod
    def from_spaces(cls, observation_space, action_space,
                    hidden_layers: Sequence[int] = (64, 64)) -> "QNetwork":
        """Build the network from the environment's own spaces.

        This is how the training script creates it, so the architecture can
        never disagree with the environment.
        """
        shape = tuple(observation_space.shape)
        if len(shape) != 1:
            raise ValueError(
                "QNetwork expects a flat observation space, got shape {}."
                .format(shape))
        return cls(input_size=int(shape[0]),
                   output_size=int(action_space.n),
                   hidden_layers=hidden_layers)

    # ------------------------------------------------------------------
    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def architecture(self) -> str:
        """The architecture as the one-line string used in reports and metadata."""
        return " -> ".join(
            [str(self.input_size)] + [str(h) for h in self.hidden_layers]
            + [str(self.output_size)])

    def __repr__(self) -> str:
        return "<QNetwork {} ({} parameters)>".format(
            self.architecture(), self.parameter_count)
