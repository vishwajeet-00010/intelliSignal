"""
IntelliSignal
agents/phase_dqn_agent.py

The Deep Q-Network that chooses GREEN DURATIONS.

    traffic state, from the served direction's view  (10 numbers)
            |
            v
      policy network                       agents/dqn_network.py
            |
            v
    Q(4 s), Q(5 s), Q(6 s) ... Q(20 s)      one per entry of the duration menu
            |
            v
    epsilon-greedy (training) / greedy (evaluation)
            |
            v
    rl/phase_action.py turns the choice into green + clearance --> SUMO

Three things are different from `agents/dqn_agent.py`, and each one is here
because the Stage 3 agent got it wrong on heavy traffic.

1. Double Q-learning
--------------------
Plain DQN takes `max_a Q_target(s', a)`, so any action whose value happens to
be over-estimated is both chosen and believed. With ten actions whose true
values differ by very little - a 6 s and a 7 s green are nearly the same
decision - that bias is large relative to the signal, and it is what a
degenerate "always pick the same action" policy looks like from the inside.

Double Q-learning splits the two roles: the POLICY network says which action
is best, the TARGET network says what it is worth.

    a*     = argmax_a Q_policy(s', a)
    target = r + gamma^duration * Q_target(s', a*)

2. Discounting by elapsed time, not by step count
-------------------------------------------------
A decision here occupies `green + yellow + all_red` seconds, and that is not
the same for every action: an 8 s green plus clearance is 12 s, a 20 s green
plus clearance is 24 s. Discounting both by one factor of gamma would tell
the agent that twenty-four seconds of consequence cost the same as twelve.

So gamma is a PER-SECOND discount and the target uses `gamma ** duration`.
At gamma = 0.98 a 12 s decision discounts the future by 0.78 and a 24 s
decision by 0.62, which is the honest ratio. This only works because
`rl/phase_environment.py` also integrates the reward over the decision's
seconds rather than averaging it; discounting by time while rewarding by
average would be worse than doing neither.

3. The action space is the control, not a request
-------------------------------------------------
Every action is legal in every state - the menu is bounded by `min_green` and
`max_green` by construction - so there is no action mask here. The Stage 3
agent needed one because SWITCH was illegal below the minimum green, and a
masked, nearly-tied two-action space is exactly the setting where the mask
and the over-estimation bias interact badly.

What this file does NOT contain: the traffic state, the action meaning, the
reward, or the signal timing. Those are `rl/phase_state.py`,
`rl/phase_action.py` and `rl/phase_environment.py`, and they are used as they
are.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.dqn_network import QNetwork                     # noqa: E402
from agents.replay_buffer import ReplayBuffer               # noqa: E402

CHECKPOINT_FORMAT = 2


# ---------------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    """Seed every generator that affects a training run."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


# ---------------------------------------------------------------------------
class EpsilonSchedule:
    """Linear decay of the exploration rate over a number of decisions."""

    def __init__(self, start: float = 1.0, end: float = 0.05,
                 decay_steps: int = 10000):
        if not 0.0 <= end <= start <= 1.0:
            raise ValueError("require 0 <= end <= start <= 1")
        if int(decay_steps) < 1:
            raise ValueError("decay_steps must be at least 1")
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = int(decay_steps)

    def value(self, step: int) -> float:
        fraction = min(max(float(step) / self.decay_steps, 0.0), 1.0)
        return self.start + fraction * (self.end - self.start)

    def describe(self) -> str:
        return "linear {:.2f} -> {:.2f} over {} decisions".format(
            self.start, self.end, self.decay_steps)


# ---------------------------------------------------------------------------
class PhaseDQNAgent:
    """A Double DQN over a menu of green durations, with time discounting."""

    def __init__(self, observation_size: int, num_actions: int,
                 hidden_layers: Sequence[int] = (128, 128),
                 learning_rate: float = 5e-4,
                 gamma_per_second: float = 0.98,
                 batch_size: int = 128,
                 replay_capacity: int = 100000,
                 learning_starts: int = 1000,
                 target_update_frequency: int = 400,
                 gradient_clip: float = 10.0,
                 double_dqn: bool = True,
                 loss: str = "huber",
                 seed: int = 42,
                 device: str = "cpu",
                 scenario: str = "",
                 durations: Sequence[float] = (),
                 observation_contract: Optional[Dict[str, Any]] = None):
        if not 0.0 < gamma_per_second <= 1.0:
            raise ValueError("gamma_per_second must be in (0, 1]")
        if loss not in ("huber", "mse"):
            raise ValueError("loss must be 'huber' or 'mse'")

        self.observation_size = int(observation_size)
        self.num_actions = int(num_actions)
        self.gamma_per_second = float(gamma_per_second)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.target_update_frequency = int(target_update_frequency)
        self.gradient_clip = float(gradient_clip)
        self.double_dqn = bool(double_dqn)
        self.loss_name = str(loss)
        self.seed = int(seed)
        self.scenario = str(scenario)
        self.durations = [float(d) for d in durations]
        self.observation_contract = dict(observation_contract or {})
        self.device = torch.device(device)

        set_seeds(seed)
        self.policy_net = QNetwork(self.observation_size, self.num_actions,
                                   hidden_layers).to(self.device)
        self.target_net = QNetwork(self.observation_size, self.num_actions,
                                   hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=float(learning_rate))
        self.memory = ReplayBuffer(capacity=replay_capacity, seed=seed,
                                   num_actions=self.num_actions)

        self.epsilon = 1.0
        self.learn_steps = 0
        self.target_updates = 0
        self.episode = 0
        self.model_path = None
        self._last_loss: Optional[float] = None
        self._random = random.Random(seed)

    # ==================================================================
    # Acting
    # ==================================================================
    def q_values(self, observation) -> np.ndarray:
        self.policy_net.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32),
                                     device=self.device)
            return self.policy_net(tensor).cpu().numpy()

    # ------------------------------------------------------------------
    def act(self, observation, training: bool = False) -> int:
        """Pick a duration index. Epsilon-greedy while training, greedy after."""
        if training and self._random.random() < self.epsilon:
            return self._random.randrange(self.num_actions)
        return int(np.argmax(self.q_values(observation)))

    # ------------------------------------------------------------------
    def set_epsilon(self, value: float) -> float:
        self.epsilon = float(min(max(value, 0.0), 1.0))
        return self.epsilon

    # ==================================================================
    # Learning
    # ==================================================================
    def remember(self, state, action: int, reward: float, next_state,
                 done: bool, duration: float) -> None:
        self.memory.push(state, action, reward, next_state, done,
                         duration=duration)

    # ------------------------------------------------------------------
    def learn(self) -> Optional[float]:
        """One gradient step. Returns the loss, or None when still filling up."""
        if len(self.memory) < max(self.learning_starts, self.batch_size):
            return None

        batch = self.memory.sample(self.batch_size)

        states = torch.from_numpy(batch.states).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        next_states = torch.from_numpy(batch.next_states).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)
        durations = torch.from_numpy(batch.durations).to(self.device)

        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # The policy network chooses, the target network scores.
                best = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, best).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values

            target_q = self.bootstrap_target(rewards, next_q, dones, durations)

        if self.loss_name == "mse":
            loss = F.mse_loss(current_q, target_q)
        else:
            loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                     self.gradient_clip)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_frequency == 0:
            self.update_target_network()

        self._last_loss = float(loss.item())
        return self._last_loss

    # ------------------------------------------------------------------
    def bootstrap_target(self, rewards, next_q, dones, durations):
        """The temporal-difference target, discounted by ELAPSED TIME.

            target = reward + gamma ** seconds * next_q      (continuing)
            target = reward                                  (terminal)

        `gamma` is a per-second rate, so a decision that occupied twenty-four
        seconds of simulation discounts its future more than one that
        occupied twelve. Kept separate from `learn` so the arithmetic that
        makes this a semi-Markov update can be tested on its own.
        """
        discount = torch.pow(
            torch.full_like(durations, self.gamma_per_second), durations)
        return torch.where(dones.bool(), rewards, rewards + discount * next_q)

    # ------------------------------------------------------------------
    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_updates += 1

    @property
    def last_loss(self) -> Optional[float]:
        return self._last_loss

    # ==================================================================
    # Explaining - the dashboard renders this next to the signal
    # ==================================================================
    def explain(self, observation) -> Dict[str, Any]:
        values = self.q_values(observation)
        action = int(np.argmax(values))
        green = self.durations[action] if self.durations else float(action)
        ranked = sorted(range(self.num_actions), key=lambda i: -values[i])
        runner_up = ranked[1] if len(ranked) > 1 else action
        return {
            "agent": "PhaseDQN",
            "action": action,
            "green_seconds": green,
            "is_learned": True,
            "q_values": {("{:g}s".format(self.durations[i]) if self.durations
                          else str(i)): float(values[i])
                         for i in range(self.num_actions)},
            "margin": float(values[action] - values[runner_up]),
            "epsilon": float(self.epsilon),
            "reason": (
                "Q is highest for a {:g} s green ({:+.3f}), ahead of {:g} s "
                "({:+.3f}), so the network expects the least congestion over "
                "the coming cycles if this green runs for {:g} s."
                .format(green, values[action],
                        self.durations[runner_up] if self.durations else runner_up,
                        values[runner_up], green)),
        }

    # ==================================================================
    # Checkpoints
    # ==================================================================
    def state_dict(self) -> Dict[str, Any]:
        return {
            "format_version": CHECKPOINT_FORMAT,
            "agent": "PhaseDQNAgent",
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": float(self.epsilon),
            "episode": int(self.episode),
            "learn_steps": int(self.learn_steps),
            "target_updates": int(self.target_updates),
            "observation_size": int(self.observation_size),
            "num_actions": int(self.num_actions),
            "hidden_layers": [int(h) for h in self.policy_net.hidden_layers],
            "architecture": self.policy_net.architecture(),
            "durations": list(self.durations),
            "gamma_per_second": float(self.gamma_per_second),
            "double_dqn": bool(self.double_dqn),
            "observation_contract": dict(self.observation_contract),
            "scenario": self.scenario,
            "seed": int(self.seed),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "torch_version": torch.__version__,
        }

    # ------------------------------------------------------------------
    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = self.state_dict()
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, path)
        self.model_path = path
        return path

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PhaseDQNAgent":
        """Rebuild an agent from a checkpoint, scales and menu included.

        The duration menu and the observation contract are read from the file
        rather than from a config, so a policy can never be evaluated against
        a different action space or different normalisation than the one it
        was trained with.
        """
        try:
            checkpoint = torch.load(path, map_location=device,
                                    weights_only=False)
        except TypeError:                      # pragma: no cover - old torch
            checkpoint = torch.load(path, map_location=device)

        agent = cls(
            observation_size=int(checkpoint["observation_size"]),
            num_actions=int(checkpoint["num_actions"]),
            hidden_layers=checkpoint.get("hidden_layers", (128, 128)),
            gamma_per_second=float(checkpoint.get("gamma_per_second", 0.98)),
            double_dqn=bool(checkpoint.get("double_dqn", True)),
            seed=int(checkpoint.get("seed", 42)),
            device=device,
            scenario=checkpoint.get("scenario", ""),
            durations=checkpoint.get("durations", ()),
            observation_contract=checkpoint.get("observation_contract"),
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(
            checkpoint.get("target_state_dict", checkpoint["policy_state_dict"]))
        agent.target_net.eval()
        agent.epsilon = float(checkpoint.get("epsilon", 0.0))
        agent.episode = int(checkpoint.get("episode", 0))
        agent.learn_steps = int(checkpoint.get("learn_steps", 0))
        agent.model_path = path
        return agent

    # ------------------------------------------------------------------
    def architecture(self) -> str:
        return self.policy_net.architecture()

    def __repr__(self) -> str:
        return ("<PhaseDQNAgent {} actions, {}, gamma={}/s, double={}>".format(
            self.num_actions, self.architecture(), self.gamma_per_second,
            self.double_dqn))


# ---------------------------------------------------------------------------
# The registry of promoted duration-choosing models
# ---------------------------------------------------------------------------
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "models", "active_models.json")

#: The section this family of models is registered under. It is deliberately
#: NOT `active_models`: that section is read by `agents/dqn_agent.py` and the
#: dashboard, both of which expect a KEEP/SWITCH checkpoint and an
#: observation contract with a `decision_interval`. Putting a duration-menu
#: model there would not be a promotion, it would be a crash.
REGISTRY_SECTION = "phase_duration_models"


class PhaseModelError(RuntimeError):
    """Raised when a registered duration-choosing model cannot be used."""


def _registry() -> Dict[str, Any]:
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    section = payload.get(REGISTRY_SECTION, {}) if isinstance(payload, dict) else {}
    return section if isinstance(section, dict) else {}


def registered_phase_model(scenario: str) -> Optional[Dict[str, Any]]:
    """The promoted duration-choosing model for a scenario, if there is one."""
    record = _registry().get(scenario)
    return dict(record) if isinstance(record, dict) else None


def load_registered_phase_model(scenario: str) -> "PhaseDQNAgent":
    """Load the promoted model, refusing a file that is not the one promoted.

    The hash is checked for the same reason the Stage 3 loader checks one: a
    checkpoint path is a mutable thing, and "the model we measured" and "the
    model in that directory" drift apart quietly.
    """
    record = registered_phase_model(scenario)
    if not record:
        raise PhaseModelError(
            "No duration-choosing model is registered for {!r} under "
            "'{}' in {}.".format(scenario, REGISTRY_SECTION, REGISTRY_PATH))

    path = str(record.get("path", ""))
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(path):
        raise PhaseModelError(
            "Registered model for {!r} is missing: {}".format(scenario, path))

    expected = str(record.get("sha256", "")).lower()
    if expected:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected:
            raise PhaseModelError(
                "Registered model for {!r} does not match its recorded "
                "sha256: {}".format(scenario, path))

    return PhaseDQNAgent.load(path)


# ---------------------------------------------------------------------------
def n_step_transitions(buffer: List[Dict[str, Any]], n: int,
                       gamma_per_second: float) -> Dict[str, Any]:
    """Collapse the first `n` steps of `buffer` into one transition.

    Multi-step returns matter here because one decision's effect shows up in
    the next direction's queue, not in its own. With a per-second discount the
    accumulated reward has to be discounted by the SECONDS already elapsed,
    not by the number of steps:

        R = r_0 + gamma^(d_0) * r_1 + gamma^(d_0 + d_1) * r_2 + ...

    and the collapsed transition's own duration is the sum of the durations
    it covers, so the bootstrap term is discounted correctly as well.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    window = buffer[:n]
    if not window:
        raise ValueError("nothing to collapse")

    total_reward = 0.0
    elapsed = 0.0
    for step in window:
        total_reward += (gamma_per_second ** elapsed) * float(step["reward"])
        elapsed += float(step["duration"])

    last = window[-1]
    return {
        "state": window[0]["state"],
        "action": window[0]["action"],
        "reward": total_reward,
        "next_state": last["next_state"],
        "done": bool(last["done"]),
        "duration": elapsed,
    }
