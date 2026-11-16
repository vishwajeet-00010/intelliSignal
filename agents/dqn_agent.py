"""
IntelliSignal - Stage 3
agents/dqn_agent.py

THE DEEP Q-NETWORK CONTROLLER. This is a real, working agent.

    traffic state (10 numbers)
            |
            v
      policy network            agents/dqn_network.py
            |
            v
    Q(keep), Q(switch)
            |
            v
    epsilon-greedy (training) / greedy (evaluation)
            |
            v
    keep / switch  ------->  rl/action.py safety controller  ---> SUMO

What this file contains and what it deliberately does not
---------------------------------------------------------
It contains the two networks, the optimiser, the exploration schedule, the
learning update and the checkpoint format. It does NOT contain the traffic
state, the action meaning, the reward, or the signal-timing safety rules:
those are Stage 2's and are used exactly as they are. The agent only ever
returns 0 or 1 and `rl/action.py` decides what a switch actually does to the
traffic light, which is why the agent cannot create an unsafe transition -
every switch still passes through the Stage 1 yellow clearance, still honours
min_green, and is still forced at max_green.

Two conventions, both supported
-------------------------------
The project has two action conventions and `agents/base_agent.py` documents
both:

  * the RL environment's KEEP / SWITCH  (what the network outputs)
  * the agent layer's PHASE SELECTION   (what `TrafficAgent.select_action`
    is defined to return)

`select_action()` accepts either input and answers in the matching convention:

    select_action(observation_array)  -> 0 = keep, 1 = switch      (env action)
    select_action(TrafficState)       -> 0 = NS green, 1 = EW green (phase)

The network itself is always asked for keep/switch; the phase answer is just
that decision translated through `phase_action_to_env_action`'s inverse. The
training loop and the evaluation runner use the observation form, the
dashboard uses the `TrafficState` form, and there is only one network behind
both.
"""

from __future__ import annotations

import glob
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agents.base_agent import (ACTION_EW_GREEN, ACTION_NAMES, ACTION_NS_GREEN,
                               ENV_ACTION_KEEP, ENV_ACTION_SWITCH, PHASE_EW,
                               PHASE_NS, TrafficAgent, TrafficState)
from agents.dqn_config import DQNConfig, load_dqn_config
from agents.replay_buffer import ReplayBuffer
from utils.config_loader import PROJECT_ROOT

# ---------------------------------------------------------------------------
# PyTorch is a Stage 3 dependency. The project must still import without it,
# so that Stage 1 and Stage 2 keep working on a machine where torch has not
# been installed - the dashboard then reports the DQN mode as unavailable
# instead of crashing.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from agents.dqn_network import QNetwork

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as _exc:                       # pragma: no cover - env specific
    torch = None
    nn = None
    F = None
    QNetwork = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(_exc)

#: Stage 3 implemented the DQN. This flag stays as the one place that says so.
DQN_IMPLEMENTED = True

#: Where trained models live: models/dqn_<scenario>/{best,final}.pt
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

#: Preference order when the caller does not name a checkpoint.
PREFERRED_CHECKPOINTS = ("best.pt", "final.pt")

TORCH_MISSING_MESSAGE = (
    "PyTorch is not installed, so the DQN controller cannot run.\n\n"
    "Install it with:  pip install -r requirements.txt\n"
    "Original import error: {}".format(TORCH_IMPORT_ERROR))

NO_MODEL_MESSAGE = (
    "No trained DQN model was found{where}.\n\n"
    "The DQN code exists, but a network with random weights is not a "
    "controller - it would produce meaningless signal decisions. Train one "
    "first:\n\n"
    "    python training/train_dqn.py --scenario {scenario} --episodes 20 --seed 42\n\n"
    "The model is then saved to models/dqn_{scenario}/ and this mode becomes "
    "available.")


class DQNModelNotFoundError(FileNotFoundError):
    """Raised when the DQN is asked to act but no trained weights exist.

    This is deliberately NOT a quiet fall back to the fixed-time controller.
    A missing model is reported; it is never hidden behind fabricated metrics.
    """


class DQNUnavailableError(RuntimeError):
    """Raised when PyTorch itself is missing."""


# ===========================================================================
# Model discovery - used by the dashboard to decide whether DQN is offered
# ===========================================================================
def model_directory(scenario: str) -> str:
    """models/dqn_<scenario>"""
    return os.path.join(MODELS_DIR, "dqn_{}".format(scenario))


def find_models(scenario: str) -> List[Dict[str, Any]]:
    """Every checkpoint saved for a scenario, best and final first.

    Each entry is {"name", "path", "metadata"}. `metadata` is whatever the
    sidecar metadata JSON holds, or {} when there is none - it is read from
    the file, never guessed.
    """
    directory = model_directory(scenario)
    if not os.path.isdir(directory):
        return []

    paths = sorted(glob.glob(os.path.join(directory, "*.pt")))
    metadata = read_metadata(directory)

    def rank(path: str) -> tuple:
        name = os.path.basename(path)
        if name in PREFERRED_CHECKPOINTS:
            return (PREFERRED_CHECKPOINTS.index(name), name)
        return (len(PREFERRED_CHECKPOINTS), name)

    return [{"name": os.path.basename(p), "path": p, "metadata": metadata}
            for p in sorted(paths, key=rank)]


def read_metadata(directory: str) -> Dict[str, Any]:
    """The metadata JSON written next to the checkpoints, or {}."""
    path = os.path.join(directory, "metadata.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def default_model_path(scenario: str) -> Optional[str]:
    """The checkpoint a caller gets when it does not name one, or None."""
    models = find_models(scenario)
    return models[0]["path"] if models else None


def dqn_model_available(scenario: str = None) -> bool:
    """True only when PyTorch is installed AND a trained model exists.

    This is what the dashboard asks before offering DQN control. With no
    scenario it answers for any scenario that has a model.
    """
    if not TORCH_AVAILABLE:
        return False
    if scenario is not None:
        return default_model_path(scenario) is not None
    if not os.path.isdir(MODELS_DIR):
        return False
    return bool(glob.glob(os.path.join(MODELS_DIR, "dqn_*", "*.pt")))


# ===========================================================================
# Exploration schedule
# ===========================================================================
class EpsilonSchedule:
    """How much the agent explores, as a function of the training step.

    Linear (the default):

        epsilon_start ------\\
                             \\______ epsilon_end
        0            decay_steps          ...

    Exponential multiplies epsilon by `rate` every step and is clamped at
    epsilon_end. Both are configured in config/dqn/training.yaml; the linear
    schedule's length is derived from the number of episodes requested, so a
    short demo run still explores for half of its steps instead of becoming
    greedy after two episodes.
    """

    def __init__(self, start: float = 1.0, end: float = 0.05,
                 decay_steps: int = 9000, kind: str = "linear",
                 rate: float = 0.9995):
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = max(int(decay_steps), 1)
        self.kind = str(kind)
        self.rate = float(rate)

    def value(self, step: int) -> float:
        step = max(int(step), 0)
        if self.kind == "exponential":
            return max(self.end, self.start * (self.rate ** step))
        fraction = min(step / float(self.decay_steps), 1.0)
        return self.start + fraction * (self.end - self.start)

    def describe(self) -> str:
        if self.kind == "exponential":
            return "exponential, x{} per step, floor {}".format(self.rate, self.end)
        return "linear {} -> {} over {} steps".format(
            self.start, self.end, self.decay_steps)


# ===========================================================================
# The agent
# ===========================================================================
class DQNAgent(TrafficAgent):
    """Chooses keep/switch from Q-values learned by a neural network."""

    name = "DQN Adaptive Controller"
    description = ("Chooses the signal phase from Q-values produced by a "
                   "trained neural network, using live queue lengths and "
                   "waiting times.")
    #: Whether the implementation can run at all here (PyTorch present).
    #: A usable *controller* also needs trained weights - see is_available().
    available = TORCH_AVAILABLE
    is_learned = True

    # ------------------------------------------------------------------
    def __init__(self, model_path: Optional[str] = None,
                 scenario: Optional[str] = None,
                 observation_size: Optional[int] = None,
                 num_actions: int = 2,
                 config: Optional[DQNConfig] = None,
                 trainable: bool = False,
                 device: Optional[str] = None,
                 seed: Optional[int] = None):
        """Build the agent.

        Three ways to create one, and nothing in between:

          DQNAgent(model_path=...)                 load a trained checkpoint
          DQNAgent(scenario="moderate")            load that scenario's best
                                                   checkpoint, or refuse
          DQNAgent(observation_size=10,
                   trainable=True, config=cfg)     a fresh, untrained network
                                                   for training/tests only

        Asking for a controller when no model exists raises
        `DQNModelNotFoundError`. It never returns an untrained network
        pretending to be a trained one.
        """
        if not TORCH_AVAILABLE:
            raise DQNUnavailableError(TORCH_MISSING_MESSAGE)

        self.config = config or load_dqn_config()
        self.scenario = scenario
        self.device = torch.device(device or "cpu")
        self.trainable = bool(trainable)
        self.model_path = ""
        self.metadata: Dict[str, Any] = {}

        # --- find the weights, unless a fresh network was asked for ----
        if model_path is None and not trainable and observation_size is None:
            if scenario is None:
                raise DQNModelNotFoundError(NO_MODEL_MESSAGE.format(
                    where=" (no scenario given)", scenario="moderate"))
            model_path = default_model_path(scenario)
            if model_path is None:
                raise DQNModelNotFoundError(NO_MODEL_MESSAGE.format(
                    where=" in {}".format(
                        os.path.relpath(model_directory(scenario), PROJECT_ROOT)),
                    scenario=scenario))

        if model_path is not None and not os.path.isfile(model_path):
            raise DQNModelNotFoundError(
                "DQN checkpoint not found: {}".format(model_path))

        # --- sizes -----------------------------------------------------
        checkpoint = None
        if model_path is not None:
            checkpoint = self._read_checkpoint(model_path)
            observation_size = int(checkpoint["observation_size"])
            num_actions = int(checkpoint["num_actions"])
            hidden = [int(h) for h in checkpoint.get(
                "hidden_layers", self.config.hidden_layers)]
        else:
            if observation_size is None:
                raise ValueError(
                    "observation_size is required to build an untrained "
                    "network. Pass the environment's "
                    "observation_space.shape[0].")
            hidden = [int(h) for h in self.config.hidden_layers]

        self.observation_size = int(observation_size)
        self.num_actions = int(num_actions)

        if seed is not None:
            set_seeds(int(seed))

        # --- the two networks ------------------------------------------
        self.policy_net = QNetwork(self.observation_size, self.num_actions,
                                   hidden).to(self.device)
        self.target_net = QNetwork(self.observation_size, self.num_actions,
                                   hidden).to(self.device)
        # The target network starts as an exact copy of the policy network.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # --- training machinery ----------------------------------------
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=self.config.learning_rate)
        self.memory = ReplayBuffer(self.config.replay_capacity, seed=seed)
        self.epsilon = float(self.config.epsilon_start)
        self.learn_steps = 0          # gradient steps taken
        self.target_updates = 0       # times the target network was refreshed
        self.episode = 0
        self._random = random.Random(seed)
        self._last_loss: Optional[float] = None

        if checkpoint is not None:
            self._restore(checkpoint, model_path)
            self.policy_net.eval()    # loaded models default to evaluation

    # ==================================================================
    # Acting
    # ==================================================================
    def q_values(self, observation) -> np.ndarray:
        """The raw Q-values for one observation: [Q(keep), Q(switch)]."""
        array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if array.size != self.observation_size:
            raise ValueError(
                "Observation has {} values but the network expects {}."
                .format(array.size, self.observation_size))
        with torch.no_grad():
            tensor = torch.from_numpy(array).to(self.device)
            values = self.policy_net(tensor)
        return values.cpu().numpy()

    # ------------------------------------------------------------------
    def act(self, observation, training: bool = False) -> int:
        """The environment action for an observation: 0 = keep, 1 = switch.

        training=True  ->  epsilon-greedy: a random action with probability
                           `self.epsilon`, otherwise the best known action.
        training=False ->  greedy, no exploration at all. This is what
                           evaluation and the dashboard use.
        """
        if training and self._random.random() < self.epsilon:
            return int(self._random.randrange(self.num_actions))
        return int(np.argmax(self.q_values(observation)))

    # ------------------------------------------------------------------
    def select_action(self, state, training: bool = False) -> int:
        """The agent's decision.

        `state` may be either form, and the answer matches the question:

          * a 10-number observation (NumPy array, list, tensor)
                -> the ENVIRONMENT action: 0 keep, 1 switch
          * a `TrafficState` (the agent-layer object the dashboard uses)
                -> the PHASE to make green: 0 NS green, 1 EW green

        The network is asked for keep/switch either way; the second form just
        translates that into the phase that keep/switch would produce.
        """
        if isinstance(state, TrafficState):
            observation = state.observation()
            if observation is None:
                raise ValueError(
                    "This TrafficState carries no 10-number observation, so "
                    "the Q-network cannot be evaluated. Build it with "
                    "TrafficState.from_env_info(info, observation=obs) - the "
                    "DQN reads the per-approach observation, not the NS/EW "
                    "summary.")
            env_action = self.act(observation, training=training)
            return self.phase_for(env_action, state.current_phase)
        return self.act(state, training=training)

    # ------------------------------------------------------------------
    @staticmethod
    def phase_for(env_action: int, current_phase: int) -> int:
        """keep/switch + the current phase -> the phase that will be green."""
        if int(env_action) == ENV_ACTION_KEEP:
            return int(current_phase)
        return PHASE_EW if int(current_phase) == PHASE_NS else PHASE_NS

    # ------------------------------------------------------------------
    def env_action(self, state: TrafficState) -> int:
        """Override: ask the network directly, with no phase round trip."""
        observation = state.observation()
        if observation is None:
            return super().env_action(state)
        return self.act(observation, training=False)

    # ------------------------------------------------------------------
    def explain(self, state) -> Dict[str, Any]:
        """Why the agent chose what it chose, including the Q-values.

        The dashboard renders this next to the signal, so a panel member can
        see the two numbers the decision was actually made from.
        """
        if isinstance(state, TrafficState):
            observation = state.observation()
            current_phase = int(state.current_phase)
            state_rows = {
                "NS queue": state.ns_queue,
                "EW queue": state.ew_queue,
                "NS waiting": state.ns_waiting,
                "EW waiting": state.ew_waiting,
                "current phase": ACTION_NAMES[current_phase],
                "time in phase": state.time_since_phase_change,
            }
        else:
            observation = np.asarray(state, dtype=np.float32)
            current_phase = PHASE_EW if float(observation[8]) >= 0.5 else PHASE_NS
            state_rows = {"observation": np.round(observation, 3).tolist()}

        if observation is None:
            raise ValueError(
                "explain() needs a TrafficState that carries its observation.")

        values = self.q_values(observation)
        env_action = int(np.argmax(values))
        phase = self.phase_for(env_action, current_phase)
        return {
            "agent": self.name,
            "action": phase,
            "action_name": ACTION_NAMES[phase],
            "is_learned": True,
            "uses_traffic_state": True,
            "env_action": env_action,
            "env_action_name": "KEEP" if env_action == ENV_ACTION_KEEP else "SWITCH",
            "q_values": {"keep": float(values[ENV_ACTION_KEEP]),
                         "switch": float(values[ENV_ACTION_SWITCH])},
            "q_gap": float(abs(values[0] - values[1])),
            "epsilon": float(self.epsilon),
            "model_path": self.model_path,
            "reason": ("Q(keep) = {:+.3f} and Q(switch) = {:+.3f}, so the "
                       "network expects less future congestion if it {}."
                       .format(values[ENV_ACTION_KEEP], values[ENV_ACTION_SWITCH],
                               "holds this green" if env_action == ENV_ACTION_KEEP
                               else "changes the signal")),
            "state": state_rows,
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Nothing to clear: the agent is memoryless between decisions.

        All of the history the DQN needs is in the observation itself (the
        current phase and how long it has been green), so there is no hidden
        per-episode state. The replay buffer is deliberately NOT cleared - it
        is the agent's long-term memory across episodes.
        """

    # ==================================================================
    # Learning
    # ==================================================================
    def remember(self, state, action: int, reward: float, next_state,
                 done: bool) -> None:
        """Store one transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    def learn(self) -> Optional[float]:
        """One gradient step of Q-learning. Returns the loss, or None.

        None means "not yet": the buffer still holds fewer transitions than
        `learning_starts`, so there is nothing worth learning from.

        The update is the standard DQN temporal-difference rule:

            current = Q_policy(state, action_taken)
            target  = reward + gamma * max_a' Q_target(next_state, a')
            target  = reward                        if the episode ended

        The maximum is taken with the TARGET network, not the policy network.
        That is the whole point of having two: the policy network is being
        changed by this very gradient step, and using it on both sides of the
        comparison makes the target move as fast as the estimate, which is
        what destabilises plain Q-learning with a function approximator.
        """
        if len(self.memory) < max(self.config.learning_starts,
                                  self.config.batch_size):
            return None

        batch = self.memory.sample(self.config.batch_size)

        states = torch.from_numpy(batch.states).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        next_states = torch.from_numpy(batch.next_states).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)

        self.policy_net.train()

        # Q-value the policy network currently gives the action that was taken
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        # Best Q-value the TARGET network sees in the next state
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            # A terminal transition has no future, so the future term is zeroed.
            target_q = rewards + self.config.gamma * next_q * (1.0 - dones)

        if self.config.loss == "mse":
            loss = F.mse_loss(current_q, target_q)
        else:
            loss = F.smooth_l1_loss(current_q, target_q)   # Huber

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.gradient_clip:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                     float(self.config.gradient_clip))
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % int(self.config.target_update_frequency) == 0:
            self.update_target_network()

        self._last_loss = float(loss.item())
        return self._last_loss

    # ------------------------------------------------------------------
    def update_target_network(self) -> None:
        """target_net <- policy_net (a full copy of the weights)."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_updates += 1

    # ------------------------------------------------------------------
    def set_epsilon(self, value: float) -> float:
        self.epsilon = float(min(max(value, 0.0), 1.0))
        return self.epsilon

    @property
    def last_loss(self) -> Optional[float]:
        return self._last_loss

    # ==================================================================
    # Checkpoints
    # ==================================================================
    def state_dict(self) -> Dict[str, Any]:
        """Everything needed to continue training or to reproduce a decision."""
        return {
            "format_version": 1,
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
            "config": self.config.as_dict(),
            "scenario": self.scenario or "",
            "seed": int(self.config.seed),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "torch_version": torch.__version__,
        }

    # ------------------------------------------------------------------
    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Write a checkpoint, creating the directory if needed.

        Both networks, the optimiser state, epsilon, the episode counter, the
        full configuration, the architecture and the seed are stored - never
        an anonymous bag of weights.
        """
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        payload = self.state_dict()
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, path)
        self.model_path = path
        return path

    # ------------------------------------------------------------------
    @staticmethod
    def _read_checkpoint(path: str) -> Dict[str, Any]:
        """Load a checkpoint file across PyTorch versions."""
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:                       # torch < 1.13 has no weights_only
            return torch.load(path, map_location="cpu")

    # ------------------------------------------------------------------
    def _restore(self, checkpoint: Dict[str, Any], path: str) -> None:
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        if "target_state_dict" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
        else:                                    # pragma: no cover - old files
            self.target_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.eval()

        if self.trainable and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:                    # pragma: no cover
                pass

        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_end))
        self.episode = int(checkpoint.get("episode", 0))
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
        self.target_updates = int(checkpoint.get("target_updates", 0))
        self.scenario = self.scenario or (checkpoint.get("scenario") or None)
        self.model_path = path
        self.metadata = {
            "architecture": checkpoint.get("architecture", ""),
            "episode": self.episode,
            "epsilon": self.epsilon,
            "scenario": checkpoint.get("scenario", ""),
            "seed": checkpoint.get("seed", ""),
            "saved_at": checkpoint.get("saved_at", ""),
            "torch_version": checkpoint.get("torch_version", ""),
            "config": checkpoint.get("config", {}),
        }
        self.metadata.update(read_metadata(os.path.dirname(os.path.abspath(path))))

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, trainable: bool = False,
             device: Optional[str] = None, **kwargs) -> "DQNAgent":
        """Build an agent from a saved checkpoint."""
        return cls(model_path=path, trainable=trainable, device=device, **kwargs)

    @classmethod
    def for_scenario(cls, scenario: str, model_path: Optional[str] = None,
                     **kwargs) -> "DQNAgent":
        """The agent the dashboard asks for: that scenario's trained model."""
        return cls(model_path=model_path, scenario=scenario, **kwargs)

    # ------------------------------------------------------------------
    @classmethod
    def is_available(cls, scenario: str = None) -> bool:
        """True only when this agent can really control a signal here."""
        return dqn_model_available(scenario)

    # ------------------------------------------------------------------
    def architecture(self) -> str:
        return self.policy_net.architecture()

    def summary(self) -> Dict[str, Any]:
        """What the dashboard's model page shows. Every value is real."""
        return {
            "name": self.name,
            "model_path": self.model_path,
            "scenario": self.scenario or "",
            "architecture": self.architecture(),
            "parameters": self.policy_net.parameter_count,
            "observation_size": self.observation_size,
            "num_actions": self.num_actions,
            "episodes_trained": self.episode,
            "epsilon": self.epsilon,
            "gamma": self.config.gamma,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "replay_capacity": self.config.replay_capacity,
            "target_update_frequency": self.config.target_update_frequency,
            "seed": self.config.seed,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return "<DQNAgent {} model={!r} episodes={}>".format(
            self.architecture(), os.path.basename(self.model_path or ""),
            self.episode)


# ===========================================================================
# Reproducibility
# ===========================================================================
def set_seeds(seed: int) -> int:
    """Seed Python, NumPy and PyTorch.

    SUMO is seeded separately, through the environment (`env.reset(seed=...)`),
    because its randomness lives in the simulator process. Even with every
    seed fixed, results can still differ between SUMO versions and between
    machines - that is documented rather than hidden.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():            # pragma: no cover - CPU in CI
            torch.cuda.manual_seed_all(seed)
    return seed


def write_metadata(directory: str, metadata: Dict[str, Any]) -> str:
    """Write models/dqn_<scenario>/metadata.json next to the checkpoints."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "metadata.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
    return path


def build_agent_for_env(env, config: DQNConfig = None, scenario: str = None,
                        seed: int = None, device: str = None) -> DQNAgent:
    """A fresh, untrained agent sized from the environment's own spaces.

    This is how training/train_dqn.py creates the agent: the input size comes
    from `env.observation_space.shape` and the output size from
    `env.action_space.n`, so the network can never disagree with Stage 2.
    """
    shape = tuple(env.observation_space.shape)
    if len(shape) != 1:
        raise ValueError(
            "Expected a flat observation space, got shape {}.".format(shape))
    return DQNAgent(
        observation_size=int(shape[0]),
        num_actions=int(env.action_space.n),
        config=config,
        scenario=scenario,
        trainable=True,
        seed=seed,
        device=device,
    )
