"""
IntelliSignal
rl/timing_profile.py

The operational timing rules a controller must obey.

A profile is a named set of limits that every controller - fixed-time,
actuated, pressure-based or learned - is held to. It exists so that a
comparison is fair by construction: you cannot accidentally measure a
learned controller that is allowed 4 s greens against a fixed plan that is
not.

    min_green       the shortest green the signal may show
    max_green       the longest
    durations       the menu a duration-choosing controller picks from
    max_red_factor  a group's continuous red may not exceed this many times
                    its own preceding completed green

Two profiles ship with the project.

RESPONSIVE (4-20 s)
    What every model up to 23 Nov 2026 was trained with. On this junction
    it is genuinely the better-performing regime: clearance is only 4 s and
    demand is modest, so short cycles win. The measured optima are
    north-south 7 s / east-west 6 s on heavy (5.754 s mean waiting) and
    6 s / 4 s on moderate (2.761 s).

STANDARD (15-30 s)
    Conventional signal practice - long enough for a pedestrian phase and
    for drivers to expect it. It is the more realistic profile and it is
    markedly more expensive on this junction. Measured over a full sweep of
    every plan with both greens in 15-30 s, seeds 6101-6102:

        heavy     best is ns 18 / ew 15 -> 42.50 s   (against 5.754 s)
        moderate  best is ns 15 / ew 15 ->  5.57 s   (against 2.761 s)

    That 7.4x is the price of the profile, not a defect in any controller.
    It is reported rather than hidden, and the two profiles are never
    compared with each other.

The red limit
-------------
`max_red_factor` is an upper bound, never a required wait. On a two-group
junction it is also provably slack: a group's red runs from the end of its
own yellow to the start of its next green, which is exactly

    other group's green + other group's yellow

so the worst case is `max_green + yellow`. Under STANDARD that is
30 + 4 = 34 s against a tightest limit of 4 x 15 = 60 s - it can never
bind. `red_limit_analysis()` returns that arithmetic so a report can state
it rather than imply the rule is doing work.

Under RESPONSIVE the same rule *can* bind (4 x 4 = 16 s against a worst
case of 20 + 4 = 24 s), which is why it is enforced rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

RESPONSIVE = "responsive"
STANDARD = "standard"


@dataclass(frozen=True)
class TimingProfile:
    """One named set of operational timing limits."""

    name: str
    min_green: float
    max_green: float
    durations: Tuple[float, ...]
    max_red_factor: float = 4.0
    description: str = ""

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.min_green <= 0:
            raise ValueError("min_green must be positive.")
        if self.max_green < self.min_green:
            raise ValueError("max_green must be at least min_green.")
        if not self.durations:
            raise ValueError("durations must not be empty.")
        if list(self.durations) != sorted(self.durations):
            raise ValueError("durations must be in increasing order.")
        if len(set(self.durations)) != len(self.durations):
            raise ValueError("durations must not repeat a value.")
        if self.durations[0] < self.min_green:
            raise ValueError(
                "the shortest duration ({:g} s) is below min_green ({:g} s)."
                .format(self.durations[0], self.min_green))
        if self.durations[-1] > self.max_green:
            raise ValueError(
                "the longest duration ({:g} s) is above max_green ({:g} s)."
                .format(self.durations[-1], self.max_green))
        if self.max_red_factor <= 0:
            raise ValueError("max_red_factor must be positive.")

    # ------------------------------------------------------------------
    def max_red_seconds(self, preceding_green: float) -> float:
        """The longest continuous red allowed after a green of that length.

        An upper limit. Nothing is required to wait this long, and the
        first red of an episode has no preceding green - see
        `initial_max_red_seconds`.
        """
        return self.max_red_factor * float(preceding_green)

    def initial_max_red_seconds(self) -> float:
        """The limit for a group that has not been green yet.

        There is no preceding green to multiply, so the minimum green is
        used. That is the strictest reading of the rule and it cannot be
        gamed by starting the episode on a very short green.
        """
        return self.max_red_factor * self.min_green

    # ------------------------------------------------------------------
    def red_limit_analysis(self, yellow: float,
                           all_red: float = 0.0) -> Dict[str, Any]:
        """Whether the red limit can ever bind on a two-group junction."""
        clearance = float(yellow) + float(all_red)
        worst_red = self.max_green + clearance
        tightest_limit = self.max_red_factor * self.min_green
        binding = worst_red >= tightest_limit
        return {
            "profile": self.name,
            "clearance_seconds": clearance,
            "worst_case_red_seconds": worst_red,
            "worst_case_formula": "max_green + yellow + all_red",
            "tightest_limit_seconds": tightest_limit,
            "tightest_limit_formula": "max_red_factor x min_green",
            "can_bind": binding,
            "explanation": (
                "On this two-group junction a group's red is exactly the "
                "other group's green plus the clearance, so the worst case "
                "is {:.0f} s. The tightest the limit can ever be is "
                "{:.0f} s, so the rule {}."
                .format(worst_red, tightest_limit,
                        "can bind and is enforced" if binding
                        else "can never bind on this junction")),
        }

    # ------------------------------------------------------------------
    @property
    def cycle_range(self) -> Tuple[float, float]:
        return (2 * self.min_green, 2 * self.max_green)

    def contains_green(self, seconds: float) -> bool:
        return self.min_green - 1e-9 <= seconds <= self.max_green + 1e-9

    def nearest_duration(self, seconds: float) -> float:
        """The menu entry closest to `seconds`. Used to port a fixed plan."""
        return min(self.durations, key=lambda d: abs(d - float(seconds)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_green": self.min_green,
            "max_green": self.max_green,
            "durations": list(self.durations),
            "max_red_factor": self.max_red_factor,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimingProfile":
        return cls(name=data["name"], min_green=float(data["min_green"]),
                   max_green=float(data["max_green"]),
                   durations=tuple(float(d) for d in data["durations"]),
                   max_red_factor=float(data.get("max_red_factor", 4.0)),
                   description=data.get("description", ""))

    def describe(self) -> str:
        return ("{}: green {:g}-{:g} s, menu {}, red at most {:g}x the "
                "preceding green".format(
                    self.name, self.min_green, self.max_green,
                    "/".join("{:g}".format(d) for d in self.durations),
                    self.max_red_factor))


# ---------------------------------------------------------------------------
PROFILES: Dict[str, TimingProfile] = {
    RESPONSIVE: TimingProfile(
        name=RESPONSIVE,
        min_green=4.0, max_green=20.0,
        durations=(4, 5, 6, 7, 8, 9, 10, 12, 15, 20),
        max_red_factor=4.0,
        description=(
            "Short greens, as every model up to 23 Nov 2026 was trained. "
            "The better-performing regime on this junction, because the "
            "clearance is only 4 s and demand is modest."),
    ),
    STANDARD: TimingProfile(
        name=STANDARD,
        min_green=15.0, max_green=30.0,
        # Dense where the measured optimum sits (15-22 s) and sparse above
        # it: a sweep of every plan in 15-30 s put the best at ns 18 / ew 15
        # on heavy and 15 / 15 on moderate, so resolution matters at the
        # bottom of the range and not at the top.
        durations=(15, 16, 17, 18, 20, 22, 25, 30),
        max_red_factor=4.0,
        description=(
            "Conventional signal practice: a 15 s minimum green, long "
            "enough for a pedestrian phase. More realistic and markedly "
            "more expensive on this junction - the best plan under it is "
            "42.50 s on heavy against 5.754 s unconstrained."),
    ),
}

DEFAULT_PROFILE = RESPONSIVE


def get_profile(name: Optional[str]) -> TimingProfile:
    """Look a profile up by name, defaulting to the historical one."""
    if name is None:
        return PROFILES[DEFAULT_PROFILE]
    if isinstance(name, TimingProfile):
        return name
    try:
        return PROFILES[str(name)]
    except KeyError:
        raise ValueError(
            "Unknown timing profile {!r}. Available: {}.".format(
                name, ", ".join(sorted(PROFILES))))


def profile_for_durations(durations: Sequence[float]) -> Optional[str]:
    """Which profile a checkpoint's menu belongs to, if any.

    Used to refuse to run a model under a profile it was not trained for -
    a 4 s-green policy loaded into the standard profile is not a
    conservative choice, it is an invalid one.
    """
    wanted = tuple(float(d) for d in durations)
    for name, profile in PROFILES.items():
        if tuple(float(d) for d in profile.durations) == wanted:
            return name
    return None


def compatible(profile: TimingProfile, durations: Sequence[float]) -> bool:
    """Can a checkpoint with this menu be run under this profile?"""
    return all(profile.contains_green(float(d)) for d in durations)
