"""Known-law planner: the reference for the normalised score S.

The planner knows the world law (q, noise, costs) and every body's execution
law, and sees exactly the learner's observations. It plans by exact dynamic
programming over a compact information state:

* anchor: the most recent valve known to have been the target (from a probe
  or a delivery), and k, the number of target transitions since then. Under
  the law, P(target = anchor) = 1/4 + 3/4 q^k.
* last: whether the last command is unset, equal to the anchor, or different.

Negative evidence (an executed actuation that delivered nothing) is dropped,
except that it clears the anchor when it contradicts it. For q in {0, 1} this
information state is exact on the planner's own path, so its DP value is the
exact expected return; for intermediate q it is a principled reference, not
an optimum, and its value must be estimated by Monte Carlo.

Burden is deterministic given the rest pattern, so welfare is handled by a
lazy rest schedule: rest only in cycles where acting would push burden over
the limit. Delaying rest is optimal because a later rest removes more burden
(f_rest(f_act(b)) < f_act(f_rest(b)) under 0.9 decay), and no rest is wasted
at zero burden.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from ..config import PROBE, REST, VALVES, Profile, WorldConfig
from ..env.runner import EpisodeContext
from ..env.water import Observation, Phase

UNSET, EQ, NEQ = 0, 1, 2
ANCHOR, LAST, FRESH = "anchor", "last", "fresh"
EPS = 1e-12


@dataclass(frozen=True)
class RestSchedule:
    forced: tuple[bool, ...]  # index t-1: must rest in cycle t
    burden: tuple[float, ...]  # burden after cycle t under the schedule
    welfare_unavoidable: bool

    @property
    def active_cycles(self) -> int:
        return sum(not f for f in self.forced)


def lazy_rest_schedule(world: WorldConfig, profile: Profile) -> RestSchedule:
    e = world.excess_noise(profile)
    b = 0.0
    forced, traj = [], []
    unavoidable = False
    for _ in range(world.horizon):
        act_b = world.burden_decay * b + e
        if act_b > world.burden_limit:
            forced.append(True)
            b = max(0.0, act_b - profile.recovery)
            if b > world.burden_limit:
                unavoidable = True
        else:
            forced.append(False)
            b = act_b
        traj.append(b)
    return RestSchedule(tuple(forced), tuple(traj), unavoidable)


def max_active_cycles(world: WorldConfig, profile: Profile) -> int:
    """Exact maximum number of non-rest cycles with burden kept within the limit.

    Pareto DP: for each count of active cycles, keep the minimum reachable
    burden (lower burden dominates since the dynamics are monotone). Used to
    verify that the lazy schedule is optimal.
    """
    e, d, r, lim = world.excess_noise(profile), world.burden_decay, profile.recovery, world.burden_limit
    front = {0: 0.0}
    for _ in range(world.horizon):
        nxt: dict[int, float] = {}
        for k, b in front.items():
            for active in (True, False):
                nb = max(0.0, d * b + e - (0.0 if active else r))
                if nb > lim + 1e-12:
                    continue
                kk = k + int(active)
                if kk not in nxt or nb < nxt[kk]:
                    nxt[kk] = nb
        front = nxt
    return max(front) if front else 0


class PlannerDP:
    """Backward induction over (cycle, anchor, k, last-relation)."""

    def __init__(self, world: WorldConfig, profile: Profile):
        if world.energy < (world.probe_cost + world.actuate_cost) * world.horizon:
            raise NotImplementedError("planner assumes energy never binds (true in Water-v1)")
        self.world = world
        self.profile = profile
        self.schedule = lazy_rest_schedule(world, profile)
        q = world.q
        self.kcap = 1 if q in (0.0, 1.0) else world.horizon + 1
        self.probe_allowed = not world.probe_shuffle
        self._value: dict[tuple, float] = {}
        self._probe: dict[tuple, bool] = {}
        self._commit: dict[tuple, str] = {}
        self._solve()

    # ----- belief helpers --------------------------------------------------------------------
    def p_anchor(self, k: int) -> float:
        n = self.world.n_valves
        return 1.0 / n + (1.0 - 1.0 / n) * self.world.q ** min(k, self.kcap)

    def p_hit(self, a: int, k: int, x: str) -> float:
        n = self.world.n_valves
        if not a:
            return 1.0 / n
        pa = self.p_anchor(k)
        return pa if x == ANCHOR else (1.0 - pa) / (n - 1)

    def p_last_is_target(self, a: int, k: int, last: int) -> float:
        if last == UNSET:
            return 0.0
        if not a:
            return 1.0 / self.world.n_valves
        return self.p_hit(a, k, ANCHOR if last == EQ else FRESH)

    def _age(self, a: int, k: int) -> int:
        return min(k + 1, self.kcap) if a else 0

    # ----- solve -----------------------------------------------------------------------------
    def V(self, t: int, a: int, k: int, last: int) -> float:
        if t > self.world.horizon:
            return 0.0
        return self._value[(t, a, k, last)]

    def commit_options(self, t: int, a: int, k: int, last: int, forced: bool) -> dict[str, float]:
        w, p = self.world, self.profile
        k1 = self._age(a, k)
        opts = {REST: self.V(t + 1, a, k1, last)}
        if forced:
            return opts
        cands = []
        if a:
            cands.append(ANCHOR)
            if last == NEQ:
                cands.append(LAST)
        elif last != UNSET:
            cands.append(LAST)
        cands.append(FRESH)
        for x in cands:
            is_last = x == LAST or (x == ANCHOR and last == EQ)
            pe = p.repeat_precision if is_last else p.switch_precision
            h = self.p_hit(a, k, ANCHOR if x == ANCHOR else FRESH)
            rel = EQ if x == ANCHOR else NEQ
            v_hit = self.V(t + 1, 1, min(1, self.kcap), EQ)
            if a and x == ANCHOR:
                v_miss = self.V(t + 1, 0, 0, NEQ)
            else:
                v_miss = self.V(t + 1, a, k1, rel)
            v_fail = self.V(t + 1, a, k1, rel)
            val = (pe * h * w.liters_per_success - w.energy_coef * w.actuate_cost
                   + pe * h * v_hit + pe * (1 - h) * v_miss + (1 - pe) * v_fail)
            opts[x] = val
        return opts

    def _best_commit(self, t, a, k, last, forced) -> tuple[str, float]:
        opts = self.commit_options(t, a, k, last, forced)
        order = [ANCHOR, LAST, FRESH, REST]  # tie-break preference
        best = max(order, key=lambda x: (opts.get(x, -1e18), -order.index(x)))
        return best, opts[best]

    def _solve(self) -> None:
        w = self.world
        ks = range(self.kcap + 1)
        for t in range(w.horizon, 0, -1):
            forced = self.schedule.forced[t - 1]
            # commit stage, for every state (including fresh post-probe states with k = 0)
            for a in (0, 1):
                for k in (ks if a else (0,)):
                    for last in (UNSET, EQ, NEQ):
                        if not a and last == EQ:
                            continue
                        x, v = self._best_commit(t, a, k, last, forced)
                        self._commit[(t, a, k, last)] = x
                        self._value[("c", t, a, k, last)] = v
            # probe stage
            for a in (0, 1):
                for k in (ks if a else (0,)):
                    for last in (UNSET, EQ, NEQ):
                        if not a and last == EQ:
                            continue
                        no_probe = self._value[("c", t, a, k, last)]
                        best, do_probe = no_probe, False
                        if self.probe_allowed:
                            pl = self.p_last_is_target(a, k, last)
                            if last == UNSET:
                                after = self._value[("c", t, 1, 0, UNSET)]
                            else:
                                after = (pl * self._value[("c", t, 1, 0, EQ)]
                                         + (1 - pl) * self._value[("c", t, 1, 0, NEQ)])
                            with_probe = -w.energy_coef * w.probe_cost + after
                            if with_probe > no_probe + EPS:
                                best, do_probe = with_probe, True
                        self._value[(t, a, k, last)] = best
                        self._probe[(t, a, k, last)] = do_probe

    # ----- queries ---------------------------------------------------------------------------
    @property
    def value(self) -> float:
        """Expected return from the start of an episode (exact for q in {0, 1})."""
        return self._value[(1, 0, 0, UNSET)]

    def should_probe(self, t: int, a: int, k: int, last: int) -> bool:
        return self._probe[(t, a, k, last)]

    def commit_choice(self, t: int, a: int, k: int, last: int) -> str:
        return self._commit[(t, a, k, last)]


@lru_cache(maxsize=64)
def solve(world: WorldConfig, profile: Profile) -> PlannerDP:
    return PlannerDP(world, profile)


def planner_value(world: WorldConfig, profile: Profile) -> float:
    return solve(world, profile).value


class KnownLawPlanner:
    """Acts in the environment from observations only, following the DP policy."""

    name = "known-law-planner"
    privileged = False

    def __init__(self, world: WorldConfig, profiles: dict[str, Profile]):
        self.world = world
        self.profiles = profiles
        self.remap = world.reward_remap.permutation if world.reward_remap else None

    def begin(self, ctx: EpisodeContext) -> None:
        self.dp = solve(self.world, self.profiles[ctx.body])
        self.anchor: Optional[int] = None
        self.k = 0
        self.last: Optional[int] = None
        self.prev_liters = 0
        self.pending: Optional[int] = None  # the commit we just sent

    def _rel(self) -> int:
        if self.last is None:
            return UNSET
        if self.anchor is not None and self.last == self.anchor:
            return EQ
        return NEQ

    def _absorb(self, obs: Observation) -> None:
        """Update the information state from the outcome of the previous commit."""
        if self.pending is None:
            return
        a = self.pending
        self.pending = None
        kcap = self.dp.kcap
        if a == REST:
            if self.anchor is not None:
                self.k = min(self.k + 1, kcap)
            return
        hit = obs.liters > self.prev_liters
        executed = bool(obs.last_success)
        self.last = a
        if executed and hit and self.remap is None:
            self.anchor, self.k = a, min(1, kcap)
        elif executed and not hit and self.remap is None and self.anchor == a:
            self.anchor, self.k = None, 0
        elif self.anchor is not None:
            self.k = min(self.k + 1, kcap)
        self.prev_liters = obs.liters

    def _valve(self, x: str) -> int:
        if x == ANCHOR:
            return self.anchor
        if x == LAST:
            return self.last
        taken = {self.anchor, self.last}
        return next(v for v in VALVES if v not in taken)

    def act(self, obs: Observation) -> int:
        dp, t = self.dp, obs.cycle
        if obs.phase is Phase.PROBE:
            self._absorb(obs)
            a = int(self.anchor is not None)
            if PROBE in obs.legal and dp.should_probe(t, a, self.k if a else 0, self._rel()):
                return PROBE
        else:  # post-probe: the probe result is a fresh anchor
            r = obs.probe_result
            self.anchor = self.remap[r - 1] if self.remap else r
            self.k = 0
        a = int(self.anchor is not None)
        x = dp.commit_choice(t, a, self.k if a else 0, self._rel())
        action = REST if x == REST else self._valve(x)
        if action not in obs.legal:
            action = REST
        self.pending = action
        self.prev_liters = obs.liters
        return action
