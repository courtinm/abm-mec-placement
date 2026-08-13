"""
Baseline CR placement strategies.

Each class exposes the same interface as CRPlacementAgent so simulator.py
needs zero modifications:
    get_state() -> tuple
    select_action(state) -> int
    apply_action(action_idx) -> None
    update(*args) -> None   (no-op — strategies do not learn)
    prev_state, prev_action, last_reward  (set by simulator, harmlessly ignored)
"""

import math
import random
from itertools import combinations

from agents.base_station import BaseStation, ComputeRessources
from agents.relay_node import RelayNode
from simulation.simulator import (
    get_path_to_bs,
    _BS_THROUGHPUT_MBPS,
    _RADIO_LATENCY_SCALE,
    _CORE_LATENCY_MS,
    _HOP_LATENCY_MS,
    _admission_order,
    _proportional_rate,
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _apply(candidate_bs, chosen_set, cr_pool):
    """Activate CRs on the BS indices in *chosen_set*; deactivate all others."""
    pool_iter = iter(cr_pool)
    for i, bs in enumerate(candidate_bs):
        if i in chosen_set:
            cr = next(pool_iter)
            cr.current_load_mbps = 0.0
            cr.demanded_load_mbps = 0.0
            bs.has_compute_resource = True
            bs.compute_resource = cr
        else:
            bs.has_compute_resource = False
            bs.compute_resource = None


def _score(users, candidate_bs, chosen_set, cr_capacity_mbps,
           radio_allocation="equal_share", admission_policy="fcfs"):
    """
    Satisfaction rate for *chosen_set* CR placement.
    Pure read — does not modify any simulation state.
    Mirrors the satisfaction logic in MetricsLogger.log(), including its
    radio_allocation ("equal_share"/"proportional") and admission_policy
    ("fcfs"/"priority") behaviour, so the greedy oracle scores each candidate
    placement under the same rules the simulator will actually apply for the
    step being decided -- otherwise it ranks placements using a model that
    doesn't match reality under M1-M3 (dissertation Ch.7).
    """
    cr_loads = {j: 0.0 for j in chosen_set}
    satisfied = 0

    # Total requested demand per node, needed for proportional sharing --
    # mirrors MetricsLogger.log()'s demand_by_node, computed once regardless
    # of admission order since a node's total demand doesn't depend on which
    # of its users is scored first.
    demand_by_node = {}
    for u in users:
        node = u.connected_to
        if node is not None:
            demand_by_node[node] = demand_by_node.get(node, 0.0) + u.throughput_req_mbps

    for user in _admission_order(users, admission_policy):
        if user.connected_to is None:
            continue

        if isinstance(user.connected_to, BaseStation):
            bs = user.connected_to
            try:
                bs_idx = candidate_bs.index(bs)
            except ValueError:
                bs_idx = -1

            total_mbps = _BS_THROUGHPUT_MBPS.get(bs.bs_type, 200.0)
            if radio_allocation == "proportional":
                data_rate = _proportional_rate(
                    total_mbps, user.throughput_req_mbps, demand_by_node.get(bs, 0.0)
                )
            else:  # "equal_share"
                data_rate = total_mbps / max(1, bs.current_load)

            if bs_idx in chosen_set:
                if cr_loads[bs_idx] + user.throughput_req_mbps <= cr_capacity_mbps:
                    cr_loads[bs_idx] += user.throughput_req_mbps
                    target = "BS"
                else:
                    target = "Core"
            else:
                target = "Core"

            radio_ms   = math.dist(user.position, bs.position) * _RADIO_LATENCY_SCALE
            compute_ms = 0.0 if target == "BS" else _CORE_LATENCY_MS
            ok = (
                data_rate >= user.throughput_req_mbps
                and radio_ms + compute_ms <= user.latency_threshold_ms
            )

        elif isinstance(user.connected_to, RelayNode):
            path = get_path_to_bs(user.connected_to)
            if path is None:
                continue
            bs = path[-1]
            try:
                bs_idx = candidate_bs.index(bs)
            except ValueError:
                bs_idx = -1

            rn = user.connected_to
            if radio_allocation == "proportional":
                data_rate = _proportional_rate(
                    rn.throughput, user.throughput_req_mbps, demand_by_node.get(rn, 0.0)
                )
            else:  # "equal_share"
                data_rate = rn.throughput / max(1, rn.current_load)
            backhaul_los = all(getattr(n, "backhaul_los", True) for n in path[:-1])
            hop_count = len(path)

            if bs_idx in chosen_set:
                if cr_loads[bs_idx] + user.throughput_req_mbps <= cr_capacity_mbps:
                    cr_loads[bs_idx] += user.throughput_req_mbps
                    target = "BS"
                else:
                    target = "Core"
            else:
                target = "Core"

            radio_ms    = math.dist(user.position, rn.position) * _RADIO_LATENCY_SCALE
            backhaul_ms = (hop_count - 1) * _HOP_LATENCY_MS
            compute_ms  = 0.0 if target == "BS" else _CORE_LATENCY_MS
            ok = (
                data_rate >= user.throughput_req_mbps
                and radio_ms + backhaul_ms + compute_ms <= user.latency_threshold_ms
                and backhaul_los
            )

        else:
            continue

        if ok:
            satisfied += 1

    return satisfied / len(users) if users else 0.0


# ── Base class ────────────────────────────────────────────────────────────────

class _BaseStrategy:
    """Interface compatible with CRPlacementAgent for simulator.py."""
    prev_state         = None
    prev_action        = None
    last_reward        = None
    last_reward_global = None
    frozen             = True
    # Empty histories so finalize() works without special-casing strategies
    q_history       = []
    epsilon_history = []
    delta_q_history = []

    def get_state(self):           return ()
    def select_action(self, _):   return 0
    def apply_action(self, _):    pass
    def update(self, *a, **kw):   pass


# ── Strategy implementations ─────────────────────────────────────────────────

class NoCRStrategy(_BaseStrategy):
    """No compute resource active anywhere."""

    def __init__(self, candidate_bs, **_):
        self.candidate_bs = candidate_bs

    def apply_action(self, _):
        for bs in self.candidate_bs:
            bs.has_compute_resource = False
            bs.compute_resource = None


class RandomStrategy(_BaseStrategy):
    """K randomly chosen BS per step (seeded for reproducibility)."""

    def __init__(self, candidate_bs, k, cr_capacity_mbps):
        self.candidate_bs    = candidate_bs
        self.k               = k
        self.cr_capacity_mbps = cr_capacity_mbps
        self._cr_pool        = [ComputeRessources(i, (0, 0), cr_capacity_mbps, 0)
                                for i in range(k)]
        self._chosen         = set(range(min(k, len(candidate_bs))))

    def select_action(self, _):
        n = len(self.candidate_bs)
        self._chosen = set(random.sample(range(n), min(self.k, n)))
        return 0

    def apply_action(self, _):
        _apply(self.candidate_bs, self._chosen, self._cr_pool)


class StaticStrategy(_BaseStrategy):
    """
    CRs always on the K macro BS.
    If fewer than K macros exist, the remaining slots go to the first
    available non-macro BS (sorted by index).
    Placement is computed once at init and never changes.
    """

    def __init__(self, candidate_bs, k, cr_capacity_mbps):
        self.candidate_bs    = candidate_bs
        self._cr_pool        = [ComputeRessources(i, (0, 0), cr_capacity_mbps, 0)
                                for i in range(k)]
        macro   = [i for i, bs in enumerate(candidate_bs) if bs.bs_type == "macro"]
        others  = [i for i in range(len(candidate_bs)) if i not in macro]
        chosen  = macro[:k] + others[:max(0, k - len(macro))]
        self._chosen = set(chosen[:k])

    def apply_action(self, _):
        _apply(self.candidate_bs, self._chosen, self._cr_pool)


class ExhaustiveGreedyStrategy(_BaseStrategy):
    """
    Per-step exhaustive greedy: evaluates all C(N,K) placements via *_score*
    (read-only, no side-effects) and applies the one maximising immediate satisfaction.
    For N=4, K=2 this is 6 evaluations × n_users per step, a negligible cost.

    This baseline selects the step-optimal placement given perfect global knowledge
    of the current state. It serves as a single-step upper bound for the RL agent
    but does not account for future state transitions.

    Requires *_users* to be set to sim.users before the first step
    (done by run_experiment.py — same list reference, always up-to-date).
    radio_allocation/cr_admission_policy must likewise be set from
    sim.radio_allocation/sim.cr_admission_policy (Ch.7 M0-M3 factorial) so
    _score evaluates candidates under the same rules the simulator actually
    applies -- default to the M0 values so any caller that never sets them
    keeps the original equal_share + fcfs behaviour.
    """

    def __init__(self, candidate_bs, k, cr_capacity_mbps):
        self.candidate_bs     = candidate_bs
        self.k                = k
        self.cr_capacity_mbps = cr_capacity_mbps
        self._cr_pool         = [ComputeRessources(i, (0, 0), cr_capacity_mbps, 0)
                                 for i in range(k)]
        self._combos  = list(combinations(range(len(candidate_bs)), k))
        self._chosen  = set(range(min(k, len(candidate_bs))))
        self._users   = []   # set from run_experiment.py
        self.radio_allocation    = "equal_share"  # set from sim.radio_allocation
        self.cr_admission_policy = "fcfs"          # set from sim.cr_admission_policy

    def select_action(self, _):
        best_score, best_idx = -1.0, 0
        for idx, combo in enumerate(self._combos):
            s = _score(self._users, self.candidate_bs,
                       set(combo), self.cr_capacity_mbps,
                       radio_allocation=self.radio_allocation,
                       admission_policy=self.cr_admission_policy)
            if s > best_score:
                best_score, best_idx = s, idx
        self._chosen = set(self._combos[best_idx])
        return best_idx

    def apply_action(self, _):
        _apply(self.candidate_bs, self._chosen, self._cr_pool)


# ── Factory ───────────────────────────────────────────────────────────────────

_REGISTRY = {
    "no_cr":          NoCRStrategy,
    "random":         RandomStrategy,
    "static":         StaticStrategy,
    "exhaustive_greedy": ExhaustiveGreedyStrategy,
}

STRATEGY_NAMES = list(_REGISTRY)


def make_strategy(name, candidate_bs, k, cr_capacity_mbps):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {STRATEGY_NAMES}")
    return _REGISTRY[name](candidate_bs, k=k, cr_capacity_mbps=cr_capacity_mbps)
