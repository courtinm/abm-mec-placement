from itertools import combinations
import os
import pickle
import random

from agents.base_station import ComputeRessources

# Must match simulator.py — core penalty saved when CR serves a user locally.
_CORE_LATENCY_MS = 50.0


class CRPlacementAgent:
    """
    Q-learning agent for dynamic Compute Resource placement.

    State:  tuple of N demand levels (0=none, 1=low, 2=high) derived from
            each candidate BS's current_load (set during user attachment).
    Action: index into the C(N,K) enumerated placements.
    Reward: shaped = r_counterfactual + lambda * r_latency_reduction.

    Q-update timing (cross-step):
        Step t  → observe S_t, choose A_t, apply A_t, metrics → R_t stored.
        Step t+1 → observe S_{t+1}, Q-update(S_t, A_t, R_t, S_{t+1}),
                   choose A_{t+1}, apply A_{t+1}, ...
    """

    _LOW_MAX = 3

    def __init__(self, candidate_bs, k, cr_capacity_mbps,
                 learning_rate=0.1, discount=0.9, epsilon=0.5,
                 epsilon_min=0.05, epsilon_decay=0.995,
                 alpha_min=0.01, alpha_decay=0.998,
                 reward_shaping_lambda=0.0):
        self.candidate_bs = candidate_bs
        self.k = k
        self.n = len(candidate_bs)
        self.actions = list(combinations(range(self.n), k))
        self._cr_pool = [
            ComputeRessources(i, (0, 0), cr_capacity_mbps, 0)
            for i in range(k)
        ]

        self.q_table = {}
        self.alpha = learning_rate
        self.alpha_min = alpha_min
        self.alpha_decay_rate = alpha_decay
        self.gamma = discount
        self.epsilon = epsilon
        self.min_epsilon = epsilon_min
        self.decay_rate = epsilon_decay
        self.reward_shaping_lambda = reward_shaping_lambda
        self.frozen = False

        self._users = []
        self._relay_nodes = []

        self.prev_state = None
        self.prev_action = None
        self.last_reward = None               # shaped reward (used for Q-update)
        self.last_reward_counterfactual = None  # counterfactual only (for logging)
        self.last_reward_shaping = None         # lambda * r_latency (for logging)
        self.last_reward_global = None          # global satisfaction (for logging)

        self.q_history = []
        self.epsilon_history = []
        self.delta_q_history = []

    # ------------------------------------------------------------------
    # MDP
    # ------------------------------------------------------------------

    def _demand_level(self, bs):
        direct = sum(
            1 for u in self._users
            if u.connected_to is bs
            and getattr(u, "app_type", None) in ("AR_VR", "streaming")
        )
        rn_users = sum(
            1 for rn in self._relay_nodes if rn.parent is bs
            for u in self._users
            if u.connected_to is rn
            and getattr(u, "app_type", None) in ("AR_VR", "streaming")
        )
        load = direct + rn_users
        if load == 0:
            return 0
        return 1 if load <= self._LOW_MAX else 2

    def get_state(self):
        return tuple(self._demand_level(bs) for bs in self.candidate_bs)

    def select_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {i: 0.0 for i in range(len(self.actions))}
        if not self.frozen and random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        return max(self.q_table[state], key=self.q_table[state].get)

    def apply_action(self, action_idx):
        chosen = set(self.actions[action_idx])
        pool_iter = iter(self._cr_pool)
        for i, bs in enumerate(self.candidate_bs):
            if i in chosen:
                cr = next(pool_iter)
                cr.current_load_mbps = 0.0
                cr.demanded_load_mbps = 0.0
                bs.has_compute_resource = True
                bs.compute_resource = cr
            else:
                bs.has_compute_resource = False
                bs.compute_resource = None

    def update(self, state, action, reward, next_state):
        if self.frozen:
            return
        n_act = len(self.actions)
        for s in (state, next_state):
            if s not in self.q_table:
                self.q_table[s] = {i: 0.0 for i in range(n_act)}
        old_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q
        self.delta_q_history.append(max(abs(new_q - old_q), 1e-10))
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        self.alpha   = max(self.alpha_min,   self.alpha   * self.alpha_decay_rate)
        self._log_learning()

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _compute_latency_shaping(self):
        """
        r_latency_reduction: mean normalised latency saving for users
        currently served by a CR-equipped BS.

        Saving = _CORE_LATENCY_MS (fixed core penalty avoided).
        Normalised by user.latency_threshold_ms, capped at 1.
        NLoS users contribute 0 (radio impairment dominates).
        """
        cr_bs_set = {bs for bs in self.candidate_bs if bs.has_compute_resource}
        if not cr_bs_set:
            return 0.0

        contributions = []
        for user in self._users:
            conn = user.connected_to
            if conn is None:
                continue

            # Direct BS connection
            if conn in cr_bs_set:
                served = True
            # RN → parent BS connection (single hop)
            elif hasattr(conn, 'parent') and conn.parent in cr_bs_set:
                served = True
            else:
                served = False

            if not served:
                continue

            if not getattr(user, 'has_los', True):
                contributions.append(0.0)
            else:
                normalized = min(1.0, _CORE_LATENCY_MS / user.latency_threshold_ms)
                contributions.append(normalized)

        return sum(contributions) / len(contributions) if contributions else 0.0

    def compute_and_set_reward(self, counterfactual):
        """Called by simulator after metrics.log(); sets all three reward attributes."""
        shaping = self._compute_latency_shaping()
        self.last_reward_counterfactual = counterfactual
        self.last_reward_shaping = self.reward_shaping_lambda * shaping
        self.last_reward = counterfactual + self.last_reward_shaping

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _log_learning(self):
        total_q, count = 0.0, 0
        for state_actions in self.q_table.values():
            for q in state_actions.values():
                total_q += q
                count += 1
        self.q_history.append(total_q / count if count else 0.0)
        self.epsilon_history.append(self.epsilon)

    def save_qtable(self, path):
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_qtable(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        max_states = 3 ** self.n
        n_actions = len(self.actions)
        print(f"[CR] Q-table loaded from '{path}': "
              f"{len(self.q_table)}/{max_states} states visited, "
              f"{n_actions} actions  (n={self.n} BS, k={self.k})")
