from itertools import combinations
import os
import pickle
import random

from agents.base_station import ComputeRessources


class CRPlacementAgent:
    """
    Q-learning agent for dynamic Compute Resource placement.

    State:  tuple of N demand levels (0=none, 1=low, 2=high) derived from
            each candidate BS's current_load (set during user attachment).
    Action: index into the C(N,K) enumerated placements.
    Reward: global satisfaction rate in [0, 1] (fraction of satisfied users).

    Q-update timing (cross-step):
        Step t  → observe S_t, choose A_t, apply A_t, metrics → R_t stored.
        Step t+1 → observe S_{t+1}, Q-update(S_t, A_t, R_t, S_{t+1}),
                   choose A_{t+1}, apply A_{t+1}, ...
    """

    # Demand discretisation: 0 → none, 1 → low (1..LOW_MAX), 2 → high (>LOW_MAX)
    _LOW_MAX = 3

    def __init__(self, candidate_bs, k, cr_capacity_mbps,
                 learning_rate=0.1, discount=0.9, epsilon=0.5):
        self.candidate_bs = candidate_bs   # ordered list of BaseStation objects
        self.k = k
        self.n = len(candidate_bs)
        # Enumerate all C(N, K) placements once
        self.actions = list(combinations(range(self.n), k))
        # Pool of K reusable ComputeRessources objects (reassigned each step)
        self._cr_pool = [
            ComputeRessources(i, (0, 0), cr_capacity_mbps, 0)
            for i in range(k)
        ]

        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.min_epsilon = 0.05
        self.decay_rate = 0.995
        self.frozen = False

        # Live references set by run_experiment after build_simulation()
        # (same list objects → always reflect current step state)
        self._users = []
        self._relay_nodes = []

        # Cross-step bookkeeping for Q-update
        self.prev_state = None
        self.prev_action = None
        self.last_reward = None

        # Learning history (one entry per Q-update)
        self.q_history = []
        self.epsilon_history = []
        self.delta_q_history = []  # |new_Q - old_Q| per update

    # ------------------------------------------------------------------
    # MDP
    # ------------------------------------------------------------------

    def _demand_level(self, bs):
        direct = bs.current_load
        rn_users = sum(
            1 for rn in self._relay_nodes if rn.parent is bs
            for u in self._users if u.connected_to is rn
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
        """Activate CR on the K chosen BS; deactivate and unlink on others."""
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
        self._log_learning()

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
