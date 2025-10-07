from math import sqrt
import random
import os
import pickle

class QAgent:
    def __init__(self, id, learning_rate=0.2, discount=0.95, epsilon=0.3):
        self.id = id
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.min_epsilon = 0.05
        self.decay_rate = 0.995  # slower decay

    def select_action(self, state, actions):
        if state not in self.q_table:
            self.q_table[state] = {i: 0.0 for i in range(len(actions))}
        if random.random() < self.epsilon:
            return random.randint(0, len(actions) - 1)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state, num_actions):
        if state not in self.q_table:
            self.q_table[state] = {i: 0.0 for i in range(num_actions)}
        if next_state not in self.q_table:
            self.q_table[next_state] = {i: 0.0 for i in range(num_actions)}

        old_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.q_table[state][action] = new_q
        self.decay_epsilon()

        # Optional: Debug print
        print(f"[RN{self.id}] Q-update: State={state}, Action={action}, OldQ={old_q:.2f}, NewQ={new_q:.2f}, Reward={reward:.2f}")

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


class RelayNode:
    def __init__(self, id, position, throughput=20, move_range=10):
        self.id = id
        self.position = position
        self.throughput = throughput
        self.range = 50
        self.capacity = 5
        self.current_load = 0
        self.status = "active"
        self.move_range = move_range
        self.agent = QAgent(id)
        self.possible_positions = self.generate_possible_positions()
        self.last_action = 0

        # Learning logs
        self.q_history = []
        self.epsilon_history = []

    def generate_possible_positions(self):
        grid_size = 10
        positions = []
        for x in range(grid_size, 100, grid_size):
            for y in range(grid_size, 100, grid_size):
                positions.append((x, y))
        return positions

    def reset(self):
        self.current_load = 0

    def move(self, new_position):
        x = max(0, min(100, int(new_position[0])))
        y = max(0, min(100, int(new_position[1])))
        self.position = (x, y)

    def step(self, users):
        state = self.get_state(users)
        action = self.agent.select_action(state, self.possible_positions)
        new_pos = self.possible_positions[action]
        self.move(new_pos)
        print(f"[RN{self.id}] moved to {self.position}")
        self.last_action = action
        self.update_q_value(users, prev_state=state)

    def get_state(self, users):
        x, y = self.position
        x_bin = int(x // 10)
        y_bin = int(y // 10)
        nearby = sum(1 for u in users if self.distance(u.position) <= self.range)
        return (x_bin, y_bin, nearby)

    def update_q_value(self, users, prev_state):
        connected_users = sum(1 for u in users if u.connected_to == self)
        reward = connected_users  # Encourages useful positions
        next_state = self.get_state(users)
        self.agent.update(prev_state, self.last_action, reward, next_state, len(self.possible_positions))
        self.log_learning()

    def log_learning(self):
        # Average Q-value across all Q-table entries
        total_q = 0.0
        count = 0
        for state_actions in self.agent.q_table.values():
            for q in state_actions.values():
                total_q += q
                count += 1
        avg_q = total_q / count if count > 0 else 0.0
        self.q_history.append(avg_q)
        self.epsilon_history.append(self.agent.epsilon)

    def save_learning_logs(self):
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/rn{self.id}_q.pkl", "wb") as f:
            pickle.dump(self.q_history, f)
        with open(f"logs/rn{self.id}_epsilon.pkl", "wb") as f:
            pickle.dump(self.epsilon_history, f)

    def distance(self, user_pos):
        return sqrt((self.position[0] - user_pos[0]) ** 2 + (self.position[1] - user_pos[1]) ** 2)

    def __str__(self):
        return f"RN{self.id}:{self.current_load}"


def evaluate_and_adjust_relay_nodes(relay_nodes, users, max_rns=15, min_rns=2):
    uncovered_users = [
        user for user in users
        if not any(rn.distance(user.position) <= rn.range for rn in relay_nodes)
    ]
    idle_rns = [rn for rn in relay_nodes if rn.current_load == 0]

    if len(uncovered_users) > 5 and len(relay_nodes) < max_rns:
        new_id = max(rn.id for rn in relay_nodes) + 1
        new_rn = RelayNode(new_id, position=(50, 50))  # Optional: smarter placement
        relay_nodes.append(new_rn)
        print(f"[Controller] ➕ Added RN{new_id}")

    elif len(idle_rns) >= 2 and len(relay_nodes) > min_rns:
        to_remove = idle_rns[0]
        relay_nodes.remove(to_remove)
        print(f"[Controller] ➖ Removed RN{to_remove.id}")
