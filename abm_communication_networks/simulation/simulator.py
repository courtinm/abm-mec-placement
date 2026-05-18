import math
import random
import os
import csv
from agents.relay_node import evaluate_and_adjust_relay_nodes
from agents.base_station import BaseStation
from agents.relay_node import RelayNode

# Total downlink throughput per BS type (Mbps).
# Macro: 5G NR 100 MHz @ ~10 b/s/Hz; Small: LTE-A 20 MHz @ ~10 b/s/Hz.
# Per-user rate = _BS_THROUGHPUT_MBPS[bs_type] / current_load.
_BS_THROUGHPUT_MBPS = {"macro": 1000.0, "small": 200.0}

def is_line_blocked(src, dst, obstacles, allow_through_large=False):
    x1, y1 = src
    x2, y2 = dst
    for obs in obstacles:
        if isinstance(obs, dict):
            ox, oy = obs['pos']
            size = obs.get('size', 'small')
        else:
            ox, oy = obs
            size = 'small'

        if size == 'large' and allow_through_large:
            continue

        if math.isclose((y2 - y1) * (ox - x1), (x2 - x1) * (oy - y1), abs_tol=1e-6):
            if min(x1, x2) <= ox <= max(x1, x2) and min(y1, y2) <= oy <= max(y1, y2):
                return True
    return False

def path_loss_mmwave(d, frequency_GHz=28, shadow_std=3.0):
    if d == 0:
        return -30
    c = 3e8
    freq = frequency_GHz * 1e9
    lambda_ = c / freq
    pl = 20 * math.log10(4 * math.pi * d / lambda_)
    shadow = random.gauss(0, shadow_std)
    return pl + shadow

class MetricsLogger:
    def __init__(self):
        self.latency_avg = []
        self.latency_max = []
        self.failed_connections = []
        self.nlos_connections = []
        self.handoffs = []
        self.optimal_connections = []
        self.prev_connections = {}
        self.hop_count_to_BS = {}  #contains each UE connected to a BS with CR the number of hops
        self.hop_count_to_core_network = {} #contains each UE connected to a BS without CR so the compute is done in the core network
        self.hop_counts_log = []  # accumulates (step, user_id, target, hop_count) for CSV
        self.satisfaction_users_log = []    # one row per user per step
        self.satisfaction_summary_log = []  # one row per step

    def log(self, step, users):
        total_latency = []
        failures = 0
        nlos = 0
        handoffs = 0
        optimal = 0

        # Reset each step — connections change as users move
        self.hop_count_to_BS = {}
        self.hop_count_to_core_network = {}

        rows_before = len(self.hop_counts_log)

        for user in users:
            connected = user.connected_to is not None

            # Count hops to compute resource and compute satisfaction
            if isinstance(user.connected_to, BaseStation):
                node = user.connected_to
                total_mbps = _BS_THROUGHPUT_MBPS.get(node.bs_type, 200.0)
                # data_rate in Mbps: node capacity shared equally among connected users
                data_rate_mbps = total_mbps / max(1, node.current_load)
                # TODO (Tier 3): add latency_threshold_ms condition when end-to-end latency model is implemented
                user.is_satisfied = data_rate_mbps >= user.throughput_req_mbps
                if node.has_compute_resource:
                    target = "BS"
                    self.hop_count_to_BS[user.id] = 1
                    self.hop_counts_log.append((step, user.id, "BS", 1))
                else:
                    target = "Core"
                    # TODO (IAB): add BS->core segment once multi-hop routing is implemented
                    self.hop_count_to_core_network[user.id] = 1
                    self.hop_counts_log.append((step, user.id, "Core", 1))
            elif isinstance(user.connected_to, RelayNode):
                target = "RN"
                data_rate_mbps = 0.0
                user.is_satisfied = False
                # TODO (IAB): hop count will be 2 (user->RN->BS) once multi-hop routing is implemented
                self.hop_counts_log.append((step, user.id, "RN", ""))
            else:
                target = "Disconnected"
                data_rate_mbps = 0.0
                user.is_satisfied = False
                self.hop_counts_log.append((step, user.id, "Disconnected", ""))

            self.satisfaction_users_log.append((
                step, user.id, user.app_type, user.throughput_req_mbps,
                round(data_rate_mbps, 2), target, user.is_satisfied,
            ))

            if connected:
                dist = math.dist(user.position, user.connected_to.position)
                latency = dist / 3.0  # simple latency model
                total_latency.append(latency)

                if not user.has_los:
                    nlos += 1

                if getattr(user.connected_to, 'current_load', 0) <= getattr(user.connected_to, 'capacity', 0):
                    if user.has_los:
                        optimal += 1
            else:
                failures += 1

            # Check for handoff
            prev = self.prev_connections.get(user.id)
            if connected and prev and prev != user.connected_to:
                handoffs += 1

            self.prev_connections[user.id] = user.connected_to

        assert len(self.hop_counts_log) - rows_before == len(users), (
            f"Step {step}: expected {len(users)} hop rows, got {len(self.hop_counts_log) - rows_before}"
        )

        app_keys = ["AR_VR", "streaming", "best_effort"]
        n_by_app = {a: 0 for a in app_keys}
        sat_by_app = {a: 0 for a in app_keys}
        for user in users:
            n_by_app[user.app_type] += 1
            if user.is_satisfied:
                sat_by_app[user.app_type] += 1
        n_total = len(users)
        n_sat = sum(sat_by_app.values())

        def _rate(a):
            return round(sat_by_app[a] / n_by_app[a], 4) if n_by_app[a] else 0.0

        self.satisfaction_summary_log.append((
            step, n_total, n_sat, round(n_sat / n_total, 4) if n_total else 0.0,
            n_by_app["AR_VR"],      sat_by_app["AR_VR"],      _rate("AR_VR"),
            n_by_app["streaming"],  sat_by_app["streaming"],   _rate("streaming"),
            n_by_app["best_effort"],sat_by_app["best_effort"], _rate("best_effort"),
        ))

        print(f"[Step {step}] hop_to_BS={len(self.hop_count_to_BS)} hop_to_core={len(self.hop_count_to_core_network)}")

        avg_latency = sum(total_latency) / len(total_latency) if total_latency else 0
        max_latency = max(total_latency) if total_latency else 0
        optimal_pct = (optimal / len(users)) * 100 if users else 0

        self.latency_avg.append((step, avg_latency))
        self.latency_max.append((step, max_latency))
        self.failed_connections.append((step, failures))
        self.nlos_connections.append((step, nlos))
        self.handoffs.append((step, handoffs))
        self.optimal_connections.append((step, optimal_pct))



    def save_all(self, folder="logs"):
        os.makedirs(folder, exist_ok=True)

        self._save_csv(os.path.join(folder, "latency_avg.csv"), self.latency_avg)
        self._save_csv(os.path.join(folder, "latency_max.csv"), self.latency_max)
        self._save_csv(os.path.join(folder, "failed_connections.csv"), self.failed_connections)
        self._save_csv(os.path.join(folder, "nlos_connections.csv"), self.nlos_connections)
        self._save_csv(os.path.join(folder, "handoffs.csv"), self.handoffs)
        self._save_csv(os.path.join(folder, "optimal_connections.csv"), self.optimal_connections)

        hop_path = os.path.join(folder, "hop_counts.csv")
        with open(hop_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "UserID", "Target", "HopCount"])
            writer.writerows(self.hop_counts_log)

        with open(os.path.join(folder, "satisfaction_users.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "UserID", "AppType", "ThroughputReq_Mbps",
                             "DataRate_Mbps", "Target", "Satisfied"])
            writer.writerows(self.satisfaction_users_log)

        with open(os.path.join(folder, "satisfaction_summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "N_Total", "N_Satisfied", "Rate_Global",
                             "N_AR_VR", "Sat_AR_VR", "Rate_AR_VR",
                             "N_Streaming", "Sat_Streaming", "Rate_Streaming",
                             "N_BestEffort", "Sat_BestEffort", "Rate_BestEffort"])
            writer.writerows(self.satisfaction_summary_log)

    def _save_csv(self, path, data):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Value"])
            writer.writerows(data)

class Simulator:
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.base_stations = []
        self.relay_nodes = []
        self.users = []
        self.obstacles = []
        self.timestep = 0
        self.debug_logs = []
        self.metrics = MetricsLogger()

    def add_base_station(self, bs):
        self.base_stations.append(bs)

    def add_relay_node(self, rn):
        self.relay_nodes.append(rn)

    def add_user(self, user):
        self.users.append(user)

    def add_obstacle(self, pos, size='small'):
        self.obstacles.append({'pos': pos, 'size': size})

    def simulate_step(self):
        self.timestep += 1
        self.debug_logs.clear()

        for user in self.users:
            dx, dy = random.randint(-5, 5), random.randint(-5, 5)
            user.move((dx, dy))

        for bs in self.base_stations:
            bs.reset()
        for rn in self.relay_nodes:
            rn.reset()

        for rn in self.relay_nodes:
            rn.step(self.users)

        for user in self.users:
            user.has_los = False
            candidates_los = []
            candidates_nlos = []
            blocked_nodes = []

            for node in self.base_stations + self.relay_nodes:
                dist = math.dist(user.position, node.position)
                can_penetrate = hasattr(node, 'bs_type') and node.bs_type == 'macro'
                is_blocked = is_line_blocked(node.position, user.position, self.obstacles, allow_through_large=can_penetrate)

                signal_loss = path_loss_mmwave(dist)
                if signal_loss > 110:
                    blocked_nodes.append((node, "Signal too weak"))
                    continue

                if is_blocked and not can_penetrate:
                    blocked_nodes.append((node, "Blocked by obstacle"))
                    candidates_nlos.append((node, signal_loss))
                else:
                    candidates_los.append((node, signal_loss))

            connected = False

            if candidates_los:
                best_node, _ = min(candidates_los, key=lambda tup: tup[1])
                capacity = getattr(best_node, 'capacity', getattr(best_node, 'throughput', 0))

                if best_node.current_load < capacity:
                    user.connected_to = best_node
                    best_node.current_load += 1
                    user.has_los = True
                    connected = True
                else:
                    self.debug_logs.append(f"[User {user.id}] LOS node overloaded: {best_node.id}")

            if not connected and candidates_nlos:
                best_node, _ = min(candidates_nlos, key=lambda tup: tup[1])
                capacity = getattr(best_node, 'capacity', getattr(best_node, 'throughput', 0))

                if best_node.current_load < capacity:
                    user.connected_to = best_node
                    best_node.current_load += 1
                    user.has_los = False
                    connected = True
                    log_line = f"[User {user.id}] Connected to node {best_node.id} WITHOUT line-of-sight (penalty applied)"
                    print(log_line)
                    self.debug_logs.append(log_line)
                else:
                    self.debug_logs.append(f"[User {user.id}] NLOS node overloaded: {best_node.id}")

            if not connected:
                user.connected_to = None
                user.disconnections += 1
                reasons = [reason for node, reason in blocked_nodes] or ["No suitable nodes nearby"]
                log_line = f"[User {user.id}] No connection. Reasons: {reasons}"
                print(log_line)
                self.debug_logs.append(log_line)

        if self.timestep % 50 == 0:
            evaluate_and_adjust_relay_nodes(self.relay_nodes, self.users)

        self.metrics.log(self.timestep, self.users)

    def finalize(self, output_dir="logs"):
        os.makedirs(output_dir, exist_ok=True)

        for rn in self.relay_nodes:
            filename = os.path.join(output_dir, f"rn{rn.id}_learning.csv")
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Max_Q", "Epsilon"])
                for i, (q, eps) in enumerate(zip(rn.q_history, rn.epsilon_history)):
                    writer.writerow([i, q, eps])
            rn.save_learning_logs(output_dir)

        self.metrics.save_all(output_dir)

        debug_file = os.path.join(output_dir, f"debug_step_{self.timestep}.log")
        with open(debug_file, "w") as f:
            f.write(f"Step: {self.timestep}\n")
            f.write("\n".join(self.debug_logs))

        print(f"[Logs saved to '{output_dir}/' folder.")
