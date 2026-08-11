import math
import random
import os
import csv
from agents.relay_node import RelayNode, evaluate_and_adjust_relay_nodes
from agents.base_station import BaseStation

# Total downlink throughput per BS type (Mbps).
# Macro: 5G NR 100 MHz @ ~10 b/s/Hz; Small: LTE-A 20 MHz @ ~10 b/s/Hz.
# Per-user rate = _BS_THROUGHPUT_MBPS[bs_type] / current_load.
_BS_THROUGHPUT_MBPS = {"macro": 1000.0, "small": 200.0}

# End-to-end latency model.
# Radio latency: dist_grid_units * _RADIO_LATENCY_SCALE ms  (0.1 ms/unit ≈ 1 ms / 10 m).
# Core penalty: additional RTT when compute is routed to the core network rather than
#               served locally by an edge CR (realistic estimate: 50 ms).
# Backhaul: _HOP_LATENCY_MS per relay hop beyond the first.
_RADIO_LATENCY_SCALE = 0.1   # ms per grid unit
_CORE_LATENCY_MS    = 50.0   # ms added when compute hits the core network
_HOP_LATENCY_MS     = 2.0    # ms per extra relay hop

# CR admission priority, used only to order the loop in MetricsLogger.log()
# below. Replaces the earlier FCFS (list-order) admission policy: under FCFS,
# CR capacity was granted to whichever user happened to appear first in
# `users` (initialisation order), which could let a best-effort user occupy
# CR capacity while a concurrently saturated AR/VR user overflowed to the
# core despite its stricter latency threshold — a limitation documented in
# Section 5.3.1 of the dissertation. Latency-critical AR/VR users are now
# admitted first, then streaming, then best-effort; ties within a class are
# broken by user.id so admission order stays deterministic under a fixed
# seed. This changes only iteration order — cr.can_serve / cr.add_user and
# all other admission logic are unchanged.
_APP_PRIORITY = {"AR_VR": 0, "streaming": 1, "best_effort": 2}


def _admission_order(users, policy="priority"):
    # "fcfs" reproduces the pre-fix list-order behaviour and exists solely so
    # that the before/after comparison experiment (dissertation Ch.7) can
    # reproduce the old policy without duplicating the simulator; every
    # normal caller uses the "priority" default and is unaffected.
    if policy == "fcfs":
        return list(users)
    return sorted(users, key=lambda u: (_APP_PRIORITY.get(u.app_type, len(_APP_PRIORITY)), u.id))

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

def get_path_to_bs(node):
    """
    Walk up through parent links from node (a RelayNode) until a
    BaseStation is reached.
    Returns the list [node, parent1, ..., base_station].
    HopCount for the user = len(returned list)
        (1 hop per link: user->RN, RN->parent, ..., parent->BS)
    Returns None if parent chain is broken (None encountered).
    """
    path = []
    current = node
    visited = set()
    while not isinstance(current, BaseStation):
        if current is None or id(current) in visited:
            return None  # broken chain
        visited.add(id(current))
        path.append(current)
        current = getattr(current, 'parent', None)
    path.append(current)  # append the BS
    return path


def _proportional_rate(capacity_mbps, user_req_mbps, node_demand_mbps):
    """Radio data rate for one user at a node with total capacity
    *capacity_mbps*, given the node's total requested demand
    *node_demand_mbps* (sum of throughput_req_mbps over all users currently
    connected to it). Fully served (= its own request) while aggregate
    demand fits under capacity; otherwise shared out proportionally to each
    user's own request rather than split equally per head. Replaces the
    earlier equal-share model (capacity / connected_count), which ignored
    how much each user actually needed.
    """
    if node_demand_mbps <= capacity_mbps:
        return user_req_mbps
    return capacity_mbps * (user_req_mbps / node_demand_mbps)


def _serving_bs_id(user):
    """The id of the base station a user is currently anchored to, whether
    directly connected or reached via a relay-node backhaul chain. None if
    disconnected or the parent chain is broken. Used only by the migration
    cost overlay in MetricsLogger.log() below.
    """
    conn = user.connected_to
    if conn is None:
        return None
    if isinstance(conn, BaseStation):
        return conn.id
    if isinstance(conn, RelayNode):
        path = get_path_to_bs(conn)
        return path[-1].id if path is not None else None
    return None


def deployment_link_budget(src_pos, dst_pos, obstacles, frequency_GHz=28, nlos_penalty_dB=30.0):
    """Deterministic path loss for deployment-time decisions (no shadow fading).
    Adds nlos_penalty_dB if the link is blocked — 3GPP TR 38.901 typical value.
    """
    d = math.dist(src_pos, dst_pos)
    if d == 0:
        return -30
    c = 3e8
    lambda_ = c / (frequency_GHz * 1e9)
    pl = 20 * math.log10(4 * math.pi * d / lambda_)
    if is_line_blocked(src_pos, dst_pos, obstacles):
        pl += nlos_penalty_dB
    return pl


class MetricsLogger:
    def __init__(self):
        # Both default to the original (M0) behaviour: FCFS admission order
        # and equal-share radio rate. Existing M0 Q-tables/results were
        # trained/produced under exactly these two defaults and must stay
        # reproducible without passing any flag — see Simulator.__init__.
        self.admission_policy = "fcfs"        # "fcfs" (M0) | "priority" (M1/M3)
        self.radio_allocation = "equal_share"  # "equal_share" (M0) | "proportional" (M2/M3)
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
        self.cr_utilization_log = []        # one row per step per BS with CR
        self.cr_counterfactual_reward = 0.0  # users saved by CR / n_total

        # Migration-cost overlay (eval only — see Simulator.migration_cost_mode
        # and the "Migration cost overlay" block in log() below). Set by
        # Simulator.simulate_step() before calling log(); default is a no-op
        # so behaviour is byte-identical to before this feature when unset
        # (always the case during train_rn/train_cr).
        self.migration_cost_mode = None            # None | "hard_cutover" | "make_before_break"
        self.migration_lost_bs_ids = set()          # BS ids that just lost their CR this step
        self.migration_lost_capacity_mbps = 0.0     # sum of capacity_mbps of those BS's (old) CRs
        self.migration_overhead_log = []            # one row per step: (step, migration_overhead_mbps)

    def log(self, step, users):
        total_latency = []
        failures = 0
        nlos = 0
        handoffs = 0
        optimal = 0

        # Reset each step — connections change as users move
        self.hop_count_to_BS = {}
        self.hop_count_to_core_network = {}
        n_helped = 0  # counterfactual: users saved by CR this step

        rows_before = len(self.hop_counts_log)

        # Total requested demand per node (BS or RN), used below for
        # proportional radio-rate sharing (_proportional_rate). One pass
        # over *users* regardless of admission order, since a node's total
        # demand does not depend on which of its users gets processed first.
        demand_by_node = {}
        for u in users:
            node = u.connected_to
            if node is not None:
                demand_by_node[node] = demand_by_node.get(node, 0.0) + u.throughput_req_mbps

        # Rows are staged here rather than appended straight to
        # satisfaction_users_log, because the migration-cost overlay below
        # (applied after this loop) may still flip user.is_satisfied for
        # "hard_cutover" and needs to contribute two extra columns; each row
        # reads user.is_satisfied fresh when committed after the overlay, so
        # every row reflects the final, post-overlay value exactly once.
        pending_rows = []

        # Priority-ordered admission (see _admission_order docstring above):
        # only the iteration order changes here, not the admission logic
        # itself (cr.can_serve / cr.add_user, below, are untouched). Rows
        # logged per user still cover every user exactly once, just in
        # priority order rather than list order — CSV row order changes,
        # per-user content does not.
        for user in _admission_order(users, self.admission_policy):
            connected = user.connected_to is not None

            # Count hops to compute resource and compute satisfaction
            if isinstance(user.connected_to, BaseStation):
                node = user.connected_to
                total_mbps = _BS_THROUGHPUT_MBPS.get(node.bs_type, 200.0)
                if self.radio_allocation == "proportional":
                    data_rate_mbps = _proportional_rate(
                        total_mbps, user.throughput_req_mbps, demand_by_node.get(node, 0.0)
                    )
                else:  # "equal_share" (M0 default) — node capacity split equally per connected user
                    data_rate_mbps = total_mbps / max(1, node.current_load)
                if node.has_compute_resource and node.compute_resource is not None:
                    cr = node.compute_resource
                    cr.demanded_load_mbps += user.throughput_req_mbps
                    if cr.can_serve(user.throughput_req_mbps):
                        cr.add_user(user.throughput_req_mbps)
                        target = "BS"
                        self.hop_count_to_BS[user.id] = 1
                        self.hop_counts_log.append((step, user.id, "BS", 1, ""))
                    else:
                        target = "Core"
                        self.hop_count_to_core_network[user.id] = 1
                        self.hop_counts_log.append((step, user.id, "Core", 1, ""))
                else:
                    target = "Core"
                    self.hop_count_to_core_network[user.id] = 1
                    self.hop_counts_log.append((step, user.id, "Core", 1, ""))
                radio_ms = math.dist(user.position, node.position) * _RADIO_LATENCY_SCALE
                compute_ms = 0.0 if target == "BS" else _CORE_LATENCY_MS
                user.is_satisfied = (
                    data_rate_mbps >= user.throughput_req_mbps
                    and radio_ms + compute_ms <= user.latency_threshold_ms
                )
                if target == "BS" and user.is_satisfied:
                    sat_no_cr = (data_rate_mbps >= user.throughput_req_mbps
                                 and radio_ms + _CORE_LATENCY_MS <= user.latency_threshold_ms)
                    if not sat_no_cr:
                        n_helped += 1
            elif isinstance(user.connected_to, RelayNode):
                path = get_path_to_bs(user.connected_to)
                if path is None:
                    # Broken parent chain — should not happen
                    print(f"[WARNING] Step {step}: broken parent chain for RN{user.connected_to.id}")
                    target = "RN"
                    data_rate_mbps = 0.0
                    user.is_satisfied = False
                    self.hop_counts_log.append((step, user.id, "RN", "", ""))
                else:
                    bs = path[-1]
                    hop_count = len(path)
                    rn = user.connected_to
                    if self.radio_allocation == "proportional":
                        data_rate_mbps = _proportional_rate(
                            rn.throughput, user.throughput_req_mbps, demand_by_node.get(rn, 0.0)
                        )
                    else:  # "equal_share" (M0 default) — RN throughput split equally per connected user
                        data_rate_mbps = rn.throughput / max(1, rn.current_load)
                    # backhaul_los is False if any RN in the path has a NLoS link to its parent
                    backhaul_los = all(getattr(n, 'backhaul_los', True) for n in path[:-1])
                    if bs.has_compute_resource and bs.compute_resource is not None:
                        cr = bs.compute_resource
                        cr.demanded_load_mbps += user.throughput_req_mbps
                        if cr.can_serve(user.throughput_req_mbps):
                            cr.add_user(user.throughput_req_mbps)
                            target = "BS"
                            self.hop_count_to_BS[user.id] = hop_count
                        else:
                            target = "Core"
                            self.hop_count_to_core_network[user.id] = hop_count
                    else:
                        target = "Core"
                        self.hop_count_to_core_network[user.id] = hop_count
                    self.hop_counts_log.append((step, user.id, target, hop_count, backhaul_los))
                    radio_ms    = math.dist(user.position, rn.position) * _RADIO_LATENCY_SCALE
                    backhaul_ms = (hop_count - 1) * _HOP_LATENCY_MS
                    compute_ms  = 0.0 if target == "BS" else _CORE_LATENCY_MS
                    user.is_satisfied = (
                        data_rate_mbps >= user.throughput_req_mbps
                        and radio_ms + backhaul_ms + compute_ms <= user.latency_threshold_ms
                        and backhaul_los
                    )
                    if target == "BS" and user.is_satisfied:
                        sat_no_cr = (data_rate_mbps >= user.throughput_req_mbps
                                     and radio_ms + backhaul_ms + _CORE_LATENCY_MS <= user.latency_threshold_ms
                                     and backhaul_los)
                        if not sat_no_cr:
                            n_helped += 1
            else:
                target = "Disconnected"
                data_rate_mbps = 0.0
                user.is_satisfied = False
                self.hop_counts_log.append((step, user.id, "Disconnected", "", ""))

            pending_rows.append((user, target, data_rate_mbps))

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

        self.cr_counterfactual_reward = n_helped / len(users) if users else 0.0

        assert len(self.hop_counts_log) - rows_before == len(users), (
            f"Step {step}: expected {len(users)} hop rows, got {len(self.hop_counts_log) - rows_before}"
        )

        # ── Migration cost overlay (dissertation Ch.7, eval only) ──────────
        # Applied last, after every latency/throughput/satisfaction value
        # above has already been computed with the existing, untouched logic
        # — cr.demanded_load_mbps and cr_counterfactual_reward (just set,
        # above) are never touched by this block. Two modes, both gated on
        # migration_lost_bs_ids (BS that just lost their CR this step,
        # populated by Simulator.simulate_step() only when migration_cost_mode
        # is not None):
        #   "hard_cutover"      — forces is_satisfied=False for every user
        #                         currently anchored (directly or via RN) to
        #                         one of those BS, independent of the normal
        #                         latency/throughput result.
        #   "make_before_break" — does NOT touch is_satisfied (so no user is
        #                         ever forced unsatisfied by this transition);
        #                         it only records migration_overhead_mbps, the
        #                         capacity of the CR(s) that would need to
        #                         stay active one extra step to make the
        #                         transition seamless — informational only,
        #                         for a future RQ2 cost discussion, not wired
        #                         into satisfaction or reward at all.
        migration_overhead_mbps = 0.0
        if self.migration_cost_mode == "hard_cutover" and self.migration_lost_bs_ids:
            for user in users:
                if _serving_bs_id(user) in self.migration_lost_bs_ids:
                    user.is_satisfied = False
                    user._migration_penalty_applied = True
        elif self.migration_cost_mode == "make_before_break" and self.migration_lost_bs_ids:
            migration_overhead_mbps = self.migration_lost_capacity_mbps
            for user in users:
                if _serving_bs_id(user) in self.migration_lost_bs_ids:
                    user._migration_overhead_active = True

        self.migration_overhead_log.append((step, round(migration_overhead_mbps, 2)))

        for user, target, data_rate_mbps in pending_rows:
            self.satisfaction_users_log.append((
                step, user.id, user.app_type, user.throughput_req_mbps,
                round(data_rate_mbps, 2), target, user.is_satisfied,
                getattr(user, "_migration_penalty_applied", False),
                getattr(user, "_migration_overhead_active", False),
            ))
            # Reset per-step flags so a user who isn't affected next step
            # doesn't carry a stale True forward.
            user._migration_penalty_applied = False
            user._migration_overhead_active = False

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
            writer.writerow(["Step", "UserID", "Target", "HopCount", "BackhaulLoS"])
            writer.writerows(self.hop_counts_log)

        with open(os.path.join(folder, "satisfaction_users.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "UserID", "AppType", "ThroughputReq_Mbps",
                             "DataRate_Mbps", "Target", "Satisfied",
                             "migration_penalty_applied", "migration_overhead_active"])
            writer.writerows(self.satisfaction_users_log)

        with open(os.path.join(folder, "satisfaction_summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "N_Total", "N_Satisfied", "Rate_Global",
                             "N_AR_VR", "Sat_AR_VR", "Rate_AR_VR",
                             "N_Streaming", "Sat_Streaming", "Rate_Streaming",
                             "N_BestEffort", "Sat_BestEffort", "Rate_BestEffort"])
            writer.writerows(self.satisfaction_summary_log)

        with open(os.path.join(folder, "cr_utilization.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "BS_ID", "CR_capacity_mbps", "CR_load_mbps", "CR_utilization"])
            writer.writerows(self.cr_utilization_log)

        # Migration-cost overlay metric — informational only (RQ2 future
        # discussion), never fed into satisfaction or reward. All-zero when
        # migration_cost_mode is None or "hard_cutover" (see log()).
        with open(os.path.join(folder, "migration_overhead.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "migration_overhead_mbps"])
            writer.writerows(self.migration_overhead_log)

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
        self.cr_agent = None      # set externally to enable dynamic CR placement
        self.dynamic_rn = True    # set to False in headless runs to disable auto add/remove RNs
        self.user_move_range = 5  # max |dx|,|dy| per step; set from config["user_mobility"] in build_simulation
        # Migration-cost model, eval only (dissertation Ch.7). None reproduces
        # exact pre-existing behaviour and MUST stay None during train_rn /
        # train_cr — see run_experiment.py's --migration-mode validation.
        self.migration_cost_mode = None  # None | "hard_cutover" | "make_before_break"

        # 2x2 factorial design (dissertation Ch.7 M0-M3): CR admission order
        # and radio-rate sharing, each independently switchable and each
        # active in ALL modes (unlike migration_cost_mode above, these change
        # user.is_satisfied and therefore the RN/CR reward signal itself, so
        # they must be set consistently for train_rn, train_cr, AND eval).
        # M0 = both left at these defaults, reproducing the exact pre-factorial
        # behaviour the existing M0 Q-tables/results were produced under.
        self.cr_admission_policy = "fcfs"    # "fcfs" (M0/M2) | "priority" (M1/M3)
        self.radio_allocation    = "equal_share"  # "equal_share" (M0/M1) | "proportional" (M2/M3)

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

        # Propagate the 2x2 factorial flags every step (cheap) so a script
        # can flip them on self before any step and have it take effect
        # immediately, consistently across train_rn/train_cr/eval.
        self.metrics.admission_policy = self.cr_admission_policy
        self.metrics.radio_allocation = self.radio_allocation

        for user in self.users:
            r = self.user_move_range
            dx, dy = random.randint(-r, r), random.randint(-r, r)
            user.move((dx, dy))

        for bs in self.base_stations:
            bs.reset()
            if bs.has_compute_resource and bs.compute_resource is not None:
                bs.compute_resource.current_load_mbps = 0.0
                bs.compute_resource.demanded_load_mbps = 0.0
        for rn in self.relay_nodes:
            rn.reset()

        for rn in self.relay_nodes:
            rn.step(self.users)

        for rn in self.relay_nodes:
            best_bs, _ = min(
                ((bs, deployment_link_budget(rn.position, bs.position, self.obstacles))
                 for bs in self.base_stations),
                key=lambda t: t[1],
            )
            rn.parent = best_bs
            rn.backhaul_los = not is_line_blocked(
                rn.position, rn.parent.position, self.obstacles
            )

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
                reasons = [reason for _, reason in blocked_nodes] or ["No suitable nodes nearby"]
                log_line = f"[User {user.id}] No connection. Reasons: {reasons}"
                print(log_line)
                self.debug_logs.append(log_line)

        if self.dynamic_rn and self.timestep % 50 == 0:
            evaluate_and_adjust_relay_nodes(self.relay_nodes, self.users)

        # CR placement agent: Q-update with prev step, then choose + apply new placement
        if self.cr_agent is not None:
            current_state = self.cr_agent.get_state()
            if (self.cr_agent.prev_state is not None
                    and self.cr_agent.prev_action is not None
                    and self.cr_agent.last_reward is not None):
                self.cr_agent.update(
                    self.cr_agent.prev_state,
                    self.cr_agent.prev_action,
                    self.cr_agent.last_reward,
                    current_state,
                )
            action = self.cr_agent.select_action(current_state)

            # Migration-cost bookkeeping (eval only): snapshot which BS carry
            # a CR *before* apply_action, purely by reading existing state —
            # apply_action() itself is completely untouched, for every
            # cr_agent type (CRPlacementAgent or a baseline strategy).
            prev_cr_by_bs_id = {}
            if self.migration_cost_mode is not None:
                prev_cr_by_bs_id = {
                    bs.id: bs.compute_resource for bs in self.base_stations
                    if bs.has_compute_resource and bs.compute_resource is not None
                }

            self.cr_agent.apply_action(action)
            self.cr_agent.prev_state = current_state
            self.cr_agent.prev_action = action

            if self.migration_cost_mode is not None:
                still_active_ids = {bs.id for bs in self.base_stations if bs.has_compute_resource}
                lost_bs_ids = set()
                lost_capacity_mbps = 0.0
                for bs_id, cr in prev_cr_by_bs_id.items():
                    if bs_id not in still_active_ids:
                        lost_bs_ids.add(bs_id)
                        lost_capacity_mbps += cr.capacity_mbps
                self.metrics.migration_cost_mode = self.migration_cost_mode
                self.metrics.migration_lost_bs_ids = lost_bs_ids
                self.metrics.migration_lost_capacity_mbps = lost_capacity_mbps
            else:
                self.metrics.migration_cost_mode = None
                self.metrics.migration_lost_bs_ids = set()
                self.metrics.migration_lost_capacity_mbps = 0.0

        self.metrics.log(self.timestep, self.users)

        # CR agent reward: counterfactual (users saved by CR) + optional shaping
        if self.cr_agent is not None:
            n = len(self.users)
            n_sat = sum(1 for u in self.users if u.is_satisfied)
            self.cr_agent.last_reward_global = n_sat / n if n > 0 else 0.0
            if hasattr(self.cr_agent, 'compute_and_set_reward'):
                self.cr_agent.compute_and_set_reward(self.metrics.cr_counterfactual_reward)
            else:
                self.cr_agent.last_reward = self.metrics.cr_counterfactual_reward

        for bs in self.base_stations:
            if bs.has_compute_resource and bs.compute_resource is not None:
                cr = bs.compute_resource
                self.metrics.cr_utilization_log.append((
                    self.timestep, bs.id,
                    cr.capacity_mbps,
                    round(cr.current_load_mbps, 2),
                    round(cr.utilization, 4),
                ))

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

        if self.cr_agent is not None and self.cr_agent.q_history:
            cr_log = os.path.join(output_dir, "cr_learning.csv")
            with open(cr_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Avg_Q", "Epsilon"])
                for i, (q, eps) in enumerate(
                    zip(self.cr_agent.q_history, self.cr_agent.epsilon_history)
                ):
                    writer.writerow([i + 1, round(q, 6), round(eps, 6)])

            dq_log = os.path.join(output_dir, "cr_delta_q.csv")
            with open(dq_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "DeltaQ"])
                for i, dq in enumerate(self.cr_agent.delta_q_history):
                    writer.writerow([i + 1, round(dq, 10)])

        self.metrics.save_all(output_dir)

        debug_file = os.path.join(output_dir, f"debug_step_{self.timestep}.log")
        with open(debug_file, "w") as f:
            f.write(f"Step: {self.timestep}\n")
            f.write("\n".join(self.debug_logs))

        print(f"[Logs saved to '{output_dir}/' folder.")
