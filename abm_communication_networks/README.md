# Agent-Based Modeling for Self-Organization in 5G Networks using Q-Learning

## Project Title
**Agent-Based Simulation for Self-Organization in Communication Networks with IAB and Q-Learning**
This project simulates the self-organization of communication networks, with a focus on optimizing latency and integrated access and backhaul (IAB) performance in small cell deployments using agent-based modeling (ABM).

---

## Project Proposal Summary

Modern 5G and 6G networks face the challenge of managing dense small cell deployments while maintaining **low latency**, **high reliability**, and **scalable resource allocation**. Centralized control strategies often fall short in such dynamic environments.

This project addresses these challenges using an **Agent-Based Modeling (ABM)** approach, where each network element (users, base stations, relay nodes) acts as an autonomous agent capable of local decision-making. Combined with **Q-learning**, this setup allows the system to **self-organize and adapt** over time with minimal external control.

### Key Themes:
- **Integrated Access and Backhaul (IAB):** Relay nodes extend coverage and forward user data to macro base stations wirelessly
- **Self-Organization:** Nodes reposition themselves over time using learned strategies
- **Q-Learning:** Reinforcement learning to optimize placement of relay nodes
- **mmWave Communication:** High-frequency signals with limited range and high path loss
- **Shadow Fading and Obstacles:** Real-world phenomena simulated to reflect NLoS conditions

---

## Project Goal

To design and implement a **simulation tool** that:
- Emulates realistic 5G environments with users, base stations, and obstacles
- Implements **Q-learning** for autonomous optimization by relay nodes
- Demonstrates **self-organization** and evaluates network performance under varying user mobility and load
- Explores how IAB architectures perform under dynamic conditions using ABM

---

## Core Features & Their Implementation

### Agent-Based Modeling (ABM)
Implemented in:
- `simulator.py`, `user.py`, `base_station.py`, and `relay_node.py`

Each entity (user, base station, relay node) acts independently:
- **Users** move randomly each step and connect to nearby nodes
- **Relay nodes** reposition themselves based on Q-learning outcomes
- **Base stations** provide direct or indirect access depending on LoS

ABM allows the simulation to scale naturally and reflect real-world decentralized network behavior.

---

### Integrated Access and Backhaul (IAB)

**What is IAB?**  
IAB allows base stations and relay nodes to provide **both user access and backhaul** over the same wireless link — reducing infrastructure costs and increasing deployment flexibility.

**In This Project:**
- Relay nodes **receive connections from users** and **forward them** to macro base stations if direct access is blocked
- Users check if LoS to a BS exists. If blocked, they fall back to relay nodes (simulated IAB behavior)
- Controlled in `simulator.py` during `simulate_step()`

Relay nodes act like intermediate IAB agents — especially critical in NLoS cases.

---

### Q-Learning for Relay Node Optimization

**Goal:** Find optimal relay node placements that maximize user connections while minimizing overload and NLoS

Implemented in `relay_node.py`:
- States: Relative position on the grid
- Actions: Move in 8 directions or stay
- Rewards: Based on number of users connected and penalties for NLoS or overload
- Epsilon-greedy strategy with decay for exploration/exploitation

Logged metrics: Q-values over time, epsilon decay, and rewards.

**Every 50 steps**, the system evaluates and adjusts the positions of RNs using:
```python
evaluate_and_adjust_relay_nodes(self.relay_nodes, self.users)

--> mmWave Path Loss Model
Implemented in:
def path_loss_mmwave(d, frequency_GHz=28, shadow_std=3.0)
(Located in simulator.py)

Frequency: 28 GHz (standard mmWave)

Based on Free-Space Path Loss (FSPL) formula

Distance-dependent loss formula used
High path loss simulates real mmWave behavior (poor coverage behind obstacles)

--> Shadow Fading
Shadow fading simulates environmental variability and is randomized each connection using:

shadow = random.gauss(0, shadow_std)
This Gaussian noise component is added to path loss, creating a more realistic signal environment where links fluctuate due to physical obstructions or signal bouncing.

Obstacle-Aware Line-of-Sight (LoS) & Non-LoS
Handled in:
def is_line_blocked(src, dst, obstacles, allow_through_large=False)
Checks if direct path between node and user is obstructed

Macro base stations can penetrate large obstacles, relay nodes cannot

NLoS links are penalized via:

Signal degradation

Lower rewards in Q-learning

Higher latency

User logs show when a connection is made without LoS:

[User 3] Connected to node RN4 WITHOUT line-of-sight (penalty applied)

--> User Mobility
Each user moves in a random direction every simulation step using:

dx, dy = random.randint(-5, 5), random.randint(-5, 5)
Simulates real-world movement like pedestrians or vehicles. Forces network to adapt continuously.

Metrics Collected
Saved in /logs/ every step via metrics.py:

latency_avg.csv – average user connection latency

latency_max.csv – worst-case latency

failed_connections.csv – unserved users

nlos_connections.csv – number of NLoS connections

handoffs.csv – users switching connections

optimal_connections.csv – % of users connected under LoS with available capacity

These metrics help evaluate if the Q-learning and ABM strategies are leading to improvement.

Visualization: What You See
When running main.py, you will see a real-time visualization:


| Symbol / Color        | Meaning                                               |
|------------------------|--------------------------------------------------------|
| 🔵 Blue circle         | **Macro Base Station** (penetrates obstacles)         |
| 🔹 Light Blue circle   | **Small Base Station** (weaker, fixed)                |
| 🟢 Green circle        | **Relay Node** (IAB node, moves using Q-learning)     |
| 🔴 Red dot             | **User Device** (connected)                           |
| ⚪ Gray dot            | **User Device** (disconnected / lost signal)          |
| 🟥 Dark red square     | **Large Obstacle**                                    |
| ⬛ Dark gray square     | **Small Obstacle**                                    |
| 🟡 Yellow line         | **LoS Connection (Line of Sight)**                    |
| 🔴 Red line            | **NLoS Connection (Non-Line of Sight, weaker)**       |


How to Run and Analyze
Run Simulation:

python main.py

Plot Metrics:

python plot_metrics.py
This will generate graphs for:

Average/max latency

Q-learning performance for each relay node (Q-value curves)

% of optimal LoS connections

Number of handoffs and NLoS links


Conclusion:

This project demonstrates how distributed, self-organizing behavior in 5G networks can be simulated using agent-based modeling. The use of Q-learning on relay nodes allows the system to adapt dynamically to changing user behavior and signal conditions without centralized coordination.

Key Takeaways:
Relay nodes learn optimal placement through local interaction and rewards

ABM mimics realistic network behaviors, reducing the complexity of centralized management

IAB architecture is naturally supported, especially for coverage extension

Shadow fading, path loss, and obstacles add realism to the simulation

Results are logged, visualized, and analyzed quantitatively for performance

This system provides a foundation for experimenting with intelligent network design, especially in preparation for densified 5G and 6G rollouts in urban areas.