# Marie Project description
## Project Proposal: Agent-Based Modeling of Mobile Edge Computing for Latency- and Throughput-Aware Optimization
Future 5G and 6G networks are not only about providing connectivity but also about enabling applications that require strict performance guarantees such as ultra-low latency, high throughput, and reliability. Mobile Edge Computing (MEC) brings compute resources closer to the users, reducing round-trip delays and enabling new use cases like augmented reality, autonomous driving, and real-time analytics. However, the placement of compute resources and network functions in heterogeneous networks is a complex optimization challenge, especially when user requirements vary dynamically.

Building on the existing agent-based modeling (ABM) simulator, this project introduces user application requirements (e.g., latency thresholds, throughput needs) and extends the model to include computing nodes that can host services or functions. The main focus will be on exploring how to optimally place compute resources in the network to satisfy application constraints while efficiently utilizing network capacity.

The project aims to extend the ABM simulator to model MEC-enabled networks where both connectivity and computing resources are dynamically optimized. The student will implement application-specific requirements for users, model compute resource placement strategies, and investigate distributed self-organization approaches for balancing latency, throughput, and resource efficiency.
## Objectives
1. Extend Simulator with Application-Aware Users:
  - Model users not just as traffic sources, but as entities with application requirements (e.g., latency < 10 ms for AR, high throughput for video streaming).
  - Track end-to-end performance metrics that combine radio conditions with compute placement.
2. Introduce Compute Nodes and Function Placement:
  - Add computing nodes (e.g., edge servers, core cloud servers) to the simulator.
  - Implement models for network function placement and service migration across different parts of the network.
3. Optimization of Compute Placement:
  - Develop algorithms to decide where to place compute (e.g., macro BS, micro BS, central server) based on user application requirements.
  - Investigate trade-offs between placing resources close to users (low latency but more distributed) versus centralized placement (higher efficiency but higher delay).
4. Evaluation and Experiments:
  - Run experiments for different scenarios (urban, suburban, rural) with mixed user applications.
  - Compare distributed agent-based strategies against centralized optimization baselines.
  - Explore how mobility of users affects the need for service migration.
## Example Scenarios
- Urban AR/VR use case: High density of users with strict latency requirements, testing the benefit of MEC placement near small cells.
- Video streaming in suburban deployment: Throughput-heavy applications distributed across macro and micro base stations.
- Rural e-health scenario: Compute placement trade-offs between sparse local servers and central cloud infrastructure.



# Project from 2024
## Project Proposal: Agent-Based Modeling for Self-Organization in Communication Networks
The growing complexity of communication networks, especially with the introduction of 5G and future 6G, requires new approaches to ensure that these networks can meet the increasing demands for low latency, high reliability, and efficient resource allocation. Traditional centralized control methods often struggle to cope with the dynamic nature of these networks, particularly in environments with a dense deployment of small cells. These cells, which serve as the backbone of integrated access and backhaul (IAB) systems, are critical for extending coverage and enhancing capacity in urban areas. However, managing these networks effectively requires solutions that can adapt in real-time, optimize performance autonomously, and reduce the operational costs associated with network maintenance.

Agent-based modeling (ABM) offers a promising approach to address these challenges by simulating the behavior of individual network elements as autonomous agents capable of self-organization. By allowing each agent to make local decisions based on its environment and interactions with neighboring agents, ABM can mimic the distributed nature of modern communication networks. This approach can lead to more resilient and adaptive networks, where optimization happens at a local level, reducing the need for extensive centralized control. In this context, the proposed project aims to develop an ABM simulation tool to explore and enhance self-organization mechanisms in communication networks, with a focus on optimizing latency and IAB performance in small cell deployments.

## Goal: Developing an Agent-Based Simulation Tool for Distributed Network Optimization
The primary goal of this project is to design and implement an agent-based modeling simulation tool that will allow us to study and improve self-organization in communication networks. This tool will enable the exploration of how autonomous agents within the network can cooperate to optimize key performance metrics such as latency and the efficiency of integrated access and backhaul in small cell environments.

To achieve this, the project will involve creating a robust simulation environment where different network configurations can be tested under varying conditions. The agents in this simulation will represent different network elements, such as base stations or relay nodes, each equipped with decision-making algorithms designed to minimize latency and maximize IAB efficiency. The project will focus on developing and fine-tuning these algorithms to enable the agents to self-organize in a way that optimizes the overall network performance.

Additionally, the tool will be designed to allow for easy customization and extension, so that different optimization strategies can be tested and compared. The ultimate objective is to identify the most effective strategies for distributed network optimization and to provide insights that can be applied to real-world communication networks, particularly in the context of small cell deployments and IAB systems.
