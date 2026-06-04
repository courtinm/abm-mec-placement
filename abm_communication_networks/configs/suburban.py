CONFIG = {
    "grid_size": 100,
    "app_mix": {"AR_VR": 0.2, "streaming": 0.5, "best_effort": 0.3},
    "n_users": 20,
    "user_mobility": "high",
    "base_stations": [
        {"x": 10, "y": 10, "type": "macro", "capacity": 50, "has_compute_resource": True, "cr_capacity_mbps": 75},
        {"x": 80, "y": 80, "type": "macro", "capacity": 50, "has_compute_resource": False},
        {"x": 50, "y": 50, "type": "small", "capacity": 30, "has_compute_resource": False},
    ],
    "relay_nodes": [
        {"x": 40, "y": 40, "throughput": 30},
    ],
    "cr_placement": {
        "k": 2,              # 2 CRs among 3 BS
        "cr_capacity_mbps": 100.0,
    },
    "obstacles": [
        {"x": 35, "y": 35, "size": "small"},
        {"x": 60, "y": 40, "size": "large"},
        {"x": 40, "y": 60, "size": "small"},
    ],
}
