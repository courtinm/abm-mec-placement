CONFIG = {
    "grid_size": 100,
    "app_mix": {"AR_VR": 0.1, "streaming": 0.3, "best_effort": 0.6},
    "n_users": 10,
    "user_mobility": "high",
    "base_stations": [
        {"x": 10, "y": 10, "type": "macro", "capacity": 50, "has_compute_resource": True, "cr_capacity_mbps": 50},
        {"x": 80, "y": 80, "type": "macro", "capacity": 50, "has_compute_resource": False},
    ],
    "relay_nodes": [],
    "cr_placement": {
        "k": 1,              # 1 CR among 2 BS
        "cr_capacity_mbps": 100.0,
    },
    "obstacles": [
        {"x": 50, "y": 50, "size": "small"},
        {"x": 70, "y": 30, "size": "small"},
    ],
}
