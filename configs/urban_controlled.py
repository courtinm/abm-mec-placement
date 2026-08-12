CONFIG = {
    "grid_size": 100,
    "app_mix": {"AR_VR": 0.4, "streaming": 0.4, "best_effort": 0.2},
    "n_users": 30,
    "user_mobility": "high",
    "base_stations": [
        {"x": 10, "y": 10, "type": "macro", "capacity": 50, "has_compute_resource": True, "cr_capacity_mbps": 100},
        {"x": 80, "y": 80, "type": "macro", "capacity": 50, "has_compute_resource": False},
        {"x": 20, "y": 70, "type": "small", "capacity": 30, "has_compute_resource": False},
        {"x": 70, "y": 20, "type": "small", "capacity": 30, "has_compute_resource": False},
    ],
    "relay_nodes": [
        {"x": 30, "y": 30, "throughput": 30},
        {"x": 60, "y": 60, "throughput": 30},
    ],
    "cr_placement": {
        "k": 2,
        "cr_capacity_mbps": 100.0,
    },
    "obstacles": [
        {"x": 25, "y": 25, "size": "large"},
        {"x": 45, "y": 45, "size": "large"},
        {"x": 50, "y": 50, "size": "large"},
        {"x": 55, "y": 45, "size": "small"},
        {"x": 45, "y": 55, "size": "large"},
        {"x": 65, "y": 65, "size": "large"},
        {"x": 35, "y": 65, "size": "small"},
        {"x": 65, "y": 35, "size": "small"},
    ],
}
