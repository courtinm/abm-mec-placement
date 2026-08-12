CONFIG = {
    "grid_size": 100,
    "app_mix": {"AR_VR": 0.5, "streaming": 0.4, "best_effort": 0.1},
    "n_users": 40,
    "user_mobility": "low",
    # 3 macro BS in triangle formation for uniform coverage, each paired with a small BS
    "base_stations": [
        {"x": 25, "y": 25, "type": "macro", "capacity": 50, "has_compute_resource": True, "cr_capacity_mbps": 100},
        {"x": 75, "y": 25, "type": "macro", "capacity": 50, "has_compute_resource": False},
        {"x": 50, "y": 75, "type": "macro", "capacity": 50, "has_compute_resource": False},
        {"x": 40, "y": 30, "type": "small", "capacity": 30, "has_compute_resource": False},
        {"x": 60, "y": 30, "type": "small", "capacity": 30, "has_compute_resource": False},
        {"x": 50, "y": 60, "type": "small", "capacity": 30, "has_compute_resource": False},
    ],
    "relay_nodes": [
        {"x": 25, "y": 55, "throughput": 30},
        {"x": 75, "y": 55, "throughput": 30},
        {"x": 50, "y": 40, "throughput": 30},
    ],
    "cr_placement": {
        "k": 3,
        "cr_capacity_mbps": 100.0,
    },
    "obstacles": [
        {"x": 15, "y": 45, "size": "small"},
        {"x": 35, "y": 15, "size": "large"},
        {"x": 65, "y": 15, "size": "large"},
        {"x": 85, "y": 45, "size": "small"},
        {"x": 20, "y": 65, "size": "large"},
        {"x": 45, "y": 50, "size": "small"},
        {"x": 55, "y": 50, "size": "small"},
        {"x": 80, "y": 65, "size": "large"},
        {"x": 30, "y": 80, "size": "small"},
        {"x": 50, "y": 85, "size": "large"},
        {"x": 70, "y": 80, "size": "small"},
        {"x": 45, "y": 35, "size": "small"},
    ],
}
