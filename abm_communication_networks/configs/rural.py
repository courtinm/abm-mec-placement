CONFIG = {
    "grid_size": 100,
    "app_mix": {"AR_VR": 0.1, "streaming": 0.3, "best_effort": 0.6},
    "n_users": 10,
    "user_mobility": "high",
    "base_stations": [
        {"x": 10, "y": 10, "type": "macro", "capacity": 50, "has_compute_resource": True},
        {"x": 80, "y": 80, "type": "macro", "capacity": 50, "has_compute_resource": False},
    ],
    "relay_nodes": [],
    "obstacles": [
        {"x": 50, "y": 50, "size": "small"},
        {"x": 70, "y": 30, "size": "small"},
    ],
}
