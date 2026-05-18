# Mirrors the hardcoded setup that was in main.py before the config refactor.
CONFIG = {
    "grid_size": 100,
    "app_mix": {"AR_VR": 0.0, "streaming": 0.0, "best_effort": 1.0},
    "n_users": 20,
    "user_mobility": "high",
    "base_stations": [
        {"x": 10, "y": 10, "type": "macro", "capacity": 50, "has_compute_resource": True},
        {"x": 80, "y": 80, "type": "small", "capacity": 30, "has_compute_resource": False},
        {"x": 10, "y": 90, "type": "macro", "capacity": 50, "has_compute_resource": False},
        {"x": 90, "y": 10, "type": "small", "capacity": 30, "has_compute_resource": False},
    ],
    "relay_nodes": [
        {"x": 30, "y": 30, "throughput": 30},
        {"x": 60, "y": 60, "throughput": 30},
    ],
    "obstacles": [
        {"x": 45, "y": 45, "size": "small"},
        {"x": 50, "y": 50, "size": "large"},
        {"x": 55, "y": 45, "size": "small"},
        {"x": 45, "y": 55, "size": "large"},
    ],
}
