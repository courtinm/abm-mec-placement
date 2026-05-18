import random
from math import sqrt

class UserDevice:
    def __init__(self, id, position, data_demand=None, traffic_type="best_effort",
                 app_type="best_effort", latency_threshold_ms=200, throughput_req_mbps=1):
        self.id = id
        self.position = position
        self.data_demand = data_demand if data_demand is not None else random.randint(1, 5)
        self.traffic_type = traffic_type
        self.app_type = app_type
        self.latency_threshold_ms = latency_threshold_ms
        self.throughput_req_mbps = throughput_req_mbps
        self.connected_to = None
        self.latency = None
        self.disconnections = 0
        self.is_satisfied = False

    def move(self, delta):
        new_x = max(0, min(100, self.position[0] + delta[0]))
        new_y = max(0, min(100, self.position[1] + delta[1]))
        self.position = (new_x, new_y)

    def calculate_distance(self, other_pos):
        return sqrt((self.position[0] - other_pos[0]) ** 2 + (self.position[1] - other_pos[1]) ** 2)
