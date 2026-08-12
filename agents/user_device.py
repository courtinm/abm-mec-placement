class UserDevice:
    def __init__(self, id, position, app_type="best_effort",
                 latency_threshold_ms=200, throughput_req_mbps=1):
        self.id = id
        self.position = position
        self.app_type = app_type
        self.latency_threshold_ms = latency_threshold_ms
        self.throughput_req_mbps = throughput_req_mbps
        self.connected_to = None
        self.has_los = False
        self.disconnections = 0
        self.is_satisfied = False

    def move(self, delta):
        new_x = max(0, min(100, self.position[0] + delta[0]))
        new_y = max(0, min(100, self.position[1] + delta[1]))
        self.position = (new_x, new_y)

    def __str__(self):
        return f"User{self.id}@{self.position} App:{self.app_type} Connected:{self.connected_to}"
