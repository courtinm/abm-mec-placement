from math import isclose

#this class presents the CR
class ComputeRessources:
    def __init__(self, id, position, capacity, current_load):
        self.id = id
        self.position = position
        self.capacity = capacity
        self.current_load = current_load



class BaseStation:
    def __init__(self, id, position, capacity, bs_type="macro", tx_power=20, custom_range=None):
        self.id = id
        self.position = position
        self.capacity = capacity
        self.current_load = 0
        self.tx_power = tx_power
        self.energy_consumed = 0
        self.status = "active"  # or "failed"
        self.bs_type = bs_type.lower()
        self.has_compute_resource = False  # as I don't put CR at each BS, the BS should mention if it has CR
        self.compute_resource = None #to access the details concerning the CR

        # Assigning range based on type, unless overridden
        if custom_range is not None:
            self.range = custom_range
        elif self.bs_type == "macro":
            self.range = 100
        elif self.bs_type == "small":
            self.range = 30
        else:
            raise ValueError(f"Unknown base station type: {bs_type}")

    def update_load(self, load):
        self.current_load += load
        self.energy_consumed += 0.1 * load

    def is_overloaded(self):
        return self.current_load > self.capacity

    def reset(self):
        """Reset load at the beginning of each simulation step."""
        self.current_load = 0

    def can_see(self, point, obstacles):
        """Return False if any obstacle lies exactly on the straight line segment."""
        x1, y1 = self.position
        x2, y2 = point
        for ox, oy in obstacles:
            # Check colinearity via cross-product ~0
            if isclose((y2 - y1) * (ox - x1), (x2 - x1) * (oy - y1), abs_tol=1e-6):
                # Then check if obstacle lies between endpoints
                if min(x1, x2) <= ox <= max(x1, x2) and min(y1, y2) <= oy <= max(y1, y2):
                    return False
        return True

    def fail(self):
        self.status = "failed"

    def recover(self):
        self.status = "active"

    def move(self, new_pos):
        self.position = new_pos  # Not used after Step 1 — BS are now static.

    def __str__(self):
        return (f"BS{self.id}@{self.position} Type:{self.bs_type} "
                f"Load:{self.current_load}/{self.capacity} En:{self.energy_consumed:.1f}J Sta:{self.status}")
