from abviralsimulator.helpers import Algebra
from abviralsimulator.structure import CsvWritable


class TraveledDistance:

    def __init__(self, agents):
        self.movement_tracker = {agent.index: (agent.home_location, 0) for agent in agents}
        self.traveled_distance = []

    def update(self, agents):
        for agent in agents:
            old_location, traveled_length = self.movement_tracker[agent.index]

            # Agent is not moving, don't update
            if agent.movement.is_moving() is False:
                self.movement_tracker[agent.index] = agent.home_location, traveled_length
                continue
            increment = Algebra.euclid_distance(old_location, agent.location)
            self.movement_tracker[agent.index] = (agent.location, traveled_length + increment)

    def daily_update(self):
        pass
