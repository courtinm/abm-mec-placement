class AgentTracker:

    def __init__(self, agents, tracked_ids):
        self.locations = {agent.index: [agent.location] for agent in agents if agent.index in tracked_ids}

    def update(self, agents):
        for agent in agents:
            if agent.index not in self.locations:
                continue
            self.locations[agent.index] += [agent.location]

    def daily_update(self):
        pass