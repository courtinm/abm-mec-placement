from scipy import spatial


class ContactCount:

    def __init__(self, agents):
        self.contacts = {agent.index: [] for agent in agents}
        self.daily_contact_count = []
        self.daily_unique_contact_count = []

    def update(self, agents):
        # Filter out agents that are home
        moving_agents = list(filter(lambda agent: agent.movement.is_moving(), agents))

        # No agent is moving, nothing to do
        if len(moving_agents) == 0:
            return

        indexes = [agent.index for agent in moving_agents]
        locations = [agent.location for agent in moving_agents]

        tree = spatial.KDTree(locations)
        multi_agent_results = tree.query_ball_point(locations, 1)

        for agent_index, agent_results in zip(indexes, multi_agent_results):
            self.contacts[agent_index] += [indexes[result] for result in agent_results if
                                           indexes[result] != agent_index]

    def daily_update(self):
        
        contact_count = [len(contacts) for contacts in self.contacts.values()]
        unique_contact_count = [len(set(contacts)) for contacts in self.contacts.values()]
        
        self.daily_contact_count.append(sum(contact_count))
        self.daily_unique_contact_count.append(sum(unique_contact_count))
        
        self.contacts = {index: [] for index in self.contacts.keys()}
