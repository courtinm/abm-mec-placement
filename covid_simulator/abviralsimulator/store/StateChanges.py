from abviralsimulator.enums import ViralState
from abviralsimulator.structure import CsvWritable, overrides


class StateChanges(CsvWritable):

    def __init__(self, agents):
        self.states = {agent.index: agent.viral_state for agent in agents}
        self.state_changes = {viral_state: [] for viral_state in ViralState}

    def update(self, agents):
        for viral_state in self.state_changes.keys():
            self.state_changes[viral_state] += [0]

        for agent in agents:
            if agent.viral_state is self.states[agent.index]:
                continue
            self.state_changes[agent.viral_state][-1] += 1
            self.states[agent.index] = agent.viral_state

    # Not needed
    def daily_update(self):
        pass

    @overrides(CsvWritable)
    def get_header(self):
        return [viral_state.name for viral_state in ViralState]

    @overrides(CsvWritable)
    def get_row_size(self):
        return len(self.state_changes[ViralState.UNAFFECTED])

    @overrides(CsvWritable)
    def get_row(self, index):
        return [self.state_changes[viral_state][index] for viral_state in ViralState]