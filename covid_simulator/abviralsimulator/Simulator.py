from abviralsimulator.managers import ScenarioManager, VirusManager
from abviralsimulator.helpers import Common, Validator
from abviralsimulator.core import LocationQuery, Healthcare, Vaccination
from abviralsimulator.store import DataStore
from abviralsimulator.statistics import Statistics


class Simulator:

    def __init__(self):
        for Virus in VirusManager.viruses.values():
            Validator.validate_virus(Virus)

    def __setup_scenario(self, Scenario):

        scenario = Scenario()

        iterations = Scenario.args['iterations']
        time_step = Scenario.args['time_step']

        healthcare = scenario.setup_healthcare()

        actors = scenario.setup()

        Healthcare(healthcare)
        Vaccination(healthcare, actors, time_step)

        LocationQuery.setup(VirusManager.viruses)

        agents = Common.initialize_agents(actors, Scenario.args, healthcare)

        DataStore(agents, time_step)
        Common.infect_agents(actors, agents)

        Statistics().add("Number of agents", len(agents))
        Statistics().add("Iterations", iterations)

        return agents

    def __run_scenario(self, iterations, agents):
        data_store = DataStore()
        vaccination = Vaccination()

        for iteration in range(iterations):

            for agent in agents:
                agent.update()

            vaccination.update(agents)
            data_store.update(agents)
            LocationQuery.update()

    def run(self):

        for name, Scenario in ScenarioManager.scenarios.items():
            print("Running %s scenario." % name)

            iterations = Scenario.args['iterations']

            agents = self.__setup_scenario(Scenario)
            self.__run_scenario(iterations, agents)
