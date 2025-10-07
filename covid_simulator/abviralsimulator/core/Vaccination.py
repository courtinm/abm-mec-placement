import math
import copy
import random

from abviralsimulator.core import Handler
from abviralsimulator.enums import HealthcareHandlers, ActorHandlers
from abviralsimulator.managers import VaccineManager
from abviralsimulator.structure import Singleton


class Vaccination(metaclass=Singleton):

    def __init__(self, healthcare, actors, time_step):
        # Vaccination priorities of agents
        priorities = []
        for index, actor in enumerate(actors):
            priority_handler = Handler.get_handler(actor, ActorHandlers.VACCINATION_PRIORITY)
            priority = priority_handler()

            priorities.append((priority, index))

        self.priorities = [x[1] for x in sorted(priorities, key=lambda x: x[0])]
        self.number_of_agents = len(priorities)

        daily_vaccine_count_handler = Handler.get_handler(healthcare, HealthcareHandlers.DAILY_VACCINE_COUNT)
        self.daily_vaccines = {vaccine_type:daily_vaccine_count_handler(vaccine_type) for vaccine_type in VaccineManager.vaccines.values()}

        self.refresh_counter = math.ceil(24*60//time_step)

        self.vaccines_per_time_step = sum(self.daily_vaccines.values())/self.refresh_counter

        self.index = 0
        self.__refresh_vaccines()

    def __refresh_vaccines(self):
        self.available_vaccines = copy.deepcopy(self.daily_vaccines)
        self.counter = 0

    def __random_available_vaccine(self):
        if len(self.available_vaccines) == 0:
            return None
        vaccine = random.choice(list(self.available_vaccines.keys()))
        self.available_vaccines[vaccine] -= 1

        if self.available_vaccines[vaccine] == 0:
            self.available_vaccines.pop(vaccine)

        return vaccine

    def update(self, agents):

        if self.index >= self.number_of_agents:
            return

        used_vaccines = 0

        while used_vaccines < self.vaccines_per_time_step:
            agent_index = self.priorities[self.index]
            agent = agents[agent_index]

            if agent.should_vaccinate():
                random_vaccine = self.__random_available_vaccine()
                if random_vaccine is None:
                    break
                agent.vaccinate_with(random_vaccine)
                used_vaccines += 1

            self.index += 1

            if self.index >= self.number_of_agents:
                return

        self.counter += 1
        if self.counter == self.refresh_counter:
            self.__refresh_vaccines()
