from random import random
from abviralsimulator.core import Handler
from abviralsimulator.enums import VirusHandlers, InfectionType


class Infection:

    def __init__(self, virus_type, transmission_info, time_step):

        self.virus_type = virus_type
        self.time_step = time_step

        duration_handler = Handler.get_handler(virus_type, VirusHandlers.DURATION)
        symptoms_trigger_handler = Handler.get_handler(virus_type, VirusHandlers.SHOWING_SYMPTOMS_AFTER)
        asymptomatic_rate_handler = Handler.get_handler(virus_type, VirusHandlers.ASYMPTOMATIC)

        if random() < asymptomatic_rate_handler(virus_type):
            self.infection_type = InfectionType.ASYMPTOMATIC
        else:
            self.infection_type = InfectionType.SYMPTOMATIC

        self.duration_counter = duration_handler(virus_type, time_step)
        self.symptoms_after_counter = symptoms_trigger_handler(virus_type, time_step)
        self.probability_evolution, self.probability, self.max_probability = transmission_info

    def is_over(self):
        return self.duration_counter == 0

    def update(self):
        self.duration_counter -= 1
        self.symptoms_after_counter -= 1
        self.probability = max(self.max_probability, self.probability_evolution(self.probability))

    def has_symptoms(self):
        return self.infection_type is not InfectionType.ASYMPTOMATIC and self.symptoms_after_counter <= 0
