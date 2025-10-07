from random import random

from abviralsimulator.enums import ViralState
# from abviralsimulator.core import Infection, LocationQuery, Immunity, Healthcare
# from abviralsimulator.core import Infection, LocationQuery, Immunity
from .Healthcare import Healthcare
from .Infection import Infection
from .LocationQuery import LocationQuery

class Agent:

    def __init__(self, index, home_location, movement, transmission_probabilities, dying_probabilities,
                 testing_probabilities):

        self.index = index
        self.viral_state = ViralState.UNAFFECTED
        self.virus_type = None
        self.home_location = home_location
        self.location = home_location
        self.movement = movement

        self.testing_probabilities = testing_probabilities
        self.transmission_probabilities = transmission_probabilities
        self.dying_probabilities = dying_probabilities

        self.infection = None
        self.immunity = None

    def infect_with(self, virus_type):
        self.virus_type = virus_type
        self.infection = Infection(virus_type, self.transmission_probabilities[virus_type], self.movement.time_step)
        self.transition_to(ViralState.INFECTED)

    def vaccinate_with(self, vaccine_type):
        self.immunity = Immunity(vaccine_type, self.movement.time_step)
        self.transition_to(ViralState.IMMUNE)

    def update(self):
        self.update_state()
        self.move()

    def update_state(self):
        if self.viral_state == ViralState.UNAFFECTED:
            infection_probabilities = LocationQuery.query(self.location)
            virus = self.__should_be_infected(infection_probabilities)

            if virus is not None:
                self.infect_with(virus)

        elif self.viral_state == ViralState.INFECTED:
            self.__should_diagnose()
            self.__progress_infection()

        elif self.viral_state == ViralState.DIAGNOSED:
            self.__progress_infection()

        elif self.viral_state == ViralState.IMMUNE:
            self.immunity.update()

        elif self.viral_state == ViralState.DECEASED:
            return
        elif self.viral_state == ViralState.RECOVERED:
            return
        else:
            raise RuntimeError("Unreachable update state.")

    def transition_to(self, target):

        if target == ViralState.INFECTED:
            # Statistics.get_instance().add_infection(self.index, self.location)
            pass

        elif target == ViralState.DIAGNOSED:
            pass

        elif target == ViralState.DECEASED:
            self.virus_type = None

        elif target == ViralState.IMMUNE:
            pass

        elif target == ViralState.RECOVERED:
            self.virus_type = None
        else:
            raise RuntimeError("Unreachable transition state.")

        self.viral_state = target

    def move(self):
        if self.viral_state == ViralState.RECOVERED or self.viral_state == ViralState.DECEASED:
            return

        self.movement.update(self.viral_state, self.virus_type)

        if self.movement.is_moving():
            self.location = self.movement.get_next_location()
        else:
            self.location = self.home_location

    def __should_die(self):
        dying_probability = self.dying_probabilities[self.virus_type]()
        return random() < dying_probability

    def __should_be_infected(self, infection_probabilities):
        infection_probabilities.sort(key=lambda x: x[0])
        for probability, virus in infection_probabilities:
            if random() < probability:
                return virus
        return None

    def __progress_infection(self):
        self.infection.update()
        if self.infection.is_over():
            if self.__should_die():
                self.transition_to(ViralState.DECEASED)
            else:
                self.transition_to(ViralState.RECOVERED)
            return True
        else:
            if self.movement.is_moving():
                LocationQuery.add(self.virus_type, self.location, self.infection.probability)
            return False

    def __should_diagnose(self):

        if not self.infection.has_symptoms():
            return False

        testing_probability = self.testing_probabilities[self.virus_type]()
        if not Healthcare().is_getting_tested(testing_probability):
            return False

        Healthcare().get_tested()
        self.transition_to(ViralState.DIAGNOSED)

        return True

    def should_vaccinate(self):
        return self.viral_state == ViralState.UNAFFECTED

    def print_state(self):

        print("Actor(id: %d):" % self.index)
        print("\tviral_state: %s, virus_type: %s" % (self.viral_state, self.virus_type))
        print("\thome_location: %s, location: %s" % (self.home_location, self.location))
        print("\tis_moving: %d, counter: %d" % (self.movement.is_moving(), self.movement.state_change_counter))
