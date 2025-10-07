from functools import partial

from abviralsimulator.managers import ActorManager
from abviralsimulator.core import Agent, Handler, Movement
from abviralsimulator.enums import ActorHandlers


class Common:

    @classmethod
    def initialize_agents(cls, actors, scenario_parameters, available_tests):
        time_step = scenario_parameters['time_step']
        agents = []
        for index, actor in enumerate(actors):
            actor_class = ActorManager.actor_classes[actor.name]

            home_location_handler = Handler.get_handler(actor_class, ActorHandlers.HOME_LOCATION)

            dying_probabilities = Common.create_dying_probabilities(actor_class, actor)
            transmission_probabilities = Common.create_transmission_probabilities(actor_class, actor, time_step)

            home_location = home_location_handler(actor)
            movement = Common.create_movement(actor_class, actor, home_location, time_step)
            testing_probaiblities = Common.create_testing_probabilities(actor_class, actor)

            agent = Agent(index, home_location, movement, transmission_probabilities, dying_probabilities,
                          testing_probaiblities)
            agents.append(agent)

        return agents

    @staticmethod
    def infect_agents(actors, agents):
        for actor, agent in zip(actors, agents):
            actor_class = ActorManager.actor_classes[actor.name]

            initially_infected_handler = Handler.get_handler(actor_class, ActorHandlers.INITIALLY_INFECTED)
            initially_infected = initially_infected_handler(actor)

            if initially_infected is not None:
                agent.infect_with(virus_type=initially_infected)

    @staticmethod
    def create_transmission_probabilities(actor_class, actor, time_step):
        transmission_probability_handlers = Handler.get_multiple_handlers(actor_class,
                                                                          ActorHandlers.TRANSMISSION_PROBABILITY)
        transmission_rules = {}

        for handler in transmission_probability_handlers:
            virus_type = handler.args['virus_type']
            initial_probability = handler.args['initial_probability']
            max_probability = handler.args['max_probability']

            transmission_rules[virus_type] = partial(handler, actor,
                                                     time_step=time_step), initial_probability, max_probability

        return transmission_rules

    @staticmethod
    def create_dying_probabilities(actor_class, actor):
        dying_probability_handlers = Handler.get_multiple_handlers(actor_class, ActorHandlers.DYING_PROBABILITY)
        dying_probabilities = {}
        for handler in dying_probability_handlers:
            virus_type = handler.args['virus_type']
            dying_probabilities[virus_type] = partial(handler, actor)
        return dying_probabilities

    @staticmethod
    def create_movement(actor_class, actor, home_location, time_step):

        at_home_handlers = Handler.get_multiple_handlers(actor_class, ActorHandlers.AT_HOME_DURATION)
        transition_rules = {}

        for handler in at_home_handlers:

            virus_type = handler.args['virus_type']
            viral_state = handler.args['viral_state']
            if viral_state not in transition_rules:
                transition_rules[viral_state] = {}
            transition_rules[viral_state][virus_type] = partial(handler, actor)

        movement_handler = Handler.get_handler(actor_class, ActorHandlers.MOVEMENT)
        bound_movement_handler = partial(movement_handler, actor, home_location, time_step)
        return Movement(bound_movement_handler, transition_rules, time_step)

    @staticmethod
    def create_testing_probabilities(actor_class, actor):
        testing_probability_handlers = Handler.get_multiple_handlers(actor_class, ActorHandlers.TESTING_PROBABILITY)
        testing_probabilities = {}

        for handler in testing_probability_handlers:
            virus_type = handler.args['virus_type']
            testing_probabilities[virus_type] = partial(handler, actor)

        return testing_probabilities

    @staticmethod
    def clamp(val, minimum, maximum):
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val