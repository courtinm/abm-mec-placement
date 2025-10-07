from abviralsimulator.enums import ActorHandlers, ViralState
from abviralsimulator.managers import ActorManager
from abviralsimulator.core import Handler


def Actor(name):
    def wrapper(cls):
        cls.name = name
        ActorManager.add_actor(name, cls)
        return cls

    return wrapper


def dying_probability(virus_type):
    def wrapper(method):
        Handler.register_handler(method, ActorHandlers.DYING_PROBABILITY, {'virus_type': virus_type})
        return method

    return wrapper


def transmission_probability(virus_type, initial_probability=0, max_probability=1):
    def wrapper(method):
        Handler.register_handler(method, ActorHandlers.TRANSMISSION_PROBABILITY, {
            'virus_type': virus_type,
            'initial_probability': initial_probability,
            'max_probability': max_probability
        })
        return method

    return wrapper


def movement(method):
    Handler.register_handler(method, ActorHandlers.MOVEMENT)
    return method


def home_location(method):
    Handler.register_handler(method, ActorHandlers.HOME_LOCATION)
    return method


def at_home(method):
    Handler.register_handler(method, ActorHandlers.AT_HOME_DURATION, {'virus_type': None, 'viral_state': None})
    return method


def at_home_infected(virus_type):
    def wrapper(method):
        Handler.register_handler(method, ActorHandlers.AT_HOME_DURATION,
                                 {'virus_type': virus_type, 'viral_state': ViralState.INFECTED})
        return method

    return wrapper


def at_home_diagnosed(virus_type):
    def wrapper(method):
        Handler.register_handler(method, ActorHandlers.AT_HOME_DURATION,
                                 {'virus_type': virus_type, 'viral_state': ViralState.DIAGNOSED})
        return method

    return wrapper


def initially_infected(method):
    Handler.register_handler(method, ActorHandlers.INITIALLY_INFECTED)
    return method


def testing_probability(virus_type):
    def wrapper(method):
        Handler.register_handler(method, ActorHandlers.TESTING_PROBABILITY, {'virus_type': virus_type})
        return method

    return wrapper


def vaccination_priority(method):
    Handler.register_handler(method, ActorHandlers.VACCINATION_PRIORITY)
    return method
