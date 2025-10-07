from ..managers import ScenarioManager


def Scenario(name, time_step, iterations):
    """
        Used for loading the parameters used for a single simulation.
    """

    def wrapper(cls):
        cls.args = {'time_step': time_step, 'iterations': iterations}

        ScenarioManager.add_scenario(name, cls)
        return cls

    return wrapper
