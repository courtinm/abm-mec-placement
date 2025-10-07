VERBOSE = True


class ScenarioManager:
    scenarios = {}

    @classmethod
    def add_scenario(cls, name, scenario):
        if VERBOSE:
            print("Registering scenario with name %s." % name)
        if name in cls.scenarios:
            raise RuntimeError("A scenario of that name already exists.")

        cls.scenarios[name] = scenario
