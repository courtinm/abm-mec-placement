
from abviralsimulator.structure import Singleton
from abviralsimulator.store import ContactCount, TraveledDistance, StateChanges


class DataStore(metaclass=Singleton):

    def __init__(self, agents, time_step):
        self.time_step = time_step

        self.modules = {}

        self.__add_module(ContactCount, agents)
        self.__add_module(StateChanges, agents)
        self.__add_module(TraveledDistance, agents)
        # self.__add_module(AgentTracker, agents, [11, 111, 7, 92, 64, 37, 55, 46, 98])

        self.counter = 24 * 60 // self.time_step

    def __add_module(self, module, *args):
        self.modules[module.__name__] = module(*args)

    def update(self, agents):
        for module in self.modules.values():
            module.update(agents)

        self.counter -= 1
        if self.counter == 0:
            self.__daily_update()
            self.__reset_counter()

    def __reset_counter(self):
        self.counter = 24 * 60 // self.time_step

    def __daily_update(self):
        for module in self.modules.values():
            module.daily_update()

    def get_modules(self):
        return self.modules

    def get_module(self, module):
        return self.modules[module.__name__]