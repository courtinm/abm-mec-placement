from abviralsimulator.core import Handler
from abviralsimulator.enums import VaccineHandlers


class Immunity:

    def __init__(self, vaccine_type, time_step):
        self.time_step = time_step
        self.counter = 0

        efficiency_handlers = Handler.get_multiple_handlers(vaccine_type, VaccineHandlers.EFFICENCY)
        self.immunity_efficiency = {}
        for handler in efficiency_handlers:
            virus_type = handler.args['virus_type']
            self.immunity_efficiency[virus_type] = handler

    def get_immunity(self, virus_type):
        return self.immunity_efficiency[virus_type](self.counter, self.time_step)

    def update(self):
        self.counter += 1
