from abviralsimulator.structure import Singleton


class Statistics(metaclass=Singleton):

    def __init__(self):
        self.statistics = {}

    def add(self, name, value):
        self.statistics[name] = value

    def print(self):
        padding = 15
        title = "STATISTICS"
        print("*"*padding, "STATISTICS", "*"*padding, sep="")
        for name, value in self.statistics.items():
            print(name, ": ", value, sep="")
        print("*" * (2*padding + len(title)))
